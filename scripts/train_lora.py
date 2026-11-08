"""Train the TwinPrune LoRA adapter for Table 2, Block 3b."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path

# Default INTERNVL_MODEL_DIR before importing latc.
os.environ.setdefault(
    "INTERNVL_MODEL_DIR",
    "OpenGVLab/InternVL3_5-8B-Instruct",
)

import torch
from PIL import Image
from torch.optim import AdamW

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.latc import LATCConfig, LATCInferenceEngine
from src.pruning import STRATEGIES
from src.model_utils import prepare_pixel_values
from transformers import DynamicCache


# ─────────────────────────────────────────────────────────────────────────────
# InternVL-specific grad-enabled prefill
# ─────────────────────────────────────────────────────────────────────────────

def prefill_with_grad_internvl(
    engine: LATCInferenceEngine,
    inputs_embeds: torch.Tensor,
    visual_indices: list[int],
):
    """Mirror `LATCInferenceEngine._latc_prefill` with grads enabled.

    Selection itself runs under `torch.no_grad()` (discrete), but surviving
    visual tokens are re-sliced from the live grad-bearing `hidden_states`
    so LoRA gradients flow through the post-compression layers.

    Returns (hidden_states, cache).
    """
    lm = engine.model.language_model.model
    layers = lm.layers
    n_layers = len(layers)
    cache = DynamicCache()

    engine._last_inputs_embeds = inputs_embeds
    h = inputs_embeds
    vis_start = visual_indices[0]
    vis_end = visual_indices[-1] + 1
    pre_n_vis = vis_end - vis_start

    merge_at = {layer_idx: ratio for layer_idx, ratio in engine.stages}
    engine._merge_checkpoints = []

    for i in range(n_layers):
        seq_len = h.shape[1]
        position_ids = torch.arange(seq_len, device=engine.device).unsqueeze(0)
        position_embeddings = lm.rotary_emb(h, position_ids)

        outputs = layers[i](
            h,
            position_embeddings=position_embeddings,
            past_key_values=cache,
            use_cache=True,
        )
        h = outputs[0] if isinstance(outputs, tuple) else outputs

        if i in merge_at:
            engine._merge_checkpoints.append((i, seq_len))
            ratio = merge_at[i]

            strategy_fn = STRATEGIES[engine.config.strategy]
            with torch.no_grad():
                _h_throwaway, _vs, _ve, n_kept, keep_mask = strategy_fn(
                    hidden=h.detach(),
                    vis_start=vis_start,
                    vis_end=vis_end,
                    ratio=ratio,
                    cache=cache,
                    current_layer=i,
                    model=engine.model,
                    seed=engine.config.seed,
                    return_keep_mask=True,
                )[:5]

            # Re-slice from the live (grad-bearing) tensor.
            pre_h = h[:, :vis_start, :]
            vis_h = h[:, vis_start:vis_end, :]
            kept_vis_h = vis_h[:, keep_mask, :]
            post_h = h[:, vis_end:, :]

            h = torch.cat([pre_h, kept_vis_h, post_h], dim=1)

            vis_end = vis_start + n_kept
            pre_n_vis = n_kept

    engine._prefill_trimmed_len = h.shape[1]
    return h, cache


# ─────────────────────────────────────────────────────────────────────────────
# A-OKVQA training subset
# ─────────────────────────────────────────────────────────────────────────────




def load_aokvqa_train_subset(arrow_dir: str, n_samples: int, seed: int = 42):
    from datasets import Dataset, concatenate_datasets
    import glob
    arrow_files = sorted(glob.glob(os.path.join(arrow_dir, "a-okvqa-train-*.arrow")))
    if not arrow_files:
        raise FileNotFoundError(f"No a-okvqa-train-*.arrow in {arrow_dir}")
    parts = [Dataset.from_file(f) for f in arrow_files]
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    n_total = len(ds)
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    samples = []
    for idx in indices:
        if len(samples) >= n_samples:
            break
        ex = ds[idx]
        choices = ex.get("choices", [])
        if not choices or len(choices) != 4:
            continue
        ans_idx = ex.get("correct_choice_idx", None)
        if ans_idx is None:
            continue
        ans_letter = "ABCD"[int(ans_idx)]
        image = ex["image"]
        if not isinstance(image, Image.Image):
            continue
        image = image.convert("RGB")
        question = ex["question"]
        opt_lines = [f"({l}) {c}" for l, c in zip("ABCD", choices)]
        prompt = "\n".join([question, *opt_lines, "Answer with the option letter only."])
        samples.append({"image": image, "question": prompt, "answer": ans_letter})
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model_path: str,
    parquet_path: str,
    save_dir: str,
    n_samples: int = 2000,
    layer: int = 0,
    ratio: float = 0.70,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lr: float = 1e-4,
    epochs: int = 1,
    grad_accum: int = 4,
    seed: int = 42,
    log_every: int = 10,
    device: str = "cuda:0",
    dataset: str = "aokvqa",
    target_modules: list = None,
    method: str = "cac_v4",
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    if min(n_samples, epochs, grad_accum) < 1:
        raise ValueError('Training counts must be positive')
    if Path(save_dir).exists():
        raise FileExistsError(f'Refusing to overwrite an adapter: {save_dir}')

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

    from twinprune.resources import DEFAULT_MANIFEST
    from twinprune.evaluate import load_model
    reference = next(cell for cell in json.loads(DEFAULT_MANIFEST.read_text())['cells']
                     if cell.get('adapter_required'))
    print(f"[load] pinned InternVL3.5-8B-Instruct from {model_path}", flush=True)
    model, tokenizer = load_model(reference, model_path, device)

    # Apply LoRA on q_proj/v_proj of every LLM decoder layer.
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(target_modules),
        modules_to_save=[],
    )
    print(f"[lora] target_modules={list(target_modules)}", flush=True)
    print("[lora] applying LoRA to language_model.model q_proj/v_proj", flush=True)
    # Freeze all params first (visual encoder, lm_head, etc.).
    for p in model.parameters():
        p.requires_grad_(False)
    # InternVL: model.language_model is Qwen2ForCausalLM. We wrap its .model
    # (Qwen2Model — the transformer body) so PEFT can find q_proj/v_proj on
    # decoder layers, while leaving model.language_model.lm_head untouched.
    causal_lm = model.language_model
    transformer = causal_lm.model
    transformer = get_peft_model(transformer, lora_cfg)
    causal_lm.model = transformer
    transformer.print_trainable_parameters()

    print(f"[engine] strategy={method}", flush=True)
    engine = LATCInferenceEngine(
        model, tokenizer,
        LATCConfig(stages=[(layer, ratio)], strategy=method, seed=seed),
    )

    print(f"[data] loading {n_samples} {dataset} train samples", flush=True)
    if dataset != "aokvqa":
        raise ValueError("The main-table LoRA recipe uses A-OKVQA")
    samples = load_aokvqa_train_subset(parquet_path, n_samples=n_samples, seed=seed)
    print(f"[data] got {len(samples)} samples", flush=True)
    if len(samples) != n_samples:
        raise ValueError(f'Requested {n_samples} training examples, found {len(samples)}')

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[opt] {sum(p.numel() for p in trainable)} trainable parameters", flush=True)
    optimizer = AdamW(trainable, lr=lr)

    model.train()

    losses = []
    step = 0
    optimizer.zero_grad()

    t0 = time.time()
    n_total_steps = len(samples) * epochs
    print(f"[train] {n_total_steps} samples × {epochs} epoch(s) = {n_total_steps} fwd/bwd", flush=True)

    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        order = list(range(len(samples)))
        rng.shuffle(order)
        for sample_idx in order:
            sample = samples[sample_idx]

            try:
                pixel_values = prepare_pixel_values(sample["image"], model, device=device)
                input_ids, inputs_embeds, visual_indices = engine._prepare_inputs(
                    pixel_values, sample["question"]
                )
                if not visual_indices:
                    continue
            except Exception as e:
                print(f"[skip] prepare failed: {type(e).__name__}: {e}", flush=True)
                continue

            try:
                hidden_states, cache = prefill_with_grad_internvl(
                    engine, inputs_embeds, visual_indices
                )
            except Exception as e:
                print(f"[skip] prefill failed: {type(e).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()
                continue

            ans_text = sample["answer"]
            ans_ids = tokenizer(ans_text, add_special_tokens=False).input_ids
            if not ans_ids:
                continue
            ans_token_id = ans_ids[0]

            # InternVL loss: norm lives on the underlying Qwen2Model (PEFT base);
            # lm_head lives on the outer Qwen2ForCausalLM (untouched by PEFT).
            # PEFT.__getattr__ forwards .norm to base_model.norm, so this works
            # whether `transformer` is the raw Qwen2Model or PEFT-wrapped.
            final_h = transformer.norm(hidden_states[:, -1:, :])  # (1,1,D)
            logits = causal_lm.lm_head(final_h)  # (1,1,V)
            target = torch.tensor([[ans_token_id]], device=device)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
            )

            (loss / grad_accum).backward()
            losses.append(loss.item())

            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % log_every == 0:
                avg = sum(losses[-log_every:]) / max(1, len(losses[-log_every:]))
                elapsed = time.time() - t0
                rate = step / elapsed
                print(f"[step {step:4d}] loss={loss.item():.4f} avg{log_every}={avg:.4f} "
                      f"rate={rate:.2f} samples/s", flush=True)

    if step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad()

    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s ({step} fwd/bwd)", flush=True)
    if step != n_total_steps:
        raise RuntimeError(f'Incomplete training: {step}/{n_total_steps} examples; refusing to save an adapter')

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"[save] LoRA adapter → {save_dir}", flush=True)
    transformer.save_pretrained(save_dir)

    with open(os.path.join(save_dir, "train_log.json"), "w") as f:
        json.dump({
            "n_samples": len(samples),
            "processed_examples": step,
            "epochs": epochs,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lr": lr,
            "grad_accum": grad_accum,
            "seed": seed,
            "method": method,
            "layer": layer,
            "prune_ratio": ratio,
            "model_hidden_size": model.config.llm_config.hidden_size,
            "base_checkpoint_identity": model._twinprune_checkpoint_identity,
            "target_modules": list(target_modules),
            "dataset": dataset,
            "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "elapsed_seconds": elapsed,
            "losses": losses,
            "final_loss_avg10": sum(losses[-10:]) / max(1, len(losses[-10:])),
        }, f, indent=2)

    return save_dir, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", "--model-path", default="OpenGVLab/InternVL3_5-8B-Instruct")
    p.add_argument("--parquet_path", "--parquet-path", required=True)
    p.add_argument("--dataset", default="aokvqa", choices=["aokvqa"])
    p.add_argument("--save_dir", "--save-dir", required=True)
    p.add_argument("--n_samples", "--n-samples", type=int, default=2000)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--ratio", type=float, default=0.70)
    p.add_argument("--lora_r", "--lora-r", type=int, default=16)
    p.add_argument("--lora_alpha", "--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad_accum", "--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target_modules", "--target-modules", default="q_proj,v_proj",
                   help="Comma-separated LoRA target modules")
    p.add_argument("--method", default="cac_v4", choices=["cac_v4"],
                   help="Compression strategy used in the forward pass")
    args = p.parse_args()

    target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]

    train(
        model_path=args.model_path,
        parquet_path=args.parquet_path,
        save_dir=args.save_dir,
        n_samples=args.n_samples,
        layer=args.layer,
        ratio=args.ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        seed=args.seed,
        device=args.device,
        dataset=args.dataset,
        target_modules=target_modules,
        method=args.method,
    )


if __name__ == "__main__":
    main()
