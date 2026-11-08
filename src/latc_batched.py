"""TRUE-batched (B>1) LATC inference engine for InternVL.

Wraps a shared image across B samples to measure batched prefill latency.
The decoder layers see (B, T, D) tensors processed in one matmul, rather
than the sequential B-time loop that LATCInferenceEngine does.

Numerical equivalence at B=1 is preserved — the prefill / decode paths use
the same strategy fns, the same per-layer rotary embeddings, the same RoPE
fix during decode, and produce the same generated text as src.latc.

Simplifying assumption: same image / same prompt across the batch. All B
samples share keep_idx at every merge layer, so we compute the strategy on
sample 0 and apply the (already-shared) trimming to the (B, T, D) tensor.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

# Reuse the accuracy engine's configuration and selector.
from src.latc import LATCConfig
from src.pruning import STRATEGIES

from src.internvl_conversation import get_conv_template, register_model


class LATCBatchedInferenceEngine:
    """Inference engine with TRUE-batched (B>1) LATC compression.

    Identical interface to src.latc.LATCInferenceEngine.generate(), but
    additionally exposes generate_batched(pixel_values, question, batch=B,
    max_new_tokens=N) that processes B copies of the same image in one
    forward pass.

    All B samples share the SAME image (caller's responsibility), so the
    keep_idx selected at each merge layer is the same per sample. We compute
    it on sample 0 and broadcast the trim across the (B, T, D) tensor and
    the (B, n_kv_heads, T, head_dim) cache.

    Per-step decode produces B argmaxes; we early-stop when all B sequences
    have hit EOS or max_new_tokens is reached.
    """

    def __init__(self, model, tokenizer, config: "LATCConfig | None" = None):
        register_model(model)
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or LATCConfig()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.last_merge_diagnostics: list[dict] = []

        self.stages = sorted(self.config.stages, key=lambda x: x[0])

        template = get_conv_template(model.template)
        self.eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        question: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        history: list | None = None,
    ) -> str:
        """Single-sample generation (B=1). Mirrors src.latc behaviour."""
        results = self.generate_batched(
            pixel_values, question, batch=1,
            max_new_tokens=max_new_tokens, do_sample=do_sample, history=history,
        )
        return results[0]

    @torch.no_grad()
    def generate_batched(
        self,
        pixel_values: torch.Tensor,
        question: str,
        batch: int = 1,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        history: list | None = None,
    ) -> list[str]:
        """Generate `batch` outputs in one forward pass on the shared image.

        Args:
            pixel_values: (n_tiles, 3, H, W) image tensor for ONE image.
            question: text prompt.
            batch: how many parallel copies to run (all share the same image).
            max_new_tokens: max tokens generated per sample.

        Returns:
            list of `batch` decoded strings.
        """
        # Build (B, T, D) inputs by replicating the single image's prefill embeds.
        input_ids_1, inputs_embeds_1, visual_indices = self._prepare_inputs(
            pixel_values, question, history,
        )
        B = batch
        # Replicate along batch dim. Use repeat so we have actual storage
        # (not just a view) — the cache stores (B, ...) tensors.
        if B > 1:
            inputs_embeds_B = inputs_embeds_1.repeat(B, 1, 1).contiguous()
            input_ids_B = input_ids_1.repeat(B, 1).contiguous()
        else:
            inputs_embeds_B = inputs_embeds_1
            input_ids_B = input_ids_1

        self.last_merge_diagnostics = []

        if not self.stages:
            # No compression — run plain (B, T, D) prefill manually so we
            # get the exact same numerical path as the compressed branch
            # (full final_norm + lm_head + greedy decode from compressed cache).
            final_hidden, cache = self._plain_prefill_batched(inputs_embeds_B)
        else:
            final_hidden, cache = self._latc_prefill_batched(
                inputs_embeds_B, visual_indices,
            )

        final_hidden = self.model.language_model.model.norm(final_hidden)
        logits = self.model.language_model.lm_head(final_hidden)

        generated_ids = self._greedy_decode_batched(logits, cache, max_new_tokens, B)

        # Decode each sample to text
        template = get_conv_template(self.model.template)
        sep = template.sep.strip()
        outputs = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            text = text.split(sep)[0].strip()
            outputs.append(text)
        return outputs

    # ------------------------------------------------------------------
    # Input preparation (single sample; will be replicated to B)
    # ------------------------------------------------------------------
    def _prepare_inputs(
        self,
        pixel_values: torch.Tensor,
        question: str,
        history: list | None = None,
    ):
        """Identical to src.latc.LATCInferenceEngine._prepare_inputs."""
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        num_patches = pixel_values.shape[0] if pixel_values is not None else 0

        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        for old_q, old_a in (history or []):
            template.append_message(template.roles[0], old_q)
            template.append_message(template.roles[1], old_a)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches
            + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)

        vit_embeds = self.model.extract_feature(pixel_values)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        Bx, N, C = input_embeds.shape
        flat_embeds = input_embeds.reshape(Bx * N, C)
        flat_ids = input_ids.reshape(Bx * N)
        selected = flat_ids == self.model.img_context_token_id
        flat_embeds[selected] = vit_embeds.reshape(-1, C).to(flat_embeds.device)
        input_embeds = flat_embeds.reshape(Bx, N, C)

        visual_indices = (
            (input_ids[0] == self.model.img_context_token_id)
            .nonzero(as_tuple=True)[0]
            .tolist()
        )
        return input_ids, input_embeds, visual_indices

    # ------------------------------------------------------------------
    # Plain (no-compression) batched prefill
    # ------------------------------------------------------------------
    def _plain_prefill_batched(
        self,
        inputs_embeds_B: torch.Tensor,
    ) -> tuple[torch.Tensor, DynamicCache]:
        lm = self.model.language_model.model
        layers = lm.layers
        cache = DynamicCache()
        h = inputs_embeds_B
        seq_len = h.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        position_embeddings = lm.rotary_emb(h, position_ids)

        for i in range(len(layers)):
            outputs = layers[i](
                h,
                position_embeddings=position_embeddings,
                past_key_values=cache,
                use_cache=True,
            )
            h = outputs[0] if isinstance(outputs, tuple) else outputs

        # No merge → all layers used the same seq_len → no RoPE fix needed.
        self._merge_checkpoints = []
        self._prefill_trimmed_len = h.shape[1]
        return h, cache

    # ------------------------------------------------------------------
    # LATC batched prefill — compress on sample 0, broadcast to (B, T, D)
    # ------------------------------------------------------------------
    def _latc_prefill_batched(
        self,
        inputs_embeds_B: torch.Tensor,
        visual_indices: list[int],
    ) -> tuple[torch.Tensor, DynamicCache]:
        lm = self.model.language_model.model
        layers = lm.layers
        n_layers = len(layers)
        cache = DynamicCache()

        # Keep input embeddings available for diagnostics.
        self._last_inputs_embeds = inputs_embeds_B

        h = inputs_embeds_B
        vis_start = visual_indices[0]
        vis_end = visual_indices[-1] + 1

        merge_at = {layer_idx: ratio for layer_idx, ratio in self.stages}
        self._merge_checkpoints: list[tuple[int, int]] = []

        for i in range(n_layers):
            seq_len = h.shape[1]
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            position_embeddings = lm.rotary_emb(h, position_ids)

            outputs = layers[i](
                h,
                position_embeddings=position_embeddings,
                past_key_values=cache,
                use_cache=True,
            )
            h = outputs[0] if isinstance(outputs, tuple) else outputs

            if i in merge_at:
                self._merge_checkpoints.append((i, seq_len))
                merge_ratio = merge_at[i]
                h, vis_start, vis_end = self._apply_merge_batched(
                    h, vis_start, vis_end, merge_ratio, cache, i,
                )

        self._prefill_trimmed_len = h.shape[1]
        return h, cache

    def _apply_merge_batched(
        self,
        hidden: torch.Tensor,  # (B, T, D)
        vis_start: int,
        vis_end: int,
        merge_ratio: float,
        cache: DynamicCache,
        current_layer: int,
    ) -> tuple[torch.Tensor, int, int]:
        """Apply compression on sample 0 then broadcast trim to all (B, T, D).

        Strategy fns operate on hidden[0, vis_start:vis_end, :] under the hood
        (they use index 0 of the batch dim). They also call _trim_cache which
        slices the (B, n_heads, T, head_dim) cache along the seq dim — which
        is broadcast-correct across batch.

        For the (B, T, D) hidden tensor itself, the strategy fns do
        `hidden[:, vis_start:vis_end, :][:, keep_mask, :]` and re-cat. That
        already handles arbitrary B since the batch dim is preserved.

        So a B>1 hidden tensor flowing through these strategies "just works",
        provided we accept that the keep decision is made on sample 0 only —
        which is exactly correct here because all B samples share the image.
        """
        strategy_name = self.config.strategy
        try:
            strategy_fn = STRATEGIES[strategy_name]
        except KeyError as exc:
            raise ValueError(f"Unknown LATC strategy '{strategy_name}'") from exc

        B, T, D = hidden.shape
        # Pass a B=1 slice into the strategy. All B samples have identical
        # hidden state (same image, deterministic forward), so the strategy's
        # per-sample-0 decision applies to all. This avoids strategies that
        # internally do `compressed.unsqueeze(0)` and expect B=1 hidden.
        # Cache trimming inside the strategy works for any B because it slices
        # along the seq dim (which is shared across batch).
        h_1 = hidden[0:1].contiguous() if B > 1 else hidden
        embeds_1 = getattr(self, "_last_inputs_embeds", None)
        if embeds_1 is not None and embeds_1.shape[0] > 1:
            embeds_1 = embeds_1[0:1].contiguous()
        result = strategy_fn(
            hidden=h_1,
            vis_start=vis_start,
            vis_end=vis_end,
            ratio=merge_ratio,
            cache=cache,
            current_layer=current_layer,
            model=self.model,
            seed=self.config.seed,
            inputs_embeds=embeds_1,
        )
        # Strategy returns (hidden, vis_start, vis_end, n_vis [, keep_mask, full_keep_idx])
        new_hidden_1 = result[0]  # (1, T', D)
        new_vs = result[1]
        new_ve = result[2]
        # Broadcast back to (B, T', D). All B identical → expand+contiguous.
        if B > 1:
            new_hidden = new_hidden_1.expand(B, -1, -1).contiguous()
        else:
            new_hidden = new_hidden_1
        return new_hidden, new_vs, new_ve



    # ------------------------------------------------------------------
    # Greedy batched decode (B sequences in parallel)
    # ------------------------------------------------------------------
    def _greedy_decode_batched(
        self,
        prefill_logits: torch.Tensor,  # (B, T, V)
        cache: DynamicCache,
        max_new_tokens: int,
        B: int,
    ) -> list[list[int]]:
        lm = self.model.language_model
        layers = lm.model.layers

        # Same RoPE-fix as src.latc — per-layer position base derived from
        # merge checkpoints. Shared across the batch since all B used the
        # same compression schedule.
        checkpoints = getattr(self, "_merge_checkpoints", [])
        trimmed_base = getattr(self, "_prefill_trimmed_len", cache.get_seq_length(0))

        def _pos_base_for_layer(layer_idx: int) -> int:
            for merge_layer, pre_len in sorted(checkpoints, key=lambda x: x[0]):
                if merge_layer >= layer_idx:
                    return pre_len
            return trimmed_base

        layer_to_base = [_pos_base_for_layer(i) for i in range(len(layers))]

        # Initial next token: argmax over last position of each sample.
        next_token_id = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)

        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        generated = [[] for _ in range(B)]

        for step in range(max_new_tokens):
            # Append already-decoded tokens (skip if finished)
            for b in range(B):
                if finished[b]:
                    continue
                tid = int(next_token_id[b, 0].item())
                if tid == self.eos_token_id:
                    finished[b] = True
                    continue
                generated[b].append(tid)

            if bool(finished.all().item()):
                break

            # Embed next token for all B (we still feed even finished ones to
            # keep cache lengths aligned — they get whatever they get; their
            # outputs are discarded by `finished` mask).
            h = lm.get_input_embeddings()(next_token_id)  # (B, 1, dim)

            cur_len = cache.get_seq_length(0)

            unique_bases = set(layer_to_base)
            pos_emb_cache: dict[int, tuple] = {}
            for base in unique_bases:
                pos = base + (cur_len - trimmed_base)
                pid = torch.tensor([[pos]], device=self.device)
                pos_emb_cache[base] = lm.model.rotary_emb(h, pid)

            for i, layer in enumerate(layers):
                position_embeddings = pos_emb_cache[layer_to_base[i]]
                outputs = layer(
                    h,
                    position_embeddings=position_embeddings,
                    past_key_values=cache,
                    use_cache=True,
                )
                h = outputs[0] if isinstance(outputs, tuple) else outputs

            h = lm.model.norm(h)
            logits = lm.lm_head(h)
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        return generated
