"""InternVL3.5 engine with decode-phase KV cache eviction.

Wraps LATCInferenceEngine, forces manual prefill (even with stages=[]), then
evicts visual KV entries before running greedy decode with position tracking
based on ORIGINAL prefill length.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from src.decode_kv_eviction import evict_visual_kv
from src.latc import LATCConfig, LATCInferenceEngine


class InternVLDecodeEvictionEngine(LATCInferenceEngine):
    """InternVL engine with decode-phase density-guided KV eviction."""

    def __init__(
        self,
        model,
        tokenizer,
        keep_ratio: float = 0.5,
        strategy: str = "density",
        seed: int = 42,
    ):
        # We need stages=[] for uncompressed prefill, but we can't use the
        # early model.chat() path — override generate below.
        no_comp_cfg = LATCConfig(stages=[], seed=seed, strategy="cac_v4")
        super().__init__(model, tokenizer, no_comp_cfg)
        self.keep_ratio = keep_ratio
        self.eviction_strategy = strategy

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        question: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        history: list | None = None,
    ) -> str:
        """Generate with decode-phase KV eviction.

        Unlike the parent class which short-circuits to model.chat() when
        stages=[], we always do manual prefill so we can evict visual KV
        entries before decoding.
        """
        input_ids, inputs_embeds, visual_indices = self._prepare_inputs(
            pixel_values, question, history
        )
        self.last_merge_diagnostics = []

        # Manual uncompressed prefill
        final_hidden, cache = self._latc_prefill(inputs_embeds, visual_indices)

        # Apply final norm + lm_head for prefill logits
        final_hidden = self.model.language_model.model.norm(final_hidden)
        prefill_logits = self.model.language_model.lm_head(final_hidden)

        # Decode-phase eviction
        original_len = cache.get_seq_length(0)
        if visual_indices and self.keep_ratio < 1.0:
            vis_start = visual_indices[0]
            vis_end = visual_indices[-1] + 1
            visual_tokens = inputs_embeds[0, vis_start:vis_end, :]
            original_len, trimmed_len, n_kept = evict_visual_kv(
                cache=cache,
                visual_tokens=visual_tokens,
                vis_start=vis_start,
                vis_end=vis_end,
                keep_ratio=self.keep_ratio,
                strategy=self.eviction_strategy,
                seed=self.config.seed,
            )

        # Decode with original-position tracking
        generated_ids = self._greedy_decode_with_offset(
            prefill_logits, cache, original_len, max_new_tokens
        )

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        try:
            from src.latc import get_conv_template
            template = get_conv_template(self.model.template)
            text = text.split(template.sep.strip())[0].strip()
        except Exception:
            text = text.strip()
        return text

    def _greedy_decode_with_offset(
        self,
        prefill_logits: torch.Tensor,
        cache: DynamicCache,
        original_len: int,
        max_new_tokens: int,
    ) -> list[int]:
        """Greedy decode preserving ORIGINAL prefill positions for RoPE."""
        lm = self.model.language_model
        layers = lm.model.layers
        generated = []

        next_token_id = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for step in range(max_new_tokens):
            token_id = next_token_id.item()
            if token_id == self.eos_token_id:
                break
            generated.append(token_id)

            h = lm.get_input_embeddings()(next_token_id)

            # Use ORIGINAL prefill length + step for position_ids
            pos = original_len + step
            position_ids = torch.tensor([[pos]], device=self.device)
            position_embeddings = lm.model.rotary_emb(h, position_ids)

            for layer in layers:
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
