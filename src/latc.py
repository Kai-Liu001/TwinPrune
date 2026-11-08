"""InternVL prefill pruning and greedy decoding for TwinPrune."""

import math
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from src.pruning import STRATEGIES
from src.internvl_conversation import get_conv_template, register_model


@dataclass
class LATCConfig:
    """Layer index and fraction of visual tokens to remove after that layer."""

    stages: list[tuple[int, float]] = field(default_factory=list)
    seed: int = 42
    strategy: str = "cac_v4"
    position_mode: str = "legacy"


def _infer_grid_shape(n_vis: int) -> tuple[int, int]:
    grid_w = max(1, int(round(math.sqrt(n_vis))))
    while grid_w > 1 and n_vis % grid_w != 0:
        grid_w -= 1
    grid_h = max(1, math.ceil(n_vis / grid_w))
    return grid_h, grid_w


def _initial_visual_positions(n_vis: int, device: torch.device) -> torch.Tensor:
    grid_h, grid_w = _infer_grid_shape(n_vis)
    indices = torch.arange(n_vis, device=device)
    rows = torch.div(indices, grid_w, rounding_mode="floor")
    cols = indices % grid_w
    return torch.stack([rows, cols], dim=-1).float()


class LATCInferenceEngine:
    """Inference engine with LATC visual token compression.

    Wraps an InternVLChatModel to apply layered token merging during prefill,
    then uses the compressed KV cache for autoregressive generation.

    Usage:
        engine = LATCInferenceEngine(model, tokenizer, LATCConfig(stages=[(3, 0.5), (8, 0.5)]))
        response = engine.generate(pixel_values, "Describe this image.")
    """

    def __init__(self, model, tokenizer, config: LATCConfig | None = None):
        register_model(model)
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or LATCConfig()
        if self.config.position_mode not in {"legacy", "original"}:
            raise ValueError("position_mode must be legacy or original")
        if self.config.position_mode == "original" and self.config.strategy not in {
            "cac_v4",
        }:
            raise ValueError("Original-position tracking requires a selector keep mask")
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.last_merge_diagnostics: list[dict] = []

        # Sort merge stages by layer index
        self.stages = sorted(self.config.stages, key=lambda x: x[0])

        # EOS token for generation stopping
        template = get_conv_template(model.template)
        self.eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        question: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        history: list | None = None,
    ) -> str:
        """Generate response with LATC compression.

        Args:
            pixel_values: (1, 3, H, W) image tensor.
            question: Text prompt (will be wrapped in conversation template).
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to sample (False = greedy).
            history: Conversation history [(q, a), ...].

        Returns:
            Generated text response.
        """
        input_ids, inputs_embeds, visual_indices = self._prepare_inputs(
            pixel_values, question, history
        )
        self.last_merge_diagnostics = []

        if not self.stages:
            # No compression — use original model.chat()
            gen_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
            return self.model.chat(
                self.tokenizer, pixel_values, question, gen_config,
                history=history or [], return_history=False,
            )

        # Manual prefill with LATC
        final_hidden, cache = self._latc_prefill(inputs_embeds, visual_indices)

        # Apply final norm + lm_head to get prefill logits
        final_hidden = self.model.language_model.model.norm(final_hidden)
        logits = self.model.language_model.lm_head(final_hidden)

        # Greedy decode from cache
        generated_ids = self._greedy_decode(logits, cache, max_new_tokens)

        # Decode to text
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        template = get_conv_template(self.model.template)
        text = text.split(template.sep.strip())[0].strip()
        return text

    def _prepare_inputs(
        self,
        pixel_values: torch.Tensor,
        question: str,
        history: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Replicate InternVLChatModel.chat() input preparation.

        Returns:
            input_ids: (1, seq_len) token IDs
            inputs_embeds: (1, seq_len, hidden_dim) embeddings with vision features inserted
            visual_indices: list of positions where visual tokens are
        """
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

        # Build conversation from template
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

        # Replace <image> with actual image tokens
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace("<image>", image_tokens, 1)

        # Tokenize
        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)

        # Get embeddings and replace visual token positions with vision features
        vit_embeds = self.model.extract_feature(pixel_values)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        B, N, C = input_embeds.shape
        flat_embeds = input_embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        selected = flat_ids == self.model.img_context_token_id
        flat_embeds[selected] = vit_embeds.reshape(-1, C).to(flat_embeds.device)
        input_embeds = flat_embeds.reshape(B, N, C)

        # Find visual token indices
        visual_indices = (input_ids[0] == self.model.img_context_token_id).nonzero(as_tuple=True)[0].tolist()

        return input_ids, input_embeds, visual_indices

    def _latc_prefill(
        self,
        inputs_embeds: torch.Tensor,
        visual_indices: list[int],
    ) -> tuple[torch.Tensor, DynamicCache]:
        """Manual layer-by-layer prefill with LATC merging.

        At each merge layer, merges visual tokens in hidden states and trims
        the KV cache for all preceding layers.

        Args:
            inputs_embeds: (1, seq_len, hidden_dim)
            visual_indices: Original positions of visual tokens

        Returns:
            final_hidden: (1, compressed_seq_len, hidden_dim)
            cache: DynamicCache populated with compressed KV entries
        """
        lm = self.model.language_model.model
        layers = lm.layers
        n_layers = len(layers)
        cache = DynamicCache()

        # Keep the input embeddings available for diagnostics.
        self._last_inputs_embeds = inputs_embeds

        h = inputs_embeds
        self._original_prefill_len = h.shape[1]
        survivor_positions = torch.arange(h.shape[1], device=self.device).unsqueeze(0)
        vis_start = visual_indices[0]
        vis_end = visual_indices[-1] + 1
        current_n_vis = vis_end - vis_start
        vis_positions = _initial_visual_positions(current_n_vis, self.device)

        # Build set of merge layers for quick lookup
        merge_at = {layer_idx: ratio for layer_idx, ratio in self.stages}

        # Track (merge_layer, pre_merge_seq_len) for RoPE position fix in decode.
        # Layer i's KV was computed with h.shape[1] BEFORE any merge at layer >= i.
        self._merge_checkpoints: list[tuple[int, int]] = []

        for i in range(n_layers):
            # Compute position embeddings for current sequence length
            seq_len = h.shape[1]
            position_ids = (survivor_positions if self.config.position_mode == "original"
                            else torch.arange(seq_len, device=self.device).unsqueeze(0))
            position_embeddings = lm.rotary_emb(h, position_ids)

            # Forward through layer with cache
            outputs = layers[i](
                h,
                position_embeddings=position_embeddings,
                past_key_values=cache,
                use_cache=True,
            )
            # transformers 5.x returns tensor directly, older returns tuple
            h = outputs[0] if isinstance(outputs, tuple) else outputs

            # Apply merge if this is a merge layer
            if i in merge_at:
                # Record seq_len BEFORE the merge — layers 0..i used this length.
                self._merge_checkpoints.append((i, seq_len))
                merge_ratio = merge_at[i]
                h, vis_start, vis_end, current_n_vis, vis_positions = self._apply_merge(
                    h, vis_start, vis_end, merge_ratio, cache, i, vis_positions
                )
                if self.config.position_mode == "original":
                    survivor_positions = survivor_positions[:, self._last_full_keep_indices]

        # Trimmed seq_len after all merges (= what layers after last merge saw).
        self._prefill_trimmed_len = h.shape[1]

        return h, cache

    def _apply_merge(
        self,
        hidden: torch.Tensor,
        vis_start: int,
        vis_end: int,
        merge_ratio: float,
        cache: DynamicCache,
        current_layer: int,
        vis_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, int, torch.Tensor]:
        """Select visual survivors and trim the cache through this layer."""
        try:
            strategy_fn = STRATEGIES[self.config.strategy]
        except KeyError as exc:
            raise ValueError("This release supports TwinPrune (cac_v4) only") from exc
        result = strategy_fn(
            hidden=hidden,
            vis_start=vis_start,
            vis_end=vis_end,
            ratio=merge_ratio,
            cache=cache,
            current_layer=current_layer,
            model=self.model,
            seed=self.config.seed,
            return_keep_mask=self.config.position_mode == "original",
        )
        if self.config.position_mode == "original":
            from src.pruning import _build_keep_indices

            n_before = vis_end - vis_start
            if result[0].shape[1] == hidden.shape[1]:
                keep_mask = torch.ones(n_before, dtype=torch.bool, device=hidden.device)
            elif len(result) >= 5 and isinstance(result[4], torch.Tensor):
                keep_mask = result[4]
            else:
                raise ValueError("Selector did not return its actual survivor mask")
            if keep_mask.dtype != torch.bool or keep_mask.numel() != n_before:
                raise ValueError("Selector returned an invalid survivor mask")
            self._last_full_keep_indices = _build_keep_indices(
                keep_mask, vis_start, vis_end, hidden.shape[1], hidden.device,
            )
            return (*result[:4], vis_positions[keep_mask])
        return (*result[:4], vis_positions)

    def _greedy_decode(
        self,
        prefill_logits: torch.Tensor,
        cache: DynamicCache,
        max_new_tokens: int,
    ) -> list[int]:
        """Simple greedy autoregressive decoding using pre-built KV cache.

        Args:
            prefill_logits: (1, seq_len, vocab_size) logits from prefill
            cache: DynamicCache from LATC prefill
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of generated token IDs
        """
        lm = self.model.language_model
        layers = lm.model.layers
        generated = []

        # Build per-layer position base to fix RoPE inconsistency after LATC merge.
        # Layer i's KV was encoded with positions from the seq_len at the time that
        # layer ran during prefill.  Layers at or before a merge layer used the
        # pre-merge seq_len; layers after used the post-merge (trimmed) seq_len.
        # _merge_checkpoints = [(merge_layer_idx, pre_merge_seq_len), ...] (ascending)
        # _prefill_trimmed_len = seq_len after all merges.
        checkpoints = getattr(self, "_merge_checkpoints", [])
        trimmed_base = getattr(self, "_prefill_trimmed_len", cache.get_seq_length(0))

        def _pos_base_for_layer(layer_idx: int) -> int:
            """Return the position base for a given layer (= its prefill seq_len)."""
            if self.config.position_mode == "original":
                return self._original_prefill_len
            for merge_layer, pre_len in sorted(checkpoints, key=lambda x: x[0]):
                if merge_layer >= layer_idx:
                    return pre_len
            return trimmed_base

        # Pre-compute the position base used by each decoder layer.
        layer_to_base = [_pos_base_for_layer(i) for i in range(len(layers))]

        # First token from prefill logits
        next_token_id = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)

        for step in range(max_new_tokens):
            token_id = next_token_id.item()
            if token_id == self.eos_token_id:
                break
            generated.append(token_id)

            # Embed the new token
            h = lm.get_input_embeddings()(next_token_id)  # (1, 1, dim)

            # Cache-length at this step (= trimmed_base + #tokens-already-generated).
            # Used as position for layers that saw the trimmed sequence.
            cur_len = cache.get_seq_length(0)

            # Pre-compute position_embeddings for each unique base.
            unique_bases = set(layer_to_base)
            pos_emb_cache: dict[int, tuple] = {}
            for base in unique_bases:
                pos = base + (cur_len - trimmed_base)
                pid = torch.tensor([[pos]], device=self.device)
                pos_emb_cache[base] = lm.model.rotary_emb(h, pid)

            # Forward through all layers with per-layer correct position embeddings.
            for i, layer in enumerate(layers):
                position_embeddings = pos_emb_cache[layer_to_base[i]]
                outputs = layer(
                    h,
                    position_embeddings=position_embeddings,
                    past_key_values=cache,
                    use_cache=True,
                )
                h = outputs[0] if isinstance(outputs, tuple) else outputs

            # Norm + lm_head
            h = lm.model.norm(h)
            logits = lm.lm_head(h)
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        return generated
