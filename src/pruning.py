"""TwinPrune hidden-state and KV-cache trimming."""

import torch
from transformers import DynamicCache


def _build_keep_indices(
    keep_mask: torch.Tensor,
    vis_start: int,
    vis_end: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    pre_vis = list(range(vis_start))
    surviving_vis = [vis_start + i for i, keep in enumerate(keep_mask.tolist()) if keep]
    post_vis = list(range(vis_end, seq_len))
    return torch.tensor(pre_vis + surviving_vis + post_vis, device=device, dtype=torch.long)


def _trim_cache(cache: DynamicCache, keep_indices: torch.Tensor, current_layer: int) -> None:
    for layer_idx in range(current_layer + 1):
        layer_cache = cache.layers[layer_idx]
        layer_cache.keys = layer_cache.keys[:, :, keep_indices, :]
        layer_cache.values = layer_cache.values[:, :, keep_indices, :]


def _apply_selector(
    variant_fn,
    hidden: torch.Tensor,
    vis_start: int,
    vis_end: int,
    ratio: float,
    cache: DynamicCache,
    current_layer: int,
    variant_kwargs: dict | None = None,
    return_keep_mask: bool = False,
):
    """Apply the selector and trim the visual span in the hidden states and cache."""
    with torch.no_grad():
        vis_tokens = hidden[0, vis_start:vis_end, :]  # [N, D]
        n_vis = vis_tokens.shape[0]
        n_remove = max(0, int(n_vis * ratio))
        if n_remove <= 0 or n_vis <= 1:
            return hidden, vis_start, vis_end, n_vis

        keep_ratio = 1.0 - ratio
        compressed, kept_indices = variant_fn(
            vis_tokens, keep_ratio, **(variant_kwargs or {})
        )
        n_kept = compressed.shape[0]

        hidden = torch.cat(
            [hidden[:, :vis_start, :], compressed.unsqueeze(0), hidden[:, vis_end:, :]],
            dim=1,
        )

        keep_mask = torch.zeros(n_vis, dtype=torch.bool, device=hidden.device)
        keep_mask[kept_indices] = True

        cache_keep_indices = _build_keep_indices(
            keep_mask=keep_mask,
            vis_start=vis_start,
            vis_end=vis_end,
            seq_len=cache.get_seq_length(0),
            device=hidden.device,
        )
        _trim_cache(cache, cache_keep_indices, current_layer)

        if return_keep_mask:
            return hidden, vis_start, vis_start + n_kept, n_kept, keep_mask
        return hidden, vis_start, vis_start + n_kept, n_kept


def apply_cac_v4(
    hidden: torch.Tensor,
    vis_start: int,
    vis_end: int,
    ratio: float,
    cache: DynamicCache,
    current_layer: int,
    model,
    seed: int = 42,
    beta: float = 0.3,
    **kwargs,
):
    """Apply TwinPrune prefill pruning after the configured decoder layer."""
    return_keep_mask = kwargs.get("return_keep_mask", False)
    precomputed_density = kwargs.get("precomputed_density")
    del model, seed, kwargs
    from src.selection import cac_v4_density_tilted_fps
    return _apply_selector(
        cac_v4_density_tilted_fps, hidden, vis_start, vis_end, ratio, cache, current_layer,
        variant_kwargs={
            "beta": beta,
            "precomputed_density": precomputed_density,
        },
        return_keep_mask=return_keep_mask,
    )


STRATEGIES = {"cac_v4": apply_cac_v4}
