"""Density-guided visual KV eviction with all text tokens retained."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from transformers import DynamicCache

from src.density import compute_spatial_density


def _infer_spatial_shape(n: int) -> Tuple[int, int]:
    h = int(math.isqrt(n))
    while h > 1 and n % h != 0:
        h -= 1
    return (h, n // h) if h > 1 else (1, n)


def _build_keep_indices(
    vis_keep_local: torch.Tensor,  # indices within [0, n_vis)
    vis_start: int,
    vis_end: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build full cache keep-indices for trimming."""
    pre = torch.arange(vis_start, device=device, dtype=torch.long)
    vis_keep = (vis_start + vis_keep_local.sort().values).to(device).long()
    post = torch.arange(vis_end, seq_len, device=device, dtype=torch.long)
    return torch.cat([pre, vis_keep, post])


def _trim_cache(cache: DynamicCache, keep_indices: torch.Tensor) -> None:
    """Trim all layers of the DynamicCache to the given indices.

    For multi-GPU sharded models (device_map='auto'), K/V of different layers
    may live on different devices; move the keep_indices tensor to each
    layer's device before indexing.
    """
    n_layers = len(cache)
    for i in range(n_layers):
        layer = cache.layers[i] if hasattr(cache, "layers") else None
        if layer is not None and hasattr(layer, "keys"):
            idx = keep_indices.to(layer.keys.device)
            layer.keys = layer.keys[:, :, idx, :].contiguous()
            layer.values = layer.values[:, :, idx, :].contiguous()
        else:
            idx = keep_indices.to(cache.key_cache[i].device)
            cache.key_cache[i] = cache.key_cache[i][:, :, idx, :].contiguous()
            cache.value_cache[i] = cache.value_cache[i][:, :, idx, :].contiguous()


def density_eviction(
    visual_tokens: torch.Tensor,  # [n_vis, D]
    keep_ratio: float,
    spatial_shape: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """Return local indices (within visual block) to keep, by density top-k."""
    n = visual_tokens.shape[0]
    n_keep = max(1, int(n * keep_ratio))
    if n_keep >= n:
        return torch.arange(n, device=visual_tokens.device)
    if spatial_shape is None or spatial_shape[0] * spatial_shape[1] != n:
        spatial_shape = _infer_spatial_shape(n)
        if spatial_shape[0] * spatial_shape[1] != n:
            spatial_shape = (1, n)
    density = compute_spatial_density(visual_tokens, spatial_shape)
    keep_idx = torch.topk(density, n_keep, largest=True).indices
    return keep_idx


def evict_visual_kv(
    cache: DynamicCache,
    visual_tokens: torch.Tensor,  # for density eviction
    vis_start: int,
    vis_end: int,
    keep_ratio: float,
    strategy: str = "density",
    spatial_shape: Tuple[int, int] | None = None,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """Perform decode-phase visual KV eviction on the cache IN-PLACE.

    Returns (original_seq_length, trimmed_seq_length, n_kept_visual).
    The caller must use original_seq_length for position_ids during decoding
    to preserve RoPE relative-position consistency with remaining KV entries.
    """
    device = visual_tokens.device
    n_vis = vis_end - vis_start
    original_len = cache.get_seq_length(0)

    if keep_ratio >= 1.0 or n_vis <= 1:
        return original_len, original_len, n_vis

    if keep_ratio <= 0.0:
        # Trim ALL visual tokens from cache — keep only pre-visual and post-visual text
        keep_indices = torch.cat([
            torch.arange(0, vis_start, device=device, dtype=torch.long),
            torch.arange(vis_end, original_len, device=device, dtype=torch.long),
        ])
        _trim_cache(cache, keep_indices)
        return original_len, cache.get_seq_length(0), 0

    if strategy != "density":
        raise ValueError("This release supports density KV eviction only")
    vis_keep_local = density_eviction(visual_tokens, keep_ratio, spatial_shape)

    keep_indices = _build_keep_indices(
        vis_keep_local, vis_start, vis_end, original_len, device
    )
    _trim_cache(cache, keep_indices)

    trimmed_len = cache.get_seq_length(0)
    n_kept = vis_keep_local.numel()
    return original_len, trimmed_len, n_kept
