"""TwinPrune prefill selection, including its optional timing backend."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from src.density import compute_spatial_density


def _infer_spatial_shape(n: int) -> tuple[int, int]:
    """Find best H x W factorization close to square."""
    h = int(math.isqrt(n))
    while h > 1 and n % h != 0:
        h -= 1
    return (h, n // h) if h > 1 else (1, n)


def _cac_v4_legacy_loop(
    visual_tokens,
    keep_ratio,
    spatial_shape,
    beta,
    density=None,
):
    """Reference Python-loop implementation kept as fallback for CUDA Graph
    failures or tiny K where capture overhead exceeds savings."""
    n = visual_tokens.shape[0]
    n_keep = max(1, int(n * keep_ratio))
    eps = 1e-8

    if density is None:
        density = compute_spatial_density(
            visual_tokens,
            spatial_shape,
        ).float()
    else:
        density = density.to(
            device=visual_tokens.device,
            dtype=torch.float32,
        )
    d_norm = (density - density.min()) / (density.max() - density.min() + eps)

    norm_v = F.normalize(visual_tokens.float(), dim=-1)
    dist = 1.0 - norm_v @ norm_v.T
    dist.fill_diagonal_(0.0)

    seed = int(density.argmax().item())
    selected = [seed]
    min_dist = dist[seed].clone()
    for _ in range(n_keep - 1):
        min_dist[selected] = -1.0
        score = min_dist + beta * d_norm
        pick = int(score.argmax().item())
        selected.append(pick)
        min_dist = torch.minimum(min_dist, dist[pick])

    keep_idx = torch.tensor(selected, device=visual_tokens.device, dtype=torch.long).sort().values
    return visual_tokens[keep_idx], keep_idx


def cac_v4_density_tilted_fps(
    visual_tokens: torch.Tensor,
    keep_ratio: float,
    spatial_shape: tuple[int, int] | None = None,
    beta: float = 0.3,
    precomputed_density: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select sorted survivors using density-tilted farthest-point sampling.

    The main-table accuracy protocol uses the legacy backend. The Triton
    backend is used for the separately specified batched timing protocol.
    """
    n = visual_tokens.shape[0]
    if n <= 1 or keep_ratio >= 1.0:
        return visual_tokens, torch.arange(n, device=visual_tokens.device)
    if spatial_shape is None:
        spatial_shape = _infer_spatial_shape(n)
    if spatial_shape[0] * spatial_shape[1] != n:
        spatial_shape = (1, n)
    if precomputed_density is None:
        density = compute_spatial_density(
            visual_tokens,
            spatial_shape,
        ).float()
    else:
        if precomputed_density.numel() != n:
            raise ValueError(
                "precomputed density length does not match visual tokens"
            )
        density = precomputed_density.to(
            device=visual_tokens.device,
            dtype=torch.float32,
        )

    # CPU / non-CUDA / tiny budget: fall back to legacy loop (graph overhead
    # exceeds savings; legacy is also the only path that runs on CPU).
    backend = os.environ.get(
        "TWINPRUNE_FPS_BACKEND",
        os.environ.get("CAC_FPS_BACKEND", "graph"),
    ).lower()
    if (
        not visual_tokens.is_cuda
        or os.environ.get("CAC_FPS_DISABLE_GRAPH") == "1"
        or backend == "legacy"
        or max(1, int(n * keep_ratio)) < 32
    ):
        return _cac_v4_legacy_loop(
            visual_tokens,
            keep_ratio,
            spatial_shape,
            beta,
            density=density,
        )

    # Triton backend: 2 fused kernels per FPS step, captured into a CUDA Graph.
    # Falls through to the cuBLAS-graph path on any error.
    if backend == "triton":
        try:
            from src.cac_v4_triton import cac_v4_density_tilted_fps_triton
            return cac_v4_density_tilted_fps_triton(
                visual_tokens, keep_ratio, spatial_shape, beta=beta,
                compute_density_fn=compute_spatial_density,
                infer_spatial_fn=_infer_spatial_shape,
                precomputed_density=density,
            )
        except Exception:
            if os.environ.get("TWINPRUNE_STRICT_BACKEND") == "1":
                raise

    n_keep = max(1, int(n * keep_ratio))
    eps = 1e-8
    device = visual_tokens.device
    D = visual_tokens.shape[1]

    d_norm = (density - density.min()) / (density.max() - density.min() + eps)
    norm_v = F.normalize(visual_tokens.float(), dim=-1)

    seed = density.argmax()
    selected_t = torch.empty(n_keep, dtype=torch.long, device=device)
    selected_t[0] = seed

    init_dist = 1.0 - (norm_v @ norm_v[seed].unsqueeze(-1)).squeeze(-1)
    init_dist[seed] = -1.0
    if n_keep == 1:
        return visual_tokens[selected_t.sort().values], selected_t.sort().values

    # Static buffers for graph capture
    min_dist = torch.empty(n, device=device, dtype=torch.float32)
    score_buf = torch.empty(n, device=device, dtype=torch.float32)
    new_dist_buf = torch.empty(n, device=device, dtype=torch.float32)
    pick_row_buf = torch.empty(D, device=device, dtype=torch.float32)
    pick_t = torch.zeros((), device=device, dtype=torch.long)
    step_t = torch.zeros(1, device=device, dtype=torch.long)
    neg_one = torch.tensor(-1.0, device=device, dtype=torch.float32)

    try:
        # Warm up on a side stream (required for CUDA Graph capture).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            min_dist.copy_(init_dist)
            for _ in range(3):
                torch.add(min_dist, d_norm, alpha=beta, out=score_buf)
                torch.argmax(score_buf, out=pick_t)
                torch.index_select(norm_v, 0, pick_t.unsqueeze(0), out=pick_row_buf.unsqueeze(0))
                torch.matmul(norm_v, pick_row_buf.unsqueeze(-1), out=new_dist_buf.unsqueeze(-1))
                new_dist_buf.neg_().add_(1.0)
                torch.minimum(min_dist, new_dist_buf, out=min_dist)
                min_dist.scatter_(0, pick_t.unsqueeze(0), neg_one.unsqueeze(0))
        torch.cuda.current_stream().wait_stream(s)

        # Re-init buffers for the captured graph.
        selected_t[0] = seed
        min_dist.copy_(init_dist)
        step_t.fill_(1)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            torch.add(min_dist, d_norm, alpha=beta, out=score_buf)
            torch.argmax(score_buf, out=pick_t)
            selected_t.scatter_(0, step_t, pick_t.unsqueeze(0))
            step_t.add_(1)
            torch.index_select(norm_v, 0, pick_t.unsqueeze(0), out=pick_row_buf.unsqueeze(0))
            torch.matmul(norm_v, pick_row_buf.unsqueeze(-1), out=new_dist_buf.unsqueeze(-1))
            new_dist_buf.neg_().add_(1.0)
            torch.minimum(min_dist, new_dist_buf, out=min_dist)
            min_dist.scatter_(0, pick_t.unsqueeze(0), neg_one.unsqueeze(0))

        for _ in range(n_keep - 1):
            g.replay()
    except Exception:
        # If CUDA Graph capture fails for any reason (driver, concurrent
        # capture, etc.), fall back to the verified legacy loop.
        if os.environ.get("TWINPRUNE_STRICT_BACKEND") == "1":
            raise
        return _cac_v4_legacy_loop(
            visual_tokens,
            keep_ratio,
            spatial_shape,
            beta,
            density=density,
        )

    keep_idx = selected_t.sort().values
    return visual_tokens[keep_idx], keep_idx
