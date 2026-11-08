"""Triton-fused FPS step for ``cac_v4`` density-tilted variant.

Replaces the per-step CUDA Graph body in ``cac_v4_density_tilted_fps``
with two Triton kernels (matmul-update + score-argmax), captured into a
CUDA Graph and replayed K-1 times. Reduces per-step kernel-launch count
from 5 to 2 and keeps all per-iteration compute on-GPU.

Numerics: not bit-identical to the legacy fp32 cuBLAS path because the
matmul reduction order differs (Triton tile-sum vs cuBLAS).  Selection
indices may differ when tokens have near-equal scores. Main-table accuracy
therefore uses the separate legacy backend; this backend is used for timing.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Module-level CUDA Graph cache.
# Key:   (n, D, n_keep, beta_x100, dev_idx, dtype_str, block_m, block_k)
# Value: dict with the captured graph + reusable buffers.
# Only Triton path uses the cache; legacy graph path remains unchanged for
# clean fallback semantics.
_FPS_GRAPH_CACHE: dict = {}


@triton.jit
def _fps_argmax_kernel(
    min_dist_ptr,    # *fp32 [N]
    d_norm_ptr,      # *fp32 [N]
    selected_ptr,    # *i64 [n_keep]
    step_ptr,        # *i64 [1]
    pick_ptr,        # *i64 [1]
    beta,            # fp32 scalar
    N,               # i32 scalar
    BLOCK: tl.constexpr,  # power-of-2 >= N
):
    """Single-block argmax of ``min_dist + beta * d_norm``.

    Writes pick to ``pick_ptr``, also writes ``selected[step] = pick``
    and increments ``step`` (in-place).
    """
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    md = tl.load(min_dist_ptr + offs, mask=mask, other=-float('inf'))
    dn = tl.load(d_norm_ptr + offs, mask=mask, other=-float('inf'))
    score = md + beta * dn
    # tl.argmax returns the index of the maximum (ties broken by first occurrence).
    pick_idx = tl.argmax(score, axis=0)
    pick64 = pick_idx.to(tl.int64)

    step = tl.load(step_ptr)
    tl.store(selected_ptr + step, pick64)
    tl.store(pick_ptr, pick64)
    tl.store(step_ptr, step + 1)


@triton.jit
def _fps_update_kernel(
    norm_v_ptr,      # *fp16 [N, D] contiguous (half memory bandwidth vs fp32)
    pick_ptr,        # *i64 [1]
    min_dist_ptr,    # *fp32 [N]  in-place
    N,
    D,
    stride_n,        # row stride (= D for contiguous)
    BLOCK_M: tl.constexpr,  # rows per program
    BLOCK_K: tl.constexpr,  # cols per inner iteration
):
    """For ``BLOCK_M`` rows, compute ``dot(norm_v[i], norm_v[pick])`` and
    update ``min_dist[i] = min(min_dist[i], 1 - dot)``.  If a row equals
    ``pick``, override to ``-1`` so the next argmax cannot re-pick it.

    norm_v is fp16 (10 mantissa bits → 0.001 cosine resolution, ~16× finer
    than the random-unit-vector cosine std at D=4096). Loads cast to fp32
    inline; accumulator stays fp32 so argmax precision is preserved.
    """
    pid = tl.program_id(0)
    pick = tl.load(pick_ptr)

    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < N

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in range(0, D, BLOCK_K):
        cols = k + tl.arange(0, BLOCK_K)
        col_mask = cols < D
        v = tl.load(
            norm_v_ptr + rows[:, None] * stride_n + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        p = tl.load(
            norm_v_ptr + pick * stride_n + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(v * p[None, :], axis=1)

    new_dist = 1.0 - acc
    md = tl.load(min_dist_ptr + rows, mask=row_mask, other=float('inf'))
    new_md = tl.minimum(md, new_dist)
    new_md = tl.where(rows == pick, -1.0, new_md)
    tl.store(min_dist_ptr + rows, new_md, mask=row_mask)


# ---------------------------------------------------------------------------
# Public entry point


def _build_graph_entry(
    n, D, n_keep, beta_f, device, block_m, block_k,
):
    """Allocate static buffers + capture a fresh CUDA Graph for this shape.

    norm_v_buf is fp16. With normalized unit vectors all values fit in
    [-1, 1], so the limited fp16 exponent range is fine; the 10 mantissa
    bits give 1e-3 cosine resolution, which is ~16× finer than the
    random-input cosine std at D=4096 — enough to keep argmax stable.
    """
    norm_v_buf = torch.empty((n, D), device=device, dtype=torch.float16)
    d_norm_buf = torch.empty(n, device=device, dtype=torch.float32)
    min_dist_buf = torch.empty(n, device=device, dtype=torch.float32)
    pick_t_buf = torch.empty(1, device=device, dtype=torch.int64)
    selected_t_buf = torch.empty(n_keep, device=device, dtype=torch.int64)
    step_t_buf = torch.empty(1, device=device, dtype=torch.int64)

    BLOCK_N = triton.next_power_of_2(n)
    grid_M = (n + block_m - 1) // block_m

    # Pre-fill buffers with deterministic values for warmup (any non-NaN ok).
    norm_v_buf.normal_()
    d_norm_buf.zero_()
    min_dist_buf.fill_(1.0)
    min_dist_buf[0] = -1.0
    pick_t_buf.zero_()
    selected_t_buf.zero_()
    step_t_buf.fill_(1)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _fps_argmax_kernel[(1,)](
                min_dist_buf, d_norm_buf, selected_t_buf, step_t_buf, pick_t_buf,
                beta_f, n, BLOCK=BLOCK_N,
            )
            _fps_update_kernel[(grid_M,)](
                norm_v_buf, pick_t_buf, min_dist_buf,
                n, D, D,
                BLOCK_M=block_m, BLOCK_K=block_k,
            )
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _fps_argmax_kernel[(1,)](
            min_dist_buf, d_norm_buf, selected_t_buf, step_t_buf, pick_t_buf,
            beta_f, n, BLOCK=BLOCK_N,
        )
        _fps_update_kernel[(grid_M,)](
            norm_v_buf, pick_t_buf, min_dist_buf,
            n, D, D,
            BLOCK_M=block_m, BLOCK_K=block_k,
        )

    return {
        "graph": g,
        "norm_v": norm_v_buf,
        "d_norm": d_norm_buf,
        "min_dist": min_dist_buf,
        "pick_t": pick_t_buf,
        "selected_t": selected_t_buf,
        "step_t": step_t_buf,
        "block_m": block_m,
        "block_k": block_k,
    }


def cac_v4_density_tilted_fps_triton(
    visual_tokens: torch.Tensor,
    keep_ratio: float,
    spatial_shape: tuple[int, int] | None = None,
    beta: float = 0.3,
    *,
    compute_density_fn,
    infer_spatial_fn,
    precomputed_density: torch.Tensor | None = None,
    block_m: int = 32,
    block_k: int = 64,
):
    """Triton-fused variant of ``cac_v4_density_tilted_fps``.

    Caches the captured CUDA Graph keyed on ``(n, D, n_keep, beta)``.
    Repeat calls with the same shape skip warmup + capture and pay only
    K-1 replays.
    """
    n = visual_tokens.shape[0]
    if n <= 1 or keep_ratio >= 1.0:
        return visual_tokens, torch.arange(n, device=visual_tokens.device)
    if spatial_shape is None:
        spatial_shape = infer_spatial_fn(n)
    if spatial_shape[0] * spatial_shape[1] != n:
        spatial_shape = (1, n)

    n_keep = max(1, int(n * keep_ratio))
    if not visual_tokens.is_cuda or n_keep < 32:
        raise RuntimeError("Triton FPS requires CUDA and n_keep>=32")

    eps = 1e-8
    device = visual_tokens.device
    D = visual_tokens.shape[1]
    beta_f = float(beta)

    density = (
        compute_density_fn(visual_tokens, spatial_shape).float()
        if precomputed_density is None
        else precomputed_density.to(
            device=visual_tokens.device,
            dtype=torch.float32,
        )
    )
    d_norm = (density - density.min()) / (density.max() - density.min() + eps)
    norm_v_fp32 = F.normalize(visual_tokens.float(), dim=-1).contiguous()
    norm_v_fp16 = norm_v_fp32.to(torch.float16)
    seed = density.argmax()

    # Seed init in fp32 — the very first matmul (norm_v_fp32 @ norm_v_fp32[seed])
    # uses fp32 to match legacy bit-exact at step 0; subsequent K-1 graph
    # replays use fp16 norm_v.
    init_dist = 1.0 - (norm_v_fp32 @ norm_v_fp32[seed].unsqueeze(-1)).squeeze(-1)
    init_dist[seed] = -1.0

    if n_keep == 1:
        keep_idx = seed.unsqueeze(0)
        return visual_tokens[keep_idx], keep_idx

    cache_key = (n, D, n_keep, int(round(beta_f * 1000)),
                 device.index if device.type == "cuda" else -1,
                 block_m, block_k)
    entry = _FPS_GRAPH_CACHE.get(cache_key)
    if entry is None:
        entry = _build_graph_entry(n, D, n_keep, beta_f, device, block_m, block_k)
        _FPS_GRAPH_CACHE[cache_key] = entry

    # Copy this call's data into the static buffers captured by the graph.
    entry["norm_v"].copy_(norm_v_fp16)
    entry["d_norm"].copy_(d_norm)
    entry["min_dist"].copy_(init_dist)
    entry["pick_t"][0] = seed
    entry["selected_t"][0] = seed
    entry["step_t"][0] = 1

    g = entry["graph"]
    for _ in range(n_keep - 1):
        g.replay()

    keep_idx = entry["selected_t"].sort().values
    return visual_tokens[keep_idx], keep_idx
