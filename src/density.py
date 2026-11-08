"""Multi-scale spatial density used by both TwinPrune phases."""

import torch
import torch.nn.functional as F


def compute_spatial_density(
    tokens: torch.Tensor,
    spatial_shape: tuple[int, int],
    window_sizes: list[int] | None = None,
    entropy_normalize: bool = False,
) -> torch.Tensor:
    """Compute per-token information density via multi-scale cosine dissimilarity.

    For each token, at each window size, computes mean cosine similarity with
    spatial neighbors.  density = 1 - mean_sim.  Takes max across scales.

    Args:
        tokens: [N, D] visual token embeddings.
        spatial_shape: (H, W) spatial grid dimensions.
        window_sizes: list of odd window sizes (default [3, 5, 7]).
        entropy_normalize: if True, divide the raw density by a local-entropy
            proxy estimated as the std-dev of cosine similarities within the
            window.  Intuition: regions of globally high variance already
            score high on raw density regardless of local structure, which
            biases selection toward "noisy" regions; dividing by local entropy
            corrects this so density reflects *local* informativeness (Fix 7).

    Returns:
        [N] tensor of density scores in [0, 1] (or rescaled if normalized).
    """
    if window_sizes is None:
        window_sizes = [3, 5, 7]

    N, D = tokens.shape
    H, W = spatial_shape
    assert H * W == N, f"spatial_shape {spatial_shape} doesn't match N={N}"

    if N <= 1:
        return tokens.new_zeros(N)

    # Normalize for cosine similarity
    normed = F.normalize(tokens.float(), dim=-1)  # [N, D]
    grid = normed.view(1, H, W, D).permute(0, 3, 1, 2)  # [1, D, H, W]

    density_scales = []

    for ws in window_sizes:
        pad = ws // 2

        # Neighbor count per position (excluding self)
        ones = torch.ones(1, 1, H, W, device=tokens.device, dtype=torch.float32)
        neighbor_sum = F.avg_pool2d(
            F.pad(ones, (pad, pad, pad, pad), mode="constant", value=0.0),
            kernel_size=ws, stride=1, padding=0,
        ) * (ws * ws)  # [1, 1, H, W]
        n_neighbors = (neighbor_sum - 1.0).clamp(min=1.0)  # exclude self

        # Sum of cosine sims = sum_j (x_i . x_j) for j in window
        # = x_i . (sum_j x_j for j in window)
        # We compute local sum via avg_pool * window_area
        padded = F.pad(grid, (pad, pad, pad, pad), mode="constant", value=0.0)
        local_sum = F.avg_pool2d(padded, kernel_size=ws, stride=1, padding=0) * (ws * ws)
        # local_sum: [1, D, H, W]

        # Dot product of each token with its local sum
        dot = (grid * local_sum).sum(dim=1, keepdim=True)  # [1, 1, H, W]

        # Subtract self-similarity (which is 1.0 for normalized vectors)
        dot = dot - 1.0

        # Mean similarity with neighbors
        mean_sim = dot.squeeze() / n_neighbors.squeeze()  # [H, W]
        density = (1.0 - mean_sim).clamp(0.0, 1.0)
        density_scales.append(density.reshape(N))

    stacked = torch.stack(density_scales, dim=0)  # [n_scales, N]
    result = stacked.max(dim=0).values

    if entropy_normalize:
        # Proxy for local entropy: std-dev of cosine similarities across the
        # three window sizes (captures how much the neighborhood varies with
        # scale).  Small std means the neighborhood is self-consistent at all
        # scales (unambiguous density); large std means the density estimate
        # itself is noisy, so we dampen it.
        local_std = stacked.std(dim=0).clamp(min=1e-4)  # [N]
        # Divide raw density by (1 + local_std).  local_std in [0, ~0.3]
        # typical range, so divisor is in [1, 1.3].  Preserves ordering in
        # consistent regions and dampens in inconsistent ones.
        result = result / (1.0 + local_std)
        # Rescale to [0, max] to preserve the dynamic range used downstream
        max_val = result.max()
        if max_val > 0:
            result = result * (stacked.max() / max_val)

    return result.to(tokens.dtype)
