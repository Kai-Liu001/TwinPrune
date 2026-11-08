"""Public entry points for the TwinPrune release candidate."""

__version__ = "0.1.0rc1"


def spatial_density(tokens, spatial_shape):
    from src.density import compute_spatial_density
    return compute_spatial_density(tokens, spatial_shape)
