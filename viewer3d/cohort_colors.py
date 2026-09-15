"""Scalar display ranges and color enhancement without changing source values."""
from functools import lru_cache
import numpy as np


def symmetric_color_limit(values, percentile=100.0):
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be in (0, 100]")
    values = np.asarray(values, dtype=float)
    magnitudes = np.abs(values[np.isfinite(values)])
    if not len(magnitudes):
        return 1.0
    limit = float(np.percentile(magnitudes, percentile))
    # Degenerate percentiles (e.g. mostly zeros) must not produce a zero range.
    return limit or float(magnitudes.max()) or 1.0


@lru_cache(maxsize=2)
def scalar_colormap(enhance=False):
    if not enhance:
        return "RdBu_r"
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap
    positions = np.linspace(-1, 1, 1025)
    positions = np.arcsinh(10 * positions) / np.arcsinh(10)
    # Warp the color lookup table, not the data: legend ticks retain actual units.
    return ListedColormap(colormaps['RdBu_r']((positions + 1) / 2), name='RdBu_r_asinh10')
