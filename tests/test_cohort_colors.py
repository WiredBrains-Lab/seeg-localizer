import unittest
import numpy as np
from viewer3d.cohort_colors import symmetric_color_limit, scalar_colormap


class ColorTests(unittest.TestCase):
    def test_robust_range_reduces_outlier_influence(self):
        values = [0.01] * 99 + [10, np.nan, np.inf]
        self.assertEqual(symmetric_color_limit(values), 10)
        self.assertAlmostEqual(symmetric_color_limit(values, 95), 0.01)
        self.assertEqual(symmetric_color_limit([-0.2, 0.1]), 0.2)

    def test_degenerate_ranges_and_tiny_values(self):
        self.assertEqual(symmetric_color_limit([]), 1)
        self.assertEqual(symmetric_color_limit([np.nan, 0]), 1)
        self.assertEqual(symmetric_color_limit([0] * 99 + [0.1], 95), 0.1)
        self.assertEqual(symmetric_color_limit([1e-10, -2e-10]), 2e-10)
        with self.assertRaises(ValueError):
            symmetric_color_limit([1], 101)

    def test_enhancement_preserves_zero_endpoints_and_expands_small_values(self):
        from matplotlib import colormaps
        cmap = scalar_colormap(True)
        base = colormaps['RdBu_r']
        np.testing.assert_allclose(cmap([0., 0.5, 1.]), base([0., 0.5, 1.]), atol=0.01)
        neutral = np.array(base(0.5))[:3]
        self.assertGreater(np.linalg.norm(np.array(cmap(0.55))[:3] - neutral),
                           np.linalg.norm(np.array(base(0.55))[:3] - neutral))
