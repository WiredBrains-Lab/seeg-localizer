import unittest
import numpy as np
from viewer3d.cohort_smoothing import CorticalGaussianField


class SmoothingTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0],
                                  [0, 5, 0], [5, 5, 0], [10, 5, 0]])
        self.faces = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
        self.field = CorticalGaussianField(self.vertices, self.faces)

    def test_weighted_average_is_symmetric_and_bounded(self):
        means, alpha = self.field.evaluate(self.vertices[[0, 2]], [-1, 1], 15)
        self.assertAlmostEqual(means[1], 0)
        self.assertLess(means[0], 0)
        self.assertGreater(means[2], 0)
        self.assertTrue(np.all(np.abs(means) <= 1))
        self.assertEqual(alpha[0], 1)
        self.assertGreater(alpha[1], 0)
        self.assertLess(alpha[1], 1)

    def test_missing_values_and_duplicate_sites(self):
        means, alpha = self.field.evaluate(self.vertices[[0, 0, 2]], [1, 3, np.nan])
        np.testing.assert_allclose(means[np.isfinite(means)], 2)
        _, single_alpha = self.field.evaluate(self.vertices[[0]], [2])
        np.testing.assert_allclose(alpha, single_alpha)
        means, alpha = self.field.evaluate(self.vertices[[0]], [np.nan])
        self.assertTrue(np.isnan(means).all())
        self.assertFalse(alpha.any())

    def test_width_support_and_kernel_cache(self):
        _, narrow = self.field.evaluate(self.vertices[[0]], [1], 2)
        self.assertEqual(narrow[2], 0)
        _, broad = self.field.evaluate(self.vertices[[0]], [1], 20)
        self.assertGreater(broad[2], 0)
        kernel = self.field._kernel
        self.field.evaluate(self.vertices[[0]], [-3], 20)
        self.assertIs(kernel, self.field._kernel)

    def test_does_not_cross_disconnected_surface(self):
        vertices = np.vstack([self.vertices, self.vertices + [0, 0, 0.1]])
        field = CorticalGaussianField(vertices, np.vstack([self.faces, self.faces + 6]))
        means, alpha = field.evaluate([[0, 0, 0]], [3], 50)
        self.assertTrue(np.isnan(means[6:]).all())
        self.assertFalse(alpha[6:].any())

    def test_empty_and_invalid_inputs(self):
        means, alpha = self.field.evaluate([], [])
        self.assertTrue(np.isnan(means).all())
        self.assertFalse(alpha.any())
        with self.assertRaises(ValueError):
            self.field.evaluate([[0, 0, 0]], [1], 0)
        with self.assertRaises(ValueError):
            self.field.evaluate([[np.nan, 0, 0]], [1])
