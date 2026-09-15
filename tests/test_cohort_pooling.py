import unittest
import numpy as np
from viewer3d.cohort_pooling import pool_mni_contacts


class PoolingTests(unittest.TestCase):
    def test_chain_does_not_bridge_distant_contacts(self):
        points = np.array([[0, 0, 0], [6, 0, 0], [12, 0, 0]])
        groups = pool_mni_contacts(points, [1, 3, 5], ['a', 'b', 'c'], 10)
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(i for g in groups for i in g['members']), [0, 1, 2])
        for group in groups:
            coords = points[group['members']]
            self.assertLessEqual(np.linalg.norm(coords[:, None] - coords, axis=2).max(), 10)

    def test_finite_mean_centroid_and_counts(self):
        group, = pool_mni_contacts([[0, 0, 0], [2, 0, 0], [4, 0, 0]],
                                  [1, np.nan, 5], ['a', 'a', 'b'])
        self.assertEqual(group['mean'], 3)
        self.assertEqual(group['center'], [2, 0, 0])
        self.assertEqual(group['value_count'], 2)
        self.assertEqual(group['patient_count'], 2)
        self.assertEqual(group['contact_count'], 3)

    def test_empty_singleton_missing_and_boundary(self):
        self.assertEqual(pool_mni_contacts([], [], []), [])
        group, = pool_mni_contacts([[0, 0, 0]], [np.nan], ['a'])
        self.assertIsNone(group['mean'])
        self.assertEqual(len(pool_mni_contacts([[0, 0, 0], [10, 0, 0]], [1, 2], ['a', 'b'])), 1)
        with self.assertRaises(ValueError):
            pool_mni_contacts([], [], [], 0)

    def test_order_and_scalar_independent_membership(self):
        points = [[0, 0, 0], [6, 0, 0], [12, 0, 0]]
        a = pool_mni_contacts(points, [1, 2, 3], ['a'] * 3)
        b = pool_mni_contacts(points[::-1], [np.nan] * 3, ['a'] * 3)
        self.assertEqual([g['center'] for g in a], [g['center'] for g in b])
