"""Gaussian-weighted descriptive scalar fields on a cortical mesh."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


class CorticalGaussianField:
    """Smooth on white-surface mesh-edge distances, in anatomical millimetres.

    Contacts project to their nearest white vertex. Kernel rows are vertices,
    columns are unique projected contact sites. Cache only the latest kernel;
    scalar changes and missing-value changes reuse it without recomputing paths.
    """

    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not np.isfinite(self.vertices).all():
            raise ValueError("Surface vertices must be finite 3D coordinates")
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError("Surface must contain triangular faces")
        if faces.min() < 0 or faces.max() >= len(self.vertices):
            raise ValueError("Invalid surface vertex index")
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        lengths = np.linalg.norm(self.vertices[edges[:, 0]] - self.vertices[edges[:, 1]], axis=1)
        self.graph = csr_matrix((np.tile(lengths, 2),
                                 (edges.ravel(order='F'), edges[:, ::-1].ravel(order='F'))),
                                shape=(len(self.vertices), len(self.vertices)))
        self.retained = np.unique(faces)
        self.tree = cKDTree(self.vertices[self.retained])
        self._kernel_key = None
        self._kernel = None

    def evaluate(self, points, values, fwhm_mm=15.0):
        """Return weighted means and coverage opacity (not statistical confidence).

        Finite contacts have equal weight before distance weighting. The Gaussian
        is truncated at 3 sigma; opacity fades from 1 to 0 at that boundary using
        the strongest contributing kernel, independent of sampling density.
        Uncovered vertices have NaN scalar and zero opacity.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        values = np.asarray(values, dtype=float)
        if values.shape != (len(points),) or not np.isfinite(points).all():
            raise ValueError("Provide finite points and one scalar per point")
        if not np.isfinite(fwhm_mm) or fwhm_mm <= 0:
            raise ValueError("Smoothing FWHM must be positive and finite")
        means = np.full(len(self.vertices), np.nan)
        opacity = np.zeros(len(self.vertices))
        if not len(points) or not np.isfinite(values).any():
            return means, opacity
        _, nearest = self.tree.query(points)
        seeds, inverse = np.unique(self.retained[nearest], return_inverse=True)
        key = (float(fwhm_mm), seeds.tobytes())
        if key != self._kernel_key:
            sigma = float(fwhm_mm) / np.sqrt(8 * np.log(2))
            rows, cols, weights = [], [], []
            for column, seed in enumerate(seeds):
                # One bounded search at a time, never a dense contacts × vertices matrix.
                distances = dijkstra(self.graph, directed=True, indices=int(seed), limit=3 * sigma)
                supported = np.flatnonzero(np.isfinite(distances))
                rows.append(supported)
                cols.append(np.full(len(supported), column))
                weights.append(np.exp(-0.5 * (distances[supported] / sigma) ** 2))
            self._kernel = coo_matrix((np.concatenate(weights),
                                      (np.concatenate(rows), np.concatenate(cols))),
                                     shape=(len(self.vertices), len(seeds))).tocsr()
            self._kernel_key = key
        finite = np.isfinite(values)
        counts = np.bincount(inverse[finite], minlength=len(seeds))
        sums = np.bincount(inverse[finite], weights=values[finite], minlength=len(seeds))
        total_weight = self._kernel @ counts
        numerator = self._kernel @ sums
        supported = total_weight > 0
        means[supported] = numerator[supported] / total_weight[supported]
        strongest = self._kernel[:, counts > 0].max(axis=1).toarray().ravel()
        floor = np.exp(-4.5)
        opacity = np.clip((strongest - floor) / (1 - floor), 0, 1)
        return means, opacity
