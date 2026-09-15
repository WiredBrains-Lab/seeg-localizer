"""Descriptive spatial pooling of MNI contacts (not ALE inference)."""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def pool_mni_contacts(points, values, patients, max_diameter=10.0):
    """Complete-linkage groups with a bounded pairwise distance in MNI mm.

    Geometry is independent of scalar availability. Finite values receive equal
    contact weight; missing values never become zero. Membership indices refer
    to the input order. Sorting coordinates makes distance ties reproducible.
    This is a hierarchical heuristic, not a globally optimal partition.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    values = np.asarray(values, dtype=float)
    if len(points) != len(values) or len(points) != len(patients):
        raise ValueError("Points, values and patients must have equal lengths")
    if not np.isfinite(max_diameter) or max_diameter <= 0:
        raise ValueError("Maximum diameter must be positive and finite")
    if not np.all(np.isfinite(points)):
        raise ValueError("MNI coordinates must be finite")
    if not len(points):
        return []
    order = np.lexsort(points.T[::-1])
    labels = (fcluster(linkage(points[order], method="complete"),
                       t=max_diameter, criterion="distance")
              if len(points) > 1 else np.ones(1, dtype=int))
    groups = []
    for label in np.unique(labels):
        members = np.sort(order[labels == label])
        finite = values[members][np.isfinite(values[members])]
        groups.append({
            "members": members.tolist(),
            "center": points[members].mean(axis=0).tolist(),
            "mean": float(finite.mean()) if len(finite) else None,
            "value_count": len(finite),
            "contact_count": len(members),
            "patient_count": len({patients[index] for index in members}),
        })
    return sorted(groups, key=lambda group: tuple(group["center"]))
