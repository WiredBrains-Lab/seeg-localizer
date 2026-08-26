"""Shaft-related utility functions: axis fitting and projections.

This module provides fast PCA-based axis fitting and helpers to project
points onto the fitted line. A simple RANSAC line fitter is also included
as an optional, more robust alternative.

Functions:
- fit_shaft_axis_pca(points) -> (centroid, unit_direction)
- project_point_to_line(pt, axis_point, axis_dir) -> (proj_point, t, perp_dist)
- fit_shaft_axis_ransac(points, n_iters=200, dist_thresh=2.0, min_inliers=3)
- split_line_clusters_by_gap(positions, labels, max_gap, min_contacts=3)
- split_labels_by_group(labels, groups, min_contacts=3)
- cluster_shaft_contacts(positions, dist_thresh=3.0, max_gap=25.0, min_contacts=5)

These are pure-numpy implementations to avoid extra dependencies; sklearn
RANSAC can be used instead if desired.
"""

from __future__ import annotations

import numpy as np
import random
from typing import Dict, List, Optional, Tuple


def fit_shaft_axis_pca(points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit a shaft axis using PCA.

    Parameters
    - points: (N,3) array-like

    Returns (centroid, unit_direction) or (None, None) if points empty.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return None, None
    c = pts.mean(axis=0)
    X = pts - c
    # SVD: principal components in vh
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    direction = vh[0]
    norm = np.linalg.norm(direction)
    if norm == 0:
        # degenerate: return a default axis
        return c, np.array([1.0, 0.0, 0.0])
    return c, direction / norm


def project_point_to_line(pt: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Project a point onto a line defined by axis_point + t * axis_dir.

    Returns (projected_point, t_scalar, perpendicular_distance).
    """
    p = np.asarray(pt, dtype=float)
    a = np.asarray(axis_point, dtype=float)
    u = np.asarray(axis_dir, dtype=float)
    u = u / np.linalg.norm(u)
    w = p - a
    t = float(np.dot(w, u))
    proj = a + t * u
    perp = w - t * u
    dist = float(np.linalg.norm(perp))
    return proj, t, dist


def fit_shaft_axis_ransac(points: np.ndarray, n_iters: int = 300, dist_thresh: float = 2.0, min_inliers: int = 3) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    """Simple RANSAC line fitter for shafts.

    Returns (axis_point, axis_dir, inlier_indices).
    If RANSAC fails, falls back to PCA on all points and returns all indices.
    """
    pts = np.asarray(points, dtype=float)
    N = len(pts)
    if N < 2:
        return None, None, []

    best_inliers: List[int] = []
    best_model = None

    for _ in range(n_iters):
        i, j = random.sample(range(N), 2)
        a = pts[i]
        b = pts[j]
        dirv = b - a
        norm = np.linalg.norm(dirv)
        if norm < 1e-6:
            continue
        dirv = dirv / norm
        # distances
        ws = pts - a
        ts = np.dot(ws, dirv)
        perps = ws - np.outer(ts, dirv)
        dists = np.linalg.norm(perps, axis=1)
        inliers_idx = list(np.where(dists <= dist_thresh)[0])
        if len(inliers_idx) > len(best_inliers):
            best_inliers = inliers_idx
            best_model = (a.copy(), dirv.copy())
            # early exit if almost all
            if len(best_inliers) >= max(min_inliers, int(0.9 * N)):
                break

    if best_model is None:
        # fallback to PCA
        centroid, direction = fit_shaft_axis_pca(pts)
        return centroid, direction, list(range(N))

    a, dirv = best_model
    # re-fit using PCA on inliers for more stable axis
    inlier_pts = pts[best_inliers]
    centroid, direction = fit_shaft_axis_pca(inlier_pts)
    return centroid, direction, best_inliers


def split_line_clusters_by_gap(
    positions: np.ndarray,
    labels: np.ndarray,
    max_gap: float,
    min_contacts: int = 3,
) -> np.ndarray:
    """Split line-like clusters where projected contact spacing exceeds max_gap.

    Fragments with fewer than min_contacts are marked unassigned (-1). This
    prevents isolated distant contacts from being pulled into an otherwise valid
    shaft just because they are collinear with it.
    """
    split_labels = np.asarray(labels, dtype=int).copy()
    pts_all = np.asarray(positions, dtype=float)
    if len(pts_all) == 0 or max_gap <= 0:
        return split_labels

    next_label = (split_labels.max() + 1) if split_labels.size else 0
    for label in sorted(set(split_labels)):
        if label < 0:
            continue
        idxs = np.where(split_labels == label)[0]
        if len(idxs) < 2:
            continue
        pts = pts_all[idxs]
        centroid, direction = fit_shaft_axis_pca(pts)
        if centroid is None or direction is None:
            continue
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction = direction / norm
        ts = (pts - centroid) @ direction
        order = np.argsort(ts)
        sorted_idxs = idxs[order]
        sorted_t = ts[order]

        segments = []
        current_segment = [sorted_idxs[0]]
        last_t = sorted_t[0]
        for idx, t in zip(sorted_idxs[1:], sorted_t[1:]):
            if abs(t - last_t) > max_gap:
                segments.append(current_segment)
                current_segment = [idx]
            else:
                current_segment.append(idx)
            last_t = t
        segments.append(current_segment)

        if len(segments) <= 1:
            continue

        valid_segment_positions = [
            segment_idx
            for segment_idx, segment in enumerate(segments)
            if len(segment) >= int(min_contacts)
        ]
        if not valid_segment_positions:
            split_labels[idxs] = -1
            continue

        labels_by_segment = {}
        for valid_idx, segment_idx in enumerate(valid_segment_positions):
            if valid_idx == 0:
                labels_by_segment[segment_idx] = label
            else:
                labels_by_segment[segment_idx] = next_label
                next_label += 1

        for segment_idx, segment in enumerate(segments):
            assigned_label = labels_by_segment.get(segment_idx)
            segment = np.asarray(segment, dtype=int)
            if assigned_label is None:
                split_labels[segment] = -1
            else:
                split_labels[segment] = assigned_label

    return split_labels


def split_labels_by_group(
    labels: np.ndarray,
    groups: np.ndarray,
    min_contacts: int = 3,
) -> np.ndarray:
    """Split existing cluster labels when contacts have different group keys.

    Group keys are arbitrary hashable values such as "left"/"right". Contacts
    with a missing group key are left in their current cluster unless the cluster
    also contains concrete groups, in which case the missing-key contacts are
    unassigned. Split fragments smaller than min_contacts are unassigned.
    """
    split_labels = np.asarray(labels, dtype=int).copy()
    group_values = np.asarray(groups, dtype=object)
    if split_labels.size == 0 or group_values.size != split_labels.size:
        return split_labels

    next_label = (int(split_labels.max()) + 1) if split_labels.size else 0
    min_contacts = max(1, int(min_contacts))

    for label in sorted(set(split_labels.tolist())):
        if label < 0:
            continue
        idxs = np.where(split_labels == label)[0]
        if len(idxs) < 2:
            continue

        concrete_groups = []
        for group in group_values[idxs]:
            if group is None:
                continue
            if group not in concrete_groups:
                concrete_groups.append(group)

        if len(concrete_groups) <= 1:
            continue

        split_labels[idxs] = -1
        valid_groups = []
        for group in sorted(concrete_groups, key=lambda value: str(value)):
            group_idxs = idxs[group_values[idxs] == group]
            if len(group_idxs) >= min_contacts:
                valid_groups.append(group_idxs)

        for group_position, group_idxs in enumerate(valid_groups):
            assigned_label = label if group_position == 0 else next_label
            if group_position > 0:
                next_label += 1
            split_labels[group_idxs] = assigned_label

    return split_labels


def _unit_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    vec = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return None
    return vec / norm


def _line_projection_distances(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return projection scalars, perpendicular distances, and residual vectors."""
    pts = np.asarray(points, dtype=float)
    point = np.asarray(axis_point, dtype=float)
    direction = _unit_vector(axis_dir)
    if direction is None:
        zeros = np.zeros(len(pts), dtype=float)
        return zeros, np.full(len(pts), np.inf, dtype=float), np.zeros_like(pts)

    offsets = pts - point
    t_values = offsets @ direction
    residuals = offsets - np.outer(t_values, direction)
    distances = np.linalg.norm(residuals, axis=1)
    return t_values, distances, residuals


def _split_indices_by_gap(
    indices: np.ndarray,
    t_values: np.ndarray,
    max_gap: float,
    min_contacts: int,
) -> List[np.ndarray]:
    """Split indices ordered along an axis when consecutive points are too far apart."""
    idxs = np.asarray(indices, dtype=int)
    ts = np.asarray(t_values, dtype=float)
    if len(idxs) == 0:
        return []
    if len(idxs) != len(ts):
        raise ValueError("indices and t_values must have the same length")

    order = np.argsort(ts)
    sorted_idxs = idxs[order]
    sorted_t = ts[order]

    segments: List[List[int]] = [[int(sorted_idxs[0])]]
    for idx, prev_t, cur_t in zip(sorted_idxs[1:], sorted_t[:-1], sorted_t[1:]):
        if max_gap > 0 and abs(float(cur_t - prev_t)) > float(max_gap):
            segments.append([])
        segments[-1].append(int(idx))

    min_contacts = max(1, int(min_contacts))
    return [
        np.asarray(segment, dtype=int)
        for segment in segments
        if len(segment) >= min_contacts
    ]


def _perpendicular_basis(axis_dir: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return two orthonormal vectors spanning the plane perpendicular to axis_dir."""
    direction = _unit_vector(axis_dir)
    if direction is None:
        return None

    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(direction, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    first = np.cross(direction, ref)
    first = _unit_vector(first)
    if first is None:
        return None
    second = np.cross(direction, first)
    second = _unit_vector(second)
    if second is None:
        return None
    return first, second


def _radius_components(coords: np.ndarray, radius: float) -> List[np.ndarray]:
    """Connected components in a small point cloud using a fixed radius graph."""
    pts = np.asarray(coords, dtype=float)
    n_points = len(pts)
    if n_points == 0:
        return []
    if radius <= 0:
        return [np.arange(n_points, dtype=int)]

    radius_sq = float(radius) ** 2
    diff = pts[:, None, :] - pts[None, :, :]
    adjacency = np.sum(diff * diff, axis=2) <= radius_sq

    seen = np.zeros(n_points, dtype=bool)
    components: List[np.ndarray] = []
    for start in range(n_points):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = np.where(adjacency[current] & ~seen)[0]
            for neighbor in neighbors:
                seen[neighbor] = True
                stack.append(int(neighbor))
        components.append(np.asarray(component, dtype=int))
    return components


def _split_lateral_components(
    points: np.ndarray,
    indices: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    dist_thresh: float,
    min_contacts: int,
) -> List[np.ndarray]:
    """Split line inliers that occupy distinct offsets in the perpendicular plane."""
    idxs = np.asarray(indices, dtype=int)
    if len(idxs) < int(min_contacts) * 2:
        return [idxs]

    basis = _perpendicular_basis(axis_dir)
    if basis is None:
        return [idxs]

    _, _, residuals = _line_projection_distances(points[idxs], axis_point, axis_dir)
    lateral_coords = np.column_stack((residuals @ basis[0], residuals @ basis[1]))
    lateral_eps = max(1.0, float(dist_thresh) * 0.5)

    groups = []
    for component in _radius_components(lateral_coords, lateral_eps):
        if len(component) >= int(min_contacts):
            groups.append(idxs[component])

    if not groups:
        return [idxs]
    return groups


def _candidate_score(
    count: int,
    t_values: np.ndarray,
    distances: np.ndarray,
    dist_thresh: float,
    max_gap: float,
) -> float:
    """Score a proposed shaft by count, straightness, span, and spacing regularity."""
    if count <= 0:
        return -np.inf

    dist_thresh = max(float(dist_thresh), 1e-6)
    mean_dist = float(np.mean(distances)) if len(distances) else 0.0
    p90_dist = float(np.percentile(distances, 90)) if len(distances) else 0.0
    line_quality = max(0.0, 1.0 - (mean_dist / dist_thresh))

    span = 0.0
    spacing_cv = 0.0
    small_gap_count = 0
    if len(t_values) > 1:
        sorted_t = np.sort(np.asarray(t_values, dtype=float))
        span = float(sorted_t[-1] - sorted_t[0])
        raw_diffs = np.diff(sorted_t)
        positive_diffs = raw_diffs[raw_diffs > 1e-6]
        if len(positive_diffs):
            median_gap = float(np.median(positive_diffs))
            if median_gap > 1e-6:
                small_gap_limit = max(1.0, min(float(max_gap) * 0.25, median_gap * 0.35))
                small_gap_count = int(np.sum(raw_diffs < small_gap_limit))
                spacing_cv = float(np.std(raw_diffs) / median_gap)

    gap_scale = max(float(max_gap), 1.0)
    span_bonus = min(span / gap_scale, float(count))
    return (
        (float(count) * 20.0)
        + (line_quality * 25.0)
        + (span_bonus * 3.0)
        - (mean_dist * 10.0)
        - (p90_dist * 4.0)
        - (spacing_cv * 20.0)
        - (small_gap_count * 35.0)
    )


def _prune_dense_projection_slots(
    points: np.ndarray,
    indices: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    max_gap: float,
) -> np.ndarray:
    """Keep the closest point to the axis when multiple contacts share a slot."""
    idxs = np.asarray(indices, dtype=int)
    if len(idxs) < 2:
        return idxs

    t_values, distances, _ = _line_projection_distances(points[idxs], axis_point, axis_dir)
    order = np.argsort(t_values)
    sorted_idxs = idxs[order]
    sorted_t = t_values[order]
    sorted_distances = distances[order]

    positive_diffs = np.diff(sorted_t)
    positive_diffs = positive_diffs[positive_diffs > 1e-6]
    if len(positive_diffs) == 0:
        best_pos = int(np.argmin(sorted_distances))
        return np.asarray([sorted_idxs[best_pos]], dtype=int)

    median_gap = float(np.median(positive_diffs))
    if median_gap <= 1e-6:
        return idxs
    slot_gap = max(1.0, min(float(max_gap) * 0.25, median_gap * 0.35))

    kept = []
    current = [0]
    for pos in range(1, len(sorted_idxs)):
        if abs(float(sorted_t[pos] - sorted_t[pos - 1])) < slot_gap:
            current.append(pos)
        else:
            best_pos = min(current, key=lambda item: float(sorted_distances[item]))
            kept.append(int(sorted_idxs[best_pos]))
            current = [pos]
    best_pos = min(current, key=lambda item: float(sorted_distances[item]))
    kept.append(int(sorted_idxs[best_pos]))
    return np.asarray(kept, dtype=int)


def _build_candidates_from_indices(
    points: np.ndarray,
    indices: np.ndarray,
    dist_thresh: float,
    max_gap: float,
    min_contacts: int,
) -> List[Dict[str, object]]:
    """Refine a rough inlier set into one or more valid shaft candidates."""
    idxs = np.asarray(indices, dtype=int)
    if len(idxs) < int(min_contacts):
        return []

    candidates: List[Dict[str, object]] = []
    centroid, direction = fit_shaft_axis_pca(points[idxs])
    if centroid is None or direction is None:
        return []

    t_values, distances, _ = _line_projection_distances(points[idxs], centroid, direction)
    keep = distances <= float(dist_thresh)
    if int(np.sum(keep)) < int(min_contacts):
        return []
    idxs = idxs[keep]

    centroid, direction = fit_shaft_axis_pca(points[idxs])
    if centroid is None or direction is None:
        return []
    t_values, distances, _ = _line_projection_distances(points[idxs], centroid, direction)
    keep = distances <= float(dist_thresh)
    if int(np.sum(keep)) < int(min_contacts):
        return []
    idxs = idxs[keep]
    t_values = t_values[keep]

    for segment in _split_indices_by_gap(idxs, t_values, max_gap, min_contacts):
        if len(segment) < int(min_contacts):
            continue
        seg_centroid, seg_direction = fit_shaft_axis_pca(points[segment])
        if seg_centroid is None or seg_direction is None:
            continue
        seg_t, seg_distances, _ = _line_projection_distances(
            points[segment],
            seg_centroid,
            seg_direction,
        )
        keep_segment = seg_distances <= float(dist_thresh)
        if int(np.sum(keep_segment)) < int(min_contacts):
            continue
        segment = segment[keep_segment]

        seg_centroid, seg_direction = fit_shaft_axis_pca(points[segment])
        if seg_centroid is None or seg_direction is None:
            continue
        segment = _prune_dense_projection_slots(
            points,
            segment,
            seg_centroid,
            seg_direction,
            max_gap,
        )
        if len(segment) < int(min_contacts):
            continue
        seg_centroid, seg_direction = fit_shaft_axis_pca(points[segment])
        if seg_centroid is None or seg_direction is None:
            continue
        seg_t, seg_distances, _ = _line_projection_distances(
            points[segment],
            seg_centroid,
            seg_direction,
        )
        for final_segment in _split_indices_by_gap(segment, seg_t, max_gap, min_contacts):
            final_centroid, final_direction = fit_shaft_axis_pca(points[final_segment])
            if final_centroid is None or final_direction is None:
                continue
            final_t, final_distances, _ = _line_projection_distances(
                points[final_segment],
                final_centroid,
                final_direction,
            )
            if np.any(final_distances > (float(dist_thresh) * 1.15)):
                continue
            score = _candidate_score(
                len(final_segment),
                final_t,
                final_distances,
                dist_thresh,
                max_gap,
            )
            order = np.argsort(final_t)
            candidates.append({
                "indices": np.asarray(final_segment[order], dtype=int),
                "score": float(score),
                "count": int(len(final_segment)),
                "mean_distance": float(np.mean(final_distances)),
                "max_distance": float(np.max(final_distances)),
            })

    return candidates


def _generate_shaft_candidates(
    points: np.ndarray,
    available_indices: np.ndarray,
    dist_thresh: float,
    max_gap: float,
    min_contacts: int,
) -> List[Dict[str, object]]:
    """Generate deterministic shaft candidates from near-neighbor line seeds."""
    available = np.asarray(available_indices, dtype=int)
    if len(available) < int(min_contacts):
        return []

    pts = np.asarray(points, dtype=float)
    rem_pts = pts[available]
    n_points = len(available)
    seed_min = max(1.0, float(dist_thresh) * 0.35)
    seed_max = np.inf if max_gap <= 0 else float(max_gap) * 1.5
    candidates_by_key: Dict[Tuple[int, ...], Dict[str, object]] = {}

    for local_i in range(n_points - 1):
        start = rem_pts[local_i]
        for local_j in range(local_i + 1, n_points):
            end = rem_pts[local_j]
            axis = end - start
            pair_distance = float(np.linalg.norm(axis))
            if pair_distance < seed_min or pair_distance > seed_max:
                continue

            t_values, distances, _ = _line_projection_distances(rem_pts, start, axis)
            inlier_mask = distances <= float(dist_thresh)
            if int(np.sum(inlier_mask)) < int(min_contacts):
                continue

            inlier_indices = available[inlier_mask]
            inlier_t = t_values[inlier_mask]
            for segment in _split_indices_by_gap(
                inlier_indices,
                inlier_t,
                max_gap,
                min_contacts,
            ):
                lateral_groups = _split_lateral_components(
                    pts,
                    segment,
                    start,
                    axis,
                    dist_thresh,
                    min_contacts,
                )
                for group in lateral_groups:
                    for candidate in _build_candidates_from_indices(
                        pts,
                        group,
                        dist_thresh,
                        max_gap,
                        min_contacts,
                    ):
                        key = tuple(sorted(int(idx) for idx in candidate["indices"]))
                        current = candidates_by_key.get(key)
                        if current is None or float(candidate["score"]) > float(current["score"]):
                            candidates_by_key[key] = candidate

    return list(candidates_by_key.values())


def cluster_shaft_contacts(
    positions: np.ndarray,
    dist_thresh: float = 3.0,
    max_gap: float = 25.0,
    min_contacts: int = 5,
) -> np.ndarray:
    """Cluster SEEG contacts into line-like shafts.

    The algorithm is deterministic: it builds candidate shaft lines from nearby
    contact pairs, splits each candidate on large axial gaps and lateral
    parallel offsets, then greedily accepts the best non-overlapping shaft.
    Contacts that do not form a minimum-sized straight segment remain -1.
    """
    pts = np.asarray(positions, dtype=float)
    if pts.size == 0:
        return np.array([], dtype=int)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("positions must be an (N, 3) array")

    n_points = len(pts)
    min_contacts = max(2, int(min_contacts))
    labels = np.full(n_points, -1, dtype=int)
    available = np.arange(n_points, dtype=int)
    next_label = 0

    while len(available) >= min_contacts:
        candidates = _generate_shaft_candidates(
            pts,
            available,
            float(dist_thresh),
            float(max_gap),
            min_contacts,
        )
        if not candidates:
            break

        candidates.sort(
            key=lambda item: (
                float(item["score"]),
                int(item["count"]),
                -float(item["mean_distance"]),
                -float(item["max_distance"]),
            ),
            reverse=True,
        )
        best = candidates[0]
        best_indices = np.asarray(best["indices"], dtype=int)
        if len(best_indices) < min_contacts:
            break

        labels[best_indices] = next_label
        next_label += 1
        available = available[labels[available] == -1]

    return labels
