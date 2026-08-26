"""MCW SEEG naming-schema spatial prior."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple


# Approximate label locations from the MCW lateral naming schematic.
# Coordinates are normalized as (anterior-posterior, superior-inferior):
#   AP: 0 = posterior, 1 = anterior
#   SI: 0 = inferior,  1 = superior
MCW_SCHEMA_LABEL_POSITIONS: Dict[str, Tuple[float, float]] = {
    "E": (0.18, 0.12),
    "A": (0.28, 0.25),
    "H": (0.36, 0.38),
    "B": (0.43, 0.34),
    "F": (0.48, 0.15),
    "C": (0.52, 0.34),
    "I": (0.60, 0.50),
    "D": (0.70, 0.43),
    "G": (0.76, 0.22),
    "O": (0.83, 0.18),
    "N": (0.73, 0.36),
    "P": (0.79, 0.48),
    "J": (0.68, 0.48),
    "K": (0.53, 0.58),
    "W": (0.34, 0.63),
    "Y": (0.22, 0.46),
}


def normalize_schema_label(label: object) -> str:
    """Return a comparable schema label such as F from F1 or A from A*."""
    text = str(label or "").strip().upper()
    text = text.replace("*", "")
    text = re.sub(r"[^A-Z0-9]", "", text)
    text = re.sub(r"\d+$", "", text)
    return text


def _trajectory_name(trajectory: dict) -> str:
    return str(trajectory.get("trajectory_name") or "").strip()


def trajectory_schema_position(trajectory: dict) -> Optional[Tuple[float, float]]:
    """Return an approximate schema position for a planning trajectory."""
    label = normalize_schema_label(trajectory.get("label"))
    if not label:
        return None
    position = MCW_SCHEMA_LABEL_POSITIONS.get(label)
    if position is None:
        return None

    ap, si = position
    name = _trajectory_name(trajectory).lower()
    raw_label = str(trajectory.get("label") or "").strip().upper()

    if re.search(r"\b(ant|anterior)\b|(?:^|[-_])a[a-z]", name):
        ap += 0.035
    if re.search(r"\b(post|posterior)\b|(?:^|[-_])p[a-z]", name):
        ap -= 0.035
    if raw_label.endswith("1"):
        ap += 0.025
    elif raw_label.endswith("2"):
        ap -= 0.025

    return (min(max(ap, 0.0), 1.0), min(max(si, 0.0), 1.0))


def _normalize_detected_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    ap_vals = [point[0] for point in points]
    si_vals = [point[1] for point in points]
    ap_min, ap_max = min(ap_vals), max(ap_vals)
    si_min, si_max = min(si_vals), max(si_vals)
    ap_span = ap_max - ap_min
    si_span = si_max - si_min

    normalized = []
    for ap, si in points:
        ap_norm = 0.5 if abs(ap_span) < 1e-6 else (ap - ap_min) / ap_span
        si_norm = 0.5 if abs(si_span) < 1e-6 else (si - si_min) / si_span
        normalized.append((ap_norm, si_norm))
    return normalized


def _linear_sum_assignment(cost_matrix):
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost_matrix)
    except Exception:
        rows = set(range(len(cost_matrix)))
        cols = set(range(len(cost_matrix[0]) if len(cost_matrix) else 0))
        row_ind = []
        col_ind = []
        while rows and cols:
            best = None
            for row in rows:
                for col in cols:
                    cost = cost_matrix[row][col]
                    if best is None or cost < best[0]:
                        best = (cost, row, col)
            if best is None:
                break
            _, row, col = best
            row_ind.append(row)
            col_ind.append(col)
            rows.remove(row)
            cols.remove(col)
        return row_ind, col_ind


def assign_schema_prior(
    detected_items: Sequence[dict],
    trajectories: Sequence[dict],
    *,
    order_weight: float = 0.12,
) -> List[dict]:
    """Assign detected shafts to trajectories using the MCW schema prior.

    Each detected item must have `name`, `point`, and `order` keys, where point
    is a 2D world-space `(AP, SI)` tuple.
    """
    detected = [
        item for item in detected_items
        if item.get("point") is not None
    ]
    planned = [
        (trajectory, trajectory_schema_position(trajectory))
        for trajectory in trajectories
    ]
    planned = [
        (trajectory, point) for trajectory, point in planned
        if point is not None
    ]
    if len(detected) < 2 or len(planned) < 2:
        return []

    detected_points = _normalize_detected_points([item["point"] for item in detected])
    planned_points = _normalize_detected_points([point for _trajectory, point in planned])
    n_detected = len(detected)
    n_planned = len(planned)
    order_denominator = max(n_detected, n_planned, 1)

    cost_matrix = []
    for det_idx, detected_point in enumerate(detected_points):
        row = []
        for plan_idx, (_trajectory, _schema_point) in enumerate(planned):
            spatial = math.dist(detected_point, planned_points[plan_idx])
            order_cost = abs(det_idx - plan_idx) / order_denominator
            row.append(spatial + order_weight * order_cost)
        cost_matrix.append(row)

    row_ind, col_ind = _linear_sum_assignment(cost_matrix)
    assignments = []
    for row, col in zip(row_ind, col_ind):
        detected_item = detected[int(row)]
        trajectory, schema_point = planned[int(col)]
        assignments.append({
            "old_name": detected_item["name"],
            "trajectory": trajectory,
            "schema_point": schema_point,
            "detected_point": detected_item["point"],
            "cost": float(cost_matrix[int(row)][int(col)]),
            "method": "schema_prior",
        })

    assignments.sort(key=lambda item: item["old_name"])
    return assignments
