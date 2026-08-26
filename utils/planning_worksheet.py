"""Helpers for reading SEEG planning worksheet trajectory names."""

from __future__ import annotations

from pathlib import Path
import re
import zipfile
from typing import List, Optional, Sequence


EXACT_PLANNING_WORKSHEET_NAME = "seeg planning worksheet.xlsx"


def _clean_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    text = text.strip()
    return text or None


def _normalize_header(value) -> str:
    text = _clean_text(value) or ""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _parse_int(value) -> Optional[int]:
    text = _clean_text(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


_TRAJECTORY_HEADER_RANKS = {
    "trajectoryname": 0,
    "implantname": 0,
    "targetname": 0,
    "target": 1,
    "trajectory": 2,
    "implant": 2,
}


def _trajectory_header_rank(normalized: str) -> Optional[int]:
    return _TRAJECTORY_HEADER_RANKS.get(normalized)


def _is_trajectory_header(normalized: str) -> bool:
    return _trajectory_header_rank(normalized) is not None


def _is_label_header(normalized: str) -> bool:
    return normalized in {
        "label",
        "name",
        "nameleft",
        "electrodelabel",
        "electrodename",
        "electrodereference",
        "reference",
    }


def _is_contacts_header(normalized: str) -> bool:
    return "contacts" in normalized and "length" not in normalized


def _workbook_rank(path: Path):
    name = path.name.lower()
    if name == EXACT_PLANNING_WORKSHEET_NAME:
        return (0, len(path.parts), str(path).lower())
    if "seeg" in name and "planning" in name and "worksheet" in name:
        return (1, len(path.parts), str(path).lower())
    if "planning" in name and "worksheet" in name:
        return (2, len(path.parts), str(path).lower())
    return (3, len(path.parts), str(path).lower())


def _looks_like_planning_workbook(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith("~$"):
        return False
    if path.suffix.lower() not in {".xlsx", ".xlsm"}:
        return False
    if name == EXACT_PLANNING_WORKSHEET_NAME:
        return True
    return "planning" in name and (
        "seeg" in name or "implant" in name or "worksheet" in name
    )


def _candidate_roots(paths: Sequence[Optional[str]], max_ancestor_depth: int):
    seen = set()
    for raw_path in paths:
        if not raw_path:
            continue
        try:
            path = Path(raw_path).expanduser()
        except (TypeError, ValueError):
            continue
        if not path.is_dir():
            path = path.parent

        for depth, root in enumerate([path, *path.parents]):
            if depth > max_ancestor_depth:
                break
            try:
                resolved = root.resolve()
            except OSError:
                resolved = root
            key = str(resolved)
            if key in seen or not root.is_dir():
                continue
            seen.add(key)
            yield root


def find_planning_worksheet(
    paths: Sequence[Optional[str]],
    *,
    max_ancestor_depth: int = 6,
) -> Optional[str]:
    """Find the nearest SEEG planning worksheet near MRI/CT patient paths."""
    candidates = []
    seen = set()
    for root in _candidate_roots(paths, max_ancestor_depth):
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file() or not _looks_like_planning_workbook(entry):
                continue
            try:
                key = str(entry.resolve())
            except OSError:
                key = str(entry)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(entry)

    if not candidates:
        return None
    candidates.sort(key=_workbook_rank)
    return str(candidates[0])


def count_embedded_images(path: str) -> int:
    """Return the number of embedded image/media parts in a workbook."""
    try:
        with zipfile.ZipFile(path) as archive:
            return len([
                name for name in archive.namelist()
                if name.startswith("xl/media/")
            ])
    except Exception:
        return 0


def read_planning_worksheet(path: str) -> List[dict]:
    """Read trajectory rows from a SEEG planning worksheet workbook."""
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError("openpyxl is required to read SEEG planning worksheets") from exc

    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    trajectories = []

    try:
        for sheet in workbook.worksheets:
            header_row = None
            trajectory_col = None
            label_col = None
            contacts_col = None

            for row_number, row in enumerate(
                sheet.iter_rows(max_row=min(sheet.max_row, 30)),
                start=1,
            ):
                normalized = [_normalize_header(cell.value) for cell in row]
                trajectory_cols = [
                    (rank, idx)
                    for idx, value in enumerate(normalized)
                    for rank in [_trajectory_header_rank(value)]
                    if rank is not None
                ]
                if not trajectory_cols:
                    continue

                header_row = row_number
                trajectory_col = min(trajectory_cols)[1]
                for idx, value in enumerate(normalized):
                    if label_col is None and _is_label_header(value):
                        label_col = idx
                    if contacts_col is None and _is_contacts_header(value):
                        contacts_col = idx
                break

            if header_row is None or trajectory_col is None:
                continue

            started_rows = False
            for row_number, row in enumerate(
                sheet.iter_rows(min_row=header_row + 1, values_only=True),
                start=header_row + 1,
            ):
                if trajectory_col >= len(row):
                    continue
                trajectory_name = _clean_text(row[trajectory_col])
                if not trajectory_name:
                    if started_rows:
                        break
                    continue
                if _is_trajectory_header(_normalize_header(trajectory_name)):
                    continue

                label = None
                if label_col is not None and label_col < len(row):
                    label = _clean_text(row[label_col])

                contacts = None
                if contacts_col is not None and contacts_col < len(row):
                    contacts = _parse_int(row[contacts_col])
                if not label and contacts is None:
                    continue

                started_rows = True
                trajectories.append({
                    "trajectory_name": trajectory_name,
                    "label": label,
                    "contacts": contacts,
                    "order": len(trajectories) + 1,
                    "sheet": sheet.title,
                    "row": row_number,
                })
    finally:
        workbook.close()

    return trajectories
