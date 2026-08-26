"""Load saved sEEG contact locations for cohort-level MNI visualization."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from viewer3d.electrode import Electrode


ELECTRODE_INFO_PATTERN = "*_elec_info.csv"


def _text(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _float(value):
    value = _text(value)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _int(value):
    value = _float(value)
    return None if value is None else int(value)


def _bool(value, default=True):
    if isinstance(value, bool):
        return value
    value = (_text(value) or "").lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    return default


def _is_synthetic_contact(row):
    """Match the coregistration loader's treatment of derived micro contacts."""
    method = (_text(row.get("synthetic_3d_method")) or "").lower()
    if method in {"macro_first_contact", "unresolved_macro_source"}:
        return True
    assignment = (_text(row.get("planning_assignment_method")) or "").lower()
    return assignment == "synthetic_micro_target"


def discover_mni_electrode_files(root_path) -> list[Path]:
    """Return saved electrode CSVs beneath a cohort root (or one CSV path)."""
    root = Path(root_path).expanduser()
    if root.is_file():
        return [root] if root.match(ELECTRODE_INFO_PATTERN) else []
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob(ELECTRODE_INFO_PATTERN) if path.is_file()),
        key=lambda path: str(path).lower(),
    )


def _patient_id_for_file(root: Path, csv_path: Path) -> str:
    """Infer a stable patient label from the first folder below the cohort root."""
    if root.is_file():
        root = root.parent
    try:
        relative_parent = csv_path.parent.relative_to(root)
    except ValueError:
        relative_parent = csv_path.parent
    if relative_parent.parts:
        return relative_parent.parts[0]
    suffix = "_elec_info"
    stem = csv_path.stem
    return stem[:-len(suffix)] if stem.lower().endswith(suffix) else stem


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _electrode_from_row(row, patient_id: str, csv_path: Path):
    mni = tuple(_float(row.get(key)) for key in ("mni_x", "mni_y", "mni_z"))
    if any(value is None for value in mni):
        return None

    source_shaft = _text(row.get("shaft")) or "Unassigned"
    source_name = _text(row.get("name"))
    mni_space = _text(row.get("mni_space")) or "mni_tal"
    return Electrode(
        x=mni[0],
        y=mni[1],
        z=mni[2],
        shaft=source_shaft,
        name=source_name,
        visible=_bool(row.get("visible"), default=True),
        detected=_bool(row.get("detected"), default=False),
        space="mni_tal",
        tissue=_text(row.get("tissue")) or "unknown",
        mni_x=mni[0],
        mni_y=mni[1],
        mni_z=mni[2],
        mni_space=mni_space,
        patient_id=patient_id,
        cohort_patient=patient_id,
        cohort_source=str(csv_path),
        source_shaft=source_shaft,
        source_name=source_name,
        shaft_contact_number=_int(row.get("shaft_contact_number")),
        parcellation=_text(row.get("parcellation")),
        trajectory_name=_text(row.get("trajectory_name")),
        trajectory_label=_text(row.get("trajectory_label")),
        trajectory_hemisphere=_text(row.get("trajectory_hemisphere")),
        map_shaft=_text(row.get("map_shaft")),
        map_contact=_text(row.get("map_contact")),
        saved_metadata=dict(row),
    )


def _deduplication_key(electrode: Electrode):
    """Identify the same saved contact repeated across exports for one patient."""
    return (
        electrode.get("cohort_patient"),
        electrode.get("source_shaft"),
        electrode.get("source_name"),
        round(float(electrode["mni_x"]), 4),
        round(float(electrode["mni_y"]), 4),
        round(float(electrode["mni_z"]), 4),
    )


def load_mni_cohort(root_path):
    """Load all valid saved MNI contacts without reading patient imaging data.

    Returns
    -------
    electrodes : list of Electrode
        Contacts expressed directly in fsaverage/MNI Talairach millimetres.
    summary : dict
        Counts, source paths, inferred patient IDs, and per-file errors.
    """
    root = Path(root_path).expanduser()
    files = discover_mni_electrode_files(root)
    electrodes = []
    seen = set()
    errors = []
    skipped_no_mni = 0
    skipped_synthetic = 0
    skipped_duplicates = 0
    files_loaded = 0

    for csv_path in files:
        try:
            rows = _read_rows(csv_path)
        except Exception as exc:
            errors.append((str(csv_path), str(exc)))
            continue
        files_loaded += 1
        patient_id = _patient_id_for_file(root, csv_path)
        for row in rows:
            if _is_synthetic_contact(row):
                skipped_synthetic += 1
                continue
            electrode = _electrode_from_row(row, patient_id, csv_path)
            if electrode is None:
                skipped_no_mni += 1
                continue
            key = _deduplication_key(electrode)
            if key in seen:
                skipped_duplicates += 1
                continue
            seen.add(key)
            electrodes.append(electrode)

    patient_ids = sorted(
        {electrode["cohort_patient"] for electrode in electrodes},
        key=lambda value: str(value).lower(),
    )
    return electrodes, {
        "root": str(root),
        "source_files": [str(path) for path in files],
        "files_found": len(files),
        "files_loaded": files_loaded,
        "contacts_loaded": len(electrodes),
        "patient_ids": patient_ids,
        "patients_loaded": len(patient_ids),
        "skipped_no_mni": skipped_no_mni,
        "skipped_synthetic": skipped_synthetic,
        "skipped_duplicates": skipped_duplicates,
        "errors": errors,
    }
