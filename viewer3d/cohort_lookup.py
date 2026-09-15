"""Read and safely join per-contact scalar tables to saved MNI cohorts."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re


LESION_COLUMNS = tuple(
    f"test_l{level}_baseline_minus_lesioned_mpr" for level in (1, 2, 3)
)


def _key(value):
    return str(value or "").strip().casefold()


def subj_subject_id(value):
    """Return the SUBJ identifier for names such as SUBJ_009, or None."""
    match = re.fullmatch(r"subj[-_]?(\d+)", _key(value))
    return f"subj{int(match[1]):02d}" if match else None


def subject_alias(value):
    """Normalize the explicit SUBJ naming convention, preserving other IDs."""
    return subj_subject_id(value) or _key(value)


def contact_alias(value):
    """Use the final underscore-separated contact name and normalize zero padding."""
    name = _key(value).rsplit("_", 1)[-1].strip()
    return re.sub(r"(\d+)$", lambda match: str(int(match[0])), name)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


@dataclass
class CohortLookup:
    path: Path
    rows: list[dict]
    columns: list[str]
    values: dict[str, list[float]]

    def color_limits(self, columns, row_indices):
        """Symmetric limits around zero, excluding missing and unmatched values."""
        indices = set(index for index in row_indices if index is not None)
        magnitude = max(
            (abs(self.values[column][index]) for column in columns for index in indices
             if math.isfinite(self.values[column][index])),
            default=0.0,
        )
        # An all-zero or all-missing table still needs a non-degenerate mapper.
        magnitude = magnitude or 1.0
        return (-magnitude, magnitude)


def load_cohort_lookup(path):
    """Read a CSV with subject/electrode_name keys and discover numeric columns.

    Missing/non-finite cells remain NaN, never zero. Duplicate exact keys are
    rejected; ambiguous aliases are resolved separately against the cohort.
    """
    path = Path(path).expanduser()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = [field.strip() for field in (reader.fieldnames or [])]
        if len(fields) != len(set(fields)):
            raise ValueError("The lookup CSV has duplicate column headers.")
        if not {"subject", "electrode_name"}.issubset(fields):
            raise ValueError("The lookup CSV must contain subject and electrode_name columns.")
        reader.fieldnames = fields
        rows = []
        seen = set()
        for line, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"CSV row {line} has more cells than column headers.")
            row = {field: str(raw.get(field) or "").strip() for field in fields}
            if not any(row.values()):
                continue
            key = (subject_alias(row["subject"]), _key(row["electrode_name"]))
            if not all(key):
                raise ValueError(f"CSV row {line} is missing subject or electrode_name.")
            if key in seen:
                raise ValueError(
                    f"Duplicate subject/electrode_name at row {line}: "
                    f"{row['subject']} / {row['electrode_name']}."
                )
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("The lookup CSV contains no electrode rows.")

    values = {}
    for column in fields:
        if column in {"subject", "electrode_name", "electrode_index"}:
            continue
        numbers = [_number(row[column]) for row in rows]
        if any(math.isfinite(value) for value in numbers):
            values[column] = numbers
    if not values:
        raise ValueError("The lookup CSV contains no finite numeric value columns.")
    # Requested scientific measures appear before numeric metadata such as counts.
    columns = [column for column in LESION_COLUMNS if column in values]
    columns += [column for column in values if column not in columns]
    return CohortLookup(path.resolve(), rows, columns, values)


@dataclass
class CohortLookupMatches:
    row_indices: list[int | None]
    ambiguous_contacts: list[int]
    unmatched_rows: list[int]

    @property
    def matched_contacts(self):
        return sum(index is not None for index in self.row_indices)


def match_cohort_lookup(electrodes, lookup):
    """Join within subject, preferring exact names over unambiguous aliases.

    Saved name/map_contact/electrode_name are alternative identifiers for the
    same contact. Conflicting identifiers and many-to-one matches are withheld
    instead of silently assigning a value to the wrong location.
    """
    exact = {}
    aliases = defaultdict(set)
    for index, row in enumerate(lookup.rows):
        subject = subject_alias(row["subject"])
        exact[subject, _key(row["electrode_name"])] = index
        alias = contact_alias(row["electrode_name"])
        if alias:
            aliases[subject, alias].add(index)

    matched = []
    ambiguous = set()
    for index, electrode in enumerate(electrodes):
        subject = subject_alias(electrode.get("cohort_patient") or electrode.get("patient_id"))
        metadata = electrode.get("saved_metadata") or {}
        names = {
            _key(source.get(field))
            for source in (electrode, metadata)
            for field in ("electrode_name", "source_name", "name", "map_contact")
            if _key(source.get(field))
        }
        candidates = {exact[subject, name] for name in names if (subject, name) in exact}
        if not candidates:
            candidates = set().union(
                *(aliases.get((subject, contact_alias(name)), set()) for name in names)
            )
        if len(candidates) == 1:
            matched.append(next(iter(candidates)))
        else:
            matched.append(None)
            if candidates:
                ambiguous.add(index)

    inverse = defaultdict(list)
    for index, row in enumerate(matched):
        if row is not None:
            inverse[row].append(index)
    for indices in inverse.values():
        if len(indices) > 1:
            ambiguous.update(indices)
            for index in indices:
                matched[index] = None
    used = set(matched)
    return CohortLookupMatches(
        matched, sorted(ambiguous),
        [index for index in range(len(lookup.rows)) if index not in used],
    )


def cohort_contact_is_visible(electrode, selected_patients=None):
    """Use the same visibility/coordinate rules for counts, meshes, and export."""
    if selected_patients is not None and electrode.get("cohort_patient") not in selected_patients:
        return False
    return bool(electrode.get("visible", True)) and all(
        math.isfinite(_number(electrode.get(key))) for key in ("mni_x", "mni_y", "mni_z")
    )
