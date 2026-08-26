"""Helpers for reading Ripple/Trellis electrode map files."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence


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


def _looks_like_map_file(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".map" and not name.startswith(".")


def _map_rank(path: Path):
    name = path.name.lower()
    exactish = 0 if re.match(r"^(mcw|fh|pt|subj).+\.map$", name) else 1
    return (exactish, -len(path.parts), str(path).lower())


def find_electrode_map(
    paths: Sequence[Optional[str]],
    *,
    max_ancestor_depth: int = 6,
) -> Optional[str]:
    """Find the nearest electrode .map file near MRI/CT patient paths."""
    candidates = []
    seen = set()
    for root in _candidate_roots(paths, max_ancestor_depth):
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file() or not _looks_like_map_file(entry):
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
    candidates.sort(key=_map_rank)
    return str(candidates[0])


def _parse_contact_label(label: str):
    """Return (shaft, contact_number) for labels like LN01 or RN05-COM."""
    text = str(label or "").strip()
    match = re.match(r"^([A-Za-z][A-Za-z0-9]*?)(\d{2})(?:\D.*)?$", text)
    if not match:
        match = re.match(r"^([A-Za-z]+)(\d+)", text)
    if not match:
        return None, None
    shaft = match.group(1).upper()
    try:
        contact_number = int(match.group(2))
    except ValueError:
        return None, None
    if not shaft or shaft[0] not in {"L", "R"}:
        return None, None
    return shaft, contact_number


def read_electrode_map(path: str) -> Dict[str, dict]:
    """Read shaft contact labels from a .map file.

    Only non-comment rows whose hardware address starts with 1.A or 1.B are
    considered. The second semicolon-separated field is interpreted as the
    contact label.
    """
    shafts: Dict[str, dict] = {}
    row_order = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [field.strip() for field in line.split(";")]
            if len(fields) < 2:
                continue
            address = fields[0]
            if not re.match(r"^1\.[AB](?:\.|$)", address, flags=re.IGNORECASE):
                continue
            contact_label = fields[1]
            shaft, contact_number = _parse_contact_label(contact_label)
            if shaft is None or contact_number is None:
                continue

            entry = shafts.setdefault(
                shaft,
                {
                    "shaft": shaft,
                    "contacts": [],
                    "order": row_order,
                },
            )
            if not entry["contacts"]:
                row_order += 1
            entry["contacts"].append({
                "address": address,
                "label": contact_label,
                "number": contact_number,
                "line": line_number,
            })

    for entry in shafts.values():
        contacts: List[dict] = entry["contacts"]
        contacts.sort(key=lambda item: (item["number"], item["line"]))
        entry["contact_count"] = len(contacts)
        entry["first_contact"] = contacts[0]["label"] if contacts else None
    return shafts
