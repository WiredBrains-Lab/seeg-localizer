"""Atlas and parcellation helpers for sEEG Localizer."""

from pathlib import Path

import mne


HCP_MMP_PARCS = {"HCPMMP1", "HCPMMP1_combined"}

def normalize_parcellation_key(parcellation):
    """Return a canonical atlas key used by the viewer controls."""
    text = str(parcellation or "").strip()
    lower = text.lower().replace("_", "-").replace(" ", "-")
    if lower in {"hcp-mmp", "hcpmmp", "hcp-mmp1", "hcpmmp1", "hcp"}:
        return "HCPMMP1"
    if lower in {"hcp-mmp-combined", "hcpmmp1-combined", "hcp-combined"}:
        return "HCPMMP1_combined"
    if lower in {"aparc+aseg", "volume", "vol"}:
        return "aparc+aseg"
    if lower in {"aparc", "surface-aparc", "freesurfer-aparc"}:
        return "aparc"
    return text or "aparc+aseg"


def ensure_hcp_mmp_parcellation(subjects_dir):
    """Fetch HCP-MMP annotations into a subjects_dir and return that path."""
    subjects_dir = Path(subjects_dir).expanduser().resolve()
    subjects_dir.mkdir(parents=True, exist_ok=True)
    mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
    mne.datasets.fetch_hcp_mmp_parcellation(
        subjects_dir=str(subjects_dir),
        combine=True,
        accept=True,
        verbose=False,
    )
    return str(subjects_dir)


def read_surface_labels(subject, subjects_dir, parcellation):
    """Read labels for a surface parcellation."""
    parc = normalize_parcellation_key(parcellation)
    if parc in HCP_MMP_PARCS:
        subjects_dir = ensure_hcp_mmp_parcellation(subjects_dir)
        subject = "fsaverage"
    return mne.read_labels_from_annot(
        subject=str(subject),
        parc=parc,
        subjects_dir=str(subjects_dir),
        verbose="error",
    )


def subject_aseg_path(subject, subjects_dir):
    """Return the first supported aseg-like atlas path for a subject."""
    subject_dir = Path(subjects_dir).expanduser().resolve() / str(subject)
    for name in ("aparc+aseg.mgz", "aparc.DKTatlas+aseg.deep.mgz"):
        path = subject_dir / "mri" / name
        if path.exists():
            return str(path)
    return ""


def aseg_name_from_path(aseg_path):
    """Return the MNE aseg name corresponding to an atlas file path."""
    name = Path(str(aseg_path)).name
    if name.endswith(".mgz"):
        name = name[:-4]
    return name or "aparc+aseg"
