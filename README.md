# sEEG Localizer

A standalone desktop application for localizing and visualizing sEEG contacts,
registering postoperative CT to preoperative MRI, inspecting
FreeSurfer surfaces and parcellations, and viewing cohort electrode locations
in MNI space.

## Included features

- Patient MRI and postoperative CT loading
- CT-to-MRI registration with FreeSurfer or optional Python backends
- Electrode detection, labeling, planning-sheet integration, and export
- Individual FreeSurfer subject and `fsaverage` visualization
- MNI cohort loading from saved `*_elec_info.csv` files

## Requirements

- Python 3.10 or newer
- A working desktop display for Qt/PyVista
- MNE's `fsaverage` data (downloaded by MNE when needed)
- Preoperative MRI and postoperative CT for patient-specific registration
- FreeSurfer for the preferred registration workflow, or at least one optional
  Python registration backend

FreeSurfer is installed separately and should expose `FREESURFER_HOME`. The
viewer also checks common macOS FreeSurfer installation locations.

## Install

Create and activate a virtual environment, then install the viewer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

To install all optional Python registration backends:

```bash
python -m pip install -e '.[registration]'
```

`antspyx`, `dipy`, and `SimpleITK` are optional because platform and Python
wheel availability can vary. FreeSurfer's `mri_robust_register` remains the
preferred backend when FreeSurfer is available.

## Run

After installation:

```bash
seeg-localizer
```

You can optionally open with an existing patient registration folder:

```bash
seeg-localizer /path/to/patient_registration_folder
```

The equivalent source-tree command is:

```bash
python -m viewer3d
```

## Project layout

```text
viewer3d/      Main viewer, registration dialog, MNI cohort, and electrode model
core/          Atlas and parcellation helpers
usercontrols/  Custom Qt widgets
utils/         Electrode map, planning worksheet, schema, and shaft helpers
```

## Data privacy

Patient MRI, CT, electrode exports, and related clinical data may contain
protected health information. Common neuroimaging and generated patient-data
formats are excluded by `.gitignore`, but review `git status` carefully before
every public push.

## License

No open-source license has been selected yet. Add a `LICENSE` file before
granting reuse or redistribution rights.
