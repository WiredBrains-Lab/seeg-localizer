# sEEG Localizer 3D

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
- Per-contact scalar lookup coloring, table-based filtering, and separate cohort figure export

<img width="1713" height="918" alt="seeg-localizer-paths-masked" src="https://github.com/user-attachments/assets/84a7e98c-c3c0-4d39-a825-12554b90f894" />

## Requirements

- Python 3.10 or newer
- A working desktop display for Qt/PyVista
- MNE's `fsaverage` data (downloaded by MNE when needed)
- Preoperative MRI and postoperative CT for patient-specific registration
- FreeSurfer for the preferred registration workflow, or at least one optional
  Python registration backend
- A large display is recommended; some controls may be crowded on a laptop.
- macOS is the currently tested platform. Windows and Linux support is
  expected but has not yet been validated comprehensively.

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

Source launches use `packaging/macos/seeg-localizer-icon.png` for the Qt
application and viewer windows. The icon is assigned before MNE initializes
its Qt backend, preventing MNE's default icon from replacing it.

## Download the macOS application

Tagged releases can include a ready-to-run Apple-silicon macOS disk image named
`sEEG-Localizer-<version>-macOS-arm64.dmg` on the GitHub Releases page. Download
and open the disk image, then drag `sEEG Localizer` into the Applications folder.
Each DMG is accompanied by a `.sha256` checksum file.

The current automated build is for Apple-silicon Macs (`arm64`). The application
is not yet notarized, so macOS may require the first launch to be performed by
Control-clicking the application, selecting **Open**, and confirming the prompt.
FreeSurfer remains a separate installation and is detected at runtime.

### Build a macOS release locally

Install the application and release dependencies, then run the build script:

```bash
python -m pip install -r requirements-build.txt
SEEG_BUILD_PYTHON=python ./scripts/build_macos.sh
```

The script builds and smoke-tests `dist/sEEG Localizer.app`, creates a compressed
DMG, and verifies both artifacts. To use an installed Developer ID certificate,
set `SEEG_MACOS_CODESIGN_IDENTITY`. A local `notarytool` keychain profile can be
provided with `SEEG_MACOS_NOTARY_PROFILE` to submit and staple the DMG.

The macOS icon is generated from
`packaging/macos/seeg-localizer-icon.png`. The release build regenerates the
bundled `.icns` automatically. To regenerate it without running a full build:

```bash
./scripts/generate_macos_icon.sh
```

The `Build macOS application` GitHub Actions workflow can also be run manually.
Pushing a version tag such as `v0.1.0` builds the DMG and attaches it to a draft
GitHub Release so it can be tested before publication.

## Input data

### Minimum data for one patient

The basic CT-to-MRI localization workflow requires:

- A preoperative structural MRI, preferably a three-dimensional T1-weighted
  NIfTI file (`.nii` or `.nii.gz`). An MGZ volume (`.mgz`) can also be selected.
- A postoperative CT containing the implanted contacts, preferably converted
  to one three-dimensional NIfTI file (`.nii` or `.nii.gz`). The file picker
  exposes individual DICOM files, but the current processing workflow is built
  around complete volumes readable by NiBabel; convert a DICOM series to NIfTI
  before use.

The MRI and CT are selected separately in the Coregistration UI, so the app
does not require a rigid folder layout. Keeping all files for one patient under
one patient directory is strongly recommended because planning files and saved
results are discovered relative to the selected MRI, CT, and output paths.

### Recommended patient layout

This is a suggested organization, not a required BIDS layout:

```text
cohort-root/
  sub-001/
    anat/
      sub-001_T1w.nii.gz
    postop/
      sub-001_postop_ct.nii.gz
      sub-001_postop_ct_transform.npy       # created after successful registration
      sub-001_postop_ct_elec_info.csv       # created by Save Electrodes
      sub-001_postop_ct_elec_info.xlsx      # created by Save Electrodes
    seeg planning worksheet.xlsx            # optional
    sub-001.map                              # optional Ripple/Trellis map
    Electrodes.mat                           # optional comparison data
    freesurfer/
      sub-001/                               # optional FreeSurfer/FastSurfer subject
        surf/
        mri/
        label/
    registration/                            # registered CT and backend transforms
  sub-002/
    ...
```

If a patient registration directory is passed on the command line, registration
outputs are written there. Otherwise they are written to a `coregistration/`
folder beside the MRI. Saved electrode CSV/XLSX files and the automatically
saved NumPy transform are named from the CT and written beside the CT.

For example, loading `sub-001_postop_ct.nii.gz` produces:

```text
sub-001_postop_ct_transform.npy
sub-001_postop_ct_elec_info.csv
sub-001_postop_ct_elec_info.xlsx
```

When that CT is loaded again, the app looks for those exact neighboring names
and restores the saved transform and electrode locations when available.

## Optional planning worksheet

The planning worksheet is not required to detect or manually group contacts.
When present, it supplies planned trajectory names, labels, contact counts, and
ordering so that generated shaft names can be matched to the implant plan.

Use an Excel workbook (`.xlsx` or `.xlsm`). The preferred filename is:

```text
seeg planning worksheet.xlsx
```

Other filenames are accepted when they contain `planning` plus `seeg`,
`implant`, or `worksheet`. Place the workbook in the patient directory or
beside the MRI/CT. The app searches those locations and their parent folders;
it does not recursively search arbitrary sibling directories.

Each usable worksheet should contain a contiguous table whose header appears
within the first 30 rows. The recommended columns are:

| Trajectory Name | Label | Contacts |
| --- | --- | ---: |
| L Amygdala | A* | 10 |
| R Amygdala | A | 10 |
| L Hippocampus | H* | 12 |
| R Hippocampus | H | 12 |

Column behavior:

- `Trajectory Name` identifies the anatomical target. The alternatives
  `Implant Name`, `Target Name`, `Target`, `Trajectory`, and `Implant` are also
  recognized.
- `Label` is the short electrode code used to match acquisition-map prefixes.
  `Name`, `Electrode Label`, `Electrode Name`, and `Reference` are also
  recognized.
- `Contacts` is the planned number of contacts. Any header containing
  `contacts`, except a contact-length field, is accepted.
- A trajectory row must contain a trajectory name and at least a label or a
  contact count.
- Do not insert blank trajectory rows inside the table; the first blank after
  trajectory data ends that table.

With the current MCW naming convention, a trajectory name beginning with `L`
or `R` determines its hemisphere. If the name does not provide a side, an
asterisk in the worksheet label is interpreted as left and a label without an
asterisk as right. Adjust the worksheet or assign trajectories manually if a
different convention is used.

If the workbook contains an embedded planning image, the app enables its
schema-position prior when matching detected shafts to planned trajectories.
Automatic assignments should always be reviewed in the UI before export.

## Optional electrode map

A neighboring `.map` file can supply recorded contact names and numbering from
a Ripple/Trellis configuration. The parser expects semicolon-separated rows:

```text
1.A.1;LA01
1.A.2;LA02
1.B.1;RA01
1.B.2;RA02
```

- Macro-contact rows must have a hardware address beginning with `1.A` or
  `1.B`.
- The second field is the contact label.
- Contact labels should begin with `L` or `R` and end in a contact number, such
  as `LA01`, `RH12`, or `RN05-COM`.
- `1.C` and `1.D` rows are treated as micro-contact bundles and can be assigned
  to planning trajectories in the map-assignment controls.

The app matches map prefixes to worksheet trajectories using hemisphere,
worksheet label, and contact count. Ambiguous mappings remain available for
manual assignment.

## Optional FreeSurfer or FastSurfer subject

A patient-specific reconstruction enables patient cortical surfaces,
parcellation lookup, and transformation of electrode coordinates to MNI
Talairach space. Select the subject directory itself, for example
`freesurfer/sub-001/` in the suggested tree.

At minimum the subject must contain:

```text
sub-001/
  surf/
    lh.pial
    rh.pial
    lh.white
    rh.white
  mri/
```

Useful optional files include:

- `surf/lh.inflated` and `surf/rh.inflated` for inflated-surface viewing
- `mri/aparc+aseg.mgz` for volume labels and tissue classification
- `label/lh.aparc.annot` and `label/rh.aparc.annot` for cortical parcellations
- `mri/transforms/talairach.xfm` for MRI-to-MNI transformation

Without a patient reconstruction, the application can still use MNE's
`fsaverage` template and perform MRI/CT localization, but patient-specific
surface labels and MRI-to-MNI mapping are unavailable.

## Saved electrode files and cohort viewing

`Save Electrodes` writes both CSV and Excel exports. The CSV contains native
contact coordinates, shaft/contact assignments, visibility, tissue and
parcellation fields, planning metadata, map metadata, and MNI coordinates when
they have been computed. If a neighboring `Electrodes.mat` is present, the app
also attempts to add comparison information to the Excel workbook; this file
is optional.

To prepare data for MNI cohort viewing:

1. Load the patient MRI, postoperative CT, and FreeSurfer/FastSurfer subject.
2. Register the CT to MRI and detect or load the contacts.
3. Review shaft grouping and planning/map assignments.
4. Select `MRI->MNI` to populate `mni_x`, `mni_y`, and `mni_z`.
5. Select `Save Electrodes`.

The cohort loader recursively searches a selected cohort root for
`*_elec_info.csv`. It loads only rows with complete finite `mni_x`, `mni_y`,
and `mni_z` values; it does not open the patient MRI or CT. The first directory
below the selected cohort root becomes the patient identifier, which is why a
one-folder-per-patient organization is recommended.

### Color and filter a cohort with a scalar lookup table

After loading the cohort, choose **Load Lookup CSV...** and select a CSV with
`subject`, `electrode_name`, and one or more numeric value columns. The table
does not need coordinates; coordinates come from each subject's saved
`*_elec_info.csv` export. For example (illustrative values):

```csv
subject,electrode_name,test_l1_baseline_minus_lesioned_mpr,test_l2_baseline_minus_lesioned_mpr,test_l3_baseline_minus_lesioned_mpr
sub-001,ns5_1_LAMY01,0.002,-0.001,0.003
sub-001,ns5_2_LAMY02,-0.001,0.004,0.000
sub-002,ns5_1_LAMY01,0.005,0.001,-0.002
```

Matching uses **both subject and contact name**. The saved electrode CSV does
not need a subject column: its subject comes from the folder hierarchy.
SUBJ names are normalized, so `SUBJ_009` matches lookup subject `SUBJ09`
and `SUBJ_010` matches `SUBJ10`. An explicit SUBJ subject folder is recognized
even with subfolders below it or when that subject folder itself is selected.
For other naming conventions, the first folder below the cohort root supplies
the subject ID (for example, `sub-001`).
Contact names are matched against the saved `name`, `map_contact`, or
`electrode_name`. Matching ignores case and surrounding spaces. Exact names
take precedence; otherwise the final underscore-separated part is used and
terminal contact zero padding is normalized, so `ns5_1_LAMY01` can match
`LAMY01` or `LAMY1`. Hemisphere letters are preserved; region abbreviations
and subject identifiers outside the SUBJ convention are not guessed.

Duplicate subject/electrode_name rows are rejected. Ambiguous matches,
including multiple saved locations for one lookup row, are excluded from the
join. The lookup status reports matched contacts, unmatched table rows, and
ambiguous contacts; hover over it for examples of unmatched rows. Table rows
without a saved MNI contact (such as non-intracranial recording channels) are
not plotted.

- **Colors** selects a numeric column or **Default location colors**. The three
  `test_l1/l2/l3_baseline_minus_lesioned_mpr` measures appear first. Colors are
  assigned to individual contacts, not averaged over a shaft.
- **Only table electrodes** is enabled when a table is loaded. It hides
  contacts without a unique match and works independently of the color
  selection. Patient checkboxes still apply. Uncheck it to show all otherwise
  visible cohort contacts; **Clear Lookup** restores default coloring and
  removes the table filter.
- Scalar colors use blue for negative values, white at zero, and red for
  positive values. Missing, invalid, or non-finite values are gray, not zero.
  A numeric color bar identifies the measure and limits.
- **Shaft transparency**, beside the brain transparency control, adjusts only
  the tubes connecting contacts. Set it to **100%** to hide shafts and show
  only the electrode contacts. It defaults to 50% and applies to both cohort
  and patient views, including exported figures. Contact coloring and brain
  transparency are unchanged; **Labels** controls the shaft labels separately.
- **Shared L1/L2/L3 scale** is enabled by default, using the largest absolute
  value across the three measures among uniquely matched cohort contacts.
  Limits are symmetric around zero and stay fixed when patients are unchecked.
  Disable this option for a separate symmetric range for each measure. Other
  numeric columns always use their own range. An all-zero/all-missing range
  falls back to −1 to +1.
- With an L1/L2/L3 measure selected, **Export series…** saves one PNG per
  available level, with the same camera, patient selection, table filter,
  surface, and scale setting. Other scalar columns export a single figure.
  PNGs include the color bar and are saved at twice the current viewport
  resolution. A new subfolder prevents overwriting earlier figures, and
  `figure_settings.json` records the source paths, measures, limits, counts,
  camera, and display settings. The original interactive view is restored
  after export. Hide **Labels** beforehand for an uncluttered figure.

## Typical workflow

1. Start the app, optionally passing a patient registration/output directory.
2. Open the Coregistration UI and select the MRI and postoperative CT.
3. Select a FreeSurfer/FastSurfer subject if one is available.
4. Register the CT to MRI, or review a neighboring saved transform that is
   loaded automatically.
5. Detect contacts and inspect the MRI/CT overlay.
6. Group contacts into shafts and review planning worksheet or `.map`
   assignments.
7. Optionally classify contacts using the patient segmentation and transform
   them to MNI space.
8. Save the electrode locations.
9. For group visualization, choose `Load Saved Locations` and select the common
   cohort root.

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

## Contributing

Contributions are welcome! If you add a feature or fix a bug, please submit a
pull request with a brief description of the change and how you tested it.
For larger changes, open an issue first to discuss the proposed approach.
Please use synthetic data in examples and tests, and do not include patient data.

## License

Licensed under the [MIT License](LICENSE). You may use, modify, and redistribute
the software, including commercially, provided you retain the copyright and
license notice in copies or substantial portions of the software.

### Spatially pooled MNI cohort view

After loading saved MNI locations and a lookup CSV, select a numeric color column
and enable **Merge nearby**. **Maximum cluster diameter** (default 10 mm) controls
complete-linkage clustering in the original MNI coordinates, before any inflated
surface mapping. Every pair of contacts in a group lies within that distance;
isolated contacts remain individual groups. This is an adjustable spatial scale,
not an automatically estimated optimum or a statistical ALE significance map.

Each group appears as a fixed-size sphere at its centroid. Color represents the
arithmetic mean of finite contact values; missing values are excluded, and groups
with no values are gray. Each contact has equal weight, so patients with more
contacts contribute more. Labels show the mean, contact count, patient count, and
number of valid values. Patient and lookup filters are applied before grouping;
changing those filters recomputes groups. Turning the toggle off restores the
individual contact view. Existing color limits are retained for comparison.

Figure export saves the current pooling mode, diameter, group centers, source
cohort contact indices, counts, and means in `figure_settings.json`. Complete
linkage uses quadratic memory in the visible contact count; large cohorts may
need to be filtered before pooling.

### Cortical flatmap

Use the **Pial / Inflated / Flatmap** radio buttons in the Surface row to switch
views. Flatmap displays both flattened hemispheres face-on with parallel
projection and 2D navigation. Contacts map to the nearest retained white-surface
vertex in their hemisphere, using the displayed patch's vertex correspondence.
Contacts near a cut or omitted medial wall project to the nearest retained
vertex. This is a cortical projection: electrode depth and 3D distances are not
preserved. Shaft connectors are hidden to avoid lines across flatmap cuts.

Lookup scalar colors, patient filtering, nearby-contact pooling, labels, and
figure export work in flatmap mode. Pooling continues to use original MNI
coordinates. MNE's fsaverage includes the flat patches; custom subjects need
`lh.cortex.patch.flat` and `rh.cortex.patch.flat`, plus matching `white` and
`sphere` surfaces in their `surf` directory. If these are missing, selecting
Flatmap leaves the current surface selected and explains which files are needed.

### Continuous scalar field

Enable **Smooth scalar field** after selecting a numeric lookup column. This
replaces contacts/groups with a continuous, softly fading cortical heatmap.
**Smoothing width (FWHM)** controls the Gaussian width in anatomical millimetres
(default 15 mm; adjustable from 2–50 mm). The default is a starting value, not an
automatically fitted optimum. Smooth field and Merge nearby are mutually
exclusive; turn both off to return to individual contacts.

Each finite contact projects to the nearest white-surface vertex in its own
hemisphere. Gaussian weights follow shortest paths along white-surface mesh
edges, not distances across folded banks or distances in the flattened display.
The same field transfers by vertex correspondence to Pial, Inflated, and Flatmap;
flatmap cuts simply omit vertices absent from the patch. This is a cortical
projection, so it does not represent the original depth of deep electrodes.

Color shows `sum(weight * scalar) / sum(weight)`. Contacts receive equal weight
before distance weighting, so subjects with more sampled contacts contribute
more. Missing/nonfinite values contribute nothing. Opacity follows the strongest
contributing Gaussian, tapering to zero at three standard deviations; unobserved
regions remain transparent rather than becoming zero-valued. Opacity describes
spatial coverage, not statistical confidence. This is a descriptive visualization,
not an ALE significance map. Shaft lines and individual labels are omitted.

Patient/table filters and the existing fixed scalar limits still apply. Exports
record smoothing width, method, cutoff, opacity definition, and per-hemisphere
contact/covered-vertex counts in `figure_settings.json`. Surface graphs and the
latest sparse kernels are cached for reuse; changing width or the projected
contact sites rebuilds the bounded-distance kernels.

### Compact viewer controls

The viewer groups its main controls into four rows: **Data** for loading and
filtering, **View** for surfaces and export, **Scalar** for the lookup column,
merging and smoothing, and **Range ±** for color limits. Open **Display details**
for percentile and contrast settings, transparency, field visibility, and the
detailed color status. Its expanded state is preserved when switching surfaces.
Contact and lookup-match counts remain visible beneath the controls.

### Seeing small scalar values

The scalar controls include a **Color range ±** number box and logarithmic slider.
Narrowing the symmetric range reveals small differences; values outside it use
the endpoint colors. A status line reports how many visible source contact values
are outside the range. These controls never alter the lookup values, pooled
means, or smoothed means.

**Robust auto-range** uses the selected percentile (95% by default) of absolute
finite matched values. **Full range** restores the maximum absolute value.
Automatic reference values use the complete matched cohort, so changing patient
filters does not rescale colors. **Shared L1/L2/L3 auto-range** includes all three
columns when computing that reference. Degenerate zero percentiles fall back to
the full range, and all-zero/all-missing data use ±1.

Manual ranges are remembered per column. **Lock across columns** freezes the
current numeric limits for every column, including exports. Editing a range or
pressing an auto-range button updates that locked value. Unlocking returns to the
current column's manual or automatic range. Loading a new lookup table clears
manual limits and the lock to avoid reusing another table's units.

**Enhance small values (asinh)** expands colors near zero while preserving sign,
zero, and endpoint colors. It warps the color table using
`asinh(10 * value / limit) / asinh(10)`; legend ticks remain in original scalar
units and the legend title identifies asinh colors. **Field visibility** scales
smooth-field opacity from 0–300%, independently of blur width and scalar values;
100% retains the original fade, and higher settings make faint regions stronger.
The spatial cutoff is unchanged. Exports record the effective limits for each
figure plus percentile, lock, manual ranges, enhancement, visibility gain, and
counts of contact values outside the color range.

### Save a publication image

Click **Save Publication Image...** beside **Export series…** to save just the
currently displayed view. Choose lossless **PNG** or **TIFF**. The image is
rendered at high resolution (not enlarged from a screen capture), with a
**4,200-pixel longest edge** and embedded **600 DPI** metadata—7 inches along
that edge. The current aspect ratio is preserved; the viewer controls are not
included. The background is white, and the selected legend, labels, camera
composition, color limits, enhancement, and field visibility are retained. The
interactive axis widget is hidden during export to avoid VTK magnification
artifacts and restored afterward. Compose the view and choose labels before saving.

A matching `<name>.settings.json` records dimensions, print size, camera, selected
column, filters, and visualization settings. This button also supports categorical
contact colors and a brain view without a lookup table. It does not cycle through
L1/L2/L3; use **Export series…** for that. The interactive camera and background are
restored after saving, including when rendering or writing fails. The Python
`export_publication_image` method also accepts `long_edge_px` and `dpi` for other
journal-specific output sizes.
