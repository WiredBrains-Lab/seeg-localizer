"""
Author: Sunil Mathew
Date: 05 December 2025

This class implements sEEG Localizer using MNE-Python's fsaverage data.
The patient's CT/MRI can be coregistered to use the 3D model and associated segmentations of different
parts of the brain. The sEEG macro electrodes can be visualized in the 3D space.

Coregistration Workflow:
------------------------
1. Patient MRI and postoperative CT are loaded through the GUI file browser
2. Automatic registration tries multiple methods in order of preference:
   a) FreeSurfer's mri_robust_register (most robust, requires FreeSurfer installation)
   b) SimpleITK registration (pure Python, good quality)
   c) DIPY registration (pure Python, fallback option)
3. The chosen method produces: registered CT volume + transformation matrix
4. Transform is converted to MNE Transform format for compatibility
5. FreeSurfer fsaverage serves as the standard template brain for visualization
6. Final transforms enable accurate electrode localization in patient MRI space

Cohort MNI Workflow:
--------------------
1. Select a common root containing patient ``*_elec_info.csv`` exports
2. Load only rows with saved ``mni_x``, ``mni_y``, and ``mni_z`` coordinates
3. Display all contacts on fsaverage, colored by location label and filterable by patients
4. No patient CT, MRI, transform, or individual FreeSurfer surface is loaded

Registration Methods:
---------------------
- FreeSurfer (preferred): Uses Normalized Mutual Information (NMI), robust to intensity differences
- SimpleITK: Multi-resolution registration with Mattes Mutual Information metric
- DIPY: Affine registration with mutual information, pure Python fallback

Requirements:
- MNE-Python with fsaverage template data (required)
- Patient preoperative MRI (T1-weighted preferred) (required)
- Patient postoperative CT with visible electrode contacts (required)
- Or, for cohort-only viewing, saved electrode CSVs with MNI coordinates
- At least ONE of the following registration tools:
  * FreeSurfer installation with FREESURFER_HOME environment variable set
  * SimpleITK: pip install SimpleITK
  * DIPY: pip install dipy

"""

import os
import sys
import re
import time
import numpy as np
import mne
import h5py
from matplotlib import colormaps
from mne.datasets import fetch_fsaverage
import pyqtgraph as pg
# pg.setConfigOption('imageAxisOrder', 'row-major')
from pyqtgraph.dockarea.Dock import Dock

from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtWidgets import QApplication 
from pathlib import Path

import atexit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from usercontrols.checkable_combobox import CheckableComboBox

PROFILE_MERGE = bool(int(os.getenv("SEEG_LOCALIZER_PROFILE_MERGE", "1")))
from viewer3d.postop_ct_reg import RegistrationDialog
from viewer3d.mni_cohort import load_mni_cohort
from viewer3d.cohort_pooling import pool_mni_contacts
from viewer3d.cohort_smoothing import CorticalGaussianField
from viewer3d.cohort_colors import symmetric_color_limit, scalar_colormap
from viewer3d.publication_export import save_publication_image
from viewer3d.cohort_lookup import (
    LESION_COLUMNS,
    cohort_contact_is_visible,
    load_cohort_lookup,
    match_cohort_lookup,
)
from core.source_parcellation import (
    HCP_MMP_PARCS,
    aseg_name_from_path,
    normalize_parcellation_key,
    read_surface_labels,
)

GRAY_MATTER_LABELS = {
    3, 42, 8, 47, 10, 11, 12, 13, 16, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60
}
WHITE_MATTER_LABELS = {2, 41, 7, 46}


APPLICATION_NAME = "sEEG Localizer"


def _application_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("seeg-localizer")
    except PackageNotFoundError:
        return "0.1.0"


def _prepare_application_identity():
    """Set the macOS bundle name before Qt or MNE creates native menus."""
    if sys.platform == "darwin":
        try:
            from Foundation import NSBundle
        except ImportError:
            pass  # Qt uses applicationDisplayName when no bundle name is set.
        else:
            bundle = NSBundle.mainBundle()
            for info in (bundle.infoDictionary(), bundle.localizedInfoDictionary()):
                if info is not None:
                    info["CFBundleName"] = APPLICATION_NAME
                    info["CFBundleDisplayName"] = APPLICATION_NAME
    QtCore.QCoreApplication.setApplicationName(APPLICATION_NAME)
    QtCore.QCoreApplication.setOrganizationName("WiredBrainsLab")


def _apply_application_identity(app):
    """Restore our identity after MNE initializes or recreates a renderer."""
    app.setApplicationName(APPLICATION_NAME)
    app.setApplicationDisplayName(APPLICATION_NAME)
    app.setOrganizationName("WiredBrainsLab")
    app.setApplicationVersion(_application_version())


def _application_icon_path():
    """Locate the runtime PNG in a source checkout or PyInstaller bundle."""
    candidates = []
    pyinstaller_root = getattr(sys, "_MEIPASS", None)
    if pyinstaller_root:
        candidates.append(Path(pyinstaller_root) / "seeg-localizer-icon.png")
    candidates.append(
        Path(__file__).resolve().parents[1]
        / "packaging"
        / "macos"
        / "seeg-localizer-icon.png"
    )
    return next((path for path in candidates if path.is_file()), None)


def _apply_application_icon(app, window=None):
    """Set our icon before MNE initializes its Qt backend."""
    _apply_application_identity(app)
    icon_path = _application_icon_path()
    if icon_path is None:
        return None
    icon = QtGui.QIcon(str(icon_path))
    if icon.isNull():
        return None
    app.setWindowIcon(icon)
    if window is not None:
        window.setWindowIcon(icon)
    return icon

class SEEGLocalizer:

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SEEGLocalizer, cls).__new__(cls)
        return cls.instance

    def __init__(self, patient_reg_folder=None, chs_3d_view=None):
        self.patient_reg_folder = patient_reg_folder
        self.chs_3d_view = chs_3d_view
        self.label_names = []
        self.active_labels = []
        self.label_parcellation = "aparc+aseg"
        self.label_parcellation_combo = None
        self._aseg_name = "aparc+aseg"
        self._surface_label_parc = None
        self._surface_labels_by_name = {}
        self._surface_label_subject = None
        self._surface_label_subjects_dir = None
        self._plot_widget = None
        self.brain = None
        self._cleanup_registered = False
        self.mri_path = None
        self.ct_path = None
        self.trans = None  # transformation matrix for coregistration
        self.lta_transform_path = None  # FreeSurfer LTA transform file
        self.ct_registered_path = None  # Registered CT output file
        self.custom_subject = None  # Custom FreeSurfer subject ID
        self.custom_subjects_dir = None  # Custom FreeSurfer subjects directory
        self.force_fsaverage = False
        self._subject_to_fsaverage_trans = None
        self._subject_to_fsaverage_key = None
        self._subject_to_fsaverage_warned = False
        self._loaded_mri_to_subject_mri_trans = None
        self._loaded_mri_to_subject_mri_key = None
        self._loaded_mri_to_subject_mri_warned = False
        self.detected_electrodes = []
        self.detected_electrode_space = None
        self.detected_electrode_native_space = None
        self.detected_selected_shaft = None
        self.mni_cohort_electrodes = []
        self.mni_cohort_summary = None
        self.mni_cohort_root = None
        self.mni_cohort_active = False
        self.mni_cohort_selected_patients = None
        self.mni_cohort_show_labels = True
        self._mni_cohort_label_positions = []
        self._mni_cohort_label_texts = []
        self._mni_cohort_label_sides = []
        self._mni_cohort_previous_view_mode = "fsaverage"
        self.mni_cohort_load_btn = None
        self.mni_cohort_patient_combo = None
        self.mni_cohort_labels_checkbox = None
        self.mni_cohort_clear_btn = None
        self.mni_cohort_status_label = None
        self.mni_cohort_lookup = None
        self.mni_cohort_lookup_matches = None
        self.mni_cohort_scalar_column = None
        self.mni_cohort_lookup_filter = False
        self.mni_cohort_shared_scale = True
        self.mni_color_auto_mode = "full"
        self.mni_color_percentile = 95.0
        self.mni_color_manual_limits = {}
        self.mni_color_locked_limit = None
        self.mni_color_enhance = False
        self.mni_field_visibility = 1.0
        self.mni_color_controls = {}
        self._display_details_expanded = False
        self.display_details_button = None
        self.display_details_panel = None
        self.mni_cohort_smooth_enabled = False
        self.mni_cohort_smooth_fwhm = 15.0
        self.mni_cohort_smooth_checkbox = None
        self.mni_cohort_smooth_spin = None
        self._mni_cohort_smooth_geometry = {}
        self._mni_cohort_smooth_summary = {}
        self.mni_cohort_pool_enabled = False
        self.mni_cohort_pool_diameter = 10.0
        self.mni_cohort_pool_checkbox = None
        self.mni_cohort_pool_spin = None
        self._mni_cohort_pooled_groups = []
        self._mni_cohort_scalar_bar_title = None
        self.mni_cohort_lookup_load_btn = None
        self.mni_cohort_color_combo = None
        self.mni_cohort_lookup_filter_checkbox = None
        self.mni_cohort_shared_scale_checkbox = None
        self.mni_cohort_lookup_clear_btn = None
        self.mni_cohort_export_btn = None
        self.publication_image_btn = None
        self.mni_cohort_lookup_status_label = None
        self._detected_electrode_actors = []
        self._detected_shaft_actors = []
        self._detected_shaft_label_actors = []
        self.brain_surface = "pial"
        self.surface_buttons = {}
        self._pending_surface_change = None
        self._surface_change_scheduled = False
        self._surface_switch_in_progress = False
        self._flat_projection_cache = {}
        self.brain_cortex_style = "classic"
        self.brain_opacity = 0.25
        self.shaft_opacity = 0.5
        self._brain_surface_actors = []
        self._surface_vertex_cache = {}
        self._surface_kdtree_cache = {}
        self._aseg_path = None
        self._aseg_data = None
        self._aseg_affine = None
        self.coreg_panel = None
        self._viewer_splitter = None
        self._viewer_right_container = None
        self._viewer_right_layout = None
        self._orientation_widget = None
        self._orientation_widget_plotter = None
        self.subject_view_combo = None
        self.brain_transparency_slider = None
        self.brain_transparency_value_label = None
        self.shaft_transparency_slider = None
        self.shaft_transparency_value_label = None

        if self.chs_3d_view is not None:
            self._register_cleanup_hooks()
            self.init_3d_layout()
            self.setup_freesurfer_environment()

    def __del__(self):
        self._teardown_renderer()

    #region freesurfer setup
    def setup_freesurfer_environment(self):
        """Configure FreeSurfer environment variables if not already set."""
        
        # Check if FREESURFER_HOME is already set
        if os.environ.get('FREESURFER_HOME'):
            print(f"✓ FREESURFER_HOME already set: {os.environ['FREESURFER_HOME']}")
            return True
        
        print("Setting up FreeSurfer environment...")
        
        # Common FreeSurfer installation locations on macOS
        possible_locations = [
            '/Applications/freesurfer',
            '/usr/local/freesurfer',
            os.path.expanduser('~/freesurfer'),
        ]
        
        freesurfer_home = None
        for location in possible_locations:
            if os.path.isdir(location):
                print(f"  Found FreeSurfer directory: {location}")
                # Check if this is a versioned installation (e.g., /Applications/freesurfer/8.1.0)
                subdirs = [d for d in Path(location).iterdir() if d.is_dir() and d.name[0].isdigit()]
                if subdirs:
                    # Use the latest versioned subdirectory
                    freesurfer_home = str(sorted(subdirs)[-1])
                    print(f"  Using versioned installation: {freesurfer_home}")
                else:
                    freesurfer_home = location
                break
        
        if not freesurfer_home:
            print("✗ FreeSurfer installation not found in common locations")
            return False
        
        # Set FreeSurfer environment variables
        os.environ['FREESURFER_HOME'] = freesurfer_home
        print(f"✓ Set FREESURFER_HOME: {freesurfer_home}")
        
        # Add FreeSurfer bin to PATH
        bin_path = os.path.join(freesurfer_home, 'bin')
        if os.path.isdir(bin_path):
            current_path = os.environ.get('PATH', '')
            if bin_path not in current_path:
                os.environ['PATH'] = f"{bin_path}:{current_path}"
            print(f"✓ Added to PATH: {bin_path}")
        
        # Set other FreeSurfer environment variables
        subjects_dir = os.path.join(freesurfer_home, 'subjects')
        if os.path.isdir(subjects_dir):
            os.environ['SUBJECTS_DIR'] = subjects_dir
            print(f"✓ Set SUBJECTS_DIR: {subjects_dir}")
        
        # Source FreeSurfer setup if needed (for library paths on macOS)
        if sys.platform == 'darwin':  # macOS
            lib_path = os.path.join(freesurfer_home, 'lib')
            if os.path.isdir(lib_path):
                dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
                if lib_path not in dyld_path:
                    os.environ['DYLD_LIBRARY_PATH'] = f"{lib_path}:{dyld_path}"
                print(f"✓ Set DYLD_LIBRARY_PATH: {lib_path}")
        
        return True
    
    def load_freesurfer_subject(self, subject_id, subjects_dir):
        """
        Load a FreeSurfer subject for visualization instead of using fsaverage.
        
        This method allows you to visualize recon-all results from any FreeSurfer
        subjects directory, including results processed on a different computer.
        
        Parameters
        ----------
        subject_id : str
            The subject ID (must match the folder name in subjects_dir)
        subjects_dir : str or Path
            Path to the FreeSurfer subjects directory containing the subject folder
            
        Returns
        -------
        bool
            True if the subject was loaded successfully, False otherwise
            
        Examples
        --------
        Load a subject processed on another computer:
        >>> localizer = SEEGLocalizer(chs_3d_view=my_3d_view)
        >>> localizer.load_freesurfer_subject(
        ...     subject_id='patient01',
        ...     subjects_dir='/path/to/transferred/subjects'
        ... )
        
        Notes
        -----
        The subject folder must contain the complete FreeSurfer reconstruction
        output, including at minimum:
        - surf/ directory with pial and white surfaces
        - mri/ directory with brain volumes
        - label/ directory with parcellations (optional but recommended)
        
        After loading, you can visualize the patient-specific brain by calling
        any of the visualization methods. The Brain object will use the custom
        subject instead of fsaverage.
        """
        subjects_dir = Path(subjects_dir)
        subject_path = subjects_dir / subject_id
        
        # Validate the subject directory exists
        if not subject_path.exists():
            print(f"✗ Subject directory not found: {subject_path}")
            return False
        
        # Check for required subdirectories
        required_dirs = ['surf', 'mri']
        missing_dirs = [d for d in required_dirs if not (subject_path / d).exists()]
        
        if missing_dirs:
            print(f"✗ Subject directory is incomplete. Missing: {', '.join(missing_dirs)}")
            print(f"  Required directories: {', '.join(required_dirs)}")
            return False
        
        # Check for surface files
        surf_dir = subject_path / 'surf'
        required_surfaces = ['lh.pial', 'rh.pial', 'lh.white', 'rh.white']
        missing_surfaces = [s for s in required_surfaces if not (surf_dir / s).exists()]
        
        if missing_surfaces:
            print(f"✗ Missing surface files: {', '.join(missing_surfaces)}")
            return False
        
        # Store custom subject information
        self.custom_subject = subject_id
        self.custom_subjects_dir = str(subjects_dir)
        if normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg")) in HCP_MMP_PARCS:
            self.label_parcellation = "aparc+aseg"
            self.active_labels = []
        
        print(f"✓ Loaded FreeSurfer subject: {subject_id}")
        print(f"  Location: {subject_path}")
        print(f"  Surfaces: {', '.join(required_surfaces)}")
        
        # Check for optional but useful files
        mri_dir = subject_path / 'mri'
        if (mri_dir / 'aparc+aseg.mgz').exists():
            print(f"  ✓ Segmentation available (aparc+aseg.mgz)")
        
        label_dir = subject_path / 'label'
        if label_dir.exists() and (label_dir / 'lh.aparc.annot').exists():
            print(f"  ✓ Cortical parcellation available")
        
        return True
    
    def _get_brain_subject_info(self):
        """
        Get the subject ID and subjects_dir to use for Brain visualization.
        
        Returns
        -------
        tuple
            (subject_id, subjects_dir) - uses custom subject if loaded, otherwise fsaverage
        """
        if not self.force_fsaverage and self.custom_subject and self.custom_subjects_dir:
            return self.custom_subject, self.custom_subjects_dir
        # Keep MNE's fsaverage files with the writable sample dataset rather
        # than relying on SUBJECTS_DIR, which can point at a root-owned
        # FreeSurfer installation (for example, under /Applications).
        subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
        fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
        return "fsaverage", subjects_dir

    #endregion freesurfer setup

    #region 3D viewer

    def _check_render_ready(self):
        """Best-effort test that a GUI-backed OpenGL surface is available."""
        if self.chs_3d_view is None:
            return False, "3D dock is not available"

        # Wayland/X11 availability check (Linux headless environments)
        if sys.platform.startswith("linux"):
            if not any(os.environ.get(var) for var in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")):
                return False, "No DISPLAY/WAYLAND target detected for Qt"

        qt_platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
        if qt_platform in {"offscreen", "minimal", "minimalistic", "headless"}:
            return False, f"Qt platform '{qt_platform}' does not expose windows"

        try:
            app = QtWidgets.QApplication.instance()
        except RuntimeError:
            app = None

        if app is None:
            return False, "QApplication instance not initialized"

        screen = None
        try:
            screen = app.primaryScreen()
        except RuntimeError:
            screen = None

        if screen is None and hasattr(QtGui, "QGuiApplication"):
            try:
                screen = QtGui.QGuiApplication.primaryScreen()
            except RuntimeError:
                screen = None

        if screen is None:
            return False, "Qt reports no available screens (likely headless)"

        gl_ready, gl_reason = self._probe_opengl()
        if not gl_ready:
            return False, gl_reason

        return True, ""

    def _probe_opengl(self):
        """Create a throwaway OpenGL context to ensure VTK can render."""
        required_version = (2, 1)

        if not hasattr(QtGui, "QSurfaceFormat"):
            return False, "Qt build lacks QSurfaceFormat for OpenGL probing"

        version = (0, 0)

        try:
            surface_format = QtGui.QSurfaceFormat()
            surface_format.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
            surface_format.setMajorVersion(required_version[0])
            surface_format.setMinorVersion(required_version[1])

            gl_context = QtGui.QOpenGLContext()
            gl_context.setFormat(surface_format)
            if not gl_context.create():
                return False, "Unable to create an OpenGL context (needed by PyVista)"

            offscreen_surface = QtGui.QOffscreenSurface()
            offscreen_surface.setFormat(surface_format)
            offscreen_surface.create()
            if not offscreen_surface.isValid():
                return False, "Failed to create an offscreen surface for OpenGL"

            if not gl_context.makeCurrent(offscreen_surface):
                return False, "Cannot activate OpenGL context; GPU/driver missing"

            fmt = gl_context.format()
            version = (fmt.majorVersion(), fmt.minorVersion())
            gl_context.doneCurrent()

        except Exception as exc:  # pragma: no cover - GUI/driver specific
            return False, f"Qt OpenGL probe failed: {exc}"

        if version < required_version:
            return False, f"OpenGL {required_version[0]}.{required_version[1]}+ required, but got {version[0]}.{version[1]}"

        return True, ""

    def _show_disabled_message(self, reason):
        message = (
            "3D viewer disabled: "
            + reason
            + ". This prevents PyVista/VTK from creating a window."
        )
        print(f"[WARN] {message}")

        if self.chs_3d_view is None:
            return

        placeholder = QtWidgets.QLabel(message)
        placeholder.setWordWrap(True)
        placeholder.setAlignment(QtCore.Qt.AlignTop)
        placeholder.setStyleSheet(
            "QLabel { color: #aa0000; padding: 12px; font-size: 12px; }"
        )
        self.chs_3d_view.addWidget(placeholder)

    def _init_empty_brain_scene(self):
        """Create a bare brain scene when no demo dataset is loaded."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        self.brain = mne.viz.Brain(
            subject=subject_id,
            subjects_dir=subjects_dir,
            cortex=self.brain_cortex_style,
            alpha=self.brain_opacity,
            background="white",
            surf=self.brain_surface,
            views="flat" if self.brain_surface == "flat" else "lateral",
            # VTK silhouette decimation is unsuitable for planar patch meshes.
            silhouette=self.brain_surface != "flat",
            show=False,
        )
        self._finalize_brain_scene(reset_camera=True)
    
    def _init_seg_labels_for_subject(self, subject_id, subjects_dir):
        """Initialize segmentation labels for a specific FreeSurfer/FastSurfer subject."""
        try:
            # Path to the subject's segmentation file
            fname_aseg = Path(subjects_dir) / subject_id / "mri" / "aparc+aseg.mgz"
            
            if not fname_aseg.exists():
                print(f"  Warning: Segmentation file not found: {fname_aseg}")
                # Try alternative location (FastSurfer sometimes uses different structure)
                fname_aseg_alt = Path(subjects_dir) / subject_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
                if fname_aseg_alt.exists():
                    fname_aseg = fname_aseg_alt
                    print(f"  Using FastSurfer segmentation: {fname_aseg}")
                else:
                    print("  No segmentation file found - labels will not be available")
                    self.label_names = []
                    self.active_labels = []
                    return

            self._aseg_path = str(fname_aseg)
            self._aseg_name = aseg_name_from_path(fname_aseg)
            if not getattr(self, "label_parcellation", None):
                self.label_parcellation = "aparc+aseg"
            self._aseg_data = None
            self._aseg_affine = None
            
            # Get all available labels from the segmentation
            self.label_names = mne.get_volume_labels_from_aseg(str(fname_aseg))
            print(f"  Found {len(self.label_names)} segmentation labels")
            
            # Keep parcellations available in the label controls, but do not
            # eagerly render a default set of temporal regions at startup.
            # temporal_areas = [
            #     'ctx-lh-superiortemporal', 'ctx-lh-inferiortemporal',
            #     'ctx-lh-middletemporal', 'ctx-lh-temporalpole',
            #     'ctx-lh-transversetemporal', 'ctx-lh-parahippocampal',
            # ]
            # self.active_labels = [
            #     label for label in self.label_names if label in temporal_areas
            # ]
            self.active_labels = []
            
            # Add volume labels to the brain visualization
            legend_kwargs = dict(bcolor=None)
            aseg_name = "aparc+aseg" if "aparc+aseg.mgz" in str(fname_aseg) else "aparc.DKTatlas+aseg.deep"
            
            if self.active_labels:
                self.brain.add_volume_labels(aseg=aseg_name, labels=self.active_labels, legend=legend_kwargs)
                print(f"  Added {len(self.active_labels)} default segmentation labels to brain")
            else:
                print("  No default sEEG labels found in the selected atlas")
                
        except Exception as e:
            print(f"  Error loading segmentation labels: {e}")
            import traceback
            traceback.print_exc()
            self.label_names = []
            self.active_labels = []

    def _load_aseg_volume(self):
        """Load aseg volume data/affine if available."""
        if not self._aseg_path or not os.path.exists(self._aseg_path):
            return None, None
        if self._aseg_data is not None and self._aseg_affine is not None:
            return self._aseg_data, self._aseg_affine
        try:
            import nibabel as nib
            aseg_img = nib.load(self._aseg_path)
            self._aseg_data = np.asarray(aseg_img.get_fdata())
            self._aseg_affine = aseg_img.affine
            return self._aseg_data, self._aseg_affine
        except Exception as exc:
            print(f"Error loading aseg volume: {exc}")
            self._aseg_data = None
            self._aseg_affine = None
            return None, None

    def _classify_detected_electrodes_tissue(self):
        """Classify detected electrodes as gray/white based on FreeSurfer aseg."""
        if not self.detected_electrodes:
            return
        aseg_data, aseg_affine = self._load_aseg_volume()
        if aseg_data is None or aseg_affine is None:
            print("No aseg volume available for tissue classification.")
            return
        if not self.mri_path or not os.path.exists(self.mri_path):
            print("No MRI path available for tissue classification.")
            return
        try:
            import nibabel as nib
            mri_affine = nib.load(self.mri_path).affine
            inv_aseg_affine = np.linalg.inv(aseg_affine)
        except Exception as exc:
            print(f"Error preparing tissue classification: {exc}")
            return

        total = 0
        gray = 0
        white = 0
        for elec in self.detected_electrodes:
            vox = np.array([elec.x, elec.y, elec.z, 1.0], dtype=float)
            world = mri_affine @ vox
            aseg_vox = inv_aseg_affine @ world
            ix, iy, iz = np.round(aseg_vox[:3]).astype(int)
            if (
                ix < 0
                or iy < 0
                or iz < 0
                or ix >= aseg_data.shape[0]
                or iy >= aseg_data.shape[1]
                or iz >= aseg_data.shape[2]
            ):
                elec.tissue = "unknown"
                total += 1
                continue

            label = int(round(aseg_data[ix, iy, iz]))
            if label in GRAY_MATTER_LABELS:
                elec.tissue = "gray"
                gray += 1
            elif label in WHITE_MATTER_LABELS:
                elec.tissue = "white"
                white += 1
            else:
                elec.tissue = "unknown"
            total += 1

        print(
            f"Tissue classification complete: gray={gray}, white={white}, unknown={total - gray - white}"
        )

    def _update_classify_button_state(self):
        """Enable tissue classification button only when electrodes exist."""
        classify_btn = None
        if self.coreg_panel is not None and hasattr(self.coreg_panel, "classify_tissue_btn"):
            classify_btn = self.coreg_panel.classify_tissue_btn
        elif hasattr(self, "classify_tissue_btn"):
            classify_btn = self.classify_tissue_btn

        if classify_btn is None:
            return

        has_electrodes = bool(self.detected_electrodes)
        has_subject = bool(self.custom_subjects_dir and self.custom_subject)
        has_aseg = bool(self._aseg_path and os.path.exists(self._aseg_path))
        classify_btn.setEnabled(has_electrodes and has_subject and has_aseg)

    def _on_classify_tissue_clicked(self):
        """Handle manual tissue classification trigger."""
        if not self.detected_electrodes:
            QtWidgets.QMessageBox.information(
                None, "No Electrodes", "Detect electrodes first before classification."
            )
            return
        if not self._aseg_path or not os.path.exists(self._aseg_path):
            QtWidgets.QMessageBox.information(
                None,
                "No FreeSurfer Aseg",
                "Load a FreeSurfer/FastSurfer subject first to classify gray/white matter."
            )
            return
        self._classify_detected_electrodes_tissue()
        QtWidgets.QMessageBox.information(
            None, "Classification Complete", "Electrode tissue classification updated."
        )
    
    def _register_cleanup_hooks(self):
        """Ensure OpenGL widgets are torn down before Qt exits (prevents BadWindow)."""
        if self._cleanup_registered:
            return

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._teardown_renderer)

        if hasattr(self.chs_3d_view, "sigClosed"):
            try:
                self.chs_3d_view.sigClosed.connect(self._on_dock_closed)
            except Exception:
                pass

        atexit.register(self._teardown_renderer)

        self._cleanup_registered = True

    def _on_dock_closed(self, *args, **kwargs):
        self._teardown_renderer()

    def _teardown_renderer(self):
        """Release PyVista/VTK resources so X11 does not complain about dead windows."""
        if getattr(self, "_plot_widget", None) is not None:
            try:
                self._plot_widget.setParent(None)
            except Exception:
                pass
            try:
                self._plot_widget.hide()
            except Exception:
                pass
            try:
                QtCore.QTimer.singleShot(0, self._plot_widget.deleteLater)
            except Exception:
                pass
            finally:
                self._plot_widget = None

        if getattr(self, "labels_widget", None) is not None:
            try:
                self.labels_widget.setParent(None)
            except Exception:
                pass
            try:
                self.labels_widget.hide()
            except Exception:
                pass
            try:
                QtCore.QTimer.singleShot(0, self.labels_widget.deleteLater)
            except Exception:
                pass
            finally:
                self.labels_widget = None
        self._orientation_widget = None
        self._orientation_widget_plotter = None
        self._brain_surface_actors = []
        self._flat_projection_cache = {}

        brain = getattr(self, "brain", None)
        self.brain = None
        if brain is not None:
            renderer = getattr(brain, "_renderer", None)
            if renderer is not None:
                plotter = getattr(renderer, "plotter", None)
                if plotter is not None:
                    app_window = getattr(plotter, "app_window", None)
                    if app_window is not None:
                        try:
                            app_window.hide()
                        except Exception:
                            pass
                        try:
                            QtCore.QTimer.singleShot(0, app_window.deleteLater)
                        except Exception:
                            pass

                    interactor = getattr(plotter, "interactor", None)
                    if interactor is not None:
                        try:
                            interactor.hide()
                        except Exception:
                            pass
                        try:
                            interactor.setParent(None)
                        except Exception:
                            pass
                        try:
                            QtCore.QTimer.singleShot(0, interactor.deleteLater)
                        except Exception:
                            pass

                    for attr in ("app_window", "interactor"):
                        widget = getattr(plotter, attr, None)
                        if widget is not None:
                            try:
                                widget.hide()
                            except Exception:
                                pass

    def init_3d_layout(self):
        self._teardown_renderer()
        can_render, reason = self._check_render_ready()
        if not can_render:
            self._show_disabled_message(reason)
            return

        try:
            if self.brain is None:
                self._init_empty_brain_scene()

            subject_id, subjects_dir = self._get_brain_subject_info()
            self._init_seg_labels_for_subject(subject_id, subjects_dir)

            self._plot_widget = self.brain._renderer.plotter.interactor
            self._plot_widget.setParent(None)
            self._ensure_coreg_panel()
            self._populate_viewer_right()
            if self.mni_cohort_active:
                self._display_mni_cohort_electrodes()
            
        except Exception as e:
            print(f"ERROR in init_3d_layout: {e}")
            import traceback
            print(traceback.format_exc())
            
            # Add error message to the view instead of crashing
            error_label = QtWidgets.QLabel(f"3D Viewer Error: {str(e)}\nCheck console for details.")
            error_label.setStyleSheet("QLabel { color: red; padding: 20px; font-size: 12px; }")
            self.chs_3d_view.addWidget(error_label)

    def _ensure_coreg_panel(self):
        """Create the left-side coregistration panel and right-side viewer container."""
        if self.chs_3d_view is None:
            return

        if self._viewer_splitter is None:
            self.coreg_panel = RegistrationDialog(
                getattr(self, "mri_path", None),
                getattr(self, "ct_path", None),
                self.patient_reg_folder or os.getcwd(),
                parent=self.chs_3d_view,
                as_panel=True,
                controller=self,
            )
            self.coreg_panel.applied.connect(lambda dlg=self.coreg_panel: self._apply_registration_results(dlg))
            self.coreg_panel.panel_hidden.connect(self._sync_coreg_toggle_button)
            self.coreg_panel.setVisible(False)

            self._viewer_right_container = QtWidgets.QWidget()
            self._viewer_right_layout = QtWidgets.QVBoxLayout()
            self._viewer_right_layout.setContentsMargins(0, 0, 0, 0)
            self._viewer_right_layout.setSpacing(6)
            self._viewer_right_container.setLayout(self._viewer_right_layout)

            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            splitter.setChildrenCollapsible(False)
            splitter.setHandleWidth(6)
            splitter.addWidget(self.coreg_panel)
            splitter.addWidget(self._viewer_right_container)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([360, 1000])

            self._viewer_splitter = splitter
            self.chs_3d_view.addWidget(self._viewer_splitter)
        elif self._viewer_splitter.parent() is None:
            self.chs_3d_view.addWidget(self._viewer_splitter)

    def _clear_viewer_right_layout(self, preserve_widgets=None):
        """Remove widgets from the right-side viewer layout."""
        layout = self._viewer_right_layout
        if layout is None:
            return
        preserve = {id(widget) for widget in (preserve_widgets or []) if widget is not None}
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                if id(widget) not in preserve:
                    widget.deleteLater()

    def _populate_viewer_right(self):
        """Populate the right-side viewer with plot and label controls."""
        if self._viewer_right_layout is None:
            return
        self._clear_viewer_right_layout(preserve_widgets=[getattr(self, "_plot_widget", None)])
        self._plot_widget.setParent(None)
        self._viewer_right_layout.addWidget(self._plot_widget, 1)
        self._viewer_right_layout.addWidget(self.get_label_chart())
        self._ensure_orientation_widget()
        self._update_classify_button_state()
        self._sync_coreg_toggle_button()

    def _sync_coreg_toggle_button(self):
        """Update the CT-MRI coregistration toggle button text."""
        if not hasattr(self, "coreg_toggle_btn"):
            return
        is_visible = bool(self.coreg_panel and self.coreg_panel.isVisible())
        self.coreg_toggle_btn.setText(
            "Hide Coregistration UI" if is_visible else "Show Coregistration UI"
        )
        self.coreg_toggle_btn.setToolTip(
            "Show or hide the CT-MRI coregistration and electrode detection panel"
        )
        self._sync_subject_view_combo()

    def create_file_browser_interface(self):
        """Compact coregistration control for the 3D view.

        The full MRI/CT browse UI lives in the coregistration panel. Here we
        present a single button that shows or hides that panel.
        """
        file_browser_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row_layout = QtWidgets.QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)

        btn = QtWidgets.QPushButton("Show Coregistration UI")
        btn.setToolTip("Show or hide the coregistration panel (MRI/CT browse + registration)")
        btn.clicked.connect(self.toggle_coregistration_panel)
        self.coreg_toggle_btn = btn
        self.coregister_btn = btn
        row_layout.addWidget(btn)

        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItem("View")
        self.view_combo.addItems([
            "Superior",
            "Inferior",
            "Anterior",
            "LAO20",
            "LAO30",
            "RAO20",
            "RAO30",
            "Posterior",
            "Left",
            "Right",
        ])
        self.view_combo.setToolTip("Orient the 3D camera to a standard view")
        self.view_combo.currentIndexChanged.connect(self._on_view_combo_changed)
        row_layout.addWidget(self.view_combo)

        self.subject_view_combo = QtWidgets.QComboBox()
        self.subject_view_combo.addItem("fsaverage", "fsaverage")
        self.subject_view_combo.addItem("Patient", "subject")
        self.subject_view_combo.setToolTip("Switch between fsaverage and patient surfaces")
        self.subject_view_combo.currentIndexChanged.connect(self._on_subject_view_combo_changed)
        row_layout.addWidget(self.subject_view_combo)

        row_layout.addStretch()

        layout.addLayout(row_layout)

        cohort_layout = QtWidgets.QHBoxLayout()
        cohort_layout.setContentsMargins(0, 0, 0, 0)
        cohort_layout.setSpacing(6)
        cohort_label = QtWidgets.QLabel("MNI cohort:")
        cohort_label.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; color: #555555; }")
        cohort_layout.addWidget(cohort_label)

        self.mni_cohort_load_btn = QtWidgets.QPushButton("Load Saved Locations...")
        self.mni_cohort_load_btn.setToolTip(
            "Select a common patient root and load saved *_elec_info.csv MNI coordinates only; "
            "patient CT and MRI volumes are not loaded"
        )
        self.mni_cohort_load_btn.clicked.connect(self._browse_mni_cohort_directory)
        cohort_layout.addWidget(self.mni_cohort_load_btn)

        self.mni_cohort_patient_combo = CheckableComboBox(width=180)
        self.mni_cohort_patient_combo.setToolTip(
            "Check the cohort patients to display; uncheck patients to hide them"
        )
        self.mni_cohort_patient_combo.view().pressed.connect(
            self._on_mni_cohort_patient_pressed
        )
        cohort_layout.addWidget(self.mni_cohort_patient_combo)

        self.mni_cohort_labels_checkbox = QtWidgets.QCheckBox("Labels")
        self.mni_cohort_labels_checkbox.setChecked(self.mni_cohort_show_labels)
        self.mni_cohort_labels_checkbox.setToolTip(
            "Show or hide 3D electrode labels in single-patient and MNI cohort views"
        )
        self.mni_cohort_labels_checkbox.stateChanged.connect(
            self._on_mni_cohort_labels_changed
        )
        cohort_layout.addWidget(self.mni_cohort_labels_checkbox)

        self.mni_cohort_clear_btn = QtWidgets.QPushButton("Clear Cohort")
        self.mni_cohort_clear_btn.setToolTip(
            "Leave cohort mode and restore the current patient's electrodes"
        )
        self.mni_cohort_clear_btn.clicked.connect(self.clear_mni_cohort)
        cohort_layout.addWidget(self.mni_cohort_clear_btn)

        self.mni_cohort_status_label = QtWidgets.QLabel("No cohort loaded")
        self.mni_cohort_status_label.setStyleSheet("QLabel { font-size: 11px; color: #555555; }")
        cohort_layout.addWidget(self.mni_cohort_status_label)
        cohort_layout.addStretch()
        layout.addLayout(cohort_layout)
        self._create_mni_lookup_controls(layout)
        self._sync_mni_cohort_controls()

        surface_layout = QtWidgets.QHBoxLayout()
        surface_layout.setContentsMargins(0, 0, 0, 0)
        surface_label = QtWidgets.QLabel("Surface:")
        surface_label.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; color: #555555; }")
        surface_layout.addWidget(surface_label)
        self._create_surface_controls(surface_layout)
        surface_layout.addSpacing(10)
        transparency_label = QtWidgets.QLabel("Transparency:")
        transparency_label.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; color: #555555; }")
        surface_layout.addWidget(transparency_label)

        self.brain_transparency_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brain_transparency_slider.setRange(0, 100)
        self.brain_transparency_slider.setSingleStep(1)
        self.brain_transparency_slider.setFixedWidth(140)
        transparency_pct = int(round((1.0 - float(self.brain_opacity)) * 100))
        self.brain_transparency_slider.setValue(transparency_pct)
        self.brain_transparency_slider.setToolTip("Adjust cortical surface transparency")
        self.brain_transparency_slider.valueChanged.connect(self._on_brain_transparency_changed)
        surface_layout.addWidget(self.brain_transparency_slider)

        self.brain_transparency_value_label = QtWidgets.QLabel(f"{transparency_pct}%")
        self.brain_transparency_value_label.setStyleSheet("QLabel { font-size: 11px; color: #555555; min-width: 36px; }")
        surface_layout.addWidget(self.brain_transparency_value_label)

        surface_layout.addSpacing(10)
        shaft_label = QtWidgets.QLabel("Shaft transparency:")
        shaft_label.setStyleSheet("QLabel { font-size: 11px; color: #555555; }")
        surface_layout.addWidget(shaft_label)
        self.shaft_transparency_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.shaft_transparency_slider.setRange(0, 100)
        self.shaft_transparency_slider.setFixedWidth(140)
        shaft_transparency = int(round((1.0 - self.shaft_opacity) * 100))
        self.shaft_transparency_slider.setValue(shaft_transparency)
        self.shaft_transparency_slider.setToolTip(
            "Adjust the connecting shafts only. Set to 100% to hide shafts and keep contacts."
        )
        self.shaft_transparency_slider.valueChanged.connect(self._on_shaft_transparency_changed)
        surface_layout.addWidget(self.shaft_transparency_slider)
        self.shaft_transparency_value_label = QtWidgets.QLabel(f"{shaft_transparency}%")
        self.shaft_transparency_value_label.setStyleSheet(
            "QLabel { font-size: 11px; color: #555555; min-width: 36px; }"
        )
        surface_layout.addWidget(self.shaft_transparency_value_label)

        surface_layout.addStretch()
        layout.addLayout(surface_layout)

        status_label = QtWidgets.QLabel("Status: Ready")
        status_label.setStyleSheet("QLabel { font-size: 11px; color: #555555; }")
        self.coreg_status_label = status_label
        layout.addWidget(status_label)

        file_browser_widget.setLayout(layout)
        self._compact_viewer_controls(file_browser_widget, layout)
        return file_browser_widget

    def _compact_viewer_controls(self, container, layout):
        """Arrange existing controls without replacing their signals or state."""
        colors = self.mni_color_controls
        widgets = [
            self.coreg_toggle_btn, self.view_combo, self.subject_view_combo,
            self.mni_cohort_load_btn, self.mni_cohort_patient_combo,
            self.mni_cohort_labels_checkbox, self.mni_cohort_clear_btn,
            self.mni_cohort_status_label, self.mni_cohort_lookup_load_btn,
            self.mni_cohort_color_combo, self.mni_cohort_lookup_clear_btn,
            self.mni_cohort_export_btn, self.publication_image_btn,
            self.mni_cohort_lookup_filter_checkbox, self.mni_cohort_shared_scale_checkbox,
            self.mni_cohort_lookup_status_label, self.mni_cohort_pool_checkbox,
            self.mni_cohort_pool_spin, self.mni_cohort_smooth_checkbox,
            self.mni_cohort_smooth_spin, self.brain_transparency_slider,
            self.brain_transparency_value_label, self.shaft_transparency_slider,
            self.shaft_transparency_value_label, self.coreg_status_label,
            *self.surface_buttons.values(), *colors.values(),
        ]
        keep = {id(widget) for widget in widgets}

        def empty(old_layout):
            while old_layout.count():
                item = old_layout.takeAt(0)
                child_layout = item.layout()
                widget = item.widget()
                if child_layout is not None:
                    empty(child_layout)
                    child_layout.deleteLater()
                elif widget is not None and id(widget) not in keep:
                    widget.hide()
                    widget.deleteLater()
        empty(layout)
        for widget in widgets:
            widget.setParent(container)
        layout.setSpacing(3)

        def row(parent_layout, title=None):
            line = QtWidgets.QHBoxLayout()
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(7)
            if title:
                label = QtWidgets.QLabel(title)
                label.setFixedWidth(55)
                label.setStyleSheet("font-size: 11px; font-weight: bold; color: #555555;")
                line.addWidget(label)
            parent_layout.addLayout(line)
            return line

        self.mni_cohort_load_btn.setText("Reload locations…" if self.mni_cohort_active else "Load locations…")
        self.mni_cohort_lookup_load_btn.setText("Load table…")
        self.mni_cohort_clear_btn.setText("Clear locations")
        self.mni_cohort_lookup_clear_btn.setText("Clear table")
        self.mni_cohort_export_btn.setText("Export series…")
        self.mni_cohort_lookup_filter_checkbox.setText("Table contacts only")
        self.mni_cohort_pool_checkbox.setText("Merge nearby")
        self.mni_cohort_smooth_checkbox.setText("Smooth field")
        self.mni_cohort_color_combo.setMinimumContentsLength(16)
        self.mni_cohort_color_combo.setMinimumWidth(140)
        self.mni_cohort_pool_spin.setMaximumWidth(95)
        self.mni_cohort_smooth_spin.setMaximumWidth(95)
        self.brain_transparency_slider.setFixedWidth(100)
        self.shaft_transparency_slider.setFixedWidth(100)

        line = row(layout, "Data")
        for widget in (self.coreg_toggle_btn, self.mni_cohort_load_btn,
                       self.mni_cohort_lookup_load_btn, self.mni_cohort_patient_combo,
                       self.mni_cohort_lookup_filter_checkbox):
            line.addWidget(widget)
        line.addStretch()
        line.addWidget(self.mni_cohort_lookup_clear_btn)
        line.addWidget(self.mni_cohort_clear_btn)

        line = row(layout, "View")
        line.addWidget(self.subject_view_combo)
        line.addWidget(self.view_combo)
        line.addSpacing(8)
        for widget in self.surface_buttons.values():
            line.addWidget(widget)
        line.addWidget(self.mni_cohort_labels_checkbox)
        line.addStretch()
        line.addWidget(self.mni_cohort_export_btn)
        line.addWidget(self.publication_image_btn)

        line = row(layout, "Scalar")
        line.addWidget(self.mni_cohort_color_combo, 1)
        line.addSpacing(8)
        line.addWidget(self.mni_cohort_pool_checkbox)
        line.addWidget(QtWidgets.QLabel("Diameter:"))
        line.addWidget(self.mni_cohort_pool_spin)
        line.addSpacing(8)
        line.addWidget(self.mni_cohort_smooth_checkbox)
        line.addWidget(QtWidgets.QLabel("FWHM:"))
        line.addWidget(self.mni_cohort_smooth_spin)

        line = row(layout, "Range ±")
        line.addWidget(colors["spin"])
        line.addWidget(colors["slider"], 1)
        line.addWidget(colors["robust"])
        line.addWidget(colors["lock"])
        self.display_details_button = QtWidgets.QToolButton()
        self.display_details_button.setText("Display details")
        self.display_details_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.display_details_button.setCheckable(True)
        self.display_details_button.setChecked(self._display_details_expanded)
        self.display_details_button.setArrowType(QtCore.Qt.DownArrow if self._display_details_expanded else QtCore.Qt.RightArrow)
        self.display_details_button.setToolTip("Percentile, full-range reset, small-value enhancement, transparency and detailed status")
        line.addWidget(self.display_details_button)

        self.display_details_panel = QtWidgets.QWidget(container)
        details_layout = QtWidgets.QVBoxLayout(self.display_details_panel)
        details_layout.setContentsMargins(0, 2, 0, 2)
        details_layout.setSpacing(3)
        line = row(details_layout, "Contrast")
        line.addWidget(QtWidgets.QLabel("Robust percentile:"))
        line.addWidget(colors["percentile"])
        line.addWidget(colors["full"])
        line.addWidget(self.mni_cohort_shared_scale_checkbox)
        line.addWidget(colors["enhance"])
        line.addStretch()
        line = row(details_layout, "Opacity")
        line.addWidget(QtWidgets.QLabel("Brain transparency:"))
        line.addWidget(self.brain_transparency_slider)
        line.addWidget(self.brain_transparency_value_label)
        line.addWidget(QtWidgets.QLabel("Shaft transparency:"))
        line.addWidget(self.shaft_transparency_slider)
        line.addWidget(self.shaft_transparency_value_label)
        line.addWidget(QtWidgets.QLabel("Field visibility:"))
        line.addWidget(colors["visibility"])
        line.addStretch()
        line = row(details_layout)
        line.addWidget(colors["status"], 1)
        line.addWidget(self.coreg_status_label)
        layout.addWidget(self.display_details_panel)
        self.display_details_panel.setVisible(self._display_details_expanded)
        self.display_details_button.toggled.connect(self._toggle_display_details)

        line = row(layout)
        for widget in (self.mni_cohort_status_label, self.mni_cohort_lookup_status_label):
            widget.setWordWrap(False)
            widget.setMinimumWidth(0)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
            line.addWidget(widget, 1)

    def _toggle_display_details(self, expanded):
        self._display_details_expanded = bool(expanded)
        self.display_details_panel.setVisible(expanded)
        self.display_details_button.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)

    def _sync_mni_cohort_controls(self):
        """Synchronize cohort controls after a load, filter, or panel rebuild."""
        summary = self.mni_cohort_summary or {}
        patient_ids = list(summary.get("patient_ids", []) or [])
        combo = getattr(self, "mni_cohort_patient_combo", None)
        if combo is not None:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All patients", None)
            for patient_id in patient_ids:
                combo.addItem(str(patient_id), patient_id)

            selected = self.mni_cohort_selected_patients
            if selected is None:
                selected = set(patient_ids)
            else:
                selected = set(selected).intersection(patient_ids)
            self.mni_cohort_selected_patients = selected

            for row in range(combo.count()):
                item = combo.model().item(row)
                if item is None:
                    continue
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                if row == 0:
                    if patient_ids and len(selected) == len(patient_ids):
                        state = QtCore.Qt.Checked
                    elif selected:
                        state = QtCore.Qt.PartiallyChecked
                    else:
                        state = QtCore.Qt.Unchecked
                else:
                    state = (
                        QtCore.Qt.Checked
                        if combo.itemData(row) in selected
                        else QtCore.Qt.Unchecked
                    )
                item.setCheckState(state)
            self._update_mni_cohort_patient_combo_summary()
            combo.setCurrentIndex(0)
            combo.setEnabled(bool(self.mni_cohort_active and patient_ids))
            combo.blockSignals(False)

        labels_checkbox = getattr(self, "mni_cohort_labels_checkbox", None)
        if labels_checkbox is not None:
            labels_checkbox.blockSignals(True)
            labels_checkbox.setChecked(bool(self.mni_cohort_show_labels))
            labels_checkbox.setEnabled(True)
            labels_checkbox.blockSignals(False)

        load_btn = getattr(self, "mni_cohort_load_btn", None)
        if load_btn is not None:
            load_btn.setText(
                "Reload locations…" if self.mni_cohort_active else "Load locations…"
            )
        clear_btn = getattr(self, "mni_cohort_clear_btn", None)
        if clear_btn is not None:
            clear_btn.setEnabled(bool(self.mni_cohort_active))

        self._sync_mni_lookup_controls()
        self._update_mni_cohort_status()

    def _update_mni_cohort_patient_combo_summary(self):
        """Show the current multi-patient selection in the closed combo box."""
        combo = getattr(self, "mni_cohort_patient_combo", None)
        if combo is None or combo.count() == 0:
            return
        patient_count = max(0, combo.count() - 1)
        selected_count = sum(
            combo.model().item(row).checkState() == QtCore.Qt.Checked
            for row in range(1, combo.count())
        )
        if selected_count == patient_count:
            text = "All patients"
        elif selected_count == 0:
            text = "No patients selected"
        else:
            text = f"{selected_count}/{patient_count} patients"
        combo.setItemText(0, text)

    def _update_mni_cohort_status(self):
        """Update cohort counts for the currently checked patients."""
        self._sync_mni_color_controls()
        status = getattr(self, "mni_cohort_status_label", None)
        if status is None:
            return
        if not self.mni_cohort_active:
            status.setText("No cohort loaded")
            return

        summary = self.mni_cohort_summary or {}
        patient_ids = set(summary.get("patient_ids", []) or [])
        selected = self.mni_cohort_selected_patients
        if selected is None:
            selected = patient_ids
        else:
            selected = set(selected).intersection(patient_ids)
        shown = self._mni_cohort_display_indices()
        contact_count = len(shown)
        if not selected:
            status.setText("No patients selected")
        elif len(selected) == 1:
            patient_id = next(iter(selected))
            status.setText(f"{patient_id}: {contact_count} contacts")
        else:
            status.setText(
                f"{contact_count} contacts / {len(selected)} of {len(patient_ids)} patients"
            )
        if self.mni_cohort_scalar_column and self.mni_cohort_lookup:
            missing = sum(not np.isfinite(self._mni_cohort_scalar_value(index)) for index in shown)
            if self.mni_cohort_smooth_enabled:
                stats = self._mni_cohort_smooth_summary
                status.setText(status.text() + f" · smooth field {self.mni_cohort_smooth_fwhm:g} mm FWHM · "
                               f"{stats.get('contributing_contacts', 0)} finite contacts · "
                               f"{stats.get('covered_vertices', 0)} supported vertices")
            elif self.mni_cohort_pool_enabled:
                groups = self._mni_cohort_pooled_groups
                gray = sum(group["mean"] is None for group in groups)
                status.setText(status.text() + f" → {len(groups)} clusters · "
                               f"{missing} missing values excluded from means · {gray} gray clusters")
            else:
                status.setText(status.text() + f" · {missing} without values (gray)")

    def _create_mni_lookup_controls(self, layout):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.mni_cohort_lookup_load_btn = QtWidgets.QPushButton("Load Lookup CSV...")
        self.mni_cohort_lookup_load_btn.setToolTip(
            "Load subject/electrode_name keys and numeric values for individual contacts"
        )
        self.mni_cohort_lookup_load_btn.clicked.connect(self._browse_mni_cohort_lookup)
        row.addWidget(self.mni_cohort_lookup_load_btn)
        row.addWidget(QtWidgets.QLabel("Colors:"))
        self.mni_cohort_color_combo = QtWidgets.QComboBox()
        self.mni_cohort_color_combo.setMinimumContentsLength(24)
        self.mni_cohort_color_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.mni_cohort_color_combo.currentIndexChanged.connect(self._on_mni_lookup_color_changed)
        row.addWidget(self.mni_cohort_color_combo, 1)
        self.mni_cohort_lookup_clear_btn = QtWidgets.QPushButton("Clear Lookup")
        self.mni_cohort_lookup_clear_btn.clicked.connect(self.clear_mni_cohort_lookup)
        row.addWidget(self.mni_cohort_lookup_clear_btn)
        self.mni_cohort_export_btn = QtWidgets.QPushButton("Export Figures...")
        self.mni_cohort_export_btn.setToolTip(
            "Save separate L1/L2/L3 PNGs (or the selected scalar column) from this camera, "
            "plus a JSON record of the figure settings"
        )
        self.mni_cohort_export_btn.clicked.connect(self._browse_export_mni_cohort_figures)
        row.addWidget(self.mni_cohort_export_btn)
        self.publication_image_btn = QtWidgets.QPushButton("Save Publication Image...")
        self.publication_image_btn.setToolTip(
            "Save this view as a lossless PNG or TIFF: 4200 pixels on the longest edge, "
            "600 DPI (7 inches), white background, current legend/labels, and a settings sidecar."
        )
        self.publication_image_btn.clicked.connect(self._browse_save_publication_image)
        row.addWidget(self.publication_image_btn)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.mni_cohort_lookup_filter_checkbox = QtWidgets.QCheckBox("Only table electrodes")
        self.mni_cohort_lookup_filter_checkbox.setToolTip(
            "Show only contacts uniquely matched to the lookup table, even with default colors"
        )
        self.mni_cohort_lookup_filter_checkbox.toggled.connect(self._on_mni_lookup_filter_changed)
        row.addWidget(self.mni_cohort_lookup_filter_checkbox)
        self.mni_cohort_shared_scale_checkbox = QtWidgets.QCheckBox("Shared L1/L2/L3 auto-range")
        self.mni_cohort_shared_scale_checkbox.setToolTip(
            "Use identical zero-centered color limits for the three lesion measures. "
            "Limits use all matched cohort contacts and stay fixed when patients are unchecked. "
            "Manual ranges are per column unless Lock across columns is enabled."
        )
        self.mni_cohort_shared_scale_checkbox.toggled.connect(self._on_mni_lookup_scale_changed)
        row.addWidget(self.mni_cohort_shared_scale_checkbox)
        self.mni_cohort_lookup_status_label = QtWidgets.QLabel("No lookup table")
        self.mni_cohort_lookup_status_label.setWordWrap(True)
        row.addWidget(self.mni_cohort_lookup_status_label, 1)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.mni_cohort_pool_checkbox = QtWidgets.QCheckBox("Merge nearby")
        self.mni_cohort_pool_checkbox.setToolTip(
            "Spatial pooling of visible MNI contacts. Color is the arithmetic mean "
            "of available contact values; this is descriptive, not statistical ALE."
        )
        self.mni_cohort_pool_checkbox.toggled.connect(self._on_mni_pool_changed)
        row.addWidget(self.mni_cohort_pool_checkbox)
        row.addWidget(QtWidgets.QLabel("Maximum cluster diameter:"))
        self.mni_cohort_pool_spin = QtWidgets.QDoubleSpinBox()
        self.mni_cohort_pool_spin.setRange(0.1, 50.0)
        self.mni_cohort_pool_spin.setSuffix(" mm")
        self.mni_cohort_pool_spin.setDecimals(1)
        self.mni_cohort_pool_spin.setKeyboardTracking(False)
        self.mni_cohort_pool_spin.setToolTip(
            "Every pair in a cluster stays within this MNI distance. "
            "10 mm is an adjustable starting value, not an estimated optimum."
        )
        self.mni_cohort_pool_spin.valueChanged.connect(self._on_mni_pool_diameter_changed)
        row.addWidget(self.mni_cohort_pool_spin)
        row.addStretch()
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.mni_cohort_smooth_checkbox = QtWidgets.QCheckBox("Smooth scalar field")
        self.mni_cohort_smooth_checkbox.setToolTip(
            "Blend finite contact values with Gaussian weights along the cortical mesh. "
            "Color is a weighted mean; fading indicates distance from sampled sites, not confidence. "
            "Turn off both Smooth scalar field and Merge nearby to show individual contacts."
        )
        self.mni_cohort_smooth_checkbox.toggled.connect(self._on_mni_smooth_changed)
        row.addWidget(self.mni_cohort_smooth_checkbox)
        row.addWidget(QtWidgets.QLabel("Smoothing width (FWHM):"))
        self.mni_cohort_smooth_spin = QtWidgets.QDoubleSpinBox()
        self.mni_cohort_smooth_spin.setRange(2.0, 50.0)
        self.mni_cohort_smooth_spin.setDecimals(1)
        self.mni_cohort_smooth_spin.setSuffix(" mm")
        self.mni_cohort_smooth_spin.setKeyboardTracking(False)
        self.mni_cohort_smooth_spin.setToolTip(
            "Gaussian full width at half maximum along the original white surface. "
            "15 mm is a starting value, not a fitted optimum. Larger values blend more broadly."
        )
        self.mni_cohort_smooth_spin.valueChanged.connect(self._on_mni_smooth_width_changed)
        row.addWidget(self.mni_cohort_smooth_spin)
        row.addStretch()
        layout.addLayout(row)

        self._create_mni_color_controls(layout)

    def _create_mni_color_controls(self, layout):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QtWidgets.QLabel("Color range ±"))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(12)
        spin.setRange(0.000000000001, 1e12)
        spin.setKeyboardTracking(False)
        spin.setMaximumWidth(170)
        spin.setToolTip("Symmetric limits in scalar units. Values outside the range saturate at the endpoint colors.")
        spin.valueChanged.connect(self._on_mni_color_limit_changed)
        row.addWidget(spin)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, 1000)
        slider.setTracking(False)
        slider.setMinimumWidth(130)
        slider.setToolTip("Narrow the color range from the full data range down to 0.1%, on a logarithmic slider.")
        slider.valueChanged.connect(self._on_mni_color_slider_changed)
        row.addWidget(slider)
        robust = QtWidgets.QPushButton("Robust auto-range")
        robust.clicked.connect(lambda: self._set_mni_color_auto_mode("robust"))
        row.addWidget(robust)
        percentile = QtWidgets.QDoubleSpinBox()
        percentile.setRange(50, 100)
        percentile.setDecimals(1)
        percentile.setSuffix("%")
        percentile.setToolTip("Percentile of absolute finite matched values. Patient filters do not change this reference.")
        percentile.valueChanged.connect(self._on_mni_color_percentile_changed)
        row.addWidget(percentile)
        full = QtWidgets.QPushButton("Full range")
        full.clicked.connect(lambda: self._set_mni_color_auto_mode("full"))
        row.addWidget(full)
        lock = QtWidgets.QCheckBox("Lock across columns")
        lock.setToolTip("Hold the current numeric limits across all scalar columns and exports. Range edits update the locked value.")
        lock.toggled.connect(self._on_mni_color_lock_changed)
        row.addWidget(lock)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        enhance = QtWidgets.QCheckBox("Enhance small values (asinh)")
        enhance.setToolTip("Expand colors near zero with an asinh curve. Data and legend labels stay in original scalar units.")
        enhance.toggled.connect(self._on_mni_color_enhance_changed)
        row.addWidget(enhance)
        row.addWidget(QtWidgets.QLabel("Field visibility:"))
        visibility = QtWidgets.QSpinBox()
        visibility.setRange(0, 300)
        visibility.setSuffix("%")
        visibility.setKeyboardTracking(False)
        visibility.setMaximumWidth(100)
        visibility.setToolTip("Scale smooth-field opacity independently of values and blur width; 100% is the original fade.")
        visibility.valueChanged.connect(self._on_mni_field_visibility_changed)
        row.addWidget(visibility)
        status = QtWidgets.QLabel()
        status.setWordWrap(True)
        row.addWidget(status, 1)
        layout.addLayout(row)
        self.mni_color_controls = dict(spin=spin, slider=slider, robust=robust, percentile=percentile,
                                       full=full, lock=lock, enhance=enhance, visibility=visibility,
                                       status=status)

    def _sync_mni_lookup_controls(self):
        lookup = self.mni_cohort_lookup
        available = bool(self.mni_cohort_active and lookup and self.mni_cohort_scalar_column)
        if self.mni_cohort_labels_checkbox is not None:
            self.mni_cohort_labels_checkbox.setEnabled(
                not (available and self.mni_cohort_smooth_enabled)
            )
        if self.mni_cohort_smooth_checkbox is not None:
            self.mni_cohort_smooth_checkbox.blockSignals(True)
            self.mni_cohort_smooth_checkbox.setChecked(self.mni_cohort_smooth_enabled)
            self.mni_cohort_smooth_checkbox.setEnabled(available)
            self.mni_cohort_smooth_checkbox.blockSignals(False)
        if self.mni_cohort_smooth_spin is not None:
            self.mni_cohort_smooth_spin.blockSignals(True)
            self.mni_cohort_smooth_spin.setValue(self.mni_cohort_smooth_fwhm)
            self.mni_cohort_smooth_spin.setEnabled(available and self.mni_cohort_smooth_enabled)
            self.mni_cohort_smooth_spin.blockSignals(False)
        if self.mni_cohort_pool_checkbox is not None:
            self.mni_cohort_pool_checkbox.blockSignals(True)
            self.mni_cohort_pool_checkbox.setChecked(self.mni_cohort_pool_enabled)
            self.mni_cohort_pool_checkbox.setEnabled(available)
            self.mni_cohort_pool_checkbox.blockSignals(False)
        if self.mni_cohort_pool_spin is not None:
            self.mni_cohort_pool_spin.blockSignals(True)
            self.mni_cohort_pool_spin.setValue(self.mni_cohort_pool_diameter)
            self.mni_cohort_pool_spin.setEnabled(available and self.mni_cohort_pool_enabled)
            self.mni_cohort_pool_spin.blockSignals(False)
        combo = self.mni_cohort_color_combo
        if combo is not None:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Default location colors", None)
            if lookup:
                for column in lookup.columns:
                    combo.addItem(column, column)
                    combo.setItemData(combo.count() - 1, column, QtCore.Qt.ToolTipRole)
            combo.setCurrentIndex(max(0, combo.findData(self.mni_cohort_scalar_column)))
            combo.setEnabled(bool(lookup))
            combo.setToolTip(self.mni_cohort_scalar_column or "Default location colors")
            combo.blockSignals(False)
        for checkbox, value, enabled in (
            (self.mni_cohort_lookup_filter_checkbox, self.mni_cohort_lookup_filter, bool(lookup)),
            (self.mni_cohort_shared_scale_checkbox, self.mni_cohort_shared_scale,
             bool(lookup and any(column in lookup.columns for column in LESION_COLUMNS))),
        ):
            if checkbox is not None:
                checkbox.blockSignals(True)
                checkbox.setChecked(value)
                checkbox.setEnabled(enabled)
                checkbox.blockSignals(False)
        if self.mni_cohort_lookup_clear_btn is not None:
            self.mni_cohort_lookup_clear_btn.setEnabled(bool(lookup))
        if self.mni_cohort_export_btn is not None:
            self.mni_cohort_export_btn.setEnabled(bool(
                self.mni_cohort_active and lookup and self.mni_cohort_scalar_column
            ))
        if self.publication_image_btn is not None:
            self.publication_image_btn.setEnabled(self.brain is not None and hasattr(self.brain, "_renderer"))
        status = self.mni_cohort_lookup_status_label
        if status is not None:
            if lookup is None:
                status.setText("No lookup table")
                status.setToolTip("")
            elif not self.mni_cohort_active:
                status.setText(f"{len(lookup.rows)} lookup rows · load a cohort to match")
                status.setToolTip(str(lookup.path))
            else:
                matches = self.mni_cohort_lookup_matches
                status.setText(
                    f"{matches.matched_contacts}/{len(self.mni_cohort_electrodes)} contacts matched · "
                    f"{len(matches.unmatched_rows)} table rows unmatched · "
                    f"{len(matches.ambiguous_contacts)} ambiguous contacts"
                )
                sample = "\n".join(
                    f"{lookup.rows[index]['subject']} / {lookup.rows[index]['electrode_name']}"
                    for index in matches.unmatched_rows[:10]
                )
                status.setToolTip(f"{lookup.path}\nUnmatched table rows (up to 10):\n{sample}")

        self._sync_mni_color_controls()

    def _sync_mni_color_controls(self):
        controls = self.mni_color_controls
        if not controls:
            return
        available = bool(self.mni_cohort_active and self.mni_cohort_lookup and self.mni_cohort_scalar_column)
        limit = self._mni_cohort_scalar_limits()[1] if available else 1.0
        full = self._mni_auto_color_limit(100) if available else 1.0
        slider_value = int(round(1000 * (1 + np.log10(max(limit / full, 0.001)) / 3)))
        values = {"spin": limit, "slider": min(1000, max(0, slider_value)),
                  "percentile": self.mni_color_percentile, "lock": self.mni_color_locked_limit is not None,
                  "enhance": self.mni_color_enhance, "visibility": int(round(100 * self.mni_field_visibility))}
        for name, value in values.items():
            widget = controls[name]
            widget.blockSignals(True)
            if name in ("lock", "enhance"):
                widget.setChecked(value)
            else:
                widget.setValue(value)
            widget.setEnabled(available and (name != "visibility" or self.mni_cohort_smooth_enabled))
            widget.blockSignals(False)
        controls["robust"].setEnabled(available)
        controls["full"].setEnabled(available)
        if available:
            samples = np.asarray([self._mni_cohort_scalar_value(i) for i in self._mni_cohort_display_indices()])
            clipped = int(np.count_nonzero(np.isfinite(samples) & (np.abs(samples) > limit)))
            mode = "locked" if self.mni_color_locked_limit is not None else (
                "manual" if self.mni_cohort_scalar_column in self.mni_color_manual_limits else self.mni_color_auto_mode)
            controls["status"].setText(f"{mode} · ±{limit:.4g} · {clipped} visible contact values outside range (saturated)")
        else:
            controls["status"].setText("Select a scalar column to adjust contrast")

    def _refresh_mni_color_display(self):
        self._display_mni_cohort_electrodes()
        self._sync_mni_color_controls()
        self._update_mni_cohort_status()

    def _on_mni_color_limit_changed(self, value):
        if not self.mni_cohort_scalar_column or value <= 0:
            return
        self.mni_color_manual_limits[self.mni_cohort_scalar_column] = float(value)
        if self.mni_color_locked_limit is not None:
            self.mni_color_locked_limit = float(value)
        self._refresh_mni_color_display()

    def _on_mni_color_slider_changed(self, value):
        self._on_mni_color_limit_changed(self._mni_auto_color_limit(100) * 10 ** (3 * (value / 1000 - 1)))

    def _set_mni_color_auto_mode(self, mode):
        self.mni_color_auto_mode = mode
        self.mni_color_manual_limits.pop(self.mni_cohort_scalar_column, None)
        if self.mni_color_locked_limit is not None:
            self.mni_color_locked_limit = self._mni_auto_color_limit(
                self.mni_color_percentile if mode == "robust" else 100)
        self._refresh_mni_color_display()

    def _on_mni_color_percentile_changed(self, value):
        self.mni_color_percentile = float(value)
        if self.mni_color_auto_mode == "robust":
            self._set_mni_color_auto_mode("robust")

    def _on_mni_color_lock_changed(self, checked):
        self.mni_color_locked_limit = self._mni_cohort_scalar_limits()[1] if checked else None
        self._refresh_mni_color_display()

    def _on_mni_color_enhance_changed(self, checked):
        self.mni_color_enhance = bool(checked)
        self._refresh_mni_color_display()

    def _on_mni_field_visibility_changed(self, value):
        self.mni_field_visibility = value / 100.0
        self._refresh_mni_color_display()

    def _match_mni_cohort_lookup(self):
        self.mni_cohort_lookup_matches = (
            match_cohort_lookup(self.mni_cohort_electrodes, self.mni_cohort_lookup)
            if self.mni_cohort_lookup else None
        )

    def load_mni_cohort_lookup(self, path):
        """Load transactionally so a malformed replacement keeps the existing view."""
        lookup = load_cohort_lookup(path)
        matches = match_cohort_lookup(self.mni_cohort_electrodes, lookup)
        self.mni_cohort_lookup = lookup
        self.mni_color_manual_limits = {}
        self.mni_color_locked_limit = None
        self.mni_cohort_lookup_matches = matches
        self.mni_cohort_scalar_column = lookup.columns[0]
        self.mni_cohort_lookup_filter = True
        self._display_mni_cohort_electrodes()
        self._sync_mni_cohort_controls()
        return matches

    def _browse_mni_cohort_lookup(self):
        start = str(self.mni_cohort_lookup.path) if self.mni_cohort_lookup else str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Electrode Scalar Lookup Table", start, "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            matches = self.load_mni_cohort_lookup(path)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(None, "Cannot Load Lookup", str(exc))
            return
        if self.mni_cohort_active and matches.matched_contacts == 0:
            QtWidgets.QMessageBox.warning(
                None, "No Matching Contacts",
                "No contacts matched uniquely. Subjects come from the saved CSV's subject folder "
                "(SUBJ_009 matches SUBJ09); no subject column is needed in that saved CSV. "
                "The final underscore-separated part of electrode_name is matched to the saved "
                "name or map_contact (ns5_1_LAMY01 matches LAMY01). Contact zero padding is "
                "handled automatically.\n\n"
                "Uncheck 'Only table electrodes' to see the cohort while checking the names."
            )

    def clear_mni_cohort_lookup(self):
        self.mni_cohort_lookup = None
        self.mni_color_manual_limits = {}
        self.mni_color_locked_limit = None
        self.mni_cohort_lookup_matches = None
        self.mni_cohort_scalar_column = None
        self.mni_cohort_lookup_filter = False
        self._display_mni_cohort_electrodes()
        self._sync_mni_cohort_controls()

    def _on_mni_lookup_color_changed(self, index):
        self.mni_cohort_scalar_column = self.mni_cohort_color_combo.itemData(index)
        self._display_mni_cohort_electrodes()
        self._sync_mni_lookup_controls()
        self._update_mni_cohort_status()

    def _on_mni_lookup_filter_changed(self, checked):
        self.mni_cohort_lookup_filter = bool(checked)
        self._display_mni_cohort_electrodes()
        self._update_mni_cohort_status()

    def _on_mni_lookup_scale_changed(self, checked):
        self.mni_cohort_shared_scale = bool(checked)
        self._refresh_mni_color_display()

    def _on_mni_pool_changed(self, checked):
        self.mni_cohort_pool_enabled = bool(checked)
        if checked:
            self.mni_cohort_smooth_enabled = False
        self._display_mni_cohort_electrodes()
        self._sync_mni_cohort_controls()

    def _on_mni_smooth_changed(self, checked):
        self.mni_cohort_smooth_enabled = bool(checked)
        if checked:
            self.mni_cohort_pool_enabled = False
        self._display_mni_cohort_electrodes()
        self._sync_mni_cohort_controls()

    def _on_mni_smooth_width_changed(self, value):
        self.mni_cohort_smooth_fwhm = float(value)
        self._display_mni_cohort_electrodes()
        self._update_mni_cohort_status()

    def _on_mni_pool_diameter_changed(self, value):
        self.mni_cohort_pool_diameter = float(value)
        self._display_mni_cohort_electrodes()
        self._update_mni_cohort_status()

    def _mni_cohort_display_indices(self):
        matches = self.mni_cohort_lookup_matches
        return [
            index for index, electrode in enumerate(self.mni_cohort_electrodes)
            if cohort_contact_is_visible(electrode, self.mni_cohort_selected_patients)
            and (not self.mni_cohort_lookup_filter or matches is None
                 or matches.row_indices[index] is not None)
        ]

    def _mni_cohort_scalar_value(self, index):
        if not self.mni_cohort_lookup or not self.mni_cohort_scalar_column:
            return np.nan
        row = self.mni_cohort_lookup_matches.row_indices[index]
        if row is None:
            return np.nan
        return self.mni_cohort_lookup.values[self.mni_cohort_scalar_column][row]

    def _mni_auto_color_limit(self, percentile):
        column = self.mni_cohort_scalar_column
        if not self.mni_cohort_lookup or not column or not self.mni_cohort_lookup_matches:
            return 1.0
        columns = [column]
        if self.mni_cohort_shared_scale and column in LESION_COLUMNS:
            columns = [name for name in LESION_COLUMNS if name in self.mni_cohort_lookup.columns]
        rows = {index for index in self.mni_cohort_lookup_matches.row_indices if index is not None}
        values = [self.mni_cohort_lookup.values[name][index] for name in columns for index in rows]
        return symmetric_color_limit(values, percentile)

    def _mni_cohort_scalar_limits(self):
        limit = self.mni_color_locked_limit
        if limit is None:
            limit = self.mni_color_manual_limits.get(self.mni_cohort_scalar_column)
        if limit is None:
            limit = self._mni_auto_color_limit(self.mni_color_percentile if self.mni_color_auto_mode == "robust" else 100)
        return (-limit, limit)

    def _mni_figure_settings(self):
        """Snapshot common display settings for single-image and batch exports."""
        plotter = self.brain._renderer.plotter
        camera = plotter.camera
        lookup = self.mni_cohort_lookup
        matches = self.mni_cohort_lookup_matches
        return {
            "lookup_csv": str(lookup.path) if lookup else None, "cohort_root": self.mni_cohort_root,
            "selected_patients": (sorted(self.mni_cohort_selected_patients)
                                  if self.mni_cohort_selected_patients is not None else None),
            "only_table_electrodes": self.mni_cohort_lookup_filter,
            "shared_l1_l2_l3_scale": self.mni_cohort_shared_scale,
            "smooth_scalar_field": self.mni_cohort_smooth_enabled,
            "smoothing_fwhm_mm": self.mni_cohort_smooth_fwhm,
            "smoothing_method": "Gaussian weighted contact mean on white mesh-edge distances",
            "smoothing_projection": "nearest white vertex, separately per hemisphere",
            "smoothing_cutoff_sigma": 3.0,
            "smoothing_opacity": "strongest finite-contact kernel, tapered to zero at cutoff; not confidence",
            "spatial_pooling": self.mni_cohort_pool_enabled,
            "pooling_method": "complete linkage; equal contact mean; finite values only",
            "maximum_cluster_diameter_mm": self.mni_cohort_pool_diameter,
            "colormap": "RdBu_r_asinh10" if self.mni_color_enhance else "RdBu_r",
            "color_auto_mode": self.mni_color_auto_mode,
            "robust_percentile": self.mni_color_percentile,
            "manual_color_limits": dict(self.mni_color_manual_limits),
            "locked_color_limit": self.mni_color_locked_limit,
            "color_enhancement": "asinh(10*x/limit)/asinh(10)" if self.mni_color_enhance else "linear",
            "field_visibility_gain": self.mni_field_visibility,
            "out_of_range_values": "saturate at endpoint colors; source values unchanged",
            "missing_value_color": "#808080",
            "surface": self.brain_surface, "brain_opacity": self.brain_opacity,
            "shaft_opacity": self.shaft_opacity,
            "labels": self.mni_cohort_show_labels,
            "camera_position": [list(point) for point in plotter.camera_position],
            "parallel_projection": bool(camera.parallel_projection),
            "parallel_scale": float(camera.parallel_scale),
            "matched_contacts": matches.matched_contacts if matches else 0,
            "unmatched_table_rows": len(matches.unmatched_rows) if matches else 0,
            "ambiguous_contacts": len(matches.ambiguous_contacts) if matches else 0,
            "figures": [],
        }

    def _browse_save_publication_image(self):
        stem = self.mni_cohort_scalar_column or "brain"
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            None, "Save Publication Image — 4200 px, 600 DPI",
            str(Path.home() / f"{self.brain_surface}_{stem}.png"),
            "PNG image (*.png);;TIFF image (*.tif *.tiff)",
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".tif" if selected_filter.startswith("TIFF") else ".png"
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            image_path, settings_path = self.export_publication_image(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(None, "Publication Image Could Not Be Saved", str(exc))
        else:
            QtWidgets.QMessageBox.information(
                None, "Publication Image Saved",
                f"Saved {image_path.name} at 600 DPI (4200-pixel longest edge).\n\n"
                f"Image: {image_path}\nSettings: {settings_path}",
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def export_publication_image(self, path, long_edge_px=4200, dpi=600):
        """Save the currently selected view, without cycling through columns."""
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            raise ValueError("The 3D viewer is not ready")
        if self.mni_cohort_active:
            if not self._mni_cohort_display_indices():
                raise ValueError("No contacts are visible. Check the patient and lookup filters.")
            if not self._display_mni_cohort_electrodes():
                raise RuntimeError("Could not render the current cohort view")
        settings = self._mni_figure_settings()
        settings.pop("figures", None)
        scalar_mode = bool(self.mni_cohort_active and self.mni_cohort_lookup and self.mni_cohort_scalar_column)
        settings.update({
            "cohort_active": self.mni_cohort_active,
            "column": self.mni_cohort_scalar_column if scalar_mode else None,
            "color_limits": list(self._mni_cohort_scalar_limits()) if scalar_mode else None,
            "pooled_groups": self._mni_cohort_pooled_groups if self.mni_cohort_active else [],
            "smooth_field": self._mni_cohort_smooth_summary if self.mni_cohort_active else {},
            "visible_contacts": len(self._mni_cohort_display_indices()) if self.mni_cohort_active else None,
            "orientation_marker": bool(getattr(self._orientation_widget, "GetEnabled", lambda: False)()),
        })
        return save_publication_image(self.brain._renderer.plotter, path, settings,
                                      long_edge_px=long_edge_px, dpi=dpi,
                                      orientation_widget=self._orientation_widget)

    def _browse_export_mni_cohort_figures(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "Save MNI Cohort Figures In")
        if not directory:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            paths = self.export_mni_cohort_figures(directory)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(None, "Figure Export Failed", str(exc))
        else:
            QtWidgets.QMessageBox.information(
                None, "Figures Saved", f"Saved {len(paths)} figures and settings to:\n{paths[0].parent}"
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def export_mni_cohort_figures(self, directory):
        """Save independent scalar figures with a fixed camera and restore the view."""
        import json
        import tempfile

        if not self.mni_cohort_active or not self.mni_cohort_lookup or not self.mni_cohort_scalar_column:
            raise ValueError("Load a cohort and lookup table, then select a scalar color column.")
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            raise ValueError("The 3D viewer is not ready.")
        shown = self._mni_cohort_display_indices()
        if not shown:
            raise ValueError("No contacts are visible. Check the patient and lookup filters.")
        lookup = self.mni_cohort_lookup
        previous_column = self.mni_cohort_scalar_column
        columns = [previous_column]
        if previous_column in LESION_COLUMNS:
            columns = [column for column in LESION_COLUMNS if column in lookup.columns]
        matched_rows = [self.mni_cohort_lookup_matches.row_indices[index] for index in shown]
        if not any(np.isfinite(lookup.values[column][row])
                   for column in columns for row in matched_rows if row is not None):
            raise ValueError("The visible contacts have no finite values for the selected figures.")
        output = Path(tempfile.mkdtemp(prefix="mni-cohort-figures-", dir=directory))
        plotter = self.brain._renderer.plotter
        camera = plotter.camera.copy()
        paths = []
        settings = self._mni_figure_settings()
        try:
            for column in columns:
                self.mni_cohort_scalar_column = column
                plotter.camera = camera.copy()
                if not self._display_mni_cohort_electrodes():
                    raise RuntimeError(f"Could not render {column}.")
                filename = "mni_" + re.sub(r"[^A-Za-z0-9_-]", "_", column) + ".png"
                path = output / filename
                plotter.screenshot(str(path), scale=2, transparent_background=False, return_img=False)
                paths.append(path)
                settings["figures"].append({
                    "file": filename, "column": column,
                    "color_limits": list(self._mni_cohort_scalar_limits()),
                    "visible_contacts": len(shown),
                    "contacts_outside_color_range": sum(int(np.isfinite(self._mni_cohort_scalar_value(i)) and
                        abs(self._mni_cohort_scalar_value(i)) > self._mni_cohort_scalar_limits()[1]) for i in shown),
                    "pooled_groups": self._mni_cohort_pooled_groups,
                    "smooth_field": self._mni_cohort_smooth_summary,
                    "contacts_with_values": sum(int(np.isfinite(self._mni_cohort_scalar_value(index)))
                                                for index in shown),
                })
            with (output / "figure_settings.json").open("w", encoding="utf-8") as handle:
                json.dump(settings, handle, indent=2)
        except Exception as exc:
            raise RuntimeError(f"Export stopped; any completed files are in {output}.\n{exc}") from exc
        finally:
            self.mni_cohort_scalar_column = previous_column
            plotter.camera = camera
            self._display_mni_cohort_electrodes()
            self._sync_mni_cohort_controls()
        return paths

    def _browse_mni_cohort_directory(self):
        """Prompt for a cohort root and load saved MNI coordinate CSVs beneath it."""
        start_dir = self.mni_cohort_root or self.patient_reg_folder or os.getcwd()
        cohort_root = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "Select Cohort Root Containing Saved Electrode Locations",
            str(start_dir),
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if not cohort_root:
            return

        summary = self.load_mni_cohort_directory(cohort_root)
        if summary.get("contacts_loaded", 0) == 0:
            if summary.get("files_found", 0) == 0:
                detail = "No *_elec_info.csv files were found beneath the selected folder."
            else:
                detail = (
                    "Saved electrode files were found, but none contained complete "
                    "mni_x, mni_y, and mni_z values."
                )
            QtWidgets.QMessageBox.warning(None, "No MNI Cohort Contacts", detail)
            return

        error_count = len(summary.get("errors", []) or [])
        skipped = summary.get("skipped_no_mni", 0)
        detail = (
            f"Loaded {summary['contacts_loaded']} contacts from "
            f"{summary['patients_loaded']} patients ({summary['files_loaded']} files).\n\n"
            "Only saved MNI coordinates were read; patient CT and MRI files were not loaded."
        )
        if skipped or error_count:
            detail += f"\n\nSkipped {skipped} rows without complete MNI coordinates"
            if error_count:
                detail += f" and {error_count} unreadable files"
            detail += "."
        QtWidgets.QMessageBox.information(None, "MNI Cohort Loaded", detail)

    def load_mni_cohort_directory(self, cohort_root):
        """Load saved cohort contacts and display them directly on fsaverage.

        This path deliberately reads only ``*_elec_info.csv`` files. It never
        opens the corresponding CT, MRI, transform, or patient FreeSurfer data.
        """
        electrodes, summary = load_mni_cohort(cohort_root)
        if not electrodes:
            return summary

        if not self.mni_cohort_active:
            self._mni_cohort_previous_view_mode = (
                "subject"
                if self.custom_subject and self.custom_subjects_dir and not self.force_fsaverage
                else "fsaverage"
            )

        self.mni_cohort_electrodes = electrodes
        self.mni_cohort_summary = summary
        self.mni_cohort_root = str(cohort_root)
        self.mni_cohort_active = True
        self.mni_cohort_selected_patients = None
        self._match_mni_cohort_lookup()

        subject_id, _ = self._get_brain_subject_info()
        if subject_id != "fsaverage":
            self.set_subject_view_mode("fsaverage")
        else:
            self._display_mni_cohort_electrodes()
        self._sync_subject_view_combo()
        self._sync_mni_cohort_controls()
        if self.coreg_panel is not None and hasattr(self.coreg_panel, "_refresh_subject_view_combo"):
            self.coreg_panel._refresh_subject_view_combo()
        if hasattr(self, "coreg_status_label"):
            self.coreg_status_label.setText(
                f"Status: MNI cohort on fsaverage - {summary['contacts_loaded']} contacts, "
                f"{summary['patients_loaded']} patients"
            )
        return summary

    def _on_mni_cohort_patient_pressed(self, index):
        """Toggle one patient, or all patients, in the cohort display."""
        combo = getattr(self, "mni_cohort_patient_combo", None)
        if combo is None or not self.mni_cohort_active:
            return

        row = index.row()
        if row < 0 or row >= combo.count():
            return
        if row == 0:
            all_checked = all(
                combo.model().item(item_row).checkState() == QtCore.Qt.Checked
                for item_row in range(1, combo.count())
            )
            state = QtCore.Qt.Unchecked if all_checked else QtCore.Qt.Checked
            for item_row in range(1, combo.count()):
                combo.model().item(item_row).setCheckState(state)
        else:
            item = combo.model().item(row)
            state = (
                QtCore.Qt.Unchecked
                if item.checkState() == QtCore.Qt.Checked
                else QtCore.Qt.Checked
            )
            item.setCheckState(state)

        selected = {
            combo.itemData(item_row)
            for item_row in range(1, combo.count())
            if combo.model().item(item_row).checkState() == QtCore.Qt.Checked
        }
        self.mni_cohort_selected_patients = selected
        all_item = combo.model().item(0)
        if selected and len(selected) == combo.count() - 1:
            all_item.setCheckState(QtCore.Qt.Checked)
        elif selected:
            all_item.setCheckState(QtCore.Qt.PartiallyChecked)
        else:
            all_item.setCheckState(QtCore.Qt.Unchecked)
        self._update_mni_cohort_patient_combo_summary()
        combo.setCurrentIndex(0)
        self._display_mni_cohort_electrodes()
        self._update_mni_cohort_status()

    @staticmethod
    def _mni_cohort_location_key(electrode):
        """Return the saved alphabetic location label used for cohort coloring."""
        for field in ("trajectory_label", "map_shaft", "source_shaft", "shaft", "name"):
            value = electrode.get(field)
            text = str(value or "").strip()
            if not text:
                continue
            if field == "name":
                text = re.sub(r"(?:[_\-\s]?\d+)(?:ref)?$", "", text, flags=re.IGNORECASE)
            if text:
                return text.upper()
        return "UNASSIGNED"

    @staticmethod
    def _mni_cohort_location_colors(location_keys):
        """Assign stable categorical colors from the location label itself."""
        palette = []
        for cmap_name in ("tab20", "tab20b", "tab20c"):
            cmap = colormaps.get_cmap(cmap_name)
            palette.extend(tuple(cmap(idx / 19.0)[:3]) for idx in range(20))
        color_map = {}
        for key in set(location_keys):
            normalized = str(key or "UNASSIGNED").strip().upper()
            if re.fullmatch(r"[A-Z]", normalized):
                palette_index = ord(normalized) - ord("A")
            else:
                stable_value = sum(
                    (index + 1) * ord(char)
                    for index, char in enumerate(normalized)
                )
                palette_index = stable_value % len(palette)
            color_map[key] = palette[palette_index % len(palette)]
        return color_map

    def _add_mni_cohort_labels(self, plotter):
        """Add independently anchored cohort labels outside the shaft endpoints."""
        if not self.mni_cohort_show_labels or not self._mni_cohort_label_positions:
            return
        label_sides = list(self._mni_cohort_label_sides or [])
        if len(label_sides) != len(self._mni_cohort_label_positions):
            label_sides = ["right"] * len(self._mni_cohort_label_positions)

        # PyVista's batched vtkLabelPlacementMapper can replace the supplied
        # point positions during its collision-avoidance pass.  In MNE's real
        # renderer this caused screen-left labels to jump back across the shaft,
        # even after their anchors were shifted by the full measured text width.
        # Billboard actors retain their exact 3-D anchor and support dependable
        # left/right text justification, so use one lightweight actor per shaft.
        try:
            from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D

            renderer = plotter.renderer
            render_window = plotter.render_window
            if render_window is not None:
                window_width = float(render_window.GetSize()[0])
            else:
                window_width = float(plotter.window_size[0])
            projected_sides = []

            for position in self._mni_cohort_label_positions:
                renderer.SetWorldPoint(
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    1.0,
                )
                renderer.WorldToDisplay()
                display_x = renderer.GetDisplayPoint()[0]
                projected_sides.append(
                    "left" if display_x < window_width / 2.0 else "right"
                )
            if len(projected_sides) == len(label_sides):
                label_sides = projected_sides
        except Exception:
            try:
                from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D
            except Exception as exc:
                print(f"Could not create MNI cohort label actors: {exc}")
                return

        for position, label, side in zip(
            self._mni_cohort_label_positions,
            self._mni_cohort_label_texts,
            label_sides,
        ):
            actor = vtkBillboardTextActor3D()
            actor.SetInput(str(label))
            actor.SetPosition(*(float(value) for value in position))
            actor.SetDisplayOffset(-5 if side == "left" else 5, 0)

            text_property = actor.GetTextProperty()
            text_property.SetFontSize(12)
            text_property.SetBold(True)
            text_property.SetFontFamilyToArial()
            text_property.SetColor(0.2, 0.2, 0.2)
            text_property.SetBackgroundColor(0.5, 0.5, 0.5)
            text_property.SetBackgroundOpacity(0.65)
            text_property.SetVerticalJustificationToCentered()
            if side == "left":
                text_property.SetJustificationToRight()
            else:
                text_property.SetJustificationToLeft()
            actor.ForceOpaqueOn()

            plotter.add_actor(
                actor,
                reset_camera=False,
                pickable=False,
                render=False,
            )
            self._detected_shaft_label_actors.append(actor)

    def _on_mni_cohort_labels_changed(self, state):
        """Show or hide 3D electrode labels without rebuilding electrode meshes."""
        self.mni_cohort_show_labels = bool(state == QtCore.Qt.Checked)
        self._refresh_mni_cohort_label_actors()
        self._sync_mni_cohort_controls()

    def _refresh_mni_cohort_label_actors(self):
        """Recreate label actors for the current camera without touching meshes."""
        if not self.mni_cohort_active:
            for actor in getattr(self, "_detected_shaft_label_actors", []):
                actor.SetVisibility(bool(self.mni_cohort_show_labels))
            if self.brain is not None and hasattr(self.brain, "_renderer"):
                self._render_plotter(self.brain._renderer.plotter)
            return
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            return
        plotter = self.brain._renderer.plotter
        for actor in getattr(self, "_detected_shaft_label_actors", []):
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        self._detected_shaft_label_actors = []
        if self.mni_cohort_show_labels:
            try:
                self._add_mni_cohort_labels(plotter)
            except Exception as exc:
                print(f"Could not show MNI cohort labels: {exc}")
        self._render_plotter(plotter)

    def _display_mni_smooth_field(self, plotter, indices):
        """Display a continuous cortical field with distance-based opacity."""
        import pyvista as pv

        self._mni_cohort_pooled_groups = []
        self._mni_cohort_label_positions = []
        self._mni_cohort_label_texts = []
        self._mni_cohort_label_sides = []
        stats = {"contributing_contacts": 0, "covered_vertices": 0, "hemispheres": {}}
        self._mni_cohort_smooth_summary = stats
        finite_indices = [index for index in indices if np.isfinite(self._mni_cohort_scalar_value(index))]
        if not finite_indices:
            self._render_plotter(plotter)
            return True
        points = np.asarray([[self.mni_cohort_electrodes[index][key]
                              for key in ("mni_x", "mni_y", "mni_z")]
                             for index in finite_indices], dtype=float)
        valid = np.all(np.isfinite(points), axis=1)
        values = np.asarray([self._mni_cohort_scalar_value(index) for index in finite_indices])[valid]
        points = points[valid]
        subject, subjects_dir = self._get_brain_subject_info()
        geometry = getattr(self.brain, "geo", {})
        title = ("Smooth mean (asinh colors) · " if self.mni_color_enhance else "Smooth mean · ") + self.mni_cohort_scalar_column.replace("_baseline_minus_", "\nbaseline_minus_")
        for hemi in ("lh", "rh"):
            selected = points[:, 0] <= 0 if hemi == "lh" else points[:, 0] > 0
            if not np.any(selected):
                continue
            geo = geometry.get(hemi)
            if geo is None:
                raise ValueError(f"No displayed {hemi} cortical geometry for smoothing")
            key = (str(subjects_dir), subject, hemi)
            if key not in self._mni_cohort_smooth_geometry:
                white, faces = mne.surface.read_surface(
                    str(Path(subjects_dir) / subject / "surf" / f"{hemi}.white")
                )
                self._mni_cohort_smooth_geometry[key] = CorticalGaussianField(white, faces)
            field = self._mni_cohort_smooth_geometry[key]
            if len(field.vertices) != len(geo.coords):
                raise ValueError("Displayed and white surface vertex counts differ")
            means, opacity = field.evaluate(points[selected], values[selected], self.mni_cohort_smooth_fwhm)
            retained = np.unique(geo.faces)
            covered = int(np.count_nonzero(opacity[retained] > 0))
            stats["contributing_contacts"] += int(selected.sum())
            stats["covered_vertices"] += covered
            stats["hemispheres"][hemi] = {"contacts": int(selected.sum()), "covered_vertices": covered}
            if not covered:
                continue
            # Lift the overlay very slightly to avoid coincident-surface flicker.
            normals = np.asarray(geo.nn)
            coords = np.asarray(geo.coords) + 0.15 * normals
            faces = np.column_stack([np.full(len(geo.faces), 3), geo.faces]).ravel()
            mesh = pv.PolyData(coords, faces)
            mesh.point_data["cohort_scalar"] = means
            mesh.point_data["field_opacity"] = np.clip(opacity * self.mni_field_visibility, 0, 1)
            actor = plotter.add_mesh(
                mesh, scalars="cohort_scalar", preference="point", cmap=scalar_colormap(self.mni_color_enhance),
                clim=self._mni_cohort_scalar_limits(), opacity="field_opacity",
                nan_opacity=0.0, lighting=False, reset_camera=False,
                interpolate_before_map=True, show_scalar_bar=True,
                scalar_bar_args={
                    "title": title, "n_labels": 5, "fmt": "%.3g",
                    "vertical": False, "position_x": 0.2, "position_y": 0.03,
                    "width": 0.6, "height": 0.12, "title_font_size": 13,
                    "label_font_size": 11,
                },
            )
            self._detected_electrode_actors.append(actor)
            self._mni_cohort_scalar_bar_title = title
        self._render_plotter(plotter)
        return True

    def _display_mni_cohort_electrodes(self):
        """Render batched contacts with categorical colors or per-contact scalars."""
        if not self.mni_cohort_active:
            return False
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            return False

        try:
            import pyvista as pv
            from matplotlib import colors as mcolors

            t0 = time.perf_counter()
            selected_patients = self.mni_cohort_selected_patients
            shown_indices = set(self._mni_cohort_display_indices())
            scalar_mode = bool(self.mni_cohort_lookup and self.mni_cohort_scalar_column)
            coords_by_shaft = {}
            for index, electrode in enumerate(self.mni_cohort_electrodes):
                patient_id = electrode.get("cohort_patient") or "Unknown"
                if not cohort_contact_is_visible(electrode, selected_patients):
                    continue
                source_shaft = electrode.get("source_shaft") or electrode.get("shaft") or "Unassigned"
                location_key = self._mni_cohort_location_key(electrode)
                coords = [
                    electrode.get("mni_x"),
                    electrode.get("mni_y"),
                    electrode.get("mni_z"),
                ]
                try:
                    world = np.asarray(coords, dtype=float)
                except (TypeError, ValueError):
                    continue
                if world.shape != (3,) or not np.all(np.isfinite(world)):
                    continue
                coords_by_shaft.setdefault(
                    (patient_id, source_shaft, location_key), []
                ).append((world, index))

            location_colors = self._mni_cohort_location_colors(
                [location_key for _, _, location_key in coords_by_shaft]
            )

            plotter = self.brain._renderer.plotter
            self._clear_detected_electrode_actors(plotter)
            self._detected_shaft_cache = None
            self._mni_cohort_smooth_summary = {}
            if scalar_mode and self.mni_cohort_smooth_enabled:
                return self._display_mni_smooth_field(plotter, sorted(shown_indices))

            contact_meshes = {}
            tube_meshes = {}
            label_positions = []
            label_texts = []
            label_sides = []
            total_contacts = 0
            self._mni_cohort_pooled_groups = []
            if scalar_mode and self.mni_cohort_pool_enabled:
                indices = sorted(shown_indices)
                indices = [index for index in indices if np.all(np.isfinite(np.asarray([
                    self.mni_cohort_electrodes[index].get(key)
                    for key in ("mni_x", "mni_y", "mni_z")], dtype=float)))]
                groups = pool_mni_contacts(
                    [[self.mni_cohort_electrodes[index][key]
                      for key in ("mni_x", "mni_y", "mni_z")] for index in indices],
                    [self._mni_cohort_scalar_value(index) for index in indices],
                    [self.mni_cohort_electrodes[index].get("cohort_patient") for index in indices],
                    self.mni_cohort_pool_diameter,
                )
                centers = np.asarray([group["center"] for group in groups]).reshape(-1, 3)
                if len(centers) and self.brain_surface in ("inflated", "flat"):
                    centers = self._map_points_to_display_surface(centers)
                for number, (group, center) in enumerate(zip(groups, centers), start=1):
                    group["members"] = [indices[index] for index in group["members"]]
                    mesh = pv.Sphere(center=center, radius=2.0, theta_resolution=24, phi_resolution=24)
                    value = group["mean"] if group["mean"] is not None else np.nan
                    mesh.cell_data["cohort_scalar"] = np.full(mesh.n_cells, value)
                    contact_meshes.setdefault(None, []).append(mesh)
                    mean_text = f"{value:.3g}" if np.isfinite(value) else "no values"
                    label_positions.append(center)
                    label_sides.append("left" if center[0] >= 0 else "right")
                    label_texts.append(f"C{number} · {mean_text} · {group['contact_count']} contacts / "
                                       f"{group['patient_count']} patients · n={group['value_count']}")
                self._mni_cohort_pooled_groups = groups
                total_contacts = len(indices)
                coords_by_shaft = {}
            for (patient_id, source_shaft, location_key), coords in coords_by_shaft.items():
                world = np.asarray([point for point, _ in coords], dtype=float)
                contact_indices = np.asarray([index for _, index in coords])
                if not any(index in shown_indices for index in contact_indices):
                    continue
                if self.brain_surface in ("inflated", "flat"):
                    world = self._map_points_to_display_surface(world)

                center = np.mean(world, axis=0)
                if len(world) > 1:
                    _, _, vh = np.linalg.svd(world - center, full_matrices=False)
                    axis = vh[0]
                    if np.linalg.norm(axis) == 0:
                        axis = np.array([0.0, 0.0, 1.0])
                else:
                    axis = np.array([0.0, 0.0, 1.0])
                axis = axis / max(np.linalg.norm(axis), 1e-6)

                projection = (world - center) @ axis
                order = np.argsort(projection)
                projection = projection[order]
                world = world[order]
                contact_indices = contact_indices[order]
                included = np.asarray([index in shown_indices for index in contact_indices])
                spacing = 2.0
                if len(projection) > 1:
                    differences = np.diff(projection)
                    differences = differences[differences > 1e-3]
                    if len(differences):
                        spacing = float(np.median(differences))
                contact_height = max(1.0, min(3.0, spacing * 0.6))

                # Fit the axis/spacing to the full shaft before masking contacts,
                # so filtering cannot change a retained contact's geometry.
                mesh_key = None if scalar_mode else location_key
                location_contact_meshes = contact_meshes.setdefault(mesh_key, [])
                for point, index in zip(world[included], contact_indices[included]):
                    mesh = pv.Cylinder(
                        center=point,
                        direction=axis,
                        radius=0.8,
                        height=contact_height,
                        resolution=24,
                    )
                    if scalar_mode:
                        mesh.cell_data["cohort_scalar"] = np.full(
                            mesh.n_cells, self._mni_cohort_scalar_value(index), dtype=float
                        )
                    location_contact_meshes.append(mesh)
                    total_contacts += 1

                line_start = center + axis * np.min(projection[included])
                line_end = center + axis * np.max(projection[included])
                # Only join adjacent retained contacts; do not draw a shaft
                # through contacts omitted by the lookup filter.
                for pair in range(len(world) - 1) if self.brain_surface != "flat" else ():
                    if not (included[pair] and included[pair + 1]):
                        continue
                    start = center + axis * projection[pair]
                    end = center + axis * projection[pair + 1]
                    if not np.allclose(start, end):
                        line = pv.Line(start, end)
                        tube_meshes.setdefault(mesh_key, []).append(
                            line.tube(radius=0.4, n_sides=16)
                        )

                # Anchor each label beyond the endpoint farthest from the brain
                # center. The text is then justified away from the midline so its
                # background does not extend back across the electrode contacts.
                start_lateral = abs(float(line_start[0]))
                end_lateral = abs(float(line_end[0]))
                if start_lateral > end_lateral:
                    outer_endpoint = line_start
                elif end_lateral > start_lateral:
                    outer_endpoint = line_end
                elif np.linalg.norm(line_start) > np.linalg.norm(line_end):
                    outer_endpoint = line_start
                else:
                    outer_endpoint = line_end
                lateral_sign = np.sign(float(outer_endpoint[0]))
                if lateral_sign == 0:
                    lateral_sign = np.sign(float(center[0])) or 1.0
                outward = np.array([lateral_sign, 0.0, 0.0])
                label_offset = max(6.0, contact_height * 3.0)
                label_positions.append(outer_endpoint + outward * label_offset)
                # This is the fallback side until the point is projected through
                # the current camera immediately before creating its label actor.
                label_sides.append("left" if outer_endpoint[0] >= 0 else "right")
                if str(source_shaft).strip().upper() == location_key:
                    label_text = f"{patient_id} · {source_shaft}"
                else:
                    label_text = f"{patient_id} · {source_shaft} ({location_key})"
                label_texts.append(label_text)

            for location_key, meshes in contact_meshes.items():
                if not meshes:
                    continue
                merged = pv.merge(meshes, merge_points=False)
                if scalar_mode:
                    title = self.mni_cohort_scalar_column.replace("_baseline_minus_", "\nbaseline_minus_")
                    if self.mni_color_enhance:
                        title = "Asinh colors · " + title
                    if self.mni_cohort_pool_enabled:
                        title = "Mean · " + title
                    actor = plotter.add_mesh(
                        merged, scalars="cohort_scalar", preference="cell",
                        cmap=scalar_colormap(self.mni_color_enhance), clim=self._mni_cohort_scalar_limits(),
                        nan_color="#808080", nan_opacity=1.0,
                        opacity=1.0, lighting=False, reset_camera=False, show_scalar_bar=True,
                        scalar_bar_args={
                            "title": title, "n_labels": 5, "fmt": "%.3g",
                            "vertical": False, "position_x": 0.2, "position_y": 0.03,
                            "width": 0.6, "height": 0.12, "title_font_size": 13,
                            "label_font_size": 11, "nan_annotation": True,
                        },
                    )
                    self._mni_cohort_scalar_bar_title = title
                else:
                    actor = plotter.add_mesh(
                        merged,
                        color=location_colors.get(location_key, "#333333"),
                        opacity=0.9, reset_camera=False,
                    )
                self._detected_electrode_actors.append(actor)

            for location_key, meshes in tube_meshes.items():
                if not meshes:
                    continue
                merged = pv.merge(meshes, merge_points=False)
                rgb = np.asarray(
                    mcolors.to_rgb(location_colors.get(location_key, "#777777")),
                    dtype=float,
                )
                muted_color = tuple((rgb * 0.4 + np.array([0.6, 0.6, 0.6]) * 0.6).tolist())
                actor = plotter.add_mesh(
                    merged, color="#888888" if scalar_mode else muted_color,
                    opacity=self.shaft_opacity, reset_camera=False,
                )
                self._detected_shaft_actors.append(actor)

            self._mni_cohort_label_positions = label_positions
            self._mni_cohort_label_texts = label_texts
            self._mni_cohort_label_sides = label_sides
            self._add_mni_cohort_labels(plotter)

            self._render_plotter(plotter)
            print(
                f"Displayed {total_contacts} saved MNI cohort contacts "
                f"({len(self._mni_cohort_pooled_groups)} pooled groups) "
                f"on fsaverage in {time.perf_counter() - t0:.2f}s."
            )
            return True
        except Exception as exc:
            print(f"Error displaying MNI cohort electrodes: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def clear_mni_cohort(self):
        """Leave cohort mode and restore the current patient's electrode display."""
        if not self.mni_cohort_active:
            return
        previous_view = self._mni_cohort_previous_view_mode
        self.mni_cohort_active = False
        self.mni_cohort_electrodes = []
        self.mni_cohort_summary = None
        self.mni_cohort_selected_patients = None
        self._match_mni_cohort_lookup()
        self._mni_cohort_label_positions = []
        self._mni_cohort_label_texts = []
        self._mni_cohort_label_sides = []

        if (
            previous_view == "subject"
            and self.custom_subject
            and self.custom_subjects_dir
            and self.force_fsaverage
        ):
            self.set_subject_view_mode("subject")
        elif self.brain is not None and hasattr(self.brain, "_renderer"):
            plotter = self.brain._renderer.plotter
            self._clear_detected_electrode_actors(plotter)
            if self.detected_electrodes:
                self._display_detected_electrodes(
                    self.detected_electrodes, self.detected_electrode_space
                )
            else:
                self._render_plotter(plotter)
        self._sync_subject_view_combo()
        self._sync_mni_cohort_controls()
        if self.coreg_panel is not None and hasattr(self.coreg_panel, "_refresh_subject_view_combo"):
            self.coreg_panel._refresh_subject_view_combo()
        if hasattr(self, "coreg_status_label"):
            self.coreg_status_label.setText("Status: MNI cohort cleared")

    def toggle_coregistration_panel(self):
        """Toggle visibility of CT-MRI coregistration controls."""
        if not self.coreg_panel:
            return
        showing = not self.coreg_panel.isVisible()
        self.coreg_panel.setVisible(showing)
        if showing and self._viewer_splitter is not None:
            total = self._viewer_splitter.size().width()
            if total <= 0:
                total = 1000
            left = int(total * 0.6)
            right = total - left
            self._viewer_splitter.setSizes([left, right])
        self._sync_coreg_toggle_button()

    def _on_view_combo_changed(self, index):
        """Handle selection from the 3D view dropdown."""
        if index <= 0:
            return
        if not hasattr(self, "view_combo"):
            return
        view_label = self.view_combo.currentText()
        self._set_brain_view(view_label)

    def _sync_subject_view_combo(self):
        combo = getattr(self, "subject_view_combo", None)
        if combo is None:
            return
        has_subject = bool(self.custom_subject and self.custom_subjects_dir)
        combo.blockSignals(True)
        combo.setEnabled(has_subject and not self.mni_cohort_active)
        mode = "fsaverage"
        if has_subject and not self.force_fsaverage:
            mode = "subject"
        if self.mni_cohort_active:
            mode = "fsaverage"
            combo.setToolTip("MNI cohort mode uses the fsaverage surface")
        else:
            combo.setToolTip("Switch between fsaverage and patient surfaces")
        idx = combo.findData(mode)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _on_subject_view_combo_changed(self, _index):
        combo = getattr(self, "subject_view_combo", None)
        if combo is None:
            return
        mode = combo.currentData() or "fsaverage"
        self.set_subject_view_mode(mode)
        if self.coreg_panel is not None and hasattr(self.coreg_panel, "_refresh_subject_view_combo"):
            self.coreg_panel._refresh_subject_view_combo()

    def _set_brain_view(self, view_label):
        """Orient the 3D viewer camera to a standard view."""
        if self.brain is None:
            return
        if self.brain_surface == "flat":
            self.brain.show_view("flat")
            plotter = self.brain._renderer.plotter
            plotter.enable_parallel_projection()
            return
        view_map = {
            "Superior": ("dorsal", None),
            "Inferior": ("ventral", None),
            "Anterior": ("rostral", None),
            "LAO20": ("rostral", "lh"),
            "LAO30": ("rostral", "lh"),
            "RAO20": ("rostral", "lh"),
            "RAO30": ("rostral", "lh"),
            "Posterior": ("caudal", None),
            "Left": ("lateral", "lh"),
            "Right": ("lateral", "rh"),
        }
        view, hemi = view_map.get(view_label, (None, None))
        if view is None:
            return
        # RAS camera azimuth: anterior is 90 degrees, patient left is 180.
        # Use explicit angles so the oblique presets are hemisphere-independent.
        view_kwargs = {}
        oblique_azimuths = {"LAO20": 110.0, "LAO30": 120.0,
                            "RAO20": 70.0, "RAO30": 60.0}
        if view_label in oblique_azimuths:
            # Leave roll unset: MNE sets superior (+Z) as camera up.
            # An explicit VTK roll of zero overrides that anatomical alignment.
            view_kwargs = dict(
                azimuth=oblique_azimuths[view_label],
                elevation=90.0,
            )
        try:
            if hemi is None:
                self.brain.show_view(view, **view_kwargs)
            else:
                self.brain.show_view(view, hemi=hemi, **view_kwargs)
        except Exception:
            try:
                self.brain.show_view(view, **view_kwargs)
            except Exception:
                pass
        if self.mni_cohort_active and self._mni_cohort_label_positions:
            # Camera projection changes can swap which anatomical hemisphere is
            # on each screen side; rebuild labels with the correct outside anchor.
            QtCore.QTimer.singleShot(0, self._refresh_mni_cohort_label_actors)

    def _cache_brain_surface_actors(self):
        """Capture current cortical surface actors so opacity changes only affect the brain mesh."""
        self._brain_surface_actors = []
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            return

        plotter = getattr(self.brain._renderer, "plotter", None)
        if plotter is None:
            return

        actor_candidates = []
        actor_sources = [
            getattr(plotter, "actors", None),
            getattr(getattr(plotter, "renderer", None), "actors", None),
        ]
        for source in actor_sources:
            if source is None:
                continue
            if hasattr(source, "values"):
                try:
                    actor_candidates.extend(list(source.values()))
                    continue
                except Exception:
                    pass
            try:
                actor_candidates.extend(list(source))
            except Exception:
                continue

        seen = set()
        for actor in actor_candidates:
            if actor is None:
                continue
            key = id(actor)
            if key in seen:
                continue
            seen.add(key)
            if hasattr(actor, "GetProperty"):
                self._brain_surface_actors.append(actor)

    def _apply_brain_opacity(self, opacity, render=True):
        """Apply cortical surface opacity to cached brain mesh actors."""
        try:
            opacity = float(opacity)
        except Exception:
            return
        opacity = max(0.0, min(1.0, opacity))
        self.brain_opacity = opacity

        if self.brain is None or not hasattr(self.brain, "_renderer"):
            return
        if not self._brain_surface_actors:
            self._cache_brain_surface_actors()

        changed = False
        for actor in list(self._brain_surface_actors):
            try:
                prop = actor.GetProperty()
                if prop is None:
                    continue
                prop.SetOpacity(opacity)
                changed = True
            except Exception:
                continue

        if changed and render:
            try:
                self.brain._renderer.plotter.render()
            except Exception:
                pass

    def _finalize_brain_scene(self, reset_camera=True):
        """Apply consistent mesh visibility and camera settings after Brain creation."""
        app = QApplication.instance()
        if app is not None:
            _apply_application_identity(app)
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            return
        self._cache_brain_surface_actors()
        self._apply_brain_opacity(self.brain_opacity, render=False)

        view_label = "Left"
        combo = getattr(self, "view_combo", None)
        if combo is not None:
            try:
                current = str(combo.currentText() or "").strip()
            except Exception:
                current = ""
            if current and current != "View":
                view_label = current
        self._set_brain_view(view_label)

        plotter = getattr(self.brain._renderer, "plotter", None)
        if plotter is None:
            return
        if reset_camera:
            try:
                plotter.reset_camera()
            except Exception:
                pass
            try:
                plotter.reset_camera_clipping_range()
            except Exception:
                pass
        self._render_plotter(plotter)

    def _on_shaft_transparency_changed(self, value):
        """Update connecting tubes immediately, leaving contacts and brain unchanged."""
        value = max(0, min(100, int(value)))
        self.shaft_opacity = (100 - value) / 100.0
        if self.shaft_transparency_value_label is not None:
            self.shaft_transparency_value_label.setText(f"{value}%")
        for actor in self._detected_shaft_actors:
            actor.GetProperty().SetOpacity(self.shaft_opacity)
        if self.brain is not None and hasattr(self.brain, "_renderer"):
            self._render_plotter(self.brain._renderer.plotter)

    def _on_brain_transparency_changed(self, value):
        """Handle UI changes for cortical transparency."""
        try:
            value = int(value)
        except Exception:
            return
        value = max(0, min(100, value))
        if self.brain_transparency_value_label is not None:
            self.brain_transparency_value_label.setText(f"{value}%")
        opacity = (100 - value) / 100.0
        self._apply_brain_opacity(opacity, render=True)

    def _surface_files_available(self, surface_name):
        """Check if the requested FreeSurfer surface files are available."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        if not subject_id or not subjects_dir:
            return False
        surf_dir = Path(subjects_dir) / subject_id / "surf"
        if surface_name == "flat":
            return all((surf_dir / f"{hemi}.{name}").is_file()
                       for hemi in ("lh", "rh")
                       for name in ("cortex.patch.flat", "sphere", "white"))
        return (surf_dir / f"lh.{surface_name}").exists() and (surf_dir / f"rh.{surface_name}").exists()

    def _get_surface_vertices(self, surface_name):
        """Load and cache FreeSurfer surface vertices for both hemispheres."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        if not subject_id or not subjects_dir:
            return None
        key = (str(subjects_dir), subject_id, surface_name)
        cache = self._surface_vertex_cache
        if key in cache:
            return cache[key]
        surf_dir = Path(subjects_dir) / subject_id / "surf"
        lh_path = surf_dir / f"lh.{surface_name}"
        rh_path = surf_dir / f"rh.{surface_name}"
        if not (lh_path.exists() and rh_path.exists()):
            return None
        try:
            rr_lh, _ = mne.surface.read_surface(str(lh_path))
            rr_rh, _ = mne.surface.read_surface(str(rh_path))
        except Exception:
            return None
        rr = np.vstack([rr_lh, rr_rh])
        cache[key] = rr
        return rr

    def _get_surface_kdtree(self, surface_name):
        """Return a cached KDTree for a surface."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        if not subject_id or not subjects_dir:
            return None
        key = (str(subjects_dir), subject_id, surface_name)
        cache = self._surface_kdtree_cache
        if key in cache:
            return cache[key]
        rr = self._get_surface_vertices(surface_name)
        if rr is None:
            return None
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return None
        tree = cKDTree(rr)
        cache[key] = tree
        return tree

    def _map_points_to_display_surface(self, points):
        if self.brain_surface != "flat":
            return self._map_points_to_inflated(points)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not len(points):
            return points
        from scipy.spatial import cKDTree

        # Use the actual displayed geometry: MNE rotates flat patches and offsets
        # hemispheres. Reading raw patch coordinates would misalign overlays.
        subject, subjects_dir = self._get_brain_subject_info()
        geometry = getattr(self.brain, "geo", {})
        mapped = np.empty_like(points)
        for hemi in ("lh", "rh"):
            selected = points[:, 0] <= 0 if hemi == "lh" else points[:, 0] > 0
            if not np.any(selected):
                continue
            geo = geometry.get(hemi)
            if geo is None:
                raise ValueError(f"No displayed {hemi} flatmap geometry")
            key = (str(subjects_dir), subject, hemi, id(geo))
            if key not in self._flat_projection_cache:
                white, _ = mne.surface.read_surface(
                    str(Path(subjects_dir) / subject / "surf" / f"{hemi}.white")
                )
                if len(white) != len(geo.coords):
                    raise ValueError("Flatmap and white surface vertex counts differ")
                # The patch omits medial-wall/cut vertices. Their zero-filled
                # coordinates must never attract contacts to the map origin.
                retained = np.unique(geo.faces)
                if not len(retained):
                    raise ValueError(f"The {hemi} flatmap has no retained triangles")
                self._flat_projection_cache[key] = (cKDTree(white[retained]), retained)
            tree, retained = self._flat_projection_cache[key]
            _, nearest = tree.query(points[selected])
            mapped[selected] = geo.coords[retained[nearest]]
        return mapped

    def _map_points_to_inflated(self, points):
        """Map MRI-space points to inflated surface using vertex correspondence."""
        white = self._get_surface_vertices("white")
        inflated = self._get_surface_vertices("inflated")
        if white is None or inflated is None or white.shape != inflated.shape:
            return self._project_points_to_surface(points, "inflated")
        tree = self._get_surface_kdtree("white")
        if tree is None:
            return self._project_points_to_surface(points, "inflated")
        _, idx = tree.query(points)
        return inflated[idx]

    def _project_points_to_surface(self, points, surface_name):
        """Project points to the nearest vertex on the given surface."""
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return points
        rr = self._get_surface_vertices(surface_name)
        if rr is None:
            return points
        tree = self._get_surface_kdtree(surface_name)
        if tree is not None:
            _, idx = tree.query(points)
            return rr[idx]
        diff = points[:, np.newaxis, :] - rr[np.newaxis, :, :]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        return rr[idx]

    def _create_surface_controls(self, layout):
        self.surface_button_group = QtWidgets.QButtonGroup(layout.parentWidget())
        self.surface_button_group.setExclusive(True)
        self.surface_buttons = {}
        for surface, label in (("pial", "Pial"), ("inflated", "Inflated"), ("flat", "Flatmap")):
            button = QtWidgets.QRadioButton(label)
            button.setChecked(self.brain_surface == surface)
            button.setEnabled(not self._surface_switch_in_progress)
            button.toggled.connect(
                lambda checked, target=surface: self._queue_surface_change(target) if checked else None
            )
            self.surface_button_group.addButton(button)
            self.surface_buttons[surface] = button
            layout.addWidget(button)
        self.surface_buttons["flat"].setToolTip(
            "Cortical flatmap. Contacts project to the nearest retained white-surface "
            "vertex in their hemisphere; depth and 3D distances are not preserved."
        )

    def _sync_surface_toggle(self):
        """Sync mutually exclusive radio buttons without triggering a reload."""
        buttons = self.surface_buttons
        for button in buttons.values():
            button.blockSignals(True)
        for surface, button in buttons.items():
            button.setChecked(self.brain_surface == surface)
        for button in buttons.values():
            button.blockSignals(False)

    def _queue_surface_change(self, target_surface):
        """Let Qt finish the radio-button event before replacing its controls.

        Brain construction can process nested Qt events. Reloading synchronously
        from toggled can delete the emitting button before setChecked finishes
        its accessibility notification, causing a native macOS crash.
        """
        if self._surface_switch_in_progress:
            return
        self._pending_surface_change = target_surface
        if not self._surface_change_scheduled:
            self._surface_change_scheduled = True
            QtCore.QTimer.singleShot(0, self._apply_pending_surface_change)

    def _apply_pending_surface_change(self):
        self._surface_change_scheduled = False
        target = self._pending_surface_change
        self._pending_surface_change = None
        if target is None or self._surface_switch_in_progress:
            return
        self._surface_switch_in_progress = True
        for button in self.surface_buttons.values():
            button.setEnabled(False)
        try:
            self._select_brain_surface(target)
        except Exception as exc:
            # Never propagate an exception out of a Qt timer callback.
            QtWidgets.QMessageBox.warning(None, "Surface Could Not Be Loaded", str(exc))
        finally:
            self._surface_switch_in_progress = False
            self._sync_surface_toggle()
            # Reload may have replaced the widgets; use the current controls.
            for button in self.surface_buttons.values():
                button.setEnabled(True)

    def _select_brain_surface(self, target_surface):
        if target_surface == self.brain_surface:
            return
        if not self._surface_files_available(target_surface):
            detail = ("Flatmap requires lh/rh.cortex.patch.flat, sphere, and white files "
                      "in the current subject's surf directory."
                      if target_surface == "flat" else
                      f"Surface files for '{target_surface}' were not found for the current subject.")
            QtWidgets.QMessageBox.warning(None, "Surface Not Found", detail)
            self._sync_surface_toggle()
            return
        previous_surface = self.brain_surface
        try:
            self._reload_brain_with_subject(surface=target_surface)
        except Exception as exc:
            self.brain_surface = previous_surface
            self._sync_surface_toggle()
            QtWidgets.QMessageBox.warning(None, "Surface Could Not Be Loaded", str(exc))
            return
        self._sync_surface_toggle()

    def _get_current_view(self):
        """Get the current view label from the view combo."""
        combo = getattr(self, "view_combo", None)
        if combo is not None:
            try:
                current = str(combo.currentText() or "").strip()
                if current and current != "View":
                    return current
            except Exception:
                pass
        return "Left"  # Default view
    
    
    def browse_mri_file(self):
        """Open file browser to select MRI file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select MRI File",
            "",
            "NIfTI Files (*.nii *.nii.gz);;MGZ Files (*.mgz);;All Files (*.*)"
        )
        if file_path:
            self.mri_path = file_path
            if self.coreg_panel is not None:
                self.coreg_panel.mri_path = file_path
                if hasattr(self.coreg_panel, "mri_path_display"):
                    self.coreg_panel.mri_path_display.setText(file_path)
                if hasattr(self.coreg_panel, "_sync_output_dir"):
                    self.coreg_panel._sync_output_dir()
            self.update_button_states()
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: MRI loaded - {os.path.basename(file_path)}")
            
            # Display MRI in 3D viewer
            # self._display_patient_volumes()
    
    def browse_ct_file(self):
        """Open file browser to select postoperative CT file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select Postoperative CT File",
            "",
            "NIfTI Files (*.nii *.nii.gz);;DICOM Files (*.dcm);;All Files (*.*)"
        )
        if file_path:
            self.ct_path = file_path
            if self.coreg_panel is not None:
                self.coreg_panel.ct_path = file_path
                if hasattr(self.coreg_panel, "ct_path_display"):
                    self.coreg_panel.ct_path_display.setText(file_path)
            self.update_button_states()
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: CT loaded - {os.path.basename(file_path)}")
            
            # Display CT in 3D viewer (unregistered)
            # self._display_patient_volumes()
    
    def _prompt_fastsurfer_reconstruction(self):
        """Ask user if they want to run FastSurfer for surface reconstruction."""
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Surface Reconstruction Options")
        msg_box.setIcon(QtWidgets.QMessageBox.Question)
        
        msg_text = (
            "Would you like to use FreeSurfer/FastSurfer surface reconstruction?\n\n"
            "FastSurfer is a deep learning-based alternative to FreeSurfer that:\n"
            "• Creates high-quality cortical surface meshes from MRI\n"
            "• Performs brain segmentation and parcellation\n"
            "• Runs 10-100x faster than FreeSurfer (minutes vs hours)\n"
            "• Produces FreeSurfer-compatible outputs\n\n"
            "Options:\n"
            "• Run FastSurfer: Start new surface reconstruction (5-20 min)\n"
            "• Browse Existing: Use previously generated FreeSurfer/FastSurfer outputs\n"
            "• Maybe Later: Skip for now\n\n"
            "Requirements for FastSurfer:\n"
            "• FastSurfer installation (Docker or local)\n"
            "• GPU recommended but not required"
        )
        msg_box.setText(msg_text)
        
        run_btn = msg_box.addButton("Run FastSurfer", QtWidgets.QMessageBox.AcceptRole)
        browse_btn = msg_box.addButton("Browse Existing", QtWidgets.QMessageBox.ActionRole)
        info_btn = msg_box.addButton("More Info", QtWidgets.QMessageBox.HelpRole)
        later_btn = msg_box.addButton("Maybe Later", QtWidgets.QMessageBox.RejectRole)
        
        msg_box.setDefaultButton(later_btn)
        msg_box.exec_()
        
        clicked_button = msg_box.clickedButton()
        
        if clicked_button == run_btn:
            self._run_fastsurfer()
        elif clicked_button == browse_btn:
            self._browse_existing_surfaces()
        elif clicked_button == info_btn:
            self._show_fastsurfer_info()
    
    def _browse_existing_surfaces(self):
        """Browse for existing FreeSurfer/FastSurfer reconstruction outputs."""
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Browse FreeSurfer/FastSurfer Outputs")
        msg_box.setIcon(QtWidgets.QMessageBox.Question)
        
        msg_text = (
            "Select what type of FreeSurfer/FastSurfer data you want to load:\n\n"
            "• Subjects Directory: Browse to a FreeSurfer/FastSurfer subjects directory\n"
            "  (contains multiple subject folders with recon-all outputs)\n\n"
            "• Single Subject: Browse to a single subject's reconstruction folder\n"
            "  (e.g., /path/to/subjects_dir/subject_name)\n\n"
            "FreeSurfer/FastSurfer directory structure should contain:\n"
            "  - surf/ folder (surface meshes)\n"
            "  - mri/ folder (volumes)\n"
            "  - label/ folder (parcellations)"
        )
        msg_box.setText(msg_text)
        
        subjects_dir_btn = msg_box.addButton("Browse Subjects Directory", QtWidgets.QMessageBox.ActionRole)
        single_subject_btn = msg_box.addButton("Browse Single Subject", QtWidgets.QMessageBox.ActionRole)
        cancel_btn = msg_box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        
        msg_box.setDefaultButton(cancel_btn)
        msg_box.exec_()
        
        clicked_button = msg_box.clickedButton()
        
        if clicked_button == subjects_dir_btn:
            self._browse_subjects_directory()
        elif clicked_button == single_subject_btn:
            self._browse_single_subject()
    
    def _browse_subjects_directory(self, start_dir=None):
        """Browse for FreeSurfer/FastSurfer subjects directory."""
        subjects_dir = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "Select FreeSurfer/FastSurfer Subjects Directory",
            start_dir or os.path.expanduser("~"),
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        
        if not subjects_dir:
            return
        
        # List available subjects in the directory
        try:
            potential_subjects = []
            for item in os.listdir(subjects_dir):
                subject_path = os.path.join(subjects_dir, item)
                if os.path.isdir(subject_path):
                    # Check if it looks like a FreeSurfer subject
                    surf_dir = os.path.join(subject_path, 'surf')
                    mri_dir = os.path.join(subject_path, 'mri')
                    if os.path.exists(surf_dir) and os.path.exists(mri_dir):
                        potential_subjects.append(item)
            
            if not potential_subjects:
                QtWidgets.QMessageBox.warning(
                    None,
                    "No Subjects Found",
                    f"No valid FreeSurfer/FastSurfer subjects found in:\n{subjects_dir}\n\n"
                    "Make sure the directory contains subject folders with 'surf' and 'mri' subdirectories."
                )
                return
            
            # Let user select a subject
            subject_id, ok = QtWidgets.QInputDialog.getItem(
                None,
                "Select Subject",
                f"Found {len(potential_subjects)} subject(s). Select one:",
                potential_subjects,
                0,
                False
            )
            
            if ok and subject_id:
                self._load_freesurfer_subject(subjects_dir, subject_id)
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                "Error",
                f"Error reading subjects directory:\n{str(e)}"
            )
    
    def _browse_single_subject(self):
        """Browse for a single FreeSurfer/FastSurfer subject folder."""
        subject_dir = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "Select FreeSurfer/FastSurfer Subject Folder",
            os.path.expanduser("~"),
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        
        if not subject_dir:
            return
        
        # Validate it's a FreeSurfer subject
        surf_dir = os.path.join(subject_dir, 'surf')
        mri_dir = os.path.join(subject_dir, 'mri')
        
        if not (os.path.exists(surf_dir) and os.path.exists(mri_dir)):
            QtWidgets.QMessageBox.warning(
                None,
                "Invalid Subject Folder",
                f"The selected folder does not appear to be a valid FreeSurfer/FastSurfer subject.\n\n"
                f"Selected: {subject_dir}\n\n"
                "A valid subject folder should contain:\n"
                "  - surf/ subdirectory (surface meshes)\n"
                "  - mri/ subdirectory (volumes)"
            )
            return
        
        # Extract subject ID and subjects directory
        subject_id = os.path.basename(subject_dir)
        subjects_dir = os.path.dirname(subject_dir)
        
        self._load_freesurfer_subject(subjects_dir, subject_id)
    
    def _load_freesurfer_subject(self, subjects_dir, subject_id):
        """Load a FreeSurfer/FastSurfer subject for use in visualization."""
        try:
            # Verify the subject has required files
            subject_path = os.path.join(subjects_dir, subject_id)
            surf_dir = os.path.join(subject_path, 'surf')
            
            # Check for key surface files
            required_surfaces = ['lh.pial', 'rh.pial', 'lh.white', 'rh.white']
            missing_surfaces = []
            for surf in required_surfaces:
                if not os.path.exists(os.path.join(surf_dir, surf)):
                    missing_surfaces.append(surf)
            
            if missing_surfaces:
                reply = QtWidgets.QMessageBox.question(
                    None,
                    "Missing Surface Files",
                    f"Some expected surface files are missing:\n{', '.join(missing_surfaces)}\n\n"
                    "Do you want to continue anyway?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.No:
                    return
            
            # Store the subject information
            self.custom_subjects_dir = subjects_dir
            self.custom_subject = subject_id
            # A loaded subject remains available for later patient viewing, but
            # cohort MNI coordinates must stay on fsaverage while cohort mode is active.
            self.force_fsaverage = bool(self.mni_cohort_active)
            if normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg")) in HCP_MMP_PARCS:
                self.label_parcellation = "aparc+aseg"
                self.active_labels = []
            self._subject_to_fsaverage_trans = None
            self._subject_to_fsaverage_key = None
            self._subject_to_fsaverage_warned = False
            self._sync_subject_view_combo()
            if self.coreg_panel is not None:
                if hasattr(self.coreg_panel, "fs_subject_path_display"):
                    subject_path = os.path.join(subjects_dir, subject_id)
                    self.coreg_panel.fs_subject_path_display.setText(subject_path)
                    self.coreg_panel.fs_subject_path = subject_path
                if hasattr(self.coreg_panel, "_refresh_subject_view_combo"):
                    self.coreg_panel._refresh_subject_view_combo()
                if hasattr(self.coreg_panel, "_update_mni_button_state"):
                    self.coreg_panel._update_mni_button_state()
            
            # Update UI
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(
                    f"Status: FreeSurfer/FastSurfer subject loaded - {subject_id}"
                )
            
            print(f"\n✓ Loaded FreeSurfer/FastSurfer subject:")
            print(f"  Subject ID: {subject_id}")
            print(f"  Subjects Dir: {subjects_dir}")
            print(f"  Subject Path: {subject_path}")
            
            # Reload the 3D brain viewer with the new subject
            try:
                print("  Reloading 3D viewer with custom subject...")
                self._reload_brain_with_subject()
                if not self.detected_electrodes and self.coreg_panel is not None:
                    self.detected_electrodes = list(getattr(self.coreg_panel, "before_electrodes", []))
                    native_space = getattr(self.coreg_panel, "electrode_space", None)
                    display_space = getattr(self.coreg_panel, "electrode_display_space", None)
                    if native_space:
                        self.detected_electrode_native_space = native_space
                    if display_space:
                        self.detected_electrode_space = display_space
                if self.detected_electrodes:
                    self._display_detected_electrodes(
                        self.detected_electrodes, self.detected_electrode_space
                    )
                if self.coreg_panel is not None and hasattr(self.coreg_panel, "_update_mni_button_state"):
                    self.coreg_panel._update_mni_button_state()
                
                QtWidgets.QMessageBox.information(
                    None,
                    "Subject Loaded",
                    f"FreeSurfer/FastSurfer subject loaded successfully!\n\n"
                    f"Subject: {subject_id}\n"
                    f"Location: {subjects_dir}\n\n"
                    "The 3D viewer has been updated with the custom brain surfaces."
                )
            except Exception as reload_error:
                print(f"  Warning: Could not reload 3D viewer: {reload_error}")
                QtWidgets.QMessageBox.warning(
                    None,
                    "Subject Loaded (Viewer Not Updated)",
                    f"FreeSurfer/FastSurfer subject loaded successfully!\n\n"
                    f"Subject: {subject_id}\n"
                    f"Location: {subjects_dir}\n\n"
                    "Note: The 3D viewer could not be automatically updated.\n"
                    f"Reason: {str(reload_error)}\n\n"
                    "The subject is loaded and will be used for future visualizations."
                )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                "Error Loading Subject",
                f"Error loading FreeSurfer/FastSurfer subject:\n{str(e)}"
            )
    
    def _find_freesurfer_subjects_dir(self):
        """Return a likely FreeSurfer SUBJECTS_DIR if available."""
        subjects_dir = os.environ.get('SUBJECTS_DIR')
        if subjects_dir and os.path.isdir(subjects_dir):
            return subjects_dir
        freesurfer_home = os.environ.get('FREESURFER_HOME')
        if freesurfer_home:
            candidate = os.path.join(freesurfer_home, 'subjects')
            if os.path.isdir(candidate):
                return candidate
        candidate = os.path.expanduser('~/freesurfer/subjects')
        if os.path.isdir(candidate):
            return candidate
        return None

    def _prompt_freesurfer_browse_if_available(self):
        """Prompt user to browse FreeSurfer subjects if a directory is available."""
        subjects_dir = self._find_freesurfer_subjects_dir()
        if not subjects_dir:
            return False
        reply = QtWidgets.QMessageBox.question(
            None,
            "FreeSurfer Subjects Found",
            f"FreeSurfer subjects directory found:\n{subjects_dir}\n\n"
            "Would you like to browse and load a subject now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._browse_subjects_directory(start_dir=subjects_dir)
            return True
        return False

    def _reload_brain_with_subject(self, surface=None):
        """Reload the 3D brain viewer with the current custom subject."""
        target_surface = surface or self.brain_surface
        if target_surface == "flat" and not self._surface_files_available("flat"):
            if surface is not None:
                raise FileNotFoundError("The selected subject does not have flatmap surface files")
            # A newly loaded patient may not have flat patches, unlike fsaverage.
            target_surface = "pial"
            print("  Flatmap unavailable for this subject; using pial surface")
        if self.brain is None:
            self.brain_surface = target_surface
            print("  No brain viewer to reload - initializing from scratch")
            self.init_3d_layout()
            return

        # Load first so invalid/mismatched patches cannot destroy the current view.
        subject_id, subjects_dir = self._get_brain_subject_info()
        new_brain = mne.viz.Brain(
            subject=subject_id,
            subjects_dir=subjects_dir,
            cortex=self.brain_cortex_style,
            alpha=self.brain_opacity,
            background="white",
            surf=target_surface,
            views="flat" if target_surface == "flat" else "lateral",
            # Avoid decimating a planar patch just to create a 3D silhouette.
            silhouette=target_surface != "flat",
            show=False,
        )
        self._teardown_renderer()
        self.brain = new_brain
        self.brain_surface = target_surface
        self._finalize_brain_scene(reset_camera=True)
        
        # Reinitialize segmentation labels with custom subject
        print("  Loading segmentation labels for custom subject...")
        self._init_seg_labels_for_subject(subject_id, subjects_dir)
        self._classify_detected_electrodes_tissue()
        self._update_classify_button_state()
        
        # Re-embed in Qt widget (similar to init_3d_layout)
        if self.chs_3d_view is not None:
            print("  Embedding brain viewer in Qt widget...")
            self._plot_widget = self.brain._renderer.plotter.interactor
            self._plot_widget.setParent(None)
            self._ensure_coreg_panel()
            self._populate_viewer_right()

            if self.coreg_panel is not None:
                self.coreg_panel.mri_path = self.mri_path
                self.coreg_panel.ct_path = self.ct_path
                if self.mri_path and hasattr(self.coreg_panel, "mri_path_display"):
                    self.coreg_panel.mri_path_display.setText(self.mri_path)
                if self.ct_path and hasattr(self.coreg_panel, "ct_path_display"):
                    self.coreg_panel.ct_path_display.setText(self.ct_path)

            if self.mni_cohort_active:
                self._display_mni_cohort_electrodes()
            elif self.detected_electrodes:
                self._display_detected_electrodes(
                    self.detected_electrodes, self.detected_electrode_space
                )
            self._sync_surface_toggle()
            if self.brain_transparency_slider is not None:
                self.brain_transparency_slider.blockSignals(True)
                transparency_pct = int(round((1.0 - float(self.brain_opacity)) * 100))
                self.brain_transparency_slider.setValue(transparency_pct)
                self.brain_transparency_slider.blockSignals(False)
            if self.brain_transparency_value_label is not None:
                transparency_pct = int(round((1.0 - float(self.brain_opacity)) * 100))
                self.brain_transparency_value_label.setText(f"{transparency_pct}%")
        
        print(f"  ✓ Brain viewer reloaded with subject: {subject_id}")

    def set_subject_view_mode(self, mode):
        """Switch between custom subject and fsaverage for visualization."""
        if self.mni_cohort_active and mode != "fsaverage":
            return
        use_fsaverage = (mode == "fsaverage")
        if use_fsaverage == self.force_fsaverage:
            return
        if not use_fsaverage and not (self.custom_subject and self.custom_subjects_dir):
            return
        if not use_fsaverage and normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg")) in HCP_MMP_PARCS:
            self.label_parcellation = "aparc+aseg"
            self.active_labels = []
        self.force_fsaverage = use_fsaverage
        try:
            self._reload_brain_with_subject()
            if hasattr(self, "coreg_status_label"):
                label = "fsaverage" if use_fsaverage else self.custom_subject
                self.coreg_status_label.setText(f"Status: Using {label} for visualization")
            self._sync_subject_view_combo()
            if self.coreg_panel is not None and hasattr(self.coreg_panel, "_refresh_subject_view_combo"):
                self.coreg_panel._refresh_subject_view_combo()
        except Exception as exc:
            print(f"Failed to switch brain subject: {exc}")
    
    def _run_fastsurfer(self):
        """Launch FastSurfer surface reconstruction process."""
        # Check if FastSurfer is installed
        import subprocess
        import shutil
        
        # Check for Docker installation (recommended method)
        has_docker = shutil.which('docker') is not None
        
        # Check for local FastSurfer installation
        fastsurfer_home = os.environ.get('FASTSURFER_HOME')
        has_local_fastsurfer = fastsurfer_home and os.path.exists(os.path.join(fastsurfer_home, 'run_fastsurfer.sh'))
        
        if not has_docker and not has_local_fastsurfer:
            QtWidgets.QMessageBox.warning(
                None,
                "FastSurfer Not Found",
                "FastSurfer installation not detected.\n\n"
                "Please install FastSurfer using one of these methods:\n\n"
                "1. Docker (Recommended):\n"
                "   docker pull deepmi/fastsurfer:latest\n\n"
                "2. Local Installation:\n"
                "   git clone https://github.com/Deep-MI/FastSurfer.git\n"
                "   Set FASTSURFER_HOME environment variable\n\n"
                "See: https://github.com/Deep-MI/FastSurfer"
            )
            return
        
        # Prompt for output directory
        default_output = str(Path(self.mri_path).parent / "fastsurfer_output")
        output_dir, ok = QtWidgets.QInputDialog.getText(
            None,
            "FastSurfer Output Directory",
            "Enter output directory path:",
            QtWidgets.QLineEdit.Normal,
            default_output
        )
        
        if not ok or not output_dir:
            return
        
        # Prompt for subject ID
        default_subject_id = Path(self.mri_path).stem.replace('.nii', '')
        subject_id, ok = QtWidgets.QInputDialog.getText(
            None,
            "Subject ID",
            "Enter subject ID:",
            QtWidgets.QLineEdit.Normal,
            default_subject_id
        )
        
        if not ok or not subject_id:
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Show progress dialog
        progress_dialog = QtWidgets.QProgressDialog(
            "Running FastSurfer surface reconstruction...\n"
            "This may take 5-20 minutes depending on your hardware.\n\n"
            "Check terminal for detailed progress.",
            "Cancel",
            0, 0,
            None
        )
        progress_dialog.setWindowTitle("FastSurfer Processing")
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.show()
        
        # Construct FastSurfer command
        if has_docker:
            # Docker command
            cmd = [
                'docker', 'run', '--rm',
                '-v', f'{os.path.abspath(os.path.dirname(self.mri_path))}:/input',
                '-v', f'{os.path.abspath(output_dir)}:/output',
                '--user', f'{os.getuid()}:{os.getgid()}',
                'deepmi/fastsurfer:latest',
                '--t1', f'/input/{os.path.basename(self.mri_path)}',
                '--sd', '/output',
                '--sid', subject_id
            ]
            
            # Add GPU flag if available
            try:
                subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', 'hello-world'], 
                             capture_output=True, timeout=5)
                cmd.insert(3, '--gpus')
                cmd.insert(4, 'all')
                print("GPU support detected - using GPU acceleration")
            except:
                print("Running on CPU (slower)")
        else:
            # Local installation command
            cmd = [
                os.path.join(fastsurfer_home, 'run_fastsurfer.sh'),
                '--t1', self.mri_path,
                '--sd', output_dir,
                '--sid', subject_id
            ]
        
        print(f"\nRunning FastSurfer command:")
        print(" ".join(cmd))
        print(f"\nOutput directory: {output_dir}")
        print(f"Subject ID: {subject_id}\n")
        
        # Run FastSurfer in background
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Stream output to console
            for line in process.stdout:
                print(line, end='')
                QtWidgets.QApplication.processEvents()
                
                if progress_dialog.wasCanceled():
                    process.terminate()
                    QtWidgets.QMessageBox.information(
                        None, "Cancelled", "FastSurfer processing was cancelled."
                    )
                    return
            
            process.wait()
            progress_dialog.close()
            
            if process.returncode == 0:
                # Success
                QtWidgets.QMessageBox.information(
                    None,
                    "FastSurfer Complete",
                    f"Surface reconstruction completed successfully!\n\n"
                    f"Output directory: {output_dir}\n"
                    f"Subject: {subject_id}\n\n"
                    "You can now use these surfaces for visualization."
                )
                
                # Update custom subject settings
                self.custom_subjects_dir = output_dir
                self.custom_subject = subject_id
                self.coreg_status_label.setText(f"Status: FastSurfer completed - {subject_id}")
            else:
                QtWidgets.QMessageBox.warning(
                    None,
                    "FastSurfer Error",
                    f"FastSurfer exited with error code {process.returncode}.\n"
                    "Check terminal output for details."
                )
        except Exception as e:
            progress_dialog.close()
            QtWidgets.QMessageBox.critical(
                None,
                "FastSurfer Error",
                f"Error running FastSurfer:\n{str(e)}"
            )
    
    def _show_fastsurfer_info(self):
        """Show detailed information about FastSurfer."""
        info_dialog = QtWidgets.QMessageBox()
        info_dialog.setWindowTitle("About FastSurfer")
        info_dialog.setIcon(QtWidgets.QMessageBox.Information)
        
        info_text = (
            "<h3>FastSurfer - Deep Learning Brain Surface Reconstruction</h3>"
            "<p><b>What is FastSurfer?</b><br>"
            "FastSurfer is a state-of-the-art deep learning tool for automated MRI brain "
            "surface reconstruction and segmentation, developed by MIT and MGH.</p>"
            
            "<p><b>Key Features:</b></p>"
            "<ul>"
            "<li>10-100x faster than FreeSurfer (5-20 minutes vs 6-12 hours)</li>"
            "<li>High quality cortical surface meshes</li>"
            "<li>Automated brain segmentation and parcellation</li>"
            "<li>FreeSurfer-compatible output format</li>"
            "<li>Robust to different MRI qualities and contrasts</li>"
            "<li>Works with or without GPU</li>"
            "</ul>"
            
            "<p><b>Installation Options:</b></p>"
            "<ol>"
            "<li><b>Docker (Recommended):</b><br>"
            "<code>docker pull deepmi/fastsurfer:latest</code></li>"
            "<li><b>Local Installation:</b><br>"
            "Clone from: <a href='https://github.com/Deep-MI/FastSurfer'>"
            "https://github.com/Deep-MI/FastSurfer</a></li>"
            "</ol>"
            
            "<p><b>Requirements:</b></p>"
            "<ul>"
            "<li>T1-weighted MRI in NIfTI format (.nii or .nii.gz)</li>"
            "<li>~8GB RAM minimum</li>"
            "<li>GPU recommended (NVIDIA with CUDA) but not required</li>"
            "<li>5-20 minutes processing time (with GPU)</li>"
            "<li>30-60 minutes processing time (CPU only)</li>"
            "</ul>"
            
            "<p><b>Learn More:</b><br>"
            "GitHub: <a href='https://github.com/Deep-MI/FastSurfer'>Deep-MI/FastSurfer</a><br>"
            "Paper: Henschel et al., NeuroImage 2020</p>"
        )
        
        info_dialog.setText(info_text)
        info_dialog.setTextFormat(QtCore.Qt.RichText)
        info_dialog.exec_()
        
        # Ask again if they want to run it
        reply = QtWidgets.QMessageBox.question(
            None,
            "Run FastSurfer?",
            "Would you like to run FastSurfer now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self._run_fastsurfer()
    
    def update_button_states(self):
        """Enable/disable buttons based on file selection."""
        has_mri = self.mri_path is not None
        has_ct = self.ct_path is not None

        if hasattr(self, "coregister_btn"):
            self.coregister_btn.setEnabled(True)
        if hasattr(self, "launch_coreg_gui_btn"):
            self.launch_coreg_gui_btn.setEnabled(has_mri)
        if hasattr(self, "save_trans_btn"):
            self.save_trans_btn.setEnabled(self.trans is not None)
    
    def _display_patient_volumes(self):
        """Display MRI and/or CT volumes in the 3D viewer."""
        if self.brain is None:
            print("3D viewer not initialized")
            return
        
        try:
            import nibabel as nib
            import pyvista as pv
            
            # Remove any existing overlays
            plotter = self.brain._renderer.plotter
            
            # Remove old MRI slice actors
            for attr_name in ['_mri_axial_id', '_mri_coronal_id', '_mri_sagittal_id', '_mri_overlay_id']:
                if hasattr(self, attr_name):
                    try:
                        actor_id = getattr(self, attr_name)
                        if actor_id is not None:
                            plotter.remove_actor(actor_id)
                    except:
                        pass
                    setattr(self, attr_name, None)
                
            # Remove old CT overlay
            if hasattr(self, '_ct_overlay_id'):
                try:
                    if self._ct_overlay_id is not None:
                        plotter.remove_actor(self._ct_overlay_id)
                except:
                    pass
                self._ct_overlay_id = None
            
            # Display MRI if available
            if self.mri_path:
                print(f"Loading MRI for display: {self.mri_path}")
                try:
                    mri_img = nib.load(self.mri_path)
                    mri_data = mri_img.get_fdata()
                    
                    print(f"  MRI shape: {mri_img.shape}")
                    print(f"  MRI data range: [{mri_data.min():.1f}, {mri_data.max():.1f}]")
                    
                    # Create volume visualization for MRI (grayscale)
                    # Use middle slices for visualization
                    self._add_mri_slices(mri_data, mri_img.affine)
                    
                except Exception as e:
                    print(f"Error loading MRI: {e}")
            
            # Display CT if available  
            if self.ct_path:
                # Use registered CT if available, otherwise use original
                ct_to_display = self.ct_registered_path if hasattr(self, 'ct_registered_path') and self.ct_registered_path else self.ct_path
                
                print(f"Loading CT for display: {ct_to_display}")
                try:
                    ct_img = nib.load(str(ct_to_display))
                    ct_data = ct_img.get_fdata()
                    
                    print(f"  CT shape: {ct_img.shape}")
                    print(f"  CT data range: [{ct_data.min():.1f}, {ct_data.max():.1f}]")
                    
                    # Threshold CT to show only high-intensity regions (electrodes/bone)
                    threshold = ct_data.mean() + 2 * ct_data.std()
                    ct_mask = ct_data > threshold
                    
                    print(f"  CT threshold: {threshold:.1f}")
                    print(f"  Voxels above threshold: {ct_mask.sum()}")
                    
                    # Visualize high-intensity CT points (electrodes)
                    self._add_ct_points(ct_data, ct_img.affine, threshold)
                    
                    if ct_to_display == self.ct_registered_path:
                        status_msg = "registered CT displayed (yellow points)"
                    else:
                        status_msg = "unregistered CT displayed (may not align)"
                    
                    current_status = self.coreg_status_label.text()
                    if "complete" in current_status.lower() or "loaded" in current_status.lower():
                        self.coreg_status_label.setText(f"{current_status} - {status_msg}")
                    
                except Exception as e:
                    print(f"Error loading CT: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Force refresh of the 3D viewer
            if hasattr(self.brain, '_renderer'):
                try:
                    self.brain._renderer.plotter.render()
                except:
                    pass
                    
        except Exception as e:
            print(f"Error displaying patient volumes: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_mri_slices(self, mri_data, affine):
        """Add MRI volume visualization to the 3D viewer using PyVista.
        
        This adds orthogonal slices (axial, coronal, sagittal) to the 3D viewer
        without requiring FreeSurfer's recon-all processing.
        """
        try:
            import pyvista as pv
            
            print("  Adding MRI volume visualization...")
            
            # Normalize MRI data to 0-255 for better visualization
            mri_normalized = mri_data.copy()
            mri_min, mri_max = np.percentile(mri_data, [2, 98])  # Robust normalization
            mri_normalized = np.clip((mri_data - mri_min) / (mri_max - mri_min) * 255, 0, 255)
            
            # Create PyVista ImageData (uniform grid) from the MRI
            # Note: PyVista expects data in (nx, ny, nz) order
            grid = pv.ImageData()
            grid.dimensions = np.array(mri_data.shape) + 1  # cell data requires n+1 points
            
            # Extract spacing and origin from affine matrix
            # Affine format: [[sx*r11, sy*r12, sz*r13, tx],
            #                 [sx*r21, sy*r22, sz*r23, ty],
            #                 [sx*r31, sy*r32, sz*r33, tz],
            #                 [0,      0,      0,      1 ]]
            spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
            origin = affine[:3, 3]
            
            grid.spacing = spacing
            grid.origin = origin
            
            # Add MRI data as cell data
            grid.cell_data["values"] = mri_normalized.flatten(order="F")  # Fortran order for correct orientation
            
            # Get the plotter
            plotter = self.brain._renderer.plotter
            
            # Add orthogonal slices at the center of the volume
            center_idx = np.array(mri_data.shape) // 2
            
            # Axial slice (horizontal, through Z)
            axial_slice = grid.slice(normal='z', origin=grid.center)
            self._mri_axial_id = plotter.add_mesh(
                axial_slice,
                cmap='gray',
                opacity=0.8,
                name='mri_axial',
                show_scalar_bar=False
            )
            
            # Coronal slice (front-back, through Y)  
            coronal_slice = grid.slice(normal='y', origin=grid.center)
            self._mri_coronal_id = plotter.add_mesh(
                coronal_slice,
                cmap='gray',
                opacity=0.7,
                name='mri_coronal',
                show_scalar_bar=False
            )
            
            # Sagittal slice (left-right, through X)
            sagittal_slice = grid.slice(normal='x', origin=grid.center)
            self._mri_sagittal_id = plotter.add_mesh(
                sagittal_slice,
                cmap='gray',
                opacity=0.7,
                name='mri_sagittal',
                show_scalar_bar=False
            )
            
            print(f"  ✓ Added MRI orthogonal slices (center: {center_idx})")
            print(f"    - Axial (Z), Coronal (Y), Sagittal (X) slices")
            print(f"    - Volume shape: {mri_data.shape}, Spacing: {spacing}, Origin: {origin}")
            
        except Exception as e:
            print(f"Error adding MRI slices: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_ct_points(self, ct_data, affine, threshold):
        """Add CT high-intensity points (electrodes) to the 3D viewer."""
        try:
            import pyvista as pv
            
            # Get coordinates of high-intensity voxels
            coords = np.argwhere(ct_data > threshold)
            
            if len(coords) == 0:
                print("  No high-intensity points found in CT")
                return
            
            # Limit to a reasonable number of points for visualization
            if len(coords) > 10000:
                print(f"  Downsampling from {len(coords)} to 10000 points")
                indices = np.random.choice(len(coords), 10000, replace=False)
                coords = coords[indices]
            
            # Convert voxel coordinates to world coordinates
            coords_homogeneous = np.concatenate([coords, np.ones((len(coords), 1))], axis=1)
            world_coords = coords_homogeneous @ affine.T
            world_coords = world_coords[:, :3]
            
            if self.brain_surface in ("inflated", "flat"):
                world_coords = self._map_points_to_display_surface(world_coords)

            # Create PyVista point cloud
            point_cloud = pv.PolyData(world_coords)
            point_cloud['intensity'] = ct_data[coords[:, 0], coords[:, 1], coords[:, 2]]
            
            # Add to the renderer
            plotter = self.brain._renderer.plotter
            self._ct_overlay_id = plotter.add_points(
                point_cloud,
                color='yellow',
                point_size=3,
                opacity=0.6,
                render_points_as_spheres=True
            )
            
            print(f"  Added {len(coords)} CT points to visualization")
            
        except Exception as e:
            print(f"Error adding CT points: {e}")
            import traceback
            traceback.print_exc()

    def _display_detected_electrodes(self, electrodes, electrode_space):
        """Display detected electrode points in the 3D viewer."""
        if not electrodes:
            print("No detected electrodes to display.")
            return
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            print("3D viewer is not initialized; cannot display electrodes.")
            return

        grouped = any(
            e.shaft not in ("Detected", "Unassigned") for e in electrodes
        )
        if grouped:
            self._display_detected_shafts(electrodes, electrode_space)
        else:
            self._display_detected_electrode_points(electrodes, electrode_space)

    @staticmethod
    def _display_space_uses_mni_world(electrode_space):
        """Whether the 3D viewer should use stored MNI-space mm coordinates."""
        space = str(electrode_space or "").strip().lower()
        return space in {"mni", "mni_tal", "mni tal", "mni_talairach"}

    def _prepare_electrode_world_groups(self, electrodes, electrode_space=None):
        """Convert detected electrodes into world-space groups keyed by shaft."""
        use_mni_world = self._display_space_uses_mni_world(electrode_space)
        voxel_to_world = None if use_mni_world else self._get_electrode_voxel_to_world_trans(
            prefer_subject_mri=bool(self.custom_subject and self.custom_subjects_dir) or self.force_fsaverage
        )
        if not use_mni_world and voxel_to_world is None:
            print("No MRI/CT affine available for electrode display.")
            return None

        groups = {}
        missing_mni = 0
        voxel_to_world = None if voxel_to_world is None else np.asarray(voxel_to_world, dtype=float)
        for elec in electrodes:
            visible = elec.get("visible", True) if hasattr(elec, "get") else getattr(elec, "visible", True)
            shaft_visible = (
                elec.get("shaft_visible", True)
                if hasattr(elec, "get")
                else getattr(elec, "shaft_visible", True)
            )
            if not visible:
                continue
            if not shaft_visible:
                continue

            shaft_name = (
                elec.get("shaft", None)
                if hasattr(elec, "get")
                else getattr(elec, "shaft", None)
            ) or "Unassigned"
            if use_mni_world:
                coords = [elec.get("mni_x"), elec.get("mni_y"), elec.get("mni_z")]
                if any(val is None for val in coords):
                    missing_mni += 1
                    continue
                world = np.asarray(coords, dtype=float)
            else:
                x = elec.get("x")
                y = elec.get("y")
                z = elec.get("z")
                if x is None or y is None or z is None:
                    continue
                vox = np.array([float(x), float(y), float(z), 1.0], dtype=float)
                world = (vox @ voxel_to_world.T)[:3]
                if self.force_fsaverage:
                    world = self._transform_world_for_fsaverage(world)[0]

            groups.setdefault(shaft_name, []).append(world)

        if use_mni_world and missing_mni:
            print(f"Skipped {missing_mni} electrodes without stored MNI coordinates.")
        return {
            shaft_name: np.asarray(points, dtype=float)
            for shaft_name, points in groups.items()
            if points
        }

    def transform_detected_electrodes_to_mni(self, electrodes=None, update_display=True):
        """Compute and store MNI Talairach coordinates for detected electrodes."""
        target_electrodes = list(electrodes if electrodes is not None else self.detected_electrodes)
        if not target_electrodes:
            raise RuntimeError("No electrodes are available to transform.")

        voxel_to_subject_mri = self._get_loaded_mri_voxel_to_subject_mri_trans()
        if voxel_to_subject_mri is None:
            raise RuntimeError(
                "Could not derive the loaded MRI voxel -> FreeSurfer MRI transform. "
                "Ensure the loaded MRI matches the FreeSurfer subject."
            )

        talxfm = self._get_subject_to_fsaverage_trans()
        if talxfm is None:
            raise RuntimeError(
                "Load a FreeSurfer/FastSurfer subject with talairach.xfm before transforming to MNI."
            )

        voxel_to_subject_mri = np.asarray(voxel_to_subject_mri, dtype=float)
        count = 0
        for elec in target_electrodes:
            x = elec.get("x")
            y = elec.get("y")
            z = elec.get("z")
            if x is None or y is None or z is None:
                continue

            vox = np.array([float(x), float(y), float(z), 1.0], dtype=float)
            mri_world = (vox @ voxel_to_subject_mri.T)[:3]
            mni_world = mne.transforms.apply_trans(talxfm, mri_world / 1000.0) * 1000.0

            elec["mri_world_x"] = float(mri_world[0])
            elec["mri_world_y"] = float(mri_world[1])
            elec["mri_world_z"] = float(mri_world[2])
            elec["mni_x"] = float(mni_world[0])
            elec["mni_y"] = float(mni_world[1])
            elec["mni_z"] = float(mni_world[2])
            elec["mni_space"] = "mni_tal"
            count += 1

        if count == 0:
            raise RuntimeError("No valid electrode coordinates were available for MRI->MNI conversion.")

        self.detected_electrodes = target_electrodes
        if self.detected_electrode_native_space is None:
            self.detected_electrode_native_space = "registered"
        self.detected_electrode_space = "mni_tal"
        if update_display:
            if self.custom_subject and self.custom_subjects_dir and not self.force_fsaverage:
                self.set_subject_view_mode("fsaverage")
            self._display_detected_electrodes(self.detected_electrodes, self.detected_electrode_space)
        return count

    def _sorted_shaft_names_for_colors(self, electrodes):
        """Return shaft names ordered by numeric suffix (S_#) then alpha."""
        names = sorted({e.shaft for e in electrodes if e.shaft})
        names = [name for name in names if name != "Unassigned"]
        trajectory_orders = {}
        for elec in electrodes:
            shaft_name = getattr(elec, "shaft", None)
            if not shaft_name or shaft_name == "Unassigned":
                continue
            order = elec.get("trajectory_order") if hasattr(elec, "get") else None
            if order is None:
                continue
            current = trajectory_orders.get(shaft_name)
            if current is None or order < current:
                trajectory_orders[shaft_name] = order
        planned = []
        numeric = []
        alpha = []
        for name in names:
            if name in trajectory_orders:
                planned.append((trajectory_orders[name], name))
                continue
            match = re.match(r"^[Ss][-_]?(\d+)$", name)
            if match:
                numeric.append((int(match.group(1)), name))
            else:
                alpha.append(name)
        planned.sort(key=lambda x: (x[0], x[1]))
        numeric.sort(key=lambda x: x[0])
        alpha.sort()
        return [name for _, name in planned] + [name for _, name in numeric] + alpha

    def _shaft_trajectory_label_map(self, electrodes):
        """Return shaft -> worksheet letter label for uniquely labeled shafts."""
        labels_by_shaft = {}
        for elec in electrodes or []:
            shaft_name = getattr(elec, "shaft", None)
            if not shaft_name or shaft_name == "Unassigned":
                continue
            label = elec.get("trajectory_label") if hasattr(elec, "get") else None
            label = str(label or "").strip()
            if not label:
                continue
            labels_by_shaft.setdefault(shaft_name, [])
            if label not in labels_by_shaft[shaft_name]:
                labels_by_shaft[shaft_name].append(label)
        return {
            shaft_name: labels[0]
            for shaft_name, labels in labels_by_shaft.items()
            if len(labels) == 1
        }

    @staticmethod
    def _normalize_label_key(label):
        """Normalize an electrode or trajectory label for comparison."""
        text = str(label or "").strip().lower()
        return re.sub(r"[^a-z0-9]", "", text)

    def _shaft_display_label(self, shaft_name, trajectory_labels=None):
        """Return shaft label text with worksheet letter label appended."""
        if shaft_name == "Unassigned":
            return shaft_name
        trajectory_labels = trajectory_labels or {}
        label = str(trajectory_labels.get(shaft_name) or "").strip()
        if not label:
            return shaft_name
        if self._normalize_label_key(label) in self._normalize_label_key(shaft_name):
            return shaft_name
        return f"{shaft_name} ({label})"

    def _display_detected_electrode_points(self, electrodes, electrode_space=None):
        """Display detected electrode points grouped by shaft color."""
        try:
            import pyvista as pv
            t0 = time.perf_counter() if PROFILE_MERGE else None

            plotter = self.brain._renderer.plotter
            self._clear_detected_electrode_actors(plotter)
            t_clear = time.perf_counter() if PROFILE_MERGE else None
            self._detected_shaft_cache = None

            shaft_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
            shaft_color_map = {"Unassigned": "#777777"}
            colored_names = self._sorted_shaft_names_for_colors(electrodes)
            for idx, name in enumerate(colored_names):
                shaft_color_map[name] = shaft_colors[idx % len(shaft_colors)]

            selected_shaft = getattr(self, "detected_selected_shaft", None)
            coords_by_shaft = self._prepare_electrode_world_groups(electrodes, electrode_space)
            if coords_by_shaft is None:
                return
            t_build = time.perf_counter() if PROFILE_MERGE else None

            total_points = 0
            for shaft_name, world in coords_by_shaft.items():
                if world.size == 0:
                    continue
                if self.brain_surface in ("inflated", "flat"):
                    world = self._map_points_to_display_surface(world)
                point_cloud = pv.PolyData(world)
                color = shaft_color_map.get(shaft_name, "#333333")
                if selected_shaft:
                    point_size = 9 if shaft_name == selected_shaft else 5
                    opacity = 1.0 if shaft_name == selected_shaft else 0.35
                else:
                    point_size = 7
                    opacity = 0.9
                actor = plotter.add_points(
                    point_cloud,
                    color=color,
                    point_size=point_size,
                    opacity=opacity,
                    render_points_as_spheres=True
                )
                self._detected_electrode_actors.append(actor)
                total_points += len(world)
            t_mesh = time.perf_counter() if PROFILE_MERGE else None

            self._render_plotter(plotter)
            space_msg = f" ({electrode_space})" if electrode_space else ""
            print(f"Displayed {total_points} detected electrodes{space_msg} in 3D viewer.")
            if PROFILE_MERGE and t0 is not None:
                parts = []
                if t_clear is not None:
                    parts.append(f"clear={t_clear - t0:.3f}s")
                if t_build is not None:
                    parts.append(f"build={t_build - (t_clear or t0):.3f}s")
                if t_mesh is not None:
                    parts.append(f"mesh={t_mesh - (t_build or t0):.3f}s")
                parts.append(f"total={time.perf_counter() - t0:.3f}s")
                print(f"[profile] display_points {', '.join(parts)}")
        except Exception as e:
            print(f"Error displaying detected electrodes: {e}")
            import traceback
            traceback.print_exc()

    def _detected_shafts_signature(self, coords_by_shaft, affine, use_inflated, trajectory_labels=None):
        """Create a stable signature for the detected shaft geometry and labels."""
        sig_parts = [str(self.brain_surface), bool(use_inflated), bool(self.force_fsaverage)]
        trajectory_labels = trajectory_labels or {}
        if isinstance(affine, str):
            sig_parts.append(affine)
        if self.force_fsaverage:
            trans = self._get_subject_to_fsaverage_trans()
            if trans is not None:
                mat = trans["trans"] if isinstance(trans, dict) and "trans" in trans else getattr(trans, "trans", None)
                if mat is not None:
                    sig_parts.extend(np.round(np.asarray(mat).reshape(-1), 6).tolist())
        if affine is not None and not isinstance(affine, str):
            sig_parts.extend(np.round(np.asarray(affine).reshape(-1), 6).tolist())
        for name in sorted(coords_by_shaft.keys()):
            sig_parts.append(name)
            sig_parts.append(trajectory_labels.get(name, ""))
            coords = np.asarray(coords_by_shaft.get(name, []), dtype=float)
            sig_parts.append(len(coords))
            for x, y, z in coords:
                sig_parts.append(round(float(x), 3))
                sig_parts.append(round(float(y), 3))
                sig_parts.append(round(float(z), 3))
        return tuple(sig_parts)

    def _update_detected_shaft_highlight(self, cache, selected_shaft):
        """Update shaft highlight without rebuilding geometry."""
        actors_by_shaft = cache.get("actors_by_shaft", {})
        for shaft_name, bundle in actors_by_shaft.items():
            is_selected = bool(selected_shaft and shaft_name == selected_shaft)
            dim_others = bool(selected_shaft and shaft_name != selected_shaft)
            tube_opacity = self.shaft_opacity
            contact_opacity = 1.0 if is_selected else (0.35 if dim_others else 0.9)
            label_opacity = 1.0 if (not selected_shaft or is_selected) else 0.35

            tube_actor = bundle.get("tube")
            if tube_actor is not None:
                try:
                    tube_actor.GetProperty().SetOpacity(tube_opacity)
                except Exception:
                    pass
            for contact_actor in bundle.get("contacts", []):
                try:
                    contact_actor.GetProperty().SetOpacity(contact_opacity)
                except Exception:
                    pass
            label_actor = bundle.get("label")
            if label_actor is not None:
                try:
                    label_actor.GetProperty().SetOpacity(label_opacity)
                except Exception:
                    pass

        unassigned = cache.get("unassigned")
        if unassigned:
            if selected_shaft:
                opacity = 0.9 if selected_shaft == "Unassigned" else 0.25
                label_opacity = 1.0 if selected_shaft == "Unassigned" else 0.35
            else:
                opacity = 0.8
                label_opacity = 1.0
            point_actor = unassigned.get("points")
            if point_actor is not None:
                try:
                    point_actor.GetProperty().SetOpacity(opacity)
                except Exception:
                    pass
            label_actor = unassigned.get("label")
            if label_actor is not None:
                try:
                    label_actor.GetProperty().SetOpacity(label_opacity)
                except Exception:
                    pass

    def _display_detected_shafts(self, electrodes, electrode_space=None):
        """Display detected shafts as tubes with cylindrical contacts."""
        try:
            import pyvista as pv
            from matplotlib import colors as mcolors
            t0 = time.perf_counter() if PROFILE_MERGE else None

            coords_by_shaft = self._prepare_electrode_world_groups(electrodes, electrode_space)
            if coords_by_shaft is None:
                return
            affine = None if self._display_space_uses_mni_world(electrode_space) else self._get_affine_for_electrodes()

            plotter = self.brain._renderer.plotter

            shaft_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
            shaft_color_map = {"Unassigned": "#777777"}
            colored_names = self._sorted_shaft_names_for_colors(electrodes)
            for idx, name in enumerate(colored_names):
                shaft_color_map[name] = shaft_colors[idx % len(shaft_colors)]

            selected_shaft = getattr(self, "detected_selected_shaft", None)
            trajectory_labels = self._shaft_trajectory_label_map(electrodes)

            def _mute_color(color, mix=0.6):
                rgb = np.array(mcolors.to_rgb(color))
                muted = rgb * (1.0 - mix) + np.array([0.6, 0.6, 0.6]) * mix
                return tuple(muted.tolist())
            t_build = time.perf_counter() if PROFILE_MERGE else None

            total_contacts = 0
            shaft_radius = 0.4
            contact_radius = 0.8
            default_contact_height = 2.0

            use_inflated = self.brain_surface in ("inflated", "flat")
            signature_affine = "mni_world" if self._display_space_uses_mni_world(electrode_space) else affine
            signature = self._detected_shafts_signature(
                coords_by_shaft,
                signature_affine,
                use_inflated,
                trajectory_labels,
            )
            cache = getattr(self, "_detected_shaft_cache", None)
            if cache and cache.get("signature") == signature:
                self._update_detected_shaft_highlight(cache, selected_shaft)
                self._render_plotter(plotter)
                if PROFILE_MERGE and t0 is not None:
                    parts = []
                    if t_build is not None:
                        parts.append(f"build={t_build - t0:.3f}s")
                    parts.append(f"total={time.perf_counter() - t0:.3f}s")
                    print(f"[profile] display_shafts reuse {', '.join(parts)}")
                return

            self._clear_detected_electrode_actors(plotter)
            t_clear = time.perf_counter() if PROFILE_MERGE else None

            world_center = self._get_reference_world_center()
            if world_center is not None:
                world_center = self._transform_world_for_fsaverage(world_center)[0]
            if use_inflated and world_center is not None:
                try:
                    world_center = self._map_points_to_display_surface(np.asarray([world_center]))[0]
                except Exception:
                    pass
            actors_by_shaft = {}
            for shaft_name, coords in coords_by_shaft.items():
                if coords is None or len(coords) == 0:
                    continue
                if shaft_name == "Unassigned":
                    continue
                try:
                    is_selected = bool(selected_shaft and shaft_name == selected_shaft)
                    dim_others = bool(selected_shaft and shaft_name != selected_shaft)
                    shaft_radius_use = shaft_radius * (1.35 if is_selected else 1.0)
                    contact_radius_use = contact_radius * (1.35 if is_selected else (0.7 if dim_others else 1.0))
                    tube_opacity = self.shaft_opacity
                    contact_opacity = 1.0 if is_selected else (0.35 if dim_others else 0.9)
                    world = np.asarray(coords, dtype=float)
                    if use_inflated:
                        world = self._map_points_to_display_surface(world)

                    center = np.mean(world, axis=0)
                    if len(world) > 1:
                        _, _, vh = np.linalg.svd(world - center, full_matrices=False)
                        axis = vh[0]
                        if np.linalg.norm(axis) == 0:
                            axis = np.array([0.0, 0.0, 1.0])
                    else:
                        axis = np.array([0.0, 0.0, 1.0])
                    axis = axis / max(np.linalg.norm(axis), 1e-6)

                    proj = (world - center) @ axis
                    order = np.argsort(proj)
                    proj_sorted = proj[order]
                    world_sorted = world[order]

                    spacing = default_contact_height
                    if len(proj_sorted) > 1:
                        diffs = np.diff(proj_sorted)
                        diffs = diffs[diffs > 1e-3]
                        if len(diffs) > 0:
                            spacing = float(np.median(diffs))
                    contact_height = max(1.0, min(3.0, spacing * 0.6))

                    line_start = center + axis * np.min(proj_sorted)
                    line_end = center + axis * np.max(proj_sorted)
                    if self.brain_surface != "flat" and not np.allclose(line_start, line_end):
                        line = pv.Line(line_start, line_end)
                        if getattr(line, "n_points", 0) > 0:
                            muted_color = _mute_color(shaft_color_map.get(shaft_name, "#777777"))
                            tube = line.tube(radius=shaft_radius_use)
                            if getattr(tube, "n_points", 0) > 0:
                                shaft_actor = plotter.add_mesh(
                                    tube,
                                    color=muted_color,
                                    opacity=tube_opacity
                                )
                                self._detected_shaft_actors.append(shaft_actor)
                            else:
                                shaft_actor = None
                        else:
                            shaft_actor = None
                    else:
                        shaft_actor = None

                    contact_actors = []

                    contact_color = shaft_color_map.get(shaft_name, "#333333")
                    for point in world_sorted:
                        cyl = pv.Cylinder(
                            center=point,
                            direction=axis,
                            radius=contact_radius_use,
                            height=contact_height
                        )
                        contact_actor = plotter.add_mesh(
                            cyl,
                            color=contact_color,
                            opacity=contact_opacity
                        )
                        self._detected_electrode_actors.append(contact_actor)
                        contact_actors.append(contact_actor)
                        total_contacts += 1

                    label_actor = None
                    try:
                        label_offset = max(2.0, contact_height * 1.5)
                        if world_center is not None:
                            dist_start = np.linalg.norm(line_start - world_center)
                            dist_end = np.linalg.norm(line_end - world_center)
                            if dist_start > dist_end:
                                label_pos = line_start - axis * label_offset
                            else:
                                label_pos = line_end + axis * label_offset
                        else:
                            label_pos = line_end + axis * label_offset
                        label_text = self._shaft_display_label(shaft_name, trajectory_labels)
                        label_actor = plotter.add_point_labels(
                            [label_pos],
                            [label_text],
                            text_color=contact_color,
                            font_size=12,
                            point_size=0,
                            show_points=False,
                            shape=None,
                        )
                        label_actor.SetVisibility(bool(self.mni_cohort_show_labels))
                        self._detected_shaft_label_actors.append(label_actor)
                    except Exception:
                        label_actor = None

                    actors_by_shaft[shaft_name] = {
                        "tube": shaft_actor,
                        "contacts": contact_actors,
                        "label": label_actor,
                    }
                except Exception as exc:
                    print(f"Skipping shaft {shaft_name}: {exc}")
                    continue
            t_mesh = time.perf_counter() if PROFILE_MERGE else None

            # Render any unassigned electrodes as points
            unassigned_bundle = None
            if "Unassigned" in coords_by_shaft:
                world = np.asarray(coords_by_shaft.get("Unassigned", []), dtype=float)
                if world.size > 0:
                    if self.brain_surface in ("inflated", "flat"):
                        world = self._map_points_to_display_surface(world)
                    point_cloud = pv.PolyData(world)
                    if selected_shaft:
                        point_size = 8 if selected_shaft == "Unassigned" else 4
                        opacity = 0.9 if selected_shaft == "Unassigned" else 0.25
                    else:
                        point_size = 6
                        opacity = 0.8
                    actor = plotter.add_points(
                        point_cloud,
                        color="#777777",
                        point_size=point_size,
                        opacity=opacity,
                        render_points_as_spheres=True
                    )
                    self._detected_electrode_actors.append(actor)
                    total_contacts += len(world)
                    unassigned_bundle = {"points": actor, "label": None}
                    try:
                        centroid = np.mean(world, axis=0)
                        label_text = f"Unassigned ({len(world)})"
                        label_actor = plotter.add_point_labels(
                            [centroid],
                            [label_text],
                            text_color="#777777",
                            font_size=12,
                            point_size=0,
                            show_points=False,
                            shape=None,
                        )
                        label_actor.SetVisibility(bool(self.mni_cohort_show_labels))
                        self._detected_shaft_label_actors.append(label_actor)
                        unassigned_bundle["label"] = label_actor
                    except Exception:
                        pass
            t_unassigned = time.perf_counter() if PROFILE_MERGE else None

            self._detected_shaft_cache = {
                "signature": signature,
                "actors_by_shaft": actors_by_shaft,
                "unassigned": unassigned_bundle,
            }
            self._update_detected_shaft_highlight(
                self._detected_shaft_cache,
                getattr(self, "detected_selected_shaft", None),
            )

            self._render_plotter(plotter)
            space_msg = f" ({electrode_space})" if electrode_space else ""
            print(f"Displayed {total_contacts} detected contacts{space_msg} in 3D viewer.")
            if PROFILE_MERGE and t0 is not None:
                parts = []
                if t_clear is not None:
                    parts.append(f"clear={t_clear - t0:.3f}s")
                if t_build is not None:
                    parts.append(f"build={t_build - t0:.3f}s")
                if t_mesh is not None:
                    parts.append(f"mesh={t_mesh - (t_build or t0):.3f}s")
                if t_unassigned is not None:
                    parts.append(f"unassigned={t_unassigned - (t_mesh or t0):.3f}s")
                parts.append(f"total={time.perf_counter() - t0:.3f}s")
                print(f"[profile] display_shafts {', '.join(parts)}")
        except Exception as e:
            print(f"Error displaying detected shafts: {e}")
            import traceback
            traceback.print_exc()

    def _get_affine_for_electrodes(self):
        """Return an affine for mapping voxel electrodes to world coordinates."""
        try:
            import nibabel as nib
            if self.mri_path and os.path.exists(self.mri_path):
                return nib.load(self.mri_path).affine
            if self.ct_registered_path and os.path.exists(self.ct_registered_path):
                return nib.load(str(self.ct_registered_path)).affine
            if self.ct_path and os.path.exists(self.ct_path):
                return nib.load(self.ct_path).affine
        except Exception:
            return None
        return None

    def _get_subject_mri_volume_path(self):
        """Return the FreeSurfer volume used to define subject MRI coordinates."""
        if not (self.custom_subject and self.custom_subjects_dir):
            return None
        subject_mri_dir = Path(self.custom_subjects_dir) / self.custom_subject / "mri"
        for name in ("orig.mgz", "T1.mgz"):
            path = subject_mri_dir / name
            if path.exists():
                return path
        return None

    def _get_loaded_mri_voxel_to_subject_mri_trans(self):
        """Map loaded MRI voxel indices to FreeSurfer subject MRI surface-RAS (mm)."""
        if not (self.mri_path and os.path.exists(self.mri_path)):
            self._loaded_mri_to_subject_mri_trans = None
            self._loaded_mri_to_subject_mri_key = None
            return None
        if not (self.custom_subject and self.custom_subjects_dir):
            self._loaded_mri_to_subject_mri_trans = None
            self._loaded_mri_to_subject_mri_key = None
            return None

        key = (os.path.abspath(self.mri_path), self.custom_subjects_dir, self.custom_subject)
        if key == self._loaded_mri_to_subject_mri_key and self._loaded_mri_to_subject_mri_trans is not None:
            return self._loaded_mri_to_subject_mri_trans

        fs_mri_path = self._get_subject_mri_volume_path()
        self._loaded_mri_to_subject_mri_key = key
        if fs_mri_path is None:
            self._loaded_mri_to_subject_mri_trans = None
            if not self._loaded_mri_to_subject_mri_warned:
                print("No FreeSurfer MRI volume (orig.mgz/T1.mgz) found; cannot align detected voxels to subject MRI coordinates.")
                self._loaded_mri_to_subject_mri_warned = True
            return None

        try:
            import nibabel as nib

            loaded_img = nib.load(self.mri_path)
            fs_img = nib.load(str(fs_mri_path))
            fs_header = fs_img.header

            if not hasattr(fs_header, "get_vox2ras") or not hasattr(fs_header, "get_vox2ras_tkr"):
                raise RuntimeError(f"{fs_mri_path} does not expose FreeSurfer vox2ras transforms.")

            loaded_vox_to_scanner = np.asarray(loaded_img.affine, dtype=float)
            fs_vox_to_scanner = np.asarray(fs_header.get_vox2ras(), dtype=float)
            fs_vox_to_mri = np.asarray(fs_header.get_vox2ras_tkr(), dtype=float)

            self._loaded_mri_to_subject_mri_trans = (
                fs_vox_to_mri @ np.linalg.inv(fs_vox_to_scanner) @ loaded_vox_to_scanner
            )
            return self._loaded_mri_to_subject_mri_trans
        except Exception as exc:
            self._loaded_mri_to_subject_mri_trans = None
            print(f"Failed to derive loaded MRI voxel -> subject MRI transform: {exc}")
            return None

    def _get_electrode_voxel_to_world_trans(self, prefer_subject_mri=False):
        """Return the best available voxel->world transform for detected electrodes."""
        if prefer_subject_mri:
            subject_trans = self._get_loaded_mri_voxel_to_subject_mri_trans()
            if subject_trans is not None:
                return subject_trans
        return self._get_affine_for_electrodes()

    def _get_reference_world_center(self):
        """Return an approximate brain center in world coordinates."""
        try:
            import nibabel as nib
        except Exception:
            return None
        if self.mri_path and os.path.exists(self.mri_path):
            voxel_to_subject_mri = self._get_loaded_mri_voxel_to_subject_mri_trans()
            if voxel_to_subject_mri is not None:
                try:
                    img = nib.load(str(self.mri_path))
                    shape = np.array(img.shape[:3], dtype=float)
                    if shape.size >= 3:
                        center_vox = (shape - 1.0) / 2.0
                        center_world = nib.affines.apply_affine(voxel_to_subject_mri, center_vox)
                        return np.asarray(center_world, dtype=float)
                except Exception:
                    pass
        for path in (self.mri_path, self.ct_registered_path, self.ct_path):
            if not path:
                continue
            if not os.path.exists(path):
                continue
            try:
                img = nib.load(str(path))
                shape = np.array(img.shape[:3], dtype=float)
                if shape.size < 3:
                    continue
                center_vox = (shape - 1.0) / 2.0
                center_world = nib.affines.apply_affine(img.affine, center_vox)
                return np.asarray(center_world, dtype=float)
            except Exception:
                continue
        return None

    def _get_subject_to_fsaverage_trans(self):
        """Return mri->mni_tal transform for the loaded subject if available."""
        if not (self.custom_subject and self.custom_subjects_dir):
            self._subject_to_fsaverage_trans = None
            self._subject_to_fsaverage_key = None
            return None
        key = (self.custom_subjects_dir, self.custom_subject)
        if key == self._subject_to_fsaverage_key and self._subject_to_fsaverage_trans is not None:
            return self._subject_to_fsaverage_trans

        xfm_path = (
            Path(self.custom_subjects_dir)
            / self.custom_subject
            / "mri"
            / "transforms"
            / "talairach.xfm"
        )
        self._subject_to_fsaverage_key = key
        if not xfm_path.exists():
            self._subject_to_fsaverage_trans = None
            if not self._subject_to_fsaverage_warned:
                print(f"No talairach.xfm found at {xfm_path}; cannot map to fsaverage.")
                self._subject_to_fsaverage_warned = True
            return None
        try:
            trans = mne.read_talxfm(self.custom_subject, self.custom_subjects_dir)
        except Exception as exc:
            self._subject_to_fsaverage_trans = None
            print(f"Failed to read talairach.xfm: {exc}")
            return None
        self._subject_to_fsaverage_trans = trans
        return trans

    def _transform_world_for_fsaverage(self, world):
        """Transform world coordinates from subject MRI -> fsaverage (mni_tal)."""
        if not self.force_fsaverage:
            return world
        trans = self._get_subject_to_fsaverage_trans()
        if trans is None:
            return world
        world = np.asarray(world, dtype=float)
        if world.ndim == 1:
            world = world[None, :]
        return mne.transforms.apply_trans(trans, world / 1000.0) * 1000.0

    def _clear_detected_electrode_actors(self, plotter):
        """Remove any previously drawn detected electrode actors."""
        scalar_bar_title = getattr(self, "_mni_cohort_scalar_bar_title", None)
        if scalar_bar_title:
            try:
                plotter.remove_scalar_bar(scalar_bar_title, render=False)
            except (KeyError, ValueError):
                pass
            self._mni_cohort_scalar_bar_title = None
        for actor in getattr(self, "_detected_electrode_actors", []):
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        for actor in getattr(self, "_detected_shaft_actors", []):
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        for actor in getattr(self, "_detected_shaft_label_actors", []):
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        self._detected_electrode_actors = []
        self._detected_shaft_actors = []
        self._detected_shaft_label_actors = []
        self._detected_shaft_cache = None

    def clear_detected_electrodes(self):
        """Clear detected electrodes from the 3D viewer."""
        self.detected_electrodes = []
        self.detected_electrode_space = None
        self.detected_electrode_native_space = None
        self.detected_selected_shaft = None
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            return
        plotter = self.brain._renderer.plotter
        self._clear_detected_electrode_actors(plotter)
        self._render_plotter(plotter)

    def _render_plotter(self, plotter):
        """Safely render the plotter if possible."""
        try:
            plotter.render()
        except Exception:
            pass

    def _ensure_orientation_widget(self):
        """Ensure the 3D plotter shows an orientation widget."""
        brain = getattr(self, "brain", None)
        renderer = getattr(brain, "_renderer", None) if brain else None
        plotter = getattr(renderer, "plotter", None) if renderer else None
        if plotter is None:
            return
        if (
            getattr(self, "_orientation_widget_plotter", None) is plotter
            and getattr(self, "_orientation_widget", None) is not None
        ):
            return
        self._orientation_widget = None
        self._orientation_widget_plotter = plotter
        try:
            if hasattr(plotter, "add_orientation_widget"):
                actor = None
                try:
                    import pyvista as pv
                    for cls_name in ("AxesActor", "Axes", "AxesAssembly"):
                        cls = getattr(pv, cls_name, None)
                        if cls is not None:
                            actor = cls()
                            break
                except Exception:
                    actor = None
                if actor is not None:
                    self._apply_orientation_labels(actor)
                    try:
                        self._orientation_widget = plotter.add_orientation_widget(
                            actor, viewport=(0.82, 0.02, 0.98, 0.18)
                        )
                    except TypeError:
                        self._orientation_widget = plotter.add_orientation_widget(actor)
                else:
                    try:
                        self._orientation_widget = plotter.add_orientation_widget(
                            viewport=(0.82, 0.02, 0.98, 0.18)
                        )
                    except TypeError:
                        self._orientation_widget = plotter.add_orientation_widget()
            elif hasattr(plotter, "add_axes"):
                try:
                    self._orientation_widget = plotter.add_axes(
                        interactive=False,
                        viewport=(0.82, 0.02, 0.98, 0.18),
                    )
                except TypeError:
                    self._orientation_widget = plotter.add_axes(interactive=False)
        except Exception as exc:
            print(f"Warning: could not add orientation widget: {exc}")
            self._orientation_widget = None

    def _apply_orientation_labels(self, actor):
        """Set RAS labels on an orientation axes actor when supported."""
        label_map = {"X": "R", "Y": "A", "Z": "S"}
        for axis, label in label_map.items():
            applied = False
            for method in (
                f"Set{axis}AxisLabelText",
                f"Set{axis}AxisLabel",
                f"Set{axis}LabelText",
                f"Set{axis}Label",
            ):
                setter = getattr(actor, method, None)
                if callable(setter):
                    try:
                        setter(label)
                        applied = True
                        break
                    except Exception:
                        pass
            if applied:
                continue
            for attr in (
                f"{axis.lower()}_label",
                f"{axis.lower()}_axis_label",
                f"{axis.lower()}_axis_label_text",
            ):
                if hasattr(actor, attr):
                    try:
                        setattr(actor, attr, label)
                        applied = True
                        break
                    except Exception:
                        pass
    
    def perform_coregistration(self):
        """Launch registration dialog with before/after visualization and progress tracking."""
        if not self.coreg_panel:
            return
        self.coreg_panel.setVisible(True)
        self._sync_coreg_toggle_button()
        if not self.mri_path or not self.ct_path:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText("Status: Load MRI and CT in Coregistration UI")

    def _apply_registration_results(self, dialog):
        """Apply registration dialog outputs and prompt for FreeSurfer subject."""
        results = dialog.get_results()
        self.trans = results['trans']
        self.ct_registered_path = results['ct_registered_path']
        self.detected_electrodes = results.get('electrodes', [])
        self.detected_electrode_space = results.get('electrode_space')
        self.detected_electrode_native_space = results.get('electrode_native_space')
        if self.detected_electrode_space is None:
            self.detected_electrode_space = self.detected_electrode_native_space
        self._update_classify_button_state()

        trans_mat = None
        if isinstance(self.trans, dict) and 'trans' in self.trans:
            trans_mat = np.array(self.trans['trans'])
        elif self.trans is not None:
            trans_mat = np.array(self.trans)
        else:
            trans_mat = np.eye(4)
            self.trans = {'trans': trans_mat}

        if not self.ct_registered_path:
            self.ct_registered_path = self.ct_path

        ct_label = None
        if self.ct_registered_path:
            ct_label = os.path.basename(self.ct_registered_path)
        elif self.ct_path:
            ct_label = os.path.basename(self.ct_path)
        else:
            ct_label = "CT"

        # Check if registration was skipped (identity transform)
        is_identity = np.allclose(trans_mat, np.eye(4))

        # Update UI
        if is_identity:
            status_msg = f"Status: CT ready (no registration) - {ct_label}"
            print("✓ Using CT as-is (already aligned)")
        else:
            status_msg = f"Status: Registration complete! Registered CT: {ct_label}"
            print("✓ Registration successful!")

        if hasattr(self, "coreg_status_label"):
            self.coreg_status_label.setText(status_msg)
        self.update_button_states()

        print(f"  - CT file: {self.ct_registered_path}")
        print("  - Transform saved to memory")

        self._browse_subjects_directory()
        self._display_detected_electrodes(self.detected_electrodes, self.detected_electrode_space)
    
    def _try_freesurfer_registration(self):
        """Attempt coregistration using FreeSurfer mri_robust_register."""
        try:
            import subprocess
            from pathlib import Path
            
            # Check if FreeSurfer is available
            try:
                result = subprocess.run(
                    ["mri_robust_register", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    raise FileNotFoundError("FreeSurfer not found")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print("FreeSurfer not found in PATH")
                return False
            
            self.coreg_status_label.setText("Status: Running FreeSurfer mri_robust_register (this may take a few minutes)...")
            QtWidgets.QApplication.processEvents()
            
            # Create output directory for coregistration results
            if self.patient_reg_folder:
                output_dir = Path(self.patient_reg_folder)
            else:
                output_dir = Path(self.mri_path).parent / "coregistration"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Output files
            ct_registered_path = output_dir / "ct_registered_to_mri.nii.gz"
            lta_transform_path = output_dir / "ct_to_mri.lta"
            
            # Run FreeSurfer's mri_robust_register for CT-to-MRI registration
            cmd = [
                "mri_robust_register",
                "--mov", str(self.ct_path),
                "--dst", str(self.mri_path),
                "--lta", str(lta_transform_path),
                "--mapmov", str(ct_registered_path),
                "--satit",
                "--iscale",
                "--cost", "nmi"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"FreeSurfer registration failed:\n{result.stderr}")
                return False
            
            print("FreeSurfer registration output:")
            print(result.stdout)
            
            # Parse the LTA file
            self.lta_transform_path = lta_transform_path
            self.ct_registered_path = ct_registered_path
            
            affine_matrix = self._parse_lta_transform(lta_transform_path)
            
            if affine_matrix is not None:
                self.trans = mne.transforms.Transform(fro="head", to="mri", trans=affine_matrix)
                self.coreg_status_label.setText(
                    f"Status: FreeSurfer registration complete! CT saved to {ct_registered_path.name}"
                )
                self.update_button_states()
                print(f"  - Registered CT: {ct_registered_path}")
                print(f"  - Transform: {lta_transform_path}")
                
                # Display registered CT in 3D viewer
                self._display_patient_volumes()
                
                return True
            
            return False
            
        except subprocess.TimeoutExpired:
            print("FreeSurfer registration timed out")
            return False
        except Exception as e:
            print(f"FreeSurfer registration error: {e}")
            return False
    
    def _try_simpleitk_registration(self):
        """Attempt coregistration using SimpleITK (pure Python)."""
        try:
            import SimpleITK as sitk
            import nibabel as nib
            from pathlib import Path
            
            print("Attempting registration with SimpleITK...")
            self.coreg_status_label.setText("Status: Running SimpleITK registration (this may take a few minutes)...")
            QtWidgets.QApplication.processEvents()
            
            # Load images
            fixed_image = sitk.ReadImage(str(self.mri_path))
            moving_image = sitk.ReadImage(str(self.ct_path))
            
            # Initialize registration
            registration_method = sitk.ImageRegistrationMethod()
            
            # Similarity metric - Mutual Information for CT-MRI
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.01)
            
            # Optimizer
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=1.0,
                numberOfIterations=100,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            
            # Setup for multi-resolution framework
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            
            # Initial transform - rigid + scale (affine would be even better)
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
            
            # Interpolator
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # Execute registration
            print("  SimpleITK: Starting registration...")
            final_transform = registration_method.Execute(fixed_image, moving_image)
            
            print(f"  SimpleITK: Registration complete!")
            print(f"  Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
            print(f"  Final metric value: {registration_method.GetMetricValue()}")
            
            # Apply transform to moving image
            resampled_image = sitk.Resample(
                moving_image,
                fixed_image,
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_image.GetPixelID()
            )
            
            # Save registered CT
            if self.patient_reg_folder:
                output_dir = Path(self.patient_reg_folder)
            else:
                output_dir = Path(self.mri_path).parent / "coregistration"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            ct_registered_path = output_dir / "ct_registered_to_mri_sitk.nii.gz"
            sitk.WriteImage(resampled_image, str(ct_registered_path))
            
            # Convert SimpleITK transform to numpy affine matrix
            affine_matrix = self._simpleitk_to_affine(final_transform, fixed_image, moving_image)
            
            if affine_matrix is not None:
                self.trans = mne.transforms.Transform(fro="head", to="mri", trans=affine_matrix)
                self.ct_registered_path = ct_registered_path
                
                self.coreg_status_label.setText(
                    f"Status: SimpleITK registration complete! CT saved to {ct_registered_path.name}"
                )
                self.update_button_states()
                print(f"  - Registered CT: {ct_registered_path}")
                
                # Display registered CT in 3D viewer
                self._display_patient_volumes()
                
                return True
            
            return False
            
        except ImportError:
            print("SimpleITK not available. Install with: pip install SimpleITK")
            return False
        except Exception as e:
            print(f"SimpleITK registration error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _try_dipy_registration(self):
        """Attempt coregistration using DIPY (pure Python fallback)."""
        try:
            from dipy.align.imaffine import AffineRegistration, MutualInformationMetric, AffineMap
            from dipy.align.transforms import RigidTransform3D, AffineTransform3D
            import nibabel as nib
            from pathlib import Path
            
            print("Attempting registration with DIPY...")
            self.coreg_status_label.setText("Status: Running DIPY registration (this may take a few minutes)...")
            QtWidgets.QApplication.processEvents()
            
            # Load images
            mri_img = nib.load(self.mri_path)
            ct_img = nib.load(self.ct_path)
            
            mri_data = mri_img.get_fdata()
            ct_data = ct_img.get_fdata()
            
            mri_affine = mri_img.affine
            ct_affine = ct_img.affine
            
            # Setup registration
            metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
            
            # Multi-resolution registration
            affreg = AffineRegistration(
                metric=metric,
                level_iters=[10000, 1000, 100],
                sigmas=[3.0, 1.0, 0.0],
                factors=[4, 2, 1]
            )
            
            print("  DIPY: Starting rigid registration...")
            # Start with rigid registration
            rigid = RigidTransform3D()
            rigid_map = affreg.optimize(
                mri_data, ct_data,
                rigid, None,
                mri_affine, ct_affine
            )
            
            print("  DIPY: Starting affine registration...")
            # Refine with affine
            affine = AffineTransform3D()
            affine_map = affreg.optimize(
                mri_data, ct_data,
                affine, None,
                mri_affine, ct_affine,
                starting_affine=rigid_map.affine
            )
            
            print("  DIPY: Registration complete!")
            
            # Apply transform
            transformed_ct = affine_map.transform(ct_data)
            
            # Save registered CT
            if self.patient_reg_folder:
                output_dir = Path(self.patient_reg_folder)
            else:
                output_dir = Path(self.mri_path).parent / "coregistration"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            ct_registered_path = output_dir / "ct_registered_to_mri_dipy.nii.gz"
            
            # Create new NIfTI with registered data
            registered_img = nib.Nifti1Image(transformed_ct, mri_affine)
            nib.save(registered_img, str(ct_registered_path))
            
            # Get affine matrix
            affine_matrix = affine_map.affine
            
            # Store transform
            self.trans = mne.transforms.Transform(fro="head", to="mri", trans=affine_matrix)
            self.ct_registered_path = ct_registered_path
            
            self.coreg_status_label.setText(
                f"Status: DIPY registration complete! CT saved to {ct_registered_path.name}"
            )
            self.update_button_states()
            print(f"  - Registered CT: {ct_registered_path}")
            
            # Display registered CT in 3D viewer
            self._display_patient_volumes()
            
            return True
            
        except ImportError:
            print("DIPY not available. Install with: pip install dipy")
            return False
        except Exception as e:
            print(f"DIPY registration error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _simpleitk_to_affine(self, sitk_transform, fixed_image, moving_image):
        """Convert SimpleITK transform to 4x4 affine matrix."""
        try:
            import SimpleITK as sitk
            import numpy as np
            
            # Get the transform parameters
            if isinstance(sitk_transform, sitk.Euler3DTransform):
                # Extract rotation and translation
                matrix = np.array(sitk_transform.GetMatrix()).reshape(3, 3)
                translation = np.array(sitk_transform.GetTranslation())
                
                # Build 4x4 affine
                affine = np.eye(4)
                affine[:3, :3] = matrix
                affine[:3, 3] = translation
                
                print(f"  SimpleITK transform matrix:\n{affine}")
                return affine
            else:
                # For other transform types, try to get affine directly
                print("  Warning: Non-Euler transform, attempting conversion")
                # This is a simplified conversion
                return np.eye(4)
                
        except Exception as e:
            print(f"Error converting SimpleITK transform: {e}")
            return None
    
    def _parse_lta_transform(self, lta_path):
        """Parse FreeSurfer LTA file and extract affine transformation matrix."""
        try:
            with open(lta_path, 'r') as f:
                lines = f.readlines()
            
            # Find the transformation matrix in the LTA file
            # LTA format has the matrix after "1 4 4" line
            matrix = []
            in_matrix = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line == "1 4 4":  # Start of matrix section
                    in_matrix = True
                    continue
                
                if in_matrix and len(matrix) < 4:
                    # Parse matrix rows
                    values = line.split()
                    if len(values) >= 4:
                        matrix.append([float(v) for v in values[:4]])
                    
                    if len(matrix) == 4:
                        break
            
            if len(matrix) == 4:
                affine = np.array(matrix)
                print(f"Parsed LTA transform:\n{affine}")
                return affine
            else:
                print(f"Error: Could not parse transformation matrix from {lta_path}")
                return None
                
        except Exception as e:
            print(f"Error parsing LTA file: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_transformation(self):
        """Save the current registration transformation matrix."""
        if self.trans is None:
            self.coreg_status_label.setText("Status: Error - No transformation to save")
            return

        try:
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                None,
                "Save Transformation Matrix",
                "trans.fif",
                "FIF Files (*.fif);;All Files (*.*)",
            )
            if save_path:
                mne.write_trans(save_path, self.trans)
                self.coreg_status_label.setText(
                    f"Status: Transformation saved to {os.path.basename(save_path)}"
                )
        except Exception as exc:
            self.coreg_status_label.setText(f"Status: Error saving - {exc}")
            print(f"Save error: {exc}")

    def load_transformation(self, trans_path):
        """Load a previously saved registration transformation matrix."""
        try:
            self.trans = mne.read_trans(trans_path)
            self.coreg_status_label.setText(
                f"Status: Transformation loaded from {os.path.basename(trans_path)}"
            )
            self.update_button_states()
            return True
        except Exception as exc:
            self.coreg_status_label.setText(
                f"Status: Error loading transformation - {exc}"
            )
            print(f"Load transformation error: {exc}")
            return False

    def _label_parcellation_is_surface(self, parcellation=None):
        parcellation = normalize_parcellation_key(parcellation or self.label_parcellation)
        return parcellation in {"aparc", *HCP_MMP_PARCS}

    def _clear_label_overlays(self, clear_annotations=True):
        """Remove existing anatomical label overlays from the Brain scene."""
        if not hasattr(self, "brain") or self.brain is None:
            return
        try:
            self.brain.remove_volume_labels()
        except Exception:
            pass
        try:
            self.brain.remove_labels()
        except Exception:
            pass
        if clear_annotations:
            try:
                self.brain.remove_annotations()
            except Exception:
                pass

    def _ensure_label_parcellation_subject(self, parcellation):
        """Ensure the active Brain subject can display the requested atlas."""
        parcellation = normalize_parcellation_key(parcellation)
        if parcellation not in HCP_MMP_PARCS:
            return True
        subject_id, _subjects_dir = self._get_brain_subject_info()
        if str(subject_id) == "fsaverage":
            return True
        self.force_fsaverage = True
        self._sync_subject_view_combo()
        self._reload_brain_with_subject()
        if hasattr(self, "coreg_status_label"):
            self.coreg_status_label.setText(
                "Status: HCP-MMP is an fsaverage annotation; switched viewer to fsaverage"
            )
        return False

    def _load_volume_label_names_for_current_subject(self):
        """Load available aparc+aseg labels for the active subject."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        fname_aseg = Path(subjects_dir) / subject_id / "mri" / "aparc+aseg.mgz"
        if not fname_aseg.exists():
            alt = Path(subjects_dir) / subject_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
            fname_aseg = alt if alt.exists() else fname_aseg
        if not fname_aseg.exists():
            self.label_names = []
            self.active_labels = []
            return False
        self._aseg_path = str(fname_aseg)
        self._aseg_name = aseg_name_from_path(fname_aseg)
        self.label_names = mne.get_volume_labels_from_aseg(str(fname_aseg))
        self._surface_labels_by_name = {}
        self._surface_label_parc = None
        return True

    def _load_surface_label_names_for_current_subject(self, parcellation):
        """Load surface label names for aparc or HCP-MMP annotations."""
        parcellation = normalize_parcellation_key(parcellation)
        subject_id, subjects_dir = self._get_brain_subject_info()
        if parcellation in HCP_MMP_PARCS:
            subject_id = "fsaverage"
        labels = read_surface_labels(subject_id, subjects_dir, parcellation)
        labels_by_name = {
            str(getattr(label, "name", "") or ""): label
            for label in labels
            if str(getattr(label, "name", "") or "")
        }
        self.label_names = sorted(
            name for name in labels_by_name
            if "unknown" not in name.lower()
        )
        self.active_labels = [label for label in self.active_labels if label in labels_by_name]
        self._surface_labels_by_name = labels_by_name
        self._surface_label_parc = parcellation
        self._surface_label_subject = subject_id
        self._surface_label_subjects_dir = str(subjects_dir)
        return True

    def _prepare_label_names_for_current_parcellation(self):
        """Refresh label names for the active label atlas."""
        parcellation = normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg"))
        self.label_parcellation = parcellation
        if self._label_parcellation_is_surface(parcellation):
            return self._load_surface_label_names_for_current_subject(parcellation)
        return self._load_volume_label_names_for_current_subject()

    def _label_combo_for_name(self, label):
        """Choose the label combo bucket for a label name."""
        text = str(label or "")
        lower = text.lower()
        if lower.startswith("left"):
            return self.cmbLeft
        if lower.startswith("right"):
            return self.cmbRight
        if (
            lower.startswith("ctx-lh")
            or lower.endswith("-lh")
            or lower.startswith(("lh.", "lh_", "lh-", "l_", "left_", "left-"))
        ):
            return self.cmbCortexLH
        if (
            lower.startswith("ctx-rh")
            or lower.endswith("-rh")
            or lower.startswith(("rh.", "rh_", "rh-", "r_", "right_", "right-"))
        ):
            return self.cmbCortexRH
        if lower.startswith("cc"):
            return self.cmbCC
        return self.cmbOthers

    def _populate_label_selection_combos(self):
        """Populate label selection combos from `self.label_names`."""
        combos = [self.cmbLeft, self.cmbRight, self.cmbCortexLH, self.cmbCortexRH, self.cmbCC, self.cmbOthers]
        for cmb in combos:
            cmb.clear()
        for label in self.label_names:
            if "unknown" in str(label).lower():
                continue
            cmb = self._label_combo_for_name(label)
            cmb.addItem(label)
            item = cmb.model().item(cmb.model().rowCount() - 1)
            if item is not None:
                item.setCheckState(QtCore.Qt.Checked if label in self.active_labels else QtCore.Qt.Unchecked)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)

    def _render_active_labels(self):
        """Render active labels for the current atlas."""
        if not hasattr(self, "brain") or self.brain is None:
            return
        parcellation = normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg"))
        if self._label_parcellation_is_surface(parcellation):
            self._clear_label_overlays(clear_annotations=False)
            try:
                self.brain.add_annotation(
                    parcellation,
                    borders=2,
                    alpha=0.35,
                    remove_existing=True,
                )
            except Exception as exc:
                print(f"Could not add surface annotation {parcellation}: {exc}")
            labels_by_name = getattr(self, "_surface_labels_by_name", {}) or {}
            for label_name in self.active_labels:
                label = labels_by_name.get(label_name)
                if label is None:
                    continue
                try:
                    self.brain.add_label(label, alpha=0.75, borders=False)
                except Exception as exc:
                    print(f"Could not add surface label {label_name}: {exc}")
        else:
            self._clear_label_overlays(clear_annotations=True)
            if self.active_labels:
                try:
                    self.brain.add_volume_labels(
                        aseg=getattr(self, "_aseg_name", "aparc+aseg"),
                        labels=self.active_labels,
                        legend=dict(bcolor=None),
                    )
                except Exception as exc:
                    print(f"Could not add volume labels: {exc}")
        try:
            self.brain._renderer.plotter.update()
        except Exception:
            pass

    def _set_label_parcellation(self, parcellation):
        """Switch anatomical label atlas in the right-side label panel."""
        parcellation = normalize_parcellation_key(parcellation)
        self.label_parcellation = parcellation
        self.active_labels = []
        if not self._ensure_label_parcellation_subject(parcellation):
            return
        try:
            self._prepare_label_names_for_current_parcellation()
            self._clear_label_overlays(clear_annotations=True)
            if self._label_parcellation_is_surface(parcellation):
                try:
                    self.brain.add_annotation(parcellation, borders=2, alpha=0.35, remove_existing=True)
                except Exception as exc:
                    print(f"Could not add surface annotation {parcellation}: {exc}")
            self._populate_label_selection_combos()
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: Label atlas set to {parcellation}")
        except Exception as exc:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: Label atlas error - {exc}")
            print(f"Label parcellation error: {exc}")

    def _on_label_parcellation_changed(self, _index):
        combo = getattr(self, "label_parcellation_combo", None)
        if combo is None:
            return
        self._set_label_parcellation(combo.currentData() or "aparc+aseg")

    def get_label_chart(self):
        # Add a checkbox for each label in vertical layout and create a widget
        self.labels_widget = QtWidgets.QWidget()
        vLyt = QtWidgets.QVBoxLayout()
        self.labels_widget.setLayout(vLyt)

        # Add file browser interface at the top
        file_browser = self.create_file_browser_interface()
        vLyt.addWidget(file_browser)
        
        # Add a separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        vLyt.addWidget(separator)

        parc_row = QtWidgets.QHBoxLayout()
        parc_row.setContentsMargins(0, 0, 0, 0)
        parc_row.setSpacing(6)
        parc_label = QtWidgets.QLabel("Parcellation:")
        parc_label.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; color: #555555; }")
        parc_row.addWidget(parc_label)
        self.label_parcellation_combo = QtWidgets.QComboBox()
        self.label_parcellation_combo.addItem("aparc+aseg", "aparc+aseg")
        self.label_parcellation_combo.addItem("aparc", "aparc")
        self.label_parcellation_combo.addItem("HCP-MMP", "HCPMMP1")
        self.label_parcellation_combo.addItem("HCP-MMP combined", "HCPMMP1_combined")
        current_parc = normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg"))
        idx = self.label_parcellation_combo.findData(current_parc)
        if idx >= 0:
            self.label_parcellation_combo.setCurrentIndex(idx)
        self.label_parcellation_combo.setToolTip("Switch anatomical label atlas")
        self.label_parcellation_combo.currentIndexChanged.connect(self._on_label_parcellation_changed)
        parc_row.addWidget(self.label_parcellation_combo)
        parc_row.addStretch()
        vLyt.addLayout(parc_row)
        
        # Label selection title
        label_title = QtWidgets.QLabel("Brain Region Labels")
        label_title.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; padding: 5px; }")
        vLyt.addWidget(label_title)
        
        # Add combo boxes for volume, cortex, corpus-callosum, and fallback labels.
        combo_grid = QtWidgets.QGridLayout()
        combo_grid.setContentsMargins(0, 0, 0, 0)
        combo_grid.setHorizontalSpacing(4)
        combo_grid.setVerticalSpacing(2)
        self.cmbLeft     = CheckableComboBox()
        self.cmbRight    = CheckableComboBox()
        self.cmbCortexLH = CheckableComboBox()
        self.cmbCortexRH = CheckableComboBox()
        self.cmbCC       = CheckableComboBox()
        self.cmbOthers   = CheckableComboBox()
        label_combo_specs = [
            ("Left vol", self.cmbLeft, "Left aparc+aseg volume labels"),
            ("Right vol", self.cmbRight, "Right aparc+aseg volume labels"),
            ("LH cortex", self.cmbCortexLH, "Left-hemisphere surface labels, including HCP-MMP"),
            ("RH cortex", self.cmbCortexRH, "Right-hemisphere surface labels, including HCP-MMP"),
            ("CC", self.cmbCC, "Corpus-callosum labels"),
            ("Other", self.cmbOthers, "Labels that do not match the standard atlas naming patterns"),
        ]
        for col, (title, combo, tooltip) in enumerate(label_combo_specs):
            combo_label = QtWidgets.QLabel(title)
            combo_label.setStyleSheet("QLabel { font-size: 10px; color: #555555; }")
            combo_label.setAlignment(QtCore.Qt.AlignCenter)
            combo.setToolTip(tooltip)
            combo_grid.addWidget(combo_label, 0, col)
            combo_grid.addWidget(combo, 1, col)

        # clear button
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_labels)
        combo_grid.addWidget(self.clear_button, 1, len(label_combo_specs))
        vLyt.addLayout(combo_grid)

        self.cmbLeft.view().pressed.connect(self.show_hide_labels)
        self.cmbRight.view().pressed.connect(self.show_hide_labels)
        self.cmbCortexLH.view().pressed.connect(self.show_hide_labels)
        self.cmbCortexRH.view().pressed.connect(self.show_hide_labels)
        self.cmbCC.view().pressed.connect(self.show_hide_labels)
        self.cmbOthers.view().pressed.connect(self.show_hide_labels)

        self.label_checkboxes = {}
        try:
            self._prepare_label_names_for_current_parcellation()
        except Exception as exc:
            print(f"Could not prepare labels for {getattr(self, 'label_parcellation', 'aparc+aseg')}: {exc}")
        self._populate_label_selection_combos()
        self._render_active_labels()

        return self.labels_widget

    def clear_labels(self):
        for cmb in [self.cmbLeft, self.cmbRight, self.cmbCortexLH, self.cmbCortexRH, self.cmbCC, self.cmbOthers]:
            for i in range(cmb.model().rowCount()):
                cmb.model().item(i).setCheckState(QtCore.Qt.Unchecked)
        self.active_labels = []

        if hasattr(self, 'brain'):
            parcellation = normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg"))
            if self._label_parcellation_is_surface(parcellation):
                try:
                    self.brain.remove_labels()
                except Exception:
                    pass
                try:
                    self.brain.add_annotation(parcellation, borders=2, alpha=0.35, remove_existing=True)
                except Exception:
                    pass
            else:
                self._clear_label_overlays(clear_annotations=True)
            try:
                self.brain._renderer.plotter.update()
            except Exception:
                pass
    
    def show_hide_labels(self, index):

        # Get combobox from QModelIndex
        cmb = index.model().parent()

        # getting the item
        item = cmb.model().itemFromIndex(index)

        # checking if item is checked
        if item.checkState() == QtCore.Qt.Checked:

            # making it unchecked
            item.setCheckState(QtCore.Qt.Unchecked)

            if item.text() in self.active_labels:
                self.active_labels.remove(item.text())

        # if not checked
        else:
            # making the item checked
            item.setCheckState(QtCore.Qt.Checked)

            if item.text() not in self.active_labels:
                self.active_labels.append(item.text())


        if hasattr(self, 'brain'):
            self._render_active_labels()

    def show_bundle_vol(self, bundle):
        legend_kwargs = dict(bcolor=None)
        if not hasattr(self, 'brain'):
            return

        # Remove any existing labels
        self._clear_label_overlays(clear_annotations=not self._label_parcellation_is_surface())
        if bundle is not None:
            if self._label_parcellation_is_surface():
                label = (getattr(self, "_surface_labels_by_name", {}) or {}).get(bundle)
                if label is not None:
                    self.brain.add_label(label, alpha=0.75, borders=False)
            else:
                self.brain.add_volume_labels(
                    aseg=getattr(self, "_aseg_name", "aparc+aseg"),
                    labels=[bundle],
                    legend=legend_kwargs,
                )
            try:
                self.brain._renderer.plotter.update()
            except Exception:
                pass

    def update_patient_mri_sensors(self, mri_path):
        pass
        
    #endregion 3D viewer
    
class SEEGLocalizerMainWindow(QtWidgets.QMainWindow):
    def __init__(self, patient_reg_folder=None):
        super(SEEGLocalizerMainWindow, self).__init__()
        self.patient_reg_folder = patient_reg_folder
        self._init_menus()
        self.initUI()

    def _init_menus(self):
        help_menu = self.menuBar().addMenu("&Help")
        self.about_action = QtWidgets.QAction("About sEEG Localizer", self)
        self.about_action.setMenuRole(QtWidgets.QAction.AboutRole)
        self.about_action.triggered.connect(self._show_about)
        help_menu.addAction(self.about_action)

    def _show_about(self):
        logo_path = Path(__file__).resolve().parent / "assets" / "wired-brains-logo.png"
        logo_url = QtCore.QUrl.fromLocalFile(str(logo_path)).toString()
        QtWidgets.QMessageBox.about(
            self,
            "About sEEG Localizer",
            f"<h3>sEEG Localizer</h3><p>Version {_application_version()}</p>"
            "<p>CT–MRI coregistration, electrode localization, and 3D visualization.</p>"
            "<p>Copyright © 2026<br>Sunil Mathew, PhD</p>"
            f'<p><img src="{logo_url}" width="280" height="193" '
            'alt="Wired Brains Lab"></p>'
            '<p><a href="https://wiredbrains.org">wiredbrains.org</a></p>'
            "<p>Licensed under the MIT License. Use, modification, and redistribution "
            "are permitted with the copyright and license notice retained.</p>"
            "<p>Built with MNE-Python, PyVista, and Qt.</p>",
        )

    def initUI(self):
        sig_view = pg.dockarea.DockArea()
        viewer_dock = Dock(name='3D viewer', closable=False)
        localizer = SEEGLocalizer(
            chs_3d_view=viewer_dock,
            patient_reg_folder=self.patient_reg_folder,
        )
        sig_view.addDock(viewer_dock, 'left')

        self.setCentralWidget(sig_view)


def main(argv=None):
    """Launch sEEG Localizer."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch sEEG Localizer")
    parser.add_argument(
        "patient_registration_folder",
        nargs="?",
        help="Optional folder containing an existing patient registration",
    )
    args = parser.parse_args(argv)

    _prepare_application_identity()
    app = QApplication.instance() or QApplication([APPLICATION_NAME])
    # MNE installs its logo when the QApplication icon is empty. Set ours
    # before constructing the viewer, which initializes MNE's Qt backend.
    _apply_application_icon(app)
    window = SEEGLocalizerMainWindow(patient_reg_folder=args.patient_registration_folder)
    window.setWindowTitle("sEEG Localizer")
    _apply_application_icon(app, window)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
