"""Build configuration for the sEEG Localizer macOS application."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tomllib

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
)


SPEC_DIR = Path(SPECPATH).resolve()
REPOSITORY_ROOT = SPEC_DIR.parents[1]
APP_ICON = SPEC_DIR / "seeg-localizer.icns"

with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
    APP_VERSION = tomllib.load(handle)["project"]["version"]


def without_tests(module_name: str) -> bool:
    parts = module_name.split(".")
    return not {"conftest", "test", "tests", "testing"}.intersection(parts)


def pyvista_runtime_module(module_name: str) -> bool:
    return (
        without_tests(module_name)
        and not module_name.startswith("pyvista.ext")
        and not module_name.startswith("pyvista.trame")
    )


def vtk_runtime_module(module_name: str) -> bool:
    excluded_prefixes = (
        "vtkmodules.gtk",
        "vtkmodules.test",
        "vtkmodules.tk",
        "vtkmodules.web",
        "vtkmodules.wx",
    )
    return (
        without_tests(module_name)
        and module_name not in {"vtkmodules.all", "vtkmodules.generate_pyi"}
        and not module_name.startswith(excluded_prefixes)
    )


datas = collect_data_files(
    "mne",
    excludes=["**/tests/**", "**/testing/**"],
)
datas.append((str(SPEC_DIR / "seeg-localizer-icon.png"), "."))
datas += collect_data_files("viewer3d", includes=["assets/*.png"])
binaries = []
hiddenimports = collect_submodules("mne", filter=without_tests)

# These libraries use lazy imports or discover plugins dynamically. Their
# PyInstaller hooks cover most files; explicit collection keeps the MNE 3D
# backend and the optional registration engines available in the frozen app.
for package_name in ("pyvista", "pyvistaqt"):
    datas += collect_data_files(
        package_name,
        excludes=["**/tests/**", "**/examples/**"],
    )
    hiddenimports += collect_submodules(
        package_name,
        filter=pyvista_runtime_module,
    )

hiddenimports += collect_submodules("vtkmodules", filter=vtk_runtime_module)

for package_name in ("ants",):
    if importlib.util.find_spec(package_name) is None:
        continue
    package_datas, package_binaries, package_imports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports


analysis = Analysis(
    [str(SPEC_DIR / "entrypoint.py")],
    pathex=[str(REPOSITORY_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "PyQt6",
        "PySide2",
        "PySide6",
        "bokeh",
        "cupy",
        "cv2",
        "dask",
        "distributed",
        "fury",
        "gevent",
        "imageio_ffmpeg",
        "ipywidgets",
        "jax",
        "jaxlib",
        "jupyter",
        "jupyterlab",
        "keras",
        "mayavi",
        "netCDF4",
        "nilearn",
        "notebook",
        "panel",
        "plotly",
        "pyarrow",
        "pygame",
        "sqlalchemy",
        "tables",
        "tensorflow",
        "tkinter",
        "torch",
        "trame",
        "xarray",
        "zmq",
    ],
    noarchive=False,
    optimize=1,
)

python_archive = PYZ(analysis.pure)

codesign_identity = os.environ.get("SEEG_MACOS_CODESIGN_IDENTITY") or None

executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="sEEG Localizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=codesign_identity,
    entitlements_file=None,
)

collected = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="sEEG Localizer",
)

application = BUNDLE(
    collected,
    name="sEEG Localizer.app",
    icon=str(APP_ICON),
    bundle_identifier="edu.mcw.wiredbrains.seeg-localizer",
    version=APP_VERSION,
    codesign_identity=codesign_identity,
    entitlements_file=None,
    info_plist={
        "CFBundleDisplayName": "sEEG Localizer",
        "CFBundleName": "sEEG Localizer",
        "LSApplicationCategoryType": "public.app-category.education",
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
    },
)
