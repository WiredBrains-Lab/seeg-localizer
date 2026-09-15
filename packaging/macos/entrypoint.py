"""PyInstaller entry point for the macOS application bundle."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import sys


# MNE and QtPy otherwise choose their GUI backends at runtime. Set explicit
# defaults before importing the application so frozen builds use the bundled
# PyQt5/PyVistaQt stack.
os.environ.setdefault("QT_API", "pyqt5")
os.environ.setdefault("MNE_3D_BACKEND", "pyvistaqt")
os.environ.setdefault("MNE_BROWSER_BACKEND", "qt")

if getattr(sys, "frozen", False):
    # PyInstaller's Matplotlib runtime hook uses a temporary configuration
    # directory. A persistent per-user cache avoids rescanning all macOS fonts
    # every time the application starts.
    matplotlib_cache = (
        Path.home() / "Library" / "Caches" / "sEEG Localizer" / "matplotlib"
    )
    try:
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    else:
        os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)


def run() -> int:
    multiprocessing.freeze_support()

    from viewer3d.seeg_localizer import main

    return int(main())


if __name__ == "__main__":
    raise SystemExit(run())
