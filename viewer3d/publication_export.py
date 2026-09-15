"""Lossless, high-resolution exports of the current VTK scene."""
from pathlib import Path
import json
import math
import os
import tempfile

import numpy as np
from PIL import Image


def save_publication_image(plotter, path, settings=None, long_edge_px=4200, dpi=600, orientation_widget=None):
    """Render at export resolution, preserve aspect ratio, and embed print DPI.

    Scaling is done by VTK, followed only by downsampling if needed. No screenshot
    upscaling. The scene camera, background, and interactive window are restored
    on success and failure. Both image and sidecar are prepared before replacing
    any existing output files.
    """
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix not in ('.png', '.tif', '.tiff'):
        raise ValueError('Choose a PNG or TIFF filename')
    if not isinstance(long_edge_px, int) or not 600 <= long_edge_px <= 12000:
        raise ValueError('Longest edge must be an integer between 600 and 12000 pixels')
    if not isinstance(dpi, int) or not 72 <= dpi <= 1200:
        raise ValueError('DPI must be an integer between 72 and 1200')
    if not path.parent.is_dir():
        raise ValueError('Choose an existing output directory')
    camera = plotter.camera.copy()
    # QtInteractor can retain its initial window_size after the embedded widget
    # resizes (including Retina scaling). VTK captures the actual render window.
    # Using the stale size can request enormous tiled images and lose mesh tiles.
    window_size = tuple(plotter.render_window.GetSize())
    if len(window_size) != 2 or min(window_size) <= 0:
        raise RuntimeError('The viewer must have a nonempty render window before export')
    orientation_enabled = None
    orientation_setter = None
    if orientation_widget is not None:
        if hasattr(orientation_widget, 'GetEnabled'):
            orientation_enabled = orientation_widget.GetEnabled()
            orientation_setter = orientation_widget.SetEnabled
        elif hasattr(orientation_widget, 'GetVisibility'):
            orientation_enabled = orientation_widget.GetVisibility()
            orientation_setter = orientation_widget.SetVisibility
    backgrounds = [(renderer, renderer.GetBackground(), renderer.GetBackground2(),
                    renderer.GetGradientBackground()) for renderer in plotter.renderers]
    try:
        # Interactive VTK axis labels can be corrupted by tiled high-res capture.
        if orientation_setter is not None:
            orientation_setter(False)
        for renderer, *_ in backgrounds:
            renderer.SetBackground(1, 1, 1)
            renderer.SetGradientBackground(False)
        # Render synchronously, including for QtInteractor whose render() can queue.
        plotter.render_window.Render()
        scale = max(1, math.ceil(long_edge_px / max(window_size)))
        pixels = plotter.screenshot(scale=scale, transparent_background=False, return_img=True)
        if pixels is None or np.asarray(pixels).ndim != 3:
            raise RuntimeError('The renderer did not return an image')
        image = Image.fromarray(pixels).convert('RGB')
        if max(image.size) < long_edge_px:
            raise RuntimeError('Renderer returned fewer pixels than requested')
        size = tuple(max(1, round(value * long_edge_px / max(image.size))) for value in image.size)
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
    finally:
        plotter.camera = camera
        for renderer, first, second, gradient in backgrounds:
            renderer.SetBackground(*first)
            renderer.SetBackground2(*second)
            renderer.SetGradientBackground(gradient)
        if orientation_setter is not None:
            orientation_setter(orientation_enabled)
        plotter.render()

    sidecar = path.with_suffix('.settings.json')
    metadata = dict(settings or {})
    metadata.update({
        'image_file': path.name, 'pixel_size': list(image.size), 'dpi': dpi,
        'print_size_inches': [value / dpi for value in image.size],
        'orientation_marker_in_image': False,
        'background': 'white', 'format': 'PNG' if suffix == '.png' else 'TIFF',
        'camera_position': [list(point) for point in plotter.camera_position],
        'parallel_projection': bool(camera.parallel_projection),
        'parallel_scale': float(camera.parallel_scale),
        'source_window_size': list(window_size),
        'render_scale': scale,
    })
    temporary = []
    try:
        for target in (path, sidecar):
            handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{target.stem}-', delete=False)
            handle.close()
            temporary.append(Path(handle.name))
        kwargs = {'format': metadata['format'], 'dpi': (dpi, dpi)}
        if suffix != '.png':
            kwargs['compression'] = 'tiff_deflate'
        image.save(temporary[0], **kwargs)
        temporary[1].write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary[0], path)
        os.replace(temporary[1], sidecar)
    finally:
        for temp in temporary:
            temp.unlink(missing_ok=True)
    return path, sidecar
