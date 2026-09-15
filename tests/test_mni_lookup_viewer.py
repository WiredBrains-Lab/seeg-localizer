"""Opt-in Qt/VTK integration checks using synthetic saved coordinates.

Run with SEEG_TEST_3D=1 QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests.
No FreeSurfer download, patient files, or running application is required.
"""

import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from viewer3d.cohort_lookup import LESION_COLUMNS
from viewer3d.mni_cohort import load_mni_cohort


@unittest.skipUnless(os.environ.get("SEEG_TEST_3D") == "1", "Set SEEG_TEST_3D=1 for Qt/VTK tests")
class MNILookupViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pyvista as pv
        from qtpy import QtWidgets
        from viewer3d.seeg_localizer import SEEGLocalizer

        cls.pv = pv
        cls.QtWidgets = QtWidgets
        cls.viewer_class = SEEGLocalizer
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for subject in ("sub-001", "sub-002"):
            directory = self.root / subject
            directory.mkdir()
            with (directory / "ct_elec_info.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["name", "shaft", "mni_x", "mni_y", "mni_z", "visible"])
                # Deliberately unsorted along each shaft to expose scalar/position mixups.
                for number in (3, 1, 2, 4):
                    writer.writerow([f"LA{number:02}", "LA", -20 if subject == "sub-001" else 20,
                                     0, number * 4, True])
        self.lookup_path = self.root / "lookup.csv"
        with self.lookup_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["subject", "electrode_name", *LESION_COLUMNS])
            writer.writerows([
                ["sub-001", "ns5_1_LA01", -0.1, 0.2, 0.3],
                ["sub-001", "ns5_3_LA03", 0.1, -0.4, 0.2],
                ["sub-002", "nf3_1_LA01", "", 0.1, 0.2],
                ["sub-002", "nf3_2_LA02", 0.2, 0.3, -0.5],
            ])
        self.plotter = self.pv.Plotter(off_screen=True, window_size=(900, 700))
        self.plotter.set_background("white")
        self.plotter.camera_position = [(80, -120, 50), (0, 0, 8), (0, 0, 1)]
        self.viewer = self.viewer_class()
        self.viewer.brain = SimpleNamespace(_renderer=SimpleNamespace(plotter=self.plotter))
        self.viewer.mni_cohort_show_labels = False
        self.panel = self.QtWidgets.QWidget()
        layout = self.QtWidgets.QVBoxLayout(self.panel)
        self.viewer._create_mni_lookup_controls(layout)
        self.viewer.mni_cohort_status_label = self.QtWidgets.QLabel()
        self.electrodes, self.summary = load_mni_cohort(self.root)
        self.viewer.mni_cohort_electrodes = self.electrodes
        self.viewer.mni_cohort_summary = self.summary
        self.viewer.mni_cohort_root = str(self.root)
        self.viewer.mni_cohort_active = True
        self.viewer.load_mni_cohort_lookup(self.lookup_path)

    def tearDown(self):
        self.viewer.brain = None
        self.plotter.close()
        self.panel.close()

    def test_smooth_field_render_filters_modes_and_export(self):
        from viewer3d.cohort_smoothing import CorticalGaussianField
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        for hemi, x in (("lh", -20), ("rh", 20)):
            white = np.array([[x, -4, 0], [x, 4, 0], [x, 4, 20], [x, -4, 20]], dtype=float)
            if not hasattr(self.viewer.brain, "geo"):
                self.viewer.brain.geo = {}
            self.viewer.brain.geo[hemi] = SimpleNamespace(
                coords=white, faces=faces, nn=np.tile([1, 0, 0], (4, 1)))
            self.viewer._mni_cohort_smooth_geometry[(str(self.root), "test", hemi)] = CorticalGaussianField(white, faces)
        with patch.object(self.viewer, "_get_brain_subject_info", return_value=("test", str(self.root))):
            self.viewer.mni_cohort_pool_checkbox.setChecked(True)
            self.viewer.mni_cohort_smooth_checkbox.setChecked(True)
            self.assertFalse(self.viewer.mni_cohort_pool_enabled)
            self.assertFalse(self.viewer.mni_cohort_pool_checkbox.isChecked())
            self.assertEqual(len(self.viewer._detected_electrode_actors), 2)
            self.assertEqual(len(self.plotter.scalar_bars), 1)
            self.assertEqual(self.viewer._detected_shaft_actors, [])
            self.assertEqual(self.viewer._mni_cohort_smooth_summary["contributing_contacts"], 3)
            mesh = self.viewer._detected_electrode_actors[0].mapper.dataset
            means = mesh.point_data["cohort_scalar"].copy()
            opacity = mesh.point_data["field_opacity"].copy()
            self.viewer.mni_color_controls["visibility"].setValue(50)
            mesh = self.viewer._detected_electrode_actors[0].mapper.dataset
            np.testing.assert_allclose(mesh.point_data["cohort_scalar"], means)
            np.testing.assert_allclose(mesh.point_data["field_opacity"], opacity * 0.5)
            self.viewer.mni_color_controls["enhance"].setChecked(True)
            mesh = self.viewer._detected_electrode_actors[0].mapper.dataset
            np.testing.assert_allclose(mesh.point_data["cohort_scalar"], means)

            camera = np.asarray(list(self.plotter.camera_position))
            paths = self.viewer.export_mni_cohort_figures(self.root)
            settings = json.loads((paths[0].parent / "figure_settings.json").read_text())
            self.assertTrue(settings["smooth_scalar_field"])
            self.assertEqual(settings["smoothing_fwhm_mm"], 15)
            self.assertEqual([f["smooth_field"]["contributing_contacts"] for f in settings["figures"]], [3, 4, 4])
            np.testing.assert_allclose(list(self.plotter.camera_position), camera)
            self.viewer.mni_cohort_selected_patients = {"sub-001"}
            self.assertTrue(self.viewer._display_mni_cohort_electrodes())
            self.assertEqual(len(self.viewer._detected_electrode_actors), 1)
            self.viewer.mni_cohort_selected_patients = set()
            self.assertTrue(self.viewer._display_mni_cohort_electrodes())
            self.assertEqual(len(self.viewer._detected_electrode_actors), 0)
            self.assertEqual(len(self.plotter.scalar_bars), 0)
            self.viewer.mni_cohort_selected_patients = None
            self.viewer.mni_cohort_pool_checkbox.setChecked(True)
            self.assertFalse(self.viewer.mni_cohort_smooth_enabled)
            self.assertFalse(self.viewer.mni_cohort_smooth_checkbox.isChecked())
            self.assertEqual(self.viewer._mni_cohort_smooth_summary, {})

    def test_color_range_lock_clipping_and_export(self):
        controls = self.viewer.mni_color_controls
        controls["spin"].setValue(0.05)
        self.assertEqual(self.viewer._mni_cohort_scalar_limits(), (-0.05, 0.05))
        self.assertIn("3 visible contact values outside", controls["status"].text())
        controls["lock"].setChecked(True)
        self.viewer.mni_cohort_color_combo.setCurrentIndex(2)
        self.assertEqual(self.viewer._mni_cohort_scalar_limits(), (-0.05, 0.05))
        controls["enhance"].setChecked(True)
        paths = self.viewer.export_mni_cohort_figures(self.root)
        settings = json.loads((paths[0].parent / "figure_settings.json").read_text())
        self.assertEqual(settings["locked_color_limit"], 0.05)
        self.assertEqual(settings["colormap"], "RdBu_r_asinh10")
        self.assertEqual([f["color_limits"] for f in settings["figures"]], [[-0.05, 0.05]] * 3)
        self.assertEqual([f["contacts_outside_color_range"] for f in settings["figures"]], [3, 4, 4])
        self.assertEqual(self.viewer.mni_cohort_scalar_column, LESION_COLUMNS[1])
        actor = self.viewer._detected_electrode_actors[0]
        self.assertEqual(tuple(actor.mapper.scalar_range), (-0.05, 0.05))
        self.assertAlmostEqual(np.nanmax(actor.mapper.dataset.cell_data["cohort_scalar"]), 0.3)
        controls["lock"].setChecked(False)
        self.assertEqual(self.viewer._mni_cohort_scalar_limits(), (-0.5, 0.5))
        self.viewer.mni_cohort_color_combo.setCurrentIndex(1)
        self.assertEqual(self.viewer._mni_cohort_scalar_limits(), (-0.05, 0.05))
        controls["full"].click()
        self.assertEqual(self.viewer._mni_cohort_scalar_limits(), (-0.5, 0.5))

    def test_robust_range_slider_and_filter_stability(self):
        controls = self.viewer.mni_color_controls
        controls["percentile"].setValue(50)
        controls["robust"].click()
        values = [self.viewer.mni_cohort_lookup.values[column][row]
                  for column in LESION_COLUMNS
                  for row in set(self.viewer.mni_cohort_lookup_matches.row_indices) if row is not None]
        expected = float(np.nanpercentile(np.abs(values), 50))
        self.assertEqual(self.viewer._mni_cohort_scalar_limits(), (-expected, expected))
        self.viewer.mni_cohort_selected_patients = {"sub-001"}
        self.viewer._display_mni_cohort_electrodes()
        self.assertEqual(self.viewer._mni_cohort_scalar_limits(), (-expected, expected))
        controls["slider"].setValue(0)
        self.assertAlmostEqual(self.viewer._mni_cohort_scalar_limits()[1], 0.0005)
        self.viewer.load_mni_cohort_lookup(self.lookup_path)
        self.assertEqual(self.viewer.mni_color_manual_limits, {})
        self.assertIsNone(self.viewer.mni_color_locked_limit)
        self.viewer.clear_mni_cohort_lookup()
        self.assertFalse(controls["spin"].isEnabled())

    def test_publication_export_resolution_dpi_current_column_and_state(self):
        from PIL import Image
        self.viewer.mni_cohort_color_combo.setCurrentIndex(2)
        self.viewer.mni_color_controls["spin"].setValue(0.05)
        self.viewer.mni_color_controls["enhance"].setChecked(True)
        self.plotter.set_background("#dddddd")
        self.viewer._orientation_widget = self.plotter.add_axes()
        self.assertTrue(self.viewer._orientation_widget.GetVisibility())
        background = tuple(self.plotter.renderer.GetBackground())
        camera = np.asarray(list(self.plotter.camera_position))
        window = tuple(self.plotter.window_size)
        image_path, settings_path = self.viewer.export_publication_image(self.root / "publication.png")
        with Image.open(image_path) as image:
            self.assertEqual(max(image.size), 4200)
            self.assertAlmostEqual(image.info["dpi"][0], 600, places=1)
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        settings = json.loads(settings_path.read_text())
        self.assertFalse(settings["orientation_marker_in_image"])
        self.assertTrue(self.viewer._orientation_widget.GetVisibility())
        self.assertEqual(settings["column"], LESION_COLUMNS[1])
        self.assertEqual(settings["color_limits"], [-0.05, 0.05])
        self.assertEqual(settings["colormap"], "RdBu_r_asinh10")
        self.assertEqual(self.viewer.mni_cohort_scalar_column, LESION_COLUMNS[1])
        np.testing.assert_allclose(list(self.plotter.camera_position), camera)
        self.assertEqual(tuple(self.plotter.renderer.GetBackground()), background)
        self.assertEqual(tuple(self.plotter.window_size), window)
        self.assertEqual(len(self.plotter.scalar_bars), 1)

    def test_publication_tiff_and_categorical_view(self):
        from PIL import Image
        self.viewer.mni_cohort_color_combo.setCurrentIndex(0)
        path, settings_path = self.viewer.export_publication_image(self.root / "publication.tif", long_edge_px=1200)
        with Image.open(path) as image:
            self.assertEqual(image.format, "TIFF")
            self.assertEqual(max(image.size), 1200)
            self.assertEqual(tuple(image.info["dpi"]), (600, 600))
        settings = json.loads(settings_path.read_text())
        self.assertIsNone(settings["column"])
        self.assertIsNone(settings["color_limits"])
        self.assertTrue(self.viewer.publication_image_btn.isEnabled())
        self.assertEqual(len(self.plotter.scalar_bars), 0)

    def test_publication_uses_actual_render_size_when_qt_size_is_stale(self):
        from unittest.mock import PropertyMock
        # Embedded Qt viewers can report the original 800x800 size while VTK
        # actually renders a wide Retina viewport. Avoid excessive capture tiles.
        actual = tuple(self.plotter.render_window.GetSize())
        with patch.object(type(self.plotter), "window_size", new_callable=PropertyMock,
                          return_value=(200, 200)):
            with patch.object(self.plotter, "screenshot", wraps=self.plotter.screenshot) as capture:
                _, settings_path = self.viewer.export_publication_image(
                    self.root / "actual-size.png", long_edge_px=1200)
        self.assertEqual(capture.call_args.kwargs["scale"], 2)
        settings = json.loads(settings_path.read_text())
        self.assertEqual(settings["source_window_size"], list(actual))
        self.assertEqual(settings["render_scale"], 2)
        self.assertEqual(settings["pixel_size"], [1200, 933])

    def test_publication_failures_restore_scene_and_preserve_existing_file(self):
        path = self.root / "publication.png"
        path.write_bytes(b"existing image")
        self.plotter.set_background("#222222", top="#444444")
        renderer = self.plotter.renderer
        before = (renderer.GetBackground(), renderer.GetBackground2(), renderer.GetGradientBackground())
        camera = np.asarray(list(self.plotter.camera_position))
        with patch.object(self.plotter, "screenshot", side_effect=OSError("render failed")):
            with self.assertRaisesRegex(OSError, "render failed"):
                self.viewer.export_publication_image(path)
        self.assertEqual(path.read_bytes(), b"existing image")
        self.assertEqual((renderer.GetBackground(), renderer.GetBackground2(), renderer.GetGradientBackground()), before)
        np.testing.assert_allclose(list(self.plotter.camera_position), camera)
        with patch("PIL.Image.Image.save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.viewer.export_publication_image(path, long_edge_px=600)
        self.assertEqual(path.read_bytes(), b"existing image")
        self.assertEqual(list(self.root.glob(".publication-*")), [])

    def test_surface_radio_exclusivity_and_unavailable_flatmap(self):
        layout = self.panel.layout()
        self.viewer._create_surface_controls(layout)
        buttons = self.viewer.surface_buttons
        self.assertTrue(buttons["pial"].isChecked())
        def reload(surface):
            self.viewer.brain_surface = surface
        with patch.object(self.viewer, "_surface_files_available", return_value=True), \
                patch.object(self.viewer, "_reload_brain_with_subject", side_effect=reload) as load:
            buttons["flat"].click()
            load.assert_not_called()
            self.app.processEvents()
            self.assertEqual(self.viewer.brain_surface, "flat")
            self.assertEqual(sum(b.isChecked() for b in buttons.values()), 1)
            buttons["inflated"].click()
            self.app.processEvents()
            self.assertEqual(self.viewer.brain_surface, "inflated")
            self.assertEqual(load.call_count, 2)
        with patch.object(self.viewer, "_surface_files_available", return_value=False), \
                patch.object(self.QtWidgets.QMessageBox, "warning"):
            buttons["flat"].click()
            self.app.processEvents()
        self.assertTrue(buttons["inflated"].isChecked())
        self.assertFalse(buttons["flat"].isChecked())

    def test_surface_switch_finishes_click_before_controls_are_destroyed(self):
        from qtpy import QtCore, QtTest
        controls = self.QtWidgets.QWidget(self.panel)
        self.panel.layout().addWidget(controls)
        self.viewer._create_surface_controls(self.QtWidgets.QHBoxLayout(controls))
        click_finished = []
        destroyed = []
        old_button = self.viewer.surface_buttons["flat"]
        self.panel.show()
        self.app.processEvents()
        old_button.clicked.connect(lambda: click_finished.append(True))
        old_button.destroyed.connect(lambda: destroyed.append(True))

        def reload(surface):
            self.assertTrue(click_finished)
            self.assertTrue(self.viewer._surface_switch_in_progress)
            self.assertFalse(old_button.isEnabled())
            controls.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            self.viewer.brain_surface = surface
            replacement = self.QtWidgets.QWidget(self.panel)
            self.panel.layout().addWidget(replacement)
            self.viewer._create_surface_controls(self.QtWidgets.QHBoxLayout(replacement))
            self.assertFalse(self.viewer.surface_buttons["pial"].isEnabled())
            # A nested event loop cannot start a second renderer switch.
            self.viewer._queue_surface_change("inflated")
            self.app.processEvents()

        with patch.object(self.viewer, "_surface_files_available", return_value=True), \
                patch.object(self.viewer, "_reload_brain_with_subject", side_effect=reload) as load:
            QtTest.QTest.mouseClick(old_button, QtCore.Qt.LeftButton, pos=QtCore.QPoint(8, old_button.height() // 2))
            self.assertTrue(click_finished)
            load.assert_not_called()
            self.assertFalse(destroyed)
            self.app.processEvents()
            self.assertTrue(destroyed)
            self.assertEqual(load.call_count, 1)
        self.assertEqual(self.viewer.brain_surface, "flat")
        self.assertTrue(self.viewer.surface_buttons["flat"].isChecked())
        self.assertTrue(all(button.isEnabled() for button in self.viewer.surface_buttons.values()))

    def test_pending_surface_changes_use_latest_selection(self):
        self.viewer._create_surface_controls(self.panel.layout())
        with patch.object(self.viewer, "_select_brain_surface") as select:
            self.viewer.surface_buttons["flat"].click()
            self.viewer.surface_buttons["inflated"].click()
            select.assert_not_called()
            self.app.processEvents()
            select.assert_called_once_with("inflated")

    def test_failed_surface_load_preserves_existing_brain(self):
        previous = self.viewer.brain
        with patch.object(self.viewer, "_get_brain_subject_info", return_value=("test", str(self.root))), \
                patch.object(self.viewer, "_surface_files_available", return_value=True), \
                patch("mne.viz.Brain", side_effect=ValueError("invalid patch")):
            with self.assertRaisesRegex(ValueError, "invalid patch"):
                self.viewer._reload_brain_with_subject(surface="flat")
        self.assertIs(self.viewer.brain, previous)
        self.assertEqual(self.viewer.brain_surface, "pial")

    def test_flatmap_uses_displayed_vertices_and_ignores_patch_holes(self):
        self.viewer.brain_surface = "flat"
        white = np.array([[0, 0, 0], [-20, 0, 0], [-20, 1, 0], [-20, 0, 1]])
        coords = np.array([[0, 0, 0], [-80, 20, 0], [-80, 21, 0], [-80, 20, 1]])
        faces = np.array([[1, 2, 3]])
        self.viewer.brain.geo = {
            "lh": SimpleNamespace(coords=coords, faces=faces),
            "rh": SimpleNamespace(coords=-coords, faces=faces),
        }
        with patch.object(self.viewer, "_get_brain_subject_info", return_value=("test", str(self.root))), \
                patch("mne.surface.read_surface", side_effect=[(white, faces), (-white, faces)]):
            # Near the omitted vertex 0: must land on a retained hemisphere vertex.
            mapped = self.viewer._map_points_to_display_surface([[-0.1, 0, 0], [0.1, 0, 0]])
        np.testing.assert_allclose(mapped, [coords[1], -coords[1]])
        self.assertEqual(self.viewer._map_points_to_display_surface([]).shape, (0, 3))

    def test_flatmap_pooling_stays_in_mni_and_preserves_values(self):
        self.viewer.brain_surface = "flat"
        with patch.object(self.viewer, "_map_points_to_display_surface", side_effect=lambda pts: np.asarray(pts) * 100):
            self.viewer.mni_cohort_pool_checkbox.setChecked(True)
            self.assertEqual(len(self.viewer._mni_cohort_pooled_groups), 2)
            np.testing.assert_allclose([g["mean"] for g in self.viewer._mni_cohort_pooled_groups], [0, 0.2])
            self.assertEqual(self.viewer._detected_shaft_actors, [])
            self.viewer.mni_cohort_pool_checkbox.setChecked(False)
            self.assertEqual(self.viewer._detected_shaft_actors, [])
        self.viewer.brain.show_view = lambda view: self.assertEqual(view, "flat")
        self.viewer._set_brain_view("Left")
        self.assertTrue(self.plotter.camera.parallel_projection)

    def test_pool_toggle_means_filter_and_export(self):
        camera = np.asarray(list(self.plotter.camera_position))
        self.viewer.mni_cohort_pool_checkbox.setChecked(True)
        groups = self.viewer._mni_cohort_pooled_groups
        self.assertEqual(len(groups), 2)
        np.testing.assert_allclose([g["mean"] for g in groups], [0, 0.2])
        self.assertEqual([g["contact_count"] for g in groups], [2, 2])
        self.assertEqual(self.viewer._detected_shaft_actors, [])
        self.assertEqual(len(self.plotter.scalar_bars), 1)
        paths = self.viewer.export_mni_cohort_figures(self.root)
        settings = json.loads((paths[0].parent / "figure_settings.json").read_text())
        self.assertTrue(settings["spatial_pooling"])
        self.assertEqual(len(settings["figures"][0]["pooled_groups"]), 2)
        self.viewer.mni_cohort_pool_spin.setValue(3)
        self.assertEqual(len(self.viewer._mni_cohort_pooled_groups), 4)
        self.viewer.mni_cohort_selected_patients = {"sub-001"}
        self.viewer._display_mni_cohort_electrodes()
        self.assertEqual(len(self.viewer._mni_cohort_pooled_groups), 2)
        self.viewer.mni_cohort_pool_checkbox.setChecked(False)
        self.assertEqual(self.viewer._mni_cohort_pooled_groups, [])
        np.testing.assert_allclose(list(self.plotter.camera_position), camera)
        self.viewer.mni_cohort_pool_checkbox.setChecked(True)
        self.viewer.mni_cohort_selected_patients = set()
        self.assertTrue(self.viewer._display_mni_cohort_electrodes())
        self.assertEqual(self.viewer._detected_electrode_actors, [])
        self.assertEqual(len(self.plotter.scalar_bars), 0)

    def test_cell_scalars_stay_on_correct_contacts_after_sorting(self):
        self.assertEqual(len(self.viewer._mni_cohort_display_indices()), 4)
        self.assertEqual(len(self.plotter.scalar_bars), 1)
        actor = self.viewer._detected_electrode_actors[0]
        mesh = actor.mapper.dataset
        scalars = mesh.cell_data["cohort_scalar"]
        for value, center in ((-0.1, (-20, 0, 4)), (0.1, (-20, 0, 12)), (0.2, (20, 0, 8))):
            subset = mesh.extract_cells(np.flatnonzero(scalars == value))
            np.testing.assert_allclose(subset.center, center, atol=0.001)
        missing = mesh.extract_cells(np.flatnonzero(np.isnan(scalars)))
        np.testing.assert_allclose(missing.center, (20, 0, 4), atol=0.001)
        self.assertEqual(tuple(actor.mapper.scalar_range), (-0.5, 0.5))
        # The missing LA02 at x=-20 must not be bridged by a visible connector.
        tubes = self.viewer._detected_shaft_actors[0].mapper.dataset
        self.assertGreater(tubes.bounds[0], 19)

    def test_default_colors_keep_filter_and_clear_removes_filter_and_legend(self):
        combo = self.viewer.mni_cohort_color_combo
        combo.setCurrentIndex(0)
        self.assertEqual(len(self.plotter.scalar_bars), 0)
        self.assertEqual(len(self.viewer._mni_cohort_display_indices()), 4)
        self.viewer.mni_cohort_lookup_filter_checkbox.setChecked(False)
        self.assertEqual(len(self.viewer._mni_cohort_display_indices()), 8)
        combo.setCurrentIndex(1)
        self.assertEqual(len(self.plotter.scalar_bars), 1)
        self.viewer.clear_mni_cohort_lookup()
        self.assertEqual(len(self.plotter.scalar_bars), 0)
        self.assertEqual(len(self.viewer._mni_cohort_display_indices()), 8)
        self.assertFalse(self.viewer.mni_cohort_export_btn.isEnabled())

    def test_patient_filter_preserves_scale_camera_and_handles_empty_selection(self):
        camera = np.asarray(list(self.plotter.camera_position))
        self.viewer.mni_cohort_selected_patients = {"sub-001"}
        self.assertTrue(self.viewer._display_mni_cohort_electrodes())
        self.assertEqual(len(self.viewer._mni_cohort_display_indices()), 2)
        self.assertEqual(self.viewer._mni_cohort_scalar_limits(), (-0.5, 0.5))
        np.testing.assert_allclose(list(self.plotter.camera_position), camera)
        self.viewer.mni_cohort_selected_patients = set()
        self.assertTrue(self.viewer._display_mni_cohort_electrodes())
        self.assertEqual(len(self.plotter.scalar_bars), 0)
        self.assertEqual(self.viewer._detected_electrode_actors, [])
        with self.assertRaisesRegex(ValueError, "No contacts"):
            self.viewer.export_mni_cohort_figures(self.root)

    def test_export_three_distinct_figures_with_settings_and_restored_view(self):
        self.viewer.mni_cohort_color_combo.setCurrentIndex(2)
        camera = np.asarray(list(self.plotter.camera_position))
        paths = self.viewer.export_mni_cohort_figures(self.root)
        self.assertEqual(len(paths), 3)
        self.assertEqual(len({path.read_bytes() for path in paths}), 3)
        settings = json.loads((paths[0].parent / "figure_settings.json").read_text())
        self.assertEqual([item["column"] for item in settings["figures"]], list(LESION_COLUMNS))
        self.assertEqual([item["color_limits"] for item in settings["figures"]], [[-0.5, 0.5]] * 3)
        self.assertEqual([item["contacts_with_values"] for item in settings["figures"]], [3, 4, 4])
        self.assertEqual(self.viewer.mni_cohort_scalar_column, LESION_COLUMNS[1])
        self.assertEqual(len(self.plotter.scalar_bars), 1)
        np.testing.assert_allclose(list(self.plotter.camera_position), camera)

    def test_individual_scale_and_all_missing_selected_contacts(self):
        self.viewer.mni_cohort_shared_scale_checkbox.setChecked(False)
        self.assertEqual(self.viewer._mni_cohort_scalar_limits(), (-0.2, 0.2))
        self.viewer.mni_cohort_lookup.values[LESION_COLUMNS[0]][3] = np.nan
        self.viewer.mni_cohort_selected_patients = {"sub-002"}
        self.assertTrue(self.viewer._display_mni_cohort_electrodes())
        self.viewer._update_mni_cohort_status()
        self.assertIn("2 without values (gray)", self.viewer.mni_cohort_status_label.text())
        mesh = self.viewer._detected_electrode_actors[0].mapper.dataset
        self.assertTrue(np.all(np.isnan(mesh.cell_data["cohort_scalar"])))

    def test_invalid_replacement_and_export_failure_preserve_selection(self):
        lookup = self.viewer.mni_cohort_lookup
        with self.assertRaises(OSError):
            self.viewer.load_mni_cohort_lookup(self.root / "missing.csv")
        self.assertIs(self.viewer.mni_cohort_lookup, lookup)
        self.viewer.mni_cohort_color_combo.setCurrentIndex(3)
        with patch.object(self.plotter, "screenshot", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                self.viewer.export_mni_cohort_figures(self.root)
        self.assertEqual(self.viewer.mni_cohort_scalar_column, LESION_COLUMNS[2])
        self.assertEqual(len(self.plotter.scalar_bars), 1)

    def test_clear_and_reload_cohort_rejoins_existing_lookup(self):
        self.viewer.clear_mni_cohort()
        self.assertEqual(len(self.plotter.scalar_bars), 0)
        self.assertFalse(self.viewer.mni_cohort_export_btn.isEnabled())
        with patch.object(self.viewer, "_get_brain_subject_info", return_value=("fsaverage", "unused")):
            self.viewer.load_mni_cohort_directory(self.root)
        self.assertEqual(self.viewer.mni_cohort_lookup_matches.matched_contacts, 4)
        self.assertEqual(len(self.viewer._mni_cohort_display_indices()), 4)
        self.assertTrue(self.viewer.mni_cohort_export_btn.isEnabled())

    def test_generic_scalar_exports_one_figure_and_loads_before_cohort(self):
        with self.lookup_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["subject", "electrode_name", "effect"])
            writer.writerow(["sub-001", "LA01", 0.01])
        self.viewer.clear_mni_cohort()
        self.viewer.load_mni_cohort_lookup(self.lookup_path)
        self.assertEqual(self.viewer.mni_cohort_scalar_column, "effect")
        self.assertFalse(self.viewer.mni_cohort_export_btn.isEnabled())
        with patch.object(self.viewer, "_get_brain_subject_info", return_value=("fsaverage", "unused")):
            self.viewer.load_mni_cohort_directory(self.root)
        paths = self.viewer.export_mni_cohort_figures(self.root)
        self.assertEqual([path.name for path in paths], ["mni_effect.png"])
        self.assertEqual(self.viewer._mni_cohort_scalar_limits(), (-0.01, 0.01))


if __name__ == "__main__":
    unittest.main()
