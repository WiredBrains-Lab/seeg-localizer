"""Check that the project icon takes precedence over MNE's Qt default."""

import os
import unittest


@unittest.skipUnless(os.environ.get("SEEG_TEST_QT") == "1", "Set SEEG_TEST_QT=1 for Qt tests")
class ApplicationIconTests(unittest.TestCase):
    def test_mne_initialization_does_not_replace_application_icon(self):
        from qtpy.QtWidgets import QApplication
        from mne.viz.backends._utils import _init_mne_qtapp
        from viewer3d.seeg_localizer import _application_icon_path, _apply_application_icon

        app = QApplication.instance() or QApplication([])
        icon = _apply_application_icon(app)
        self.assertIsNotNone(_application_icon_path())
        self.assertIsNotNone(icon)
        self.assertFalse(app.windowIcon().isNull())
        project_icon_key = app.windowIcon().cacheKey()

        self.assertIs(_init_mne_qtapp(enable_icon=True), app)
        self.assertEqual(app.windowIcon().cacheKey(), project_icon_key)
        from viewer3d.seeg_localizer import _apply_application_identity
        _apply_application_identity(app)
        self.assertEqual(app.applicationName(), "sEEG Localizer")
        self.assertEqual(app.applicationDisplayName(), "sEEG Localizer")

    def test_about_action_opens_application_information(self):
        from unittest.mock import patch
        from qtpy.QtWidgets import QApplication, QAction
        from viewer3d.seeg_localizer import SEEGLocalizerMainWindow

        app = QApplication.instance() or QApplication([])
        with patch.object(SEEGLocalizerMainWindow, "initUI"):
            window = SEEGLocalizerMainWindow()
        try:
            self.assertEqual(window.about_action.menuRole(), QAction.AboutRole)
            with patch("qtpy.QtWidgets.QMessageBox.about") as about:
                window.about_action.trigger()
            title, body = about.call_args.args[1:]
            self.assertEqual(title, "About sEEG Localizer")
            self.assertIn("Sunil Mathew", body)
            self.assertIn("MIT License", body)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
