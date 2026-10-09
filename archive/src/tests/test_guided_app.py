import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import UnidentifiedImageError
from streamlit.testing.v1 import AppTest

import app_support as support

APP = support.ROOT / 'app.py'
TIMEOUT_ERROR = subprocess.TimeoutExpired('test', 1)
SAMPLE_IMAGE = support.decode_image(support.SAMPLE.read_bytes())


class GuidedTests(unittest.TestCase):
    def test_sample_decodes(self):
        self.assertEqual(support.decode_image(support.SAMPLE.read_bytes()).mode, 'RGB')

    def test_empty_image(self):
        with self.assertRaises(ValueError):
            support.decode_image(b'')

    def test_corrupt_image(self):
        with self.assertRaises(UnidentifiedImageError):
            support.decode_image(b'not an image')

    def test_byte_limit(self):
        with patch.object(support, 'MAX_BYTES', 1), self.assertRaises(ValueError):
            support.decode_image(b'xx')

    def test_pixel_limit(self):
        with patch.object(support, 'MAX_PIXELS', 1), self.assertRaises(ValueError):
            support.decode_image(support.SAMPLE.read_bytes())

    def test_partial_reports(self):
        with self.assertRaises(ValueError):
            support.validate_reports({'phone': {'results': [{}]}}, 'both')

    def test_process_failure(self):
        with self.assertRaises(ValueError):
            support.locate_results(SimpleNamespace(returncode=1, stderr=''))

    def test_result_path(self):
        result = SimpleNamespace(returncode=0, stderr='INFO Results {"run": "' + 'a' * 32 + '"}')
        self.assertEqual(support.locate_results(result), support.REPORTS / ('a' * 32))

    def test_unique_uploads(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(support, 'UPLOADS', Path(d)),
        ):
            assert support.save_input(SAMPLE_IMAGE) != support.save_input(SAMPLE_IMAGE)

    def test_welcome(self):
        app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.radio[0].options), 3)

    def test_missing_upload_message(self):
        app = AppTest.from_file(str(APP)).run()
        app.radio[1].set_value('Choose my photo')
        app.button[0].click().run()
        self.assertIn('Choose a photo', app.error[0].value)

    def test_run_failure_message(self):
        app = AppTest.from_file(str(APP)).run()
        with patch.object(support, 'run_comparison', side_effect=ValueError(support.RUN_ERROR)):
            app.button[0].click().run()
        assert app.error

    def test_timeout_message(self):
        app = AppTest.from_file(str(APP)).run()
        with patch.object(support, 'run_comparison', side_effect=TIMEOUT_ERROR):
            app.button[0].click().run()
        assert app.error
