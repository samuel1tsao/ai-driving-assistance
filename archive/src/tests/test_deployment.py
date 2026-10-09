import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import patch

from filelock import FileLock, Timeout

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/scripts'))
import run_deployment as runner


class DeploymentTests(unittest.TestCase):
    def test_both_modes(self):
        self.assertEqual(runner.selected_modes('both'), ['phone', 'backend'])

    def test_single_mode(self):
        self.assertEqual(runner.selected_modes('phone'), ['phone'])

    def test_invalid_mode(self):
        with self.assertRaises(SystemExit):
            runner.PARSER.parse_args(['--mode', 'invalid', '--image', 'unused'])

    def test_missing_model(self):
        with self.assertRaises(FileNotFoundError):
            runner.require_model('nonexistent-checkpoint.pt')

    def test_invalid_frames(self):
        args = argparse.Namespace(frames=0, image=Path(__file__), mode='backend')
        with self.assertRaises(ValueError):
            runner.validate_inputs(args)

    def test_missing_image(self):
        args = argparse.Namespace(frames=1, image=Path('missing.jpg'), mode='phone')
        with self.assertRaises(ValueError):
            runner.validate_inputs(args)

    def test_subprocess_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            assert_failed_process(Path(directory))

    def test_lock_contention(self):
        with tempfile.TemporaryDirectory() as directory:
            assert_lock_contention(Path(directory))

    def test_unique_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            assert_unique_runs(Path(directory))


def assert_failed_process(directory):
    with (
        patch.object(runner, 'LOCK', directory / 'lock'),
        unittest.TestCase().assertRaises(CalledProcessError),
    ):
        runner.execute([[sys.executable, '-c', 'raise SystemExit(7)']])


def assert_lock_contention(directory):
    with (
        patch.object(runner, 'LOCK', directory / 'lock'),
        FileLock(str(directory / 'lock')),
        unittest.TestCase().assertRaises(Timeout),
    ):
        runner.execute([])


def assert_unique_runs(directory):
    args = argparse.Namespace(mode='phone', image=Path(__file__), frames=1)
    with patch.object(runner, 'RESULTS', directory):
        assert runner.prepare_run(args) != runner.prepare_run(args)


if __name__ == '__main__':
    unittest.main()
