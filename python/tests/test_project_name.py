"""Project branding must not rename or merge the distinct baseline engine."""
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts.audit.check_project_name import naming_errors


class ProjectNameTests(unittest.TestCase):
    def test_rejects_personal_paths_without_historical_exemptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.py'
            source.write_text('model = "' + '/home' + '/example/models/checkpoint"\n')
            with patch('scripts.audit.check_project_name.subprocess.check_output', return_value=b'source.py\0'):
                self.assertEqual(naming_errors(root), ['personal filesystem path: source.py:1'])
            source.write_text('engine = "' + 'vv' + 'llm"\n')
            with patch('scripts.audit.check_project_name.subprocess.check_output', return_value=b'source.py\0'):
                self.assertEqual(naming_errors(root), ['obsolete identifier: source.py:1'])
            source.write_text('directory = "' + 'mlsys' + '-verif"\n')
            with patch('scripts.audit.check_project_name.subprocess.check_output', return_value=b'source.py\0'):
                self.assertEqual(naming_errors(root), ['obsolete identifier: source.py:1'])

    def test_no_stale_active_identifiers_or_paths(self):
        self.assertEqual(naming_errors(ROOT), [])

    def test_native_and_baseline_workers_are_distinct(self):
        import importlib.util
        from scripts.determinism_tests.protocol import ALL_EXECUTION_CONFIGS
        spec = importlib.util.find_spec('vosti_kernels')
        self.assertIsNotNone(spec)
        self.assertEqual(Path(spec.origin).resolve(), ROOT / 'python/vosti_kernels/__init__.py')
        workers = ROOT / 'scripts/determinism_tests'
        self.assertTrue((workers / 'vosti_worker.py').is_file())
        self.assertTrue((workers / 'vllm_worker.py').is_file())
        configs = {config.key: config for config in ALL_EXECUTION_CONFIGS}
        self.assertEqual(configs['vosti-padded-graph'].engine, 'vosti')
        self.assertEqual(configs['vllm-fast-auto'].engine, 'vllm')


if __name__ == '__main__':
    unittest.main()
