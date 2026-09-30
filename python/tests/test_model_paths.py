"""Checkpoint defaults must be portable and independent of personal directories."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.common.model_paths import ROOT, checkpoint_path


class ModelPathTests(unittest.TestCase):
    def test_default_is_checkout_relative(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(checkpoint_path('checkpoint'), str(ROOT / 'models/checkpoint'))

    def test_explicit_model_root(self):
        with patch.dict(os.environ, {'VOSTI_MODEL_ROOT': '/model-store'}):
            self.assertEqual(checkpoint_path('checkpoint'), '/model-store/checkpoint')

    def test_relative_root_is_resolved(self):
        with patch.dict(os.environ, {'VOSTI_MODEL_ROOT': 'weights'}):
            self.assertEqual(checkpoint_path('checkpoint'), str(Path('weights/checkpoint').resolve()))
