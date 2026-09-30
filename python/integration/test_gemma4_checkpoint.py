"""Opt-in checkpoint/profile compatibility; no GPU or weight loading."""
from pathlib import Path
import unittest

from scripts.common.model_paths import checkpoint_path
from vosti_kernels.model_families.gemma4 import profile
from vosti_kernels.model_families.gemma4.loader import inspect_text_checkpoint


class Gemma4CheckpointTests(unittest.TestCase):
    def test_local_checkpoints_match_profiles(self):
        checkpoints = [(size, Path(checkpoint_path(f"gemma-4-{size}b-it")))
                       for size in (12, 31)]
        self.assertTrue(any(path.is_dir() for _, path in checkpoints),
                        "Set VOSTI_MODEL_ROOT to a directory containing "
                        "gemma-4-12b-it or gemma-4-31b-it")
        for size, path in checkpoints:
            with self.subTest(size=size):
                if not path.is_dir():
                    continue
                config = inspect_text_checkpoint(path).as_runtime_dict()
                self.assertEqual(profile.model_profile_for_config(config)["model"]["name"],
                                 f"gemma-4-{size}b-it-text")


if __name__ == "__main__":
    unittest.main()
