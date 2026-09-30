import json
from pathlib import Path
import tempfile
import unittest

from vosti_kernels.model_families.qwen3.loader import inspect_text_checkpoint
from vosti_kernels.model_families.qwen3.profile import (
    model_config,
    model_profile_for_name,
    model_profiles,
)


class Qwen3CheckpointTests(unittest.TestCase):
    def test_checkpoint_inspection_accepts_registered_profile(self) -> None:
        name = model_profiles()[0]["model"]["name"]
        expected = model_config(model_profile_for_name(name))
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            (model_path / "config.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )
            actual = inspect_text_checkpoint(model_path)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
