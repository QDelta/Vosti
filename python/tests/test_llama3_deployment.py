import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.deployment.model_families import llama3 as llama3_deployment_bundle
from vosti_kernels.model_families.llama3.profile import (
    model_config,
    model_profile_for_name,
)


class Llama3DeploymentTests(unittest.TestCase):
    def test_candidate_checkpoint_projection_uses_exact_profile(self) -> None:
        for name in ("llama-3.1-8b", "llama-3.2-3b", "llama-3.3-70b"):
            with self.subTest(profile=name):
                profile = model_profile_for_name(name)
                resolved = model_config(profile)

                class ParsedConfig:
                    def as_runtime_dict(self) -> dict:
                        return resolved

                with tempfile.TemporaryDirectory() as directory, mock.patch(
                    "scripts.deployment.model_families.llama3.inspect_text_checkpoint",
                    return_value=ParsedConfig(),
                ):
                    path = Path(directory)
                    raw = json.dumps(resolved).encode()
                    (path / "config.json").write_bytes(raw)
                    actual, selected, config_sha256 = (
                        llama3_deployment_bundle.CANDIDATE_ARCHITECTURE.checkpoint_profile(
                            path
                        )
                    )

                self.assertEqual(actual, resolved)
                self.assertEqual(selected["model"]["name"], name)
                self.assertEqual(
                    config_sha256,
                    hashlib.sha256(raw).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
