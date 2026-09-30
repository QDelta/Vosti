"""CPU regression checks for offline benchmark reference configurations."""
import unittest

import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.attention import deployed_configs


class AttentionScreenTests(unittest.TestCase):
    def test_full_and_sliding_references_follow_their_own_selectors(self):
        configs = deployed_configs(128, 1024)
        self.assertEqual(configs['full']['num_warps'], 4)
        self.assertEqual(configs['sliding']['num_warps'], 2)
        self.assertEqual(configs['full']['D_HEAD'], 128)
        self.assertEqual(configs['sliding']['D_HEAD'], 128)
        configs = deployed_configs(256, 1024)
        self.assertEqual(configs['full'], configs['sliding'])


@pytest.mark.parametrize("kernel", ("attention", "matmul"))
def test_kernel_screen_help_outside_checkout_without_pythonpath(kernel, tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(root / f"kernels/benchmarks/{kernel}.py"), "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert "--output" in result.stdout


if __name__ == '__main__':
    unittest.main()
