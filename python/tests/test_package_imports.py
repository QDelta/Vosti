"""CPU metadata imports must not shadow packages or load the tensor runtime."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_audit_keeps_kernel_namespace_without_site_packages():
    code = '''
import importlib.util
import sys
before = importlib.util.find_spec("kernels")
import scripts.audit.check_kernel_sources
after = importlib.util.find_spec("kernels")
assert before.origin == after.origin == None
assert list(before.submodule_search_locations) == list(after.submodule_search_locations)
assert "torch" not in sys.modules
assert "triton" not in sys.modules
'''
    subprocess.run([sys.executable, "-B", "-S", "-c", code], cwd=ROOT,
                   env={**os.environ, "PYTHONPATH": str(ROOT)}, check=True)


def test_public_tensor_exports_keep_identity():
    code = '''
import vosti_kernels
from vosti_kernels import kernels
for name in vosti_kernels.__all__:
    assert name in dir(vosti_kernels)
    assert getattr(vosti_kernels, name) is getattr(kernels, name)
'''
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "python")}, check=True)
