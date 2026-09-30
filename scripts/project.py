#!/usr/bin/env python3
"""Sequential project gates behind the root Makefile; see scripts/README.md."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Gates:
    def __init__(self):
        self.env = os.environ.copy()
        self.verus = self.env.get("VERUS", str(Path.home() / ".local/verus/verus"))
        self.cargo_verus = self.env.get("CARGO_VERUS", str(Path.home() / ".local/verus/cargo-verus"))
        self.env["VERUS"] = self.verus
        self.env.setdefault("PYO3_PYTHON", sys.executable)
        self.env["PYTHONPATH"] = os.pathsep.join(map(str, (
            ROOT, ROOT / "python", ROOT / "kernels", ROOT / "scripts",
        ))) + os.pathsep + self.env.get("PYTHONPATH", "")
        self.env["LD_LIBRARY_PATH"] = str(sysconfig.get_config_var("LIBDIR") or "") + os.pathsep + self.env.get("LD_LIBRARY_PATH", "")
        executable = shutil.which(self.verus)
        if executable:
            self.env["PATH"] = str(Path(executable).absolute().parent) + os.pathsep + self.env.get("PATH", "")
        self.env["VOSTI_FRAMEWORK_ROOT"] = str(ROOT)
        self.env["VOSTI_KERNEL_ROOT"] = self.env.get("KERNELS_DIR") or str(ROOT / "kernels")
        self.target = Path(self.env.get("VERUS_CLAIM_TARGET_DIR") or ROOT / "target/verus-claim").absolute()
        self.vir_dir = Path(self.env.get("VERUS_CLAIM_VIR_LOG_DIR") or self.target / "trust-vir").absolute()

    def run(self, *args, cwd=ROOT):
        command = list(map(str, args))
        print(f"+ {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=cwd, env=self.env, check=True)

    def python(self, script, *args, cwd=ROOT):
        self.run(sys.executable, ROOT / script, *args, cwd=cwd)

    def require_verus(self):
        if not shutil.which(self.verus):
            raise ValueError("Set VERUS to an installed verifier; compiler-backed checks must not skip")

    def architecture(self):
        self.python("scripts/audit/check_project_name.py")
        self.python("scripts/audit/check_model_architecture_boundary.py")

    def kernel_sources(self):
        self.python("scripts/audit/check_kernel_sources.py", "--kernel-root", self.env["VOSTI_KERNEL_ROOT"])

    def verify_engine(self):
        self.require_verus()
        self.architecture()
        if self.target.resolve() in {ROOT, Path.home(), Path("/")}:
            raise ValueError("VERUS_CLAIM_TARGET_DIR must be a dedicated build directory")
        self.vir_dir.mkdir(parents=True, exist_ok=True)
        self.target.mkdir(parents=True, exist_ok=True)
        # Standard cache-directory magic, not a source hash or proof assumption.
        (self.target / "CACHEDIR.TAG").write_text(
            "Signature: 8a477f597d28d172789f06886806bc55\n"
            "# Disposable Cargo build cache; https://bford.info/cachedir/\n"
        )
        # cargo-verus can omit forwarded flags from Cargo's fingerprint. Clear
        # this package, not its dependencies; never accept a stale focused VIR.
        self.run("cargo", "clean", "-p", "vosti-verus", "--target-dir", self.target)
        vir = self.vir_dir / "crate-simple.vir"
        vir.unlink(missing_ok=True)
        self.run(self.cargo_verus, "verify", "--fwd-verus-args-to", "roots",
                 "--target-dir", self.target, "--", "--log", "vir-simple",
                 "--log-dir", self.vir_dir)
        if not vir.is_file() or not vir.stat().st_size:
            raise ValueError("Verus did not emit a fresh, nonempty crate-simple.vir")
        self.python("scripts/audit/check_verus_trust_manifest.py", vir)

    def examples(self):
        self.require_verus()
        self.run(self.cargo_verus, "verify", "--target-dir", self.target, "--examples")

    def verify_kernels(self):
        self.kernel_sources()
        self.python("scripts/verification/verify_kernel_contracts.py")

    def verify(self):
        self.verify_engine()
        self.verify_kernels()

    def test(self, suite="all"):
        # CPU gates must never consume a device simply because one is visible.
        self.env["CUDA_VISIBLE_DEVICES"] = ""
        if suite != "checkpoint":
            self.require_verus()
        if suite in {"all", "rust"}:
            self.run("cargo", "test", "--locked", "--features", "openai-server",
                     "--lib", "--tests", "--", "--nocapture")
            self.run("cargo", "build", "--locked", "--features", "openai-server", "--examples")
        if suite in {"all", "python"}:
            self.python("scripts/effort/dependencies.py")
            self.run(sys.executable, "-m", "pytest", "-q", "python/tests")
        if suite in {"all", "kernels"}:
            self.kernel_sources()
            self.run(sys.executable, "-m", "pytest", "-q", "tests", cwd=ROOT / "kernels")
        if suite == "checkpoint":
            self.run(sys.executable, "-m", "unittest", "discover", "-s",
                     "python/integration", "-p", "test_*.py", "-v")

    def diagnostics(self):
        self.kernel_sources()
        # Independently scoped evidence, not premises of the relational proof.
        race_drivers = sorted((ROOT / "scripts/verification/diagnostics").glob("verify_*_kernel_races.py"))
        if not race_drivers:
            raise ValueError("no independent logical race diagnostics found")
        for driver in race_drivers:
            self.python(driver, cwd=ROOT / "kernels")
        self.python("kernels/scripts/verify_suffix_independence.py", cwd=ROOT / "kernels")

    def check(self):
        self.verify()
        self.examples()
        self.test()
        self.python("scripts/audit/claim_ledger.py")
        self.diagnostics()
        self.python("scripts/audit/tcb.py", "--check")

    def generate(self):
        self.kernel_sources()
        self.python("scripts/verification/verify_kernel_contracts.py", "--write-generated")
        self.python("scripts/audit/claim_ledger.py", "--write")

    def test_gpu(self):
        # Import the existing family catalog without importing Torch or kernels.
        from scripts.checks.engine import FAMILIES

        family = self.env.get("FAMILY", "")
        if family not in FAMILIES:
            raise ValueError(f"FAMILY must be one of {', '.join(sorted(FAMILIES))}")
        for key in ("CUDA_VISIBLE_DEVICES", "MODEL_PATH", "VOSTI_DEPLOYMENT_BUNDLE"):
            if not self.env.get(key, "").strip() or self.env[key] == "-1":
                raise ValueError(f"{key} is required for test-gpu")
        for key in ("MODEL_PATH", "VOSTI_DEPLOYMENT_BUNDLE"):
            path = Path(self.env[key]).expanduser().resolve()
            if not path.is_dir():
                raise ValueError(f"{key} must name an existing directory: {path}")
            self.env[key] = str(path)
        self.kernel_sources()
        for script in ("kv_store_effect.py", "engine.py", "reference.py"):
            self.python(f"scripts/checks/{script}", family)
        self.python("scripts/checks/causal_confinement.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "verify-engine", "verify-kernels",
        "test", "test-gpu", "check", "generate", "architecture", "examples", "diagnostics"))
    parser.add_argument("--suite", choices=("all", "rust", "python", "kernels", "checkpoint"))
    args = parser.parse_args()
    if args.suite is not None and args.command != "test":
        parser.error("--suite is only valid with test")
    gates = Gates()
    if args.command != "test-gpu":
        gates.env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        if args.command == "test":
            gates.test(args.suite or "all")
        else:
            getattr(gates, args.command.replace("-", "_"))()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"project gate failed: {error}\n")


if __name__ == "__main__":
    main()
