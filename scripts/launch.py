#!/usr/bin/env python3
"""Launch a supported Engine example or OpenAI server in the locked environment."""

import argparse
import os
from pathlib import Path
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.model_paths import supported_families


def launch_command(kind: str, family: str, args: list[str]) -> list[str]:
    if kind not in ("engine", "server") or family not in supported_families():
        raise ValueError("unsupported example kind or model family")
    command = ["uv", "run", "--locked", "cargo", "run"]
    if kind == "server":
        command += ["--release", "--features", "openai-server"]
    command += ["--example", f"verus_{kind}_{family}"]
    if args:
        command += ["--", *args]
    return command


def launch_environment(environment: dict[str, str]) -> dict[str, str]:
    result = environment.copy()
    libdir = sysconfig.get_config_var("LIBDIR")
    if not libdir:
        raise RuntimeError("the selected Python does not report its library directory")
    python_paths = os.pathsep.join((str(ROOT), str(ROOT / "python")))
    for key, prefix in (("LD_LIBRARY_PATH", libdir), ("PYTHONPATH", python_paths)):
        result[key] = prefix + (os.pathsep + result[key] if result.get(key) else "")
    result["VOSTI_FRAMEWORK_ROOT"] = str(ROOT)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("engine", "server"))
    parser.add_argument("--family", required=True, choices=supported_families())
    parser.add_argument("args", nargs=argparse.REMAINDER, help="example arguments after --")
    args = parser.parse_args(argv)
    forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
    command = launch_command(args.kind, args.family, forwarded)
    environment = launch_environment(os.environ)
    os.chdir(ROOT)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
