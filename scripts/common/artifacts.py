"""Shared experiment metadata and atomic progress-file updates."""

import json
from pathlib import Path
import subprocess


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def package_inventory(python):
    code = ('import importlib.metadata as m,json; '
            'print(json.dumps(sorted((d.metadata["Name"],d.version) '
            'for d in m.distributions())))')
    return json.loads(subprocess.check_output([str(python), '-c', code], text=True))


def shared_helper_sources():
    """Include shared implementations in retained benchmark harness snapshots."""
    return sorted(Path(__file__).resolve().parent.rglob('*.py'))
