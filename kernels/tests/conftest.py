"""Make repository-relative kernel fixtures independent of pytest's caller cwd."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionstart(session) -> None:
    del session
    os.chdir(REPOSITORY_ROOT)
