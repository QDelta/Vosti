"""Reject obsolete project identifiers and personal paths in public source."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OLD_NAME = re.compile('vv' + 'llm' + '|mlsys' + '[-_]verif', re.IGNORECASE)
PERSONAL_PATH = re.compile(r'/(?:home|Users)/[^/\s]+/')


def naming_errors(root: Path = ROOT) -> list[str]:
    errors = []
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    for relative in filter(None, paths):
        if OLD_NAME.search(relative):
            errors.append(f'obsolete tracked path: {relative}')
        try:
            content = (root / relative).read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if OLD_NAME.search(line):
                errors.append(f'obsolete identifier: {relative}:{number}')
            if PERSONAL_PATH.search(line):
                errors.append(f'personal filesystem path: {relative}:{number}')
    return errors


if __name__ == '__main__':
    errors = naming_errors()
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Vosti project naming check passed')
