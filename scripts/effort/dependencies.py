"""Source-pinned dependency identities extracted from typed Verus VIR.

The compiler resolves paths, reexports and method receivers. Focused
extraction must never be presented as full-project proof success.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit.check_verus_trust_manifest import _ITEM_START, _paren_delta

DESTINATION = ROOT / 'target/effort/effort_dependencies.json'
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|[()]|[^\s()]+')
PROJECT_PATH = re.compile(r'\bvosti_verus::[^\s()"]+')
RESOLVED_REFERENCE = re.compile(r'\(\s*(?:Fun\s+:path|Dt\s+Path)\s+(vosti_verus::[^\s()"]+)\s*\)')
MODULE_METADATA = {':owning_module', ':visibility', ':transparency',
                   ':body_visibility', ':opaqueness'}


def resolved_references(fields, mode):
    references = set()
    for key, value in fields.items():
        if key == ':name' or (key == ':body' and mode != 'Spec'):
            continue
        # String literals (including source spans) cannot create references.
        value = ' '.join(m.group() for m in TOKEN.finditer(value)
                         if not m.group().startswith('"'))
        references.update(RESOLVED_REFERENCE.findall(value))
        rest = RESOLVED_REFERENCE.sub('', value)
        rest = re.sub(r'\(\s*Visibility\s+:restricted_to\s+[^\s()]+\s*\)', '', rest)
        remaining = set(PROJECT_PATH.findall(rest))
        if key in MODULE_METADATA:
            continue
        if key == ':kind':
            remaining = {name for name in remaining if not re.search(r'::impl&%\d+$', name)}
        if remaining:
            raise ValueError(f'unrecognized project reference encoding in {key}: {sorted(remaining)}')
    return references


def source_hashes(root):
    paths = subprocess.check_output(
        ['git', 'ls-files', 'src', 'Cargo.toml', 'Cargo.lock', 'build.rs',
         'rust-toolchain*', '.cargo'], cwd=root, text=True).splitlines()
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in paths}


def extractor_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def top_fields(text):
    """Read only the named fields of the outer Function/Datatype node."""
    depth, start, name, fields = 0, None, None, {}
    for match in TOKEN.finditer(text):
        token = match.group()
        if depth == 2 and (token.startswith(':') or token == ')'):
            if name is not None:
                fields[name] = text[start:match.start()].strip()
                name = None
            if token.startswith(':'):
                name, start = token, match.end()
        if token == '(':
            depth += 1
        elif token == ')':
            depth -= 1
    if depth or name is not None:
        raise ValueError('malformed typed VIR declaration')
    return fields


def parse_declaration(text, path, line, kind):
    fields = top_fields(text)
    names = PROJECT_PATH.findall(fields.get(':name', ''))
    if len(names) != 1:
        raise ValueError(f'expected one resolved project identity at {path}:{line}')
    symbol = names[0]
    mode = fields.get(':mode')
    if kind == 'Function' and mode not in ('Exec', 'Proof', 'Spec'):
        raise ValueError(f'unsupported function mode at {path}:{line}: {mode}')
    if kind == 'Function' and ':body' not in fields:
        raise ValueError(f'missing body field at {path}:{line}')
    # A spec definition's body determines its meaning. An executable/proof
    # body's implementation is NOT part of the theorem statement's dependency
    # graph. In particular, calls to helper lemmas cannot seed main specs.
    references = sorted(resolved_references(fields, mode) - {symbol})
    return dict(symbol=symbol, path=path, line=line, kind=kind,
                mode=mode, references=references)


def extract(vir, root):
    records, pieces, current, depth = {}, [], None, 0
    with vir.open() as stream:
        for line in stream:
            if current is None:
                match = _ITEM_START.match(line)
                if match is None:
                    continue
                path = Path(match['path'])
                if path.is_absolute():
                    if not path.is_relative_to(root):
                        continue
                    path = path.relative_to(root)
                if not path.as_posix().startswith('src/'):
                    continue
                current = (path.as_posix(), int(match['line']), match['kind'])
                pieces = []
            pieces.append(line)
            depth += _paren_delta(line)
            if depth == 0:
                record = parse_declaration(''.join(pieces), *current)
                if record['symbol'] in records:
                    raise ValueError(f"duplicate compiler identity: {record['symbol']}")
                records[record['symbol']] = record
                current = None
    if current is not None or not records:
        raise ValueError('incomplete or empty project VIR')
    missing = {ref for record in records.values() for ref in record['references']
               if ref not in records}
    if missing:
        raise ValueError(f'unresolved project function/type identities: {sorted(missing)}')
    return dict(sorted(records.items()))


def load_dependencies(root):
    path = root / 'target/effort/effort_dependencies.json'
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError('missing typed dependencies; run python3 scripts/effort/dependencies.py') from error
    if (data.get('schema') != 'vosti.effort-dependencies.v1'
            or data['source_sha256'] != source_hashes(root)
            or data.get('extractor_sha256') != extractor_hash()):
        raise ValueError('typed specification dependencies are stale; run python3 scripts/effort/dependencies.py')
    return data


def main():
    target = ROOT / 'target/effort-compiler'
    target.mkdir(parents=True, exist_ok=True)
    before = source_hashes(ROOT)
    env = dict(os.environ)
    verus = Path(os.environ.get('CARGO_VERUS', str(Path.home() / '.local/verus/cargo-verus')))
    env['PATH'] = str(verus.parent) + os.pathsep + env['PATH']
    env.setdefault('PYO3_PYTHON', str(ROOT / '.venv/bin/python'))
    # Focus uses a separate partial-verification cache. Keep its build and
    # Verus logs separate: Verus clears its log directory before extraction.
    cache = target / 'build/verus-partial'
    if (cache / 'CACHEDIR.TAG').exists():
        subprocess.run(['cargo', 'clean', '-p', 'vosti-verus', '--target-dir', str(cache)],
                       cwd=ROOT, env=env, check=True)
    vir = target / 'vir/crate-simple.vir'
    vir.unlink(missing_ok=True)
    with (target / 'extraction.log').open('w') as log:
        subprocess.run([str(verus), 'focus', '--locked', '--offline',
                        '--fwd-verus-args-to', 'roots',
                        '--target-dir', str(target / 'build'), '--',
                        '--verify-only-module', 'boundary::scalar',
                        '--log', 'vir-simple', '--log-dir', str(vir.parent)],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    declarations = extract(vir, ROOT)
    if source_hashes(ROOT) != before:
        raise ValueError('Rust sources changed during compiler extraction')
    data = dict(schema='vosti.effort-dependencies.v1', source_sha256=before,
                extractor_sha256=extractor_hash(),
                compiler=subprocess.check_output([str(verus.parent / 'verus'), '--version'],
                                                 text=True).strip(),
                proof_checked=False, declarations=declarations)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(data, indent=2) + '\n')
    print(f'Extracted {len(declarations)} compiler-resolved declarations; no proof claim.')


if __name__ == '__main__':
    main()
