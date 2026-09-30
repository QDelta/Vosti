"""Build only an external benchmark driver against a clean frozen Engine."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts.serving_benchmark.campaign import source_identity, file_hash
from scripts.serving_benchmark.multi_turn import write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-dir', type=Path)
    a = p.parse_args()
    frozen, out = a.frozen.resolve(), a.output.resolve()
    identity = source_identity(frozen)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('native.rs')
    shutil.copyfile(source, out/'main.rs')
    # Generated build artifact, not a modification of the frozen crate/lockfile.
    manifest = ('[package]\nname="vosti-aligned-driver"\nversion="0.1.0"\nedition="2021"\n'
        '[[bin]]\nname="vosti-aligned-driver"\npath="main.rs"\n[dependencies]\n'
        f'vosti-verus={{path={json.dumps(str(frozen))},features=["openai-server"]}}\n'
        'pyo3={version="0.22",features=["auto-initialize"]}\nserde_json="1"\n'
        'vstd="=0.0.0-2026-08-23-0033"\n')
    (out/'Cargo.toml').write_text(manifest)
    shutil.copyfile(frozen/'Cargo.lock', out/'Cargo.lock')
    env = dict(os.environ, PYO3_PYTHON=str(frozen/'.venv/bin/python'),
        VOSTI_ALIGNED_ENGINE_SETUP=str(frozen/'examples/support/engine_setup.rs'))
    target = a.target_dir.resolve() if a.target_dir else out/'target'
    with (out/'build.log').open('x') as log:
        subprocess.run(['cargo','build','--offline','--release','--manifest-path',str(out/'Cargo.toml'),
                        '--target-dir',str(target)],
            env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy2(target/'release/vosti-aligned-driver',out/'vosti-aligned-driver')
    assert source_identity(frozen) == identity
    write_new(out/'build.json',dict(source=identity, driver_sha256=file_hash(source),
        engine_setup_sha256=file_hash(frozen/'examples/support/engine_setup.rs'),
        binary=str(out/'vosti-aligned-driver'),
        binary_sha256=file_hash(out/'vosti-aligned-driver')))


if __name__ == '__main__':
    main()
