"""Kernel source checks bind relevant bytes without requiring Git metadata."""
import json
from pathlib import Path
import shutil

import pytest

from scripts.audit.check_kernel_sources import ROOT, resolve_kernel_sources


@pytest.fixture
def source_export(tmp_path):
    families = Path('python/vosti_kernels/model_families')
    for path in (ROOT / families).glob('*/scope.json'):
        dest = tmp_path / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True)
        shutil.copyfile(path, dest)
    shutil.copyfile(ROOT / 'python/vosti_kernels/kernel_catalog.json',
                    tmp_path / 'python/vosti_kernels/kernel_catalog.json')
    shutil.copytree(ROOT / 'kernels/triton_kernels', tmp_path / 'kernels/triton_kernels',
                    ignore=shutil.ignore_patterns('__pycache__'))
    return tmp_path


def test_export_without_git_matches_checkout(source_export):
    assert not (source_export / '.git').exists()
    assert resolve_kernel_sources(source_export)[1] == resolve_kernel_sources()[1]


def test_unrelated_files_do_not_invalidate_sources(source_export):
    expected = resolve_kernel_sources(source_export)[1]
    (source_export / 'kernels/README.md').write_text('Documentation edit')
    (source_export / 'kernels/experiment.py').write_text('print("unrelated experiment")')
    assert resolve_kernel_sources(source_export)[1] == expected


@pytest.mark.parametrize('name', ['matmul.py', 'runtime_contracts.py', 'constants.py', '__init__.py'])
def test_changed_kernel_or_support_bytes_fail(source_export, name):
    path = source_export / 'kernels/triton_kernels' / name
    path.write_text(path.read_text() + '\n# changed\n')
    with pytest.raises(RuntimeError, match='differs from its scope digest'):
        resolve_kernel_sources(source_export)


def test_each_family_scope_is_checked(source_export):
    path = source_export / 'python/vosti_kernels/model_families/gemma4/scope.json'
    scope = json.loads(path.read_text())
    scope['kernel_contracts'][0]['kernel'] = 'unknown_kernel'
    path.write_text(json.dumps(scope))
    with pytest.raises(RuntimeError, match='unknown kernel catalog identity'):
        resolve_kernel_sources(source_export)


def test_missing_support_binding_and_traversal_fail(source_export):
    path = source_export / 'python/vosti_kernels/kernel_catalog.json'
    original = json.loads(path.read_text())
    changed = dict(original, runtime_support_sources=[])
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match='scope must bind'):
        resolve_kernel_sources(source_export)
    original['kernels']['matmul_kernel']['source'] = '../outside.py'
    path.write_text(json.dumps(original))
    with pytest.raises(RuntimeError, match='invalid kernel source'):
        resolve_kernel_sources(source_export)


def test_alternate_directory_checks_same_sources(source_export, tmp_path):
    alternative = tmp_path / 'alternate'
    shutil.copytree(source_export / 'kernels', alternative)
    assert resolve_kernel_sources(source_export, alternative) == (
        alternative.resolve(), resolve_kernel_sources(source_export)[1])


def test_family_cannot_override_shared_identity(source_export):
    path = source_export / 'python/vosti_kernels/model_families/qwen3/scope.json'
    scope = json.loads(path.read_text())
    scope['kernel_contracts'][0]['source_sha256'] = '0' * 64
    path.write_text(json.dumps(scope))
    with pytest.raises(RuntimeError, match='must not override'):
        resolve_kernel_sources(source_export)


def test_expansion_is_detached_and_keeps_family_policy(source_export):
    from vosti_kernels.model_profile import load_scope

    path = source_export / 'python/vosti_kernels/model_families/qwen3/scope.json'
    raw = json.loads(path.read_text())
    expanded = load_scope(path)
    for declared, contract in zip(raw['kernel_contracts'], expanded['kernel_contracts']):
        assert all(contract[key] == value for key, value in declared.items())
        assert {'source', 'source_sha256', 'module', 'entrypoint'} <= contract.keys()
    expanded['kernel_contracts'][0]['source_sha256'] = 'mutated'
    assert load_scope(path)['kernel_contracts'][0]['source_sha256'] != 'mutated'
