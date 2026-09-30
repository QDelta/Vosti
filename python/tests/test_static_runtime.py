"""Immutable binding tests independent of any family or device."""

from copy import deepcopy
import importlib
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import pytest
import torch

from vosti_kernels.static_runtime import freeze_launch_inventory


@pytest.mark.parametrize('family', ['qwen3', 'llama3', 'gemma3', 'gemma4'])
def test_all_families_attest_origins_without_gpu_work(family, tmp_path):
    root = Path(__file__).resolve().parents[2]
    runtime = importlib.import_module(f'vosti_kernels.model_families.{family}.runtime')
    profile = importlib.import_module(f'vosti_kernels.model_families.{family}.profile')
    config = profile.model_config(profile.model_profiles()[0])
    with patch('torch.cuda._lazy_init', side_effect=AssertionError('unexpected GPU work')):
        staged = runtime.load_runtime(config, device='cuda:0', framework_root=root,
                                      kernel_root=root / 'kernels')
        report = staged.report()
        assert report['module_origins']['vosti_kernels.kernel_catalog'] == str(
            root / 'python/vosti_kernels/kernel_catalog.json')
        assert report['deployment_sha256'] is None
        with pytest.raises((RuntimeError, ValueError)):
            runtime.QualifiedRuntime(staged)
        with pytest.raises(RuntimeError, match='loaded from'):
            runtime.load_runtime(config, device='cuda:0', framework_root=tmp_path,
                                 kernel_root=root / 'kernels')


def test_common_admission_checks_bundle_digest_and_environment(tmp_path):
    from vosti_kernels.static_runtime import admit_runtime

    deployment = Mock()
    deployment.load_bundle.return_value = {'candidate': 'sealed'}
    deployment.validate_runtime_binding.return_value = {'deployment_sha256': 'checked'}
    kwargs = dict(family_module='test', scope={'architecture': 'test'}, config={'model': 1},
                  deployment=deployment, deployment_bundle='bundle', device='cuda:0',
                  environment={'backend': 'observed'}, kernel_root=tmp_path)
    with patch('vosti_kernels.static_runtime.attest_framework_package', return_value=('root', {}, {})), \
         patch('vosti_kernels.kernel_modules.load_attested_modules', return_value=({}, {}, {})):
        with pytest.raises(ValueError, match='model config digest'):
            admit_runtime(**kwargs)
        deployment.validate_runtime_binding.assert_not_called()
        admitted = admit_runtime(**kwargs, model_config_sha256='config-digest')
        deployment.validate_runtime_binding.assert_called_once_with(
            {'candidate': 'sealed'}, resolved_config={'model': 1},
            model_config_sha256='config-digest', environment={'backend': 'observed'})
        assert admitted.qualification == {'deployment_sha256': 'checked'}
        assert admitted.dtype == torch.bfloat16


class StaticLaunchBindingTests(unittest.TestCase):
    def launches(self):
        return [
            {"wrapper": "linear", "key": {"n": 8, "k": 4},
             "sites": ["a", "b"], "config": {"BLOCK_M": 16}},
            {"wrapper": "linear", "key": {"n": 4, "k": 8},
             "sites": ["c"], "config": {"BLOCK_M": 32}},
        ]

    def test_maps_are_detached_immutable_and_share_bound_configs(self):
        launches = self.launches()
        keyed, sites = freeze_launch_inventory(launches)
        identity = ("linear", (("k", 4), ("n", 8)))
        self.assertIs(keyed[identity], sites["a"])
        self.assertIs(sites["a"], sites["b"])
        launches[0]["config"]["BLOCK_M"] = 64
        launches[0]["key"]["k"] = 99
        launches[0]["sites"].append("d")
        self.assertEqual(dict(keyed[identity]), {"BLOCK_M": 16})
        self.assertNotIn("d", sites)
        for mapping, key in ((keyed, identity), (sites, "a"), (sites["a"], "BLOCK_M")):
            with self.subTest(key=key), self.assertRaises(TypeError):
                mapping[key] = None

    def test_empty_duplicate_keys_and_duplicate_sites_are_rejected(self):
        duplicates = self.launches()
        duplicates.append(deepcopy(duplicates[0]))
        duplicate_sites = self.launches()
        duplicate_sites[1]["sites"].append("a")
        for launches in ([], duplicates, duplicate_sites):
            with self.subTest(launches=launches), self.assertRaises(RuntimeError):
                freeze_launch_inventory(launches)


if __name__ == "__main__":
    unittest.main()
