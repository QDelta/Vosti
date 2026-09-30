from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from scripts.determinism_tests.sglang_worker import _backend_evidence, _cuda_graph_evidence, _cuda_graph_kwargs


class GraphConfigurationTests(unittest.TestCase):
    def test_resolved_settings_not_raw_auto_inputs_are_recorded(self):
        resolved = dict(attention_backend="fa3", sampling_backend="flashinfer",
                        enable_deterministic_inference=False, disable_cuda_graph=False,
                        disable_radix_cache=False,
                        cuda_graph_config={"decode": {"backend": "full", "max_bs": 16}})
        args = SimpleNamespace(attention_backend=None, cuda_graph_config=None,
                               resolved_dict=lambda: resolved)
        self.assertEqual(_backend_evidence(args), resolved)

    def test_old_and_new_api_keep_the_same_decode_limit(self):
        @dataclass
        class Old:
            cuda_graph_max_bs: int = 160

        @dataclass
        class New:
            cuda_graph_max_bs_decode: int = 160

        self.assertEqual(_cuda_graph_kwargs(Old, 16), {"cuda_graph_max_bs": 16})
        self.assertEqual(_cuda_graph_kwargs(New, 16), {"cuda_graph_max_bs_decode": 16})
        self.assertEqual(_cuda_graph_evidence(Old(16)), {"cuda_graph_max_bs": 16})
        self.assertEqual(_cuda_graph_evidence(SimpleNamespace(cuda_graph_config=New(16))),
                         {"cuda_graph_config": {"cuda_graph_max_bs_decode": 16}})

    def test_unrecognized_api_fails_instead_of_silently_ignoring_limit(self):
        class Unknown:
            def __init__(self):
                pass

        with self.assertRaisesRegex(RuntimeError, "no recognized"):
            _cuda_graph_kwargs(Unknown, 16)
