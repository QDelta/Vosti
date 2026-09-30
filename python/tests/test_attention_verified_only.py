"""Attention never substitutes an unverified implementation for a missing kernel."""
import types
import unittest
from unittest import mock

import torch

from vosti_kernels import kernels


class VerifiedOnlyAttentionTests(unittest.TestCase):
    def setUp(self):
        q = torch.zeros((1, 2, 8))
        cache = torch.zeros((1, 64, 1, 8))
        lengths = torch.tensor([0, 1], dtype=torch.int32)
        self.args = (q, cache, cache.clone(), lengths, lengths, 1, 1,
                     torch.zeros((1, 1), dtype=torch.int32))
        self.wrappers = (kernels.paged_attention,
                         kernels.paged_attention_from_verified_caller)

    def test_no_runtime_is_rejected(self):
        for wrapper in self.wrappers:
            with self.subTest(wrapper=wrapper.__name__):
                with self.assertRaisesRegex(RuntimeError, "explicit family runtime"):
                    wrapper(*self.args)

    def test_no_kernel_binding_is_rejected(self):
        runtime = types.SimpleNamespace(runtime_config=lambda: {"head_dim": 8},
                                        verified_for=lambda tensor: None)
        for wrapper in self.wrappers:
            with self.subTest(wrapper=wrapper.__name__):
                with self.assertRaisesRegex(RuntimeError, "verified kernel binding"):
                    wrapper(*self.args, runtime=runtime)

    def test_kernel_failure_propagates_without_sdpa(self):
        failure = RuntimeError("verified launch failed")
        launch = mock.Mock(side_effect=failure)
        runtime = types.SimpleNamespace(
            runtime_config=lambda: {"head_dim": 8},
            verified_for=lambda tensor: object(),
            static_launch_config=lambda wrapper, key: {"sealed": 1},
            kernel_entrypoint=lambda namespace, wrapper: launch,
        )
        for wrapper in self.wrappers:
            with self.subTest(wrapper=wrapper.__name__), mock.patch.object(
                kernels, "_validate_paged_attention_runtime_contract"
            ), mock.patch.object(torch.nn.functional, "scaled_dot_product_attention") as sdpa:
                with self.assertRaises(RuntimeError) as raised:
                    wrapper(*self.args, runtime=runtime)
                self.assertIs(raised.exception, failure)
                sdpa.assert_not_called()


if __name__ == "__main__":
    unittest.main()
