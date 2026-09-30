"""CPU capability/dispatch checks, not CUDA or full-model qualification."""

import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from vosti_kernels.model_families.gemma4 import profile, runtime
from vosti_kernels.model_families.gemma4.loader import FULL_ATTENTION, SLIDING_ATTENTION
from vosti_kernels.static_runtime import StaticPrimitiveRuntime


class Gemma4RuntimeTests(unittest.TestCase):
    def make_runtime(self, modules=None):
        return runtime.Runtime(modules or {}, {}, {}, "kernels", None,
            profile.model_profile_for_name("gemma-4-31b-it-text"), "cuda:3", torch.bfloat16)

    def test_source_attestation_does_not_grant_engine_or_backend_authority(self):
        value = runtime.load_runtime(runtime.config_for_profile("gemma-4-31b-it-text"))
        report = value.report()
        self.assertIsInstance(value, StaticPrimitiveRuntime)
        self.assertEqual(report["architecture"], "gemma4_text")
        self.assertEqual(len(report["launches"]), 23)
        self.assertFalse(report["backend_qualified"])
        self.assertFalse(report["engine_reachable"])
        self.assertIsNone(report["deployment_sha256"])
        with self.assertRaisesRegex(ValueError, "sealed backend bundle"):
            runtime.QualifiedRuntime(value)

    def test_configuration_and_launches_are_detached_and_read_only(self):
        value = self.make_runtime()
        config = value.model_config()
        config["layer_types"][0] = "changed"
        self.assertEqual(value.model_config()["layer_types"][0], SLIDING_ATTENTION)
        with self.assertRaises(TypeError):
            value.runtime_config()["layer_types"][0] = "changed"
        report = value.report()
        report["launches"][0]["config"]["D"] = 1
        self.assertNotEqual(value.report()["launches"], report["launches"])
        with self.assertRaises(TypeError):
            value._config_for("global.q_norm")["D"] = 1
        with self.assertRaisesRegex(RuntimeError, "no static launch"):
            value._config_for("runtime_selected_site")

    def test_load_rejects_wrong_geometry_device_and_dtype(self):
        config = runtime.config_for_profile("gemma-4-31b-it-text")
        with self.assertRaises(ValueError):
            runtime.load_runtime({**config, "num_global_key_value_heads": 8})
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            runtime.load_runtime(config, device="cpu")
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            runtime.load_runtime(config, dtype=torch.float32)

    def test_qkv_norm_dispatch_uses_each_kind_geometry(self):
        norm = mock.Mock(side_effect=lambda x, *args, **kwargs: x.clone())
        value = self.make_runtime({"qk_norm": SimpleNamespace(head_rms_norm=norm)})
        for kind, heads, width in ((SLIDING_ATTENTION, 16, 256), (FULL_ATTENTION, 4, 512)):
            norm.reset_mock()
            q = torch.zeros(2, 32 * width)
            kv = torch.zeros(2, heads * width)
            q_weight, k_weight = torch.ones(width), torch.ones(width)
            q_out, k_out = value.qk_norm(q, kv, q_weight, k_weight, attention_kind=kind)
            v_out = value.value_norm(kv, attention_kind=kind)
            self.assertEqual(q_out.shape, q.shape)
            self.assertEqual(k_out.shape, kv.shape)
            self.assertEqual(v_out.shape, (2, heads, width))
            self.assertEqual([call.args[2] for call in norm.call_args_list], [32, heads, heads])
            self.assertTrue(torch.equal(norm.call_args_list[2].args[1], torch.ones(width)))
            self.assertEqual([call.kwargs["launch_config"]["D"] for call in norm.call_args_list], [width] * 3)
        with self.assertRaisesRegex(ValueError, "static layer geometry"):
            value.value_norm(torch.zeros(2, 16 * 256), attention_kind=FULL_ATTENTION)
        with self.assertRaisesRegex(ValueError, "unsupported attention kind"):
            value.value_norm(kv, attention_kind="runtime_kind")

    def test_rope_dispatch_and_unrotated_global_pairs(self):
        rope = mock.Mock(side_effect=lambda x, *args, **kwargs: x.clone())
        value = self.make_runtime({"rope": SimpleNamespace(rope=rope)})
        positions = torch.tensor([0, 7], dtype=torch.int64)
        for kind, heads, width in ((SLIDING_ATTENTION, 16, 256), (FULL_ATTENTION, 4, 512)):
            rope.reset_mock()
            q, k = value.rotary_embed(torch.zeros(2, 32 * width),
                torch.zeros(2, heads * width), positions, attention_kind=kind)
            self.assertEqual(q.shape, (2, 32, width))
            self.assertEqual(k.shape, (2, heads, width))
            self.assertEqual(rope.call_args_list[0].args[1].shape, (64, width // 2))
            self.assertEqual(rope.call_args_list[1].args[1].shape, (2 * heads, width // 2))
        cos, sin = value._compute_rope_tables(positions, attention_kind=FULL_ATTENTION, dtype=torch.float32)
        # The kernel consumes one frequency per half-rotation pair, not
        # Transformers' duplicated full-width tables.
        self.assertEqual(cos.shape, (2, 256))
        self.assertTrue(torch.equal(cos[:, 64:], torch.ones_like(cos[:, 64:])))
        self.assertTrue(torch.equal(sin[:, 64:], torch.zeros_like(sin[:, 64:])))

    def test_attention_dispatch_preserves_conservative_operator_interface(self):
        full, sliding = mock.Mock(return_value="full"), mock.Mock(return_value="sliding")
        value = self.make_runtime({"fattn_paged": SimpleNamespace(fattn_varlen_paged_fwd_block_ptr=full),
            "fattn_paged_swa": SimpleNamespace(fattn_varlen_paged_swa=sliding)})
        step = SimpleNamespace(cu_seqlens_q="cu_q", cu_seqlens_k="cu_k",
            max_seqlen_q=1, max_seqlen_k=32768, block_table="pages")
        for kind, expected, fn, width in ((FULL_ATTENTION, "full", full, 512),
                                        (SLIDING_ATTENTION, "sliding", sliding, 256)):
            self.assertEqual(value.paged_attention("q", "k", "v", step, attention_kind=kind), expected)
            self.assertEqual(fn.call_args.args[-1], 32768)
            self.assertEqual(fn.call_args.kwargs["softmax_scale"], 1.0)
            self.assertEqual(fn.call_args.kwargs["launch_config"]["D_HEAD"], width)
            self.assertTrue(fn.call_args.kwargs["value_checks"])
        self.assertEqual(sliding.call_args.kwargs["window_size"], 1024)

    def test_scale_softcap_and_norm_route_to_verified_modules(self):
        scale, softcap, norm = (mock.Mock(return_value="out") for _ in range(3))
        value = self.make_runtime({"scale": SimpleNamespace(scale=scale),
            "softcap": SimpleNamespace(softcap=softcap), "rmsnorm": SimpleNamespace(rmsnorm=norm)})
        self.assertEqual(value.scale("x", "scalar"), "out")
        self.assertEqual(value.softcap("logits"), "out")
        self.assertEqual(value.rms_norm("x", "weight", site="input_norm"), "out")
        self.assertEqual(scale.call_args.args, ("x", "scalar"))
        self.assertEqual(softcap.call_args.args, ("logits", 30.0))
        self.assertEqual(norm.call_args.args[1], "weight")

    def test_qualified_family_operations_reject_identity_drift(self):
        identity = {"engine_reachable": False, "backend_qualified": True,
            "qualification": {"deployment_sha256": "a" * 64}}
        raw = mock.Mock()
        raw.report.return_value = copy.deepcopy(identity)
        raw.binding_identity.return_value = copy.deepcopy(identity)
        value = runtime.QualifiedRuntime(raw)
        value.scale("x", "scalar")
        raw.scale.assert_called_once_with("x", "scalar")
        raw.binding_identity.return_value["qualification"]["deployment_sha256"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            value.softcap("logits")
        raw.softcap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
