import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from vosti_kernels.model_families.gemma3 import physical as gemma3_physical
from vosti_kernels.model_families.gemma3.loader import (
    FULL_ATTENTION,
    GEMMA3_LAYER_WEIGHT_ROLES,
    SLIDING_ATTENTION,
    Gemma3TextConfig,
    expected_checkpoint_shapes,
    load_text_weights,
    parse_text_config,
    validate_checkpoint_layout,
)


def tiny_resolved_config() -> dict:
    return {
        "model_type": "gemma3_text",
        "vocab_size": 7,
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "rope_parameters": {
            SLIDING_ATTENTION: {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            FULL_ATTENTION: {
                "rope_type": "linear",
                "factor": 8.0,
                "rope_theta": 1000000.0,
            },
        },
        "attention_bias": False,
        "attention_dropout": 0.0,
        "query_pre_attn_scalar": 2,
        "sliding_window": 16,
        "layer_types": [SLIDING_ATTENTION, FULL_ATTENTION],
        "attn_logit_softcapping": None,
        "final_logit_softcapping": None,
        "use_bidirectional_attention": False,
    }


class Gemma3ConfigTests(unittest.TestCase):
    def test_parses_outer_multimodal_config_as_text_only(self) -> None:
        parsed = parse_text_config(
            {
                "model_type": "gemma3",
                "text_config": tiny_resolved_config(),
                "vision_config": {"model_type": "siglip_vision_model"},
            }
        )

        self.assertEqual(parsed.hidden_size, 4)
        self.assertEqual(parsed.layer_types, (SLIDING_ATTENTION, FULL_ATTENTION))
        self.assertEqual(parsed.local_rope_theta, 10000.0)
        self.assertEqual(parsed.global_rope_theta, 1000000.0)
        self.assertEqual(parsed.global_rope_factor, 8.0)

    def test_rejects_unsupported_numerical_conventions(self) -> None:
        cases = (
            ({"hidden_activation": "silu"}, "gelu_pytorch_tanh"),
            ({"attention_bias": True}, "bias-free"),
            ({"tie_word_embeddings": False}, "tied"),
            ({"layer_types": [SLIDING_ATTENTION]}, "cover every"),
            (
                {"layer_types": ["linear_attention", FULL_ATTENTION]},
                "unsupported.*attention",
            ),
            ({"attn_logit_softcapping": 50.0}, "softcapping"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                config = tiny_resolved_config()
                config.update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    parse_text_config(config)


class Gemma3CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = parse_text_config(tiny_resolved_config())

    def _write_checkpoint(
        self,
        directory: Path,
        *,
        shape_override: tuple[str, tuple[int, ...]] | None = None,
        dtype_override: tuple[str, torch.dtype] | None = None,
    ) -> None:
        shapes = expected_checkpoint_shapes(self.config)
        tensors = {}
        for key, shape in shapes.items():
            actual_shape = shape_override[1] if shape_override and key == shape_override[0] else shape
            dtype = dtype_override[1] if dtype_override and key == dtype_override[0] else torch.bfloat16
            tensors[key] = torch.zeros(actual_shape, dtype=dtype)
        save_file(tensors, directory / "model.safetensors")
        (directory / "config.json").write_text(
            json.dumps(tiny_resolved_config()), encoding="utf-8"
        )

    def test_exact_text_layout_and_role_order_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            self._write_checkpoint(model_path)

            validate_checkpoint_layout(model_path, self.config)
            loaded = load_text_weights(model_path)

        self.assertEqual(loaded["architecture"], "gemma3_text")
        self.assertEqual(len(loaded["model_config_sha256"]), 64)
        self.assertEqual(
            loaded["attention_kinds"],
            [SLIDING_ATTENTION, FULL_ATTENTION],
        )
        self.assertEqual(len(loaded["layers"]), 2)
        self.assertEqual(len(loaded["layers"][0]), len(GEMMA3_LAYER_WEIGHT_ROLES))
        gate_up_index = GEMMA3_LAYER_WEIGHT_ROLES.index("gate_up_proj")
        self.assertEqual(tuple(loaded["layers"][0][gate_up_index].shape), (12, 4))
        self.assertIs(loaded["embed_weight"], loaded["lm_head"])

    def test_shape_and_dtype_drift_fail_closed(self) -> None:
        q_key = "language_model.model.layers.0.self_attn.q_proj.weight"
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            self._write_checkpoint(model_path, shape_override=(q_key, (3, 4)))
            with self.assertRaisesRegex(ValueError, "q_proj.*shape"):
                validate_checkpoint_layout(model_path, self.config)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            self._write_checkpoint(
                model_path, dtype_override=(q_key, torch.float32)
            )
            with self.assertRaisesRegex(ValueError, "must be BF16"):
                validate_checkpoint_layout(model_path, self.config)

    def test_physical_role_contract_rejects_architecture_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            self._write_checkpoint(model_path)
            loaded = load_text_weights(model_path)

        def validate(
            *,
            layers=loaded["layers"],
            attention_kinds=loaded["attention_kinds"],
            lm_head=loaded["lm_head"],
            final_norm=loaded["final_norm"],
        ):
            return gemma3_physical.validate_model_weights_runtime_contract(
                loaded["embed_weight"],
                layers,
                attention_kinds,
                final_norm,
                lm_head,
                loaded["config"],
                2,
            )

        pre_ff = GEMMA3_LAYER_WEIGHT_ROLES.index("pre_feedforward_norm")
        malformed_layer = list(loaded["layers"][0])
        malformed_layer[pre_ff] = torch.zeros((3,), dtype=torch.bfloat16)
        cases = (
            (
                {"attention_kinds": [FULL_ATTENTION, FULL_ATTENTION]},
                "attention kinds disagree",
            ),
            ({"lm_head": loaded["lm_head"].clone()}, "tied embedding"),
            ({"layers": [loaded["layers"][0][:-1], loaded["layers"][1]]},
             "exact role tuple"),
            ({"layers": [tuple(malformed_layer), loaded["layers"][1]]},
             "pre_feedforward_norm has shape"),
            ({"final_norm": loaded["final_norm"].float()}, "one BF16"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(RuntimeError, message):
                    validate(**changes)

    def test_physical_role_contract_rejects_changed_fixed_numerics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            self._write_checkpoint(model_path)
            loaded = load_text_weights(model_path)

        changes = {
            "rms_norm_eps": 1e-5,
            "local_rope_theta": 20_000.0,
            "global_rope_theta": 500_000.0,
            "global_rope_factor": 4.0,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                config = dict(loaded["config"])
                config[field] = value
                with self.assertRaisesRegex(RuntimeError, f"exact {field}"):
                    gemma3_physical.validate_model_weights_runtime_contract(
                        loaded["embed_weight"],
                        loaded["layers"],
                        loaded["attention_kinds"],
                        loaded["final_norm"],
                        loaded["lm_head"],
                        config,
                        2,
                    )

    def test_physical_role_contract_accepts_profile_varying_attention_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            self._write_checkpoint(model_path)
            loaded = load_text_weights(model_path)

        config = dict(loaded["config"])
        config["query_pre_attn_scalar"] = 168.0
        self.assertIsNone(
            gemma3_physical.validate_model_weights_runtime_contract(
                loaded["embed_weight"],
                loaded["layers"],
                loaded["attention_kinds"],
                loaded["final_norm"],
                loaded["lm_head"],
                config,
                2,
            )
        )

    def test_permission_contract_is_config_independent_but_shape_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            self._write_checkpoint(model_path)
            loaded = load_text_weights(model_path)

        gemma3_physical.validate_model_weights_permission_contract(
            loaded["embed_weight"],
            loaded["layers"],
            loaded["attention_kinds"],
            loaded["final_norm"],
            loaded["lm_head"],
            loaded["config"]["hidden_size"],
            loaded["config"]["sliding_window"],
            2,
        )

        malformed_layer = list(loaded["layers"][0])
        down = GEMMA3_LAYER_WEIGHT_ROLES.index("down_proj")
        malformed_layer[down] = torch.zeros((4, 5), dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "projection shapes are inconsistent"):
            gemma3_physical.validate_model_weights_permission_contract(
                loaded["embed_weight"],
                [tuple(malformed_layer), loaded["layers"][1]],
                loaded["attention_kinds"],
                loaded["final_norm"],
                loaded["lm_head"],
                loaded["config"]["hidden_size"],
                loaded["config"]["sliding_window"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
