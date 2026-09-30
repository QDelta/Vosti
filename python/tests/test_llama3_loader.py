import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from vosti_kernels.model_families.llama3 import physical as llama3_physical
from vosti_kernels.model_families.llama3.loader import (
    LLAMA3_LAYER_WEIGHT_ROLES,
    expected_checkpoint_shapes,
    inspect_text_checkpoint,
    load_text_weights,
    parse_text_config,
    validate_checkpoint_layout,
)


def tiny_config() -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 7,
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "tie_word_embeddings": False,
        "pretraining_tp": 1,
    }


class Llama3LoaderTests(unittest.TestCase):
    def _write_checkpoint(
        self,
        directory: Path,
        *,
        shape_override: tuple[str, tuple[int, ...]] | None = None,
        dtype_override: tuple[str, torch.dtype] | None = None,
    ) -> None:
        config = parse_text_config(tiny_config())
        tensors = {}
        for key, shape in expected_checkpoint_shapes(config).items():
            actual_shape = (
                shape_override[1]
                if shape_override is not None and key == shape_override[0]
                else shape
            )
            dtype = (
                dtype_override[1]
                if dtype_override is not None and key == dtype_override[0]
                else torch.bfloat16
            )
            tensors[key] = torch.zeros(actual_shape, dtype=dtype)
        save_file(tensors, directory / "model.safetensors")
        (directory / "config.json").write_text(
            json.dumps(tiny_config()), encoding="utf-8"
        )

    def test_parser_rejects_composition_drift(self) -> None:
        cases = (
            ({"hidden_act": "gelu"}, "SiLU"),
            ({"attention_bias": True}, "bias-free attention"),
            ({"mlp_bias": True}, "bias-free MLP"),
            ({"tie_word_embeddings": None}, "boolean tie_word_embeddings"),
            ({"pretraining_tp": 2}, "pretraining_tp=1"),
            ({"rope_scaling": None}, "llama3 RoPE"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                config = tiny_config()
                config.update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    parse_text_config(config)

    def test_production_inspection_rejects_unregistered_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self._write_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "exact model profiles"):
                inspect_text_checkpoint(path)

    def test_exact_layout_and_role_order_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self._write_checkpoint(path)
            with mock.patch(
                "vosti_kernels.model_families.llama3.loader.inspect_text_checkpoint",
                return_value=parse_text_config(tiny_config()),
            ):
                loaded = load_text_weights(path)

        self.assertEqual(loaded["architecture"], "llama3")
        self.assertEqual(len(loaded["layers"]), 2)
        self.assertEqual(len(loaded["layers"][0]), len(LLAMA3_LAYER_WEIGHT_ROLES))
        self.assertIsNot(loaded["embed_weight"], loaded["lm_head"])
        gate = loaded["layers"][0][LLAMA3_LAYER_WEIGHT_ROLES.index("gate_up_proj")]
        self.assertEqual(tuple(gate.shape), (12, 4))

    def test_tied_checkpoint_uses_embedding_as_lm_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            tied = {**tiny_config(), "tie_word_embeddings": True}
            parsed = parse_text_config(tied)
            tensors = {
                key: torch.zeros(shape, dtype=torch.bfloat16)
                for key, shape in expected_checkpoint_shapes(parsed).items()
            }
            self.assertNotIn("lm_head.weight", tensors)
            save_file(tensors, path / "model.safetensors")
            (path / "config.json").write_text(json.dumps(tied), encoding="utf-8")
            with mock.patch(
                "vosti_kernels.model_families.llama3.loader.inspect_text_checkpoint",
                return_value=parsed,
            ):
                loaded = load_text_weights(path)

        self.assertTrue(loaded["config"]["tie_word_embeddings"])
        self.assertIs(loaded["embed_weight"], loaded["lm_head"])

    def test_shape_dtype_and_extra_role_drift_fail_closed(self) -> None:
        q_key = "model.layers.0.self_attn.q_proj.weight"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self._write_checkpoint(path, shape_override=(q_key, (3, 4)))
            with self.assertRaisesRegex(ValueError, "q_proj.*shape"):
                validate_checkpoint_layout(path, parse_text_config(tiny_config()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self._write_checkpoint(path, dtype_override=(q_key, torch.float32))
            with self.assertRaisesRegex(ValueError, "must be BF16"):
                validate_checkpoint_layout(path, parse_text_config(tiny_config()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self._write_checkpoint(path)
            tensors = {
                key: torch.zeros(shape, dtype=torch.bfloat16)
                for key, shape in expected_checkpoint_shapes(
                    parse_text_config(tiny_config())
                ).items()
            }
            tensors["unexpected.weight"] = torch.zeros((1,), dtype=torch.bfloat16)
            save_file(tensors, path / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "extra=.*unexpected"):
                validate_checkpoint_layout(path, parse_text_config(tiny_config()))

    def test_physical_contract_rejects_unconfigured_alias_and_numeric_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self._write_checkpoint(path)
            with mock.patch(
                "vosti_kernels.model_families.llama3.loader.inspect_text_checkpoint",
                return_value=parse_text_config(tiny_config()),
            ):
                loaded = load_text_weights(path)

        config = dict(loaded["config"])
        config["rms_norm_eps"] = 1e-6
        with self.assertRaisesRegex(RuntimeError, "exact rms_norm_eps"):
            llama3_physical.validate_model_weights_runtime_contract(
                loaded["embed_weight"], loaded["layers"], loaded["final_norm"],
                loaded["lm_head"], config, 2,
            )

        with self.assertRaisesRegex(RuntimeError, "distinct configured LM head"):
            llama3_physical.validate_model_weights_runtime_contract(
                loaded["embed_weight"], loaded["layers"], loaded["final_norm"],
                loaded["embed_weight"], loaded["config"], 2,
            )
        tied_config = {**loaded["config"], "tie_word_embeddings": True}
        with self.assertRaisesRegex(RuntimeError, "configured tied embedding"):
            llama3_physical.validate_model_weights_runtime_contract(
                loaded["embed_weight"], loaded["layers"], loaded["final_norm"],
                loaded["lm_head"], tied_config, 2,
            )

        layer = list(loaded["layers"][0])
        layer[LLAMA3_LAYER_WEIGHT_ROLES.index("v_proj")] = layer[
            LLAMA3_LAYER_WEIGHT_ROLES.index("k_proj")
        ]
        with self.assertRaisesRegex(RuntimeError, "share storage"):
            llama3_physical.validate_model_weights_runtime_contract(
                loaded["embed_weight"], [tuple(layer), loaded["layers"][1]],
                loaded["final_norm"], loaded["lm_head"], loaded["config"], 2,
            )


if __name__ == "__main__":
    unittest.main()
