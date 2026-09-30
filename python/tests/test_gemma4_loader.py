"""CPU-only configuration/layout checks; not model inference qualification."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from vosti_kernels.model_families.gemma4.loader import (
    FULL_ATTENTION, SLIDING_ATTENTION, LAYER_WEIGHT_ROLES, TEXT_PREFIX,
    expected_checkpoint_shapes, layer_checkpoint_keys, load_text_weights,
    parse_text_config, validate_checkpoint_layout,
)


def tiny_config() -> dict:
    return dict(model_type="gemma4_text", vocab_size=7, hidden_size=8,
        intermediate_size=12, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, num_global_key_value_heads=1, head_dim=4,
        global_head_dim=8, max_position_embeddings=128, rms_norm_eps=1e-6,
        sliding_window=16, hidden_activation="gelu_pytorch_tanh", tie_word_embeddings=True,
        attention_k_eq_v=True, use_bidirectional_attention="vision", final_logit_softcapping=30.0,
        hidden_size_per_layer_input=0, num_kv_shared_layers=0, enable_moe_block=False,
        use_double_wide_mlp=False, attention_bias=False, attention_dropout=0.0,
        layer_types=[SLIDING_ATTENTION, FULL_ATTENTION], rope_parameters={
            SLIDING_ATTENTION: dict(rope_type="default", rope_theta=10000.0),
            FULL_ATTENTION: dict(rope_type="proportional", rope_theta=1000000.0,
                                 partial_rotary_factor=0.25)})


class Gemma4LoaderTests(unittest.TestCase):
    def test_unified_text_uses_same_dense_schema_and_rejects_other_features(self):
        raw = {**tiny_config(), "model_type": "gemma4_unified_text"}
        self.assertEqual(parse_text_config(raw), parse_text_config(tiny_config()))
        for field, value in (("enable_moe_block", True), ("num_kv_shared_layers", 1),
                             ("hidden_size_per_layer_input", 8), ("model_type", "unknown")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_text_config({**raw, field: value})

    def test_unified_checkpoint_load_does_not_need_multimodal_autoconfig(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, extra="model.vision_tower.ignored.weight")
            raw = {"model_type": "gemma4_unified", "text_config": {
                **tiny_config(), "model_type": "gemma4_unified_text"}}
            (root / "config.json").write_text(json.dumps(raw))
            with patch("vosti_kernels.model_families.gemma4.loader.AutoConfig.from_pretrained",
                       side_effect=AssertionError("must not load multimodal AutoConfig")):
                loaded = load_text_weights(root)
            self.assertEqual(loaded["config"], parse_text_config(tiny_config()).as_runtime_dict())
            self.assertEqual(json.loads((root / "config.json").read_text()), raw)

    def test_static_geometry_and_shared_projection(self):
        c = parse_text_config(dict(model_type="gemma4", text_config=tiny_config(),
                                   vision_config={"irrelevant": True}))
        self.assertEqual(c.attention_geometry(0).kv_tail_shape, (2, 4))
        self.assertEqual(c.attention_geometry(1).kv_tail_shape, (1, 8))
        self.assertEqual(c.attention_geometry(1).query_width, 32)
        self.assertFalse(c.shared_kv_projection(0))
        self.assertTrue(c.shared_kv_projection(1))
        keys = layer_checkpoint_keys(1, c)
        self.assertEqual(keys["k_proj"], keys["v_proj"])
        expected = expected_checkpoint_shapes(c)
        self.assertNotIn(f"{TEXT_PREFIX}layers.1.self_attn.v_proj.weight", expected)
        self.assertEqual(expected[f"{TEXT_PREFIX}layers.1.layer_scalar"], (1,))
        for bad in (-1, 2, True):
            with self.assertRaisesRegex(ValueError, "index"):
                c.attention_geometry(bad)

    def test_rejects_unsupported_features_and_invalid_numerics(self):
        for field, value in (("num_kv_shared_layers", 1), ("hidden_size_per_layer_input", 8),
            ("enable_moe_block", True), ("use_double_wide_mlp", True),
            ("attention_bias", True), ("hidden_activation", "silu"),
            ("tie_word_embeddings", False), ("use_bidirectional_attention", True),
            ("attention_dropout", 0.1), ("attn_logit_softcapping", 50.0),
            ("head_dim", 3), ("num_global_key_value_heads", 3),
            ("rms_norm_eps", float("nan")), ("final_logit_softcapping", 0.0),
            ("layer_types", [FULL_ATTENTION]), ("attention_k_eq_v", 1)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_text_config({**tiny_config(), field: value})
        for value in (-1, float("nan"), 1.5, True):
            config = tiny_config()
            config["rope_parameters"][FULL_ATTENTION]["partial_rotary_factor"] = value
            with self.assertRaises(ValueError):
                parse_text_config(config)

    def _write(self, root, *, change=None, extra=None):
        config = tiny_config()
        tensors = {k: torch.full(s, 0.25, dtype=torch.bfloat16)
                   for k, s in expected_checkpoint_shapes(parse_text_config(config)).items()}
        if change:
            tensors[change[0]] = change[1]
        if extra:
            tensors[extra] = torch.zeros(1, dtype=torch.bfloat16)
        save_file(tensors, root / "model.safetensors")
        (root / "config.json").write_text(json.dumps(config))

    def test_loads_exact_roles_tied_head_and_raw_global_kv_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, extra="model.vision_tower.ignored.weight")
            loaded = load_text_weights(root)
        self.assertEqual(loaded["architecture"], "gemma4_text")
        self.assertIs(loaded["embed_weight"], loaded["lm_head"])
        idx = {role: i for i, role in enumerate(LAYER_WEIGHT_ROLES)}
        local, global_ = loaded["layers"]
        self.assertIsNot(local[idx["k_proj"]], local[idx["v_proj"]])
        self.assertIs(global_[idx["k_proj"]], global_[idx["v_proj"]])
        self.assertEqual(tuple(global_[idx["gate_up_proj"]].shape), (24, 8))
        self.assertEqual(global_[idx["layer_scalar"]].item(), 0.25)

    def test_rejects_checkpoint_shape_dtype_and_extra_text_tensors(self):
        key = f"{TEXT_PREFIX}layers.1.self_attn.k_proj.weight"
        for change, extra in ((None, f"{TEXT_PREFIX}unexpected.weight"),
            ((key, torch.zeros((9, 8), dtype=torch.bfloat16)), None),
            ((key, torch.zeros((8, 8), dtype=torch.float32)), None)):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write(root, change=change, extra=extra)
                with self.assertRaises(ValueError):
                    validate_checkpoint_layout(root, parse_text_config(tiny_config()))


if __name__ == "__main__":
    unittest.main()
