"""Weight admission and heterogeneous KV allocation, without GPU or full weights."""

import unittest

import torch

from vosti_kernels import physical
from vosti_kernels.model_families.gemma4 import physical as family
from vosti_kernels.model_families.gemma4.loader import parse_text_config, config_from_runtime_dict
from python.tests.test_gemma4_loader import tiny_config


class Gemma4PhysicalTests(unittest.TestCase):
    def setUp(self):
        self.config = parse_text_config(tiny_config())
        self.embed = torch.zeros((7, 8), dtype=torch.bfloat16)
        self.final_norm = torch.ones(8, dtype=torch.bfloat16)
        self.layers = []
        for i in range(2):
            shapes = physical.four_norm_gated_layer_shapes(8, 12,
                self.config.attention_geometry(i), layer_scale=True)
            tensors = {role: torch.ones(shape, dtype=torch.bfloat16) for role, shape in shapes.items()}
            if self.config.shared_kv_projection(i):
                tensors["v_proj"] = tensors["k_proj"]
            self.layers.append(tuple(tensors[role] for role in family.LAYER_WEIGHT_ROLES))

    def check(self, *, config=None, layers=None, embed=None, norm=None, head=None):
        return family.validate_model_weights_runtime_contract(
            self.embed if embed is None else embed, self.layers if layers is None else layers,
            list(self.config.layer_types), self.final_norm if norm is None else norm,
            self.embed if head is None else head,
            self.config.as_runtime_dict() if config is None else config, 2)

    def test_valid_roles_and_fresh_heterogeneous_cache(self):
        self.check()
        caches = family.init_model_kv_caches(self.embed, self.layers, self.config.as_runtime_dict(), 65)
        self.assertEqual(tuple(caches[0][0].shape), (2, physical.PAGE_SIZE, 2, 4))
        self.assertEqual(tuple(caches[1][0].shape), (2, physical.PAGE_SIZE, 1, 8))
        self.assertEqual(len({t.untyped_storage().data_ptr() for pair in caches for t in pair}), 4)
        self.assertEqual(config_from_runtime_dict(self.config.as_runtime_dict()), self.config)

    def test_rejects_weight_and_config_drift(self):
        for config in ({**self.config.as_runtime_dict(), "runtime_batch": 4},
            {**self.config.as_runtime_dict(), "global_head_dim": 4},
            {**self.config.as_runtime_dict(), "num_global_key_value_heads": 2},
            {**self.config.as_runtime_dict(), "rms_norm_eps": float("nan")}):
            with self.subTest(config=config), self.assertRaises(RuntimeError):
                self.check(config=config)
        with self.assertRaisesRegex(RuntimeError, "tied"):
            self.check(head=self.embed.clone())
        with self.assertRaisesRegex(RuntimeError, "BF16"):
            self.check(norm=self.final_norm.float())
        with self.assertRaisesRegex(RuntimeError, "gradients"):
            self.check(norm=self.final_norm.clone().requires_grad_())
        altered = list(self.layers[1])
        v = family.LAYER_WEIGHT_ROLES.index("v_proj")
        altered[v] = altered[v].clone()
        with self.assertRaisesRegex(RuntimeError, "share its raw K/V"):
            self.check(layers=[self.layers[0], tuple(altered)])
        with self.assertRaisesRegex(RuntimeError, "configured geometry"):
            family.init_model_kv_caches(self.embed, self.layers,
                {**self.config.as_runtime_dict(), "num_global_key_value_heads": 2}, 65)


if __name__ == "__main__":
    unittest.main()
