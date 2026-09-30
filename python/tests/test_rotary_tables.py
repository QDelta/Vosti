import unittest

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from vosti_kernels import rotary


class RotaryTableTests(unittest.TestCase):
    def test_proportional_tables_match_reference_and_preserve_nope_pairs(self) -> None:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        policy = {"rope_type": "proportional", "partial_rotary_factor": 0.25, "factor": 1.0}
        config = Gemma4TextConfig()
        config.global_head_dim = 512
        config.rope_parameters["full_attention"] = {**policy, "rope_theta": 1_000_000.0}
        reference, _ = ROPE_INIT_FUNCTIONS["proportional"](
            config, device=torch.device("cpu"), layer_type="full_attention",
            head_dim_key="global_head_dim")
        ours = rotary.inverse_frequencies(head_dim=512, theta=1_000_000.0,
            scaling=policy, device=torch.device("cpu"))
        self.assertTrue(torch.equal(ours.view(torch.uint8), reference.view(torch.uint8)))
        self.assertEqual(ours.count_nonzero().item(), 64)
        positions = torch.tensor([0, 1, 1023, 32768, 262143])
        cos, sin = rotary.tables(positions, head_dim=512, theta=1_000_000.0,
                                 scaling=policy, dtype=torch.bfloat16)
        self.assertTrue(torch.all(cos[:, 64:] == 1))
        self.assertTrue(torch.all(sin[:, 64:] == 0))
        for value in (-1, 1.1, float("nan"), True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fraction"):
                rotary.inverse_frequencies(head_dim=512, theta=1_000_000.0,
                    scaling={**policy, "partial_rotary_factor": value}, device=torch.device("cpu"))

    def test_precomputed_rows_match_direct_tables_exactly(self) -> None:
        positions = torch.tensor(
            [0, 1, 7, 8191, 8192, 50000, 131071], dtype=torch.int64
        )
        scaling = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
        cached = rotary.precompute_tables(
            max_positions=131072,
            head_dim=128,
            theta=500000.0,
            scaling=scaling,
            device="cpu",
            dtype=torch.bfloat16,
        )
        selected = cached.select(positions)
        direct = rotary.tables(
            positions,
            head_dim=128,
            theta=500000.0,
            scaling=scaling,
            dtype=torch.bfloat16,
        )
        self.assertTrue(torch.equal(selected[0], direct[0]))
        self.assertTrue(torch.equal(selected[1], direct[1]))

    def test_llama31_tables_match_transformers_exactly(self) -> None:
        config = LlamaConfig(
            hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
            max_position_embeddings=131072,
            rope_parameters={
                "rope_type": "llama3", "rope_theta": 500000.0,
                "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        )
        positions = torch.tensor([0, 1, 8191, 8192, 131071], dtype=torch.int64)
        ours = rotary.tables(
            positions,
            head_dim=128,
            theta=500000.0,
            scaling={
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
            dtype=torch.bfloat16,
        )
        reference = LlamaRotaryEmbedding(config)
        x = torch.zeros((1, positions.numel(), 128), dtype=torch.bfloat16)
        expected_cos, expected_sin = reference(x, positions.unsqueeze(0))
        self.assertTrue(torch.equal(ours[0], expected_cos[0, :, :64]))
        self.assertTrue(torch.equal(ours[1], expected_sin[0, :, :64]))

    def test_policy_is_closed_and_validated(self) -> None:
        positions = torch.tensor([0], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "closed policy schema"):
            rotary.tables(
                positions,
                head_dim=128,
                theta=500000.0,
                scaling={"rope_type": "llama3", "factor": 8.0},
                dtype=torch.float32,
            )
        with self.assertRaisesRegex(ValueError, "unsupported sealed"):
            rotary.runtime_scaling({"rope_scaling_kind": "auto"})


if __name__ == "__main__":
    unittest.main()
