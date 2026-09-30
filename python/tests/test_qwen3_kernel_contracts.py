"""Runtime contract tests for the production Qwen3 kernel adapter."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import types
import unittest
from unittest import mock

import torch

from vosti_kernels import (
    kernels,
    primitive_runtime,
)
from vosti_kernels.model_families.qwen3 import physical as qwen3_physical
from vosti_kernels.model_families.qwen3 import runtime as qwen3_runtime


def _test_runtime(**kwargs):
    return qwen3_runtime.runtime_for_tests(**kwargs)


class KernelContractTests(unittest.TestCase):
    def test_block_table_padding_is_a_valid_page_id(self) -> None:
        out = kernels.block_tables_tensor([[3], [4, 5]])
        self.assertEqual(out.tolist(), [[3, 0], [4, 5]])

    def test_step_plan_materializers_preserve_values_and_allocate_fresh(self) -> None:
        tensors = (
            kernels.token_tensor([7, 3]),
            kernels.position_tensor([11, 12]),
            kernels.slot_tensor([65, 130]),
            kernels.block_tables_tensor([[3], [4, 5]]),
            kernels.seq_lens_tensor([0, 1, 3]),
        )
        self.assertEqual(
            [tensor.tolist() for tensor in tensors],
            [[7, 3], [11, 12], [65, 130], [[3, 0], [4, 5]], [0, 1, 3]],
        )
        self.assertTrue(all(tensor.dtype == torch.int64 for tensor in tensors))
        self.assertTrue(all(tensor.is_contiguous() for tensor in tensors))
        self.assertEqual(
            len({tensor.untyped_storage().data_ptr() for tensor in tensors}),
            len(tensors),
        )

        runtime = self._configure_cpu_qwen()
        configured = (
            kernels.token_tensor([7, 3], runtime=runtime),
            kernels.position_tensor([11, 12], runtime=runtime),
            kernels.slot_tensor([65, 130], runtime=runtime),
            kernels.block_tables_tensor([[3], [4, 5]], runtime=runtime),
            kernels.seq_lens_tensor([0, 1, 3], runtime=runtime),
        )
        self.assertEqual(configured[1].dtype, torch.int64)
        self.assertTrue(
            all(
                tensor.dtype == torch.int32
                for i, tensor in enumerate(configured)
                if i != 1
            )
        )
        self.assertEqual(
            [tensor.tolist() for tensor in configured],
            [[7, 3], [11, 12], [65, 130], [[3, 0], [4, 5]], [0, 1, 3]],
        )

    def test_step_plan_materializers_reject_non_cuda_device_anchor(self) -> None:
        cpu_anchor = torch.zeros((1,))
        for materialize, values in (
            (kernels.token_tensor, [7]),
            (kernels.position_tensor, [11]),
            (kernels.slot_tensor, [65]),
            (kernels.block_tables_tensor, [[3]]),
            (kernels.seq_lens_tensor, [0, 1]),
        ):
            with self.subTest(materialize=materialize.__name__):
                with self.assertRaisesRegex(ValueError, "CUDA tensor"):
                    materialize(values, cpu_anchor)

    def _configure_cpu_qwen(self):
        return _test_runtime(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=1e-6,
        )

    def _tiny_model_weights(self, *, tied: bool = False):
        self.runtime = _test_runtime(
            hidden_size=4,
            num_heads=2,
            num_kv_heads=1,
            head_dim=2,
            intermediate_size=6,
            rms_norm_eps=1e-6,
        )
        embed = torch.zeros((11, 4))
        layer = (
            torch.zeros((4,)),
            torch.zeros((4, 4)),
            torch.zeros((2, 4)),
            torch.zeros((2, 4)),
            torch.zeros((2,)),
            torch.zeros((2,)),
            torch.zeros((4, 4)),
            torch.zeros((4,)),
            torch.zeros((12, 4)),
            torch.zeros((4, 6)),
        )
        final_norm = torch.zeros((4,))
        lm_head = embed if tied else torch.zeros((11, 4))
        return embed, [layer], final_norm, lm_head

    def test_model_weight_contract_accepts_intentional_tied_embedding(self) -> None:
        embed, layers, final_norm, lm_head = self._tiny_model_weights(tied=True)
        self.assertIs(embed, lm_head)
        self.assertIsNone(
            qwen3_physical.validate_model_weights_runtime_contract(
                embed,
                layers,
                final_norm,
                lm_head,
                self.runtime.runtime_config(),
                1,
            )
        )

    def test_model_weight_contract_rejects_changed_fixed_numerics(self) -> None:
        embed, layers, final_norm, lm_head = self._tiny_model_weights()
        for field, value in (("rms_norm_eps", 1e-5), ("rope_theta", 10_000.0)):
            with self.subTest(field=field):
                config = dict(self.runtime.runtime_config())
                config[field] = value
                with self.assertRaisesRegex(RuntimeError, f"exact {field}"):
                    qwen3_physical.validate_model_weights_runtime_contract(
                        embed, layers, final_norm, lm_head, config, 1,
                    )

    def test_unconfigured_model_weight_contract_rejects_odd_gate_width(self) -> None:
        embed = torch.zeros((11, 4))
        layer = (
            torch.zeros((4,)),
            torch.zeros((4, 4)),
            torch.zeros((2, 4)),
            torch.zeros((2, 4)),
            torch.zeros((2,)),
            torch.zeros((2,)),
            torch.zeros((4, 4)),
            torch.zeros((4,)),
            torch.zeros((11, 4)),
            torch.zeros((4, 5)),
        )
        with self.assertRaisesRegex(RuntimeError, "even output width"):
            qwen3_physical.validate_model_weights_permission_contract(
                embed, [layer], torch.zeros((4,)), torch.zeros((11, 4)), 1
            )

    def test_model_weight_contract_rejects_malformed_physical_roles(self) -> None:
        embed, layers, final_norm, lm_head = self._tiny_model_weights()

        def validate(*, embed_weight=embed, layer_values=layers,
                     final=final_norm, head=lm_head, count=1):
            return qwen3_physical.validate_model_weights_runtime_contract(
                embed_weight,
                layer_values,
                final,
                head,
                self.runtime.runtime_config(),
                count,
            )

        noncontiguous_q = torch.zeros((4, 4)).t()
        grad_final = final_norm.clone().requires_grad_()
        cases = (
            ({"count": 2}, "wrong number of layers"),
            ({"layer_values": [layers[0][:-1]]}, "exact role tuple"),
            (
                {"layer_values": [(layers[0][0], torch.zeros((3, 4)), *layers[0][2:])]},
                "q_proj has shape",
            ),
            (
                {"layer_values": [(layers[0][0], noncontiguous_q, *layers[0][2:])]},
                "not contiguous",
            ),
            ({"final": grad_final}, "requires gradients"),
            ({"head": lm_head.to(torch.float64)}, "one dtype and device"),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    validate(**overrides)

    def test_int32_materializers_reject_before_narrowing(self) -> None:
        runtime = self._configure_cpu_qwen()
        too_large = 1 << 31
        for materialize, values in (
            (kernels.token_tensor, [too_large]),
            (kernels.slot_tensor, [too_large]),
            (kernels.seq_lens_tensor, [0, too_large]),
            (kernels.block_tables_tensor, [[too_large]]),
        ):
            with self.subTest(materialize=materialize.__name__):
                with self.assertRaisesRegex(OverflowError, "int32 index range"):
                    materialize(values, runtime=runtime)

    def test_kv_allocator_returns_exact_independent_cpu_collection(self) -> None:
        caches = kernels.init_kv_caches(3, kernels._BLOCK_SIZE + 1)
        self.assertEqual(len(caches), 3)
        tensors = [tensor for pair in caches for tensor in pair]
        self.assertTrue(
            all(tuple(tensor.shape) == (2, kernels._BLOCK_SIZE) for tensor in tensors)
        )
        self.assertEqual(len({id(tensor) for tensor in tensors}), len(tensors))
        self.assertEqual(
            len({tensor.untyped_storage().data_ptr() for tensor in tensors}),
            len(tensors),
        )

    def test_kv_allocator_accepts_distinct_zero_sized_tensors(self) -> None:
        caches = kernels.init_kv_caches(2, 0)
        tensors = [tensor for pair in caches for tensor in pair]
        self.assertEqual(len({id(tensor) for tensor in tensors}), len(tensors))
        self.assertTrue(all(tensor.numel() == 0 for tensor in tensors))
        self.assertEqual(
            {tensor.untyped_storage().data_ptr() for tensor in tensors}, {0}
        )

    def test_kv_allocator_runtime_contract_rejects_malformed_results(self) -> None:
        valid = kernels.init_kv_caches(1, kernels._BLOCK_SIZE)
        base = torch.zeros((2, kernels._BLOCK_SIZE), dtype=torch.float32)
        cases = (
            ([], 1, "wrong number of layers"),
            ([(valid[0][0],)], 1, "exact K/V pair"),
            ([(valid[0][0], valid[0][0])], 1, "same tensor object"),
            ([(base[0:1], base[1:2])], 1, "shared storage"),
            (
                [(torch.zeros((2, kernels._BLOCK_SIZE)), valid[0][1])],
                1,
                "has shape",
            ),
            (
                [(valid[0][0].to(torch.float64), valid[0][1])],
                1,
                "dtype/device",
            ),
        )
        for caches, layers, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    kernels._validate_init_kv_caches_runtime_contract(
                        caches, layers, kernels._BLOCK_SIZE
                    )

    def test_configured_kv_allocator_pins_full_physical_shape(self) -> None:
        runtime = self._configure_cpu_qwen()
        caches = kernels.init_kv_caches(
            2, kernels._BLOCK_SIZE + 1, runtime
        )
        for pair in caches:
            for tensor in pair:
                self.assertEqual(
                    tuple(tensor.shape), (2, kernels._BLOCK_SIZE, 8, 128)
                )
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(tensor.device, torch.device("cpu"))

    def test_paged_attention_rejects_before_narrowing(self) -> None:
        runtime = self._configure_cpu_qwen()
        q = torch.zeros((1, 16, 128))
        cache = torch.zeros((1, kernels._BLOCK_SIZE, 8, 128))
        cu_q = torch.tensor([0, 1], dtype=torch.int64)
        cu_k = torch.tensor([0, 1 << 31], dtype=torch.int64)
        block_table = torch.tensor([[0]], dtype=torch.int64)
        with self.assertRaisesRegex(OverflowError, "int32 index range"):
            kernels.paged_attention(
                q, cache, cache, cu_q, cu_k, 1, 1 << 31, block_table,
                runtime,
            )

    def test_verified_paged_attention_runtime_contract_fails_closed(self) -> None:
        runtime = self._configure_cpu_qwen()

        def launch(
            *,
            q=None,
            k_cache=None,
            v_cache=None,
            cu_q=None,
            cu_k=None,
            max_q=1,
            max_k=2,
            block_table=None,
        ) -> None:
            q = torch.zeros((2, 16, 128)) if q is None else q
            k_cache = (
                torch.zeros((2, kernels._BLOCK_SIZE, 8, 128))
                if k_cache is None else k_cache
            )
            v_cache = torch.zeros_like(k_cache) if v_cache is None else v_cache
            cu_q = torch.tensor([0, 1, 2], dtype=torch.int32) if cu_q is None else cu_q
            cu_k = torch.tensor([0, 2, 4], dtype=torch.int32) if cu_k is None else cu_k
            block_table = (
                torch.tensor([[0], [1]], dtype=torch.int32)
                if block_table is None else block_table
            )
            kernels._validate_paged_attention_runtime_contract(
                q, k_cache, v_cache, cu_q, cu_k,
                max_q, max_k, block_table, runtime,
            )

        # The otherwise-valid CPU fixture reaches the final deployment check.
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            launch()

        cases = (
            (
                {"q": torch.zeros((2, 8, 128))},
                "query geometry",
            ),
            (
                {"v_cache": torch.zeros((3, kernels._BLOCK_SIZE, 8, 128))},
                "cache shapes must match",
            ),
            (
                {"cu_q": torch.tensor([0, 1, 2], dtype=torch.int64)},
                "metadata tensors must have int32 dtype",
            ),
            (
                {"cu_q": torch.tensor([0, 0, 2], dtype=torch.int32)},
                "strictly increasing",
            ),
            (
                {
                    "cu_q": torch.tensor([0, 2], dtype=torch.int32),
                    "cu_k": torch.tensor([0, 1], dtype=torch.int32),
                    "block_table": torch.tensor([[0]], dtype=torch.int32),
                    "max_q": 2,
                    "max_k": 1,
                },
                "q_len <= k_len",
            ),
            (
                {"max_q": 0},
                "positive int32",
            ),
            (
                {
                    "cu_k": torch.tensor([0, 65, 130], dtype=torch.int32),
                    "max_k": 65,
                },
                "does not cover every KV row",
            ),
            (
                {"block_table": torch.tensor([[0], [2]], dtype=torch.int32)},
                "in-bounds cache page",
            ),
        )
        for arguments, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    launch(**arguments)

    def test_verified_paged_attention_dispatch_pins_physical_launch_roles(self) -> None:
        runtime = self._configure_cpu_qwen()
        q = torch.zeros((2, 16, 128))
        k_cache = torch.zeros((2, kernels._BLOCK_SIZE, 8, 128))
        v_cache = torch.ones_like(k_cache)
        cu_q = torch.tensor([0, 1, 2], dtype=torch.int32)
        cu_k = torch.tensor([0, 2, 4], dtype=torch.int32)
        block_table = torch.tensor([[0], [1]], dtype=torch.int32)
        observed = {}

        def launch(*args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs
            return torch.empty_like(q)

        vk = types.SimpleNamespace(
            fattn_paged=types.SimpleNamespace(
                fattn_varlen_paged_fwd_block_ptr=launch
            )
        )
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                return_value={"sealed": 1},
            ),
            mock.patch.object(
                kernels, "_validate_paged_attention_runtime_contract"
            ) as validate,
        ):
            output = kernels.paged_attention(
                q, k_cache, v_cache, cu_q, cu_k, 1, 2, block_table,
                runtime,
            )

        validate.assert_called_once_with(
            q, k_cache, v_cache, cu_q, cu_k, 1, 2, block_table,
            runtime,
        )
        self.assertIs(observed["args"][0], q)
        self.assertIs(observed["args"][1], k_cache)
        self.assertIs(observed["args"][2], v_cache)
        self.assertTrue(torch.equal(observed["args"][3], cu_q))
        self.assertTrue(torch.equal(observed["args"][4], cu_k))
        self.assertEqual(observed["args"][5:7], (1, 2))
        self.assertTrue(torch.equal(observed["kwargs"]["block_table"], block_table))
        self.assertEqual(observed["kwargs"]["launch_config"], {"sealed": 1})
        self.assertNotIn("causal", observed["kwargs"])
        self.assertEqual(observed["kwargs"]["softmax_scale"], 128**-0.5)
        self.assertEqual(tuple(output.shape), (2, 16 * 128))
        self.assertTrue(output.is_contiguous())

    def test_verified_caller_attention_skips_device_value_validation(self) -> None:
        runtime = self._configure_cpu_qwen()
        q = torch.zeros((2, 16, 128))
        k_cache = torch.zeros((2, kernels._BLOCK_SIZE, 8, 128))
        v_cache = torch.ones_like(k_cache)
        cu_q = torch.tensor([0, 1, 2], dtype=torch.int32)
        cu_k = torch.tensor([0, 2, 4], dtype=torch.int32)
        block_table = torch.tensor([[0], [1]], dtype=torch.int32)
        observed = {}

        def launch(*args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs
            return torch.empty_like(q)

        vk = types.SimpleNamespace(
            fattn_paged=types.SimpleNamespace(
                fattn_varlen_paged_fwd_block_ptr=launch
            )
        )
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                return_value={"sealed": 1},
            ),
            mock.patch.object(
                kernels,
                "_require_nonnegative_int32_tensor",
                side_effect=AssertionError("must not read device metadata"),
            ),
            mock.patch.object(
                kernels,
                "_validate_paged_attention_runtime_contract",
                side_effect=AssertionError("must not copy device metadata"),
            ),
            mock.patch.object(
                kernels,
                "paged_attention",
                side_effect=AssertionError("must not enter checked fallback"),
            ),
        ):
            output = kernels.paged_attention_from_verified_caller(
                q, k_cache, v_cache, cu_q, cu_k, 1, 2, block_table,
                runtime,
            )

        self.assertIs(observed["args"][0], q)
        self.assertIs(observed["args"][1], k_cache)
        self.assertIs(observed["args"][2], v_cache)
        self.assertIs(observed["args"][3], cu_q)
        self.assertIs(observed["args"][4], cu_k)
        self.assertIs(observed["kwargs"]["block_table"], block_table)
        self.assertIs(observed["kwargs"]["value_checks"], False)
        self.assertNotIn("causal", observed["kwargs"])
        self.assertEqual(observed["kwargs"]["launch_config"], {"sealed": 1})
        self.assertEqual(tuple(output.shape), (2, 16 * 128))

    def test_select_clones_exact_last_row_without_kernel_dispatch(self) -> None:
        logits = torch.arange(30, dtype=torch.float32).reshape(6, 5)
        cu = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
        with mock.patch.object(
            primitive_runtime,
            "verified_for",
            side_effect=AssertionError("selection must not resolve a custom kernel"),
        ):
            row = kernels.select_sample_logits(logits, cu, 1)
        self.assertTrue(torch.equal(row, logits[4]))
        self.assertTrue(row.is_contiguous())
        self.assertNotEqual(row.data_ptr(), logits[4].data_ptr())
        row[0] = -1
        self.assertNotEqual(row[0], logits[4, 0])

    def test_select_rejects_invalid_segment_metadata(self) -> None:
        logits = torch.zeros((3, 5), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "2D logits and 1D"):
            kernels.select_sample_logits(logits.reshape(-1), torch.tensor([0, 3]), 0)
        with self.assertRaisesRegex(RuntimeError, "2D logits and 1D"):
            kernels.select_sample_logits(logits, torch.tensor([[0, 3]]), 0)
        cases = (
            (torch.tensor([0, 0], dtype=torch.int32), 0, "nonempty in-bounds"),
            (torch.tensor([0, 4], dtype=torch.int32), 0, "nonempty in-bounds"),
            (torch.tensor([0, 3], dtype=torch.int32), -1, "outside cu_seqlens_q"),
            (torch.tensor([0, 3], dtype=torch.int32), 1, "outside cu_seqlens_q"),
        )
        for cu, index, message in cases:
            with self.subTest(cu=cu.tolist(), index=index):
                with self.assertRaisesRegex(RuntimeError, message):
                    kernels.select_sample_logits(logits, cu, index)

    def test_sampling_row_gather_copies_each_exact_last_hidden_row(self) -> None:
        hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        cu = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
        rows = kernels.select_rows_for_sampling(hidden, cu)

        self.assertTrue(torch.equal(rows, hidden[torch.tensor([1, 4, 5])]))
        self.assertTrue(rows.is_contiguous())
        self.assertNotEqual(rows.untyped_storage().data_ptr(), hidden.data_ptr())
        rows[0, 0] = -1
        self.assertNotEqual(rows[0, 0], hidden[1, 0])

    def test_batched_sampler_matches_existing_sampler_per_row(self) -> None:
        rows = torch.tensor(
            [
                [-3.0, 7.0, 7.0, 2.0],
                [float("nan"), 5.0, float("nan"), 1.0],
                [float("-inf"), float("inf"), 4.0, float("inf")],
            ]
        )
        expected = [kernels.sample(row) for row in rows]
        self.assertEqual(kernels.sample_tokens_rows(rows), expected)
        self.assertEqual(kernels.sample_tokens_rows(rows[:0]), [])
        self.assertEqual(
            kernels.sample_tokens_rows(torch.empty((2, 0))),
            [kernels.sample(torch.empty(0)), kernels.sample(torch.empty(0))],
        )

    def test_batched_sampler_observer_flag_visits_each_row(self) -> None:
        rows = torch.tensor([[1.0, 2.0], [4.0, 3.0]])
        with (
            mock.patch.dict(
                os.environ, {"VOSTI_KERNELS_LOGITS_OBSERVER": "1"}, clear=True
            ),
            mock.patch.object(kernels, "_maybe_digest_logits") as observe,
        ):
            self.assertEqual(kernels.sample_tokens_rows(rows), [1, 0])
        self.assertEqual(observe.call_count, 2)
        self.assertTrue(torch.equal(observe.call_args_list[0].args[0], rows[0]))
        self.assertEqual(observe.call_args_list[0].args[1], 0)
        self.assertTrue(torch.equal(observe.call_args_list[1].args[0], rows[1]))
        self.assertEqual(observe.call_args_list[1].args[1], 1)

    def test_sampling_adapters_reject_wrong_ranks(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "2D hidden states and 1D"):
            kernels.select_rows_for_sampling(
                torch.zeros(4), torch.tensor([0, 4], dtype=torch.int32)
            )
        with self.assertRaisesRegex(RuntimeError, "2D logits"):
            kernels.sample_tokens_rows(torch.zeros(4))

    def test_verified_store_enforces_slot_contract(self) -> None:
        vk = types.SimpleNamespace(
            store_kv_cache=types.SimpleNamespace(store_kv_cache=lambda *args: None)
        )
        k = torch.zeros((2, 1, 1))
        v = torch.zeros_like(k)
        kc = torch.zeros((1, kernels._BLOCK_SIZE, 1, 1))
        vc = torch.zeros_like(kc)
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                return_value={"sealed": 1},
            ),
            mock.patch.object(
                primitive_runtime,
                "kernel_entrypoint",
                return_value=vk.store_kv_cache.store_kv_cache,
            ),
        ):
            cases = (
                (torch.tensor([7], dtype=torch.int32), "one slot per input row"),
                (torch.tensor([7, kernels._BLOCK_SIZE], dtype=torch.int32), "cache bounds"),
                (torch.tensor([-2, 7], dtype=torch.int32), "cache bounds"),
                (torch.tensor([7, 7], dtype=torch.int32), "injective nonnegative"),
            )
            for slots, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(RuntimeError, message):
                        kernels.store_kv_cache(k, v, kc, vc, slots)

    def test_store_no_write_sentinel_preserves_cache(self) -> None:
        k = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
        v = torch.tensor([[[4.0]], [[5.0]], [[6.0]]])
        slots = torch.tensor([2, -1, -1], dtype=torch.int32)
        k_cache = torch.full((1, kernels._BLOCK_SIZE, 1, 1), -7.0)
        v_cache = torch.full_like(k_cache, -8.0)

        with mock.patch.object(primitive_runtime, "verified_for", return_value=None):
            kernels.store_kv_cache(k, v, k_cache, v_cache, slots)

        self.assertEqual(k_cache[0, 2, 0, 0].item(), 1.0)
        self.assertEqual(v_cache[0, 2, 0, 0].item(), 4.0)
        self.assertEqual(k_cache[0, -1, 0, 0].item(), -7.0)
        self.assertEqual(v_cache[0, -1, 0, 0].item(), -8.0)

    def test_store_fallback_uses_cache_page_size_and_last_write_wins(self) -> None:
        k = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])
        v = torch.tensor([[[7.0, 8.0]], [[9.0, 10.0]], [[11.0, 12.0]]])
        slots = torch.tensor([2, 3, 3], dtype=torch.int64)

        k_cache = torch.full((2, 3, 1, 2), -1.0)
        v_cache = torch.full_like(k_cache, -2.0)
        with mock.patch.object(primitive_runtime, "verified_for", return_value=None):
            kernels.store_kv_cache(k, v, k_cache, v_cache, slots)

        self.assertTrue(torch.equal(k_cache[0, 2], k[0]))
        self.assertTrue(torch.equal(v_cache[0, 2], v[0]))
        self.assertTrue(torch.equal(k_cache[1, 0], k[2]))
        self.assertTrue(torch.equal(v_cache[1, 0], v[2]))
        self.assertEqual(k_cache[0, 0, 0, 0].item(), -1.0)
        self.assertEqual(v_cache[0, 0, 0, 0].item(), -2.0)

        toy_k_cache = torch.full((2, 3), -1.0)
        toy_v_cache = torch.full_like(toy_k_cache, -2.0)
        kernels.store_kv_cache(
            k.reshape(3, -1),
            v.reshape(3, -1),
            toy_k_cache,
            toy_v_cache,
            slots,
        )
        self.assertEqual(toy_k_cache[0, 2].item(), k[0].mean().item())
        self.assertEqual(toy_v_cache[0, 2].item(), v[0].mean().item())
        self.assertEqual(toy_k_cache[1, 0].item(), k[2].mean().item())
        self.assertEqual(toy_v_cache[1, 0].item(), v[2].mean().item())

    def test_verified_store_runtime_contract_fails_closed(self) -> None:
        def validate(**overrides):
            arguments = {
                "k": torch.zeros((2, 1, 1)),
                "v": torch.ones((2, 1, 1)),
                "k_cache": torch.zeros((1, kernels._BLOCK_SIZE, 1, 1)),
                "v_cache": torch.ones((1, kernels._BLOCK_SIZE, 1, 1)),
                "slot_mapping": torch.tensor([0, 63], dtype=torch.int32),
            }
            arguments.update(overrides)
            return kernels._store_kv_cache_runtime_contract(**arguments)

        self.assertIsNone(validate())
        short_pages = torch.zeros((1, 32, 1, 1))
        wide_rows = torch.zeros((2, 2, 1))
        noncontiguous_slots = torch.tensor([0, 4, 63, 5], dtype=torch.int32)[::2]
        shared_cache = torch.zeros((1, kernels._BLOCK_SIZE, 1, 1))
        cases = (
            ({"v": torch.zeros((1, 1, 1))}, "matching K/V row shapes"),
            ({"v_cache": torch.zeros((2, kernels._BLOCK_SIZE, 1, 1))},
             "matching K/V cache shapes"),
            ({"k_cache": short_pages, "v_cache": torch.zeros_like(short_pages)},
             "64-token pages"),
            ({"k": wide_rows, "v": torch.zeros_like(wide_rows)},
             "source rows to match cache rows"),
            ({"v": torch.ones((2, 1, 1), dtype=torch.float64)},
             "implicit dtype conversion"),
            ({"slot_mapping": noncontiguous_slots}, "contiguous tensors"),
            ({"slot_mapping": torch.tensor([0, 63], dtype=torch.int64)},
             "1D int32"),
            ({"k_cache": shared_cache, "v_cache": shared_cache},
             "independent storage"),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    validate(**overrides)

    def test_verified_store_dispatch_preserves_exact_roles(self) -> None:
        def scatter(k, v, k_cache, v_cache, slots, *, launch_config):
            self.assertEqual(launch_config, {"sealed": 1})
            indices = slots.to(torch.int64)
            k_cache.view(-1, 2).index_copy_(0, indices, k.view(-1, 2))
            v_cache.view(-1, 2).index_copy_(0, indices, v.view(-1, 2))

        vk = types.SimpleNamespace(
            store_kv_cache=types.SimpleNamespace(store_kv_cache=scatter)
        )
        k = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        v = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])
        k_cache = torch.full((1, kernels._BLOCK_SIZE, 1, 2), -1.0)
        v_cache = torch.full_like(k_cache, -2.0)
        old_k_pointer, old_v_pointer = k_cache.data_ptr(), v_cache.data_ptr()
        slots = torch.tensor([63, 0], dtype=torch.int32)
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                return_value={"sealed": 1},
            ),
            mock.patch.object(
                primitive_runtime,
                "kernel_entrypoint",
                return_value=scatter,
            ),
        ):
            kernels.store_kv_cache(k, v, k_cache, v_cache, slots)

        self.assertTrue(torch.equal(k_cache[0, 63], k[0]))
        self.assertTrue(torch.equal(k_cache[0, 0], k[1]))
        self.assertTrue(torch.equal(v_cache[0, 63], v[0]))
        self.assertTrue(torch.equal(v_cache[0, 0], v[1]))
        self.assertEqual(k_cache[0, 1, 0, 0].item(), -1.0)
        self.assertEqual(v_cache[0, 1, 0, 0].item(), -2.0)
        self.assertEqual(k_cache.data_ptr(), old_k_pointer)
        self.assertEqual(v_cache.data_ptr(), old_v_pointer)

    def test_verified_caller_store_skips_device_value_validation(self) -> None:
        observed = {}

        def scatter(*args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs

        vk = types.SimpleNamespace(
            store_kv_cache=types.SimpleNamespace(store_kv_cache=scatter)
        )
        k = torch.zeros((2, 1, 1))
        v = torch.ones_like(k)
        k_cache = torch.zeros((1, kernels._BLOCK_SIZE, 1, 1))
        v_cache = torch.ones_like(k_cache)
        slots = torch.tensor([0, 63], dtype=torch.int32)
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                return_value={"sealed": 1},
            ),
            mock.patch.object(
                primitive_runtime,
                "kernel_entrypoint",
                return_value=scatter,
            ),
            mock.patch.object(
                kernels,
                "_store_kv_cache_runtime_contract",
                side_effect=AssertionError("must not copy device slots"),
            ),
            mock.patch.object(
                kernels,
                "store_kv_cache",
                side_effect=AssertionError("must not enter checked fallback"),
            ),
        ):
            kernels.store_kv_cache_from_verified_caller(
                k, v, k_cache, v_cache, slots
            )

        for actual, expected in zip(
            observed["args"], (k, v, k_cache, v_cache, slots)
        ):
            self.assertIs(actual, expected)
        self.assertEqual(observed["kwargs"]["launch_config"], {"sealed": 1})

    def test_view_as_kv_returns_independent_storage(self) -> None:
        """The Verus permission model treats the result as freshly owned."""
        runtime = self._configure_cpu_qwen()
        original = torch.arange(2 * 8 * 128, dtype=torch.float32).reshape(2, -1)
        viewed = kernels.view_as_kv(original, runtime)

        self.assertEqual(tuple(viewed.shape), (2, 8, 128))
        self.assertTrue(torch.equal(viewed.reshape(2, -1), original))
        self.assertTrue(viewed.is_contiguous())
        self.assertNotEqual(viewed.data_ptr(), original.data_ptr())
        viewed[0, 0, 0] = -1.0
        self.assertEqual(original[0, 0].item(), 0.0)

    def test_sample_is_deterministic_argmax_with_explicit_empty_case(self) -> None:
        logits = torch.tensor([-3.0, 7.0, 7.0, 2.0])
        self.assertEqual(kernels.sample(logits), 1)
        self.assertEqual(kernels.sample(logits.clone()), 1)
        self.assertEqual(kernels.sample(torch.empty(0)), 0)

    def test_embed_uses_the_explicit_weight_argument(self) -> None:
        ids = torch.tensor([2, 0], dtype=torch.int64)
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.assertTrue(torch.equal(kernels.embed(ids, weight), weight[ids]))

    def test_deployed_shape_guard_is_exact(self) -> None:
        small = qwen3_runtime.runtime_for_tests(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=1e-6,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        small_config = dict(small.runtime_config())
        self.assertIsNone(qwen3_runtime._FAMILY.verified_scope_error(small_config))
        self.assertEqual(
            qwen3_runtime._FAMILY.matched_scope_shape(small_config)["name"],
            "qwen3-0.6b",
        )

        large = qwen3_runtime.runtime_for_tests(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=12288,
            rms_norm_eps=1e-6,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        large_config = dict(large.runtime_config())
        self.assertIsNone(qwen3_runtime._FAMILY.verified_scope_error(large_config))
        self.assertEqual(
            qwen3_runtime._FAMILY.matched_scope_shape(large_config)["name"],
            "qwen3-8b",
        )

        wrong_heads = qwen3_runtime.runtime_for_tests(
            hidden_size=1024,
            num_heads=8,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=1e-6,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        self.assertIn(
            "outside",
            qwen3_runtime._FAMILY.verified_scope_error(
                dict(wrong_heads.runtime_config())
            ),
        )

        wrong_dtype = qwen3_runtime.runtime_for_tests(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=1e-6,
            device="cuda:0",
            dtype=torch.float16,
        )
        self.assertIn(
            "outside",
            qwen3_runtime._FAMILY.verified_scope_error(
                dict(wrong_dtype.runtime_config())
            ),
        )

    def test_test_runtime_cannot_be_promoted_to_qualified(self) -> None:
        runtime = qwen3_runtime.runtime_for_tests(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "sealed backend bundle"):
            qwen3_runtime.QualifiedRuntime(runtime)

    def test_environment_cannot_select_an_alternate_serving_path(self) -> None:
        fake_cuda_tensor = types.SimpleNamespace(is_cuda=True)
        verified = types.SimpleNamespace()
        runtime = qwen3_runtime.runtime_for_tests(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=1e-6,
        )
        with mock.patch.object(runtime, "_test_only", False), mock.patch.object(
            runtime, "_kernel_namespace", verified
        ), mock.patch.dict(
            os.environ, {"VOSTI_VERIFIED_KERNELS": "0"}, clear=False
        ):
            self.assertIs(runtime.verified_for(fake_cuda_tensor), verified)

    def test_production_configuration_requires_a_sealed_bundle(self) -> None:
        runtime = qwen3_runtime.runtime_for_tests(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "sealed backend bundle"):
            qwen3_runtime.QualifiedRuntime(runtime)

    def test_configured_strict_path_rejects_non_cuda_calls(self) -> None:
        runtime = qwen3_runtime.runtime_for_tests(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=1e-6,
        )
        with mock.patch.object(runtime, "_test_only", False):
            with self.assertRaisesRegex(RuntimeError, "non-CUDA tensor"):
                runtime.verified_for(torch.zeros(1))

    def test_explicit_test_configuration_allows_cpu_helpers(self) -> None:
        runtime = self._configure_cpu_qwen()
        self.assertIsNone(runtime.verified_for(torch.zeros(1)))

    def test_framework_package_origin_is_bound_to_explicit_root(self) -> None:
        root = Path(__file__).resolve().parents[2]
        fake_package = types.ModuleType("vosti_kernels")
        fake_package.__file__ = "/tmp/unreviewed/vosti_kernels/__init__.py"
        fake_package.__path__ = ["/tmp/unreviewed/vosti_kernels"]
        fake_package.__spec__ = types.SimpleNamespace(origin=fake_package.__file__)
        with mock.patch.dict(
            os.environ, {"VOSTI_FRAMEWORK_ROOT": str(root)}, clear=False
        ), mock.patch.dict(
            sys.modules, {"vosti_kernels": fake_package}, clear=False
        ):
            with self.assertRaisesRegex(RuntimeError, "loaded from"):
                from vosti_kernels.static_runtime import attest_framework_package
                attest_framework_package("qwen3", root)

    def test_preloaded_kernel_package_from_another_root_is_rejected(self) -> None:
        root = Path(__file__).resolve().parents[2]
        kernel_root = root / "kernels"
        fake_modules = {}
        for name in (
            "triton_kernels",
            *(
                f"triton_kernels.{name}"
                for name in qwen3_runtime.ENGINE_KERNEL_MODULES
            ),
        ):
            module = types.ModuleType(name)
            module.__file__ = f"/tmp/unreviewed/{name.replace('.', '/')}.py"
            module.__spec__ = types.SimpleNamespace(origin=module.__file__)
            if name == "triton_kernels":
                module.__path__ = ["/tmp/unreviewed/triton_kernels"]
            fake_modules[name] = module
        with mock.patch.dict(
            os.environ,
            {
                "VOSTI_FRAMEWORK_ROOT": str(root),
                "VOSTI_KERNEL_ROOT": str(kernel_root),
            },
            clear=False,
        ), mock.patch.dict(sys.modules, fake_modules, clear=False):
            with self.assertRaisesRegex(RuntimeError, "loaded from"):
                from vosti_kernels.kernel_modules import load_attested_modules
                load_attested_modules(str(kernel_root), qwen3_runtime._FAMILY.scope, label="Qwen3")

    def test_catalog_entries_are_reachable_from_the_verified_engine(self) -> None:
        root = Path(__file__).resolve().parents[2]
        execution_sources = [
            root / "src" / "exec" / "model.rs",
            root / "src" / "exec" / "engine.rs",
            root / "src" / "exec" / "dense_swiglu_decoder.rs",
            root / "src" / "exec" / "dense_swiglu_model.rs",
            root / "src" / "boundary" / "dense_swiglu_decoder.rs",
            *(root / "src" / "exec" / "model_families").rglob("*.rs"),
            *(
                root / "src" / "boundary" / "model_families" / "qwen3"
            ).rglob("*.rs"),
        ]
        engine_source = "\n".join(
            path.read_text(encoding="utf-8") for path in execution_sources
        )
        runtime_calls = set(
            re.findall(
                r"\b(?:RT|DLP|QWEN_BOUNDARY)::([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                engine_source,
            )
        )
        catalog = {
            contract["wrapper"]
            for contract in qwen3_runtime.ENGINE_KERNEL_CONTRACTS
        }
        self.assertLessEqual(catalog, runtime_calls)
        self.assertIn("select_last_hidden_rows", runtime_calls)
        self.assertIn("sample_tokens_rows", runtime_calls)
        self.assertNotIn("add", catalog)


if __name__ == "__main__":
    unittest.main()
