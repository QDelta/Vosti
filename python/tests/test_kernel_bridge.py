"""Adversarial checks for framework-kernel bridge attestations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest

from spec_test_support import page_constants_source
from unittest import mock

import torch

from vosti_kernels import kernels, primitive_runtime
from vosti_kernels.kernel_interfaces import RECTANGULAR_KERNEL_INTERFACES
from vosti_kernels.model_families.llama3 import runtime as llama3_runtime
from vosti_kernels.model_families.qwen3 import runtime as qwen3_runtime


def rectangular_fixture(raw, axis, fragment, binding):
    """Exercise the current adapter directly, without the retired manifest wrapper."""
    return types.SimpleNamespace(body=fragment.body + render_raw_rectangular_axis_adapter(
        raw, axis, fragment, binding))


def _qwen_test_runtime(**kwargs):
    return qwen3_runtime.runtime_for_tests(**kwargs)


def _llama_test_runtime(**kwargs):
    return llama3_runtime.runtime_for_tests(**kwargs)


ROOT = Path(__file__).resolve().parents[2]
KERNEL_ROOT = next(
    (
        path
        for path in (ROOT / "kernels",)
        if (path / "triton_kernels" / "matmul.py").is_file()
    ),
    None,
)
if KERNEL_ROOT is not None:
    sys.path.insert(0, str(KERNEL_ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from scripts.deployment.model_families.qwen3_scope import (  # noqa: E402
        SCOPE as qwen3_scope,
        contract_for,
        linear_static_policy,
        validate_bridge_surfaces,
        validate_contract_catalog,
        validate_post_surface,
    )
    from scripts.audit.runtime_bridge_scope import (  # noqa: E402
        runtime_bridge_for,
        validate_runtime_bridge_surfaces,
    )
    from ir.axis_projection_contract import (  # noqa: E402
        VerifiedAxisProjectionContract,
        normalize_axis_projection_contract,
    )
    from ir.relational_dataflow import (  # noqa: E402
        prove_relational_dataflow_from_annotations,
    )
    from ir.relational_artifact import VerifiedDataflowContract  # noqa: E402
    from ir.relational_verifier import verify_annotations, validate_dataflow_artifact  # noqa: E402
    from ir.verus_contract import (  # noqa: E402
        render_verified_contract_to_verus, render_verified_kernel_to_verus,
    )
    from scripts.verification.kernel_contract_codegen import (  # noqa: E402
        render_raw_rectangular_axis_adapter,
        validate_axis_projection_binding,
        validate_artifact_origin,
    )
    from scripts.verification.rectangular_interface_codegen import inferred_binding
    from scripts.audit.claim_ledger import (  # noqa: E402
        SURFACE_PATH,
        certificate_import_sets,
        read_json,
        validate_all,
    )
    import scripts.audit.claim_ledger as claim_ledger_module  # noqa: E402
    from scripts.audit.tcb import (  # noqa: E402
        rust_source_files,
        scan_rust,
        scan_external_type_records,
        scan_rust_trust,
        scan_rust_trust_records,
    )
    from scripts.audit.check_verus_trust_manifest import compiler_trust_records  # noqa: E402


@unittest.skipIf(KERNEL_ROOT is None, "pinned kernel checkout is unavailable")
class KernelBridgeSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (KERNEL_ROOT / "triton_kernels" / "matmul.py").read_text(
            encoding="utf-8"
        )
        cls.silu_source = (
            KERNEL_ROOT / "triton_kernels" / "silu_mul.py"
        ).read_text(encoding="utf-8")
        cls.embedding_source = (
            KERNEL_ROOT / "triton_kernels" / "embedding.py"
        ).read_text(encoding="utf-8")
        cls.rms_source = (
            KERNEL_ROOT / "triton_kernels" / "rmsnorm.py"
        ).read_text(encoding="utf-8")
        cls.residual_rms_source = (
            KERNEL_ROOT / "triton_kernels" / "rmsnorm_residual.py"
        ).read_text(encoding="utf-8")
        cls.qk_norm_source = (
            KERNEL_ROOT / "triton_kernels" / "qk_norm.py"
        ).read_text(encoding="utf-8")
        cls.qkv_source = (
            KERNEL_ROOT / "triton_kernels" / "qkv_matmul.py"
        ).read_text(encoding="utf-8")
        cls.rope_source = (
            KERNEL_ROOT / "triton_kernels" / "rope.py"
        ).read_text(encoding="utf-8")
        cls.store_source = (
            KERNEL_ROOT / "triton_kernels" / "store_kv_cache.py"
        ).read_text(encoding="utf-8")
        cls.attention_source = (
            KERNEL_ROOT / "triton_kernels" / "fattn_paged.py"
        ).read_text(encoding="utf-8")
        cls.runtime_contract_source = (
            KERNEL_ROOT / "triton_kernels" / "runtime_contracts.py"
        ).read_text(encoding="utf-8")
        cls.framework_source = (
            ROOT / "src" / "boundary" / "tensor_runtime.rs"
        ).read_text(encoding="utf-8")
        cls.qwen_weights_source = (
            ROOT
            / "src"
            / "boundary"
            / "model_families"
            / "qwen3"
            / "weights.rs"
        ).read_text(encoding="utf-8")
        cls.dense_weights_source = (
            ROOT / "src" / "boundary" / "dense_swiglu_decoder.rs"
        ).read_text(encoding="utf-8")
        cls.gemma_weights_source = (
            ROOT
            / "src"
            / "boundary"
            / "model_families"
            / "gemma3"
            / "weights.rs"
        ).read_text(encoding="utf-8")
        cls.llama_weights_source = (
            ROOT
            / "python"
            / "vosti_kernels"
            / "model_families"
            / "llama3"
            / "physical.py"
        ).read_text(encoding="utf-8")
        cls.python_source = (
            ROOT / "python" / "vosti_kernels" / "kernels.py"
        ).read_text(encoding="utf-8")
        cls.physical_source = (
            ROOT / "python" / "vosti_kernels" / "physical.py"
        ).read_text(encoding="utf-8")
        cls.qwen_physical_source = (
            ROOT / "python" / "vosti_kernels" / "model_families" / "qwen3" / "physical.py"
        ).read_text(encoding="utf-8")
        cls.gemma_physical_source = (
            ROOT / "python" / "vosti_kernels" / "model_families" / "gemma3" / "physical.py"
        ).read_text(encoding="utf-8")
        cls.gemma_loader_source = (
            ROOT / "python" / "vosti_kernels" / "model_families" / "gemma3" / "loader.py"
        ).read_text(encoding="utf-8")
        cls.qwen_loader_source = (
            ROOT / "python" / "vosti_kernels" / "model_families" / "qwen3" / "loader.py"
        ).read_text(encoding="utf-8")
        cls.llama_loader_source = (
            ROOT / "python" / "vosti_kernels" / "model_families" / "llama3" / "loader.py"
        ).read_text(encoding="utf-8")
        cls.contract = contract_for("matmul.py", "matmul_kernel")
        cls.silu_contract = contract_for("silu_mul.py", "silu_mul_kernel")
        cls.embedding_contract = contract_for("embedding.py", "embedding_kernel")
        cls.rms_contract = contract_for("rmsnorm.py", "rmsnorm_kernel")
        cls.residual_rms_contract = contract_for(
            "rmsnorm_residual.py", "rmsnorm_residual_kernel"
        )
        cls.qk_norm_contract = contract_for(
            "qk_norm.py", "head_rms_norm_kernel"
        )
        cls.rope_contract = contract_for("rope.py", "rope_kernel")
        cls.store_contract = contract_for(
            "store_kv_cache.py", "store_cache_kernel"
        )
        cls.attention_contract = contract_for(
            "fattn_paged.py", "fattn_varlen_paged_fwd_block_ptr_kernel"
        )
        cls.init_kv_caches_bridge = runtime_bridge_for("init_kv_caches")
        cls.model_weights_bridge = runtime_bridge_for("model_weights")
        cls.gemma_checkpoint_loader_bridge = runtime_bridge_for(
            "gemma3_checkpoint_loader"
        )
        cls.qwen_checkpoint_loader_bridge = runtime_bridge_for(
            "qwen3_checkpoint_loader"
        )
        cls.llama_checkpoint_loader_bridge = runtime_bridge_for(
            "llama3_checkpoint_loader"
        )
        cls.step_plan_materializers_bridge = runtime_bridge_for(
            "step_plan_materializers"
        )
        cls.exact_runtime_adapters_bridge = runtime_bridge_for(
            "exact_runtime_adapters"
        )
        cls.matmul_config = {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}
        cls.silu_config = {"BLOCK_M": 1, "BLOCK_N": 1024}
        cls.embedding_config = {"D": 1024, "BLOCK_M": 1, "BLOCK_D": 1024}
        cls.embedding_wide_config = {
            "D": 4096,
            "BLOCK_M": 1,
            "BLOCK_D": 4096,
        }
        cls.rms_config = {"BLOCK_M": 1, "BLOCK_N": 1024}
        cls.rms_wide_config = {"BLOCK_M": 1, "BLOCK_N": 4096}
        cls.residual_rms_config = {"BLOCK_M": 1, "BLOCK_N": 1024}
        cls.residual_rms_wide_config = {"BLOCK_M": 1, "BLOCK_N": 4096}
        cls.q_head_norm_config = {"H": 16, "D": 128, "BLOCK_M": 1}
        cls.q_head_norm_32_config = {"H": 32, "D": 128, "BLOCK_M": 1}
        cls.k_head_norm_config = {"H": 8, "D": 128, "BLOCK_M": 1}
        cls.rope_config = {"D": 128, "HD": 64, "BLOCK_M": 1}
        cls.qkv_config = {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128}
        cls.attention_config = {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "D_HEAD": 128,
            "PAGE_BLOCK_SIZE": 64,
        }

        structural = verify_annotations(
            cls.attention_source,
            "fattn_varlen_paged_fwd_block_ptr_kernel",
            cls.attention_config,
            goal_name="batch_invariance",
        )
        if not structural.proved or structural.verified_contract is None:
            raise AssertionError("deployed attention structural proof failed")
        cls.attention_raw = structural.verified_contract
        cls.attention_selected_dataflow = prove_relational_dataflow_from_annotations(
            cls.attention_source,
            "fattn_varlen_paged_fwd_block_ptr_kernel",
            cls.attention_config,
            goal_name="selected_row_prefix_equivalence",
        )
        if not cls.attention_selected_dataflow.proved:
            raise AssertionError("deployed attention generic dataflow proof failed")
        cls.attention_selected = cls.attention_selected_dataflow.verified_contract

    def linear_axis_contract(
        self, source: str | None = None, config: dict | None = None
    ):
        result = verify_annotations(
            self.source if source is None else source,
            "matmul_kernel",
            self.matmul_config if config is None else config,
        )
        self.assertTrue(result.proved)
        self.assertIsNotNone(result.verified_contract)
        return normalize_axis_projection_contract(result.verified_contract)

    def silu_axis_contract(
        self, source: str | None = None, config: dict | None = None
    ):
        result = verify_annotations(
            self.silu_source if source is None else source,
            "silu_mul_kernel",
            self.silu_config if config is None else config,
        )
        self.assertTrue(result.proved)
        self.assertIsNotNone(result.verified_contract)
        return normalize_axis_projection_contract(result.verified_contract)

    def silu_generated_certificate(
        self, proof_name: str, config: dict | None = None
    ):
        selected_config = self.silu_config if config is None else config
        return self.raw_generated_certificate(
            self.silu_source,
            "silu_mul_kernel",
            selected_config,
            proof_name,
        )

    def raw_generated_certificate(
        self,
        source: str,
        kernel: str,
        config: dict,
        proof_name: str,
    ):
        result = verify_annotations(
            source,
            kernel,
            config,
        )
        self.assertTrue(result.proved)
        self.assertIsNotNone(result.verified_contract)
        raw = result.verified_contract
        axis = normalize_axis_projection_contract(raw)
        prefix = proof_name.removesuffix("_certificate") + "_raw"
        fragment = render_verified_contract_to_verus(raw, symbol_prefix=prefix)
        binding = inferred_binding(axis, raw, fragment)
        binding["proof_name"] = proof_name
        return rectangular_fixture(raw, axis, fragment, binding)

    def embedding_axis_contract(
        self, source: str | None = None, config: dict | None = None
    ):
        result = verify_annotations(
            self.embedding_source if source is None else source,
            "embedding_kernel",
            self.embedding_config if config is None else config,
        )
        self.assertTrue(result.proved)
        self.assertIsNotNone(result.verified_contract)
        return normalize_axis_projection_contract(result.verified_contract)

    def rms_axis_contract(
        self, source: str | None = None, config: dict | None = None
    ):
        result = verify_annotations(
            self.rms_source if source is None else source,
            "rmsnorm_kernel",
            self.rms_config if config is None else config,
        )
        self.assertTrue(result.proved)
        self.assertIsNotNone(result.verified_contract)
        return normalize_axis_projection_contract(result.verified_contract)

    def residual_rms_axis_contract(
        self, source: str | None = None, config: dict | None = None
    ):
        result = verify_annotations(
            self.residual_rms_source if source is None else source,
            "rmsnorm_residual_kernel",
            self.residual_rms_config if config is None else config,
        )
        self.assertTrue(result.proved)
        self.assertIsNotNone(result.verified_contract)
        return normalize_axis_projection_contract(result.verified_contract)

    def qk_norm_axis_contract(self, config: dict):
        result = verify_annotations(
            self.qk_norm_source,
            "head_rms_norm_kernel",
            config,
        )
        self.assertTrue(result.proved)
        self.assertIsNotNone(result.verified_contract)
        return normalize_axis_projection_contract(result.verified_contract)

    def rope_axis_contract(self, source: str | None = None):
        result = verify_annotations(
            self.rope_source if source is None else source,
            "rope_kernel",
            self.rope_config,
        )
        self.assertTrue(result.proved)
        self.assertIsNotNone(result.verified_contract)
        return normalize_axis_projection_contract(result.verified_contract)


    def test_real_linear_bridge_matches_annotation(self) -> None:
        validate_contract_catalog()
        validate_bridge_surfaces(self.contract, self.source)

    def test_conditional_contract_rejects_retired_manual_bindings(self) -> None:
        contracts = deepcopy(list(qwen3_scope.contracts))
        attention = next(c for c in contracts
                         if c["evidence"] == "conditional_relational_certificate")
        self.assertNotIn("semantic_contracts", attention["bridge"])
        attention["bridge"]["semantic_contracts"] = [{"kind": "axis_projection"}]
        with mock.patch.object(qwen3_scope, "contracts", tuple(contracts)):
            with self.assertRaisesRegex(ValueError, "invalid bridge contract fields"):
                validate_contract_catalog()

    def test_registered_row_contracts_have_no_manual_binding(self) -> None:
        for contract in qwen3_scope.contracts:
            if (contract["source"], contract["kernel"]) in RECTANGULAR_KERNEL_INTERFACES:
                self.assertNotIn("semantic_contracts", contract["bridge"])
        validate_contract_catalog()

    def test_config_erased_linear_contract_rejects_semantic_binding(self) -> None:
        contracts = list(qwen3_scope.contracts)
        linear_index = next(
            index
            for index, contract in enumerate(contracts)
            if contract["wrapper"] == "linear"
        )
        linear = deepcopy(contracts[linear_index])
        linear["bridge"]["semantic_contracts"] = [{"kind": "axis_projection"}]
        contracts[linear_index] = linear
        with mock.patch.object(qwen3_scope, "contracts", tuple(contracts)):
            with self.assertRaisesRegex(ValueError, "invalid bridge contract fields"):
                validate_contract_catalog()

    def test_artifact_origin_rejects_source_or_specialization_substitution(self) -> None:
        axis = self.linear_axis_contract()
        validate_artifact_origin(
            self.contract,
            axis,
            source=self.source,
            constants=self.matmul_config,
        )
        with self.assertRaisesRegex(ValueError, "whole kernel source"):
            validate_artifact_origin(
                self.contract,
                axis,
                source=self.source + "\n# drift",
                constants=self.matmul_config,
            )
        wrong_config = {**self.matmul_config, "BLOCK_M": 32}
        with self.assertRaisesRegex(ValueError, "specialization differs"):
            validate_artifact_origin(
                self.contract,
                axis,
                source=self.source,
                constants=wrong_config,
            )
        wrong_contract = {**self.contract, "kernel": "embedding_kernel"}
        with self.assertRaisesRegex(ValueError, "different kernel"):
            validate_artifact_origin(
                wrong_contract,
                axis,
                source=self.source,
                constants=self.matmul_config,
            )

    def test_attention_artifact_rejects_changed_causal_mask(self) -> None:
        mutated = self.attention_source.replace(
            "attn_mask &= k_indices[None, :] <= (q_indices[:, None] + q_shift)",
            "attn_mask &= k_indices[None, :] > (q_indices[:, None] + q_shift)",
            1,
        )
        self.assertNotEqual(mutated, self.attention_source)
        with self.assertRaisesRegex(ValueError, "whole kernel source"):
            validate_artifact_origin(
                self.attention_contract,
                self.attention_raw,
                source=mutated,
                constants=self.attention_config,
            )

    def test_attention_bridge_rejects_weakened_physical_layout_guard(self) -> None:
        mutated = self.runtime_contract_source.replace(
            "if not tensor.is_contiguous():\n            raise ValueError(\n"
            "                f\"launch tensor",
            "if False:\n            raise ValueError(\n"
            "                f\"launch tensor",
            1,
        )
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.attention_contract,
                self.attention_source,
                source_overrides={
                    "kernels/triton_kernels/runtime_contracts.py": mutated
                },
            )

    def test_real_kv_allocator_runtime_bridge_matches_reviewed_sources(self) -> None:
        validate_runtime_bridge_surfaces(self.init_kv_caches_bridge)

    def test_four_norm_primitive_bridge_pins_policy_and_dispatch(self) -> None:
        bridge = runtime_bridge_for("qualified_kernel_plan_capability")
        path = "src/boundary/four_norm_gated_primitives.rs"
        source = (ROOT / path).read_text(encoding="utf-8")
        for before, after in (
            ("runtime_norm_policy(runtime) == Some(policy)", "true"),
            ("runtime_qk_norm_policy(runtime, attention_kind) == Some(policy)", "true"),
            ("runtime_rotary_config(runtime, attention_kind) == Some(rotary)", "true"),
            ("runtime_policy(runtime).unwrap().value_norm_epsilon == Some(epsilon)", "true"),
            ("runtime_attention_geometry(runtime, attention_kind) == Some(geometry)", "true"),
            ("runtime_policy(runtime).unwrap().layer_scale,", "true,"),
            ("runtime_policy(runtime).unwrap().logits_softcap == Some(cap)", "true"),
            ('call_method1("value_norm",', 'call_method1("qk_norm",'),
            ('call_method1("scale",', 'call_method1("add",'),
            ('call_method1("softcap",', 'call_method1("rms_norm",'),
            ("attention_kind_name(attention_kind),", "attention_kind_name(AttentionKind::Full),"),
            ('"scaled_embed",', '"embed",'),
            ("ROWS::norm_row(input[i], weight, policy)",
             "ROWS::norm_row(input[i], weight, NormPolicyRepr::UnitOffset)"),
        ):
            with self.subTest(before=before):
                mutated = source.replace(before, after, 1)
                self.assertNotEqual(source, mutated)
                with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
                    validate_runtime_bridge_surfaces(bridge, source_overrides={path: mutated})

    def test_layer_kv_allocator_bridge_pins_independent_return_validation(self) -> None:
        path = "src/boundary/tensor_runtime.rs"
        source = (ROOT / path).read_text(encoding="utf-8")
        mutated = source.replace('getattr("validate_layer_model_kv_caches")',
                                 'getattr("init_layer_model_kv_caches")', 1)
        self.assertNotEqual(source, mutated)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(self.init_kv_caches_bridge,
                                            source_overrides={path: mutated})

    def test_real_model_weights_runtime_bridge_matches_reviewed_sources(self) -> None:
        validate_runtime_bridge_surfaces(self.model_weights_bridge)

    def test_real_gemma_checkpoint_loader_bridge_matches_reviewed_sources(self) -> None:
        validate_runtime_bridge_surfaces(self.gemma_checkpoint_loader_bridge)

    def test_real_qwen_checkpoint_loader_bridge_matches_reviewed_sources(self) -> None:
        validate_runtime_bridge_surfaces(self.qwen_checkpoint_loader_bridge)

    def test_real_llama_checkpoint_loader_bridge_matches_reviewed_sources(self) -> None:
        validate_runtime_bridge_surfaces(self.llama_checkpoint_loader_bridge)

    def test_gemma_checkpoint_bridge_rejects_changed_role_mapping(self) -> None:
        mutated = self.gemma_loader_source.replace(
            '"q_proj": f"{prefix}.self_attn.q_proj.weight",',
            '"q_proj": f"{prefix}.self_attn.k_proj.weight",',
            1,
        )
        self.assertNotEqual(mutated, self.gemma_loader_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.gemma_checkpoint_loader_bridge,
                source_overrides={
                    "python/vosti_kernels/model_families/gemma3/loader.py": mutated
                },
            )

    def test_four_norm_checkpoint_bridges_pin_shared_role_deserializer(self) -> None:
        path = "src/boundary/four_norm_gated_weights.rs"
        source = (ROOT / path).read_text(encoding="utf-8")
        mutated = source.replace(
            "let [input_norm, q_proj, k_proj, v_proj, q_norm, k_norm, o_proj,",
            "let [input_norm, q_proj, v_proj, k_proj, q_norm, k_norm, o_proj,", 1,
        )
        self.assertNotEqual(mutated, source)
        for family in ("gemma3", "gemma4"):
            with self.subTest(family=family), self.assertRaisesRegex(
                ValueError, "source span .* digest mismatch"
            ):
                validate_runtime_bridge_surfaces(
                    runtime_bridge_for(f"{family}_checkpoint_loader"),
                    source_overrides={path: mutated},
                )

    def test_gemma4_checkpoint_bridge_pins_optional_parameter_and_kv_policy(self) -> None:
        bridge = runtime_bridge_for("gemma4_checkpoint_loader")
        validate_runtime_bridge_surfaces(bridge)
        for path, before, after in (
            ("src/boundary/model_deployment.rs",
             "value.map(f64::to_bits)", "Some(1.0f64.to_bits())"),
            ("src/boundary/model_families/gemma4/deployment.rs",
             'loaded.config_usize["global_head_dim"]', 'loaded.config_usize["head_dim"]'),
            ("python/vosti_kernels/model_families/gemma4/loader.py",
             'keys["v_proj"] = keys["k_proj"]', 'keys["v_proj"] = keys["q_proj"]'),
        ):
            with self.subTest(path=path):
                source = (ROOT / path).read_text(encoding="utf-8")
                mutated = source.replace(before, after, 1)
                self.assertNotEqual(mutated, source)
                with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
                    validate_runtime_bridge_surfaces(bridge, source_overrides={path: mutated})

    def test_qwen_checkpoint_bridge_rejects_changed_role_mapping(self) -> None:
        mutated = self.qwen_loader_source.replace(
            "attn.q_proj.weight.contiguous(),",
            "attn.k_proj.weight.contiguous(),",
            1,
        )
        self.assertNotEqual(mutated, self.qwen_loader_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.qwen_checkpoint_loader_bridge,
                source_overrides={
                    "python/vosti_kernels/model_families/qwen3/loader.py": mutated
                },
            )

    def test_llama_checkpoint_bridge_rejects_changed_role_mapping(self) -> None:
        mutated = self.llama_loader_source.replace(
            '"q_proj": f"{prefix}.self_attn.q_proj.weight",',
            '"q_proj": f"{prefix}.self_attn.k_proj.weight",',
            1,
        )
        self.assertNotEqual(mutated, self.llama_loader_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.llama_checkpoint_loader_bridge,
                source_overrides={
                    "python/vosti_kernels/model_families/llama3/loader.py": mutated
                },
            )

    def test_model_weights_bridge_rejects_changed_verus_role_binding(self) -> None:
        mutated = self.dense_weights_source.replace(
            "weights.q_proj.id() == perms.q_proj.id()",
            "weights.q_proj.id() == perms.k_proj.id()",
            1,
        )
        self.assertNotEqual(mutated, self.dense_weights_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.model_weights_bridge,
                source_overrides={
                    "src/boundary/dense_swiglu_decoder.rs": mutated
                },
            )

    def test_model_weights_bridge_rejects_changed_gemma_role_binding(self) -> None:
        mutated = self.gemma_weights_source.replace(
            "&& weights.layer_scale.is_none()",
            "&& weights.layer_scale.is_some()",
            1,
        )
        self.assertNotEqual(mutated, self.gemma_weights_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.model_weights_bridge,
                source_overrides={
                    "src/boundary/model_families/gemma3/weights.rs": mutated
                },
            )

    def test_model_weights_bridge_rejects_changed_four_norm_role_binding(self) -> None:
        path = "src/boundary/four_norm_gated_weights.rs"
        source = (ROOT / path).read_text(encoding="utf-8")
        mutated = source.replace(
            "weights.pre_feedforward_norm.id()"
            " == perms.pre_feedforward_norm.id()",
            "weights.pre_feedforward_norm.id()"
            " == perms.post_feedforward_norm.id()",
            1,
        )
        self.assertNotEqual(mutated, source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.model_weights_bridge,
                source_overrides={
                    path: mutated
                },
            )

    def test_model_weights_bridge_rejects_changed_gemma4_binding_chain(self) -> None:
        for path, before, after in (
            ("src/boundary/four_norm_gated_weights.rs",
             "layer.q_proj.inner.bind(py),", "layer.k_proj.inner.bind(py),"),
            ("src/boundary/model_families/gemma4/weights.rs",
             "layer.k_proj.id() == layer.v_proj.id()",
             "layer.k_proj.id() == layer.q_proj.id()"),
            ("src/boundary/model_families/gemma4/weights.rs",
             '("global_head_dim", config.global_head_dim)',
             '("global_head_dim", config.geometry.head_dim)'),
            ("python/vosti_kernels/model_families/gemma4/physical.py",
             'roles["v_proj"] is not roles["k_proj"]',
             'roles["v_proj"] is roles["k_proj"]'),
            ("python/vosti_kernels/model_families/gemma4/loader.py",
             'self.attention_k_eq_v and self.layer_types[layer] == FULL_ATTENTION',
             'self.attention_k_eq_v and self.layer_types[layer] == SLIDING_ATTENTION'),
        ):
            with self.subTest(path=path, before=before):
                source = (ROOT / path).read_text(encoding="utf-8")
                mutated = source.replace(before, after, 1)
                self.assertNotEqual(mutated, source)
                with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
                    validate_runtime_bridge_surfaces(
                        self.model_weights_bridge, source_overrides={path: mutated},
                    )

    def test_model_weights_bridge_rejects_changed_llama_geometry(self) -> None:
        mutated = self.llama_weights_source.replace(
            "tuple(o_proj.shape) != (hidden_size, q_width)",
            "tuple(o_proj.shape) != (hidden_size, kv_width)",
            1,
        )
        self.assertNotEqual(mutated, self.llama_weights_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.model_weights_bridge,
                source_overrides={
                    "python/vosti_kernels/model_families/llama3/physical.py": mutated
                },
            )

    def test_model_weights_bridge_rejects_changed_physical_geometry(self) -> None:
        mutated = self.qwen_physical_source.replace(
            '"q_proj": (q_width, hidden),',
            '"q_proj": (hidden, q_width),',
            1,
        )
        self.assertNotEqual(mutated, self.qwen_physical_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.model_weights_bridge,
                source_overrides={
                    "python/vosti_kernels/model_families/qwen3/physical.py": mutated
                },
            )

    def test_real_step_plan_materializer_bridge_matches_reviewed_sources(self) -> None:
        validate_runtime_bridge_surfaces(self.step_plan_materializers_bridge)

    def test_real_exact_runtime_adapter_bridge_matches_reviewed_sources(self) -> None:
        validate_runtime_bridge_surfaces(self.exact_runtime_adapters_bridge)

    def test_exact_adapter_bridge_rejects_changed_verus_selection(self) -> None:
        mutated = self.framework_source.replace(
            "logits_repr[cu_q_repr[index as int + 1] - 1]",
            "logits_repr[cu_q_repr[index as int]]",
            1,
        )
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.exact_runtime_adapters_bridge,
                source_overrides={"src/boundary/tensor_runtime.rs": mutated},
            )

    def test_exact_adapter_bridge_rejects_changed_python_sampling(self) -> None:
        mutated = self.python_source.replace(
            "torch.argmax(rows, dim=1)", "torch.argmin(rows, dim=1)", 1
        )
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.exact_runtime_adapters_bridge,
                source_overrides={"python/vosti_kernels/kernels.py": mutated},
            )

    def test_step_plan_bridge_rejects_changed_verus_value_binding(self) -> None:
        mutated = self.framework_source.replace(
            "u64_seq_to_int_repr(tokens@)),",
            "u64_seq_to_int_repr(positions@)),",
            1,
        )
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.step_plan_materializers_bridge,
                source_overrides={"src/boundary/tensor_runtime.rs": mutated},
            )

    def test_step_plan_bridge_rejects_changed_python_padding(self) -> None:
        mutated = self.python_source.replace(
            "out = torch.zeros((len(block_ids), max_len), dtype=dtype, device=device)",
            "out = torch.empty((len(block_ids), max_len), dtype=dtype, device=device)",
            1,
        )
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.step_plan_materializers_bridge,
                source_overrides={"python/vosti_kernels/kernels.py": mutated},
            )

    def test_step_plan_bridge_rejects_weakened_int32_guard(self) -> None:
        mutated = self.python_source.replace(
            "if index < 0 or index > _INT32_MAX:",
            "if index < 0 or index > _INT32_MAX + 1:",
            1,
        )
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.step_plan_materializers_bridge,
                source_overrides={"python/vosti_kernels/kernels.py": mutated},
            )

    def test_kv_allocator_bridge_rejects_changed_verus_contract(self) -> None:
        mutated = self.framework_source.replace(
            "&& kv_perms_ids_distinct(perms@)",
            "&& true",
            1,
        )
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.init_kv_caches_bridge,
                source_overrides={"src/boundary/tensor_runtime.rs": mutated},
            )

    def test_kv_allocator_bridge_rejects_disabled_storage_guard(self) -> None:
        mutated = self.physical_source.replace(
            "if storage_identity in storage_owners:",
            "if False:",
            1,
        )
        self.assertNotEqual(mutated, self.physical_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.init_kv_caches_bridge,
                source_overrides={"python/vosti_kernels/physical.py": mutated},
            )

    def test_kv_allocator_bridge_rejects_changed_allocator(self) -> None:
        mutated = self.physical_source.replace(
            "factory = torch.zeros if zero_initialize else torch.empty",
            "factory = torch.empty if zero_initialize else torch.zeros",
            1,
        )
        self.assertNotEqual(mutated, self.physical_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.init_kv_caches_bridge,
                source_overrides={"python/vosti_kernels/physical.py": mutated},
            )

    def test_kv_allocator_bridge_rejects_changed_model_geometry(self) -> None:
        mutated = self.physical_source.replace(
            "return (kv_width // head_dim, head_dim), dtypes.pop(), devices.pop()",
            "return (1, head_dim), dtypes.pop(), devices.pop()",
            1,
        )
        self.assertNotEqual(mutated, self.physical_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_runtime_bridge_surfaces(
                self.init_kv_caches_bridge,
                source_overrides={"python/vosti_kernels/physical.py": mutated},
            )

    def test_raw_rectangular_certificates_have_checked_adapters(self) -> None:
        certificates = [
            self.raw_generated_certificate(
                self.residual_rms_source,
                "rmsnorm_residual_kernel",
                self.residual_rms_config,
                "add_rms_norm_axis_projection_certificate",
            ),
            self.raw_generated_certificate(
                self.residual_rms_source,
                "rmsnorm_residual_kernel",
                self.residual_rms_wide_config,
                "add_rms_norm_wide_axis_projection_certificate",
            ),
            self.raw_generated_certificate(
                self.embedding_source,
                "embedding_kernel",
                self.embedding_config,
                "embed_axis_projection_certificate",
            ),
            self.raw_generated_certificate(
                self.embedding_source,
                "embedding_kernel",
                self.embedding_wide_config,
                "embed_wide_axis_projection_certificate",
            ),
            self.raw_generated_certificate(
                self.qk_norm_source,
                "head_rms_norm_kernel",
                self.k_head_norm_config,
                "k_head_rms_norm_axis_projection_certificate",
            ),
            self.raw_generated_certificate(
                self.qk_norm_source,
                "head_rms_norm_kernel",
                self.q_head_norm_config,
                "q_head_rms_norm_axis_projection_certificate",
            ),
            self.raw_generated_certificate(
                self.qk_norm_source,
                "head_rms_norm_kernel",
                self.q_head_norm_32_config,
                "q_head_rms_norm_32_axis_projection_certificate",
            ),
            self.raw_generated_certificate(
                self.rope_source,
                "rope_kernel",
                self.rope_config,
                "rope_axis_projection_certificate",
            ),
        ]
        certificates.extend([
            self.raw_generated_certificate(
                self.rms_source,
                "rmsnorm_kernel",
                self.rms_config,
                "rms_norm_axis_projection_certificate",
            ),
            self.raw_generated_certificate(
                self.rms_source,
                "rmsnorm_kernel",
                self.rms_wide_config,
                "rms_norm_wide_axis_projection_certificate",
            ),
            self.silu_generated_certificate(
                "silu_mul_axis_projection_certificate",
            ),
        ])
        for certificate in certificates:
            self.assertEqual(certificate.body.count("#[verifier::external_body]"), 1)
            self.assertIn("// Checked rectangular instantiation", certificate.body)

    def test_real_embedding_bridge_matches_annotation(self) -> None:
        validate_bridge_surfaces(self.embedding_contract, self.embedding_source)

    def test_embedding_certificate_records_complete_roles(self) -> None:
        axis = self.embedding_axis_contract()
        data = axis.to_data()
        self.assertEqual(data["projected_inputs"], ["ids"])
        self.assertEqual(data["shared_inputs"], ["weight"])
        self.assertEqual(len(data["derived_preconditions"]), 1)
        generated = self.raw_generated_certificate(
            self.embedding_source,
            "embedding_kernel",
            self.embedding_config,
            "embed_axis_projection_certificate",
        )
        self.assertIn("input_ids.len() == rows", generated.body)
        self.assertIn("TS::tensor2d_shape(input_weight, V, 1024)", generated.body)

    def embedding_inferred_parts(self):
        raw = verify_annotations(self.embedding_source, "embedding_kernel", self.embedding_config).verified_contract
        axis = normalize_axis_projection_contract(raw)
        fragment = render_verified_contract_to_verus(raw, symbol_prefix="raw")
        return raw, axis, fragment, inferred_binding(axis, raw, fragment)

    def test_embedding_certificate_rejects_omitted_shared_weight(self) -> None:
        raw, axis, fragment, binding = self.embedding_inferred_parts()
        binding["arguments"].pop()
        with self.assertRaisesRegex(ValueError, "shared tensor binding differs"):
            render_raw_rectangular_axis_adapter(raw, axis, fragment, binding)

    def test_importer_rejects_unknown_axis_schema_fields(self) -> None:
        data = self.embedding_axis_contract().to_data()
        data["future_unhandled_role"] = []
        axis = VerifiedAxisProjectionContract(
            canonical_json=json.dumps(data, sort_keys=True, separators=(",", ":"))
        )
        _, _, _, binding = self.embedding_inferred_parts()
        binding["contract_digest"] = axis.digest
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            validate_axis_projection_binding(
                axis,
                {key: value for key, value in binding.items() if key not in {"verus_lowering", "raw_contract_digest"}},
            )

    def test_real_rms_norm_bridge_matches_annotation(self) -> None:
        validate_bridge_surfaces(self.rms_contract, self.rms_source)

    def test_rms_norm_shared_scalar_is_inferred_and_must_keep_its_type(self) -> None:
        result = verify_annotations(self.rms_source, "rmsnorm_kernel", self.rms_config)
        self.assertTrue(result.proved)
        raw = result.verified_contract
        axis = normalize_axis_projection_contract(raw)
        fragment = render_verified_contract_to_verus(raw, symbol_prefix="raw")
        binding = inferred_binding(axis, raw, fragment)
        eps = next(a for a in binding["arguments"] if a["kernel"] == "eps")
        self.assertEqual((eps["type"], eps["mode"]), ("Scalar", "shared"))
        generated = render_raw_rectangular_axis_adapter(raw, axis, fragment, binding)
        self.assertIn("input_eps: Scalar", generated)
        self.assertIn("row_projection_repr(seq![input_x[i as int]], input_w, input_eps, N)", generated)
        self.assertIn("row_projection_launch_equivalence", generated)

        omitted = deepcopy(binding)
        omitted["arguments"] = [a for a in omitted["arguments"] if a["kernel"] != "eps"]
        with self.assertRaisesRegex(ValueError, "shared scalar binding differs"):
            render_raw_rectangular_axis_adapter(raw, axis, fragment, omitted)

        wrong_type = deepcopy(binding)
        next(a for a in wrong_type["arguments"] if a["kernel"] == "eps")["type"] = "Tensor1D"
        with self.assertRaisesRegex(ValueError, "does not match kernel argument"):
            render_raw_rectangular_axis_adapter(raw, axis, fragment, wrong_type)

    def test_wrong_framework_output_type_rejected(self) -> None:
        raw, axis, fragment, binding = self.embedding_inferred_parts()
        binding["outputs"][0]["type"] = "Tensor1D"
        with self.assertRaisesRegex(ValueError, "does not match kernel argument"):
            render_raw_rectangular_axis_adapter(raw, axis, fragment, binding)

    def test_residual_rms_inference_preserves_both_outputs(self) -> None:
        validate_bridge_surfaces(self.residual_rms_contract, self.residual_rms_source)
        result = verify_annotations(self.residual_rms_source, "rmsnorm_residual_kernel", self.residual_rms_config)
        self.assertTrue(result.proved)
        raw = result.verified_contract
        axis = normalize_axis_projection_contract(raw)
        fragment = render_verified_contract_to_verus(raw, symbol_prefix="raw")
        binding = inferred_binding(axis, raw, fragment)
        self.assertEqual([o["kernel"] for o in binding["outputs"]], ["o", "residual_out"])
        generated = render_raw_rectangular_axis_adapter(raw, axis, fragment, binding)
        self.assertIn("row_projection_o_repr", generated)
        self.assertIn("row_projection_residual_out_repr", generated)

        omitted = deepcopy(binding)
        omitted["outputs"].pop()
        with self.assertRaisesRegex(ValueError, "output binding differs"):
            render_raw_rectangular_axis_adapter(raw, axis, fragment, omitted)
        duplicate = deepcopy(binding)
        duplicate["outputs"][1] = deepcopy(duplicate["outputs"][0])
        with self.assertRaisesRegex(ValueError, "duplicate kernel output binding"):
            render_raw_rectangular_axis_adapter(raw, axis, fragment, duplicate)
        reordered = deepcopy(binding)
        reordered["outputs"].reverse()
        with self.assertRaisesRegex(ValueError, "output order differs"):
            render_raw_rectangular_axis_adapter(raw, axis, fragment, reordered)

    def test_engine_scope_rejects_omitted_residual_output(self) -> None:
        """Partial goals are valid, but cannot satisfy the engine's full surface."""

        mutated = self.residual_rms_source.replace(
            "#     left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N],\n"
            "#     left(residual_out)[b:b+1, 0:N] == right(residual_out)[0:1, 0:N]\n",
            "#     left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N]\n",
        )
        self.assertNotEqual(mutated, self.residual_rms_source)
        with self.assertRaisesRegex(ValueError, "omits required outputs.*residual_out"):
            validate_post_surface(
                "rmsnorm_residual.py", "rmsnorm_residual_kernel", mutated
            )

    def test_residual_rms_wrapper_output_order_is_attested(self) -> None:
        mutated = self.python_source.replace(
            "return (normed, new_res)\n# @kernel-bridge-end vosti_kernels::add_rms_norm",
            "return (new_res, normed)\n# @kernel-bridge-end vosti_kernels::add_rms_norm",
        )
        self.assertNotEqual(mutated, self.python_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.residual_rms_contract,
                self.residual_rms_source,
                source_overrides={"python/vosti_kernels/kernels.py": mutated},
            )

    def test_qk_norm_import_requires_both_model_specializations(self) -> None:
        validate_bridge_surfaces(self.qk_norm_contract, self.qk_norm_source)
        q_axis = self.qk_norm_axis_contract(self.q_head_norm_config)
        k_axis = self.qk_norm_axis_contract(self.k_head_norm_config)
        self.assertNotEqual(q_axis.digest, k_axis.digest)
        for axis, proof_name, config in (
            (
                q_axis,
                "q_head_rms_norm_axis_projection_certificate",
                self.q_head_norm_config,
            ),
            (
                k_axis,
                "k_head_rms_norm_axis_projection_certificate",
                self.k_head_norm_config,
            ),
        ):
            generated = self.raw_generated_certificate(
                self.qk_norm_source,
                "head_rms_norm_kernel",
                config,
                proof_name,
            )
            self.assertEqual(axis.to_data()["domain_preconditions"], [])
            self.assertIn("Raw ContractIR lowering", generated.body)

        q_raw = verify_annotations(self.qk_norm_source, "head_rms_norm_kernel", self.q_head_norm_config).verified_contract
        k_raw = verify_annotations(self.qk_norm_source, "head_rms_norm_kernel", self.k_head_norm_config).verified_contract
        q_fragment = render_verified_contract_to_verus(q_raw, symbol_prefix="raw")
        k_fragment = render_verified_contract_to_verus(k_raw, symbol_prefix="raw")
        wrong = inferred_binding(k_axis, k_raw, k_fragment)
        with self.assertRaisesRegex(ValueError, "contract digest mismatch"):
            render_raw_rectangular_axis_adapter(q_raw, q_axis, q_fragment, wrong)

    def test_rope_import_binds_all_projected_rows(self) -> None:
        validate_bridge_surfaces(self.rope_contract, self.rope_source)
        axis = self.rope_axis_contract()
        data = axis.to_data()
        self.assertEqual(
            data["projected_inputs"], ["cos_table", "sin_table", "x"]
        )
        self.assertEqual(data["projected_outputs"], ["o"])
        self.assertEqual(data["domain_preconditions"], [])
        generated = self.raw_generated_certificate(
            self.rope_source,
            "rope_kernel",
            self.rope_config,
            "rope_axis_projection_certificate",
        )
        self.assertIn("rope_axis_projection_repr", generated.body)

        raw = verify_annotations(self.rope_source, "rope_kernel", self.rope_config).verified_contract
        fragment = render_verified_contract_to_verus(raw, symbol_prefix="raw")
        binding = inferred_binding(axis, raw, fragment)
        omitted = deepcopy(binding)
        omitted["arguments"].pop()
        with self.assertRaisesRegex(ValueError, "projected input binding differs"):
            render_raw_rectangular_axis_adapter(raw, axis, fragment, omitted)

        wrong_role = deepcopy(binding)
        wrong_role["arguments"][1]["kernel"] = "x"
        with self.assertRaisesRegex(ValueError, "duplicate kernel argument"):
            render_raw_rectangular_axis_adapter(raw, axis, fragment, wrong_role)

    def test_rope_wrapper_head_row_expansion_is_attested(self) -> None:
        mutated = self.python_source.replace(
            "cos_t.repeat_interleave(heads, dim=0)",
            "cos_t.repeat_interleave(1, dim=0)",
        )
        self.assertNotEqual(mutated, self.python_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.rope_contract,
                self.rope_source,
                source_overrides={"python/vosti_kernels/kernels.py": mutated},
            )

    def test_partial_kernel_output_cannot_be_imported_as_whole_row(self) -> None:
        mutated = self.source.replace(
            "0:N] == right(c)[0:1, 0:N]",
            "0:N-1] == right(c)[0:1, 0:N-1]",
        )
        result = verify_annotations(
            mutated, "matmul_kernel", self.matmul_config
        )
        self.assertTrue(result.proved)
        self.assertIsNotNone(result.verified_contract)
        with self.assertRaisesRegex(ValueError, "complete selected-row equality"):
            normalize_axis_projection_contract(result.verified_contract)

    def test_positive_batch_and_width_remain_in_raw_domain(self) -> None:
        source = self.silu_source.replace(
            "right(M) == 1,",
            "right(M) == 1, left(M) > 0, N > 0,",
        )
        result = verify_annotations(
            source, "silu_mul_kernel", self.silu_config
        )
        self.assertTrue(result.proved)
        self.assertIsNotNone(result.verified_contract)
        raw = result.verified_contract
        axis = normalize_axis_projection_contract(raw)
        fragment = render_verified_contract_to_verus(raw, symbol_prefix="silu_mul_axis_projection_raw")
        binding = inferred_binding(axis, raw, fragment)
        binding["proof_name"] = "silu_mul_axis_projection_certificate"
        binding["contract_digest"] = axis.digest
        binding["raw_contract_digest"] = raw.digest
        binding["domain_policy"] = "raw_preconditions"
        prefix = binding["proof_name"].removesuffix("_certificate") + "_raw"
        certificate = rectangular_fixture(raw, axis, render_verified_contract_to_verus(raw, symbol_prefix=prefix), binding)
        self.assertIn("left.M > 0", certificate.body)
        self.assertIn("left.N > 0", certificate.body)

    def test_adapter_rejects_retired_domain_policies(self) -> None:
        raw, axis, fragment, binding = self.embedding_inferred_parts()
        for policy in ("none", "positive_batch_and_width", "ordered_positive_qkv_geometry"):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, "only exact raw preconditions"):
                    render_raw_rectangular_axis_adapter(
                        raw, axis, fragment, {**binding, "domain_policy": policy})

    def shared_execution_silu_certificate(self):
        raw = verify_annotations(self.silu_source, "silu_mul_kernel",
                                 self.silu_config).verified_contract
        axis = normalize_axis_projection_contract(raw)
        bundle = render_verified_kernel_to_verus((raw,), symbol_prefix="shared_silu")
        binding = inferred_binding(axis, raw, bundle.contracts[0])
        binding["proof_name"] = "silu_mul_axis_projection_certificate"
        generated = rectangular_fixture(raw, axis, bundle.contracts[0], binding)
        return generated

    def test_rectangular_adapter_accepts_shared_execution_binding(self) -> None:
        generated = self.shared_execution_silu_certificate()
        self.assertIn("shared_silu_batch_invariance_certificate(left, right, free)", generated.body)
        self.assertIn("shared_silu_execute(left)", generated.body)
        # Only the raw annotation theorem is assumed; the existing rectangular
        # domain/projection adapters continue to have checked bodies.
        self.assertEqual(generated.body.count("#[verifier::external_body]"), 1)

    @unittest.skipUnless(os.environ.get("VERUS"), "set VERUS to check the shared-execution adapter")
    def test_verus_checks_shared_execution_rectangular_adapter(self) -> None:
        generated = self.shared_execution_silu_certificate()
        module = f'''#![allow(non_snake_case)]
use vstd::prelude::*;
pub mod fixture_types {{
    use vstd::prelude::*;
    pub type Tensor2D = Seq<Seq<super::Scalar>>;
    pub type Tensor3D = Seq<Tensor2D>;
    pub type Tensor4D = Seq<Tensor3D>;
}}
#[path = "{ROOT / 'src/proof/tensor/shape.rs'}"]
mod TS;
verus! {{
#[verifier::external_body]
#[verifier::ext_equal]
pub struct Scalar {{ _private: () }}
pub type Tensor2D = Seq<Seq<Scalar>>;
pub uninterp spec fn generated_kernel_allocation_cell() -> Scalar;
{generated.body}
}}

mod proof {{
pub mod model {{ pub mod types {{ pub use crate::fixture_types::*; }} }}
pub mod tensor {{
pub mod types {{ pub use crate::fixture_types::*; }}

}}
}}

mod types {{
    pub use crate::fixture_types::*;
    use vstd::prelude::*;
    verus! {{ {page_constants_source()} }}
}}

fn main() {{}}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared_adapter.rs"
            path.write_text(module)
            result = subprocess.run([os.environ["VERUS"], str(path)],
                                    capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("0 errors", result.stdout)

    def test_ordered_positive_qkv_geometry_remains_in_raw_domain(self) -> None:
        certificate = self.raw_generated_certificate(
            self.qkv_source, "qkv_matmul_kernel", self.qkv_config,
            "qkv_axis_projection_certificate",
        )
        self.assertIn("left.QN >= left.KVN", certificate.body)
        self.assertIn("left.KVN > 0", certificate.body)
        self.assertIn("left.K > 0", certificate.body)

    def test_real_silu_and_mul_bridge_matches_annotation(self) -> None:
        validate_bridge_surfaces(self.silu_contract, self.silu_source)

    def test_trusted_store_bridge_attests_every_effect_surface(self) -> None:
        self.assertEqual(
            self.store_contract["evidence"],
            "exact_effect_certificate",
        )
        self.assertNotIn(
            "semantic_contracts", self.store_contract["bridge"]
        )
        self.assertEqual(
            {
                (span["path"], span["name"])
                for span in self.store_contract["bridge"]["source_spans"]
            },
            {
                ("src/boundary/tensor_runtime.rs", "boundary::tensor_runtime::paged_cache_geometry"),
                ("src/boundary/tensor_runtime.rs", "boundary::tensor_runtime::store_kv_cache_metadata_ready"),
                ("src/boundary/tensor_runtime.rs", "boundary::tensor_runtime::store_kv_cache_launch_ready"),
                ("src/boundary/tensor_runtime.rs", "boundary::tensor_runtime::store_kv_cache_repr"),
                ("src/boundary/tensor_runtime.rs", "boundary::tensor_runtime::store_kv_cache"),
                (
                    "python/vosti_kernels/kernels.py",
                    "vosti_kernels::store_kv_cache_from_verified_caller",
                ),
                (
                    "kernels/triton_kernels/store_kv_cache.py",
                    "store_kv_cache::store_cache_kernel",
                ),
                ("kernels/triton_kernels/store_kv_cache.py", "store_kv_cache::store_kv_cache"),
            },
        )
        validate_bridge_surfaces(self.store_contract, self.store_source)

    def test_store_bridge_rejects_changed_exact_effect(self) -> None:
        mutated = self.framework_source.replace(
            "mid.0[page].update(offset, kr[kr.len() - 1])",
            "mid.0[page].update(offset, vr[vr.len() - 1])",
            1,
        )
        self.assertNotEqual(mutated, self.framework_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.store_contract,
                self.store_source,
                source_overrides={"src/boundary/tensor_runtime.rs": mutated},
            )

    def test_store_bridge_rejects_changed_verified_caller_dispatch(self) -> None:
        mutated = self.python_source.replace(
            '        PRIMITIVE_RUNTIME.kernel_entrypoint(\n'
            '            runtime, vk, "store_kv_cache"\n'
            '        )(\n'
            '            k,\n'
            '            v,\n'
            '            k_cache,\n'
            '            v_cache,\n'
            '            slot_mapping,\n'
            '            launch_config=launch_config,\n'
            '        )\n'
            '        return\n'
            '    # CPU and toy-shape tests retain',
            '        PRIMITIVE_RUNTIME.kernel_entrypoint(\n'
            '            runtime, vk, "store_kv_cache"\n'
            '        )(\n'
            '            v,\n'
            '            k,\n'
            '            v_cache,\n'
            '            k_cache,\n'
            '            slot_mapping,\n'
            '            launch_config=launch_config,\n'
            '        )\n'
            '        return\n'
            '    # CPU and toy-shape tests retain',
            1,
        )
        self.assertNotEqual(mutated, self.python_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.store_contract,
                self.store_source,
                source_overrides={"python/vosti_kernels/kernels.py": mutated},
            )

    def test_store_bridge_rejects_swapped_runtime_roles(self) -> None:
        mutated = self.python_source.replace(
            '        PRIMITIVE_RUNTIME.kernel_entrypoint(\n'
            '            runtime, vk, "store_kv_cache"\n'
            '        )(\n'
            '            k,\n'
            '            v,\n'
            '            k_cache,\n'
            '            v_cache,\n'
            '            slot_mapping,\n'
            '            launch_config=launch_config,\n'
            '        )\n'
            '        return\n'
            '    # CPU and toy-shape tests retain',
            '        PRIMITIVE_RUNTIME.kernel_entrypoint(\n'
            '            runtime, vk, "store_kv_cache"\n'
            '        )(\n'
            '            k,\n'
            '            v,\n'
            '            v_cache,\n'
            '            k_cache,\n'
            '            slot_mapping,\n'
            '            launch_config=launch_config,\n'
            '        )\n'
            '        return\n'
            '    # CPU and toy-shape tests retain',
            1,
        )
        self.assertNotEqual(mutated, self.python_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.store_contract,
                self.store_source,
                source_overrides={"python/vosti_kernels/kernels.py": mutated},
            )

    def test_store_bridge_rejects_changed_kernel_address_or_pairing(self) -> None:
        for mutated in (
            self.store_source.replace("offsets=(s, 0)", "offsets=(s + 1, 0)", 1),
            self.store_source.replace(
                "for src, cache in ((k, k_cache), (v, v_cache)):",
                "for src, cache in ((k, v_cache), (v, k_cache)):",
                1,
            ),
        ):
            with self.subTest():
                self.assertNotEqual(mutated, self.store_source)
                with self.assertRaisesRegex(
                    ValueError, "source span .* digest mismatch"
                ):
                    validate_bridge_surfaces(
                        self.store_contract,
                        self.store_source,
                        source_overrides={
                            "kernels/triton_kernels/store_kv_cache.py": mutated
                        },
                    )





    def test_selected_dataflow_qualification_rejects_identity_or_scope_drift(self) -> None:
        from ir.proof_preparation import prepare_annotation_proof
        prepared = prepare_annotation_proof(self.attention_source,
            "fattn_varlen_paged_fwd_block_ptr_kernel", self.attention_config,
            goal_name="selected_row_prefix_equivalence")
        report = self.attention_selected_dataflow
        alignment = report.alignments[0]
        cases = {
            "kernel identity": replace(report, kernel_name="other_kernel"),
            "goal identity": replace(report, goal_name="batch_invariance"),
            "source identity": replace(report, source_sha256="0" * 64),
            "specialization": replace(
                report,
                constants=tuple(
                    (name, 32 if name == "BLOCK_M" else value)
                    for name, value in report.constants
                ),
            ),
            "duplicate specialization": replace(
                report,
                constants=report.constants + (report.constants[0],),
            ),
            "theorem identity": replace(report, theorem_sha256="0" * 64),
            "failed proof": replace(
                report,
                checks=(replace(report.checks[0], proved=False), *report.checks[1:]),
            ),
            "tensor dependencies": replace(
                report,
                required_tensor_reads=("q",),
            ),
            "output alignments": replace(report, alignments=()),
            "vacuous": replace(
                report,
                alignments=(replace(alignment, theorem_vacuous=True),),
            ),
            "external obligation": replace(
                report,
                external_obligations=("undeclared numeric premise",),
            ),
            "used assumption": replace(
                report,
                used_assumptions=("undeclared analyzer oracle",),
            ),
        }
        for label, mutated in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    validate_dataflow_artifact(prepared, mutated)














    def test_attention_bridge_rejects_changed_numeric_domain_declaration(self) -> None:
        mutated = self.framework_source.replace(
            "pub uninterp spec fn paged_attention_numeric_domain() -> bool;",
            "pub uninterp spec fn paged_attention_numeric_domain() -> int;",
        )
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.attention_contract,
                self.attention_source,
                source_overrides={"src/boundary/tensor_runtime.rs": mutated},
            )

    def test_attention_bridge_rejects_changed_verus_dispatch(self) -> None:
        mutated = self.framework_source.replace(
            'let f = m.getattr("paged_attention_from_verified_caller")?;',
            'let f = m.getattr("linear")?;',
            1,
        )
        self.assertNotEqual(mutated, self.framework_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.attention_contract,
                self.attention_source,
                source_overrides={"src/boundary/tensor_runtime.rs": mutated},
            )

    def test_attention_bridge_rejects_reenabled_device_value_checks(self) -> None:
        mutated = self.python_source.replace(
            "            value_checks=False,",
            "            value_checks=True,",
            1,
        )
        self.assertNotEqual(mutated, self.python_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.attention_contract,
                self.attention_source,
                source_overrides={"python/vosti_kernels/kernels.py": mutated},
            )

    def test_silu_certificate_covers_both_projected_halves(self) -> None:
        axis = self.silu_axis_contract()
        self.assertEqual(axis.to_data()["projected_inputs"], ["x", "y"])
        self.assertEqual(axis.to_data()["domain_preconditions"], [])
        generated = self.silu_generated_certificate(
            "silu_mul_axis_projection_certificate"
        )
        self.assertIn("silu_mul_axis_projection_raw_raw_pre", generated.body)
        self.assertIn("// Checked rectangular instantiation", generated.body)
        self.assertNotIn(
            "#[verifier::external_body]\npub proof fn silu_mul_axis_projection_certificate",
            generated.body,
        )

    def test_silu_certificate_rejects_an_omitted_half(self) -> None:
        raw = verify_annotations(self.silu_source, "silu_mul_kernel", self.silu_config).verified_contract
        axis = normalize_axis_projection_contract(raw)
        fragment = render_verified_contract_to_verus(raw, symbol_prefix="silu_mul_axis_projection_raw")
        binding = inferred_binding(axis, raw, fragment)
        binding["arguments"].pop()
        with self.assertRaisesRegex(ValueError, "projected input binding differs"):
            rectangular_fixture(raw, axis, fragment, binding)

    def test_rejects_changed_silu_split_representation(self) -> None:
        mutated = self.framework_source.replace(
            "let middle = (row.len() / 2) as int;",
            "let middle = 0int;",
        )
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.silu_contract,
                self.silu_source,
                source_overrides={"src/boundary/tensor_runtime.rs": mutated},
            )

    def test_rejects_changed_precondition(self) -> None:
        mutated = self.source.replace("right(M) == 1", "right(M) == 2")
        with self.assertRaisesRegex(ValueError, "annotation digest mismatch"):
            validate_bridge_surfaces(self.contract, mutated)

    def test_rejects_wrong_singleton_output_projection(self) -> None:
        mutated = self.source.replace(
            "right(c)[0:1, 0:N]", "right(c)[1:2, 0:N]"
        )
        with self.assertRaisesRegex(ValueError, "annotation digest mismatch"):
            validate_bridge_surfaces(self.contract, mutated)

    def test_rejects_changed_framework_semantic_declaration(self) -> None:
        mutated = self.framework_source.replace(
            "linear_kernel_cell_repr(row, wr, col)",
            "linear_kernel_cell_repr(row, wr, col + 1)",
        )
        self.assertNotEqual(mutated, self.framework_source)
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.contract,
                self.source,
                source_overrides={"src/boundary/tensor_runtime.rs": mutated},
            )

    def test_rejects_changed_framework_wrapper_postcondition(self) -> None:
        mutated = self.framework_source.replace(
            "tensor_repr_2d(perm@, t, linear_repr(xr, wr))",
            "tensor_repr_2d(perm@, t, linear_repr(wr, xr))",
        )
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.contract,
                self.source,
                source_overrides={"src/boundary/tensor_runtime.rs": mutated},
            )

    def test_rejects_changed_python_launch_adapter(self) -> None:
        mutated = self.python_source.replace("weight.t()", "weight")
        with self.assertRaisesRegex(ValueError, "source span .* digest mismatch"):
            validate_bridge_surfaces(
                self.contract,
                self.source,
                source_overrides={"python/vosti_kernels/kernels.py": mutated},
            )

    def test_rejects_source_path_outside_repository(self) -> None:
        contract = deepcopy(self.contract)
        contract["bridge"]["source_spans"][0]["path"] = "../outside.rs"
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            validate_bridge_surfaces(contract, self.source)


@unittest.skipIf(KERNEL_ROOT is None, "pinned kernel checkout is unavailable")
class ClaimLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.surface = read_json(SURFACE_PATH)
        cls.imports = {
            family: read_json(descriptor["imports_path"])
            for family, descriptor in certificate_import_sets().items()
        }

    def test_real_claim_ledger_is_closed(self) -> None:
        ledger = validate_all(deepcopy(self.surface), deepcopy(self.imports))
        self.assertEqual(len(ledger["claims"]), 3)
        self.assertEqual(len(ledger["imports"]), 49)
        self.assertEqual(len(ledger["proofs"]), 58)
        self.assertEqual(len(ledger["uninterpreted"]), 127)
        self.assertEqual(len(ledger["declaration_digests"]), 256)
        self.assertEqual(len(ledger["nonproduction"]), 0)
        self.assertEqual(set(self.imports), {"attention", "kv_store"} | set(RECTANGULAR_KERNEL_INTERFACES.values()))
        self.assertTrue(all(r["consumers"] for r in ledger["imports"].values()))





    def test_claim_ledger_rejects_wrong_checked_consumer(self) -> None:
        surface = deepcopy(self.surface)
        record = next(
            item for item in surface["certificate_consumers"]
            if item["role"] == "claim_supporting"
        )
        record["consumers"] = [
            "batch_invariance::linear_batch_invariance"
        ]
        with self.assertRaisesRegex(ValueError, "consumer mismatch"):
            validate_all(surface, deepcopy(self.imports))


    def test_claim_ledger_rejects_unclassified_runtime_boundary(self) -> None:
        surface = deepcopy(self.surface)
        surface["trusted_runtime_boundaries"].pop()
        with self.assertRaisesRegex(ValueError, "inventory is not closed"):
            validate_all(surface, deepcopy(self.imports))

    def test_claim_ledger_rejects_unclassified_trusted_proof(self) -> None:
        surface = deepcopy(self.surface)
        surface["trusted_proof_boundaries"].pop()
        with self.assertRaisesRegex(ValueError, "trusted proof inventory is not closed"):
            validate_all(surface, deepcopy(self.imports))

    def test_claim_ledger_rejects_unclassified_uninterpreted_spec(self) -> None:
        surface = deepcopy(self.surface)
        surface["uninterpreted_specifications"].pop()
        with self.assertRaisesRegex(
            ValueError, "uninterpreted specification inventory is not closed"
        ):
            validate_all(surface, deepcopy(self.imports))

    def test_claim_ledger_rejects_trusted_declaration_source_drift(self) -> None:
        surface = deepcopy(self.surface)
        surface["trusted_declaration_source_sha256"][
            "boundary::backend_certificates::qkv::raw_oq_cell"
        ] = "0" * 64
        with self.assertRaisesRegex(
            ValueError, "trusted declaration source digest mismatch"
        ):
            validate_all(surface, deepcopy(self.imports))

    def test_claim_ledger_rejects_unclosed_declaration_digest_inventory(self) -> None:
        surface = deepcopy(self.surface)
        surface["trusted_declaration_source_sha256"].pop(
            "boundary::tensor_runtime::Tensor"
        )
        with self.assertRaisesRegex(
            ValueError, "source-digest inventory is not closed"
        ):
            validate_all(surface, deepcopy(self.imports))

    def test_claim_ledger_rejects_trust_outside_boundary(self) -> None:
        original = claim_ledger_module.scan_rust_trust_records

        def mutated(path):
            rows = list(original(path))
            if path == ROOT / "src" / "exec" / "engine.rs":
                rows.append(
                    {
                        "identity": "exec::engine::unreviewed_boundary",
                        "category": "exec-trusted",
                        "source_sha256": "0" * 64,
                    }
                )
            return rows

        with mock.patch.object(
            claim_ledger_module, "scan_rust_trust_records", mutated
        ):
            with self.assertRaisesRegex(
                ValueError, "trusted declaration outside src/boundary"
            ):
                validate_all(deepcopy(self.surface), deepcopy(self.imports))

    def test_claim_ledger_rejects_unclosed_nonproduction_trust(self) -> None:
        original = claim_ledger_module.scan_rust_trust_records

        def mutated(path):
            rows = list(original(path))
            if path.name == "engine_step.rs":
                rows.append(
                    {
                        "identity": "unreviewed_test_boundary",
                        "category": "exec-trusted",
                        "source_sha256": "0" * 64,
                    }
                )
            return rows

        with mock.patch.object(
            claim_ledger_module, "scan_rust_trust_records", mutated
        ):
            with self.assertRaisesRegex(
                ValueError, "non-production trusted declaration inventory is not closed"
            ):
                validate_all(deepcopy(self.surface), deepcopy(self.imports))

    def test_claim_ledger_rejects_live_trusted_declaration_growth(self) -> None:
        original = claim_ledger_module.scan_rust_trust

        def mutated(path):
            rows = list(original(path))
            if path.name == "engine.rs":
                rows.append(
                    ("engine::unreviewed_axiom", "proof-trusted", 1, 999_999)
                )
            return rows

        with mock.patch.object(claim_ledger_module, "scan_rust_trust", mutated):
            with self.assertRaisesRegex(
                ValueError, "trusted proof inventory is not closed"
            ):
                validate_all(deepcopy(self.surface), deepcopy(self.imports))

    def test_claim_ledger_rejects_live_uninterpreted_spec_growth(self) -> None:
        original = claim_ledger_module.scan_rust_trust

        def mutated(path):
            rows = list(original(path))
            if path.name == "engine.rs":
                rows.append(("engine::unreviewed_model", "uninterp", 1, 999_999))
            return rows

        with mock.patch.object(claim_ledger_module, "scan_rust_trust", mutated):
            with self.assertRaisesRegex(
                ValueError, "uninterpreted specification inventory is not closed"
            ):
                validate_all(deepcopy(self.surface), deepcopy(self.imports))

    def test_trust_scanner_normalizes_attributes_and_excludes_disabled_items(self) -> None:
        source = """
#[verifier ::
  external_body]
pub(in crate::proof) proof fn spaced_axiom() {}

#[cfg(any())]
#[verifier::external_body]
pub proof fn disabled_axiom() {}

pub uninterp spec fn semantic_point() -> bool;
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.rs"
            path.write_text(source, encoding="utf-8")
            rows = scan_rust_trust(path)
        self.assertEqual(
            [(identity, category) for identity, category, _, _ in rows],
            [
                ("sample::spaced_axiom", "proof-trusted"),
                ("sample::semantic_point", "uninterp"),
            ],
        )

    def test_accounting_separates_host_only_exec_from_verified_exec(self) -> None:
        source = """
#[cfg(not(verus_only))]
pub fn host_runtime_gate() -> bool { true }

pub fn verified_engine_step() {}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.rs"
            path.write_text(source, encoding="utf-8")
            rows = scan_rust(path)
            trusted_rows = scan_rust_trust(path)
        self.assertEqual(
            [(name, category) for name, category, _, _ in rows],
            [
                ("host_runtime_gate", "exec-host"),
                ("verified_engine_step", "exec-verified"),
            ],
        )
        self.assertEqual(trusted_rows, [])

    def test_accounting_rejects_host_only_verus_declaration(self) -> None:
        source = """
#[cfg(not(verus_only))]
pub proof fn hidden_proof() {}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.rs"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "host-only Verus"):
                scan_rust(path)

    def test_rust_source_discovery_recurses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            top = root / "top.rs"
            nested = root / "scheduler" / "invariants.rs"
            nested.parent.mkdir()
            top.write_text("fn top() {}\n", encoding="utf-8")
            nested.write_text("proof fn nested() {}\n", encoding="utf-8")
            discovered = rust_source_files(root)
        self.assertEqual(discovered, [nested, top])

    def test_trust_scanner_source_digest_covers_contract_text(self) -> None:
        first = """
#[verifier::external_body]
pub proof fn axiom() ensures true {}

pub uninterp spec fn model() -> bool;

#[verifier::external_body]
pub struct Handle { field: u64 }
"""
        second = first.replace("ensures true", "ensures false")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.rs"
            path.write_text(first, encoding="utf-8")
            first_functions = scan_rust_trust_records(path)
            first_types = scan_external_type_records(path)
            path.write_text(second, encoding="utf-8")
            second_functions = scan_rust_trust_records(path)
            second_types = scan_external_type_records(path)
        self.assertNotEqual(
            first_functions[0]["source_sha256"],
            second_functions[0]["source_sha256"],
        )
        self.assertEqual(
            first_functions[1]["source_sha256"],
            second_functions[1]["source_sha256"],
        )
        self.assertEqual(
            first_types[0]["source_sha256"], second_types[0]["source_sha256"]
        )

    def test_trust_scanner_rejects_unrecognized_external_body_target(self) -> None:
        source = """
#[verifier::external_body]
pub enum HiddenTrust { Value }
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.rs"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unrecognized external_body"):
                scan_rust_trust(path)

    def test_trust_scanner_rejects_split_uninterpreted_declaration(self) -> None:
        source = """
pub uninterp
spec fn hidden_model() -> bool;
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.rs"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "unrecognized uninterpreted declaration"
            ):
                scan_rust_trust(path)

    def test_trust_scanner_rejects_cfg_attr_external_body_bypass(self) -> None:
        source = """
#[cfg_attr(verus_only, verifier::external_body)]
pub proof fn hidden_axiom() {}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.rs"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "unsupported external_body attribute"
            ):
                scan_rust_trust(path)

    def test_trust_scanner_rejects_external_body_spec_function(self) -> None:
        source = """
#[verifier::external_body]
pub spec fn hidden_model() -> bool { true }
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.rs"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "unsupported external_body spec function"
            ):
                scan_rust_trust(path)

    def test_compiler_trust_parser_reads_typed_modes(self) -> None:
        vir = """
(@ "src/sample.rs:4:1: 4:20 (#0)" (Function
 :body_visibility (BodyVisibility Visibility (Visibility :restricted_to crate))
 :owning_module crate::sample :mode Proof
 :attrs (FunctionAttrs :is_external_body true)
 :body None))
(@ "src/sample.rs:7:1: 7:20 (#0)" (Function
 :body_visibility (BodyVisibility Uninterpreted)
 :owning_module crate::sample :mode Spec
 :attrs (FunctionAttrs :is_external_body true)
 :body None))
(@ "src/sample.rs:10:1: 10:20 (#0)" (Datatype
 :name (Dt Path crate::sample::Handle)
 :transparency (DatatypeTransparency Never)
 :variants ()))
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crate-simple.vir"
            path.write_text(vir, encoding="utf-8")
            records = compiler_trust_records(path)
        self.assertEqual(
            records,
            [
                ("src/sample.rs", 4, "proof-trusted"),
                ("src/sample.rs", 7, "uninterp"),
                ("src/sample.rs", 10, "external-type"),
            ],
        )

    def test_compiler_trust_parser_rejects_typed_assume(self) -> None:
        vir = """
(@ "src/sample.rs:4:1: 6:2 (#0)" (Function
 :body_visibility (BodyVisibility Visibility (Visibility :restricted_to crate))
 :owning_module crate::sample :mode Proof
 :attrs (FunctionAttrs :is_external_body false)
 :body (> AssertAssume :is_assume true :expr true)))
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crate-simple.vir"
            path.write_text(vir, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contains assume/admit"):
                compiler_trust_records(path)

    def test_compiler_trust_parser_ignores_has_resolved_assume(self) -> None:
        vir = """
(@ "src/sample.rs:4:1: 6:2 (#0)" (Function
 :body_visibility (BodyVisibility Visibility (Visibility :restricted_to crate))
 :owning_module crate::sample :mode Exec
 :attrs (FunctionAttrs :is_external_body false)
 :body (> AssertAssume :is_assume true
   :expr (> UnaryOpr (UnaryOpr HasResolved (Typ MutRef (Typ Bool))) true)
   :msg None)))
(@ "src/sample.rs:8:1: 8:20 (#0)" (Function
 :body_visibility (BodyVisibility Visibility (Visibility :restricted_to crate))
 :owning_module crate::sample :mode Proof
 :attrs (FunctionAttrs :is_external_body true)
 :body None))
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crate-simple.vir"
            path.write_text(vir, encoding="utf-8")
            records = compiler_trust_records(path)
        self.assertEqual(records, [("src/sample.rs", 8, "proof-trusted")])

    def test_claim_ledger_rejects_top_theorem_drift(self) -> None:
        surface = deepcopy(self.surface)
        surface["top_level_claims"][0]["source_span"]["digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source span digest mismatch"):
            validate_all(surface, deepcopy(self.imports))

    def test_claim_ledger_rejects_unexposed_assumption(self) -> None:
        surface = deepcopy(self.surface)
        # Keep the checked claims well formed, so this exercises assumption
        # exposure rather than the earlier nonempty-requires schema guard.
        surface["deployment_assumptions"][0]["predicate"] = (
            "tensor_runtime::unexposed_numeric_domain"
        )
        with self.assertRaisesRegex(
            ValueError, "not exposed by the checked claim surface"
        ):
            validate_all(surface, deepcopy(self.imports))


class LinearRuntimeBridgeTests(unittest.TestCase):
    def test_verified_launch_uses_declared_weight_transpose(self) -> None:
        seen: dict[str, torch.Tensor] = {}

        def matmul(
            a: torch.Tensor, b: torch.Tensor, *, launch_config: dict
        ) -> torch.Tensor:
            seen["a"] = a
            seen["b"] = b
            seen["launch_config"] = launch_config
            return a @ b

        vk = types.SimpleNamespace(matmul=types.SimpleNamespace(matmul=matmul))
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        launch_config = {"sealed": 1}
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                return_value=launch_config,
            ),
            mock.patch.object(
                primitive_runtime, "kernel_entrypoint", return_value=matmul
            ),
        ):
            output = kernels.linear(x, weight)

        self.assertIs(seen["a"], x)
        self.assertTrue(torch.equal(seen["b"], weight.t()))
        self.assertIs(seen["launch_config"], launch_config)
        self.assertTrue(torch.equal(output, x @ weight.t()))

    def test_qkv_rejects_zero_reduction_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive reduction width"):
            kernels.qkv_linear(torch.empty(2, 0), torch.empty(4, 0),
                               torch.empty(2, 0), torch.empty(2, 0))

    def test_verified_normalization_rejects_incompatible_shapes(self) -> None:
        with mock.patch.object(primitive_runtime, "verified_for", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "weight width differs"):
                kernels.rms_norm(torch.empty(2, 3), torch.empty(4))
            with self.assertRaisesRegex(RuntimeError, "shapes differ"):
                kernels.add_rms_norm(torch.empty(2, 3), torch.empty(1, 3), torch.empty(3))
            with self.assertRaisesRegex(RuntimeError, "shapes differ"):
                kernels.add_rms_norm(torch.empty(2, 3), torch.empty(2, 3), torch.empty(4))

    def test_configured_embedding_rejects_empty_vocabulary(self) -> None:
        config = {"hidden_size": 4, "dtype": torch.float32}
        with mock.patch.object(primitive_runtime, "runtime_config_or_none", return_value=config), \
                mock.patch.object(primitive_runtime, "runtime_config", return_value=config):
            with self.assertRaisesRegex(ValueError, "weight disagrees"):
                kernels.embed(torch.empty((0,), dtype=torch.int32), torch.empty((0, 4)))

    def test_verified_qkv_launch_uses_three_declared_weight_transposes(self) -> None:
        seen: dict[str, object] = {}

        def qkv_matmul(
            x: torch.Tensor,
            wq: torch.Tensor,
            wk: torch.Tensor,
            wv: torch.Tensor,
            *,
            launch_config: dict,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            seen.update(x=x, wq=wq, wk=wk, wv=wv, launch_config=launch_config)
            return x @ wq, x @ wk, x @ wv

        vk = types.SimpleNamespace(
            qkv_matmul=types.SimpleNamespace(qkv_matmul=qkv_matmul)
        )
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        q_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        k_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        v_weight = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
        launch_config = {"sealed": 1}
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                return_value=launch_config,
            ) as static_config,
            mock.patch.object(
                primitive_runtime,
                "kernel_entrypoint",
                return_value=qkv_matmul,
            ),
        ):
            outputs = kernels.qkv_linear(x, q_weight, k_weight, v_weight)

        static_config.assert_called_once_with(
            None,
            "qkv_linear",
            {"q_width": 4, "kv_width": 2, "k": 3},
        )
        self.assertIs(seen["x"], x)
        self.assertTrue(torch.equal(seen["wq"], q_weight.t()))
        self.assertTrue(torch.equal(seen["wk"], k_weight.t()))
        self.assertTrue(torch.equal(seen["wv"], v_weight.t()))
        self.assertIs(seen["launch_config"], launch_config)
        self.assertTrue(torch.equal(outputs[0], x @ q_weight.t()))
        self.assertTrue(torch.equal(outputs[1], x @ k_weight.t()))
        self.assertTrue(torch.equal(outputs[2], x @ v_weight.t()))


class RowLayoutAdapterTests(unittest.TestCase):
    def test_merge_attention_heads_flattens_rows_into_fresh_storage(self) -> None:
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        output = kernels.merge_attention_heads(x)

        self.assertEqual(output.shape, (2, 12))
        self.assertTrue(torch.equal(output, x.reshape(2, 12)))
        self.assertNotEqual(
            output.untyped_storage().data_ptr(),
            x.untyped_storage().data_ptr(),
        )

    def test_split_last_axis_halves_copies_both_outputs(self) -> None:
        x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        left, right = kernels.split_last_axis_halves(x)

        self.assertTrue(torch.equal(left, x[:, :4]))
        self.assertTrue(torch.equal(right, x[:, 4:]))
        self.assertNotEqual(
            left.untyped_storage().data_ptr(), x.untyped_storage().data_ptr()
        )
        self.assertNotEqual(
            right.untyped_storage().data_ptr(), x.untyped_storage().data_ptr()
        )
        self.assertNotEqual(
            left.untyped_storage().data_ptr(),
            right.untyped_storage().data_ptr(),
        )

    def test_row_layout_adapters_reject_invalid_rank_or_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-3"):
            kernels.merge_attention_heads(torch.zeros((2, 4)))
        with self.assertRaisesRegex(ValueError, "even-width rank-2"):
            kernels.split_last_axis_halves(torch.zeros((2, 3)))


class RMSNormRuntimeBridgeTests(unittest.TestCase):
    def test_verified_launch_uses_fixed_epsilon_and_flat_weight(self) -> None:
        seen = {}

        def rmsnorm(
            x: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
            *,
            launch_config: dict,
        ) -> torch.Tensor:
            seen["x"] = x
            seen["weight"] = weight
            seen["eps"] = eps
            seen["launch_config"] = launch_config
            return x.clone()

        runtime = _qwen_test_runtime(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=3e-6,
        )
        vk = types.SimpleNamespace(
            rmsnorm=types.SimpleNamespace(rmsnorm=rmsnorm)
        )
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        weight = torch.ones((1, 3), dtype=torch.float32)
        launch_config = {"sealed": 1}
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                return_value=launch_config,
            ),
        ):
            output = kernels.rms_norm(x, weight, runtime)

        self.assertIs(seen["x"], x)
        self.assertEqual(seen["weight"].shape, (3,))
        self.assertEqual(seen["eps"], 3e-6)
        self.assertIs(seen["launch_config"], launch_config)
        self.assertTrue(torch.equal(output, x))

    def test_residual_launch_preserves_argument_and_output_order(self) -> None:
        seen = {}

        def rmsnorm_residual(x, residual, weight, eps, *, launch_config):
            seen["arguments"] = (x, residual, weight, eps, launch_config)
            return x + 1, residual + 2

        runtime = _qwen_test_runtime(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=3e-6,
        )
        vk = types.SimpleNamespace(
            rmsnorm_residual=types.SimpleNamespace(
                rmsnorm_residual=rmsnorm_residual
            )
        )
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        residual = x + 10
        weight = torch.ones((1, 3), dtype=torch.float32)
        launch_config = {"sealed": 1}
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                return_value=launch_config,
            ),
        ):
            normed, residual_out = kernels.add_rms_norm(
                x, residual, weight, runtime
            )

        self.assertIs(seen["arguments"][0], x)
        self.assertIs(seen["arguments"][1], residual)
        self.assertEqual(seen["arguments"][2].shape, (3,))
        self.assertEqual(seen["arguments"][3], 3e-6)
        self.assertIs(seen["arguments"][4], launch_config)
        self.assertTrue(torch.equal(normed, x + 1))
        self.assertTrue(torch.equal(residual_out, residual + 2))

    def test_qk_norm_launches_both_pinned_head_specializations(self) -> None:
        calls = []

        def head_rms_norm(x, weight, heads, eps, *, launch_config):
            calls.append((x, weight, heads, eps, launch_config))
            return x + heads

        runtime = _qwen_test_runtime(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=3e-6,
        )
        vk = types.SimpleNamespace(
            qk_norm=types.SimpleNamespace(head_rms_norm=head_rms_norm)
        )
        q = torch.zeros((2, 16, 128), dtype=torch.float32)
        k = torch.zeros((2, 8, 128), dtype=torch.float32)
        q_weight = torch.ones((1, 128), dtype=torch.float32)
        k_weight = torch.ones((1, 128), dtype=torch.float32) * 2
        q_config = {"H": 16}
        k_config = {"H": 8}
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                side_effect=[q_config, k_config],
            ),
        ):
            nq, nk = kernels.qk_norm(
                q, k, q_weight, k_weight, runtime
            )

        self.assertEqual([call[2] for call in calls], [16, 8])
        self.assertEqual([call[3] for call in calls], [3e-6, 3e-6])
        self.assertEqual([call[4] for call in calls], [q_config, k_config])
        self.assertEqual([call[0].shape for call in calls], [(2, 2048), (2, 1024)])
        self.assertTrue(torch.equal(calls[0][1], q_weight.reshape(-1)))
        self.assertTrue(torch.equal(calls[1][1], k_weight.reshape(-1)))
        self.assertTrue(torch.equal(nq, q + 16))
        self.assertTrue(torch.equal(nk, k + 8))


class SiluAndMulRuntimeBridgeTests(unittest.TestCase):
    def test_verified_launch_splits_each_row_on_the_last_axis(self) -> None:
        seen: dict[str, torch.Tensor] = {}

        def silu_mul(
            a: torch.Tensor, b: torch.Tensor, *, launch_config: dict
        ) -> torch.Tensor:
            seen["a"] = a
            seen["b"] = b
            seen["launch_config"] = launch_config
            return torch.nn.functional.silu(a) * b

        vk = types.SimpleNamespace(
            silu_mul=types.SimpleNamespace(silu_mul=silu_mul)
        )
        x = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        launch_config = {"sealed": 1}
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                return_value=launch_config,
            ),
            mock.patch.object(
                primitive_runtime, "kernel_entrypoint", return_value=silu_mul
            ),
        ):
            output = kernels.silu_and_mul(x)

        self.assertTrue(torch.equal(seen["a"], x[:, :3]))
        self.assertTrue(torch.equal(seen["b"], x[:, 3:]))
        self.assertIs(seen["launch_config"], launch_config)
        self.assertTrue(
            torch.equal(output, torch.nn.functional.silu(x[:, :3]) * x[:, 3:])
        )


class RotaryRuntimeBridgeTests(unittest.TestCase):
    def test_verified_launch_expands_each_token_table_row_per_head(self) -> None:
        calls = []

        def rope(x, cos_rows, sin_rows, *, launch_config):
            calls.append((x, cos_rows, sin_rows, launch_config))
            return x + len(calls)

        runtime = _qwen_test_runtime(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=3e-6,
        )
        vk = types.SimpleNamespace(rope=types.SimpleNamespace(rope=rope))
        positions = torch.tensor([1, 7], dtype=torch.int64)
        q = torch.zeros((2, 16, 128), dtype=torch.float32)
        k = torch.zeros((2, 8, 128), dtype=torch.float32)
        launch_config = {"sealed": 1}
        with (
            mock.patch.object(primitive_runtime, "verified_for", return_value=vk),
            mock.patch.object(
                primitive_runtime,
                "static_launch_config",
                return_value=launch_config,
            ),
        ):
            nq, nk = kernels.rotary_embed(positions, q, k, runtime)

        self.assertEqual(len(calls), 2)
        self.assertEqual([call[3] for call in calls], [launch_config, launch_config])
        self.assertEqual(calls[0][0].shape, (32, 128))
        self.assertEqual(calls[1][0].shape, (16, 128))
        self.assertEqual(calls[0][1].shape, (32, 64))
        self.assertEqual(calls[0][2].shape, (32, 64))
        self.assertEqual(calls[1][1].shape, (16, 64))
        self.assertEqual(calls[1][2].shape, (16, 64))

        for table_index in (1, 2):
            q_rows = calls[0][table_index]
            k_rows = calls[1][table_index]
            self.assertTrue(torch.equal(q_rows[:16], q_rows[0].expand(16, -1)))
            self.assertTrue(
                torch.equal(q_rows[16:], q_rows[16].expand(16, -1))
            )
            self.assertTrue(torch.equal(k_rows[:8], k_rows[0].expand(8, -1)))
            self.assertTrue(torch.equal(k_rows[8:], k_rows[8].expand(8, -1)))
            self.assertTrue(torch.equal(q_rows[0], k_rows[0]))
            self.assertTrue(torch.equal(q_rows[16], k_rows[8]))

        self.assertTrue(torch.equal(nq, q + 1))
        self.assertTrue(torch.equal(nk, k + 2))

    def test_shared_fallback_matches_transformers_half_rotation(self) -> None:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        runtime = _qwen_test_runtime(
            hidden_size=8,
            num_heads=2,
            num_kv_heads=1,
            head_dim=4,
            intermediate_size=16,
            rms_norm_eps=1e-6,
        )
        positions = torch.tensor([1, 7], dtype=torch.int64)
        q = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) / 10
        k = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4) / 10
        with mock.patch.object(primitive_runtime, "verified_for", return_value=None):
            actual_q, actual_k = kernels.rotary_embed(
                positions, q, k, runtime
            )

        config = runtime.runtime_config()
        inv_freq = 1.0 / (
            config["rope_theta"]
            ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4)
        )
        position_ids = positions.unsqueeze(0)
        frequencies = (
            inv_freq[None, :, None].expand(1, -1, 1)
            @ position_ids[:, None, :].float()
        ).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        expected_q, expected_k = apply_rotary_pos_emb(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            embedding.cos(),
            embedding.sin(),
        )
        expected_q = expected_q.squeeze(0).transpose(0, 1).contiguous()
        expected_k = expected_k.squeeze(0).transpose(0, 1).contiguous()
        self.assertTrue(torch.equal(actual_q, expected_q))
        self.assertTrue(torch.equal(actual_k, expected_k))


if __name__ == "__main__":
    unittest.main()
