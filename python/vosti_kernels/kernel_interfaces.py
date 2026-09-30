"""Admission of exact proof cases against the generated kernel interfaces.

This runs when validating a static deployment, never during token serving.
It binds a separately qualified specialization to the interface used by Verus;
proving a new specialization alone does not silently extend that interface.
"""

import hashlib
import json
from pathlib import Path

RECTANGULAR_KERNEL_INTERFACES = {
    ("matmul.py", "matmul_kernel"): "linear",
    ("qkv_matmul.py", "qkv_matmul_kernel"): "qkv",
    ("rmsnorm.py", "rmsnorm_kernel"): "rms_norm",
    ("rmsnorm_residual.py", "rmsnorm_residual_kernel"): "residual_rms_norm",
    ("gemma_rmsnorm.py", "gemma_rmsnorm_kernel"): "offset_rms_norm",
    ("add.py", "add_kernel"): "add",
    ("silu_mul.py", "silu_mul_kernel"): "silu_mul",
    ("gelu_tanh_mul.py", "gelu_tanh_mul_kernel"): "gelu_tanh_mul",
    ("scale.py", "scale_kernel"): "scale",
    ("softcap.py", "softcap_kernel"): "softcap",
    ("embedding.py", "embedding_kernel"): "embedding",
    ("scaled_embedding.py", "scaled_embedding_kernel"): "scaled_embedding",
    ("qk_norm.py", "head_rms_norm_kernel"): "head_rms_norm",
    ("gemma_qk_norm.py", "gemma_head_rms_norm_kernel"): "offset_head_rms_norm",
    ("rope.py", "rope_kernel"): "rotary",
}


def load_attention_interfaces() -> dict:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "audit/attention_kernel_interfaces.json").read_text())
    if set(manifest) != {"schema_version", "kind", "generated_body_sha256", "interfaces", "inventory_contributions"}:
        raise ValueError("invalid generated attention interface manifest")
    if manifest["schema_version"] != 1 or manifest["kind"] != "geometry_dispatched_attention_interfaces":
        raise ValueError("unsupported generated attention interface manifest")
    body = (root / "src/boundary/backend_certificates/attention.rs").read_bytes()
    if hashlib.sha256(body).hexdigest() != manifest["generated_body_sha256"]:
        raise ValueError("generated attention interface source differs from its manifest")
    return manifest


def load_rectangular_interface(name: str) -> dict:
    if name not in RECTANGULAR_KERNEL_INTERFACES.values():
        raise ValueError("unimplemented rectangular interface admission")
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / f"audit/{name}_kernel_interface.json").read_text())
    fields = {
        "rectangular_kernel_interface": {"checked_adapter_sha256", "raw"},
        "geometry_dispatched_rectangular_interfaces": {"checked_dispatch_sha256", "dimensions", "interfaces"},
    }.get(manifest.get("kind"))
    if fields is None or set(manifest) != {"schema_version", "kind", "generated_body_sha256",
            "inventory_contributions"} | fields or manifest["schema_version"] != 1:
        raise ValueError("invalid rectangular interface manifest")
    body = (root / f"src/boundary/backend_certificates/{name}.rs").read_bytes()
    if hashlib.sha256(body).hexdigest() != manifest["generated_body_sha256"]:
        raise ValueError("generated rectangular interface source differs from its manifest")
    return manifest


def rectangular_implementation_records(manifest: dict) -> list[dict]:
    if manifest.get("kind") == "rectangular_kernel_interface":
        return manifest["raw"]["implementations"]
    if manifest.get("kind") == "geometry_dispatched_rectangular_interfaces":
        return [implementation for interface in manifest["interfaces"]
                for implementation in interface["raw"]["implementations"]]
    raise ValueError("unsupported rectangular interface manifest")


def load_mutation_interface() -> dict:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "audit/kv_store_kernel_interface.json").read_text())
    if (set(manifest) != {"schema_version", "kind", "generated_body_sha256",
            "inventory_contributions", "checked_dispatch_sha256", "dimensions", "interfaces"}
            or manifest["schema_version"] != 1
            or manifest["kind"] != "geometry_dispatched_mutation_interface"):
        raise ValueError("invalid mutation interface manifest")
    body = (root / "src/boundary/backend_certificates/kv_store.rs").read_bytes()
    if hashlib.sha256(body).hexdigest() != manifest["generated_body_sha256"]:
        raise ValueError("generated mutation interface source differs from its manifest")
    return manifest


def validate_mutation_interface_case(case: dict, manifest: dict) -> None:
    contributions = [c for c in manifest["inventory_contributions"]
        if c["source"] == case["source"] and c["kernel"] == case["kernel"]
        and json.dumps(c["constants"], sort_keys=True, allow_nan=False)
            == json.dumps(case["specialization"], sort_keys=True, allow_nan=False)]
    if len(contributions) != 1:
        raise ValueError("mutation specialization is not bound to a unique generated Verus interface")
    contribution = contributions[0]
    implementations = [i for entry in manifest["interfaces"] for i in entry["raw"]["implementations"]
                       if i["execution_identity"] == contribution["execution_identity"]]
    if len(implementations) != 1:
        raise ValueError("mutation specialization has no unique qualified implementation")
    goals = implementations[0]["standalone_contracts"]
    if (len(goals) != 1
            or goals[0].get("proof_kind") != "exact_effect"
            or goals[0].get("state_relation") != "before_after"):
        raise ValueError("mutation interface lacks its exact before/after contract")
    if case["structural_contract_digest"] != contribution["structural_contract_digest"]:
        raise ValueError("mutation batch qualification differs from its generated Verus interface")
    semantic = case.get("semantic_proof")
    if (not isinstance(semantic, dict) or semantic.get("kind") != "exact_effect"
            or semantic.get("contract_digest") != goals[0]["raw_contract_digest"]):
        raise ValueError("mutation effect proof differs from its generated Verus interface")


def validate_rectangular_interface_case(case: dict, manifest: dict) -> None:
    contributions = [c for c in manifest["inventory_contributions"]
        if c["source"] == case["source"] and c["kernel"] == case["kernel"]
        and json.dumps(c["constants"], sort_keys=True, allow_nan=False)
            == json.dumps(case["specialization"], sort_keys=True, allow_nan=False)]
    if len(contributions) != 1:
        raise ValueError("rectangular specialization is not bound to a unique generated Verus interface")
    implementations = [i for i in rectangular_implementation_records(manifest)
                       if i["execution_identity"] == contributions[0]["execution_identity"]]
    if len(implementations) != 1:
        raise ValueError("rectangular specialization has no unique qualified implementation")
    goals = implementations[0]["standalone_contracts"]
    if len(goals) != 1 or "proof_kind" in goals[0]:
        raise ValueError("rectangular interface lacks its exact row-projection contract")
    if case["structural_contract_digest"] != goals[0]["raw_contract_digest"]:
        raise ValueError("rectangular proof differs from its generated Verus interface")


def validate_attention_interface_case(case: dict, manifest: dict) -> None:
    contributions = [entry for entry in manifest["inventory_contributions"]
                     if entry["source"] == case["source"] and entry["kernel"] == case["kernel"]
                     and json.dumps(entry["constants"], sort_keys=True, allow_nan=False)
                     == json.dumps(case["specialization"], sort_keys=True, allow_nan=False)]
    if len(contributions) != 1:
        raise ValueError("attention specialization is not bound to a unique generated Verus interface")
    identity = contributions[0]["execution_identity"]
    implementations = [implementation for interface in manifest["interfaces"]
                       for implementation in interface["raw"]["implementations"]
                       if implementation["execution_identity"] == identity]
    if len(implementations) != 1:
        raise ValueError("attention specialization has no unique qualified implementation")
    contracts = implementations[0]["standalone_contracts"]
    by_digest = {record["raw_contract_digest"]: record for record in contracts}
    if len(contracts) != 2 or len(by_digest) != 2 or any("proof_kind" in r for r in contracts):
        raise ValueError("attention interface lacks its exact batch and selected-row contracts")
    if case["structural_contract_digest"] not in by_digest:
        raise ValueError("attention batch proof differs from its generated Verus interface")
    semantic = case.get("semantic_proof")
    if not isinstance(semantic, dict) or semantic.get("kind") != "conditional_selected_row" or (
        semantic.get("contract_digest") not in by_digest
        or semantic.get("contract_digest") == case["structural_contract_digest"]
    ):
        raise ValueError("attention selected-row proof differs from its generated Verus interface")
