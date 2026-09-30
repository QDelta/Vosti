"""A fresh proof cannot bypass the Engine's generated interface coverage."""

from copy import deepcopy

import pytest

from vosti_kernels.kernel_interfaces import load_attention_interfaces, validate_attention_interface_case


def cases():
    manifest = load_attention_interfaces()
    implementations = {item["execution_identity"]: item
                       for interface in manifest["interfaces"] for item in interface["raw"]["implementations"]}
    result = []
    for contribution in manifest["inventory_contributions"]:
        contracts = {c["symbol_prefix"]: c for c in implementations[contribution["execution_identity"]]["standalone_contracts"]}
        result.append(dict(source=contribution["source"], kernel=contribution["kernel"],
            specialization=deepcopy(contribution["constants"]),
            structural_contract_digest=contracts["raw_batch_invariance"]["raw_contract_digest"],
            semantic_proof=dict(kind="conditional_selected_row",
                contract_digest=contracts["raw_selected_row_prefix_equivalence"]["raw_contract_digest"])))
    return manifest, result


def test_all_catalog_specializations_are_admitted():
    manifest, inventory = cases()
    assert inventory
    for case in inventory:
        validate_attention_interface_case(case, manifest)


@pytest.mark.parametrize("mutation", ["geometry", "tile", "batch", "selected", "execution", "duplicate", "omitted_goal"])
def test_unbound_or_mismatched_proof_is_rejected(mutation):
    manifest, inventory = cases()
    case = inventory[0]
    if mutation == "geometry":
        case["specialization"]["D_HEAD"] = 64
    elif mutation == "tile":
        case["specialization"]["BLOCK_M"] = 1024
    elif mutation == "batch":
        case["structural_contract_digest"] = "0" * 64
    elif mutation == "selected":
        case["semantic_proof"]["contract_digest"] = "0" * 64
    elif mutation == "execution":
        manifest["inventory_contributions"][0]["execution_identity"] = "0" * 64
    elif mutation == "duplicate":
        manifest["inventory_contributions"].append(deepcopy(manifest["inventory_contributions"][0]))
    else:
        identity = manifest["inventory_contributions"][0]["execution_identity"]
        for interface in manifest["interfaces"]:
            for implementation in interface["raw"]["implementations"]:
                if implementation["execution_identity"] == identity:
                    implementation["standalone_contracts"].pop()
    with pytest.raises(ValueError):
        validate_attention_interface_case(case, manifest)
