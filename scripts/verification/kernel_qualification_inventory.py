"""Model-independent collection of complete static qualification cases."""

import importlib
import json

from scripts.deployment.common import _selected_proof_cases, required_proof_goals
from vosti_kernels.model_profile import load_scope
from scripts.verification.kernel_interface_registry import ROOT, load_registry
from scripts.verification.engine_kernel_bindings import required_kernel_goal_bindings


def case_key(contract, constants):
    # JSON preserves Boolean/int/float distinctions that Python tuple equality
    # would erase. No request length, batch size, or scheduling state is read.
    return (contract["source"], contract["kernel"],
            json.dumps(constants, sort_keys=True, allow_nan=False))


def family_cases(family, entry):
    profile = importlib.import_module(f"vosti_kernels.model_families.{family}.profile")
    scope = profile.scope()
    if scope != load_scope(ROOT / entry["scope"]):
        raise ValueError(f"profile scope differs from reviewed inventory: {family}")
    cases = _selected_proof_cases(scope, [launch
        for model in profile.model_profiles() for launch in model["launches"]])
    if entry["inventory"] == "contract_catalog":
        policy = importlib.import_module(f"scripts.deployment.model_families.{family}_scope").SCOPE
        alternatives = policy.deployed_cases()
        policy.validate_deployed_case_coverage(alternatives)
        cases += [(policy.contract_for(source, kernel), constants)
                  for source, kernel, constants in alternatives]
    elif entry["inventory"] != "profile_catalog":
        raise ValueError(f"unknown kernel qualification inventory: {family}")
    if {c["wrapper"] for c, _ in cases} != {c["wrapper"] for c in scope["kernel_contracts"]}:
        raise ValueError(f"qualification inventory does not cover the complete kernel scope: {family}")
    return cases


def selected_kernel_cases(families=None):
    qualification, _ = load_registry()
    selected = set(qualification) if families is None else set(families)
    if not selected or not selected <= qualification.keys():
        raise ValueError("qualification requested an empty or unknown family set")
    cases = {}
    for family in sorted(selected):
        for contract, constants in family_cases(family, qualification[family]):
            key = case_key(contract, constants)
            if key in cases:
                previous = cases[key][0]
                # Families may use different wrapper aliases for one kernel.
                # The raw execution identity does not contain those aliases
                # or optional scope-level output-role metadata. The complete
                # typed annotation determines the proved output surface.
                for field in ("source_sha256",):
                    if previous[field] != contract[field]:
                        raise ValueError(f"qualification contributors disagree on {field}: {key}")
                if required_kernel_goal_bindings(previous) != required_kernel_goal_bindings(contract):
                    raise ValueError(f"qualification contributors disagree on required goals: {key}")
            required_proof_goals(contract)
            cases.setdefault(key, (contract, constants, set()))[2].add(family)
    return [cases[key] for key in sorted(cases)]
