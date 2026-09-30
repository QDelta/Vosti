#!/usr/bin/env python3
"""Qualify exact static cases once and generate shared kernel interfaces.

Family filters check that family's complete inventory against the installed
interfaces. Regeneration requires the complete inventory: a family cannot own
or overwrite a shared interface. Attention, read-only row operators and mutation
effects have separate checked consumers of these generated raw interfaces.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "python", ROOT / "kernels"):
    sys.path.insert(0, str(path))

from scripts.deployment.common import qualify_proof_case
from scripts.verification.kernel_qualification_inventory import selected_kernel_cases
from scripts.verification.kernel_interface_registry import load_registry
from scripts.verification.verify_attention_interfaces import check_or_write, render_qualified_cases
import scripts.verification.verify_rectangular_interfaces as rectangular
import scripts.verification.verify_mutation_interfaces as mutation
from vosti_kernels.kernel_interfaces import (load_attention_interfaces, validate_attention_interface_case,
    load_mutation_interface, validate_mutation_interface_case,
    load_rectangular_interface, validate_rectangular_interface_case)


def verify(*, families=None, write=False):
    if write and families is not None:
        raise ValueError("shared interfaces can only be generated from the complete family inventory")
    cases = selected_kernel_cases(families)
    _, interfaces = load_registry()
    if set(interfaces) != {"attention", "kv_store"} | rectangular.OPERATORS.keys():
        raise ValueError("registered kernel interface has no generator")
    attention = []
    mutations = []
    rows = {name: [] for name in rectangular.OPERATORS}
    installed = None if write else load_attention_interfaces()
    installed_mutation = None if write else load_mutation_interface()
    installed_rows = {} if write else {name: load_rectangular_interface(name) for name in rows}
    for contract, constants, contributors in cases:
        qualified = qualify_proof_case(contract, constants, ("bfloat16", "float32"))
        if contract["evidence"] == "exact_effect_certificate":
            if installed_mutation is not None:
                validate_mutation_interface_case(qualified.receipt, installed_mutation)
            mutations.append((contract, constants, contributors, qualified))
        if contract["wrapper"] in {"paged_attention", "paged_attention_swa"}:
            if installed is not None:
                validate_attention_interface_case(qualified.receipt, installed)
            attention.append((contract, constants, contributors, qualified))
        for name, source_identity in rectangular.OPERATORS.items():
            if (contract["source"], contract["kernel"]) == source_identity:
                if not write:
                    validate_rectangular_interface_case(qualified.receipt, installed_rows[name])
                rows[name].append((contract, constants, contributors, qualified))
        print(f"[QUALIFIED] {contract['kernel']} {constants}", flush=True)
    if families is None:
        check_or_write(*render_qualified_cases(attention), write=write)
        mutation.check_or_write(*mutation.render_mutation_interface(mutations), write=write)
        for name, qualified_cases in rows.items():
            rectangular.check_or_write(name, *rectangular.render_qualified_cases(qualified_cases), write=write)
    row_count = sum(len(items) for items in rows.values())
    print(f"[PASS] {len(cases)} exact kernel cases; {len(attention)} attention and "
          f"{row_count} rectangular and {len(mutations)} mutation implementations. "
          "Generated adapters additionally require Verus.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append")
    parser.add_argument("--write-generated", action="store_true")
    args = parser.parse_args()
    if args.write_generated and args.family is not None:
        parser.error("use scripts/verification/verify_kernel_contracts.py --write-generated for the shared inventory")
    verify(families=args.family, write=args.write_generated)


if __name__ == "__main__":
    main()
