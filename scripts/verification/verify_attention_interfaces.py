#!/usr/bin/env python3
"""Qualify and generate the common attention catalog from admitted inventories."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "python", ROOT / "kernels"):
    sys.path.insert(0, str(path))

from scripts.verification.attention_interface_catalog import render_attention_catalog
from scripts.deployment.common import qualify_proof_case
from ir.verus_contract import render_verified_kernel_to_verus
from scripts.verification.kernel_interface_registry import load_registry
from scripts.verification.kernel_qualification_inventory import selected_kernel_cases
from scripts.verification.engine_kernel_bindings import attention_binding_for

GENERATED = ROOT / "src/boundary/backend_certificates/attention.rs"
MANIFEST = ROOT / "audit/attention_kernel_interfaces.json"


def selected_cases():
    return [(contract, constants, families)
            for contract, constants, families in selected_kernel_cases()
            if contract["wrapper"] in {"paged_attention", "paged_attention_swa"}]


def render_qualified_cases(cases):
    """Compose interfaces from the same artifacts used for deployment receipts."""
    implementations = []
    contributions = []
    for contract, constants, families, qualified in cases:
        implementations.append((qualified.contracts, attention_binding_for(contract)))
        execution_identity = render_verified_kernel_to_verus(qualified.contracts, symbol_prefix="raw").execution_identity
        contributions.append(dict(source=contract["source"], kernel=contract["kernel"],
                                  constants=constants, families=sorted(families),
                                  execution_identity=execution_identity))
    catalog = render_attention_catalog(tuple(implementations))
    manifest = json.loads(catalog.manifest())
    manifest["inventory_contributions"] = contributions
    return catalog.body, json.dumps(manifest, sort_keys=True, indent=2) + "\n"


def generate():
    cases = []
    for contract, constants, families in selected_cases():
        qualified = qualify_proof_case(contract, constants, ("bfloat16", "float32"))
        cases.append((contract, constants, families, qualified))
        print(f"[RAW PASS] {contract['kernel']} {constants}", flush=True)
    return render_qualified_cases(cases)


def check_or_write(body, manifest, *, write=False):
    _, interfaces = load_registry()
    entry = interfaces["attention"]
    if (entry["generated"] != GENERATED.relative_to(ROOT).as_posix()
            or entry["manifest"] != MANIFEST.relative_to(ROOT).as_posix()
            or entry["generator"] != Path(__file__).relative_to(ROOT).as_posix()):
        raise SystemExit("attention generator paths differ from the reviewed registry")
    for path, text in ((GENERATED, body), (MANIFEST, manifest)):
        if write:
            path.write_text(text)
        elif not path.is_file() or path.read_text() != text:
            raise SystemExit(f"stale generated attention interface: {path}")
    print("Attention raw catalog " + ("updated" if write else "matches qualified sources")
          + "; checked adapters additionally require Verus.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    check_or_write(*generate(), write=args.write)


if __name__ == "__main__":
    main()
