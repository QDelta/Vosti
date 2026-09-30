"""Qualify and generate the neutral KV mutation interface."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "python", ROOT / "kernels"):
    sys.path.insert(0, str(path))

from scripts.deployment.common import qualify_proof_case
from scripts.verification.kernel_qualification_inventory import selected_kernel_cases
from scripts.verification.mutation_interface_codegen import render_mutation_interface
from scripts.verification.kernel_interface_registry import load_registry


def selected_cases():
    return [(c, constants, families) for c, constants, families in selected_kernel_cases()
            if c["evidence"] == "exact_effect_certificate"]


def generate():
    cases = [(c, constants, families, qualify_proof_case(c, constants, ("bfloat16", "float32")))
             for c, constants, families in selected_cases()]
    return render_mutation_interface(cases)


def check_or_write(body, manifest, *, write=False):
    paths = (ROOT / "src/boundary/backend_certificates/kv_store.rs", ROOT / "audit/kv_store_kernel_interface.json")
    _, interfaces = load_registry()
    entry = interfaces["kv_store"]
    if (entry["generated"] != paths[0].relative_to(ROOT).as_posix()
            or entry["manifest"] != paths[1].relative_to(ROOT).as_posix()
            or entry["generator"] != Path(__file__).relative_to(ROOT).as_posix()):
        raise ValueError("mutation generator paths differ from reviewed registry")
    for path, text in zip(paths, (body, manifest)):
        if write:
            path.write_text(text)
        elif not path.is_file() or path.read_text() != text:
            raise ValueError(f"stale mutation interface: {path}")
    print("Mutation interface " + ("updated" if write else "matches qualified source")
          + "; checked adapters additionally require Verus.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    check_or_write(*generate(), write=args.write)
