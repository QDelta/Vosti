"""One abstract contract for alternative, separately qualified static kernels.

This is interface equality, never execution equality. A serving deployment
chooses one implementation before execution. Every admitted implementation must
independently prove exactly the exported Verus theory. Its execution identity
and proof artifacts remain separate in the manifest; no merged execution ID or
theorem equating two implementations is produced.
"""

from dataclasses import dataclass
import hashlib
import json

from ir.contract_schema import validate_relational_contract
from ir.relational_artifact import VerifiedDataflowContract
from ir.exact_effect_artifact import VerifiedExactEffectContract
from ir.verus_contract import (
    RenderedVerusKernel, render_contract_manifest, render_verified_kernel_to_verus,
)


def _logical_body(body: str) -> str:
    # Remove only renderer-owned provenance lines. In particular preserve all
    # formulas, types, and program mappings. Analyzer-condition labels live
    # in manifest records and are compared separately below. A new
    # renderer surface fails comparison until it is deliberately supported.
    metadata = ("// Proved relational contract: ", "// Proved exact-effect contract: ",
                "// Whole Triton source: ")
    return "\n".join(line for line in body.splitlines()
                     if not line.startswith(metadata)) + "\n"


@dataclass(frozen=True)
class RenderedKernelInterface:
    implementations: tuple[RenderedVerusKernel, ...]
    logical_body: str

    @property
    def representative(self) -> RenderedVerusKernel:
        """Names/layout only; this does not select the deployed implementation."""
        return self.implementations[0]

    @property
    def body(self) -> str:
        return (
            "// Abstract interface for ONE fixed, separately qualified deployment.\n"
            "// Alternative implementations prove this same theory; their outputs\n"
            "// are not equated. See the separate implementation proof records.\n"
            + self.logical_body
        )

    def manifest(self) -> str:
        return json.dumps({
            "schema_version": 1,
            "kind": "static_implementation_interface",
            "interpretation": "one_fixed_deployed_implementation",
            "logical_body_sha256": hashlib.sha256(self.logical_body.encode()).hexdigest(),
            "generated_body_sha256": hashlib.sha256(self.body.encode()).hexdigest(),
            "implementations": [json.loads(render_contract_manifest(bundle))
                                for bundle in self.implementations],
        }, sort_keys=True, indent=2) + "\n"


def render_kernel_interface(
    implementations: tuple[tuple[VerifiedDataflowContract | VerifiedExactEffectContract, ...], ...],
    *, symbol_prefix: str,
) -> RenderedKernelInterface:
    """Require the same complete logical export for every static alternative.

    A bundle still denotes exactly one execution: the existing raw renderer
    rejects mixing specializations inside it. Only the resulting *theories*
    are compared here. This is deliberately stricter than matching annotation
    names, output roles, or a subset of theorem conclusions.
    """
    if not implementations:
        raise ValueError("kernel interface has no qualified implementation")
    bundles = {}
    source_identity = None
    logical_body = None
    logical_records = None
    for contracts in implementations:
        bundle = render_verified_kernel_to_verus(contracts, symbol_prefix=symbol_prefix)
        if isinstance(contracts[0], VerifiedExactEffectContract):
            surface = contracts[0].to_data()["plan"]["execution"]
        else:
            surface = validate_relational_contract(contracts[0], family="static kernel interface",
                                                  preserve_analyzer_conditions=True).raw
        current_source = (surface["kernel"], surface["source_sha256"])
        if source_identity is None:
            source_identity = current_source
        if source_identity != current_source:
            raise ValueError("kernel interface mixes kernel sources")
        current_body = _logical_body(bundle.body)
        if logical_body is None:
            logical_body = current_body
        if logical_body != current_body:
            raise ValueError("static implementations export different logical contracts")
        # Match the independent audit's theory comparison, including labels
        # whose interpretations are not present in the rendered predicate body.
        current_records = [
            {key: value for key, value in record.items()
             if key not in {"raw_contract_digest", "generated_body_sha256"}}
            for record in json.loads(render_contract_manifest(bundle))["standalone_contracts"]
        ]
        if logical_records is None:
            logical_records = current_records
        if logical_records != current_records:
            raise ValueError("static implementations export different logical contract records")
        if bundle.execution_identity in bundles:
            raise ValueError("duplicate implementation in kernel interface")
        bundles[bundle.execution_identity] = bundle
    assert logical_body is not None
    return RenderedKernelInterface(tuple(bundles[key] for key in sorted(bundles)), logical_body)
