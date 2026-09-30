"""Infer checked row adapters directly from qualified typed kernel contracts.

No model family, wrapper, engine semantic function, or manually named theorem
binding is input. Static alternatives must export the same complete raw theory
and checked adapter. The resulting row map denotes ONE deployed implementation,
not equality between implementations. Rendering alone is not Verus verification.
"""

from dataclasses import dataclass

from ir.axis_projection_contract import normalize_axis_projection_contract
from scripts.verification.kernel_contract_codegen import (
    render_raw_rectangular_axis_adapter,
)
from scripts.verification.kernel_interface_codegen import RenderedKernelInterface, render_kernel_interface


def inferred_binding(axis_contract, raw_contract, raw_fragment):
    axis = axis_contract.to_data()
    types = axis["parameter_types"]

    def framework_type(name):
        typ = types[name]
        if typ == {"kind": "float"}:
            return "Scalar"
        if typ["kind"] == "tensor":
            result = {("float", 1): "Tensor1D", ("float", 2): "Tensor2D",
                      ("int", 1): "IntTensor1D"}.get(
                          (typ["element"]["kind"], len(typ["shape"])))
            if result:
                return result
        raise ValueError(f"unsupported rectangular parameter type: {name}: {typ}")

    arguments = [dict(name="input_" + name, kernel=name,
                      type=framework_type(name), mode=mode)
                 for mode, names in (
                     ("projected", axis["projected_inputs"]),
                     ("shared", axis["shared_inputs"] + axis["shared_scalar_parameters"]))
                 for name in names]
    return dict(
        kind="axis_projection", contract_digest=axis_contract.digest,
        raw_contract_digest=raw_contract.digest,
        specialization={k: v["value"] for k, v in axis["constants"].items()},
        proof_name="row_projection_certificate", verus_lowering="raw_rectangular",
        arguments=arguments,
        outputs=[dict(kernel=name, function="output_" + name, type=framework_type(name))
                 for name in raw_fragment.output_parameters],
        domain_policy="raw_preconditions",
    )


@dataclass(frozen=True)
class RenderedRectangularInterface:
    raw: RenderedKernelInterface
    adapter_body: str

    @property
    def body(self):
        return self.raw.body + self.adapter_body


def render_rectangular_interface(implementations, *, symbol_prefix="raw"):
    raw = render_kernel_interface(implementations, symbol_prefix=symbol_prefix)
    adapters = []
    for contracts in implementations:
        if len(contracts) != 1:
            raise ValueError("rectangular interface requires exactly one row-projection goal")
        contract = contracts[0]
        axis = normalize_axis_projection_contract(contract)
        # Every implementation has already passed the complete raw theory
        # comparison; names/types are common but execution identities stay apart.
        own_fragment = next(bundle.contracts[0] for bundle in raw.implementations
                            if bundle.contracts[0].raw_contract_digest == contract.digest)
        binding = inferred_binding(axis, contract, own_fragment)
        adapter = render_raw_rectangular_axis_adapter(contract, axis, own_fragment, binding)
        if "external_body" in adapter or "assume(" in adapter:
            raise ValueError("rectangular adapter must not introduce trusted proof bodies")
        adapters.append(adapter)
    if any(body != adapters[0] for body in adapters):
        raise ValueError("static implementations need different checked rectangular adapters")
    return RenderedRectangularInterface(raw, adapters[0])
