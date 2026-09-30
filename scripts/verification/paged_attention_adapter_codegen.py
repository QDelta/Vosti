"""Checked paged-layout consumers of the general raw Verus contract generator.

The raw renderer owns every imported theorem and analyzer predicate. This
consumer supplies representation proofs shared by full/SWA attention, without
model-family or kernel-name dispatch. Its output must be checked by Verus before
use; successful rendering is not itself evidence that an adapter is proved.
"""

from dataclasses import dataclass, fields
from pathlib import Path
import re

from ir.contract_schema import validate_relational_contract
from ir.relational_artifact import VerifiedDataflowContract
from ir.verus_contract import RenderedVerusKernel, render_verified_kernel_to_verus, _identifier
from scripts.verification.kernel_interface_codegen import RenderedKernelInterface, render_kernel_interface
from scripts.verification.engine_kernel_bindings import PagedAttentionBinding


TEMPLATES = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class PagedAttentionGeometry:
    query_tokens: str
    query_heads: str
    kv_heads: str
    pages: str
    batch: str
    table_width: str
    head_dim: int


def paged_attention_geometry(contracts, binding):
    """Validate the bound paged representation and infer its tensor extents."""
    _require(isinstance(binding, PagedAttentionBinding), "expected typed paged binding")
    for field in fields(binding):
        value = getattr(binding, field.name)
        if value is not None:
            _identifier(value, f"invalid paged binding {field.name}")
        else:
            _require(field.name == "window", "only the window binding may be absent")
    goals = {validate_relational_contract(c, family="checked paged layout",
                preserve_analyzer_conditions=True).raw["goal_name"]: c for c in contracts}
    _require(binding.batch_goal != binding.selected_goal
             and len(goals) == len(contracts)
             and set(goals) == {binding.batch_goal, binding.selected_goal},
             "expected bound batch and selected-row goals over one execution")
    surface = validate_relational_contract(goals[binding.batch_goal], family="checked paged layout",
                                          preserve_analyzer_conditions=True)
    params = surface.parameters

    def tensor_shape(port, kind, rank):
        _require(port in params, f"missing bound port {port!r}")
        typ = params[port]["type"]
        _require(typ["kind"] == "tensor" and typ["element"] == {"kind": kind}
                 and len(typ["shape"]) == rank, f"incorrect bound tensor type: {port}")
        return typ["shape"]

    def dynamic(expr):
        _require(expr["kind"] == "var", "expected a dynamic paged extent")
        return expr["name"]

    q_tokens, q_heads, dim = tensor_shape(binding.query, "float", 3)
    pages, page, kv_heads, k_dim = tensor_shape(binding.keys, "float", 4)
    batch, table_width = tensor_shape(binding.page_table, "int", 2)
    _require(dim["kind"] == "int" and type(dim["value"]) is int and dim["value"] > 0
             and dim == k_dim, "positive common static head dimension is required")
    # This is the compiled ENGINE representation, not a kernel tile restriction.
    # Kernel tile constants and their names are deliberately not inspected.
    _require(page == {"kind": "int", "value": 64}, "incompatible compiled engine page extent")
    geometry = PagedAttentionGeometry(*(dynamic(d) for d in
        (q_tokens, q_heads, kv_heads, pages, batch, table_width)), dim["value"])
    dynamic_names = [getattr(geometry, f.name) for f in fields(geometry) if f.name != "head_dim"]
    _require(len(set(dynamic_names + [binding.total_keys])) == 7, "aliased paged extent roles")
    _require(binding.total_keys not in surface.constants, "total key extent must remain dynamic")
    port_names = [getattr(binding, name) for name in (
        "query", "keys", "values", "output", "logsumexp", "page_table", "query_offsets",
        "key_offsets", "scale", "max_query")]
    if binding.window is not None:
        port_names.append(binding.window)
    _require(len(set(port_names)) == len(port_names), "aliased paged port roles")
    def tensor(kind, shape):
        return {"kind": "tensor", "element": {"kind": kind}, "shape": shape}
    offsets = [{"kind": "binary", "op": "+", "lhs": batch, "rhs": {"kind": "int", "value": 1}}]
    expected = {
        binding.query: tensor("float", [q_tokens, q_heads, dim]),
        binding.keys: tensor("float", [pages, page, kv_heads, dim]),
        binding.values: tensor("float", [pages, page, kv_heads, dim]),
        binding.output: tensor("float", [q_tokens, q_heads, dim]),
        binding.logsumexp: tensor("float", [q_heads, q_tokens]),
        binding.page_table: tensor("int", [batch, table_width]),
        binding.query_offsets: tensor("int", offsets), binding.key_offsets: tensor("int", offsets),
        binding.scale: {"kind": "float"}, binding.max_query: {"kind": "int"},
    }
    if binding.window is not None:
        expected[binding.window] = {"kind": "int"}
    _require({name: p["type"] for name, p in params.items()} == expected,
             "typed launch parameters differ from the bound paged layout")
    return geometry


@dataclass(frozen=True)
class RenderedPagedAttentionAdapter:
    raw: RenderedVerusKernel
    adapter_body: str

    @property
    def body(self) -> str:
        # Consumers place each specialization in its own module. The template
        # function names are module-local; only the raw artifact uses a prefix.
        return "verus! {\n" + self.raw.body + "\n}\n" + self.adapter_body


@dataclass(frozen=True)
class RenderedPagedAttentionInterface:
    raw: RenderedKernelInterface
    adapter_body: str

    @property
    def body(self) -> str:
        return "verus! {\n" + self.raw.body + "\n}\n" + self.adapter_body


def render_checked_paged_attention_interface(
    implementations: tuple[tuple[VerifiedDataflowContract, ...], ...],
    *, symbol_prefix: str, binding: PagedAttentionBinding,
) -> RenderedPagedAttentionInterface:
    raw = render_kernel_interface(implementations, symbol_prefix=symbol_prefix)
    adapters = [render_checked_paged_attention(contracts, symbol_prefix=symbol_prefix, binding=binding)
                for contracts in implementations]
    body = adapters[0].adapter_body
    _require(all(adapter.adapter_body == body for adapter in adapters),
             "static implementations need different checked adapters")
    return RenderedPagedAttentionInterface(raw, body)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"unsupported checked paged attention adapter: {message}")


def render_checked_paged_attention(
    contracts: tuple[VerifiedDataflowContract, ...], *, symbol_prefix: str, binding: PagedAttentionBinding,
) -> RenderedPagedAttentionAdapter:
    raw = render_verified_kernel_to_verus(contracts, symbol_prefix=symbol_prefix)
    geometry = paged_attention_geometry(contracts, binding)
    dim, swa = geometry.head_dim, binding.window is not None
    fragments = {fragment.raw_contract_digest: fragment for fragment in raw.contracts}
    goals = {c.to_data()["theorem_contract"]["goal_name"]: c for c in contracts}
    batch = fragments[goals[binding.batch_goal].digest]
    selected = fragments[goals[binding.selected_goal].digest]
    _require(not batch.analyzer_conditions,
             "batch conditions require an explicit consumer interface before use")
    _require(set(raw.output_parameters) == {binding.output, binding.logsumexp}, "unexpected written output surface")
    values = {
        "SIDE": raw.side_type, "FREE": batch.free_type, "PRE": batch.pre_name,
        "POST": batch.post_name, "CERT": batch.certificate_name,
        "EXEC": raw.execute_name, "D": str(dim),
        "WINDOW_FIELD": f"{binding.window}: window as int," if swa else "",
        "WINDOW_REQUIREMENT": "window > 0," if swa else "",
        "WINDOW_VALIDITY": "&& window > 0" if swa else "",
        "SELECTED_FREE": selected.free_type, "SELECTED_PRE": selected.pre_name,
        "SELECTED_POST": selected.post_name, "SELECTED_CERT": selected.certificate_name,
        # Preserve every condition, including separate assumption/obligation
        # entries with the same label. No interpretation or implication axiom.
        "SELECTED_CONDITIONS": "\n    && ".join(
            f"{condition.predicate_name}(left, right, free)"
            for condition in selected.analyzer_conditions) or "true",
    }
    values.update({"PORT_" + role.upper(): getattr(binding, role) for role in (
        "query", "keys", "values", "output", "logsumexp", "page_table", "query_offsets",
        "key_offsets", "scale", "max_query", "total_keys", "batch_selector", "left_selector", "right_selector")})
    values.update({"EXTENT_" + f.name.upper(): getattr(geometry, f.name)
                   for f in fields(geometry) if f.name != "head_dim"})
    template = "\n".join((TEMPLATES / name).read_text() for name in (
        "paged_attention_batch_adapter.rs", "paged_attention_selected_adapter.rs",
        "paged_attention_canonical_adapter.rs", "paged_attention_launch_adapter.rs",
        "paged_attention_geometry_adapter.rs"))
    _require("external_body" not in template and "assume(" not in template,
             "checked templates must not introduce trusted proof bodies")
    body = re.sub(r"__([A-Z_]+)__", lambda match: values[match[1]], template)
    return RenderedPagedAttentionAdapter(raw, body)
