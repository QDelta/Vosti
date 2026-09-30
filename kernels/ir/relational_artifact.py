"""Canonical artifact for the unified relational analysis.

This is a serialization of the typed objects used by that analysis, not a new
IR, a diagnostic-text parser, or an independently checkable solver proof. The
closed node inventory prevents new IR/evidence fields from silently escaping
review. Raw Verus and projection consumers accept its exact theorem only when
they can preserve every additional condition. Production batch and selected-row
catalogs consume this artifact through the qualified relational verifier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import json
import types
from typing import get_args, get_origin, get_type_hints

from . import __dict__ as _ir_types
from . import Assign, Kernel, MaskedStore, TensorType, Var
from .contract_schema import validate_relational_contract_data
from .identity_transition import (
    AllFalseFact, IdentityTransitionReport, RegionalFactRequirement,
    StateIdentityEvidence,
)
from .positional import NeutralityReport, Pred, TensorFacts
from .preprocess import build_type_env
from .regions import GuardedRegion, collect_write_stmts, get_write_tensors
from .relational_contract import _type_data, relational_contract_data
from .relational_dataflow import (
    AxiswiseOrdinalMap, ConditionalIterationIdentity, ControlAlignment,
    DiscreteValueAlignment, LoopAlignment, OrderedAlignmentSummary, PairedDemand,
    RangeDifferenceNeutrality, RelevantStatementAlignment, RelationalDataflowReport,
    discrete_definition_graph,
)
from .proof_preparation import PreparedAnnotationProof
from .smt import ProofCheck
from .regional_obligations import relevant_output_writes, extract_loop_iter_ranges


@dataclass(frozen=True)
class DataflowEvidence:
    kernel: Kernel
    checks: tuple[ProofCheck, ...]
    required_tensor_reads: tuple[str, ...]
    used_assumptions: tuple[str, ...]
    external_obligations: tuple[str, ...]
    alignments: tuple[OrderedAlignmentSummary, ...]


# Explicit field inventories, not unrestricted dataclasses.asdict/pickle.
# Adding a typed node or field requires an intentional schema decision here.
_IR_FIELDS = {
    "IntType": "", "FloatType": "", "BoolType": "",
    "TensorType": "elem_type dims",
    "Var": "type name", "IntLit": "type value", "FloatLit": "type value",
    "BoolLit": "type value", "BinOp": "type op lhs rhs",
    "Min": "type args", "Max": "type args", "Zeros": "type shape",
    "Full": "type shape value", "Arange": "type start stop",
    "Where": "type cond on_true on_false",
    "ReduceMax": "type value axis", "ReduceSum": "type value axis",
    "Exp2": "type value", "Sigmoid": "type value", "Rsqrt": "type value",
    "Log2": "type value", "Not": "type value", "Maximum": "type lhs rhs",
    "Cast": "type value kind target", "Unsqueeze": "type value axis",
    "Squeeze": "type value axis", "BroadcastTo": "type value shape",
    "Transpose": "type value permutation", "Slice": "start stop",
    "TensorIndex": "type base indices", "TensorView": "type base region",
    "MaskedLoad": "type base region mask", "VarDecl": "var type",
    "Let": "var value", "Assign": "target op value",
    "MaskedStore": "base region value mask", "Range": "start stop",
    "For": "var iters body", "If": "cond then_body else_body",
    "Grid": "iters decls body", "GridIter": "var iters",
    "Param": "name type", "Kernel": "name params grid",
}
_FIELDS = {
    **{_ir_types[name]: names.split() for name, names in _IR_FIELDS.items()},
    Pred: ["rank", "body"],
    TensorFacts: "rank zero_where neg_inf_where one_where true_where false_where unchanged_from".split(),
    GuardedRegion: ["region", "guard"],
    AxiswiseOrdinalMap: ["rank"],
    PairedDemand: "tensor left right coordinates justification".split(),
    RelevantStatementAlignment: "ordinal write iterators control_predicates".split(),
    AllFalseFact: ["variable"],
    RegionalFactRequirement: "fact state demand".split(),
    StateIdentityEvidence: "state demand final_facts dependencies fact_requirement".split(),
    NeutralityReport: "proved failures used_assumptions final_facts".split(),
    IdentityTransitionReport: "fact fact_facts whole_report states requirements failures used_assumptions external_assumptions".split(),
    ConditionalIterationIdentity: "fact state_values left right".split(),
    RangeDifferenceNeutrality: "iterator identity left_exclusive_checks right_exclusive_checks".split(),
    LoopAlignment: "iterator scope strategy proof_check carried_values live_out_values identity_state_values conditional_identities range_difference_neutrality".split(),
    ControlAlignment: "predicate iterators proof_check".split(),
    DiscreteValueAlignment: "variable statement_ordinal strategy proof_checks".split(),
    OrderedAlignmentSummary: "output_tensor paired_demands relevant_statements loop_alignments control_alignments discrete_value_alignments theorem_vacuous".split(),
    ProofCheck: ["name", "proved"],
    DataflowEvidence: "kernel checks required_tensor_reads used_assumptions external_obligations alignments".split(),
}
_NODES = {cls.__name__: cls for cls in _FIELDS}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid unified artifact: {message}")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@lru_cache(maxsize=None)
def _hints(cls: type) -> dict:
    # Only diagnostic prose is excluded. No proof meaning is recovered from it.
    omitted = {"details"} if cls is ProofCheck else set()
    _require(
        {item.name for item in fields(cls)} == set(_FIELDS[cls]) | omitted,
        f"unreviewed fields on {cls.__name__}",
    )
    return get_type_hints(cls)


def _encode(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        # IR contains -inf constants. Hex retains signed zero and exact bits;
        # JSON non-finite numbers and locale/repr-dependent encodings are avoided.
        _require(value == value, "NaN literal")
        return {"float": value.hex()}
    if type(value) in (tuple, list):
        return [_encode(item) for item in value]
    if type(value) in (set, frozenset):
        return sorted((_encode(item) for item in value), key=_canonical)
    if type(value) is dict:
        _require(all(type(key) is str for key in value), "non-string dictionary key")
        return {key: _encode(item) for key, item in value.items()}
    cls = type(value)
    _require(cls in _FIELDS, f"unsupported typed node {cls.__name__}")
    _hints(cls)
    return {"node": cls.__name__, **{
        name: _encode(getattr(value, name)) for name in _FIELDS[cls]
    }}


def _decode(value, expected):
    origin, args = get_origin(expected), get_args(expected)
    if origin is types.UnionType:
        for choice in args:
            try:
                return _decode(value, choice)
            except (ValueError, TypeError, AssertionError):
                pass
        raise ValueError(f"invalid unified artifact: expected {expected}")
    if expected in (str, int, bool, type(None)):
        _require(type(value) is expected, f"expected {expected.__name__}")
        return value
    if expected is float:
        _require(isinstance(value, dict) and set(value) == {"float"}, "float fields")
        _require(type(value["float"]) is str, "float encoding")
        number = float.fromhex(value["float"])
        _require(number == number and number.hex() == value["float"], "noncanonical float")
        return number
    if origin in (tuple, list, set, frozenset, Sequence):
        _require(type(value) is list, "expected array")
        if origin is tuple and args[-1:] != (Ellipsis,):
            _require(len(value) == len(args), "tuple length")
            decoded = [_decode(item, typ) for item, typ in zip(value, args, strict=True)]
        else:
            decoded = [_decode(item, args[0]) for item in value]
        if origin in (set, frozenset):
            keys = [_canonical(item) for item in value]
            _require(keys == sorted(set(keys)), "noncanonical set")
            return origin(decoded)
        return tuple(decoded) if origin is tuple else decoded
    if origin is dict:
        _require(type(value) is dict and args[0] is str, "expected string-keyed object")
        return {_decode(key, str): _decode(item, args[1]) for key, item in value.items()}
    _require(isinstance(value, dict) and type(value.get("node")) is str, "missing node tag")
    cls = _NODES.get(value["node"])
    _require(cls is not None and issubclass(cls, expected), "unknown or wrongly typed node")
    _require(set(value) == {"node", *_FIELDS[cls]}, f"unexpected {cls.__name__} fields")
    hints = _hints(cls)
    kwargs = {name: _decode(value[name], hints[name]) for name in _FIELDS[cls]}
    if cls.__name__ == "BinOp":
        _require(kwargs["op"] in {
            "+", "-", "*", "/", "//", "%", "cdiv", "@",
            "<", "<=", ">", ">=", "==", "!=", "and", "or",
        }, "unsupported binary operation")
    if cls.__name__ == "Assign":
        _require(kwargs["op"] in {None, "+", "-", "*", "/", "//", "%", "@", "and", "or"},
                 "unsupported assignment operation")
    if cls.__name__ == "Cast":
        _require(kwargs["kind"] in {"float", "int32"}, "unsupported cast kind")
    if cls is ProofCheck:
        kwargs["details"] = ""
    return cls(**kwargs)


def _validate_evidence(evidence: DataflowEvidence, theorem: dict) -> None:
    _require(evidence.kernel.name == theorem["kernel"], "kernel identity differs")
    _require(bool(evidence.checks) and all(c.proved for c in evidence.checks), "failed proof")
    for name in ("required_tensor_reads", "used_assumptions", "external_obligations"):
        values = getattr(evidence, name)
        _require(tuple(sorted(set(values))) == values and all(values), f"malformed {name}")
    parameters = {p.name: p for p in evidence.kernel.params}
    _require(set(evidence.required_tensor_reads) <= parameters.keys(), "unknown input read")
    _require(
        [p.name for p in evidence.kernel.params] == [p["name"] for p in theorem["parameters"]],
        "parameter inventory differs",
    )
    _require(all(_type_data(parameters[p["name"]].type) == p["type"]
                 for p in theorem["parameters"]), "parameter types differ")
    outputs = [item["left"]["tensor"] for item in theorem["theorem"]["post"]]
    _require([a.output_tensor for a in evidence.alignments] == outputs, "output coverage differs")
    for alignment in evidence.alignments:
        _require(not alignment.theorem_vacuous, "vacuous proof")
        _require(bool(alignment.relevant_statements), "missing ordered operations")
        _require(tuple(s.ordinal for s in alignment.relevant_statements)
                 == tuple(range(len(alignment.relevant_statements))), "statement order differs")
        _require(any(s.target == alignment.output_tensor for s in alignment.relevant_statements),
                 "no output write")
        _require(bool(alignment.paired_demands)
                 and alignment.paired_demands[0].tensor == alignment.output_tensor,
                 "missing output demand")
        for demand in alignment.paired_demands:
            _require(demand.coordinates.rank == len(demand.left.region) == len(demand.right.region),
                     "coordinate rank differs")
        for loop in alignment.loop_alignments:
            allowed = ({"same_range", "filtered_relevant_range", "singleton_relevant"}
                       if loop.scope == "grid" else
                       {"same_range_fold", "same_range_map", "ordered_common_range_with_identity_difference"}
                       if loop.scope == "source" else set())
            _require(loop.strategy in allowed, "unsupported loop alignment")
            if loop.strategy == "ordered_common_range_with_identity_difference":
                _require(loop.range_difference_neutrality is not None
                         and loop.range_difference_neutrality.proved, "unproved range difference")
                _require(loop.range_difference_neutrality.iterator == loop.iterator
                         and loop.range_difference_neutrality.identity in loop.conditional_identities,
                         "range identity differs")
            for identity in loop.conditional_identities:
                _require(identity.left.proved and identity.right.proved, "failed conditional identity")
                _require(not identity.left.external_assumptions and not identity.right.external_assumptions,
                         "unaccounted identity assumption")
        discrete_ordinals: set[int] = set()
        _, discrete_dependencies, required_discrete = discrete_definition_graph(
            evidence.kernel, relevant_output_writes(evidence.kernel, alignment.output_tensor),
            build_type_env(evidence.kernel),
        )
        proved_checks = {check.name for check in evidence.checks if check.proved}
        for discrete in alignment.discrete_value_alignments:
            _require(discrete.strategy in {"exact_element_relation", "same_operation_congruence"},
                     "unsupported discrete alignment")
            ordinal = discrete.statement_ordinal
            _require(0 <= ordinal < len(alignment.relevant_statements)
                     and alignment.relevant_statements[ordinal].target == discrete.variable,
                     "discrete definition identity differs")
            _require(ordinal not in discrete_ordinals, "duplicate discrete definition alignment")
            discrete_ordinals.add(ordinal)
            _require(bool(discrete.proof_checks)
                     and all(f"{alignment.output_tensor}:{name}" in proved_checks
                             and name.startswith(f"definition_{ordinal}:discrete_{discrete.variable}_")
                             for name in discrete.proof_checks),
                     "missing discrete definition proof checks")
            if discrete.strategy == "same_operation_congruence":
                required_discrete.update(discrete_dependencies.get(ordinal, {-1}))
        _require(required_discrete <= discrete_ordinals, "incomplete discrete definition coverage")


@dataclass(frozen=True)
class VerifiedDataflowContract:
    """Canonical unified evidence used by production relational-contract consumers.

    Schema validation is not authentication or independent proof checking. Like
    the existing artifacts, issuance relies on the trusted in-process verifier.
    """

    canonical_json: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    def written_tensor_parameters(self) -> tuple[str, ...]:
        """Execution outputs from validated typed IR, independent of a goal.

        This identifies written parameters, not their exact functional effect
        or proof coverage. Unmentioned outputs must not become preserved state
        simply because a particular annotation only relates another output.
        """
        kernel = _decode(self.to_data()["evidence"]["kernel"], Kernel)
        env = build_type_env(kernel)
        written: set[str] = set()
        for conditional in collect_write_stmts(kernel):
            match conditional.write:
                case Assign(target=Var(name=name)) if not isinstance(env[name], TensorType):
                    continue
                case Assign(target=target):
                    written |= get_write_tensors(target, env)
                case MaskedStore(base=base):
                    written.add(base.name)
                case _:
                    raise ValueError("unsupported execution write")
        return tuple(p.name for p in kernel.params
                     if isinstance(p.type, TensorType) and p.name in written)

    def execution_iterator_names(self) -> tuple[str, ...]:
        """Iteration coordinates are not caller-chosen theorem witnesses."""
        kernel = _decode(self.to_data()["evidence"]["kernel"], Kernel)
        return tuple(sorted(extract_loop_iter_ranges(kernel)))

    def to_data(self) -> dict:
        try:
            data = json.loads(self.canonical_json)
            _require(type(data) is dict and set(data) == {
                "schema_version", "theorem_contract", "evidence", "annotation_preconditions_satisfiable",
            }, "top-level fields")
            _require(type(data["schema_version"]) is int and data["schema_version"] == 3,
                     "schema version")
            # This covers the typed annotation, not physical realization or
            # the interpretation of separately reported numeric obligations.
            _require(data["annotation_preconditions_satisfiable"] is True,
                     "annotation preconditions are not established satisfiable")
            _require(_canonical(data) == self.canonical_json, "noncanonical JSON")
            validate_relational_contract_data(
                data["theorem_contract"], family="unified", proof_kind="relational_dataflow",
            )
            evidence = _decode(data["evidence"], DataflowEvidence)
            _validate_evidence(evidence, data["theorem_contract"])
            return data
        except (TypeError, KeyError, AssertionError) as error:
            raise ValueError(f"invalid unified artifact: {error}") from error


def _build_verified_dataflow_contract(
    prepared: PreparedAnnotationProof, report: RelationalDataflowReport,
) -> VerifiedDataflowContract | None:
    """Freeze a supported proof after annotation SAT (not proof authentication)."""
    if (not report.proved or report.unsupported_reason is not None
            or report.annotation_satisfiability != "sat"
            or not report.alignments
            or any(a.theorem_vacuous or not a.relevant_statements for a in report.alignments)):
        return None
    theorem = relational_contract_data(
        source=prepared.source, kernel=prepared.kernel,
        declared_parameters=prepared.declared_parameters, annotation=prepared.annotation,
        constants=dict(prepared.constants), proof_kind="relational_dataflow",
    )
    _require(report.kernel_name == theorem["kernel"] and report.goal_name == theorem["goal_name"]
             and report.source_sha256 == theorem["source_sha256"]
             and report.constants == prepared.constants
             and report.theorem_sha256 == hashlib.sha256(_canonical(theorem["theorem"]).encode()).hexdigest(),
             "report does not belong to the prepared theorem")
    evidence = DataflowEvidence(
        kernel=prepared.kernel, checks=report.checks,
        required_tensor_reads=report.required_tensor_reads,
        used_assumptions=report.used_assumptions,
        external_obligations=report.external_obligations, alignments=report.alignments,
    )
    artifact = VerifiedDataflowContract(_canonical({
        "schema_version": 3, "theorem_contract": theorem, "evidence": _encode(evidence),
        "annotation_preconditions_satisfiable": True,
    }))
    artifact.to_data()
    return artifact
