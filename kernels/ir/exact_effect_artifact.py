"""Source-bound exact-effect evidence, separate from relational certificates.

Validation rederives the complete typed proof plan without calling a solver.
The serialized formulas are an audit of that plan, not an alternative theorem
language: consumers lower the source-parsed typed annotation. Solver receipts
are trusted issuer evidence, not independently checkable or signed Z3 proofs.
"""

from dataclasses import dataclass, fields
import hashlib
import json

from . import annotations as ann
from .exact_effects import ExactEffectPlan, prepare_exact_effect_plan
from .relational_artifact import _encode
from .relational_contract import _constant_data, _type_data
from .smt import ProofCheck


# Closed inventory: new AST nodes/fields require an intentional export review.
_ANNOTATION_FIELDS = {
    ann.Before: "name", ann.After: "name", ann.DTypeOf: "name",
    ann.FreeVar: "name", ann.IntConst: "value", ann.AnnBinOp: "op lhs rhs",
    ann.AnnIndex: "base indices", ann.AnnSlice: "start stop",
    ann.RegionRef: "side slices", ann.RegionEquiv: "left right given",
    ann.AnnAnd: "args", ann.AnnOr: "args", ann.AnnNot: "body",
    ann.AnnImplies: "antecedent consequent", ann.AnnComparison: "op lhs rhs",
    ann.ForAllConstraint: "vars body",
    ann.RelationalProofGoal: "name pre_conditions post_conditions singletons same_vars",
}

# These are physical correspondence obligations, not conclusions of logical
# copy/coverage/framing analysis. The runtime adapter must retain them.
EXTERNAL_OBLIGATIONS = (
    "executed source and specialization correspond to the analyzed typed IR",
    "physical tensor shapes, strides, element dtypes and metadata match the logical launch state",
    "distinct tensor parameters have nonoverlapping physical storage",
)


def _require(condition, message):
    if not condition:
        raise ValueError(f"invalid exact-effect artifact: {message}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _annotation_data(value):
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) in (list, tuple):
        return [_annotation_data(item) for item in value]
    if type(value) is set:
        return sorted((_annotation_data(item) for item in value), key=_canonical)
    cls = type(value)
    _require(cls in _ANNOTATION_FIELDS, f"unsupported annotation node {cls.__name__}")
    names = _ANNOTATION_FIELDS[cls].split()
    _require({f.name for f in fields(cls)} == set(names), f"unreviewed fields on {cls.__name__}")
    return {"node": cls.__name__, **{name: _annotation_data(getattr(value, name)) for name in names}}


def _plan_document(plan: ExactEffectPlan):
    prepared = plan.prepared
    goal, = [g for g in prepared.goals if g.name == plan.goal_name]
    declared = {p.name: p for p in prepared.declared_parameters}
    # This signature deliberately matches the shared relational execution
    # identity. Goal names and proof kinds are not separate kernel executions.
    signature = {
        "kernel": prepared.kernel.name,
        "source_sha256": hashlib.sha256(prepared.source.encode()).hexdigest(),
        "constants": {name: _constant_data(value) for name, value in prepared.constants},
        "parameters": [{"name": p.name, "declared_type": _type_data(declared[p.name].type),
                        "type": _type_data(p.type)} for p in prepared.kernel.params],
        "analyzed_kernel": _encode(prepared.kernel),
    }
    return {
        "execution": signature,
        "execution_identity": _digest(signature),
        "theorem": _annotation_data(goal),
        "written_tensors": list(plan.written_tensors),
        "external_obligations": list(EXTERNAL_OBLIGATIONS),
        "obligations": [{
            "name": o.name,
            "kind": "satisfiable" if o.claim is None else "prove",
            "assumptions": [a.sexpr() for a in o.assumptions],
            "claim": None if o.claim is None else o.claim.sexpr(),
        } for o in plan.obligations],
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class VerifiedExactEffectContract:
    canonical_json: str

    def _validated(self):
        data = json.loads(self.canonical_json, object_pairs_hook=_unique_object)
        _require(type(data) is dict and set(data) == {
            "schema_version", "source", "kernel_name", "goal_name", "constants", "plan", "checks",
        }, "document fields")
        _require(type(data["schema_version"]) is int and data["schema_version"] == 1, "schema version")
        for name in ("source", "kernel_name", "goal_name"):
            _require(type(data[name]) is str and bool(data[name]), f"{name} must be nonempty text")
        constants = data["constants"]
        _require(type(constants) is dict and all(type(k) is str and type(v) is int
                                                for k, v in constants.items()), "integer specializations")
        plan = prepare_exact_effect_plan(data["source"], data["kernel_name"], constants, goal_name=data["goal_name"])
        expected = _plan_document(plan)
        _require(_canonical(data["plan"]) == _canonical(expected), "source-derived proof plan differs")
        checks = [{"name": o.name, "proved": True} for o in plan.obligations]
        _require(bool(checks) and _canonical(data["checks"]) == _canonical(checks), "incomplete or unsuccessful solver receipts")
        return data, plan

    def to_data(self):
        return self._validated()[0]

    def prepared_plan(self):
        """Validated typed input for consumers; never reconstruct meaning from SMT text."""
        return self._validated()[1]

    @property
    def digest(self):
        return _digest(self.to_data())

    @property
    def execution_identity(self):
        return self.to_data()["plan"]["execution_identity"]

    def written_tensor_parameters(self):
        return tuple(self.to_data()["plan"]["written_tensors"])

    @classmethod
    def from_data(cls, data):
        result = cls(_canonical(data))
        result.to_data()
        return result


def _issue_exact_effect_contract(plan: ExactEffectPlan, checks: list[ProofCheck]):
    """Trusted producer only; a mutable diagnostic report is not an input."""
    _require(bool(plan.obligations) and len(checks) == len(plan.obligations), "incomplete solver results")
    _require(all(type(c) is ProofCheck and c.name == o.name and c.proved is True
                 for c, o in zip(checks, plan.obligations)), "unsuccessful or mismatched solver results")
    data = {
        "schema_version": 1,
        "source": plan.prepared.source,
        "kernel_name": plan.prepared.kernel.name,
        "goal_name": plan.goal_name,
        "constants": dict(plan.prepared.constants),
        "plan": _plan_document(plan),
        "checks": [{"name": c.name, "proved": c.proved} for c in checks],
    }
    return VerifiedExactEffectContract.from_data(data)
