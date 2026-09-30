"""Forward loop facts must be safe without caller-provided state metadata."""

import pytest
import z3

from ir import (
    Assign, BinOp, BoolLit, BoolType, FloatLit, FloatType, For, Full, If, IntLit, IntType, Let,
    MaskedStore, Range, Slice, TensorType, TensorView, Var, Where, Zeros,
)
from ir.positional import NeutralityAssumptions, PositionalAnalyzer, pred_true
from ir.reaching_definitions import assigned_variables
from ir.regions import GuardedRegion
from ir.relational_dataflow import _conditional_iteration_identities
from ir.identity_transition import prove_report_requirement_on_context


T = TensorType(FloatType(), [IntLit(1)])


def _write(name, value):
    return Assign(Var(name, type=T), None, value)


def _loop(body, count=2):
    return For(Var("i"), Range(IntLit(0), IntLit(count)), body)


def _analyzer():
    return PositionalAnalyzer(NeutralityAssumptions(), type_env={"v": T, "saved": T, "fixed": T})


def _zero():
    return Zeros([IntLit(1)], type=T)


def _unknown_zero(facts):
    assert facts.zero_where.body == BoolLit(False)


def test_zero_iterations_cannot_export_the_body_zero_fact():
    analyzer = _analyzer()
    analyzer.exec_stmt(_write("v", Full([IntLit(1)], FloatLit(1.0), type=T)))
    body = _write("v", _zero())
    analyzer.exec_stmt(_loop([body], count=0))
    # Concrete execution preserves one. The symbolic body does derive zero,
    # but that local fact must not escape the possibly empty loop.
    assert analyzer.trace().after(body)["v"].zero_where.body == BoolLit(True)
    _unknown_zero(analyzer.env["v"])
    assert "v" not in analyzer.elem_body


@pytest.mark.parametrize("nested", [False, True])
def test_arbitrary_iteration_cannot_reuse_first_iteration_fact(nested):
    analyzer = _analyzer()
    analyzer.exec_stmt(_write("v", _zero()))
    read = _write("saved", Var("v", type=T))
    overwrite = _write("v", Full([IntLit(1)], FloatLit(1.0), type=T))
    if nested:
        overwrite = For(Var("j"), Range(IntLit(0), IntLit(1)), [overwrite])
    analyzer.exec_stmt(_loop([read, overwrite]))
    # v is zero only in the first outer iteration, and one in the second.
    _unknown_zero(analyzer.trace().before(read)["v"])
    _unknown_zero(analyzer.trace().after(read)["saved"])
    _unknown_zero(analyzer.env["saved"])


def test_branch_writes_are_effects_but_unwritten_values_retain_facts():
    analyzer = _analyzer()
    for name in ("v", "fixed"):
        analyzer.exec_stmt(_write(name, _zero()))
    read = _write("saved", Var("v", type=T))
    branch = If(BoolLit(True), [_write("v", Full([IntLit(1)], FloatLit(1.0), type=T))], [])
    analyzer.exec_stmt(_loop([read, branch]))
    _unknown_zero(analyzer.trace().before(read)["v"])
    assert analyzer.env["fixed"].zero_where.body == BoolLit(True)


def test_effect_inventory_includes_partial_writes_lets_and_nested_indices():
    region = [Slice(IntLit(0), IntLit(1))]
    statements = [
        Let(Var("bound"), IntLit(1)),
        Assign(TensorView(Var("partial", type=T), region, type=T), None, _zero()),
        MaskedStore(base=Var("cache", type=T), region=region, mask=region, value=_zero()),
        For(Var("j"), Range(IntLit(0), IntLit(0)), [_write("nested", _zero())]),
    ]
    assert assigned_variables(statements) == {"bound", "partial", "cache", "j", "nested"}


def test_unknown_statement_and_missing_trace_exit_fail_closed():
    with pytest.raises(ValueError, match="unsupported definition effect"):
        assigned_variables([object()])
    with pytest.raises(ValueError, match="no exit"):
        _analyzer().trace().after(_write("v", _zero()))
    with pytest.raises(ValueError, match="unsupported forward fact"):
        _analyzer().exec_stmt(object())


@pytest.mark.parametrize("masked", [False, True])
def test_partial_write_cannot_preserve_whole_tensor_facts(masked):
    analyzer = _analyzer()
    analyzer.exec_stmt(_write("v", _zero()))
    region = [Slice(IntLit(0), IntLit(1))]
    value = Full([IntLit(1)], FloatLit(1.0), type=T)
    base = Var("v", type=T)
    write = (MaskedStore(base=base, region=region, mask=region, value=value) if masked
             else Assign(TensorView(base, region, type=T), None, value))
    analyzer.exec_stmt(write)
    _unknown_zero(analyzer.env["v"])
    assert "v" not in analyzer.elem_body


def test_conditional_identity_cannot_promote_initializer_to_loop_invariant():
    integer = TensorType(IntType(), [IntLit(1)])
    boolean = TensorType(BoolType(), [IntLit(1)])
    state, gate = Var("state", type=integer), Var("gate", type=boolean)
    zero = Full([IntLit(1)], IntLit(0), type=integer)
    one = Full([IntLit(1)], IntLit(1), type=integer)
    ordinal = Full([IntLit(1)], Var("k", type=IntType()), type=integer)
    condition = BinOp("or", BinOp("<", ordinal, one, type=boolean),
                      BinOp(">", state, zero, type=boolean), type=boolean)
    body = [Assign(gate, None, condition),
            Assign(state, None, Where(gate, BinOp("+", state, one, type=integer), state, type=integer))]
    loop = For(Var("k"), Range(IntLit(0), IntLit(2)), body)
    types = {"state": integer, "gate": boolean, "k": IntType()}
    analyzer = PositionalAnalyzer(NeutralityAssumptions(), type_env=types)
    analyzer.exec_stmt(Assign(state, None, zero))
    analyzer.exec_stmt(loop)
    demand = GuardedRegion([Slice(IntLit(0), IntLit(1))], pred_true(1))
    identity, = _conditional_iteration_identities(
        loop, frozenset({"state"}), {"state": demand}, {"state": demand},
        "state", demand.region, demand.region, types, analyzer.trace())
    # The selection really is identity when gate is false. But gate is true
    # in BOTH concrete iterations: k=0 makes state=1; k=1 makes state=2.
    concrete = 0
    for k in range(2):
        assert k < 1 or concrete > 0
        concrete += 1
    assert concrete == 2
    assert identity.left.proved
    check = prove_report_requirement_on_context(
        identity.left, identity.left.requirements[0], env={"k": z3.IntVal(1)},
        assumptions=[], context=True, check_name="later_iteration_not_neutral")
    assert not check.proved
