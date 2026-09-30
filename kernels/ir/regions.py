from dataclasses import dataclass, fields, is_dataclass
import networkx as nx
import z3

from . import *
from .preprocess import build_type_env


@dataclass(frozen=True)
class ConditionalRegion:
    """An output region guarded by path conditions and enclosing loops."""

    conditions: Sequence[Expr]
    region: Region
    iterators: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ConditionalWrite:
    conditions: Sequence[Expr]
    write: Assign | MaskedStore
    iterators: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RecurrenceGroup:
    """One cyclic write SCC and the tensor values carried through it."""

    variables: frozenset[str]
    writes: tuple[ConditionalWrite, ...]


@dataclass(frozen=True)
class WriteTraversalPlan:
    """Dependency-ordered writes plus explicit region-stable recurrences."""

    ordered_writes: tuple[ConditionalWrite, ...]
    recurrences: tuple[RecurrenceGroup, ...]


def collect_write_stmts(kernel: Kernel) -> list[ConditionalWrite]:
    """Collect all Assign statements with their path conditions.

    Returns a list of ``(conditions, assign)`` where *conditions* is the
    list of ``If`` condition conjuncts enclosing the assignment.
    """
    write_stmts: list[ConditionalWrite] = []

    def visit_stmt(
        stmt: Stmt,
        conditions: list[Expr],
        iterators: frozenset[str],
    ) -> None:
        match stmt:
            case Assign():
                write_stmts.append(ConditionalWrite(conditions, stmt, iterators))
            case For(var=var, body=body):
                for inner_stmt in body:
                    visit_stmt(inner_stmt, conditions, iterators | {var.name})
            case If(cond=cond, then_body=then_body, else_body=else_body):
                then_conds = conditions + [cond]
                for inner_stmt in then_body:
                    visit_stmt(inner_stmt, then_conds, iterators)
                else_conds = conditions + [Not(cond)]
                for inner_stmt in else_body:
                    visit_stmt(inner_stmt, else_conds, iterators)
            case Let():
                pass  # Let bindings are skipped
            case MaskedStore():
                write_stmts.append(ConditionalWrite(conditions, stmt, iterators))
            case _:
                raise AssertionError(f"unhandled stmt: {stmt}")

    for stmt in kernel.grid.body:
        visit_stmt(stmt, [], frozenset())

    return write_stmts


def get_read_tensors(expr: Expr, type_env: dict[str, Type]) -> set[str]:
    match expr:
        case Var(name=name):
            match type_env[name]:
                case TensorType():
                    return {name}
                case IntType() | FloatType() | BoolType():
                    return set()
                case _:
                    raise AssertionError(f"unhandled type: {type_env[name]}")
        case IntLit() | FloatLit() | BoolLit():
            return set()
        case BinOp(lhs=lhs, rhs=rhs) | Maximum(lhs=lhs, rhs=rhs):
            return get_read_tensors(lhs, type_env) | get_read_tensors(rhs, type_env)
        case Min(args=args) | Max(args=args):
            vars = set()
            for arg in args:
                vars |= get_read_tensors(arg, type_env)
            return vars
        case Zeros(shape=shape):
            vars = set()
            for dim in shape:
                vars |= get_read_tensors(dim, type_env)
            return vars
        case Full(shape=shape, value=value):
            vars = get_read_tensors(value, type_env)
            for dim in shape:
                vars |= get_read_tensors(dim, type_env)
            return vars
        case Arange(start=start, stop=stop):
            return get_read_tensors(start, type_env) | get_read_tensors(stop, type_env)
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            return (
                get_read_tensors(cond, type_env)
                | get_read_tensors(on_true, type_env)
                | get_read_tensors(on_false, type_env)
            )
        case ReduceMax(value=value) | ReduceSum(value=value):
            return get_read_tensors(value, type_env)
        case Exp2(value=value) | Sigmoid(value=value) | Rsqrt(value=value) | Log2(value=value) | Cast(value=value) | Not(value=value) | Unsqueeze(value=value) | Squeeze(value=value) | BroadcastTo(value=value) | Transpose(value=value):
            return get_read_tensors(value, type_env)
        case TensorView(base=base, region=region):
            vars = {base.name}
            for sl in region:
                vars |= get_read_tensors(sl.start, type_env)
                vars |= get_read_tensors(sl.stop, type_env)
            return vars
        case TensorIndex(base=base, indices=indices):
            vars = {base.name}
            for idx in indices:
                vars |= get_read_tensors(idx, type_env)
            return vars
        case MaskedLoad(base=base, region=region, mask=mask):
            vars = {base.name}
            for sl in region:
                vars |= get_read_tensors(sl.start, type_env)
                vars |= get_read_tensors(sl.stop, type_env)
            for sl in mask:
                vars |= get_read_tensors(sl.start, type_env)
                vars |= get_read_tensors(sl.stop, type_env)
            return vars
        case _:
            raise AssertionError(f"unhandled expr: {expr}")


def get_write_tensors(expr: Expr, type_env: dict[str, Type]) -> set[str]:
    match expr:
        case Var(name=name):
            match type_env[name]:
                case TensorType():
                    return {name}
                case _:
                    raise AssertionError(f"unhandled type: {type_env[name]}")
        case TensorView(base=base):
            return {base.name}
        case _:
            raise AssertionError(f"unhandled expr: {expr}")


def _assert_elementwise_for_cyclic_vars(
    expr: Expr, cyclic_vars: set[str], type_env: dict[str, Type]
) -> None:
    """Assert that all reads of cyclic_vars in expr are reachable only
    through element-wise operations.

    For element-wise ops, recurse into children. For non-element-wise ops
    (Reduce, Unsqueeze, Squeeze, BroadcastTo, Transpose, matmul @,
    TensorView, TensorIndex), assert that no cyclic variable appears in their subtree.
    """
    match expr:
        case Var():
            return  # leaf — no operation to check
        case IntLit() | FloatLit() | BoolLit() | Zeros() | Full() | Arange():
            return  # no tensor reads
        case BinOp(op=op, lhs=lhs, rhs=rhs):
            if op == "@":
                # matmul changes region — cyclic vars must not appear
                assert len(get_read_tensors(lhs, type_env) & cyclic_vars) == 0, (
                    f"cyclic vars in matmul lhs: {get_read_tensors(lhs, type_env) & cyclic_vars}"
                )
                assert len(get_read_tensors(rhs, type_env) & cyclic_vars) == 0, (
                    f"cyclic vars in matmul rhs: {get_read_tensors(rhs, type_env) & cyclic_vars}"
                )
            else:  # element-wise
                _assert_elementwise_for_cyclic_vars(lhs, cyclic_vars, type_env)
                _assert_elementwise_for_cyclic_vars(rhs, cyclic_vars, type_env)
        case Maximum(lhs=lhs, rhs=rhs):
            _assert_elementwise_for_cyclic_vars(lhs, cyclic_vars, type_env)
            _assert_elementwise_for_cyclic_vars(rhs, cyclic_vars, type_env)
        case Exp2(value=value) | Sigmoid(value=value) | Rsqrt(value=value) | Log2(value=value) | Cast(value=value) | Not(value=value):
            _assert_elementwise_for_cyclic_vars(value, cyclic_vars, type_env)
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            _assert_elementwise_for_cyclic_vars(cond, cyclic_vars, type_env)
            _assert_elementwise_for_cyclic_vars(on_true, cyclic_vars, type_env)
            _assert_elementwise_for_cyclic_vars(on_false, cyclic_vars, type_env)
        case Min(args=args) | Max(args=args):
            for arg in args:
                _assert_elementwise_for_cyclic_vars(arg, cyclic_vars, type_env)
        case ReduceMax(value=value) | ReduceSum(value=value):
            reads = get_read_tensors(value, type_env) & cyclic_vars
            assert len(reads) == 0, f"cyclic vars in reduce: {reads}"
        case Unsqueeze(value=value) | Squeeze(value=value):
            reads = get_read_tensors(value, type_env) & cyclic_vars
            assert len(reads) == 0, f"cyclic vars in unsqueeze/squeeze: {reads}"
        case BroadcastTo(value=value):
            reads = get_read_tensors(value, type_env) & cyclic_vars
            assert len(reads) == 0, f"cyclic vars in broadcast_to: {reads}"
        case Transpose(value=value):
            reads = get_read_tensors(value, type_env) & cyclic_vars
            assert len(reads) == 0, f"cyclic vars in transpose: {reads}"
        case TensorView(base=base):
            assert base.name not in cyclic_vars, (
                f"cyclic var {base.name} read via TensorView"
            )
        case TensorIndex(base=base):
            assert base.name not in cyclic_vars, (
                f"cyclic var {base.name} read via TensorIndex"
            )
        case MaskedLoad(base=base):
            assert base.name not in cyclic_vars, (
                f"cyclic var {base.name} read via MaskedLoad"
            )
        case _:
            raise AssertionError(f"unhandled expr: {expr}")


def plan_write_stmts(
    kernel: Kernel, type_env: dict[str, Type]
) -> WriteTraversalPlan:
    """Build the checked write traversal and expose its recurrence SCCs."""
    cond_write_stmts = collect_write_stmts(kernel)
    n = len(cond_write_stmts)
    wr_tensors: list[tuple[set[str], set[str]]] = []
    for cond_write in cond_write_stmts:
        match cond_write.write:
            case Assign(target=target, op=op, value=value):
                writes = get_write_tensors(target, type_env)
                reads = get_read_tensors(value, type_env)
                if op is not None:
                    # ``x op= rhs`` reads the pre-write value of ``x``.  The
                    # recurrence graph must contain that self/back edge; using
                    # only RHS reads misclassifies ordered accumulator folds as
                    # acyclic assignments.
                    reads |= writes
                wr_tensors.append(
                    (writes, reads)
                )
            case MaskedStore(base=base, value=value):
                wr_tensors.append(
                    (
                        {base.name},
                        get_read_tensors(value, type_env),
                    )
                )

    g = nx.DiGraph()
    # Isolated writes are still semantically relevant.  NetworkX does not
    # create a node until it appears in an edge, so a direct `out = input`
    # kernel would otherwise produce an empty backward traversal.
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            conflict = wr_tensors[i][0].intersection(wr_tensors[j][1])
            if len(conflict) > 0:
                _ = g.add_edge(i, j)

    assign_idx_list = []
    recurrences: list[RecurrenceGroup] = []

    dependency_graph = g
    g = nx.condensation(dependency_graph)
    scc_idx_list = list(nx.topological_sort(g))
    for idx in reversed(scc_idx_list):
        scc: set[int] = g.nodes[idx]["members"]
        # A one-node SCC with a self-edge is still a recurrence.  Exempting it
        # lets region-changing updates such as `acc = acc @ w` pass through a
        # single backward step even though repeated loop iterations require a
        # transitive closure, which is an under-approximation.
        cyclic = len(scc) > 1 or any(
            dependency_graph.has_edge(member, member) for member in scc
        )
        if cyclic:
            # Collect the set of tensor variables written by this SCC.
            cyclic_vars: set[str] = set()
            for member in scc:
                cyclic_vars |= wr_tensors[member][0]

            for member in scc:
                stmt = cond_write_stmts[member].write
                # (1) Target must be a plain Var, not TensorView.
                if isinstance(stmt, Assign):
                    assert isinstance(stmt.target, Var), (
                        f"cyclic SCC member {member}: target must be Var, got {stmt.target}"
                    )
                # (2) All operations on the path to cyclic var reads must
                # be element-wise, and cyclic vars must not appear inside
                # region-transforming operations, TensorView, or TensorIndex.
                _assert_elementwise_for_cyclic_vars(stmt.value, cyclic_vars, type_env)
            recurrences.append(
                RecurrenceGroup(
                    variables=frozenset(cyclic_vars),
                    writes=tuple(
                        cond_write_stmts[member] for member in sorted(scc)
                    ),
                )
            )
        # Backward analysis must traverse the writes of one source iteration
        # in reverse program order.  For an allowed element-wise recurrence,
        # this reaches the pre-iteration value at the same region; repeating
        # earlier iterations cannot expand that region.  Forward order can
        # silently miss dependencies introduced by a later assignment in the
        # SCC (for example scores_max <- next_max in online softmax).
        for member in sorted(scc, reverse=True):
            assign_idx_list.append(member)

    return WriteTraversalPlan(
        ordered_writes=tuple(cond_write_stmts[i] for i in assign_idx_list),
        recurrences=tuple(recurrences),
    )


def order_write_stmts(
    kernel: Kernel, type_env: dict[str, Type]
) -> list[ConditionalWrite]:
    """Return the checked dependency traversal without recurrence metadata."""

    return list(plan_write_stmts(kernel, type_env).ordered_writes)


def region_inter(r1: Region | None, r2: Region | None) -> Region | None:
    if r1 is None or r2 is None:
        return None
    assert len(r1) == len(r2)
    r = []
    for slice1, slice2 in zip(r1, r2):
        start = max_([slice1.start, slice2.start])
        stop = min_([slice1.stop, slice2.stop])
        r.append(Slice(start, stop))
    return r


def region_union(r1: Region | None, r2: Region | None) -> Region | None:
    if r1 is None:
        return r2
    if r2 is None:
        return r1
    assert len(r1) == len(r2)
    r = []
    for slice1, slice2 in zip(r1, r2):
        start = min_([slice1.start, slice2.start])
        stop = max_([slice1.stop, slice2.stop])
        r.append(Slice(start, stop))
    return r


def region_add(r: Region | None, offset: Expr) -> Region | None:
    if r is None:
        return None
    result = []
    for slice in r:
        result.append(Slice(add_(slice.start, offset), add_(slice.stop, offset)))
    return result


def region_sub(r: Region | None, offset: Expr) -> Region | None:
    if r is None:
        return None
    result = []
    for slice in r:
        result.append(Slice(sub_(slice.start, offset), sub_(slice.stop, offset)))
    return result


def bound_variable_regions(
    kernel: Kernel,
    output_tensor: str,
    output_region: Region,
) -> tuple[dict[str, Region], list[ConditionalRegion]]:
    type_env = build_type_env(kernel)

    internal_regions: dict[str, Region | None] = {output_tensor: output_region}

    output_cond_regions: list[ConditionalRegion] = []
    for cond_write in order_write_stmts(kernel, type_env):
        o_region, is_output = regions_write(
            cond_write.write, type_env, internal_regions, output_tensor
        )
        if is_output and o_region is not None:
            output_cond_regions.append(
                ConditionalRegion(
                    cond_write.conditions,
                    o_region,
                    cond_write.iterators,
                )
            )

    regions: dict[str, Region] = {}
    for var, typ in type_env.items():
        match typ:
            case TensorType(dims=dims):
                if var not in internal_regions:
                    continue
                region = region_inter(
                    internal_regions[var], [Slice(IntLit(0), dim) for dim in dims]
                )
                if region is not None:
                    regions[var] = region
            case IntType() | FloatType() | BoolType():
                pass
            case _:
                raise AssertionError(f"unhandled type: {typ}")

    return (regions, output_cond_regions)


def regions_expr(
    expr: Expr,
    target_region: Region | None,
    type_env: dict[str, Type],
    regions: dict[str, Region | None],
):
    if target_region is None:
        return

    def _child_region(child: Expr) -> Region:
        """Return target_region for tensor-typed children, [] for scalars.

        Element-wise ops propagate the target region to tensor operands
        and use an empty region for scalar operands (scalars broadcast
        implicitly to the tensor shape).
        """
        assert child.type is not None
        return target_region if isinstance(child.type, TensorType) else []

    match expr:
        case Var(name=name):
            typ = type_env[name]
            match typ:
                case TensorType():
                    regions[name] = region_union(
                        regions.get(name), target_region
                    )
                case IntType() | FloatType() | BoolType():
                    pass
                case _:
                    raise AssertionError(f"unhandled type: {typ}")
        case TensorView(base=base, region=region):
            src_region = []
            for dst, src in zip(target_region, region):
                src_region.append(
                    Slice(add_(src.start, dst.start), add_(src.start, dst.stop))
                )
            regions[base.name] = region_union(regions.get(base.name), src_region)
            for src in region:
                regions_expr(src.start, [], type_env, regions)
                regions_expr(src.stop, [], type_env, regions)
        case TensorIndex(base=base, indices=indices):
            src_region = [Slice(idx, add_(idx, IntLit(1))) for idx in indices]
            regions[base.name] = region_union(regions.get(base.name), src_region)
            for idx in indices:
                regions_expr(idx, [], type_env, regions)
        case BinOp(op=op, lhs=lhs, rhs=rhs):
            if op == "@":
                lhs_typ = lhs.type
                rhs_typ = rhs.type
                assert isinstance(lhs_typ, TensorType) and isinstance(
                    rhs_typ, TensorType
                )
                regions_expr(
                    rhs,
                    [Slice(IntLit(0), rhs_typ.dims[0]), target_region[1]],
                    type_env,
                    regions,
                )
                regions_expr(
                    lhs,
                    [target_region[0], Slice(IntLit(0), lhs_typ.dims[1])],
                    type_env,
                    regions,
                )
            else:
                # Element-wise: propagate region to tensor operands,
                # empty region to scalar operands.
                regions_expr(lhs, _child_region(lhs), type_env, regions)
                regions_expr(rhs, _child_region(rhs), type_env, regions)
        case Min(args=args) | Max(args=args):
            for arg in args:
                regions_expr(arg, target_region, type_env, regions)
        case Zeros() | IntLit() | Full() | Arange() | FloatLit() | BoolLit():
            return
        case Exp2(value=value) | Sigmoid(value=value) | Rsqrt(value=value) | Log2(value=value) | Cast(value=value) | Not(value=value):
            regions_expr(value, target_region, type_env, regions)
        case Maximum(lhs=lhs, rhs=rhs):
            regions_expr(lhs, _child_region(lhs), type_env, regions)
            regions_expr(rhs, _child_region(rhs), type_env, regions)
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            regions_expr(cond, _child_region(cond), type_env, regions)
            regions_expr(on_true, _child_region(on_true), type_env, regions)
            regions_expr(on_false, _child_region(on_false), type_env, regions)
        case ReduceMax(value=value, axis=axis) | ReduceSum(value=value, axis=axis):
            assert isinstance(value.type, TensorType)
            full_axis_slice = Slice(IntLit(0), value.type.dims[axis])
            expanded = list(target_region)
            expanded.insert(axis, full_axis_slice)
            regions_expr(value, expanded, type_env, regions)
        case Unsqueeze(value=value, axis=axis):
            contracted = list(target_region)
            del contracted[axis]
            regions_expr(value, contracted, type_env, regions)
        case Squeeze(value=value, axis=axis):
            expanded = list(target_region)
            expanded.insert(axis, Slice(IntLit(0), IntLit(1)))
            regions_expr(value, expanded, type_env, regions)
        case BroadcastTo(value=value):
            assert isinstance(value.type, TensorType)
            src_region = []
            for sl, src_dim in zip(target_region, value.type.dims):
                if src_dim == IntLit(1):
                    src_region.append(Slice(IntLit(0), IntLit(1)))
                else:
                    src_region.append(sl)
            regions_expr(value, src_region, type_env, regions)
        case Transpose(value=value, permutation=permutation):
            inv_perm = [0] * len(permutation)
            for i, p in enumerate(permutation):
                inv_perm[p] = i
            src_region = [
                target_region[inv_perm[i]] for i in range(len(permutation))
            ]
            regions_expr(value, src_region, type_env, regions)
        case MaskedLoad(base=base, region=region, mask=mask):
            src_region = []
            for dst, src in zip(target_region, region):
                src_region.append(
                    Slice(add_(src.start, dst.start), add_(src.start, dst.stop))
                )
            src_region = region_inter(src_region, mask)
            regions[base.name] = region_union(regions.get(base.name), src_region)
            for src in region:
                regions_expr(src.start, [], type_env, regions)
                regions_expr(src.stop, [], type_env, regions)
            for src in mask:
                regions_expr(src.start, [], type_env, regions)
                regions_expr(src.stop, [], type_env, regions)
        case _:
            raise AssertionError(f"unhandled expr: {expr}")


def regions_write(
    write: Assign | MaskedStore,
    type_env: dict[str, Type],
    regions: dict[str, Region | None],
    output_var: str,
) -> tuple[Region | None, bool]:
    output_region = (None, False)
    target_region = None
    match write:
        case Assign(target=target):
            match target:
                case Var(name=name):
                    match type_env[name]:
                        case TensorType():
                            target_region = regions.get(name)
                            if name == output_var:
                                output_region = (target_region, True)
                        case IntType() | FloatType() | BoolType():
                            target_region = []
                        case _:
                            raise AssertionError(
                                f"unhandled target type: {type_env[name]}"
                            )
                case TensorView(base=base, region=region):
                    target_base_region = region_inter(regions.get(base.name), region)
                    if target_base_region is not None:
                        target_region = []
                        for r, sl in zip(target_base_region, region):
                            target_region.append(
                                Slice(sub_(r.start, sl.start), sub_(r.stop, sl.start))
                            )
                    if base.name == output_var:
                        output_region = (target_base_region, True)
        case MaskedStore(base=base, region=region, mask=mask):
            target_base_region = region_inter(regions.get(base.name), region)
            target_base_region = region_inter(target_base_region, mask)
            if target_base_region is not None:
                target_region = []
                for r, sl in zip(target_base_region, region):
                    target_region.append(
                        Slice(sub_(r.start, sl.start), sub_(r.stop, sl.start))
                    )
            if base.name == output_var:
                output_region = (target_base_region, True)
    regions_expr(write.value, target_region, type_env, regions)
    return output_region


# ===========================================================================
# Mask-aware backward narrowing
# ===========================================================================
#
# The functions below extend the backward region pass with a per-position
# *guard* predicate. The standard pass produces, for each input variable T,
# a bounding hyper-rectangle `region` of positions read. The mask-aware
# pass additionally produces a Boolean predicate `guard` for value-relevant
# dependencies.  A false guard means the value at that position cannot affect
# the selected output under the stated algebraic assumptions; it does NOT mean
# the generated GPU program avoids the physical memory load.  Triton may still
# evaluate both `where` branches or load values later multiplied by zero.
#
# The guard is composed from the FORWARD per-position facts in
# `ir.positional` (`zero_where`, `true_where`, etc.). Specifically, this
# pass narrows reads at three sites:
#
#   * `Where(c, t, f)` consumed under guard G:
#       on_true  contributes under  G ∧ c.true_where
#       on_false contributes under  G ∧ c.false_where
#   * `a * b` consumed under G:
#       finite(a) permits a under G ∧ ¬b.zero_where
#       finite(b) permits b under G ∧ ¬a.zero_where
#   * `a @ b` consumed under G with the m-axis target restricted to a
#     single row index `i` and finite(b): b contributes only under
#     positions where `¬a.zero_where[i, _i0]`.
#
# Other operations propagate the guard unchanged (with appropriate
# index remapping for shape ops). When we cannot narrow, the guard
# remains TRUE — which is sound but gives no extra information.
#
# COORDINATES: each variable's guard is a `Pred` of rank `r` over the
# canonical free indices `_i0..._i{r-1}`, where `_ik` is the *absolute*
# index along T's k-th axis (NOT an offset within the bounding region).
# This makes guards composable across multiple write paths: contributions
# from different Assign statements share the same coordinate frame and
# can be OR'd directly.

from .positional import (
    FREE_IDX_PREFIX,
    ForwardFactTrace,
    NeutralityAssumptions,
    PositionalAnalyzer,
    Pred,
    TensorFacts,
    facts_unknown,
    free_idx,
    pred_and,
    pred_false,
    pred_not,
    pred_or,
    pred_true,
    subst_free_indices,
)


@dataclass(frozen=True)
class GuardedRegion:
    """A bounding region paired with a guard predicate restricting which
    positions of the variable are actually read.

    Soundness contract: positions whose values can affect the selected output
    are a SUBSET of `{x in region : guard(x)}`, under every explicitly supplied
    IEEE/algebraic assumption. This is a dependency set, not a physical memory
    access set. The guard is over the variable's canonical free indices
    `_i0..._i{rank-1}` interpreted as absolute indices along the corresponding
    axes.
    """

    region: Region
    guard: Pred

    def __post_init__(self) -> None:
        assert self.guard.rank == len(self.region), (
            "guard rank must match the guarded tensor region: "
            f"{self.guard.rank} != {len(self.region)}"
        )


def guarded_union(a: GuardedRegion | None, b: GuardedRegion | None) -> GuardedRegion | None:
    """Combine two contributing paths to the same variable.

    Bounding region: union (existing semantics). Guard: OR of the two
    guards. NB: this is an over-approximation of the precise semantics
    `{x in r_a : g_a(x)} ∪ {x in r_b : g_b(x)}` — sound (every actually-
    read position is captured) but loose when r_a and r_b differ.
    """
    if a is None:
        return b
    if b is None:
        return a
    region = region_union(a.region, b.region)
    if region is None:
        return None
    assert a.guard.rank == b.guard.rank
    return GuardedRegion(region=region, guard=pred_or(a.guard, b.guard))


def _facts_for(name: str, pos_env: dict[str, TensorFacts], rank: int) -> TensorFacts:
    """Look up forward positional facts for a variable; default to
    `facts_unknown(rank)` if absent (e.g., input parameters not present
    in `pos_env`)."""
    return pos_env.get(name, facts_unknown(rank))


def _facts_for_expr(
    expr: Expr,
    type_env: dict[str, Type],
    pos_env: dict[str, TensorFacts],
    finite_vars: frozenset[str],
) -> TensorFacts:
    """Evaluate positional facts for an inline compound-assignment RHS."""
    rank = len(expr.type.dims) if isinstance(expr.type, TensorType) else 0
    analyzer = PositionalAnalyzer(
        assumptions=NeutralityAssumptions(finite_vars=finite_vars),
        type_env=type_env,
        env=dict(pos_env),
    )
    facts, _ = analyzer._eval(expr, rank)
    return facts


def _local_guard_to_source_coords(target_guard: Pred, region: Region) -> Pred:
    """Map a view/load-local guard to absolute source tensor coordinates."""
    assert target_guard.rank == len(region)
    mapping = {
        f"{FREE_IDX_PREFIX}{k}": sub_(free_idx(k), sl.start)
        for k, sl in enumerate(region)
    }
    return Pred(
        len(region), subst_free_indices(target_guard.body, mapping)
    )


def _source_guard_to_local_coords(source_guard: Pred, region: Region) -> Pred:
    """Map an absolute destination/store guard to value-local coordinates."""
    assert source_guard.rank == len(region)
    mapping = {
        f"{FREE_IDX_PREFIX}{k}": add_(sl.start, free_idx(k))
        for k, sl in enumerate(region)
    }
    return Pred(
        len(region), subst_free_indices(source_guard.body, mapping)
    )


def _references_any_free_index(expr: Expr, indices: set[int]) -> bool:
    """Whether ``expr`` mentions a canonical index in ``indices``."""
    wanted = {f"{FREE_IDX_PREFIX}{k}" for k in indices}

    def visit(value: object) -> bool:
        if isinstance(value, Var):
            return value.name in wanted
        if not is_dataclass(value):
            return False
        for field in fields(value):
            if field.name == "type":
                continue
            child = getattr(value, field.name)
            if isinstance(child, (list, tuple)):
                if any(visit(item) for item in child):
                    return True
            elif visit(child):
                return True
        return False

    return visit(expr)


def regions_expr_masked(
    expr: Expr,
    target_region: Region | None,
    target_guard: Pred,
    type_env: dict[str, Type],
    regions: dict[str, GuardedRegion | None],
    pos_env: dict[str, TensorFacts],
    finite_vars: frozenset[str] = frozenset(),
) -> None:
    """Mask-aware variant of `regions_expr`.

    `target_guard` is a predicate over the consumer's canonical free
    indices (`_i0..._i{rank-1}`); only positions of `expr` where
    `target_guard` is true are value-relevant to the selected output.
    """
    if target_region is None:
        return

    def _child_region(child: Expr) -> Region:
        assert child.type is not None
        return target_region if isinstance(child.type, TensorType) else []

    def _child_guard(child: Expr) -> Pred:
        # Same rank as parent for tensor children; for scalar children the
        # guard is rank-0 (we conservatively use rank-0 TRUE, which means
        # "always read").
        assert child.type is not None
        if isinstance(child.type, TensorType):
            return target_guard
        return pred_true(0)

    match expr:
        case Var(name=name):
            typ = type_env[name]
            match typ:
                case TensorType():
                    new = GuardedRegion(target_region, target_guard)
                    regions[name] = guarded_union(regions.get(name), new)
                case IntType() | FloatType() | BoolType():
                    pass
                case _:
                    raise AssertionError(f"unhandled type: {typ}")
        case TensorView(base=base, region=region):
            src_region = []
            for dst, src in zip(target_region, region):
                src_region.append(
                    Slice(add_(src.start, dst.start), add_(src.start, dst.stop))
                )
            source_guard = _local_guard_to_source_coords(target_guard, region)
            new = GuardedRegion(src_region, source_guard)
            regions[base.name] = guarded_union(regions.get(base.name), new)
            for src in region:
                regions_expr_masked(
                    src.start, [], pred_true(0), type_env, regions, pos_env, finite_vars
                )
                regions_expr_masked(
                    src.stop, [], pred_true(0), type_env, regions, pos_env, finite_vars
                )
        case TensorIndex(base=base, indices=indices):
            src_region = [Slice(idx, add_(idx, IntLit(1))) for idx in indices]
            # TensorIndex is scalar-valued. Lift its rank-0 consumer guard to
            # the indexed source tensor's coordinate space.
            assert target_guard.rank == 0
            new = GuardedRegion(
                src_region, Pred(len(indices), target_guard.body)
            )
            regions[base.name] = guarded_union(regions.get(base.name), new)
            for idx in indices:
                regions_expr_masked(
                    idx, [], pred_true(0), type_env, regions, pos_env, finite_vars
                )
        case BinOp(op=op, lhs=lhs, rhs=rhs):
            if op == "@":
                _matmul_masked_propagate(
                    lhs, rhs, target_region, target_guard, type_env, regions, pos_env,
                    finite_vars,
                )
            else:
                # In particular, finite(x) does not erase x from x * 0:
                # the sign of zero still depends on x. Numerical zero facts
                # alone cannot justify bitwise dependency elimination.
                regions_expr_masked(
                    lhs, _child_region(lhs), _child_guard(lhs), type_env, regions, pos_env,
                    finite_vars,
                )
                regions_expr_masked(
                    rhs, _child_region(rhs), _child_guard(rhs), type_env, regions, pos_env,
                    finite_vars,
                )
        case Min(args=args) | Max(args=args):
            for arg in args:
                regions_expr_masked(
                    arg, target_region, target_guard, type_env, regions, pos_env,
                    finite_vars,
                )
        case Zeros() | IntLit() | Full() | Arange() | FloatLit() | BoolLit():
            return
        case Exp2(value=value) | Sigmoid(value=value) | Rsqrt(value=value) | Log2(value=value) | Cast(value=value) | Not(value=value):
            regions_expr_masked(
                value, target_region, target_guard, type_env, regions, pos_env,
                finite_vars,
            )
        case Maximum(lhs=lhs, rhs=rhs):
            regions_expr_masked(
                lhs, _child_region(lhs), _child_guard(lhs), type_env, regions, pos_env,
                finite_vars,
            )
            regions_expr_masked(
                rhs, _child_region(rhs), _child_guard(rhs), type_env, regions, pos_env,
                finite_vars,
            )
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            # Where(c, t, f): t is read iff c is true; f is read iff c
            # is false. We have LOWER BOUNDS on c's true/false positions
            # (`c.true_where`, `c.false_where`). To get an UPPER BOUND on
            # t's reads we need positions where c is provably NOT-false,
            # i.e. ¬c.false_where. Symmetrically for f: ¬c.true_where.
            #
            # Why not narrow by `c.true_where` directly? Because that's a
            # lower bound — it tells us c IS true at those positions, but
            # c could also be true elsewhere (where we just have no
            # proof). Narrowing by a lower bound would drop legal reads.
            cond_var = _underlying_var_name(cond)
            cond_rank = len(cond.type.dims) if isinstance(cond.type, TensorType) else 0
            if cond_var is not None:
                cond_facts = _facts_for_expr(
                    cond, type_env, pos_env, finite_vars
                )
                t_extra = pred_not(cond_facts.false_where)
                f_extra = pred_not(cond_facts.true_where)
            else:
                t_extra = pred_true(cond_rank)
                f_extra = pred_true(cond_rank)
            t_guard = pred_and(target_guard, t_extra) if isinstance(on_true.type, TensorType) else target_guard
            f_guard = pred_and(target_guard, f_extra) if isinstance(on_false.type, TensorType) else target_guard
            regions_expr_masked(
                cond, _child_region(cond), _child_guard(cond), type_env, regions, pos_env,
                finite_vars,
            )
            regions_expr_masked(
                on_true, _child_region(on_true), t_guard, type_env, regions, pos_env,
                finite_vars,
            )
            regions_expr_masked(
                on_false, _child_region(on_false), f_guard, type_env, regions, pos_env,
                finite_vars,
            )
        case ReduceMax(value=value, axis=axis) | ReduceSum(value=value, axis=axis):
            assert isinstance(value.type, TensorType)
            full_axis_slice = Slice(IntLit(0), value.type.dims[axis])
            expanded = list(target_region)
            expanded.insert(axis, full_axis_slice)
            # Remap the parent guard from rank r → rank r+1: a child index
            # `_ik` maps to the parent index that was at the same axis
            # position before insertion, i.e. `_ik` for k<axis stays;
            # `_ik` for k>=axis maps to `_i{k-1}` in the PARENT (since
            # the parent's k corresponds to the child's k+1 once the
            # reduction axis is added).
            # We are going parent→child, so the substitution is from
            # parent's _ik into child indices.
            mapping = _shift_indices_for_reduce_unfold(target_guard.rank, axis)
            child_rank = target_guard.rank + 1
            child_guard_body = subst_free_indices(target_guard.body, mapping)
            child_guard = Pred(child_rank, child_guard_body)
            # -inf has one bit pattern and is a max identity. Numerical zero
            # facts do not determine the sign of a sum's input lanes; retain
            # those dependencies rather than assuming a +0-initialized fold.
            value_var = _underlying_var_name(value)
            if isinstance(expr, ReduceMax) and value_var is not None:
                value_facts = _facts_for_expr(
                    value, type_env, pos_env, finite_vars
                )
                child_guard = pred_and(child_guard, pred_not(value_facts.neg_inf_where))
            regions_expr_masked(
                value, expanded, child_guard, type_env, regions, pos_env, finite_vars
            )
        case Unsqueeze(value=value, axis=axis):
            contracted = list(target_region)
            del contracted[axis]
            # Parent rank r, child rank r-1. The parent's `_ik` for
            # k<axis is the child's `_ik`; the parent's `_iaxis` is the
            # constant 0 (size-1 axis); for k>axis maps to `_i{k-1}`.
            mapping = {f"{FREE_IDX_PREFIX}{axis}": IntLit(0)}
            for k in range(target_guard.rank):
                if k == axis:
                    continue
                child_k = k if k < axis else k - 1
                mapping[f"{FREE_IDX_PREFIX}{k}"] = free_idx(child_k)
            child_guard = Pred(
                target_guard.rank - 1,
                subst_free_indices(target_guard.body, mapping),
            )
            regions_expr_masked(
                value, contracted, child_guard, type_env, regions, pos_env, finite_vars
            )
        case Squeeze(value=value, axis=axis):
            expanded = list(target_region)
            expanded.insert(axis, Slice(IntLit(0), IntLit(1)))
            mapping = {}
            for k in range(target_guard.rank):
                child_k = k if k < axis else k + 1
                mapping[f"{FREE_IDX_PREFIX}{k}"] = free_idx(child_k)
            child_guard = Pred(
                target_guard.rank + 1,
                subst_free_indices(target_guard.body, mapping),
            )
            regions_expr_masked(
                value, expanded, child_guard, type_env, regions, pos_env, finite_vars
            )
        case BroadcastTo(value=value):
            assert isinstance(value.type, TensorType)
            src_region = []
            for sl, src_dim in zip(target_region, value.type.dims):
                if src_dim == IntLit(1):
                    src_region.append(Slice(IntLit(0), IntLit(1)))
                else:
                    src_region.append(sl)
            # Backward through a broadcast requires existentially eliminating
            # every expanded output axis: the one source position is relevant
            # if *any* selected output position on that axis is relevant.
            # Predicate IR has no quantifiers, so retaining a guard that
            # depends on such an axis would be an under-approximation.  Keep
            # guards independent of expanded axes and otherwise fall back to
            # TRUE.  (Forward positional facts use the different, valid
            # substitution output_index -> source_index=0.)
            expanded_axes = {
                k for k, src_dim in enumerate(value.type.dims)
                if src_dim == IntLit(1)
            }
            child_guard = (
                pred_true(target_guard.rank)
                if _references_any_free_index(target_guard.body, expanded_axes)
                else target_guard
            )
            regions_expr_masked(
                value, src_region, child_guard, type_env, regions, pos_env, finite_vars
            )
        case Transpose(value=value, permutation=permutation):
            inv_perm = [0] * len(permutation)
            for i, p in enumerate(permutation):
                inv_perm[p] = i
            src_region = [target_region[inv_perm[i]] for i in range(len(permutation))]
            # Parent _ik came from child _i{perm[k]}. To express parent's
            # body in child indices: parent's `_ik` → child's `_i{perm[k]}`.
            mapping = {
                f"{FREE_IDX_PREFIX}{k}": free_idx(permutation[k])
                for k in range(len(permutation))
            }
            child_guard = Pred(
                target_guard.rank, subst_free_indices(target_guard.body, mapping)
            )
            regions_expr_masked(
                value, src_region, child_guard, type_env, regions, pos_env, finite_vars
            )
        case MaskedLoad(base=base, region=region, mask=mask):
            src_region = []
            for dst, src in zip(target_region, region):
                src_region.append(
                    Slice(add_(src.start, dst.start), add_(src.start, dst.stop))
                )
            src_region = region_inter(src_region, mask)
            source_guard = _local_guard_to_source_coords(target_guard, region)
            new = (
                GuardedRegion(src_region, source_guard)
                if src_region is not None
                else None
            )
            if new is not None:
                regions[base.name] = guarded_union(regions.get(base.name), new)
            for src in region:
                regions_expr_masked(
                    src.start, [], pred_true(0), type_env, regions, pos_env, finite_vars
                )
                regions_expr_masked(
                    src.stop, [], pred_true(0), type_env, regions, pos_env, finite_vars
                )
            for src in mask:
                regions_expr_masked(
                    src.start, [], pred_true(0), type_env, regions, pos_env, finite_vars
                )
                regions_expr_masked(
                    src.stop, [], pred_true(0), type_env, regions, pos_env, finite_vars
                )
        case _:
            raise AssertionError(f"unhandled expr: {expr}")


def _underlying_var_name(expr: Expr) -> str | None:
    """Peel off shape-only wrappers to find the underlying Var name, used
    to look up forward positional facts. Returns None for non-Var cores."""
    match expr:
        case Var(name=name):
            return name
        case Unsqueeze(value=v) | Squeeze(value=v) | BroadcastTo(value=v) | Transpose(value=v):
            return _underlying_var_name(v)
        case _:
            return None


def _shift_indices_for_reduce_unfold(parent_rank: int, axis: int) -> dict[str, Expr]:
    """Build a substitution mapping parent's `_ik` to child indices when
    the child has one extra axis at position `axis` (a reduction's input
    relative to its output). Parent `_ik` for k<axis maps to child `_ik`;
    parent `_ik` for k>=axis maps to child `_i{k+1}` (skipping the
    reduction axis)."""
    mapping: dict[str, Expr] = {}
    for k in range(parent_rank):
        child_k = k if k < axis else k + 1
        mapping[f"{FREE_IDX_PREFIX}{k}"] = free_idx(child_k)
    return mapping


def _matmul_masked_propagate(
    lhs: Expr,
    rhs: Expr,
    target_region: Region,
    target_guard: Pred,
    type_env: dict[str, Type],
    regions: dict[str, GuardedRegion | None],
    pos_env: dict[str, TensorFacts],
    finite_vars: frozenset[str],
) -> None:
    """Backward narrowing for `lhs @ rhs` (rank-2 matmul).

    `target_region` is `[r_m, r_n]`; `target_guard` is over `_i0` (m-axis)
    and `_i1` (n-axis). Standard pass would assign:
        rhs ← [Slice(0, K), r_n]
        lhs ← [r_m, Slice(0, K)]

    Mask-aware narrowing fires only when `r_m` is a singleton (a single
    output row). Otherwise we fall back to TRUE guards (the standard
    over-approximation), since narrowing across multiple output rows
    needs an OR over the rows that we don't compute here.
    """
    lhs_typ = lhs.type
    rhs_typ = rhs.type
    assert isinstance(lhs_typ, TensorType) and isinstance(rhs_typ, TensorType)
    k_dim = lhs_typ.dims[1]
    r_m, r_n = target_region

    # rhs's region: full k axis × r_n. rhs's coords: `_i0` (k), `_i1` (n).
    rhs_region: Region = [Slice(IntLit(0), k_dim), r_n]
    # rhs guard from parent: parent `_i1` (n-axis) maps to rhs `_i1`.
    # Parent `_i0` (m-axis) is FIXED; if r_m is a singleton, substitute
    # the literal row index. Otherwise drop narrowing (set TRUE).
    rhs_guard: Pred = pred_true(2)
    fixed_row = _singleton_index(r_m)
    if fixed_row is not None:
        # Try to narrow rhs by lhs.zero_where[fixed_row, _i0] (rhs's _i0
        # corresponds to lhs's _i1 which is the k-axis).
        rhs_var = _underlying_var_name(rhs)
        # The lhs may be an inline expression such as ``p.to(v.dtype)``.
        # Evaluate its positional facts directly; requiring a shape-wrapper
        # Var core would unnecessarily discard the cast's proved zero mask.
        if rhs_var is not None and rhs_var in finite_vars:
            lhs_facts = _facts_for_expr(
                lhs, type_env, pos_env, finite_vars
            )
            # Substitute lhs's `_i0` ← fixed_row, `_i1` ← rhs's `_i0`.
            mapping = {
                f"{FREE_IDX_PREFIX}0": fixed_row,
                f"{FREE_IDX_PREFIX}1": free_idx(0),
            }
            zero_expr = subst_free_indices(lhs_facts.zero_where.body, mapping)
            rhs_guard_extra = pred_not(Pred(2, zero_expr))
            # Combine with the parent guard mapped to rhs's coords. Parent
            # `_i0` (m) → fixed_row; parent `_i1` (n) → rhs `_i1`.
            parent_mapping = {
                f"{FREE_IDX_PREFIX}0": fixed_row,
                f"{FREE_IDX_PREFIX}1": free_idx(1),
            }
            parent_in_rhs = Pred(
                2, subst_free_indices(target_guard.body, parent_mapping)
            )
            rhs_guard = pred_and(parent_in_rhs, rhs_guard_extra)
        else:
            parent_mapping = {
                f"{FREE_IDX_PREFIX}0": fixed_row,
                f"{FREE_IDX_PREFIX}1": free_idx(1),
            }
            rhs_guard = Pred(2, subst_free_indices(target_guard.body, parent_mapping))

    regions_expr_masked(
        rhs, rhs_region, rhs_guard, type_env, regions, pos_env, finite_vars
    )

    # lhs's region: r_m × full k axis. lhs's coords: `_i0` (m), `_i1` (k).
    lhs_region: Region = [r_m, Slice(IntLit(0), k_dim)]
    # We don't currently narrow lhs by rhs's facts (would need a "for some
    # j in r_n" disjunction). Pass guard unchanged but reindex parent's
    # `_i1` (n-axis) — it doesn't appear in lhs, so we use TRUE for now.
    # Parent `_i0` (m) corresponds to lhs's `_i0`.
    if fixed_row is not None:
        # Fixed row consumer: parent guard is reduced to a function of
        # `_i1` only, which is unrelated to lhs's coords (lhs has m, k).
        # We conservatively use TRUE.
        lhs_guard = pred_true(2)
    else:
        # Parent `_i0` → lhs `_i0`; parent `_i1` → unconstrained (TRUE).
        # Conservative: TRUE.
        lhs_guard = pred_true(2)
    regions_expr_masked(
        lhs, lhs_region, lhs_guard, type_env, regions, pos_env, finite_vars
    )


def _singleton_index(s: Slice) -> Expr | None:
    """Return the sole integer index of a slice whenever this is proved.

    Region intersections turn a syntactic singleton ``[i, i+1)`` into
    expressions such as ``[max(i, lo)-base, min(i+1, hi)-base)``. The slice
    may be empty for some loop iterations, but whenever it is non-empty it is
    still a singleton. After the cheap syntactic case, ask Z3 to prove

        start < stop  ==>  stop = start + 1.

    Returning ``start`` is then sound: on non-empty executions it is the sole
    row, and on empty executions there is no dependency to narrow.
    """
    # Recognize the common construction `Slice(i, add_(i, IntLit(1)))`.
    if s.start == s.stop:
        return None
    # Match `stop == start + 1`.
    if isinstance(s.stop, BinOp) and s.stop.op == "+":
        if s.stop.lhs == s.start and s.stop.rhs == IntLit(1):
            return s.start
        if s.stop.rhs == s.start and s.stop.lhs == IntLit(1):
            return s.start

    env: dict[str, object] = {}
    _collect_z3_symbols(s.start, env)
    _collect_z3_symbols(s.stop, env)
    try:
        # Lazy import avoids the module-import cycle: verif imports regions.
        from .smt import expr_to_z3

        start = expr_to_z3(s.start, env)  # type: ignore[arg-type]
        stop = expr_to_z3(s.stop, env)  # type: ignore[arg-type]
        solver = z3.Solver()
        solver.set("timeout", 1000)
        solver.add(
            z3.Not(
                z3.Implies(
                    start < stop,
                    stop == start + 1,
                )
            )
        )
        if solver.check() == z3.unsat:
            return s.start
    except (AssertionError, KeyError, TypeError, z3.Z3Exception):
        # Unsupported expressions fail closed: no singleton narrowing.
        return None
    return None


def _collect_z3_symbols(expr: Expr, env: dict[str, object]) -> None:
    """Build a fresh, sort-correct Z3 environment for an index expression."""

    def scalar_sort(typ: Type | None):
        if isinstance(typ, BoolType):
            return z3.BoolSort()
        if isinstance(typ, FloatType):
            return z3.RealSort()
        return z3.IntSort()

    match expr:
        case Var(name=name, type=typ):
            if name not in env:
                env[name] = z3.Const(name, scalar_sort(typ))
        case TensorIndex(base=base, indices=indices):
            if base.name not in env:
                result_type: Type | None = None
                if isinstance(base.type, TensorType):
                    result_type = base.type.elem_type
                env[base.name] = z3.Function(
                    base.name,
                    *([z3.IntSort()] * len(indices)),
                    scalar_sort(result_type),
                )
            for index in indices:
                _collect_z3_symbols(index, env)
        case _:
            if not is_dataclass(expr):
                return
            for field in fields(expr):
                if field.name == "type":
                    continue
                value = getattr(expr, field.name)
                if isinstance(value, Expr):
                    _collect_z3_symbols(value, env)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, Expr):
                            _collect_z3_symbols(item, env)


def regions_write_masked(
    write: Assign | MaskedStore,
    type_env: dict[str, Type],
    regions: dict[str, GuardedRegion | None],
    pos_env: dict[str, TensorFacts],
    output_var: str,
    finite_vars: frozenset[str] = frozenset(),
) -> tuple[GuardedRegion | None, bool]:
    output: tuple[GuardedRegion | None, bool] = (None, False)
    target_region = None
    target_guard: Pred = pred_true(0)
    match write:
        case Assign(target=target, op=op, value=value):
            match target:
                case Var(name=name):
                    match type_env[name]:
                        case TensorType():
                            existing = regions.get(name)
                            target_region = existing.region if existing else None
                            target_guard = (
                                existing.guard if existing else pred_true(0)
                            )
                            if name == output_var and existing is not None:
                                output = (existing, True)
                            if existing is not None and op is not None:
                                # A compound assignment reads the target's
                                # pre-update value. Narrow that dependency only
                                # under an explicit IEEE-safe assumption.
                                rhs_facts = _facts_for_expr(
                                    value, type_env, pos_env, finite_vars
                                )
                                previous_guard = existing.guard
                                if op == "+" and name in finite_vars:
                                    # finite(old) + -inf = -inf, independent
                                    # of old at those positions.
                                    previous_guard = pred_and(
                                        previous_guard,
                                        pred_not(rhs_facts.neg_inf_where),
                                    )
                                # Multiplication by zero retains the old
                                # value's sign even under finiteness.
                                regions[name] = GuardedRegion(
                                    existing.region, previous_guard
                                )
                        case IntType() | FloatType() | BoolType():
                            target_region = []
                        case _:
                            raise AssertionError(
                                f"unhandled target type: {type_env[name]}"
                            )
                case TensorView(base=base, region=region):
                    existing = regions.get(base.name)
                    base_region = existing.region if existing else None
                    target_base_region = region_inter(base_region, region)
                    if target_base_region is not None:
                        target_region = []
                        for r, sl in zip(target_base_region, region):
                            target_region.append(
                                Slice(sub_(r.start, sl.start), sub_(r.stop, sl.start))
                            )
                    if existing is not None:
                        target_guard = _source_guard_to_local_coords(
                            existing.guard, region
                        )
                    if base.name == output_var and existing is not None:
                        output = (
                            GuardedRegion(target_base_region, existing.guard)
                            if target_base_region is not None
                            else None,
                            True,
                        )
        case MaskedStore(base=base, region=region, mask=mask):
            existing = regions.get(base.name)
            base_region = existing.region if existing else None
            target_base_region = region_inter(base_region, region)
            target_base_region = region_inter(target_base_region, mask)
            if target_base_region is not None:
                target_region = []
                for r, sl in zip(target_base_region, region):
                    target_region.append(
                        Slice(sub_(r.start, sl.start), sub_(r.stop, sl.start))
                    )
            if existing is not None:
                target_guard = _source_guard_to_local_coords(
                    existing.guard, region
                )
            if base.name == output_var and existing is not None:
                output = (
                    GuardedRegion(target_base_region, existing.guard)
                    if target_base_region is not None
                    else None,
                    True,
                )
    regions_expr_masked(
        write.value, target_region, target_guard, type_env, regions, pos_env,
        finite_vars,
    )
    return output


def bound_variable_regions_masked(
    kernel: Kernel,
    output_tensor: str,
    output_region: Region,
    output_guard: Pred,
    forward_facts: ForwardFactTrace,
    finite_vars: frozenset[str] = frozenset(),
) -> dict[str, GuardedRegion]:
    """Mask-aware backward region analysis.

    ``forward_facts`` is the program-point-sensitive result of the preceding
    forward pass. ``output_guard`` is the consumer's guard on `output_region`, in the
    canonical `_i0..._i{rank-1}` coords matching `output_tensor`'s
    natural axes.

    Returns per-input-tensor guarded dependency regions. Value-relevant
    positions are at most `{x in region : guard(x)}`; physical loads may be a
    strict superset.
    """
    type_env = build_type_env(kernel)

    internal_regions: dict[str, GuardedRegion | None] = {
        output_tensor: GuardedRegion(output_region, output_guard)
    }

    for cond_write in order_write_stmts(kernel, type_env):
        regions_write_masked(
            cond_write.write,
            type_env,
            internal_regions,
            forward_facts.before(cond_write.write),
            output_tensor,
            finite_vars,
        )

    out: dict[str, GuardedRegion] = {}
    for var, typ in type_env.items():
        match typ:
            case TensorType(dims=dims):
                if var not in internal_regions:
                    continue
                gr = internal_regions[var]
                if gr is None:
                    continue
                clipped = region_inter(gr.region, [Slice(IntLit(0), d) for d in dims])
                if clipped is None:
                    continue
                out[var] = GuardedRegion(clipped, gr.guard)
            case IntType() | FloatType() | BoolType():
                pass
            case _:
                raise AssertionError(f"unhandled type: {typ}")

    return out
