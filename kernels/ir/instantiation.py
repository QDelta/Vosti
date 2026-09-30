"""Structural suggestions for quantified-schema instantiation, never evidence.

Only paths containing a schema variable are matched. Fixed siblings may differ
(for example, two coordinates related by premises); consumers must prove the
FULL instantiated domain and region claims. No equality follows from a match.
"""
from itertools import product

import z3


def _terms(expressions):
    seen = set()
    pending = list(reversed(expressions))
    while pending:
        term = pending.pop()
        if not z3.is_app(term) or term.get_id() in seen:
            continue  # Never descend under quantifiers or expose bound variables.
        seen.add(term.get_id())
        yield term
        pending.extend(reversed(term.children()))


def suggest_instantiations(patterns, demands, symbols):
    """Suggest complete, sort-correct substitutions from shared term structure.

    For example, F(a, p) and F(b, i * B / P) suggest p := i * B / P.
    The suggestion proves neither a=b nor any bounds on p. Unsupported matches
    produce no suggestion; the caller retains its existing proof-search path.
    """
    symbols = tuple(symbols)
    if not symbols:
        return []
    holes = {symbol.get_id(): index for index, symbol in enumerate(symbols)}
    if len(holes) != len(symbols) or any(
        not z3.is_const(s) or s.decl().kind() != z3.Z3_OP_UNINTERPRETED
        for s in symbols
    ):
        raise ValueError('schema symbols must be distinct constants')
    demand_terms = list(_terms(demands))
    candidates = [[] for _ in symbols]
    seen = [set() for _ in symbols]

    def contains_hole(term):
        return any(node.get_id() in holes for node in _terms([term]))

    def admissible(term):
        # Substitutions must be closed with respect to binders and independent
        # of the schema placeholders. Ordinary free proof-context symbols are OK.
        return (z3.is_app(term) and term.get_id() not in holes
                and all(admissible(child) for child in term.children()))

    def match(pattern, demand, bindings):
        index = holes.get(pattern.get_id())
        if index is not None:
            if not pattern.sort().eq(demand.sort()) or not admissible(demand):
                return False
            if index in bindings and not bindings[index].eq(demand):
                return False
            bindings[index] = demand
            return True
        if not contains_hole(pattern):
            return True  # Fixed context is checked by the actual proof, not here.
        if not (z3.is_app(pattern) and z3.is_app(demand)
                and pattern.decl().eq(demand.decl())
                and pattern.num_args() == demand.num_args()):
            return False
        return all(match(a, b, bindings) for a, b in zip(pattern.children(), demand.children()))

    for pattern in _terms(patterns):
        # A bare placeholder has no structural anchor; it would suggest every
        # subterm. The caller already supplies its ordinary fallback candidates.
        if pattern.get_id() in holes or not contains_hole(pattern):
            continue
        for demand in demand_terms:
            bindings = {}
            if match(pattern, demand, bindings):
                for index, value in bindings.items():
                    if value.get_id() not in seen[index]:
                        candidates[index].append(value)
                        seen[index].add(value.get_id())
    return list(product(*candidates))
