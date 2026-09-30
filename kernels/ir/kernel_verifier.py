"""Qualify named typed goals without kernel-name or theorem-name dispatch.

Relational equality and temporal effects use their respective analyses of the
same prepared IR. Neither proof kind can silently fall back to the other.
"""

from dataclasses import fields, is_dataclass

from .annotations import After, Before, DTypeOf
from .exact_effects import verify_prepared_exact_effects
from .proof_preparation import PreparedKernelProofs, prepare_goal_proof
from .relational_verifier import verify_prepared_annotations


def _temporal(value):
    if isinstance(value, (Before, After, DTypeOf)):
        return True
    if isinstance(value, (list, tuple, set)):
        return any(_temporal(item) for item in value)
    if is_dataclass(value):
        return any(_temporal(getattr(value, f.name)) for f in fields(value))
    return False


def verify_kernel_goal(prepared: PreparedKernelProofs, goal_name: str, *, preserve_analyzer_conditions: bool = False):
    """Return qualified evidence; extra premises need an explicit consumer opt-in."""
    if type(preserve_analyzer_conditions) is not bool:
        raise TypeError("preserve_analyzer_conditions must be boolean")
    goals = [g for g in prepared.goals if g.name == goal_name]
    if len(goals) != 1:
        raise ValueError(f"No @verif proof goal named {goal_name!r} for {prepared.kernel.name}")
    goal = goals[0]
    if _temporal(goal):
        report = verify_prepared_exact_effects(prepared, goal_name=goal_name)
        if not report.proved or report.verified_contract is None:
            failures = [c.name for c in report.checks if not c.proved]
            raise ValueError(f"exact-effect verifier: {report.unsupported_reason or failures}")
        if not preserve_analyzer_conditions:
            raise ValueError("exact-effect verifier: consumer must preserve physical correspondence obligations")
    else:
        report = verify_prepared_annotations(prepare_goal_proof(prepared, goal_name),
                                             preserve_analyzer_conditions=preserve_analyzer_conditions)
    assert report.verified_contract is not None
    return report.verified_contract
