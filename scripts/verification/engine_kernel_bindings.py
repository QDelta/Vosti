"""Declarative bindings for Vosti's current kernel interfaces.

Identifier spellings belong here, at the engine boundary, not in generic
analysis or adapter rules. These declarations supply no propositions or axioms;
all generated adapters still require Verus checking against qualified artifacts.
"""

from dataclasses import dataclass, replace
import re


@dataclass(frozen=True)
class PagedAttentionBinding:
    """Engine roles mapped to arbitrary raw ports, goals and coordinates.

    No semantic fact is granted by this mapping. The generated representation
    and projection proofs must still be checked against the actual raw goals.
    """

    query: str
    keys: str
    values: str
    output: str
    logsumexp: str
    page_table: str
    query_offsets: str
    key_offsets: str
    scale: str
    max_query: str
    total_keys: str
    batch_goal: str
    selected_goal: str
    batch_selector: str
    left_selector: str
    right_selector: str
    window: str | None


# Required engine properties and their current annotation bindings. These are
# integration declarations, not verifier dispatch rules. A scope may explicitly
# bind different goal names while preserving exactly the same required roles.
KERNEL_PROOF_GOALS = {
    "regional_certificate": {"batch": "batch_invariance"},
    "structural_batch_decomposition_not_numeric_correctness": {"batch": "batch_invariance"},
    "conditional_relational_certificate": {
        "batch": "batch_invariance", "selected": "selected_row_prefix_equivalence"},
    "exact_effect_certificate": {"batch": "batch_invariance", "effect": "exact_effect"},
}


def required_kernel_goal_bindings(contract: dict) -> dict[str, str]:
    defaults = KERNEL_PROOF_GOALS.get(contract["evidence"])
    if defaults is None:
        raise ValueError(f"unknown kernel qualification evidence role: {contract['evidence']!r}")
    binding = contract.get("proof_goals", defaults)
    if (not isinstance(binding, dict) or set(binding) != set(defaults)
            or not all(isinstance(v, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", v)
                       for v in binding.values())
            or len(set(binding.values())) != len(binding)):
        raise ValueError("kernel proof-goal binding must cover each required role with a distinct goal")
    return {role: binding[role] for role in defaults}


FULL_ATTENTION_BINDING = PagedAttentionBinding(
    query="q", keys="k_cache", values="v_cache", output="o", logsumexp="lse",
    page_table="block_table", query_offsets="cu_seqlens_q", key_offsets="cu_seqlens_k",
    scale="scale_log2", max_query="max_seqlen_q", total_keys="Tk",
    batch_goal="batch_invariance", selected_goal="selected_row_prefix_equivalence",
    batch_selector="x", left_selector="selected_left_row", right_selector="selected_right_row",
    window=None,
)
SLIDING_ATTENTION_BINDING = replace(FULL_ATTENTION_BINDING, window="window_size")

ATTENTION_BINDINGS = {
    "paged_attention": FULL_ATTENTION_BINDING,
    "paged_attention_swa": SLIDING_ATTENTION_BINDING,
}


def attention_binding_for(contract):
    """Bind the interface's ports and the scope's required annotation goals."""
    binding = ATTENTION_BINDINGS[contract["wrapper"]]
    goals = required_kernel_goal_bindings(contract)
    if set(goals) != {"batch", "selected"}:
        raise ValueError("paged attention needs batch and selected-row proof roles")
    return replace(binding, batch_goal=goals["batch"], selected_goal=goals["selected"])
