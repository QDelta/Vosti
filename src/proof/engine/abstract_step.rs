// Construct the ghost abstract step and prove `ibm_step` for the executable
// `refinement_step`. Cache contracts follow from verified scheduler geometry
// and `model_forward` output; see `proof::cache::semantics` and
// `proof::cache::plan_layout`.
//
// Given the engine's real step (`engine_step_relation`) and `inv(old)`, we build
// `new_ibm` explicitly: each surviving request's machine takes the engine's new
// request-state (legitimate because `machine_step_transition`/`can_step`/… depend
// on the state only through its `@`-views, and `shared_rid_coherence` gives
// view-equality between the engine's `live_requests@[rid]` and the machine's
// `request_state`).  The machine's KV cache contents are set to the per-layer
// machine *stores* (so the same `new_ibm` also satisfies the cache contract),
// but only their *length* matters for `ibm_step`.
//
// The architecture-neutral constructor evaluates each selected request with
// the same closed model dispatch as Engine and relocates its logical cache
// prefix through the shared block-table contract.

use crate::exec::engine::{Engine, StepReprs};
use crate::proof::reference::independent_batch_model::*;
#[cfg(verus_only)]
use crate::proof::engine::refinement::inv;
#[cfg(verus_only)]
use crate::proof::reference::request_machine::{
    machine_step_transition_full, request_machine_alive, singleton_block_rows, slots_from,
    token_seq_to_int, RequestMachine,
};
use crate::exec::request_state::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
#[cfg(verus_only)]
use crate::{proof::model::architecture as MA, proof::model::family_layout as LAYOUT, proof::model::relocation as RELOCATION};
use vstd::prelude::*;

verus! {

// View-equal pre-states finish, step, and are steppable alike (all depend only
// on `@`-views).  These transfer the engine-side facts to the abstract machine.
pub proof fn lemma_view_eq_can_step(a: RequestState, b: RequestState)
    requires request_state_view_eq(a, b), can_step(a),
    ensures can_step(b),
{
    lemma_request_lifecycle_view_eq_fields(a, b);
    lemma_eos_tokens_from_policy_eq(a, b);
    lemma_same_eos_tokens_view_eq(eos_tokens(a), eos_tokens(b));
}

pub proof fn lemma_view_eq_should_finish(a: RequestState, b: RequestState, e: TokenId)
    requires request_state_view_eq(a, b),
    ensures should_finish_after_append(a, e) == should_finish_after_append(b, e),
{
    lemma_request_lifecycle_view_eq_fields(a, b);
    lemma_eos_tokens_from_policy_eq(a, b);
    lemma_same_eos_tokens_contains(eos_tokens(a), eos_tokens(b), e);
}

pub proof fn lemma_transition_respects_view_eq(
    pre_a: RequestState, pre_b: RequestState, post: RequestState,
    nss: SamplerState, e: TokenId,
)
    requires
        request_state_view_eq(pre_a, pre_b),
        machine_step_transition_full(pre_a, post, nss, e),
    ensures
        machine_step_transition_full(pre_b, post, nss, e),
{
    lemma_request_lifecycle_view_eq_transitive(post, pre_a, pre_b);
}

// Architecture-neutral row lookup over the scheduler's common `StepReprs`
// sequence.
pub open spec fn architecture_scheduled_index_of(
    reprs: StepReprs,
    rid: RequestId,
) -> int {
    choose|i: int| 0 <= i < reprs.scheduled.len()
        && reprs.scheduled[i] == rid
}

pub open spec fn architecture_machine_query_len(
    reprs: StepReprs,
    rid: RequestId,
) -> nat {
    let i = architecture_scheduled_index_of(reprs, rid);
    (reprs.cu_q[i + 1] - reprs.cu_q[i]) as nat
}

pub open spec fn architecture_machine_key_len(
    reprs: StepReprs,
    rid: RequestId,
) -> nat {
    let i = architecture_scheduled_index_of(reprs, rid);
    (reprs.cu_k[i + 1] - reprs.cu_k[i]) as nat
}

pub open spec fn architecture_machine_prefix_len(
    reprs: StepReprs,
    rid: RequestId,
) -> nat {
    let q_len = architecture_machine_query_len(reprs, rid);
    let k_len = architecture_machine_key_len(reprs, rid);
    (k_len - q_len) as nat
}

// Complete token history processed by one materialized row.  Decode consumes
// the engine request's current history; prefill consumes the prompt prefix
// through the row's key length.  This vocabulary is independent of the model
// family and excludes the newly sampled token, which is not in KV yet.
pub open spec fn architecture_machine_processed_tokens(
    old_e: Engine,
    reprs: StepReprs,
    rid: RequestId,
) -> IntTensor1D {
    let i = architecture_scheduled_index_of(reprs, rid);
    let state = old_e.cs.live_requests@[rid];
    let k_len = architecture_machine_key_len(reprs, rid);
    if old_e.cs.running@.contains(rid) {
        token_seq_to_int(history(state))
    } else {
        token_seq_to_int(state.prompt_tokens@).subrange(0, k_len as int)
    }
}

// Private contiguous cache base for an independently evaluated request.  Its
// logical prefix is copied from the engine's pre-forward paged cache through
// that request's own block-table row; capacity and all suffix cells come from
// the old private machine cache.
pub open spec fn architecture_machine_cache_base(
    old_e: Engine,
    old_ibm: IndependentBatchModel,
    reprs: StepReprs,
    rid: RequestId,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> {
    let num_layers = old_ibm.model_config.num_layers as nat;
    let i = architecture_scheduled_index_of(reprs, rid);
    let bt_row = reprs.bt[i];
    let prefix_len = architecture_machine_prefix_len(reprs, rid);
    Seq::new(num_layers, |layer: int| {
        let old_machine = old_ibm.machines[rid].kv_cache_reprs[layer];
        let engine = old_e.kv_caches_repr@[layer];
        (
            LAYOUT::relocated_prefix_base(
                old_machine.0, engine.0, bt_row, prefix_len,
            ),
            LAYOUT::relocated_prefix_base(
                old_machine.1, engine.1, bt_row, prefix_len,
            ),
        )
    })
}

// Post-forward private cache for one selected request. This is the
// architecture's ordinary singleton forward over a neutral contiguous layout.
// Request projection plus relocation relates it to the engine's batched
// post-cache.
pub open spec fn architecture_machine_cache_after(
    old_e: Engine,
    old_ibm: IndependentBatchModel,
    reprs: StepReprs,
    rid: RequestId,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> {
    let i = architecture_scheduled_index_of(reprs, rid);
    let lo = reprs.cu_q[i];
    let hi = reprs.cu_q[i + 1];
    let q_len = architecture_machine_query_len(reprs, rid);
    let k_len = architecture_machine_key_len(reprs, rid);
    let prefix_len = architecture_machine_prefix_len(reprs, rid);
    MA::model_forward_kv_reprs(
        old_ibm.wr,
        old_ibm.architecture_repr,
        reprs.input_ids.subrange(lo, hi),
        reprs.positions.subrange(lo, hi),
        architecture_machine_cache_base(old_e, old_ibm, reprs, rid),
        slots_from(prefix_len, q_len),
        seq![0int, q_len as int],
        seq![0int, k_len as int],
        q_len,
        k_len,
        singleton_block_rows(k_len),
    )
}

// Logits produced by the same private contiguous singleton forward.  Naming
// this projection keeps observable refinement independent of family modules.
pub open spec fn architecture_machine_logits_after(
    old_e: Engine,
    old_ibm: IndependentBatchModel,
    reprs: StepReprs,
    rid: RequestId,
) -> Tensor2D {
    let i = architecture_scheduled_index_of(reprs, rid);
    let lo = reprs.cu_q[i];
    let hi = reprs.cu_q[i + 1];
    let q_len = architecture_machine_query_len(reprs, rid);
    let k_len = architecture_machine_key_len(reprs, rid);
    let prefix_len = architecture_machine_prefix_len(reprs, rid);
    MA::model_forward_logits_repr(
        old_ibm.wr,
        old_ibm.architecture_repr,
        reprs.input_ids.subrange(lo, hi),
        reprs.positions.subrange(lo, hi),
        architecture_machine_cache_base(old_e, old_ibm, reprs, rid),
        slots_from(prefix_len, q_len),
        seq![0int, q_len as int],
        seq![0int, k_len as int],
        q_len,
        k_len,
        singleton_block_rows(k_len),
    )
}

// The private singleton forward is definitionally a continuation over the
// row's complete processed history.  Scheduler layout is the only nontrivial
// bridge: it identifies the packed row with the suffix beginning at the
// cached-prefix length and supplies the corresponding logical positions.
#[verifier::spinoff_prover]
pub proof fn lemma_architecture_machine_forward_is_prefix_continuation(
    old_e: Engine,
    old_ibm: IndependentBatchModel,
    reprs: StepReprs,
    rid: RequestId,
)
    requires
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_ok(old_e, reprs),
        reprs.scheduled.contains(rid),
    ensures ({
        let i = architecture_scheduled_index_of(reprs, rid);
        let lo = reprs.cu_q[i];
        let hi = reprs.cu_q[i + 1];
        let q_len = architecture_machine_query_len(reprs, rid);
        let k_len = architecture_machine_key_len(reprs, rid);
        let prefix_len = architecture_machine_prefix_len(reprs, rid);
        let tokens = architecture_machine_processed_tokens(
            old_e, reprs, rid,
        );
        let base = architecture_machine_cache_base(
            old_e, old_ibm, reprs, rid,
        );
        &&& tokens.len() == k_len
        &&& reprs.input_ids.subrange(lo, hi)
            == tokens.subrange(prefix_len as int, k_len as int)
        &&& reprs.positions.subrange(lo, hi)
            == crate::proof::tensor::geometry::positions_from(prefix_len, q_len)
        &&& architecture_machine_cache_after(
            old_e, old_ibm, reprs, rid,
        ) == crate::proof::model::cache::prefix_continuation_cache_reprs(
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
            tokens,
            prefix_len,
            base,
        )
        &&& architecture_machine_logits_after(
            old_e, old_ibm, reprs, rid,
        ) == crate::proof::model::cache::prefix_continuation_logits_repr(
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
            tokens,
            prefix_len,
            base,
        )
    }),
{
    reveal(crate::exec::engine::step_reprs_wf);
    let i = architecture_scheduled_index_of(reprs, rid);
    assert(exists|index: int| 0 <= index < reprs.scheduled.len()
        && reprs.scheduled[index] == rid);
    assert(0 <= i < reprs.scheduled.len()
        && reprs.scheduled[i] == rid);
    assert(crate::exec::engine::reprs_forward_layout_at(old_e, reprs, i));
    reveal(crate::exec::engine::reprs_forward_layout_at);
    let lo = reprs.cu_q[i];
    let hi = reprs.cu_q[i + 1];
    let q_len = architecture_machine_query_len(reprs, rid);
    let k_len = architecture_machine_key_len(reprs, rid);
    let prefix_len = architecture_machine_prefix_len(reprs, rid);
    let state = old_e.cs.live_requests@[rid];
    let tokens = architecture_machine_processed_tokens(
        old_e, reprs, rid,
    );
    assert(0 <= lo < hi <= reprs.input_ids.len() as int);
    assert(q_len == (hi - lo) as nat);
    assert(hi - lo == q_len as int);
    assert(q_len > 0);
    assert(q_len <= k_len);
    assert(prefix_len == k_len - q_len);
    if old_e.cs.running@.contains(rid) {
        let h = history(state);
        assert(k_len == h.len());
        assert(q_len == 1);
        assert(prefix_len + 1 == k_len);
        assert(tokens.len() == k_len);
        assert(reprs.input_ids.subrange(lo, hi)
            =~= tokens.subrange(prefix_len as int, k_len as int)) by {
            assert forall|x: int| 0 <= x < hi - lo implies
                reprs.input_ids.subrange(lo, hi)[x]
                    == tokens.subrange(
                        prefix_len as int, k_len as int,
                    )[x]
            by {
                assert(x == 0);
                assert(reprs.input_ids[lo] == h[h.len() - 1] as int);
                assert(tokens[prefix_len as int]
                    == h[prefix_len as int] as int);
            }
        }
        assert(reprs.positions.subrange(lo, hi)
            =~= crate::proof::tensor::geometry::positions_from(prefix_len, q_len)) by {
            assert forall|x: int| 0 <= x < hi - lo implies
                reprs.positions.subrange(lo, hi)[x]
                    == crate::proof::tensor::geometry::positions_from(
                        prefix_len, q_len,
                    )[x]
            by {
                assert(x == 0);
            }
        }
    } else {
        let prompt = token_seq_to_int(state.prompt_tokens@);
        let cached = (reprs.cu_k[i + 1] - reprs.cu_k[i])
            - (hi - lo);
        assert(cached == prefix_len as int);
        assert(0 <= k_len as int <= prompt.len());
        assert(tokens.len() == k_len);
        assert(reprs.input_ids.subrange(lo, hi)
            =~= tokens.subrange(prefix_len as int, k_len as int)) by {
            assert forall|x: int| 0 <= x < hi - lo implies
                reprs.input_ids.subrange(lo, hi)[x]
                    == tokens.subrange(
                        prefix_len as int, k_len as int,
                    )[x]
            by {
                let p = prefix_len as int + x;
                assert(lo <= lo + x < hi);
                assert(0 <= p < k_len as int);
                assert(reprs.input_ids[lo + x]
                    == state.prompt_tokens@[p] as int);
                assert(tokens[p] == prompt[p]);
                assert(prompt[p] == state.prompt_tokens@[p] as int);
            }
        }
        assert(reprs.positions.subrange(lo, hi)
            =~= crate::proof::tensor::geometry::positions_from(prefix_len, q_len)) by {
            assert forall|x: int| 0 <= x < hi - lo implies
                reprs.positions.subrange(lo, hi)[x]
                    == crate::proof::tensor::geometry::positions_from(
                        prefix_len, q_len,
                    )[x]
            by {
                let p = prefix_len as int + x;
                assert(reprs.input_ids[lo + x]
                    == state.prompt_tokens@[p] as int);
                assert(reprs.positions[lo + x] == p);
            }
        }
    }
    reveal(architecture_machine_cache_after);
    reveal(architecture_machine_logits_after);
    reveal(crate::proof::model::cache::prefix_continuation_cache_reprs);
    reveal(crate::proof::model::cache::prefix_continuation_logits_repr);
    reveal(crate::proof::reference::independent_batch_model::ibm_semantic_model);
}

// A sampled row has processed the request's complete pre-step history.  The
// only materialized rows that stop at an intermediate prompt prefix are the
// KV-only chunk rows excluded by `sample_mask`.
pub proof fn lemma_architecture_emitting_processed_tokens_are_history(
    old_e: Engine,
    reprs: StepReprs,
    row: int,
)
    requires
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_sample_policy(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_ok(old_e, reprs),
        crate::exec::cache_scheduler::waiting_unstarted(&old_e.cs),
        0 <= row < reprs.scheduled.len(),
        reprs.sample_mask[row],
    ensures ({
        let rid = reprs.scheduled[row];
        &&& old_e.cs.live_requests@.contains_key(rid)
        &&& architecture_machine_processed_tokens(old_e, reprs, rid)
            == token_seq_to_int(history(old_e.cs.live_requests@[rid]))
    }),
{
    let rid = reprs.scheduled[row];
    assert(reprs.scheduled.contains(rid));
    let selected = architecture_scheduled_index_of(reprs, rid);
    assert(selected == row) by {
        assert(reprs.scheduled.no_duplicates());
    }
    assert(crate::exec::engine::reprs_forward_layout_at(old_e, reprs, row));
    if old_e.cs.running@.contains(rid) {
        reveal(architecture_machine_processed_tokens);
    } else {
        crate::exec::engine::lemma_waiting_sample_row_is_final(
            old_e, reprs, row,
        );
        reveal(crate::exec::engine::reprs_forward_layout_at);
        reveal(architecture_machine_processed_tokens);
        assert(old_e.cs.live_requests@[rid].generated_tokens@.len() == 0);
        assert(history(old_e.cs.live_requests@[rid])
            == old_e.cs.live_requests@[rid].prompt_tokens@);
    }
}

pub proof fn lemma_architecture_machine_cache_after_len(
    old_e: Engine,
    old_ibm: IndependentBatchModel,
    reprs: StepReprs,
    rid: RequestId,
)
    requires
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        reprs.scheduled.contains(rid),
        old_ibm.model_config == old_e.model_config,
        old_ibm.wr == reprs.wr,
        model_weights_architecture_repr_valid(
            old_ibm.wr, old_ibm.architecture_repr,
        ),
    ensures
        architecture_machine_cache_after(
            old_e, old_ibm, reprs, rid,
        ).len() == old_ibm.model_config.num_layers as nat,
{
    reveal(crate::exec::engine::step_reprs_wf);
    let i = architecture_scheduled_index_of(reprs, rid);
    assert(exists|index: int| 0 <= index < reprs.scheduled.len()
        && reprs.scheduled[index] == rid);
    assert(0 <= i < reprs.scheduled.len()
        && reprs.scheduled[i] == rid);
    crate::proof::tensor::geometry::lemma_cu_int_bounds(
        reprs.cu_q, reprs.scheduled.len() as int,
    );
    let lo = reprs.cu_q[i];
    let hi = reprs.cu_q[i + 1];
    let q_len = architecture_machine_query_len(reprs, rid);
    let k_len = architecture_machine_key_len(reprs, rid);
    let prefix_len = architecture_machine_prefix_len(reprs, rid);
    let base = architecture_machine_cache_base(
        old_e, old_ibm, reprs, rid,
    );
    let machine_slots = slots_from(prefix_len, q_len);
    assert(base.len() == old_ibm.model_config.num_layers as nat);
    assert(old_ibm.wr.layers.len()
        == old_ibm.model_config.num_layers as nat);
    assert(reprs.input_ids.subrange(lo, hi).len()
        == reprs.positions.subrange(lo, hi).len());
    assert(machine_slots.len() == q_len);
    MA::lemma_model_forward_kv_reprs_len(
        old_ibm.wr,
        old_ibm.architecture_repr,
        reprs.input_ids.subrange(lo, hi),
        reprs.positions.subrange(lo, hi),
        base,
        machine_slots,
        seq![0int, q_len as int],
        seq![0int, k_len as int],
        q_len,
        k_len,
        singleton_block_rows(k_len),
    );
}

// Exact scheduler/layout obligation for relating the engine's batched forward
// to the architecture-neutral private singleton forward.  Keeping this as one
// named predicate lets the scheduler proof remain model-opaque.
pub open spec fn architecture_machine_relocation_ready(
    old_e: Engine,
    old_ibm: IndependentBatchModel,
    reprs: StepReprs,
    rid: RequestId,
) -> bool {
    let i = architecture_scheduled_index_of(reprs, rid);
    let q_len = architecture_machine_query_len(reprs, rid);
    let k_len = architecture_machine_key_len(reprs, rid);
    let prefix_len = architecture_machine_prefix_len(reprs, rid);
    RELOCATION::model_forward_engine_to_machine_ready(
        reprs.wr,
        old_ibm.architecture_repr,
        reprs.input_ids,
        reprs.positions,
        old_e.kv_caches_repr@,
        reprs.slots,
        reprs.cu_q,
        reprs.cu_k,
        reprs.max_q,
        reprs.max_k,
        reprs.bt,
        reprs.slots.subrange(reprs.cu_q[i], reprs.cu_q[i + 1]),
        i as nat,
        architecture_machine_cache_base(old_e, old_ibm, reprs, rid),
        slots_from(prefix_len, q_len),
        singleton_block_rows(k_len)[0],
        q_len,
        k_len,
    )
}

// The central cache-construction bridge: the selected engine logits and every
// logical post-forward KV cell agree with the independent singleton forward
// used as the request machine's private cache.
pub proof fn lemma_architecture_machine_forward_matches_engine(
    old_e: Engine,
    old_ibm: IndependentBatchModel,
    reprs: StepReprs,
    rid: RequestId,
)
    requires
        architecture_machine_relocation_ready(
            old_e, old_ibm, reprs, rid,
        ),
        reprs.scheduled.contains(rid),
        old_ibm.wr == reprs.wr,
        reprs.wr == RT::model_weights_repr_of(&old_e.weights_perms@),
        old_ibm.architecture_repr
            == RT::model_weights_architecture_repr_of(
                &old_e.weights_perms@,
            ),
    ensures ({
        let i = architecture_scheduled_index_of(reprs, rid);
        let q_len = architecture_machine_query_len(reprs, rid);
        let k_len = architecture_machine_key_len(reprs, rid);
        let prefix_len = architecture_machine_prefix_len(reprs, rid);
        let machine_base = architecture_machine_cache_base(
            old_e, old_ibm, reprs, rid,
        );
        let machine_slots = slots_from(prefix_len, q_len);
        let machine_bt = singleton_block_rows(k_len);
        &&& crate::exec::engine::architecture_step_logits_repr(
            old_e, reprs,
        ).subrange(reprs.cu_q[i], reprs.cu_q[i + 1])
            == MA::model_forward_logits_repr(
                old_ibm.wr,
                old_ibm.architecture_repr,
                reprs.input_ids.subrange(
                    reprs.cu_q[i], reprs.cu_q[i + 1],
                ),
                reprs.positions.subrange(
                    reprs.cu_q[i], reprs.cu_q[i + 1],
                ),
                machine_base,
                machine_slots,
                seq![0int, q_len as int],
                seq![0int, k_len as int],
                q_len,
                k_len,
                machine_bt,
            )
        &&& LAYOUT::cache_sequence_logical_prefix_equal(
            crate::exec::engine::architecture_engine_post_kv_of(
                old_e, reprs, old_e.kv_caches_repr@,
            ),
            reprs.bt[i],
            architecture_machine_cache_after(
                old_e, old_ibm, reprs, rid,
            ),
            machine_bt[0],
            old_ibm.wr.layers.len(),
            k_len,
        )
    }),
{
    reveal(architecture_machine_relocation_ready);
    let i = architecture_scheduled_index_of(reprs, rid);
    assert(exists|index: int| 0 <= index < reprs.scheduled.len()
        && reprs.scheduled[index] == rid);
    assert(0 <= i < reprs.scheduled.len()
        && reprs.scheduled[i] == rid);
    let q_len = architecture_machine_query_len(reprs, rid);
    let k_len = architecture_machine_key_len(reprs, rid);
    let prefix_len = architecture_machine_prefix_len(reprs, rid);
    let machine_base = architecture_machine_cache_base(
        old_e, old_ibm, reprs, rid,
    );
    let machine_slots = slots_from(prefix_len, q_len);
    let machine_bt = singleton_block_rows(k_len);
    assert(machine_bt.len() == 1);
    assert(machine_bt =~= seq![machine_bt[0]]);
    RELOCATION::lemma_model_forward_engine_to_machine(
        reprs.wr,
        old_ibm.architecture_repr,
        reprs.input_ids,
        reprs.positions,
        old_e.kv_caches_repr@,
        reprs.slots,
        reprs.cu_q,
        reprs.cu_k,
        reprs.max_q,
        reprs.max_k,
        reprs.bt,
        reprs.slots.subrange(reprs.cu_q[i], reprs.cu_q[i + 1]),
        i as nat,
        machine_base,
        machine_slots,
        machine_bt[0],
        q_len,
        k_len,
    );
    reveal(crate::exec::engine::architecture_step_logits_repr);
    reveal(crate::exec::engine::architecture_engine_post_kv_of);
    reveal(architecture_machine_cache_after);
    assert(crate::exec::engine::architecture_step_logits_repr(
        old_e, reprs,
    ).subrange(reprs.cu_q[i], reprs.cu_q[i + 1])
        == MA::model_forward_logits_repr(
            old_ibm.wr,
            old_ibm.architecture_repr,
            reprs.input_ids.subrange(
                reprs.cu_q[i], reprs.cu_q[i + 1],
            ),
            reprs.positions.subrange(
                reprs.cu_q[i], reprs.cu_q[i + 1],
            ),
            machine_base,
            machine_slots,
            seq![0int, q_len as int],
            seq![0int, k_len as int],
            q_len,
            k_len,
            machine_bt,
        ));
    assert(LAYOUT::cache_sequence_logical_prefix_equal(
        crate::exec::engine::architecture_engine_post_kv_of(
            old_e, reprs, old_e.kv_caches_repr@,
        ),
        reprs.bt[i],
        architecture_machine_cache_after(
            old_e, old_ibm, reprs, rid,
        ),
        machine_bt[0],
        old_ibm.wr.layers.len(),
        k_len,
    ));
}

pub open spec fn architecture_stepped_machine(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    reprs: StepReprs,
    rid: RequestId,
) -> RequestMachine {
    if emitted.contains_key(rid) {
        let new_state = new_e.cs.live_requests@[rid];
        RequestMachine {
            request_state: new_state,
            kv_initialized: true,
            kv_tokens: (history(new_state).len() - 1) as nat,
            kv_cache_reprs: architecture_machine_cache_after(
                old_e, old_ibm, reprs, rid,
            ),
        }
    } else {
        old_ibm.machines[rid]
    }
}

pub open spec fn architecture_stepped_ibm(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    reprs: StepReprs,
) -> IndependentBatchModel {
    IndependentBatchModel {
        model_config: old_ibm.model_config,
        wr: old_ibm.wr,
        architecture_repr: old_ibm.architecture_repr,
        machines: Map::new(
            old_ibm.machines.dom().intersect(
                new_e.cs.live_requests@.dom(),
            ),
            |rid: RequestId| architecture_stepped_machine(
                old_e, new_e, old_ibm, emitted, reprs, rid,
            ),
        ),
    }
}

// One transitioned-or-unchanged machine.

// The constructed post-step abstract model.

// Structural interface between a cache constructor and the abstract request
// transition.  The proof of `ibm_step` needs only the selected machine's cache
// length; cache contents remain opaque and are handled by the separate
// coherence/fidelity proof. The architecture-neutral cache constructor
// discharges this predicate.
pub open spec fn constructed_ibm_shape(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    new_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
) -> bool {
    &&& new_ibm.model_config == old_ibm.model_config
    &&& new_ibm.wr == old_ibm.wr
    &&& new_ibm.architecture_repr == old_ibm.architecture_repr
    &&& new_ibm.machines.dom() == old_ibm.machines.dom().intersect(
        new_e.cs.live_requests@.dom(),
    )
    &&& forall|rid: RequestId|
        #![trigger new_ibm.machines.contains_key(rid)]
        new_ibm.machines.contains_key(rid) ==> {
            if emitted.contains_key(rid) {
                let machine = new_ibm.machines[rid];
                let new_state = new_e.cs.live_requests@[rid];
                &&& machine.request_state == new_state
                &&& machine.kv_initialized
                &&& machine.kv_tokens
                    == (history(new_state).len() - 1) as nat
                &&& machine.kv_cache_reprs.len()
                    == old_ibm.model_config.num_layers as nat
            } else {
                new_ibm.machines[rid] == old_ibm.machines[rid]
            }
        }
}

pub proof fn construct_abstract_step_from_shape(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    new_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: StepReprs,
    logits_repr: Tensor2D,
    post_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
)
    requires
        inv(&old_e, &old_ibm),
        crate::exec::engine::engine_step_relation_with_payload(
            old_e, new_e, emitted, samples, reprs, logits_repr, post_kv,
        ),
        constructed_ibm_shape(
            old_e, new_e, old_ibm, new_ibm, emitted,
        ),
    ensures ibm_step(old_ibm, new_ibm, emitted.dom(), samples),
{
    reveal(constructed_ibm_shape);
    let selected = emitted.dom();
    assert(old_e.cs.live_requests@.dom() == old_ibm.machines.dom());
    assert(selected.subset_of(old_ibm.machines.dom())) by {
        assert forall|rid: RequestId| selected.contains(rid)
            implies old_ibm.machines.contains_key(rid)
        by {
            assert(emitted.contains_key(rid));
            assert(old_e.cs.live_requests@.contains_key(rid));
        }
    }
    assert(new_ibm.machines.dom().subset_of(old_ibm.machines.dom()));
    assert forall|rid: RequestId|
        #[trigger] old_ibm.machines.contains_key(rid)
            && !selected.contains(rid)
        implies new_ibm.machines.contains_key(rid)
            && new_ibm.machines[rid] == old_ibm.machines[rid]
    by {
        assert(!emitted.contains_key(rid));
        assert(old_e.cs.live_requests@.contains_key(rid));
        assert(new_e.cs.live_requests@.contains_key(rid));
        assert(new_ibm.machines.contains_key(rid));
    }
    assert forall|rid: RequestId| #[trigger] selected.contains(rid) implies {
        let pre = old_ibm.machines[rid].request_state;
        &&& can_step(pre)
        &&& samples.contains_key(rid)
        &&& (if should_finish_after_append(pre, samples[rid].1) {
                !new_ibm.machines.contains_key(rid)
            } else {
                new_ibm.machines.contains_key(rid)
                && machine_step_transition_full(
                    pre,
                    new_ibm.machines[rid].request_state,
                    samples[rid].0,
                    samples[rid].1,
                )
                && request_machine_alive(
                    new_ibm.machines[rid], old_ibm.model_config,
                )
            })
    } by {
        assert(emitted.contains_key(rid));
        let pre = old_ibm.machines[rid].request_state;
        let engine_pre = old_e.cs.live_requests@[rid];
        assert(old_e.cs.live_requests@.contains_key(rid));
        assert(request_state_view_eq(engine_pre, pre));
        assert(can_step(engine_pre));
        lemma_view_eq_can_step(engine_pre, pre);
        assert(samples.contains_key(rid));
        assert(emitted[rid] == samples[rid].1);
        lemma_view_eq_should_finish(engine_pre, pre, emitted[rid]);
        if should_finish_after_append(pre, samples[rid].1) {
            assert(should_finish_after_append(engine_pre, emitted[rid]));
            assert(!new_e.cs.live_requests@.contains_key(rid));
            assert(!new_ibm.machines.contains_key(rid));
        } else {
            assert(!should_finish_after_append(engine_pre, emitted[rid]));
            assert(new_e.cs.live_requests@.contains_key(rid));
            assert(new_ibm.machines.contains_key(rid));
            let machine = new_ibm.machines[rid];
            let new_state = new_e.cs.live_requests@[rid];
            assert(machine.request_state == new_state);
            assert(machine_step_transition_full(
                engine_pre,
                new_state,
                samples[rid].0,
                samples[rid].1,
            ));
            lemma_transition_respects_view_eq(
                engine_pre,
                pre,
                new_state,
                samples[rid].0,
                samples[rid].1,
            );
            assert(valid_request_state(new_state));
            assert(history(new_state).len() > 0);
            assert(machine.kv_cache_reprs.len()
                == old_ibm.model_config.num_layers as nat);
            assert(machine.kv_tokens
                == (history(new_state).len() - 1) as nat);
            assert(machine.kv_initialized);
            assert(request_machine_alive(
                machine, old_ibm.model_config,
            ));
        }
    }
}

// Prove that the constructed abstract model satisfies `ibm_step`.

// Architecture-neutral constructor: selected private caches are the ordinary
// singleton forward of the selected architecture, while the structural IBM
// transition reuses the cache-opaque proof above.
pub proof fn construct_architecture_abstract_step(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: StepReprs,
)
    requires
        inv(&old_e, &old_ibm),
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        old_ibm.wr == reprs.wr,
        model_weights_architecture_repr_valid(
            old_ibm.wr, old_ibm.architecture_repr,
        ),
    ensures
        ibm_step(
            old_ibm,
            architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
            emitted.dom(),
            samples,
        ),
{
    reveal(crate::exec::engine::architecture_engine_step_relation);
    let new_ibm = architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    assert(constructed_ibm_shape(
        old_e, new_e, old_ibm, new_ibm, emitted,
    )) by {
        reveal(constructed_ibm_shape);
        assert forall|rid: RequestId|
            #![trigger new_ibm.machines.contains_key(rid)]
            new_ibm.machines.contains_key(rid) implies {
                if emitted.contains_key(rid) {
                    let machine = new_ibm.machines[rid];
                    let new_state = new_e.cs.live_requests@[rid];
                    &&& machine.request_state == new_state
                    &&& machine.kv_initialized
                    &&& machine.kv_tokens
                        == (history(new_state).len() - 1) as nat
                    &&& machine.kv_cache_reprs.len()
                        == old_ibm.model_config.num_layers as nat
                } else {
                    new_ibm.machines[rid] == old_ibm.machines[rid]
                }
            }
        by {
            assert(new_ibm.machines[rid]
                == architecture_stepped_machine(
                    old_e, new_e, old_ibm, emitted, reprs, rid,
                ));
            if emitted.contains_key(rid) {
                assert(crate::exec::engine::reprs_emits(reprs, rid));
                let i = choose|i: int| 0 <= i < reprs.scheduled.len()
                    && reprs.scheduled[i] == rid
                    && reprs.sample_mask[i];
                assert(reprs.scheduled.contains(rid));
                lemma_architecture_machine_cache_after_len(
                    old_e, old_ibm, reprs, rid,
                );
            }
        }
    }
    construct_abstract_step_from_shape(
        old_e,
        new_e,
        old_ibm,
        new_ibm,
        emitted,
        samples,
        reprs,
        crate::exec::engine::architecture_step_logits_repr(old_e, reprs),
        crate::exec::engine::architecture_engine_post_kv_of(
            old_e, reprs, old_e.kv_caches_repr@,
        ),
    );
}

} // verus!
