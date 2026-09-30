// Per-request stepping abstraction. Mirrors the earlier Dafny request machine
// but recast as a pure datatype (Dafny had a heap class with mutable state;
// Verus prefers state-passing datatypes).
//
// The exec-side stepping (which threads tracked KV-cache perms and calls
// `model_forward`) lives in the engine layer; this file is structural —
// helpers, predicates, and the `RequestMachine` value type.

use crate::model_config::ModelConfig;
#[cfg(verus_only)]
pub use crate::proof::tensor::geometry::{
    slots_from, contiguous_block_ids, singleton_block_rows,
    seq_lens_for_single, synthetic_cache_reprs,
};
#[cfg(verus_only)]
use crate::proof::tensor::geometry::positions_from;
use crate::exec::request_state::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// ---------------------------------------------------------------------------
// Singleton helpers — used to feed the abstract machine's `model_forward`
// as if it were a batch of size 1.
// ---------------------------------------------------------------------------


// The machine's contiguous geometry is identity-mapped: reading logical position
// `pos` through `contiguous_block_ids` resolves to physical slot `pos`.  This
// discharges all the machine-side (B) relocation alignment hypotheses when the
// engine forward is relocated onto the machine's contiguous cache.
pub proof fn lemma_contiguous_block_table_slot(bc: nat, pos: nat)
    requires
        (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int) < bc as int,
        bc <= u64::MAX as nat,
    ensures
        crate::proof::tensor::geometry::block_table_slot(contiguous_block_ids(bc), pos) == pos,
{
    let bs = crate::types::BLOCK_SIZE_SPEC as int;
    let p = (pos as int) / bs;
    // The block id at index p is just p (contiguous), and it fits in u64.
    assert(0 <= p < bc as int);
    assert(contiguous_block_ids(bc)[p] as int == p);
    // block_table_slot = p*bs + pos%bs == pos.
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(pos as int, bs);
}


pub proof fn lemma_synthetic_cache_slot_in_cache(
    context_len: nat,
    num_layers: nat,
    layer: int,
    pos: nat,
)
    requires
        0 <= layer < num_layers as int,
        pos < context_len,
    ensures
        crate::proof::tensor::geometry::slot_in_cache(
            synthetic_cache_reprs(context_len, num_layers)[layer].0,
            pos,
        ),
        crate::proof::tensor::geometry::slot_in_cache(
            synthetic_cache_reprs(context_len, num_layers)[layer].1,
            pos,
        ),
{
    crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, context_len);
}

// Convert a Seq<TokenId> (Seq<nat>) to IntTensor1D (Seq<int>).  Tokens are
// non-negative; the cast is value-preserving.
pub open spec fn token_seq_to_int(s: Seq<TokenId>) -> IntTensor1D {
    Seq::new(s.len(), |i: int| s[i] as int)
}

// What `model_forward` would return for this machine's singleton step,
// projected to the last row's logits.  The bridge between the machine's
// structural state and the kernel-level composition.
pub open spec fn machine_step_transition(
    pre_state: RequestState,
    post_state: RequestState,
    emitted: TokenId,
) -> bool {
    request_lifecycle_view_eq(post_state, pre_state)
    && post_state.prompt_tokens@ == pre_state.prompt_tokens@
    && post_state.generated_tokens@ == pre_state.generated_tokens@.push(emitted)
}

// The COMPLETE internal machine transition also pins the next sampler state.
// Lifecycle equality remains opaque here: it is needed for executable
// engine/IBM lockstep, but is deliberately absent from cross-execution output
// determinism (`request_sampling_view_eq`).
pub open spec fn machine_step_transition_full(
    pre_state: RequestState,
    post_state: RequestState,
    next_sampler_state: SamplerState,
    emitted: TokenId,
) -> bool {
    machine_step_transition(pre_state, post_state, emitted)
    && post_state.sampler_state == next_sampler_state
}

// Local elimination boundary for the executable engine: a surviving request
// remains valid and steppable after one full machine transition.  This proof
// deliberately opens the EOS view here so large scheduler/engine contexts
// carry only the finite policy value's equality.
pub proof fn lemma_machine_step_survivor_ready(
    pre: RequestState,
    post: RequestState,
    next_sampler_state: SamplerState,
    emitted: TokenId,
)
    requires
        can_step(pre),
        machine_step_transition_full(pre, post, next_sampler_state, emitted),
        !should_finish_after_append(pre, emitted),
    ensures
        valid_request_state(post),
        !is_finished(post),
        can_step(post),
        history(post).len() > 0,
{
    lemma_request_lifecycle_view_eq_fields(post, pre);
    lemma_eos_tokens_from_policy_eq(post, pre);
    lemma_same_eos_tokens_view_eq(eos_tokens(post), eos_tokens(pre));
    assert(post.generated_tokens@.len() == pre.generated_tokens@.len() + 1);
    assert(post.generated_tokens@.len() != post.max_tokens as int);
    assert(pre.ignore_eos || !eos_tokens(pre).contains(emitted));
    assert(post.generated_tokens@.len() > 0);
    assert(post.generated_tokens@[
        post.generated_tokens@.len() - 1
    ] == emitted) by {
        let i = pre.generated_tokens@.len() as int;
        assert(post.generated_tokens@[i] == emitted);
        assert(i == post.generated_tokens@.len() - 1);
    }
}

pub proof fn lemma_machine_step_preserves_history_capacity(
    pre: RequestState,
    post: RequestState,
    next_sampler_state: SamplerState,
    emitted: TokenId,
)
    requires
        request_history_capacity_safe(pre),
        machine_step_transition_full(pre, post, next_sampler_state, emitted),
    ensures
        request_history_capacity_safe(post),
{
    lemma_request_lifecycle_view_eq_fields(post, pre);
}

// NOTE: token Vecs are related only by their `@` views, and lifecycle equality
// is carried as one opaque frame fact.  `shared_rid_coherence` therefore stays
// at the request-state view level; raw struct equality is not required.

// ---------------------------------------------------------------------------
// RequestMachine — the per-request stepping value.  Structural only here;
// the executable Step (with tracked KV perms threading) lives in the engine
// and is wired up via independent_batch_model.
// ---------------------------------------------------------------------------

pub struct RequestMachine {
    pub request_state: RequestState,
    pub kv_initialized: bool,
    pub kv_tokens: nat,
    // KV cache tensor handles + their reprs are held externally — at the
    // engine boundary — so this datatype is heap-free for refinement
    // reasoning.
    pub kv_cache_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
}

pub open spec fn request_machine_alive(m: RequestMachine, config: ModelConfig) -> bool {
    valid_request_state(m.request_state)
    && history(m.request_state).len() > 0
    && m.kv_cache_reprs.len() == config.num_layers as nat
    && (m.kv_initialized ==> m.kv_tokens + 1 == history(m.request_state).len())
    && (!m.kv_initialized ==> m.kv_tokens == 0)
}

} // verus!
