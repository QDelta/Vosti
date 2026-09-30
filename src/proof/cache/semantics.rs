//! Concrete, architecture-neutral discharge of semantic cache obligations.
//!
//! `model_cache` states the family-independent canonical-cache laws, while
//! `cache_provenance` certifies reusable physical prefixes.  This module joins
//! those laws to scheduler rows and private request-machine caches.  It must
//! not dispatch on a model family: a newly supported architecture reaches this
//! proof solely through `model_architecture::cache_refinement_supported`.

use crate::exec::engine::{Engine, StepReprs};
use crate::proof::reference::independent_batch_model::IndependentBatchModel;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// An emitted row materializes exactly the post-machine history excluding the
// newly sampled token.  This fact is scheduler/model independent and is the
// history bridge used by canonical cache extension below.
#[verifier::spinoff_prover]
pub proof fn lemma_architecture_emitted_cache_history_alignment(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
    rid: RequestId,
)
    requires
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm),
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        new_e.cs.live_requests@.contains_key(rid),
        crate::exec::engine::reprs_emits(reprs, rid),
    ensures ({
        let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
            old_e, new_e, old_ibm, emitted, reprs,
        );
        let old_machine = old_ibm.machines[rid];
        let new_machine = new_ibm.machines[rid];
        let processed = crate::proof::engine::abstract_step::architecture_machine_processed_tokens(
            old_e, reprs, rid,
        );
        let new_tokens = crate::proof::reference::request_machine::token_seq_to_int(
            crate::exec::request_state::history(new_machine.request_state),
        );
        let old_tokens = crate::proof::reference::request_machine::token_seq_to_int(
            crate::exec::request_state::history(old_machine.request_state),
        );
        let k_len = crate::proof::engine::abstract_step::architecture_machine_key_len(
            reprs, rid,
        );
        &&& old_ibm.machines.contains_key(rid)
        &&& new_ibm.machines.contains_key(rid)
        &&& new_machine.kv_tokens == k_len
        &&& old_machine.kv_tokens <= k_len
        &&& crate::proof::model::cache::histories_share_prefix(
            processed, new_tokens, k_len,
        )
        &&& crate::proof::model::cache::histories_share_prefix(
            old_tokens, new_tokens, old_machine.kv_tokens,
        )
    }),
{
    reveal(crate::proof::engine::refinement::architecture_semantic_inv);
    reveal(crate::exec::engine::architecture_engine_step_relation);
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    assert(old_e.cs.live_requests@.contains_key(rid));
    assert(old_ibm.machines.contains_key(rid));
    assert(emitted.contains_key(rid));
    assert(new_ibm.machines.contains_key(rid));
    let old_machine = old_ibm.machines[rid];
    let new_machine = new_ibm.machines[rid];
    let old_state = old_e.cs.live_requests@[rid];
    let new_state = new_e.cs.live_requests@[rid];
    let old_tokens = crate::proof::reference::request_machine::token_seq_to_int(
        crate::exec::request_state::history(old_machine.request_state),
    );
    let engine_tokens = crate::proof::reference::request_machine::token_seq_to_int(
        crate::exec::request_state::history(old_state),
    );
    let new_tokens = crate::proof::reference::request_machine::token_seq_to_int(
        crate::exec::request_state::history(new_state),
    );
    let processed = crate::proof::engine::abstract_step::architecture_machine_processed_tokens(
        old_e, reprs, rid,
    );
    let row = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
        reprs, rid,
    );
    let k_len = crate::proof::engine::abstract_step::architecture_machine_key_len(
        reprs, rid,
    );
    assert(crate::exec::request_state::request_state_view_eq(
        old_state, old_machine.request_state,
    ));
    assert(crate::exec::request_state::history(old_state)
        == crate::exec::request_state::history(old_machine.request_state));
    assert(old_tokens == engine_tokens);
    assert(crate::proof::reference::request_machine::request_machine_alive(
        old_machine, old_ibm.model_config,
    ));
    assert(crate::proof::reference::request_machine::machine_step_transition_full(
        old_state, new_state, samples[rid].0, samples[rid].1,
    ));
    reveal(crate::proof::reference::request_machine::machine_step_transition_full);
    reveal(crate::proof::reference::request_machine::machine_step_transition);
    assert(crate::exec::request_state::history(new_state).len()
        == crate::exec::request_state::history(old_state).len() + 1);
    assert(new_machine
        == crate::proof::engine::abstract_step::architecture_stepped_machine(
            old_e, new_e, old_ibm, emitted, reprs, rid,
        ));
    assert(new_machine.request_state == new_state);
    assert(new_machine.kv_tokens
        == (crate::exec::request_state::history(new_state).len() - 1) as nat);
    assert(exists|i: int| 0 <= i < reprs.scheduled.len()
        && reprs.scheduled[i] == rid);
    assert(0 <= row < reprs.scheduled.len()
        && reprs.scheduled[row] == rid);
    assert(crate::exec::engine::reprs_forward_layout_at(old_e, reprs, row));
    if old_e.cs.running@.contains(rid) {
        reveal(crate::exec::engine::reprs_forward_layout_at);
        assert(k_len == crate::exec::request_state::history(old_state).len());
        assert(processed == engine_tokens);
        assert(old_machine.kv_initialized) by {
            assert(crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm));
        }
        assert(old_machine.kv_tokens + 1 == engine_tokens.len());
        assert(old_machine.kv_tokens < k_len);
    } else {
        assert(old_e.cs.waiting@.contains(rid));
        let emitting_row = choose|i: int| 0 <= i < reprs.scheduled.len()
            && reprs.scheduled[i] == rid
            && reprs.sample_mask[i];
        assert(row == emitting_row) by {
            assert(reprs.scheduled.no_duplicates());
        }
        assert(reprs.sample_mask[row]);
        crate::exec::engine::lemma_waiting_sample_row_is_final(
            old_e, reprs, row,
        );
        assert(old_state.generated_tokens@.len() == 0) by {
            assert(crate::exec::cache_scheduler::waiting_unstarted(&old_e.cs));
        }
        assert(k_len == old_state.prompt_tokens@.len());
        assert(crate::exec::request_state::history(old_state)
            == old_state.prompt_tokens@);
        assert(processed == engine_tokens) by {
            assert(engine_tokens.subrange(0, k_len as int)
                =~= engine_tokens);
        }
        assert(!old_machine.kv_initialized) by {
            assert(crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm));
        }
        assert(old_machine.kv_tokens == 0);
    }
    assert(new_machine.kv_tokens == k_len);
    assert(processed.len() == k_len);
    assert(new_tokens.len() == k_len + 1);
    assert(processed.subrange(0, k_len as int) =~= processed);
    assert(processed.subrange(0, k_len as int)
        =~= new_tokens.subrange(0, k_len as int)) by {
        assert forall|j: int| 0 <= j < k_len as int implies
            #[trigger] processed[j] == new_tokens[j]
        by {
            let prompt_len = old_state.prompt_tokens@.len();
            assert(processed[j] == engine_tokens[j]);
            if j < prompt_len {
                assert(engine_tokens[j] == old_state.prompt_tokens@[j] as int);
                assert(new_tokens[j] == new_state.prompt_tokens@[j] as int);
            } else {
                let generated_index = j - prompt_len;
                assert(0 <= generated_index
                    < old_state.generated_tokens@.len());
                assert(engine_tokens[j]
                    == old_state.generated_tokens@[generated_index] as int);
                assert(new_tokens[j]
                    == new_state.generated_tokens@[generated_index] as int);
            }
        }
    }
    assert(crate::proof::model::cache::histories_share_prefix(
        processed, new_tokens, k_len,
    )) by {
        reveal(crate::proof::model::cache::histories_share_prefix);
    }
    assert(old_machine.kv_tokens <= k_len);
    assert(old_tokens.subrange(0, old_machine.kv_tokens as int)
        == new_tokens.subrange(0, old_machine.kv_tokens as int)) by {
        assert forall|j: int| 0 <= j < old_machine.kv_tokens as int
            implies old_tokens[j] == new_tokens[j]
        by {
            assert(old_tokens[j] == processed[j]);
            assert(processed[j] == new_tokens[j]);
        }
    }
    assert(crate::proof::model::cache::histories_share_prefix(
        old_tokens, new_tokens, old_machine.kv_tokens,
    )) by {
        reveal(crate::proof::model::cache::histories_share_prefix);
    }
}

// A shaped private cache has every logical position covered by the finite
// page budget.  Keeping this arithmetic lemma here avoids repeating physical
// page reasoning in each semantic row case.
pub proof fn lemma_machine_cache_has_position(
    ibm: &IndependentBatchModel,
    num_blocks: nat,
    rid: RequestId,
    layer: int,
    context_len: nat,
    pos: nat,
)
    requires
        crate::proof::engine::refinement::mach_cache_shape_ok(ibm, num_blocks),
        ibm.machines.contains_key(rid),
        0 <= layer < ibm.machines[rid].kv_cache_reprs.len(),
        crate::proof::tensor::geometry::blocks_needed_for(context_len) <= num_blocks,
        pos < context_len,
    ensures
        crate::proof::tensor::geometry::slot_in_cache(
            ibm.machines[rid].kv_cache_reprs[layer].0, pos,
        ),
        crate::proof::tensor::geometry::slot_in_cache(
            ibm.machines[rid].kv_cache_reprs[layer].1, pos,
        ),
{
    let page = (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int);
    let offset = (pos as int) % (crate::types::BLOCK_SIZE_SPEC as int);
    crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, context_len);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
        pos as int, crate::types::BLOCK_SIZE_SPEC as int,
    );
    vstd::arithmetic::div_mod::lemma_mod_bound(
        pos as int, crate::types::BLOCK_SIZE_SPEC as int,
    );
    assert(0 <= page < num_blocks as int);
    assert(0 <= offset < crate::types::BLOCK_SIZE_SPEC as int);
}

// Architecture-neutral physical frame for persistent provenance: a cache cell
// outside the shared scatter-slot sequence is unchanged by the selected
// model's forward.  Family-specific row construction remains behind the
// `model_architecture` dispatch boundary.
pub proof fn lemma_architecture_engine_step_preserves_unwritten_cache_cell(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
    layer: nat,
    slot: nat,
)
    requires
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        layer < reprs.wr.layers.len(),
        old_e.kv_caches_repr@.len() >= reprs.wr.layers.len(),
        !reprs.slots.contains(slot as int),
        crate::proof::tensor::geometry::slot_in_cache(
            old_e.kv_caches_repr@[layer as int].0, slot,
        ),
        crate::proof::tensor::geometry::slot_in_cache(
            old_e.kv_caches_repr@[layer as int].1, slot,
        ),
    ensures
        crate::proof::tensor::geometry::slot_in_cache(
            new_e.kv_caches_repr@[layer as int].0, slot,
        ),
        crate::proof::tensor::geometry::slot_in_cache(
            new_e.kv_caches_repr@[layer as int].1, slot,
        ),
        crate::proof::tensor::geometry::cache_at(
            new_e.kv_caches_repr@[layer as int].0, slot,
        ) == crate::proof::tensor::geometry::cache_at(
            old_e.kv_caches_repr@[layer as int].0, slot,
        ),
        crate::proof::tensor::geometry::cache_at(
            new_e.kv_caches_repr@[layer as int].1, slot,
        ) == crate::proof::tensor::geometry::cache_at(
            old_e.kv_caches_repr@[layer as int].1, slot,
        ),
{
    reveal(crate::proof::engine::refinement::architecture_semantic_inv);
    reveal(crate::exec::engine::architecture_engine_step_relation);
    assert(reprs.wr == old_ibm.wr);
    crate::proof::reference::independent_batch_model::lemma_ibm_valid_semantic_model(old_ibm);
    assert(crate::proof::model::types::model_weights_architecture_repr_valid(
        reprs.wr, old_ibm.architecture_repr,
    ));
    crate::proof::model::architecture::lemma_model_forward_kv_preserves_unwritten_slot(
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
        layer,
        slot,
    );
    assert(new_e.kv_caches_repr@
        == crate::exec::engine::architecture_engine_post_kv_of(
            old_e, reprs, old_e.kv_caches_repr@,
        ));
    reveal(crate::exec::engine::architecture_engine_post_kv_of);
}

// Lift one executed row's canonical private forward to any post-step
// provenance chain reaching the same published physical page.  Registered
// ancestry identifies the common page index, while exact token placement
// shows that the two logical requests share the history through that page.
#[verifier::spinoff_prover]
#[verifier::rlimit(500)]
pub proof fn lemma_architecture_published_row_origin_cell_is_canonical(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
    chain: Seq<BlockId>,
    request_tokens: Seq<TokenId>,
    executed_tokens: Seq<TokenId>,
    c_tokens: nat,
    k: int,
    l: int,
    layer: nat,
    pos: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm),
        crate::proof::cache::provenance::registered_cache_fidelity(
            &old_e,
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
        ),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::proof::engine::refinement::mach_cache_shape_ok(
            &old_ibm, old_e.cs.num_blocks as nat,
        ),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        crate::proof::cache::provenance::provenance_prefix_candidate(
            &new_e.cs, chain, request_tokens, c_tokens,
        ),
        0 <= k < reprs.scheduled.len(),
        k < reprs.bt.len(),
        0 <= l < reprs.bt[k].len(),
        ((pos / crate::types::BLOCK_SIZE_SPEC) as int) < chain.len(),
        old_e.cs.live_requests@.contains_key(reprs.scheduled[k]),
        ({
            let end = reprs.cu_k[k + 1] - reprs.cu_k[k];
            let full = end / (crate::types::BLOCK_SIZE_SPEC as int);
            &&& 0 <= full <= reprs.bt[k].len()
            &&& 0 <= end <= executed_tokens.len()
            &&& executed_tokens.len() <= u64::MAX as int
            &&& crate::proof::tensor::geometry::blocks_needed_for(end as nat) <= u64::MAX as nat
            &&& 0 <= l < end / (crate::types::BLOCK_SIZE_SPEC as int)
            &&& crate::proof::reference::request_machine::token_seq_to_int(executed_tokens).subrange(0, end)
                == crate::proof::engine::abstract_step::architecture_machine_processed_tokens(
                    old_e, reprs, reprs.scheduled[k],
                )
            &&& crate::exec::cache_scheduler::registered_prefix_chain(
                new_e.cs.blocks@, reprs.bt[k].subrange(0, full),
            )
            &&& crate::exec::cache_scheduler::token_placement_prefix(
                new_e.cs.blocks@, reprs.bt[k], executed_tokens,
                full * (crate::types::BLOCK_SIZE_SPEC as int),
            )
        }),
        reprs.bt[k][l]
            == chain[(pos / crate::types::BLOCK_SIZE_SPEC) as int],
        layer < reprs.wr.layers.len(),
        pos < c_tokens,
    ensures
        crate::proof::cache::provenance::registered_cache_cell_is_canonical(
            &new_e,
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
            chain,
            request_tokens,
            layer as int,
            pos,
        ),
{
    let bs = crate::types::BLOCK_SIZE_SPEC;
    let pages = c_tokens / bs;
    let j = (pos / bs) as int;
    let rid = reprs.scheduled[k];
    let row_tokens = executed_tokens;
    let row_history = crate::proof::reference::request_machine::token_seq_to_int(executed_tokens);
    let request = crate::proof::reference::request_machine::token_seq_to_int(request_tokens);
    let end = reprs.cu_k[k + 1] - reprs.cu_k[k];
    let full = end / (bs as int);
    let left = chain.subrange(0, pages as int);
    let right = reprs.bt[k].subrange(0, full);
    let upto = ((j + 1) * (bs as int)) as nat;
    let model = crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm);

    reveal(crate::proof::cache::provenance::provenance_prefix_candidate);
    assert(0 <= j < pages as int) by {
        vstd::arithmetic::div_mod::lemma_div_is_ordered(
            pos as int, c_tokens as int, bs as int,
        );
    }
    reveal(crate::exec::engine::architecture_engine_step_relation);
    assert(crate::exec::cache_scheduler::registered_prefix_chain(
        new_e.cs.blocks@, right,
    ));
    assert(crate::exec::cache_scheduler::registered_prefix_chain(
        new_e.cs.blocks@, left,
    ));
    assert(left[j] == chain[j]);
    assert(right[l] == reprs.bt[k][l]);
    crate::exec::cache_scheduler::lemma_registered_chains_match_through_target(
        new_e.cs.blocks@, left, right, j, l,
    );
    assert(j == l);
    assert(left.subrange(0, j + 1)
        == right.subrange(0, j + 1));
    assert(upto <= c_tokens) by {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
            c_tokens as int, bs as int,
        );
    }
    assert(upto <= row_tokens.len()) by {
        assert(j < full);
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
            end, bs as int,
        );
        assert(upto <= end);
    }
    assert(request_tokens.subrange(0, upto as int)
        =~= row_tokens.subrange(0, upto as int)) by {
        assert forall|p: int| 0 <= p < upto as int implies
            request_tokens.subrange(0, upto as int)[p]
                == row_tokens.subrange(0, upto as int)[p]
        by {
            crate::exec::cache_scheduler::lemma_token_placement_prefix_at(
                new_e.cs.blocks@, chain, request_tokens,
                c_tokens as int, p,
            );
            crate::exec::cache_scheduler::lemma_token_placement_prefix_at(
                new_e.cs.blocks@, reprs.bt[k], row_tokens,
                full * (bs as int), p,
            );
            reveal(crate::exec::cache_scheduler::token_placement_at);
            let page = p / (bs as int);
            assert(chain[page] == reprs.bt[k][page]) by {
                assert(left.subrange(0, j + 1)[page]
                    == right.subrange(0, j + 1)[page]);
            }
        }
    }
    assert(crate::proof::model::cache::histories_share_prefix(
        request, row_history, upto,
    )) by {
        reveal(crate::proof::model::cache::histories_share_prefix);
        assert(request.subrange(0, upto as int)
            =~= row_history.subrange(0, upto as int)) by {
            assert forall|p: int| 0 <= p < upto as int implies
                request[p] == row_history[p]
            by {
                assert(request[p] == request_tokens[p] as int);
                assert(row_history[p] == row_tokens[p] as int);
                assert(request_tokens.subrange(0, upto as int)[p]
                    == row_tokens.subrange(0, upto as int)[p]);
            }
        }
    }

    assert(reprs.scheduled.contains(rid));
    assert(crate::exec::engine::reprs_forward_layout_at(old_e, reprs, k));
    let row = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
        reprs, rid,
    );
    assert(row == k) by {
        assert(reprs.scheduled.no_duplicates());
    }
    let processed = crate::proof::engine::abstract_step::architecture_machine_processed_tokens(
        old_e, reprs, rid,
    );
    let prefix_len = crate::proof::engine::abstract_step::architecture_machine_prefix_len(
        reprs, rid,
    );
    let base = crate::proof::engine::abstract_step::architecture_machine_cache_base(
        old_e, old_ibm, reprs, rid,
    );
    let private = crate::proof::engine::abstract_step::architecture_machine_cache_after(
        old_e, old_ibm, reprs, rid,
    );
    assert(processed == row_history.subrange(0, end));
    derive_architecture_machine_canonical_prefix_forward_ready(
        old_e, old_ibm, reprs, rid,
    );
    crate::proof::model::cache::lemma_canonical_prefix_forward_cache_fidelity(
        model, processed, prefix_len, base,
    );
    crate::proof::engine::abstract_step::lemma_architecture_machine_forward_is_prefix_continuation(
        old_e, old_ibm, reprs, rid,
    );
    assert(crate::proof::model::cache::cache_reprs_match_canonical(
        private, model, processed, end as nat,
    ));

    crate::proof::cache::coherence::derive_architecture_request_projection_common_domain(
        old_e, old_ibm, reprs, rid,
    );
    crate::proof::model::architecture::lemma_model_forward_request_projection_domain_from_common(
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
        k as nat,
    );
    crate::proof::cache::coherence::derive_architecture_forward_relocation_ready_from_projection(
        old_e, old_ibm, reprs, rid,
    );
    crate::proof::engine::abstract_step::lemma_architecture_machine_forward_matches_engine(
        old_e, old_ibm, reprs, rid,
    );

    let machine_bt = crate::proof::reference::request_machine::singleton_block_rows(end as nat);
    assert(machine_bt.len() == 1);
    assert(crate::proof::model::family_layout::cache_sequence_logical_prefix_equal(
        crate::exec::engine::architecture_engine_post_kv_of(
            old_e, reprs, old_e.kv_caches_repr@,
        ),
        reprs.bt[k],
        private,
        machine_bt[0],
        model.weights.layers.len(),
        end as nat,
    ));
    assert(new_e.kv_caches_repr@
        == crate::exec::engine::architecture_engine_post_kv_of(
            old_e, reprs, old_e.kv_caches_repr@,
        ));
    assert(pos < end as nat) by {
        assert(pos < upto);
        assert(upto <= end);
    }
    crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, end as nat);
    assert(crate::proof::tensor::geometry::blocks_needed_for(end as nat)
        <= u64::MAX as nat);
    crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(
        crate::proof::tensor::geometry::blocks_needed_for(end as nat), pos,
    );
    assert(crate::proof::tensor::geometry::block_table_slot(machine_bt[0], pos) == pos);
    assert(crate::proof::tensor::geometry::block_table_slot(chain, pos)
        == crate::proof::tensor::geometry::block_table_slot(reprs.bt[k], pos)) by {
        assert(chain[j] == reprs.bt[k][j]);
    }
    reveal(crate::proof::model::family_layout::cache_sequence_logical_prefix_equal);
    reveal(crate::proof::model::family_layout::cache_pair_logical_prefix_equal);
    assert(crate::proof::model::family_layout::cache_pair_logical_prefix_equal(
        new_e.kv_caches_repr@[layer as int],
        reprs.bt[k],
        private[layer as int],
        machine_bt[0],
        end as nat,
    ));
    assert(crate::proof::tensor::geometry::slot_in_cache(
        new_e.kv_caches_repr@[layer as int].0,
        crate::proof::tensor::geometry::block_table_slot(reprs.bt[k], pos),
    ));
    assert(crate::proof::tensor::geometry::slot_in_cache(
        new_e.kv_caches_repr@[layer as int].1,
        crate::proof::tensor::geometry::block_table_slot(reprs.bt[k], pos),
    ));
    assert(crate::proof::tensor::geometry::slot_in_cache(
        new_e.kv_caches_repr@[layer as int].0,
        crate::proof::tensor::geometry::block_table_slot(chain, pos),
    ));
    assert(crate::proof::tensor::geometry::slot_in_cache(
        new_e.kv_caches_repr@[layer as int].1,
        crate::proof::tensor::geometry::block_table_slot(chain, pos),
    ));
    assert(crate::proof::model::cache::cache_pair_has_position(
        private, layer as int, pos,
    ));
    assert(crate::proof::model::cache::canonical_kv_defined(
        model, processed, layer as int, pos,
    ));
    assert(processed.subrange(0, pos as int + 1)
        =~= row_history.subrange(0, pos as int + 1));
    assert(request.subrange(0, pos as int + 1)
        =~= processed.subrange(0, pos as int + 1)) by {
        assert forall|p: int| 0 <= p < pos as int + 1 implies
            request[p] == processed[p]
        by {
            assert(request.subrange(0, upto as int)[p]
                == row_history.subrange(0, upto as int)[p]);
            assert(processed[p] == row_history[p]);
        }
    }
    assert(crate::proof::model::cache::canonical_kv_defined(
        model, request, layer as int, pos,
    ));
    assert(crate::proof::model::cache::histories_share_prefix(
        request, processed, upto,
    )) by {
        reveal(crate::proof::model::cache::histories_share_prefix);
        assert(request.subrange(0, upto as int)
            =~= processed.subrange(0, upto as int)) by {
            assert forall|p: int| 0 <= p < upto as int implies
                request[p] == processed[p]
            by {
                assert(request.subrange(0, upto as int)[p]
                    == row_history.subrange(0, upto as int)[p]);
                assert(processed[p] == row_history[p]);
            }
        }
    }
    crate::proof::model::cache::lemma_canonical_kv_respects_shared_prefix(
        model, request, processed, upto, layer as int, pos,
    );
    reveal(crate::proof::cache::provenance::registered_cache_cell_is_canonical);
    assert((
        crate::proof::tensor::geometry::cache_at(
            new_e.kv_caches_repr@[layer as int].0,
            crate::proof::tensor::geometry::block_table_slot(chain, pos),
        ),
        crate::proof::tensor::geometry::cache_at(
            new_e.kv_caches_repr@[layer as int].1,
            crate::proof::tensor::geometry::block_table_slot(chain, pos),
        ),
    ) == crate::proof::model::cache::cache_pair_at(
        private, layer as int, pos,
    ));
}

// The pre-existing-page branch of per-cell preservation.  Keeping the
// admission-origin disjunction out of this solver context makes the large
// unchanged-page argument stable under unrelated changes to the module graph.
#[verifier::spinoff_prover]
#[verifier::rlimit(600)]
proof fn lemma_architecture_preexisting_origin_cell_is_canonical(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
    chain: Seq<BlockId>,
    request_tokens: Seq<TokenId>,
    c_tokens: nat,
    layer: int,
    pos: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm),
        crate::proof::cache::provenance::provenance_cache_fidelity(
            &old_e,
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
        ),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::proof::engine::refinement::mach_cache_shape_ok(
            &old_ibm, old_e.cs.num_blocks as nat,
        ),
        crate::exec::cache_scheduler::tail_write_exclusive(&old_e.cs),
        crate::exec::cache_scheduler::residency_history_aligned(&old_e.cs),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        crate::proof::cache::provenance::provenance_prefix_candidate(
            &new_e.cs, chain, request_tokens, c_tokens,
        ),
        0 <= layer < crate::proof::reference::independent_batch_model::ibm_semantic_model(
            old_ibm,
        ).weights.layers.len(),
        pos < c_tokens,
        !(exists|k: int, l: int|
            0 <= k < reprs.scheduled.len()
            && k < reprs.bt.len()
            && !old_e.cs.running@.contains(reprs.scheduled[k])
            && old_e.cs.live_requests@.contains_key(reprs.scheduled[k])
            && 0 <= reprs.cu_k[k + 1] - reprs.cu_k[k]
                <= old_e.cs.live_requests@[reprs.scheduled[k]]
                    .prompt_tokens@.len()
            && 0 <= l < (reprs.cu_k[k + 1] - reprs.cu_k[k])
                / (crate::types::BLOCK_SIZE_SPEC as int)
            && #[trigger] reprs.bt[k][l]
                == chain[(pos / crate::types::BLOCK_SIZE_SPEC) as int]),
        forall|x: int|
            0 <= x <= (pos / crate::types::BLOCK_SIZE_SPEC) as int ==>
                #[trigger] crate::exec::cache_scheduler::positive_page_unchanged_from_pre(
                    &old_e.cs, &new_e.cs, chain[x],
                ),
    ensures
        crate::proof::cache::provenance::registered_cache_cell_is_canonical(
            &new_e,
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
            chain,
            request_tokens,
            layer,
            pos,
        ),
{
    let model = crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm);
    crate::proof::cache::provenance::lemma_provenance_fidelity_implies_registered_fidelity(
        &old_e, model,
    );
    reveal(crate::proof::engine::refinement::architecture_semantic_inv);
    reveal(crate::exec::engine::architecture_engine_step_relation);
    assert(reprs.wr == old_ibm.wr);
    assert(old_e.kv_caches_repr@.len() >= reprs.wr.layers.len());
    let bs = crate::types::BLOCK_SIZE_SPEC;
    let pages = c_tokens / bs;
    let j = (pos / bs) as int;
    let upto = ((j + 1) * (bs as int)) as nat;
    let prefix = chain.subrange(0, pages as int);
    let left = chain.subrange(0, j + 1);
    reveal(crate::proof::cache::provenance::provenance_prefix_candidate);
    assert(0 <= j < pages as int) by {
        vstd::arithmetic::div_mod::lemma_div_is_ordered(
            pos as int, c_tokens as int, bs as int,
        );
    }
    assert(crate::exec::cache_scheduler::registered_prefix_chain(
        new_e.cs.blocks@, prefix,
    ));
    assert(prefix[j] == chain[j]);
    crate::exec::cache_scheduler::lemma_registered_prefix_chain_prefix(
        new_e.cs.blocks@, prefix, j + 1,
    );
    assert(left =~= prefix.subrange(0, j + 1)) by {
        assert forall|x: int| 0 <= x < left.len() implies
            left[x] == prefix.subrange(0, j + 1)[x]
        by {
            assert(left[x] == chain[x]);
            assert(prefix.subrange(0, j + 1)[x] == prefix[x]);
            assert(prefix[x] == chain[x]);
        }
    }
    assert(crate::exec::cache_scheduler::registered_prefix_chain(
        new_e.cs.blocks@, left,
    ));
    let slot = crate::proof::tensor::geometry::block_table_slot(chain, pos);
    assert(!reprs.slots.contains(slot as int)) by {
        if reprs.slots.contains(slot as int) {
            let q = reprs.slots.index_of(slot as int);
            let nrows = reprs.scheduled.len() as int;
            assert(crate::exec::engine::step_reprs_wf(old_e, reprs));
            assert(reprs.cu_q[nrows] == reprs.slots.len() as int);
            assert(nrows >= 1) by {
                if nrows == 0 {
                    assert(reprs.cu_q[0] == 0);
                }
            }
            let m = crate::proof::tensor::geometry::lemma_cu_locate(
                reprs.cu_q, nrows, q,
            );
            assert(crate::exec::engine::reprs_forward_layout_at(
                old_e, reprs, m,
            ));
            crate::proof::tensor::geometry::block_table_slot_block(chain, pos);
            assert(reprs.slots[q] / (bs as int)
                == chain[j] as int);
            let rid = reprs.scheduled[m];
            if old_e.cs.running@.contains(rid) {
                reveal(crate::exec::engine::reprs_forward_layout_at);
                let hist = crate::exec::request_state::history(
                    old_e.cs.live_requests@[rid],
                );
                let ids = old_e.cs.request_residency@[rid].block_ids@;
                let tail = ids[ids.len() - 1];
                assert(reprs.bt[m] == ids);
                assert(reprs.slots[q] as nat
                    == crate::proof::tensor::geometry::block_table_slot(
                        ids, (hist.len() - 1) as nat,
                    ));
                crate::proof::tensor::geometry::block_table_slot_block(
                    ids, (hist.len() - 1) as nat,
                );
                assert((hist.len() as int - 1) / (bs as int)
                    == ids.len() - 1) by {
                    let tail_len = old_e.cs.blocks@[tail]
                        .tokens@.len() as int;
                    assert(hist.len() as int
                        == (ids.len() - 1) * (bs as int) + tail_len);
                    assert(1 <= tail_len <= bs as int);
                    vstd::arithmetic::div_mod::
                        lemma_fundamental_div_mod_converse_div(
                            hist.len() as int - 1,
                            bs as int,
                            ids.len() - 1,
                            tail_len - 1,
                        );
                }
                assert(chain[j] == tail);
                reveal(crate::exec::cache_scheduler::tail_write_exclusive);
                assert(old_e.cs.blocks@[tail].prefix_depth == 0);
                assert(crate::exec::cache_scheduler::positive_page_unchanged_from_pre(
                    &old_e.cs, &new_e.cs, chain[j],
                ));
                assert(new_e.cs.blocks@[chain[j]].prefix_depth > 0);
                assert(false);
            } else {
                reveal(crate::exec::engine::reprs_forward_layout_at);
                let s0 = reprs.cu_q[m];
                let s1 = reprs.cu_q[m + 1];
                let end = reprs.cu_k[m + 1] - reprs.cu_k[m];
                let c = end - (s1 - s0);
                let p = c + q - s0;
                let lp = p / (bs as int);
                assert(reprs.input_ids[q]
                    == old_e.cs.live_requests@[rid]
                        .prompt_tokens@[p] as int);
                assert(0 <= p < end);
                assert(reprs.slots[q] as nat
                    == crate::proof::tensor::geometry::block_table_slot(
                        reprs.bt[m], p as nat,
                    ));
                crate::proof::tensor::geometry::block_table_slot_block(
                    reprs.bt[m], p as nat,
                );
                assert(reprs.bt[m][lp] == chain[j]);
                assert(crate::exec::cache_scheduler::published_admission_row_prefixes(
                    &old_e.cs,
                    &new_e.cs,
                    reprs.scheduled,
                    reprs.bt,
                    reprs.cu_k,
                ));
                reveal(crate::exec::cache_scheduler::published_admission_row_prefixes);
                let full = end / (bs as int);
                if lp < full {
                    assert(exists|k: int, l: int|
                        0 <= k < reprs.scheduled.len()
                        && k < reprs.bt.len()
                        && !old_e.cs.running@.contains(
                            reprs.scheduled[k])
                        && old_e.cs.live_requests@.contains_key(
                            reprs.scheduled[k])
                        && 0 <= reprs.cu_k[k + 1] - reprs.cu_k[k]
                            <= old_e.cs.live_requests@[
                                reprs.scheduled[k]].prompt_tokens@.len()
                        && 0 <= l < (reprs.cu_k[k + 1]
                            - reprs.cu_k[k]) / (bs as int)
                        && #[trigger] reprs.bt[k][l] == chain[j]) by {
                        assert(!old_e.cs.running@.contains(
                            reprs.scheduled[m]));
                    }
                    assert(false);
                } else {
                    assert(lp < reprs.bt[m].len());
                    assert(new_e.cs.blocks@.contains_key(chain[j]));
                    assert(new_e.cs.blocks@[reprs.bt[m][lp]]
                        .prefix_depth == 0);
                    assert(new_e.cs.blocks@[chain[j]].prefix_depth > 0);
                    assert(false);
                }
            }
        }
    }

    assert forall|x: int| 0 <= x < left.len() implies {
        let bid = #[trigger] left[x];
        &&& old_e.cs.blocks@.contains_key(bid)
        &&& old_e.cs.blocks@[bid].prefix_depth
            == new_e.cs.blocks@[bid].prefix_depth
        &&& old_e.cs.blocks@[bid].parent_block
            == new_e.cs.blocks@[bid].parent_block
    } by {
        assert(left[x] == chain[x]);
        assert(crate::exec::cache_scheduler::positive_page_unchanged_from_pre(
            &old_e.cs, &new_e.cs, chain[x],
        ));
    }
    crate::exec::cache_scheduler::lemma_registered_prefix_chain_transfer(
        new_e.cs.blocks@, old_e.cs.blocks@, left,
    );
    assert(upto <= request_tokens.len()) by {
        assert(upto <= c_tokens) by {
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
                c_tokens as int, bs as int,
            );
        }
    }
    assert(crate::exec::cache_scheduler::token_placement_prefix(
        new_e.cs.blocks@, left, request_tokens, upto as int,
    )) by {
        assert forall|p: int|
            #![trigger crate::exec::cache_scheduler::token_placement_at(
                new_e.cs.blocks@, left, request_tokens, p,
            )]
            0 <= p < upto as int implies
            crate::exec::cache_scheduler::token_placement_at(
                new_e.cs.blocks@, left, request_tokens, p,
            )
        by {
            crate::exec::cache_scheduler::lemma_token_placement_prefix_at(
                new_e.cs.blocks@,
                chain,
                request_tokens,
                c_tokens as int,
                p,
            );
            reveal(crate::exec::cache_scheduler::token_placement_at);
            let x = p / (bs as int);
            assert(left[x] == chain[x]);
        }
    }
    assert forall|x: int| 0 <= x < left.len()
        && #[trigger] new_e.cs.blocks@.contains_key(left[x])
        implies old_e.cs.blocks@.contains_key(left[x])
            && old_e.cs.blocks@[left[x]].tokens@
                == new_e.cs.blocks@[left[x]].tokens@
    by {
        assert(left[x] == chain[x]);
        assert(crate::exec::cache_scheduler::positive_page_unchanged_from_pre(
            &old_e.cs, &new_e.cs, chain[x],
        ));
    }
    crate::exec::cache_scheduler::lemma_token_placement_prefix_transfer_for_ids(
        new_e.cs.blocks@,
        old_e.cs.blocks@,
        left,
        request_tokens,
        upto as int,
    );
    assert(crate::proof::cache::provenance::provenance_prefix_candidate(
        &old_e.cs, left, request_tokens, upto,
    )) by {
        reveal(crate::proof::cache::provenance::provenance_prefix_candidate);
        assert(upto / bs == (j + 1) as nat) by {
            vstd::arithmetic::div_mod::
                lemma_fundamental_div_mod_converse_div(
                    upto as int, bs as int, j + 1, 0,
                );
        }
    }
    assert(crate::proof::cache::provenance::registered_cache_cell_is_canonical(
        &old_e, model, left, request_tokens, layer, pos,
    ));
    reveal(crate::proof::cache::provenance::registered_cache_cell_is_canonical);
    assert(crate::proof::tensor::geometry::block_table_slot(left, pos) == slot) by {
        assert(left[j] == chain[j]);
    }
    lemma_architecture_engine_step_preserves_unwritten_cache_cell(
        old_e,
        new_e,
        old_ibm,
        emitted,
        samples,
        reprs,
        layer as nat,
        slot,
    );
}

// Per-cell preservation for the architecture-neutral physical provenance
// certificate. A post-step positive page is published by an executed prefill
// or decode row, or is a pre-existing page unchanged by the scheduler step.
#[verifier::spinoff_prover]
#[verifier::rlimit(100)]
pub proof fn lemma_architecture_provenance_cache_cell_preserved(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
    chain: Seq<BlockId>,
    request_tokens: Seq<TokenId>,
    c_tokens: nat,
    layer: int,
    pos: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm),
        crate::proof::cache::provenance::provenance_cache_fidelity(
            &old_e,
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
        ),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::proof::engine::refinement::mach_cache_shape_ok(
            &old_ibm, old_e.cs.num_blocks as nat,
        ),
        crate::exec::cache_scheduler::tail_write_exclusive(&old_e.cs),
        crate::exec::cache_scheduler::residency_history_aligned(&old_e.cs),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        crate::proof::cache::provenance::provenance_prefix_candidate(
            &new_e.cs, chain, request_tokens, c_tokens,
        ),
        0 <= layer < crate::proof::reference::independent_batch_model::ibm_semantic_model(
            old_ibm,
        ).weights.layers.len(),
        pos < c_tokens,
    ensures
        crate::proof::cache::provenance::registered_cache_cell_is_canonical(
            &new_e,
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
            chain,
            request_tokens,
            layer,
            pos,
        ),
{
    let model = crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm);
    crate::proof::cache::provenance::lemma_provenance_fidelity_implies_registered_fidelity(
        &old_e, model,
    );
    let bs = crate::types::BLOCK_SIZE_SPEC;
    let pages = c_tokens / bs;
    let j = (pos / bs) as int;
    let prefix = chain.subrange(0, pages as int);
    reveal(crate::proof::cache::provenance::provenance_prefix_candidate);
    assert(0 <= j < pages as int) by {
        vstd::arithmetic::div_mod::lemma_div_is_ordered(
            pos as int, c_tokens as int, bs as int,
        );
    }
    assert(prefix[j] == chain[j]);
    reveal(crate::exec::engine::architecture_engine_step_relation);
    assert(crate::exec::cache_scheduler::positive_chains_from_pre_or_executed_rows(
        &old_e.cs, &new_e.cs, reprs.scheduled, reprs.bt, reprs.cu_k,
    ));
    reveal(crate::exec::cache_scheduler::positive_chains_from_pre_or_executed_rows);
    if exists|k: int, l: int|
        0 <= k < reprs.scheduled.len()
        && k < reprs.bt.len()
        && !old_e.cs.running@.contains(reprs.scheduled[k])
        && old_e.cs.live_requests@.contains_key(reprs.scheduled[k])
        && 0 <= reprs.cu_k[k + 1] - reprs.cu_k[k]
            <= old_e.cs.live_requests@[reprs.scheduled[k]].prompt_tokens@.len()
        && 0 <= l < (reprs.cu_k[k + 1] - reprs.cu_k[k]) / (bs as int)
        && #[trigger] reprs.bt[k][l] == chain[j]
    {
        let k = choose|k: int| exists|l: int|
            0 <= k < reprs.scheduled.len()
            && k < reprs.bt.len()
            && !old_e.cs.running@.contains(#[trigger] reprs.scheduled[k])
            && old_e.cs.live_requests@.contains_key(reprs.scheduled[k])
            && 0 <= reprs.cu_k[k + 1] - reprs.cu_k[k]
                <= old_e.cs.live_requests@[reprs.scheduled[k]]
                    .prompt_tokens@.len()
            && 0 <= l < (reprs.cu_k[k + 1] - reprs.cu_k[k]) / (bs as int)
            && #[trigger] reprs.bt[k][l] == chain[j];
        let l = choose|l: int|
            0 <= k < reprs.scheduled.len()
            && k < reprs.bt.len()
            && !old_e.cs.running@.contains(reprs.scheduled[k])
            && old_e.cs.live_requests@.contains_key(reprs.scheduled[k])
            && 0 <= reprs.cu_k[k + 1] - reprs.cu_k[k]
                <= old_e.cs.live_requests@[reprs.scheduled[k]]
                    .prompt_tokens@.len()
            && 0 <= l < (reprs.cu_k[k + 1] - reprs.cu_k[k]) / (bs as int)
            && #[trigger] reprs.bt[k][l] == chain[j];
        assert(crate::exec::engine::reprs_forward_layout_at(old_e, reprs, k));
        reveal(crate::exec::engine::reprs_forward_layout_at);
        assert(0 <= l < reprs.bt[k].len());
        assert(j < chain.len());
        assert(crate::exec::cache_scheduler::published_admission_row_prefixes(
            &old_e.cs, &new_e.cs, reprs.scheduled, reprs.bt, reprs.cu_k,
        ));
        reveal(crate::exec::cache_scheduler::published_admission_row_prefixes);
        let row = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
            reprs, reprs.scheduled[k],
        );
        assert(row == k) by { assert(reprs.scheduled.no_duplicates()); }
        lemma_architecture_published_row_origin_cell_is_canonical(
            old_e,
            new_e,
            old_ibm,
            emitted,
            samples,
            reprs,
            chain,
            request_tokens,
            old_e.cs.live_requests@[reprs.scheduled[k]].prompt_tokens@,
            c_tokens,
            k,
            l,
            layer as nat,
            pos,
        );
    } else if crate::exec::cache_scheduler::decode_page_origin(
        &old_e.cs, &new_e.cs, reprs.scheduled, reprs.bt, reprs.cu_k, chain[j],
    ) {
        reveal(crate::exec::cache_scheduler::decode_page_origin);
        let k = choose|k: int|
            crate::exec::cache_scheduler::executed_decode_row(
                &old_e.cs, &new_e.cs, reprs.scheduled, reprs.bt, reprs.cu_k, k,
            ) && reprs.bt[k][reprs.bt[k].len() - 1] == chain[j];
        let tokens = crate::exec::request_state::history(old_e.cs.live_requests@[reprs.scheduled[k]]);
        let end = reprs.cu_k[k + 1] - reprs.cu_k[k];
        let full = end / (bs as int);
        assert(full == reprs.bt[k].len()) by (nonlinear_arith)
            requires end == reprs.bt[k].len() * 64, bs == 64,
                full == end / (bs as int), {}
        assert(reprs.bt[k].subrange(0, full) =~= reprs.bt[k]);
        let row = crate::proof::engine::abstract_step::architecture_scheduled_index_of(reprs, reprs.scheduled[k]);
        assert(row == k) by { assert(reprs.scheduled.no_duplicates()); }
        assert(crate::proof::tensor::geometry::blocks_needed_for(end as nat) == full) by {
            crate::proof::tensor::geometry::lemma_blocks_needed_for_full_pages(reprs.bt[k].len());
        }
        assert(crate::proof::reference::request_machine::token_seq_to_int(tokens).subrange(0, end)
            =~= crate::proof::reference::request_machine::token_seq_to_int(tokens));
        assert(crate::proof::engine::abstract_step::architecture_machine_processed_tokens(
            old_e, reprs, reprs.scheduled[k],
        ) == crate::proof::reference::request_machine::token_seq_to_int(tokens));
        assert(crate::exec::cache_scheduler::token_placement_prefix(
            new_e.cs.blocks@, reprs.bt[k], tokens, full * (bs as int),
        ));
        lemma_architecture_published_row_origin_cell_is_canonical(
            old_e, new_e, old_ibm, emitted, samples, reprs, chain,
            request_tokens, tokens, c_tokens, k, reprs.bt[k].len() - 1,
            layer as nat, pos,
        );
    } else {
        assert forall|x: int| 0 <= x <= j implies
            #[trigger] crate::exec::cache_scheduler::positive_page_unchanged_from_pre(
                &old_e.cs, &new_e.cs, chain[x],
            ) by {
                assert(prefix[x] == chain[x]);
            }
        lemma_architecture_preexisting_origin_cell_is_canonical(
            old_e,
            new_e,
            old_ibm,
            emitted,
            samples,
            reprs,
            chain,
            request_tokens,
            c_tokens,
            layer,
            pos,
        );
    }
}

// Whole-step lift of the per-cell provenance theorem.  Isolating the large
// scheduler case split above keeps the quantified invariant fold small and
// stable for the solver.
pub proof fn derive_architecture_provenance_cache_fidelity(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
)
    requires
        RT::paged_attention_numeric_domain(),
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm),
        crate::proof::cache::provenance::provenance_cache_fidelity(
            &old_e,
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
        ),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::proof::engine::refinement::mach_cache_shape_ok(
            &old_ibm, old_e.cs.num_blocks as nat,
        ),
        crate::exec::cache_scheduler::tail_write_exclusive(&old_e.cs),
        crate::exec::cache_scheduler::residency_history_aligned(&old_e.cs),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
    ensures
        crate::proof::cache::provenance::provenance_cache_fidelity(
            &new_e,
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
        ),
{
    let model = crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm);
    assert forall|chain: Seq<BlockId>, request_tokens: Seq<TokenId>,
            c_tokens: nat, layer: int, pos: nat|
        #![trigger
            crate::proof::cache::provenance::registered_cache_cell_is_canonical(
                &new_e, model, chain, request_tokens, layer, pos,
            ),
            crate::proof::cache::provenance::provenance_prefix_candidate(
                &new_e.cs, chain, request_tokens, c_tokens,
            )
        ]
        crate::proof::cache::provenance::provenance_prefix_candidate(
            &new_e.cs, chain, request_tokens, c_tokens,
        )
        && 0 <= layer < model.weights.layers.len()
        && pos < c_tokens
        implies crate::proof::cache::provenance::registered_cache_cell_is_canonical(
            &new_e, model, chain, request_tokens, layer, pos,
        )
    by {
        lemma_architecture_provenance_cache_cell_preserved(
            old_e,
            new_e,
            old_ibm,
            emitted,
            samples,
            reprs,
            chain,
            request_tokens,
            c_tokens,
            layer,
            pos,
        );
    }
}

// Every materialized scheduler row supplies a canonical writable prefix to the
// architecture's ordinary singleton continuation.  Decode obtains the prefix
// through live engine/private-cache coherence; admission obtains it through
// the persistent physical-prefix certificate.  Both branches end in the same
// model-opaque predicate consumed by `model_cache`.
#[verifier::spinoff_prover]
#[verifier::rlimit(400)]
pub proof fn derive_architecture_machine_canonical_prefix_forward_ready(
    old_e: Engine,
    old_ibm: IndependentBatchModel,
    reprs: StepReprs,
    rid: RequestId,
)
    requires
        RT::paged_attention_numeric_domain(),
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm),
        crate::proof::cache::provenance::registered_cache_fidelity(
            &old_e,
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
        ),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::proof::engine::refinement::mach_cache_shape_ok(
            &old_ibm, old_e.cs.num_blocks as nat,
        ),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::step_reprs_block_tables_bounded(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_ok(old_e, reprs),
        crate::exec::engine::reprs_cached_prefix_origins(old_e, reprs),
        reprs.scheduled.contains(rid),
    ensures
        crate::proof::model::cache::canonical_prefix_forward_ready(
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
            crate::proof::engine::abstract_step::architecture_machine_processed_tokens(
                old_e, reprs, rid,
            ),
            crate::proof::engine::abstract_step::architecture_machine_prefix_len(
                reprs, rid,
            ),
            crate::proof::engine::abstract_step::architecture_machine_cache_base(
                old_e, old_ibm, reprs, rid,
            ),
        ),
{
    let row = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
        reprs, rid,
    );
    assert(exists|i: int| 0 <= i < reprs.scheduled.len()
        && reprs.scheduled[i] == rid);
    assert(0 <= row < reprs.scheduled.len()
        && reprs.scheduled[row] == rid);
    assert(crate::exec::engine::reprs_forward_layout_at(old_e, reprs, row));

    let q_len = crate::proof::engine::abstract_step::architecture_machine_query_len(
        reprs, rid,
    );
    let k_len = crate::proof::engine::abstract_step::architecture_machine_key_len(
        reprs, rid,
    );
    let prefix_len = crate::proof::engine::abstract_step::architecture_machine_prefix_len(
        reprs, rid,
    );
    let tokens = crate::proof::engine::abstract_step::architecture_machine_processed_tokens(
        old_e, reprs, rid,
    );
    let base = crate::proof::engine::abstract_step::architecture_machine_cache_base(
        old_e, old_ibm, reprs, rid,
    );
    let model = crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm);
    let state = old_e.cs.live_requests@[rid];
    let old_machine = old_ibm.machines[rid];
    reveal(crate::proof::engine::abstract_step::architecture_machine_cache_base);

    crate::proof::engine::abstract_step::lemma_architecture_machine_forward_is_prefix_continuation(
        old_e, old_ibm, reprs, rid,
    );
    assert(tokens.len() == k_len);
    reveal(crate::exec::engine::reprs_forward_layout_at);
    assert(q_len > 0);
    assert(q_len <= k_len);
    assert(prefix_len == k_len - q_len);
    assert(prefix_len < k_len);
    assert(crate::proof::tensor::geometry::blocks_needed_for(k_len) <= reprs.bt[row].len());
    assert(reprs.bt[row].len() <= old_e.cs.num_blocks) by {
        assert(crate::exec::engine::step_reprs_block_tables_bounded(old_e, reprs));
    }
    assert(crate::proof::tensor::geometry::blocks_needed_for(k_len) <= old_e.cs.num_blocks);
    assert(crate::proof::model::cache::reference_history_supported(tokens));

    crate::proof::engine::refinement::lemma_architecture_semantic_inv_cache_laws(
        &old_e, &old_ibm,
    );
    reveal(crate::proof::engine::refinement::architecture_semantic_inv);
    assert(crate::proof::engine::refinement::inv(&old_e, &old_ibm));
    assert(crate::proof::reference::independent_batch_model::ibm_valid(old_ibm));
    crate::proof::reference::independent_batch_model::lemma_ibm_valid_semantic_model(old_ibm);
    assert(model.weights.layers.len()
        == old_ibm.model_config.num_layers as nat);
    assert(model.weights.layers.len() > 0);
    crate::proof::model::architecture::lemma_cache_refinement_support_implies_projection_configuration_ready(
        model,
    );
    assert(base.len() == model.weights.layers.len());

    assert forall|layer: int, pos: nat| #![auto]
        0 <= layer < model.weights.layers.len() && pos < tokens.len()
        implies crate::proof::tensor::geometry::slot_in_cache(base[layer].0, pos)
            && crate::proof::tensor::geometry::slot_in_cache(base[layer].1, pos)
    by {
        assert(0 <= layer < old_machine.kv_cache_reprs.len());
        lemma_machine_cache_has_position(
            &old_ibm,
            old_e.cs.num_blocks as nat,
            rid,
            layer,
            k_len,
            pos,
        );
        crate::proof::model::family_layout::lemma_relocated_prefix_base_at(
            old_machine.kv_cache_reprs[layer].0,
            old_e.kv_caches_repr@[layer].0,
            reprs.bt[row],
            prefix_len,
            pos,
        );
        crate::proof::model::family_layout::lemma_relocated_prefix_base_at(
            old_machine.kv_cache_reprs[layer].1,
            old_e.kv_caches_repr@[layer].1,
            reprs.bt[row],
            prefix_len,
            pos,
        );
    }

    if old_e.cs.running@.contains(rid) {
        let machine_tokens = crate::proof::reference::request_machine::token_seq_to_int(
            crate::exec::request_state::history(old_machine.request_state),
        );
        assert(crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm));
        assert(old_machine.kv_initialized);
        assert(crate::proof::reference::request_machine::request_machine_alive(
            old_machine, old_ibm.model_config,
        ));
        assert(crate::exec::request_state::request_state_view_eq(
            state, old_machine.request_state,
        ));
        assert(crate::exec::request_state::history(state)
            == crate::exec::request_state::history(old_machine.request_state));
        assert(tokens == machine_tokens);
        assert(k_len == crate::exec::request_state::history(state).len());
        assert(q_len == 1);
        assert(prefix_len == old_machine.kv_tokens);
        assert(crate::proof::model::cache::machine_cache_fidelity(
            old_machine, model,
        ));
        reveal(crate::proof::model::cache::machine_cache_fidelity);
        assert(crate::proof::model::cache::cache_reprs_match_canonical(
            old_machine.kv_cache_reprs,
            model,
            tokens,
            prefix_len,
        ));
        assert(reprs.bt[row]
            == old_e.cs.request_residency@[rid].block_ids@);
        reveal(crate::proof::engine::refinement::engine_kv_coherent);
        reveal(crate::proof::engine::refinement::engine_kv_coherent_at);
        assert(crate::proof::engine::refinement::engine_kv_coherent_at(
            old_e.kv_caches_repr@,
            &old_e.cs,
            &old_ibm,
            old_e.model_config.num_layers as nat,
        ));
        assert forall|layer: int, pos: nat|
            #![trigger crate::proof::model::cache::cache_pair_at(base, layer, pos)]
            #![trigger crate::proof::model::cache::cache_pair_has_position(
                base, layer, pos,
            )]
            0 <= layer < model.weights.layers.len() && pos < prefix_len
            implies {
                &&& crate::proof::model::cache::cache_pair_has_position(
                    base, layer, pos,
                )
                &&& crate::proof::model::cache::canonical_kv_defined(
                    model, tokens, layer, pos,
                )
                &&& crate::proof::model::cache::cache_pair_at(base, layer, pos)
                    == crate::proof::model::cache::canonical_kv_at(
                        model, tokens, layer, pos,
                    )
            }
        by {
            assert(crate::proof::tensor::geometry::slot_in_cache(base[layer].0, pos));
            assert(crate::proof::tensor::geometry::slot_in_cache(base[layer].1, pos));
            assert(crate::proof::model::cache::cache_pair_has_position(
                base, layer, pos,
            ));
            assert(crate::proof::model::cache::cache_pair_has_position(
                old_machine.kv_cache_reprs, layer, pos,
            ));
            assert(crate::proof::model::cache::canonical_kv_defined(
                model, tokens, layer, pos,
            ));
            crate::proof::cache::coherence::lemma_plan_slot_in_pre_cache(
                old_e, reprs, row, layer, pos,
            );
            lemma_machine_cache_has_position(
                &old_ibm,
                old_e.cs.num_blocks as nat,
                rid,
                layer,
                k_len,
                pos,
            );
            crate::proof::model::family_layout::lemma_relocated_prefix_base_at(
                old_machine.kv_cache_reprs[layer].0,
                old_e.kv_caches_repr@[layer].0,
                reprs.bt[row],
                prefix_len,
                pos,
            );
            crate::proof::model::family_layout::lemma_relocated_prefix_base_at(
                old_machine.kv_cache_reprs[layer].1,
                old_e.kv_caches_repr@[layer].1,
                reprs.bt[row],
                prefix_len,
                pos,
            );
            assert(old_e.cs.live_requests@.contains_key(rid));
            assert(old_ibm.machines.contains_key(rid));
            assert(0 <= layer < old_e.model_config.num_layers as int);
            assert(pos < old_machine.kv_tokens);
            assert(crate::proof::engine::refinement::residency_block_table(
                &old_e.cs, rid,
            ) == reprs.bt[row]);
            assert(crate::proof::tensor::geometry::cache_at(
                old_e.kv_caches_repr@[layer].0,
                crate::proof::tensor::geometry::block_table_slot(reprs.bt[row], pos),
            ) == crate::proof::tensor::geometry::cache_at(
                old_machine.kv_cache_reprs[layer].0, pos,
            ));
            assert(crate::proof::tensor::geometry::cache_at(
                old_e.kv_caches_repr@[layer].1,
                crate::proof::tensor::geometry::block_table_slot(reprs.bt[row], pos),
            ) == crate::proof::tensor::geometry::cache_at(
                old_machine.kv_cache_reprs[layer].1, pos,
            ));
            assert(crate::proof::model::cache::cache_pair_at(base, layer, pos)
                == crate::proof::model::cache::cache_pair_at(
                    old_machine.kv_cache_reprs, layer, pos,
                ));
        }
    } else {
        let prompt_tokens = state.prompt_tokens@;
        let prompt = crate::proof::reference::request_machine::token_seq_to_int(prompt_tokens);
        assert(old_e.cs.waiting@.contains(rid));
        assert(0 < k_len <= prompt_tokens.len());
        assert(tokens == prompt.subrange(0, k_len as int));
        assert(crate::proof::model::cache::histories_share_prefix(
            tokens, prompt, k_len,
        )) by {
            reveal(crate::proof::model::cache::histories_share_prefix);
            assert(tokens.subrange(0, k_len as int) =~= tokens);
            assert(prompt.subrange(0, k_len as int) == tokens);
        }
        assert(crate::exec::engine::reprs_cached_prefix_origin_at(
            old_e, reprs, row,
        )) by {
            reveal(crate::exec::engine::reprs_cached_prefix_origins);
        }
        assert(crate::proof::cache::provenance::registered_prefix_candidate(
            &old_e.cs,
            reprs.bt[row],
            prompt_tokens,
            prefix_len,
        )) by {
            reveal(crate::exec::engine::reprs_cached_prefix_origin_at);
            reveal(crate::proof::cache::provenance::registered_prefix_candidate);
        }
        assert forall|layer: int, pos: nat|
            #![trigger crate::proof::model::cache::cache_pair_at(base, layer, pos)]
            #![trigger crate::proof::model::cache::cache_pair_has_position(
                base, layer, pos,
            )]
            0 <= layer < model.weights.layers.len() && pos < prefix_len
            implies {
                &&& crate::proof::model::cache::cache_pair_has_position(
                    base, layer, pos,
                )
                &&& crate::proof::model::cache::canonical_kv_defined(
                    model, tokens, layer, pos,
                )
                &&& crate::proof::model::cache::cache_pair_at(base, layer, pos)
                    == crate::proof::model::cache::canonical_kv_at(
                        model, tokens, layer, pos,
                    )
            }
        by {
            assert(crate::proof::cache::provenance::registered_cache_cell_is_canonical(
                &old_e,
                model,
                reprs.bt[row],
                prompt_tokens,
                layer,
                pos,
            ));
            reveal(crate::proof::cache::provenance::registered_cache_cell_is_canonical);
            assert(tokens.subrange(0, pos as int + 1)
                =~= prompt.subrange(0, pos as int + 1)) by {
                assert forall|p: int| 0 <= p < pos as int + 1 implies
                    tokens[p] == prompt[p]
                by {
                    assert(tokens.subrange(0, k_len as int)[p]
                        == prompt.subrange(0, k_len as int)[p]);
                }
            }
            assert(crate::proof::model::cache::canonical_kv_defined(
                model, tokens, layer, pos,
            ));
            crate::proof::model::cache::lemma_canonical_kv_respects_shared_prefix(
                model, tokens, prompt, k_len, layer, pos,
            );
            lemma_machine_cache_has_position(
                &old_ibm,
                old_e.cs.num_blocks as nat,
                rid,
                layer,
                k_len,
                pos,
            );
            crate::proof::model::family_layout::lemma_relocated_prefix_base_at(
                old_machine.kv_cache_reprs[layer].0,
                old_e.kv_caches_repr@[layer].0,
                reprs.bt[row],
                prefix_len,
                pos,
            );
            crate::proof::model::family_layout::lemma_relocated_prefix_base_at(
                old_machine.kv_cache_reprs[layer].1,
                old_e.kv_caches_repr@[layer].1,
                reprs.bt[row],
                prefix_len,
                pos,
            );
        }
    }

    reveal(crate::proof::model::cache::canonical_prefix_forward_ready);
}

// The canonical singleton continuation is exactly the semantic cache
// extension of an emitted request machine.  This is the first concrete
// per-request theorem whose statement and proof are shared unchanged by all
// supported architectures.
#[verifier::spinoff_prover]
#[verifier::rlimit(300)]
pub proof fn derive_architecture_emitted_machine_cache_extension(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
    rid: RequestId,
)
    requires
        RT::paged_attention_numeric_domain(),
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm),
        crate::proof::cache::provenance::registered_cache_fidelity(
            &old_e,
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
        ),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::proof::engine::refinement::mach_cache_shape_ok(
            &old_ibm, old_e.cs.num_blocks as nat,
        ),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        new_e.cs.live_requests@.contains_key(rid),
        crate::exec::engine::reprs_emits(reprs, rid),
    ensures ({
        let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
            old_e, new_e, old_ibm, emitted, reprs,
        );
        crate::proof::model::cache::machine_cache_extension(
            old_ibm.machines[rid],
            new_ibm.machines[rid],
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
        )
    }),
{
    let model = crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm);
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    let old_machine = old_ibm.machines[rid];
    let new_machine = new_ibm.machines[rid];
    let processed = crate::proof::engine::abstract_step::architecture_machine_processed_tokens(
        old_e, reprs, rid,
    );
    let prefix_len = crate::proof::engine::abstract_step::architecture_machine_prefix_len(
        reprs, rid,
    );
    let k_len = crate::proof::engine::abstract_step::architecture_machine_key_len(
        reprs, rid,
    );
    let base = crate::proof::engine::abstract_step::architecture_machine_cache_base(
        old_e, old_ibm, reprs, rid,
    );
    let post = crate::proof::engine::abstract_step::architecture_machine_cache_after(
        old_e, old_ibm, reprs, rid,
    );
    let old_tokens = crate::proof::reference::request_machine::token_seq_to_int(
        crate::exec::request_state::history(old_machine.request_state),
    );
    let new_tokens = crate::proof::reference::request_machine::token_seq_to_int(
        crate::exec::request_state::history(new_machine.request_state),
    );

    reveal(crate::exec::engine::architecture_engine_step_relation);
    assert(old_e.cs.live_requests@.contains_key(rid));
    assert(old_ibm.machines.contains_key(rid));
    assert(emitted.contains_key(rid));
    assert(new_ibm.machines.contains_key(rid));
    assert(new_machine
        == crate::proof::engine::abstract_step::architecture_stepped_machine(
            old_e, new_e, old_ibm, emitted, reprs, rid,
        ));
    assert(new_machine.kv_cache_reprs == post);

    derive_architecture_machine_canonical_prefix_forward_ready(
        old_e, old_ibm, reprs, rid,
    );
    crate::proof::model::cache::lemma_canonical_prefix_forward_cache_fidelity(
        model, processed, prefix_len, base,
    );
    crate::proof::engine::abstract_step::lemma_architecture_machine_forward_is_prefix_continuation(
        old_e, old_ibm, reprs, rid,
    );
    assert(post == crate::proof::model::cache::prefix_continuation_cache_reprs(
        model, processed, prefix_len, base,
    ));
    assert(crate::proof::model::cache::cache_reprs_match_canonical(
        post, model, processed, k_len,
    ));

    lemma_architecture_emitted_cache_history_alignment(
        old_e, new_e, old_ibm, emitted, samples, reprs, rid,
    );
    assert(new_machine.kv_tokens == k_len);
    assert(old_machine.kv_tokens <= k_len);
    assert(crate::proof::model::cache::histories_share_prefix(
        processed, new_tokens, k_len,
    ));
    assert(crate::proof::model::cache::histories_share_prefix(
        old_tokens, new_tokens, old_machine.kv_tokens,
    ));
    crate::proof::model::cache::lemma_cache_fidelity_respects_shared_history(
        post, model, processed, new_tokens, k_len,
    );
    assert(crate::proof::model::cache::cache_reprs_match_canonical(
        new_machine.kv_cache_reprs,
        model,
        new_tokens,
        new_machine.kv_tokens,
    ));

    assert(crate::proof::model::cache::machine_cache_fidelity(
        old_machine, model,
    ));
    reveal(crate::proof::model::cache::machine_cache_fidelity);
    assert(crate::proof::model::cache::cache_reprs_match_canonical(
        old_machine.kv_cache_reprs,
        model,
        old_tokens,
        old_machine.kv_tokens,
    ));
    assert(crate::proof::reference::request_machine::request_machine_alive(
        old_machine, old_ibm.model_config,
    ));
    assert(crate::exec::request_state::valid_request_state(
        old_machine.request_state,
    ));
    assert(crate::exec::request_state::valid_request_state(
        new_machine.request_state,
    ));
    assert(crate::proof::model::cache::cached_prefix_supported(
        new_machine.kv_tokens,
    ));
    assert(new_machine.kv_cache_reprs.len()
        == model.weights.layers.len());

    assert(old_machine.kv_tokens <= new_machine.kv_tokens);
    assert(new_machine.kv_tokens <= new_tokens.len());
    assert(crate::proof::model::cache::histories_share_prefix(
        old_tokens, new_tokens, old_machine.kv_tokens,
    ));
    assert(crate::proof::model::cache::cache_sequences_agree_on_prefix_in_range(
        old_machine.kv_cache_reprs,
        new_machine.kv_cache_reprs,
        0,
        model.weights.layers.len(),
        old_machine.kv_tokens,
    )) by {
        reveal(crate::proof::model::cache::cache_sequences_agree_on_prefix_in_range);
        assert forall|layer: int, pos: nat|
            #![auto]
            0 <= layer < model.weights.layers.len()
                && pos < old_machine.kv_tokens
            implies {
                &&& crate::proof::tensor::geometry::slot_in_cache(
                    old_machine.kv_cache_reprs[layer].0, pos,
                )
                &&& crate::proof::tensor::geometry::slot_in_cache(
                    old_machine.kv_cache_reprs[layer].1, pos,
                )
                &&& crate::proof::tensor::geometry::slot_in_cache(
                    new_machine.kv_cache_reprs[layer].0, pos,
                )
                &&& crate::proof::tensor::geometry::slot_in_cache(
                    new_machine.kv_cache_reprs[layer].1, pos,
                )
                &&& crate::proof::tensor::geometry::cache_at(
                    old_machine.kv_cache_reprs[layer].0, pos,
                ) == crate::proof::tensor::geometry::cache_at(
                    new_machine.kv_cache_reprs[layer].0, pos,
                )
                &&& crate::proof::tensor::geometry::cache_at(
                    old_machine.kv_cache_reprs[layer].1, pos,
                ) == crate::proof::tensor::geometry::cache_at(
                    new_machine.kv_cache_reprs[layer].1, pos,
                )
            }
        by {
            assert(crate::proof::model::cache::cache_pair_has_position(
                new_machine.kv_cache_reprs, layer, pos,
            ));
            assert(crate::proof::model::cache::canonical_kv_defined(
                model, new_tokens, layer, pos,
            ));
            assert(crate::proof::model::cache::cache_pair_at(
                new_machine.kv_cache_reprs, layer, pos,
            ) == crate::proof::model::cache::canonical_kv_at(
                model, new_tokens, layer, pos,
            ));
            if pos < old_machine.kv_tokens {
                assert(crate::proof::model::cache::cache_pair_has_position(
                    old_machine.kv_cache_reprs, layer, pos,
                ));
                assert(crate::proof::model::cache::canonical_kv_defined(
                    model, old_tokens, layer, pos,
                ));
                crate::proof::model::cache::lemma_canonical_kv_respects_shared_prefix(
                    model,
                    old_tokens,
                    new_tokens,
                    old_machine.kv_tokens,
                    layer,
                    pos,
                );
                assert(crate::proof::model::cache::cache_pair_at(
                    old_machine.kv_cache_reprs, layer, pos,
                ) == crate::proof::model::cache::canonical_kv_at(
                    model, old_tokens, layer, pos,
                ));
            }
            assert(crate::proof::model::cache::cache_pair_at(
                new_machine.kv_cache_reprs, layer, pos,
            ) == crate::proof::model::cache::cache_pair_at(
                old_machine.kv_cache_reprs, layer, pos,
            ));
        }
    }
    crate::proof::model::cache::lemma_machine_cache_extension_intro(
        old_machine, new_machine, model,
    );
}

// Lift the emitted-row theorem over the constructed IBM.  Unemitted survivors
// are literal machine stutters; emitted survivors all use the same canonical
// singleton-forward theorem above.  This eliminates the final family/layout
// cache-extension premise from architecture-native one-step refinement.
#[verifier::spinoff_prover]
pub proof fn derive_architecture_ibm_cache_extension(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
)
    requires
        RT::paged_attention_numeric_domain(),
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm),
        crate::proof::cache::provenance::registered_cache_fidelity(
            &old_e,
            crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
        ),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::proof::engine::refinement::mach_cache_shape_ok(
            &old_ibm, old_e.cs.num_blocks as nat,
        ),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
    ensures
        crate::proof::model::cache::ibm_cache_extension(
            old_ibm,
            crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
{
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    let model = crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm);
    reveal(crate::exec::engine::architecture_engine_step_relation);
    reveal(crate::proof::engine::refinement::architecture_semantic_inv);
    assert(crate::proof::reference::independent_batch_model::ibm_semantic_model(new_ibm)
        == model);
    assert forall|rid: RequestId|
        #![trigger new_ibm.machines.contains_key(rid)]
        new_ibm.machines.contains_key(rid) implies {
            &&& old_ibm.machines.contains_key(rid)
            &&& crate::proof::model::cache::machine_cache_extension(
                old_ibm.machines[rid],
                new_ibm.machines[rid],
                crate::proof::reference::independent_batch_model::ibm_semantic_model(new_ibm),
            )
        }
    by {
        assert(new_e.cs.live_requests@.contains_key(rid));
        assert(old_e.cs.live_requests@.contains_key(rid));
        assert(old_ibm.machines.contains_key(rid));
        if crate::exec::engine::reprs_emits(reprs, rid) {
            derive_architecture_emitted_machine_cache_extension(
                old_e,
                new_e,
                old_ibm,
                emitted,
                samples,
                reprs,
                rid,
            );
        } else {
            assert(!emitted.contains_key(rid));
            assert(new_ibm.machines[rid]
                == crate::proof::engine::abstract_step::architecture_stepped_machine(
                    old_e, new_e, old_ibm, emitted, reprs, rid,
                ));
            assert(new_ibm.machines[rid] == old_ibm.machines[rid]);
            assert(crate::proof::reference::request_machine::request_machine_alive(
                old_ibm.machines[rid], old_ibm.model_config,
            ));
            assert(crate::proof::model::cache::machine_cache_fidelity(
                old_ibm.machines[rid], model,
            ));
            crate::proof::model::cache::lemma_machine_cache_extension_reflexive(
                old_ibm.machines[rid], model,
            );
        }
    }
    reveal(crate::proof::model::cache::ibm_cache_extension);
}

} // verus!
