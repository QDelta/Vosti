//! Gemma3 executable model-forward readiness contract.

use crate::model_config::ModelArchitecture;
use crate::boundary::model_families::gemma3 as GEMMA_BOUNDARY;
#[cfg(verus_only)]
use crate::proof::model::families::four_norm_gated::batch_invariance as G3BI;
#[cfg(verus_only)]
use crate::proof::model::families::four_norm_gated::semantics as G3;
use crate::exec::engine::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// Common packed-row relation shared by the exact forward domain and the
// scheduler-facing common domain below. Naming it avoids
// transporting the same quantified formula through multiple opaque dispatch
// predicates.
#[verifier::opaque]
pub open spec fn packed_attention_rows_ready(
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    num_seqs: nat,
) -> bool {
    &&& forall|i: int| 0 <= i < num_seqs as int ==>
        cu_q_repr[i] < #[trigger] cu_q_repr[i + 1]
    &&& forall|i: int| 0 <= i < num_seqs as int ==>
        cu_k_repr[i] < #[trigger] cu_k_repr[i + 1]
    &&& forall|i: int| 0 <= i < num_seqs as int ==>
        cu_q_repr[i + 1] - cu_q_repr[i]
            <= cu_k_repr[i + 1] - #[trigger] cu_k_repr[i]
}

pub proof fn packed_attention_rows_ready_at(
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    num_seqs: nat,
    i: int,
)
    requires
        packed_attention_rows_ready(cu_q_repr, cu_k_repr, num_seqs),
        cu_q_repr.len() == num_seqs + 1,
        cu_k_repr.len() == num_seqs + 1,
        0 <= i < num_seqs,
    ensures
        cu_q_repr[i] < cu_q_repr[i + 1],
        cu_k_repr[i] < cu_k_repr[i + 1],
        cu_q_repr[i + 1] - cu_q_repr[i]
            <= cu_k_repr[i + 1] - cu_k_repr[i],
{
    reveal(packed_attention_rows_ready);
    assert(0 <= i < cu_q_repr.len());
    assert(0 <= i + 1 < cu_q_repr.len());
    assert(0 <= i < cu_k_repr.len());
    assert(0 <= i + 1 < cu_k_repr.len());
    assert(0 <= i < num_seqs as int);
    assert(cu_q_repr[i + 1] == cu_q_repr[i + 1]);
    assert(cu_k_repr[i + 1] == cu_k_repr[i + 1]);
    assert(cu_q_repr[i] < cu_q_repr[i + 1]);
    assert(cu_k_repr[i] < cu_k_repr[i + 1]);
}

// Logical launch domain for the checked Gemma executable composition. It follows
// the exact evolving hidden/cache fold used by the Gemma semantics, while the
// common packed-metadata conditions stay explicit at this family boundary.
// The predicate deliberately retains every layer's complete causal KV cache;
// sliding attention changes only the layer-local score mask.
#[verifier::opaque]
pub open spec fn forward_ready(
    wp: &RT::ModelWeightsPerms,
    input_ids_repr: IntTensor1D,
    positions_repr: IntTensor1D,
    pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
    num_seqs: nat,
) -> bool {
    let wr = RT::model_weights_repr_of(wp);
    let gemma = GEMMA_BOUNDARY::weights_extension_repr_of(wp);
    &&& wp.architecture() == ModelArchitecture::Gemma3Text
    &&& gemma3_config_valid(gemma)
    &&& wr.layers.len() == wp.num_layers()
    &&& gemma.layers.len() == wp.num_layers()
    &&& pre_kv_reprs.len() == wp.num_layers()
    &&& input_ids_repr.len() > 0
    &&& input_ids_repr.len() == positions_repr.len()
    &&& slot_repr.len() == input_ids_repr.len()
    &&& cu_q_repr.len() == num_seqs + 1
    &&& cu_k_repr.len() == num_seqs + 1
    &&& num_seqs > 0
    &&& bt_repr.len() == num_seqs
    &&& cu_q_repr[0] == 0
    &&& cu_k_repr[0] == 0
    &&& cu_q_repr[num_seqs as int] == input_ids_repr.len() as int
    &&& packed_attention_rows_ready(cu_q_repr, cu_k_repr, num_seqs)
    &&& forall|i: int| 0 <= i < wp.num_layers() as int ==>
        #[trigger] RT::store_kv_cache_metadata_ready(
            input_ids_repr.len(), pre_kv_reprs[i].0.len(), slot_repr,
        )
    &&& forall|i: int| 0 <= i < wp.num_layers() as int ==>
        #[trigger] RT::store_kv_cache_launch_ready(
            input_ids_repr.len(), pre_kv_reprs[i].0,
            pre_kv_reprs[i].1, slot_repr,
        )
    &&& forall|i: int| 0 <= i < wp.num_layers() as int ==>
        #[trigger] RT::paged_attention_launch_ready(
            input_ids_repr.len(), pre_kv_reprs[i].0, pre_kv_reprs[i].1,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        )
    &&& G3BI::layer_chain_attention_launch_ready(
        wr.layers, gemma.layers,
        G3::scaled_embed_repr(
            input_ids_repr, wr.embed_weight, gemma.geometry.hidden_size,
        ),
        positions_repr, pre_kv_reprs, slot_repr,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr, 0,
    )
}
// Scheduler-facing premises for the checked Gemma forward. This retains
// the common packed metadata, KV-scatter, and per-layer paged-attention domain
// while hiding the exact evolving Gemma layer fold consumed by
// `forward_ready`.
#[verifier::opaque]
pub open spec fn forward_common_domain_ready(
    wp: &RT::ModelWeightsPerms,
    input_ids_repr: IntTensor1D,
    positions_repr: IntTensor1D,
    pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
    num_seqs: nat,
) -> bool {
    let wr = RT::model_weights_repr_of(wp);
    let gemma = GEMMA_BOUNDARY::weights_extension_repr_of(wp);
    &&& wp.architecture() == ModelArchitecture::Gemma3Text
    &&& gemma3_config_valid(gemma)
    &&& wr.layers.len() == wp.num_layers()
    &&& gemma.layers.len() == wp.num_layers()
    &&& pre_kv_reprs.len() == wp.num_layers()
    &&& input_ids_repr.len() > 0
    &&& input_ids_repr.len() == positions_repr.len()
    &&& slot_repr.len() == input_ids_repr.len()
    &&& cu_q_repr.len() == num_seqs + 1
    &&& cu_k_repr.len() == num_seqs + 1
    &&& num_seqs > 0
    &&& bt_repr.len() == num_seqs
    &&& cu_q_repr[0] == 0
    &&& cu_k_repr[0] == 0
    &&& cu_q_repr[num_seqs as int] == input_ids_repr.len() as int
    &&& packed_attention_rows_ready(cu_q_repr, cu_k_repr, num_seqs)
    &&& forall|i: int| 0 <= i < wp.num_layers() as int ==>
        #[trigger] RT::store_kv_cache_metadata_ready(
            input_ids_repr.len(), pre_kv_reprs[i].0.len(), slot_repr,
        )
    &&& forall|i: int| 0 <= i < wp.num_layers() as int ==>
        #[trigger] RT::paged_attention_launch_ready(
            input_ids_repr.len(), pre_kv_reprs[i].0, pre_kv_reprs[i].1,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        )
}

// The scheduler's ordinary full-cache launch domain is sufficient for Gemma's
// exact Full/SWA fold.  SWA still consumes the complete causal cache here; the
// family-local theorem proves only the positive-window launch choice, not a
// smaller dependency set or eviction safety.
pub proof fn lemma_forward_ready_from_common_domain(
    wp: &RT::ModelWeightsPerms,
    input_ids_repr: IntTensor1D,
    positions_repr: IntTensor1D,
    pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
    num_seqs: nat,
)
    requires
        forward_common_domain_ready(
            wp, input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
            num_seqs,
        ),
    ensures
        forward_ready(
            wp, input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
            num_seqs,
        ),
{
    reveal(forward_common_domain_ready);
    reveal(forward_ready);
    assert(forward_common_domain_ready(
        wp, input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        num_seqs,
    ));
    let wr = RT::model_weights_repr_of(wp);
    let gemma = GEMMA_BOUNDARY::weights_extension_repr_of(wp);
    GEMMA_BOUNDARY::lemma_extension_attention_configs_valid(wp);
    let hidden = G3::scaled_embed_repr(
        input_ids_repr, wr.embed_weight, gemma.geometry.hidden_size,
    );
    reveal(G3::scaled_embed_repr);
    assert(hidden.len() == input_ids_repr.len());
    assert forall|i: int| 0 <= i < wr.layers.len() implies
        #[trigger] RT::paged_attention_launch_ready(
            hidden.len(), pre_kv_reprs[i].0, pre_kv_reprs[i].1,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ) by {
        assert(0 <= i < wp.num_layers() as int);
    }
    G3BI::layer_chain_attention_launch_ready_from_common_domain(
        wr.layers, gemma.layers, hidden, positions_repr, pre_kv_reprs,
        slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
        bt_repr, 0,
    );
    assert(G3BI::layer_chain_attention_launch_ready(
        wr.layers, gemma.layers, hidden, positions_repr, pre_kv_reprs,
        slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
        bt_repr, 0,
    ));
    assert(wp.architecture() == ModelArchitecture::Gemma3Text);
    assert(wr.layers.len() == wp.num_layers());
    assert(gemma.layers.len() == wp.num_layers());
    assert(pre_kv_reprs.len() == wp.num_layers());
    assert(input_ids_repr.len() > 0);
    assert(input_ids_repr.len() == positions_repr.len());
    assert(slot_repr.len() == input_ids_repr.len());
    assert(cu_q_repr.len() == num_seqs + 1);
    assert(cu_k_repr.len() == num_seqs + 1);
    assert(num_seqs > 0);
    assert(bt_repr.len() == num_seqs);
    assert(cu_q_repr[0] == 0);
    assert(cu_k_repr[0] == 0);
    assert(cu_q_repr[num_seqs as int] == input_ids_repr.len() as int);
    assert(packed_attention_rows_ready(
        cu_q_repr, cu_k_repr, num_seqs,
    ));
    assert forall|i: int| 0 <= i < wp.num_layers() as int implies
        #[trigger] RT::store_kv_cache_metadata_ready(
            input_ids_repr.len(), pre_kv_reprs[i].0.len(), slot_repr,
        ) by {}
    assert forall|i: int| 0 <= i < wp.num_layers() as int implies
        #[trigger] RT::store_kv_cache_launch_ready(
            input_ids_repr.len(), pre_kv_reprs[i].0,
            pre_kv_reprs[i].1, slot_repr,
        ) by {
        RT::lemma_paged_attention_launch_ready_parts(
            input_ids_repr.len(), pre_kv_reprs[i].0, pre_kv_reprs[i].1,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        );
        RT::lemma_store_kv_cache_launch_ready_from_parts(
            input_ids_repr.len(), pre_kv_reprs[i].0,
            pre_kv_reprs[i].1, slot_repr,
        );
    }
    assert(forward_ready(
        wp, input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        num_seqs,
    ));
}

proof fn lemma_engine_forward_common_domain_ready(
    old_e: Engine,
    reprs: StepReprs,
    wp: &RT::ModelWeightsPerms,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
)
    requires
        wp.architecture() == ModelArchitecture::Gemma3Text,
        RT::model_execution_valid(&old_e.weights, &old_e.runtime, wp),
        gemma3_config_valid(GEMMA_BOUNDARY::weights_extension_repr_of(wp)),
        step_reprs_wf(old_e, reprs),
        reprs.wr == RT::model_weights_repr_of(wp),
        reprs.input_ids.len() > 0,
        pre_kv.len() == wp.num_layers(),
        forall|i: int| 0 <= i < wp.num_layers() as int ==>
            #[trigger] RT::store_kv_cache_metadata_ready(
                reprs.input_ids.len(), pre_kv[i].0.len(), reprs.slots,
            ),
        forall|i: int| 0 <= i < wp.num_layers() as int ==>
            #[trigger] RT::paged_attention_launch_ready(
                reprs.input_ids.len(), pre_kv[i].0, pre_kv[i].1,
                reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt,
            ),
    ensures
        forward_common_domain_ready(
            wp, reprs.input_ids, reprs.positions, pre_kv, reprs.slots,
            reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt,
            reprs.scheduled.len(),
        ),
{
    RT::lemma_model_weights_architecture_repr_valid(
        &old_e.weights, &old_e.runtime, wp,
    );
    GEMMA_BOUNDARY::lemma_architecture_repr(wp);
    let architecture_repr = RT::model_weights_architecture_repr_of(wp);
    let family = GEMMA_BOUNDARY::weights_extension_repr_of(wp);
    GEMMA_BOUNDARY::lemma_weights_extension_repr(wp);
    assert(model_weights_architecture_repr_valid(
        RT::model_weights_repr_of(wp), architecture_repr,
    ));
    assert(packed_attention_rows_ready(
        reprs.cu_q, reprs.cu_k, reprs.scheduled.len(),
    )) by {
        reveal(packed_attention_rows_ready);
    }
    assert(forward_common_domain_ready(
        wp, reprs.input_ids, reprs.positions, pre_kv, reprs.slots,
        reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt,
        reprs.scheduled.len(),
    )) by {
        reveal(forward_common_domain_ready);
    }
}

pub proof fn lemma_engine_model_forward_ready(
    old_e: Engine,
    reprs: StepReprs,
    wp: &RT::ModelWeightsPerms,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
)
    requires
        wp.architecture() == ModelArchitecture::Gemma3Text,
        crate::proof::model::families::engine_forward_context(old_e, reprs, wp, pre_kv),
    ensures
        crate::exec::model::architecture_model_forward_ready(
            wp, reprs.input_ids, reprs.positions, pre_kv, reprs.slots,
            reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt,
            reprs.scheduled.len(),
        ),
{
    reveal(crate::proof::model::families::engine_forward_context);
    GEMMA_BOUNDARY::lemma_execution_valid_implies_configuration_ready(
        &old_e.weights, &old_e.runtime, wp,
    );
    reveal(GEMMA_BOUNDARY::configuration_ready);
    assert forall|i: int| 0 <= i < wp.num_layers() as int implies
        #[trigger] RT::store_kv_cache_metadata_ready(
            reprs.input_ids.len(), pre_kv[i].0.len(), reprs.slots,
        ) by {
        assert(RT::store_kv_cache_launch_ready(
            reprs.input_ids.len(), pre_kv[i].0, pre_kv[i].1, reprs.slots,
        ));
        reveal(RT::store_kv_cache_launch_ready);
    }
    lemma_engine_forward_common_domain_ready(old_e, reprs, wp, pre_kv);
    lemma_forward_ready_from_common_domain(
        wp, reprs.input_ids, reprs.positions, pre_kv, reprs.slots,
        reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k,
        reprs.bt, reprs.scheduled.len(),
    );
    reveal(crate::exec::model::architecture_model_forward_ready);
}


} // verus!
