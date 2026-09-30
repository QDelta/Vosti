//! Shared four-norm decoder capstones, independent of the model-family tag.

#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::model as SEMANTICS;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::batch_invariance as PROJECTION;
#[cfg(verus_only)]
pub use crate::proof::model::four_norm_gated::batch_invariance::request_projection_ready;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub open spec fn cache_configuration_ready(
    wr: ModelWeightsRepr, config: FourNormGatedDecoderConfigRepr,
) -> bool {
    config.layers.len() == wr.layers.len()
    && forall|layer: int| 0 <= layer < config.layers.len() ==>
        layer_attention_config_valid(#[trigger] config.layers[layer].attention)
}

pub open spec fn forward_logits_repr(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
) -> Tensor2D {
    SEMANTICS::model_forward_logits_repr(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    )
}

pub open spec fn forward_kv_reprs(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> {
    SEMANTICS::model_forward_kv_reprs(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    )
}

pub open spec fn reference_logits_last_row(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    history: IntTensor1D,
) -> Tensor1D
    recommends
        family.layers.len() == wr.layers.len(),
        wr.layers.len() > 0,
        history.len() > 0,
{
    let n = history.len();
    let logits = forward_logits_repr(
        wr,
        family,
        history,
        crate::proof::tensor::geometry::positions_from(0, n),
        crate::proof::reference::request_machine::synthetic_cache_reprs(n, wr.layers.len()),
        crate::proof::reference::request_machine::slots_from(0, n),
        crate::proof::reference::request_machine::seq_lens_for_single(n),
        crate::proof::reference::request_machine::seq_lens_for_single(n),
        n,
        n,
        crate::proof::reference::request_machine::singleton_block_rows(n),
    );
    logits[n as int - 1]
}

pub proof fn lemma_reference_logits_last_row_is_forward_last(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    history: IntTensor1D,
)
    requires
        family.layers.len() == wr.layers.len(),
        wr.layers.len() > 0,
        history.len() > 0,
    ensures
        reference_logits_last_row(wr, family, history)
            == forward_logits_repr(
                wr,
                family,
                history,
                crate::proof::tensor::geometry::positions_from(0, history.len()),
                crate::proof::reference::request_machine::synthetic_cache_reprs(
                    history.len(), wr.layers.len(),
                ),
                crate::proof::reference::request_machine::slots_from(0, history.len()),
                crate::proof::reference::request_machine::seq_lens_for_single(history.len()),
                crate::proof::reference::request_machine::seq_lens_for_single(history.len()),
                history.len(),
                history.len(),
                crate::proof::reference::request_machine::singleton_block_rows(history.len()),
            )[history.len() as int - 1],
{
    reveal(reference_logits_last_row);
}

pub proof fn lemma_request_projection_ready_from_common_domain(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    architecture: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    row: nat,
)
    requires
        crate::proof::model::architecture::four_norm_decoder_config(architecture) == Some(family),
        crate::proof::model::family_layout::request_projection_common_domain(
            wr,
            architecture,
            input_ids,
            positions,
            pre_kv,
            slots,
            cu_q,
            cu_k,
            max_q,
            max_k,
            block_table,
            row,
        ),
    ensures
        request_projection_ready(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, row,
        ),
{
    reveal(crate::proof::model::family_layout::request_projection_common_domain);
    let hidden = SEMANTICS::scaled_embed_repr(
        input_ids, wr.embed_weight, family.geometry.hidden_size,
    );
    SEMANTICS::lemma_scaled_embed_repr_shape(
        input_ids, wr.embed_weight, family.geometry.hidden_size,
    );
    assert forall|layer: int| 0 <= layer < wr.layers.len() implies
        #[trigger] RT::paged_attention_launch_ready(
            hidden.len(), pre_kv[layer].0, pre_kv[layer].1,
            cu_q, cu_k, max_q, max_k, block_table,
        ) by {}
    PROJECTION::layer_chain_attention_launch_ready_from_common_domain(
        wr.layers,
        family.layers,
        hidden,
        positions,
        pre_kv,
        slots,
        cu_q,
        cu_k,
        max_q,
        max_k,
        block_table,
        0,
    );
    reveal(request_projection_ready);
}

pub proof fn lemma_logits_request_isolation(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    i: nat,
)
    requires request_projection_ready(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, i,
    ),
    ensures
        forward_logits_repr(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ).subrange(cu_q[i as int], cu_q[i as int + 1])
        == forward_logits_repr(
            wr, family,
            input_ids.subrange(cu_q[i as int], cu_q[i as int + 1]),
            positions.subrange(cu_q[i as int], cu_q[i as int + 1]),
            pre_kv, slots.subrange(cu_q[i as int], cu_q[i as int + 1]),
            seq![0int, cu_q[i as int + 1] - cu_q[i as int]],
            seq![0int, cu_k[i as int + 1] - cu_k[i as int]],
            (cu_q[i as int + 1] - cu_q[i as int]) as nat,
            (cu_k[i as int + 1] - cu_k[i as int]) as nat,
            seq![block_table[i as int]],
        ),
{
    reveal(request_projection_ready);
    PROJECTION::model_forward_logits_request_isolation_from_layout(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, i,
    );
}

pub proof fn lemma_logits_repr_shape(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
)
    requires
        family.layers.len() == wr.layers.len(),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
        pre_kv.len() == wr.layers.len(),
    ensures forward_logits_repr(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ).len() == input_ids.len(),
{
    SEMANTICS::lemma_model_forward_logits_repr_shape(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
}

pub proof fn lemma_kv_reprs_len(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
)
    requires
        family.layers.len() == wr.layers.len(),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
        pre_kv.len() >= wr.layers.len(),
    ensures forward_kv_reprs(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ).len() == pre_kv.len(),
{
    SEMANTICS::lemma_model_forward_kv_reprs_len(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
}

pub proof fn lemma_kv_reprs_empty(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
)
    requires
        family.layers.len() == wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == 0,
        positions.len() == 0,
        slots.len() == 0,
    ensures forward_kv_reprs(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ) == pre_kv,
{
    SEMANTICS::lemma_model_forward_kv_reprs_empty(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
}

pub proof fn lemma_forward_cache_shape_preserved(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    num_pages: nat,
)
    requires
        family.layers.len() == wr.layers.len(),
        pre_kv.len() == wr.layers.len(),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
        crate::proof::model::cache::cache_sequence_page_shape(pre_kv, num_pages),
    ensures
        crate::proof::model::cache::cache_sequence_page_shape(
            forward_kv_reprs(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
            num_pages,
        ),
{
    assert(SEMANTICS::cache_sequence_page_shape(
        pre_kv, num_pages,
    )) by {
        reveal(SEMANTICS::cache_sequence_page_shape);
        reveal(crate::proof::model::cache::cache_sequence_page_shape);
    }
    SEMANTICS::lemma_model_forward_cache_shape_preserved(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, num_pages,
    );
    reveal(SEMANTICS::cache_sequence_page_shape);
    reveal(crate::proof::model::cache::cache_sequence_page_shape);
}

} // verus!
