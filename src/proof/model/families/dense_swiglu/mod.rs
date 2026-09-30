//! Shared dense-SwiGLU implementation of the model-family proof contract.
//!
//! Qwen3 and Llama3 share this entire proof composition. Family boundaries
//! establish which configurations are admitted; these proofs consume the
//! exact configuration carried by the model and contain no family policy.

use crate::proof::model::dense_swiglu::cache_semantics as CACHE;
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::relational as RELATIONAL;
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::semantics as SEMANTICS;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

mod cache_refinement;
mod layer_witnesses;

#[cfg(verus_only)]
pub use layer_witnesses::{
    layer_kv_rows, lemma_forward_layer_store, lemma_forward_relocation, lemma_kv_request_isolation,
    lemma_layer_kv_rows_request_isolation, lemma_layer_kv_rows_shape,
};

verus! {

pub open spec fn forward_logits_repr(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
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
    SEMANTICS::model_forward_logits_repr(dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    )
}

pub open spec fn forward_kv_reprs(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
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
    CACHE::model_forward_kv_reprs(dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    )
}

pub open spec fn reference_logits_last_row(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
    history: IntTensor1D,
) -> Tensor1D
    recommends wr.layers.len() > 0, history.len() > 0,
{
    SEMANTICS::reference_logits_last_row(
        dense_swiglu_forward_config_repr(family), wr, history,
    )
}

pub proof fn lemma_reference_logits_last_row_is_forward_last(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
    history: IntTensor1D,
)
    requires
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
    reveal(forward_logits_repr);
    reveal(SEMANTICS::reference_logits_last_row);
    reveal(SEMANTICS::model_forward_logits_repr);
}

// Relates the closed public architecture tag to the common dense decoder
// configuration consumed below. Adding another architecture with this exact
// composition extends this one reviewed relation, not the proof body.
pub open spec fn architecture_uses_config(
    architecture: ModelWeightsArchitectureRepr,
    family: DenseSwiGluDecoderConfigRepr,
) -> bool {
    match architecture {
        ModelWeightsArchitectureRepr::Qwen3(config) => config == family,
        ModelWeightsArchitectureRepr::Llama3(config) => config == family,
        ModelWeightsArchitectureRepr::Gemma3Text(_) => false,
        ModelWeightsArchitectureRepr::Gemma4Text(_) => false,
    }
}

pub open spec fn cache_refinement_supported(
    _wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
    family: DenseSwiGluDecoderConfigRepr,
) -> bool {
    architecture_uses_config(architecture, family)
        && match architecture {
            ModelWeightsArchitectureRepr::Qwen3(_) => qwen3_config_valid(family),
            ModelWeightsArchitectureRepr::Llama3(_) => llama3_config_valid(family),
            ModelWeightsArchitectureRepr::Gemma3Text(_) => false,
        ModelWeightsArchitectureRepr::Gemma4Text(_) => false,
        }
}

pub open spec fn semantic_model(
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
) -> SemanticModelRepr {
    SemanticModelRepr {
        weights: wr,
        architecture,
    }
}

pub proof fn lemma_cache_refinement_laws(
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
    family: DenseSwiGluDecoderConfigRepr,
)
    requires
        semantic_model_repr_valid(semantic_model(wr, architecture)),
        cache_refinement_supported(wr, architecture, family),
    ensures
        crate::proof::model::cache::cache_refinement_laws(
            semantic_model(wr, architecture),
        ),
{
    cache_refinement::lemma_cache_refinement_laws(wr, architecture, family);
}

pub open spec fn request_projection_ready(
    wr: ModelWeightsRepr,
    _family: DenseSwiGluDecoderConfigRepr,
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
) -> bool {
    &&& RT::paged_attention_numeric_domain()
    &&& wr.layers.len() > 0
    &&& pre_kv.len() >= wr.layers.len()
    &&& input_ids.len() == positions.len()
    &&& slots.len() == input_ids.len()
    &&& cu_q.len() == cu_k.len()
    &&& cu_k.len() == block_table.len() + 1
    &&& i < block_table.len()
    &&& cu_q[0] == 0
    &&& cu_k[0] == 0
    &&& cu_q[block_table.len() as int] == input_ids.len() as int
    &&& forall|j: int| 0 <= j < block_table.len() as int ==>
        cu_q[j] < #[trigger] cu_q[j + 1]
    &&& forall|j: int| 0 <= j < block_table.len() as int ==>
        cu_k[j] < #[trigger] cu_k[j + 1]
    &&& 0 <= cu_q[i as int] < cu_q[i as int + 1]
        <= input_ids.len() as int
    &&& 0 <= cu_k[i as int] < cu_k[i as int + 1]
    &&& max_q > 0
    &&& cu_q[i as int + 1] - cu_q[i as int] <= max_q as int
    &&& cu_k[i as int + 1] - cu_k[i as int] <= max_k as int
    &&& cu_q[i as int + 1] - cu_q[i as int]
        <= cu_k[i as int + 1] - cu_k[i as int]
    &&& crate::proof::tensor::geometry::blocks_needed_for(
        (cu_k[i as int + 1] - cu_k[i as int]) as nat,
    ) <= block_table[i as int].len()
    &&& forall|m: int, l: int|
        #![trigger slots.subrange(0, cu_q[i as int])[m]
            / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
        0 <= m < cu_q[i as int]
            && 0 <= l < block_table[i as int].len() ==>
            slots.subrange(0, cu_q[i as int])[m]
                / (crate::types::BLOCK_SIZE_SPEC as int)
                != block_table[i as int][l] as int
    &&& forall|m: int, l: int|
        #![trigger slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
            / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
        0 <= m < slots.len() - cu_q[i as int + 1]
            && 0 <= l < block_table[i as int].len() ==>
            slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
                / (crate::types::BLOCK_SIZE_SPEC as int)
                != block_table[i as int][l] as int
    &&& forall|layer: int, pos: nat|
        #![trigger pre_kv[layer].0,
            crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos)]
        0 <= layer < wr.layers.len()
            && pos < (cu_k[i as int + 1] - cu_k[i as int]) as nat ==>
            crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].0,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].1,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
            )
    &&& forall|layer: int| 0 <= layer < wr.layers.len() ==>
        #[trigger] RT::paged_attention_launch_ready(
            input_ids.len(), pre_kv[layer].0, pre_kv[layer].1,
            cu_q, cu_k, max_q, max_k, block_table,
        )
}

#[verifier::rlimit(20)]
pub proof fn lemma_request_projection_ready_from_common_domain(
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
    family: DenseSwiGluDecoderConfigRepr,
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
        architecture_uses_config(architecture, family),
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
    reveal(crate::proof::model::architecture::request_projection_configuration_ready);
    reveal(request_projection_ready);
    reveal(architecture_uses_config);
    match architecture {
        ModelWeightsArchitectureRepr::Qwen3(config) => {
            assert(config == family);
        },
        ModelWeightsArchitectureRepr::Llama3(config) => {
            assert(config == family);
        },
        ModelWeightsArchitectureRepr::Gemma3Text(_) => {
            assert(false);
        },
        ModelWeightsArchitectureRepr::Gemma4Text(_) => { assert(false); },
    }
    assert(family.geometry.num_layers == wr.layers.len());
    assert(wr.layers.len() > 0);
    assert forall|j: int| 0 <= j < block_table.len() as int implies
        cu_q[j] < #[trigger] cu_q[j + 1] by {};
    assert forall|j: int| 0 <= j < block_table.len() as int implies
        cu_k[j] < #[trigger] cu_k[j + 1] by {};
    assert forall|m: int, l: int|
        #![trigger slots.subrange(0, cu_q[row as int])[m]
            / (crate::types::BLOCK_SIZE_SPEC as int), block_table[row as int][l]]
        0 <= m < cu_q[row as int]
            && 0 <= l < block_table[row as int].len() implies
            slots.subrange(0, cu_q[row as int])[m]
                / (crate::types::BLOCK_SIZE_SPEC as int)
                != block_table[row as int][l] as int by {};
    assert forall|m: int, l: int|
        #![trigger slots.subrange(cu_q[row as int + 1], slots.len() as int)[m]
            / (crate::types::BLOCK_SIZE_SPEC as int), block_table[row as int][l]]
        0 <= m < slots.len() - cu_q[row as int + 1]
            && 0 <= l < block_table[row as int].len() implies
            slots.subrange(cu_q[row as int + 1], slots.len() as int)[m]
                / (crate::types::BLOCK_SIZE_SPEC as int)
                != block_table[row as int][l] as int by {};
    assert forall|layer: int, pos: nat|
        #![trigger pre_kv[layer].0,
            crate::proof::tensor::geometry::block_table_slot(block_table[row as int], pos)]
        0 <= layer < wr.layers.len()
            && pos < (cu_k[row as int + 1] - cu_k[row as int]) as nat implies
            crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].0,
                crate::proof::tensor::geometry::block_table_slot(block_table[row as int], pos),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].1,
                crate::proof::tensor::geometry::block_table_slot(block_table[row as int], pos),
            ) by {};
    assert forall|layer: int| 0 <= layer < wr.layers.len() implies
        #[trigger] RT::paged_attention_launch_ready(
            input_ids.len(), pre_kv[layer].0, pre_kv[layer].1,
            cu_q, cu_k, max_q, max_k, block_table,
        ) by {};
}

pub proof fn lemma_logits_request_isolation(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
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
    RELATIONAL::model_forward_request_isolation(dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
        slots.subrange(cu_q[i as int], cu_q[i as int + 1]), i,
    );
}

pub proof fn lemma_logits_repr_shape(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
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
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
        pre_kv.len() == wr.layers.len(),
    ensures forward_logits_repr(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ).len() == input_ids.len(),
{
    SEMANTICS::lemma_model_forward_logits_repr_shape(dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
}

pub proof fn lemma_kv_reprs_len(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
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
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
        pre_kv.len() >= wr.layers.len(),
    ensures forward_kv_reprs(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ).len() == pre_kv.len(),
{
    CACHE::lemma_model_forward_kv_reprs_len(dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
}

pub proof fn lemma_kv_reprs_empty(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
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
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == 0,
        positions.len() == 0,
        slots.len() == 0,
    ensures forward_kv_reprs(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ) == pre_kv,
{
    CACHE::lemma_model_forward_kv_reprs_empty(dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
}

pub proof fn lemma_forward_cache_shape_preserved(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
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
    let post = forward_kv_reprs(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
    CACHE::lemma_model_forward_kv_reprs_len(dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
    assert(post.len() == pre_kv.len());
    assert forall|layer: int| 0 <= layer < post.len() implies {
        &&& (#[trigger] post[layer]).0.len() == num_pages
        &&& post[layer].1.len() == num_pages
        &&& (forall|page: int| 0 <= page < num_pages ==>
            (#[trigger] post[layer].0[page]).len()
                == crate::types::BLOCK_SIZE_SPEC as int)
        &&& (forall|page: int| 0 <= page < num_pages ==>
            (#[trigger] post[layer].1[page]).len()
                == crate::types::BLOCK_SIZE_SPEC as int)
    } by {
        assert(layer < wr.layers.len());
        layer_witnesses::lemma_forward_layer_store(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer as nat,
        );
        let rows = layer_witnesses::layer_kv_rows(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer as nat,
        );
        RT::lemma_store_kv_cache_repr_shape_unconditional(
            rows.0, rows.1, pre_kv[layer].0, pre_kv[layer].1, slots,
        );
        assert(crate::proof::model::cache::cache_sequence_page_shape(
            pre_kv, num_pages,
        ));
    }
}

} // verus!
