//! Shared four-norm gated decoder batch invariance proofs.
//!
//! The fold is parameterized by immutable row/attention policies. Attention
//! uses the common generated raw operator and checked projection lemmas;
//! admitting a new profile still requires exact kernel-interface qualification.
//! Full causal-prefix cache retention remains the conservative contract.

use crate::model_config::AttentionKind;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::model as MODEL;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::layers as FOUR_NORM;
#[cfg(verus_only)]
use crate::boundary::backend_certificates::support as GAC;
#[cfg(verus_only)]
use crate::boundary::dense_layer_primitives as LAYERS;
#[cfg(verus_only)]
use crate::boundary::attention_operator as AO;
#[cfg(verus_only)]
use crate::proof::model::layer_properties as BI;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub proof fn scaled_embed_subrange_invariance(
    input_ids: IntTensor1D,
    weight: Tensor2D,
    hidden_size: nat,
    a: int,
    b: int,
)
    requires 0 <= a <= b <= input_ids.len(),
    ensures MODEL::scaled_embed_repr(input_ids.subrange(a, b), weight, hidden_size)
        == MODEL::scaled_embed_repr(input_ids, weight, hidden_size).subrange(a, b),
{
    reveal(MODEL::scaled_embed_repr);
    assert(MODEL::scaled_embed_repr(input_ids.subrange(a, b), weight, hidden_size)
        =~= MODEL::scaled_embed_repr(input_ids, weight, hidden_size).subrange(a, b));
}

pub proof fn split_last_axis_half_subrange_invariance(
    input: Tensor2D,
    right_half: bool,
    a: int,
    b: int,
)
    requires 0 <= a <= b <= input.len(),
    ensures
        RT::split_last_axis_half_repr(input.subrange(a, b), right_half)
            == RT::split_last_axis_half_repr(input, right_half).subrange(a, b),
{
    reveal(RT::split_last_axis_half_repr);
    assert(RT::split_last_axis_half_repr(input.subrange(a, b), right_half)
        =~= RT::split_last_axis_half_repr(input, right_half).subrange(a, b));
}

// Composition of all row-local work through the exact KV-store inputs.  This
// theorem deliberately stops before the scatter, where full-batch and
// singleton cache values cease to be definitionally identical.
pub proof fn attention_pre_store_subrange_invariance(
    common: LayerWeightsRepr,
    extension: FourNormGatedLayerExtensionRepr,
    hidden: Tensor2D,
    positions: IntTensor1D,
    a: int,
    b: int,
)
    requires
        hidden.len() == positions.len(),
        0 <= a <= b <= hidden.len(),
    ensures
        MODEL::attention_pre_store_repr(
            common, extension, hidden.subrange(a, b),
            positions.subrange(a, b),
        ).0 == MODEL::attention_pre_store_repr(
            common, extension, hidden, positions,
        ).0.subrange(a, b),
        MODEL::attention_pre_store_repr(
            common, extension, hidden.subrange(a, b),
            positions.subrange(a, b),
        ).1 == MODEL::attention_pre_store_repr(
            common, extension, hidden, positions,
        ).1.subrange(a, b),
        MODEL::attention_pre_store_repr(
            common, extension, hidden.subrange(a, b),
            positions.subrange(a, b),
        ).2 == MODEL::attention_pre_store_repr(
            common, extension, hidden, positions,
        ).2.subrange(a, b),
{
    FOUR_NORM::attention_pre_store_subrange_invariance(
        MODEL::four_norm_layer_repr(common, extension), hidden, positions, a, b,
    );
}

// Once attention rows are supplied, the rest of a four-norm block is row-local.
// This packages residual ordering, the four-norm convention, the gate/up
// split, GELU-tanh multiplication, and the down projection into one reusable
// request-segment theorem.
pub proof fn post_attention_and_mlp_subrange_invariance(
    common: LayerWeightsRepr,
    extension: FourNormGatedLayerExtensionRepr,
    attention_residual: Tensor2D,
    attended: Tensor2D,
    a: int,
    b: int,
)
    requires
        attention_residual.len() == attended.len(),
        0 <= a <= b <= attended.len(),
    ensures
        MODEL::post_attention_and_mlp_repr(
            common, extension, attention_residual.subrange(a, b),
            attended.subrange(a, b),
        ) == MODEL::post_attention_and_mlp_repr(
            common, extension, attention_residual, attended,
        ).subrange(a, b),
{
    FOUR_NORM::post_attention_and_mlp_subrange_invariance(
        MODEL::four_norm_layer_repr(common, extension), attention_residual, attended, a, b,
    );
}

// Architecture-dispatched launch and indexed-cache contracts.  Keeping these
// two match points separate from the layer proof makes the full/SWA choice
// explicit while leaving the rest of the decomposition architecture-neutral.
pub open spec fn paged_cache_geometry(
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    attention: AttentionConfigRepr,
) -> bool {
    match attention {
        AttentionConfigRepr::Full =>
            GAC::paged_cache_geometry(k_cache, v_cache),
        AttentionConfigRepr::SlidingWindow(window_size) =>
            GAC::paged_cache_geometry(k_cache, v_cache),
    }
}

pub open spec fn paged_attention_launch_ready(
    q: Tensor2D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    attention: AttentionConfigRepr,
) -> bool {
    match attention {
        AttentionConfigRepr::Full => GAC::paged_attention_launch_ready(
            q.len(), k_cache, v_cache, cu_q, cu_k,
            max_q, max_k, block_table,
        ),
        AttentionConfigRepr::SlidingWindow(window_size) =>
            GAC::swa_paged_attention_launch_ready(
                q.len(), k_cache, v_cache, cu_q, cu_k,
                max_q, max_k, block_table,
                window_size,
            ),
    }
}

pub proof fn paged_attention_launch_ready_segment_bounds(
    q: Tensor2D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    attention: AttentionConfigRepr,
    i: nat,
)
    requires
        i < block_table.len(),
        paged_attention_launch_ready(
            q, k_cache, v_cache, cu_q, cu_k,
            max_q, max_k, block_table, attention,
        ),
    ensures 0 <= cu_q[i as int] < cu_q[i as int + 1] <= q.len(),
{
    reveal(paged_attention_launch_ready);
    match attention {
        AttentionConfigRepr::Full => {
            reveal(GAC::paged_attention_launch_ready);
            reveal(GAC::paged_attention_metadata_ready);
        },
        AttentionConfigRepr::SlidingWindow(_) => {
            reveal(GAC::swa_paged_attention_launch_ready);
            reveal(GAC::paged_attention_metadata_ready);
        },
    }
    assert forall|j: int| 0 <= j < block_table.len() as int implies
        cu_q[j] < #[trigger] cu_q[j + 1] by {
    }
    crate::proof::tensor::geometry::lemma_cu_mono(
        cu_q, block_table.len() as int, 0, i as int,
    );
    crate::proof::tensor::geometry::lemma_cu_mono(
        cu_q, block_table.len() as int,
        i as int + 1, block_table.len() as int,
    );
}

pub proof fn paged_attention_launch_ready_cache_geometry(
    q: Tensor2D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    attention: AttentionConfigRepr,
)
    requires
        paged_attention_launch_ready(
            q, k_cache, v_cache, cu_q, cu_k,
            max_q, max_k, block_table, attention,
        ),
    ensures
        paged_cache_geometry(k_cache, v_cache, attention),
{
    reveal(paged_attention_launch_ready);
    reveal(paged_cache_geometry);
    match attention {
        AttentionConfigRepr::Full => {
            reveal(GAC::paged_attention_launch_ready);
        },
        AttentionConfigRepr::SlidingWindow(window_size) => {
            reveal(GAC::swa_paged_attention_launch_ready);
        },
    }
}

pub open spec fn paged_attention_selected_cache_pages_equal(
    k_cache_a: KVCacheLayerRepr,
    v_cache_a: KVCacheLayerRepr,
    bt_row_a: Seq<BlockId>,
    k_cache_b: KVCacheLayerRepr,
    v_cache_b: KVCacheLayerRepr,
    bt_row_b: Seq<BlockId>,
    k_len: nat,
    attention: AttentionConfigRepr,
) -> bool {
    match attention {
        AttentionConfigRepr::Full =>
            GAC::paged_attention_selected_cache_pages_equal(
                k_cache_a, v_cache_a, bt_row_a,
                k_cache_b, v_cache_b, bt_row_b, k_len,
            ),
        AttentionConfigRepr::SlidingWindow(window_size) =>
            GAC::paged_attention_selected_cache_pages_equal(
                k_cache_a, v_cache_a, bt_row_a,
                k_cache_b, v_cache_b, bt_row_b, k_len,
            ),
    }
}

// Derive the singleton launch domain from the selected row of a valid full
// launch.  A relocated cache may differ in contents, but it must retain the
// same page count and the appropriate full/SWA page geometry.
pub proof fn selected_attention_launch_ready_from_full(
    q: Tensor2D,
    full_k_cache: KVCacheLayerRepr,
    full_v_cache: KVCacheLayerRepr,
    selected_k_cache: KVCacheLayerRepr,
    selected_v_cache: KVCacheLayerRepr,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    attention: AttentionConfigRepr,
    i: nat,
)
    requires
        i < block_table.len(),
        paged_attention_launch_ready(
            q, full_k_cache, full_v_cache, cu_q, cu_k,
            max_q, max_k, block_table, attention,
        ),
        paged_cache_geometry(
            selected_k_cache, selected_v_cache, attention,
        ),
        selected_k_cache.len() == full_k_cache.len(),
    ensures
        paged_attention_launch_ready(
            q.subrange(cu_q[i as int], cu_q[i as int + 1]),
            selected_k_cache, selected_v_cache,
            seq![0int, cu_q[i as int + 1] - cu_q[i as int]],
            seq![0int, cu_k[i as int + 1] - cu_k[i as int]],
            (cu_q[i as int + 1] - cu_q[i as int]) as nat,
            (cu_k[i as int + 1] - cu_k[i as int]) as nat,
            seq![block_table[i as int]], attention,
        ),
{
    let q_len = (cu_q[i as int + 1] - cu_q[i as int]) as nat;
    let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
    let selected_cu_q = seq![0int, cu_q[i as int + 1] - cu_q[i as int]];
    let selected_cu_k = seq![0int, cu_k[i as int + 1] - cu_k[i as int]];
    let selected_rows = seq![block_table[i as int]];
    reveal(paged_attention_launch_ready);
    reveal(paged_cache_geometry);
    match attention {
        AttentionConfigRepr::Full => {
            reveal(GAC::paged_attention_launch_ready);
            reveal(GAC::paged_attention_metadata_ready);
            assert forall|j: int| #![trigger selected_cu_q[j + 1]]
                0 <= j < 1 implies {
                let row_q_len = selected_cu_q[j + 1] - selected_cu_q[j];
                let row_k_len = selected_cu_k[j + 1] - selected_cu_k[j];
                &&& selected_cu_q[j] < selected_cu_q[j + 1]
                &&& selected_cu_k[j] < selected_cu_k[j + 1]
                &&& row_q_len <= q_len as int
                &&& row_k_len <= k_len as int
                &&& row_q_len <= row_k_len
                &&& crate::proof::tensor::geometry::blocks_needed_for(row_k_len as nat)
                    <= selected_rows[j].len()
                &&& (forall|l: int| 0 <= l < selected_rows[j].len() ==>
                    #[trigger] selected_rows[j][l] < selected_k_cache.len())
            } by {
                assert(j == 0);
            }
            assert(GAC::paged_attention_launch_ready(
                q_len, selected_k_cache, selected_v_cache,
                selected_cu_q, selected_cu_k, q_len, k_len, selected_rows,
            ));
        },
        AttentionConfigRepr::SlidingWindow(window_size) => {
            reveal(GAC::swa_paged_attention_launch_ready);
            reveal(GAC::paged_attention_metadata_ready);
            assert forall|j: int| #![trigger selected_cu_q[j + 1]]
                0 <= j < 1 implies {
                let row_q_len = selected_cu_q[j + 1] - selected_cu_q[j];
                let row_k_len = selected_cu_k[j + 1] - selected_cu_k[j];
                &&& selected_cu_q[j] < selected_cu_q[j + 1]
                &&& selected_cu_k[j] < selected_cu_k[j + 1]
                &&& row_q_len <= q_len as int
                &&& row_k_len <= k_len as int
                &&& row_q_len <= row_k_len
                &&& crate::proof::tensor::geometry::blocks_needed_for(row_k_len as nat)
                    <= selected_rows[j].len()
                &&& (forall|l: int| 0 <= l < selected_rows[j].len() ==>
                    #[trigger] selected_rows[j][l] < selected_k_cache.len())
            } by {
                assert(j == 0);
            }
            assert(GAC::swa_paged_attention_launch_ready(
                q_len, selected_k_cache, selected_v_cache,
                selected_cu_q, selected_cu_k, q_len, k_len, selected_rows,
                window_size,
            ));
        },
    }
    assert forall|j: int| 0 <= j < block_table.len() as int implies
        cu_q[j] < #[trigger] cu_q[j + 1] by {}
    crate::proof::tensor::geometry::lemma_cu_mono(
        cu_q, block_table.len() as int, 0, i as int,
    );
    crate::proof::tensor::geometry::lemma_cu_mono(
        cu_q, block_table.len() as int,
        i as int + 1, block_table.len() as int,
    );
    assert(0 <= cu_q[i as int] < cu_q[i as int + 1] <= q.len() as int);
    assert(q.subrange(cu_q[i as int], cu_q[i as int + 1]).len() == q_len);
}

// Checked projection into a distinct cache pool and block-table row. Retain
// the existing stronger whole-page relation, then derive the logical-prefix
// equality consumed by the shared mapped operation.
pub proof fn paged_attention_request_segment_relocation(
    q: Tensor2D,
    full_k_cache: KVCacheLayerRepr,
    full_v_cache: KVCacheLayerRepr,
    selected_k_cache: KVCacheLayerRepr,
    selected_v_cache: KVCacheLayerRepr,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    selected_bt_row: Seq<BlockId>,
    attention: AttentionConfigRepr,
    parameters: AttentionParametersRepr,
    i: nat,
)
    requires
        i < block_table.len(),
        paged_attention_launch_ready(
            q, full_k_cache, full_v_cache, cu_q, cu_k,
            max_q, max_k, block_table, attention,
        ),
        paged_attention_launch_ready(
            q.subrange(cu_q[i as int], cu_q[i as int + 1]),
            selected_k_cache, selected_v_cache,
            seq![0int, cu_q[i as int + 1] - cu_q[i as int]],
            seq![0int, cu_k[i as int + 1] - cu_k[i as int]],
            (cu_q[i as int + 1] - cu_q[i as int]) as nat,
            (cu_k[i as int + 1] - cu_k[i as int]) as nat,
            seq![selected_bt_row], attention,
        ),
        paged_attention_selected_cache_pages_equal(
            full_k_cache, full_v_cache, block_table[i as int],
            selected_k_cache, selected_v_cache, selected_bt_row,
            (cu_k[i as int + 1] - cu_k[i as int]) as nat,
            attention,
        ),
    ensures
        MODEL::paged_attention_repr(
            q, full_k_cache, full_v_cache, cu_q, cu_k, max_q, max_k,
            block_table, attention, parameters,
        ).subrange(cu_q[i as int], cu_q[i as int + 1])
        == MODEL::paged_attention_repr(
            q.subrange(cu_q[i as int], cu_q[i as int + 1]),
            selected_k_cache, selected_v_cache,
            seq![0int, cu_q[i as int + 1] - cu_q[i as int]],
            seq![0int, cu_k[i as int + 1] - cu_k[i as int]],
            (cu_q[i as int + 1] - cu_q[i as int]) as nat,
            (cu_k[i as int + 1] - cu_k[i as int]) as nat,
            seq![selected_bt_row], attention, parameters,
        ),
{
    paged_attention_launch_ready_segment_bounds(
        q, full_k_cache, full_v_cache, cu_q, cu_k,
        max_q, max_k, block_table, attention, i,
    );

    reveal(paged_attention_launch_ready);
    reveal(paged_attention_selected_cache_pages_equal);
    reveal(MODEL::paged_attention_repr);
    match attention {
        AttentionConfigRepr::Full => {

            reveal(GAC::paged_attention_launch_ready);
            reveal(GAC::paged_attention_selected_cache_pages_equal);
            reveal(LAYERS::full_paged_attention_repr);
            reveal(LAYERS::sliding_window_paged_attention_repr);
            let kl = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
            assert forall|pos: nat| #![trigger crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos)]
                pos < kl implies {
                &&& crate::proof::tensor::geometry::cache_at(full_k_cache, crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos))
                    == crate::proof::tensor::geometry::cache_at(selected_k_cache, crate::proof::tensor::geometry::block_table_slot(selected_bt_row, pos))
                &&& crate::proof::tensor::geometry::cache_at(full_v_cache, crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos))
                    == crate::proof::tensor::geometry::cache_at(selected_v_cache, crate::proof::tensor::geometry::block_table_slot(selected_bt_row, pos))
            } by {
                crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, kl);
            }
            AO::lemma_batch_projection_relocated(q, full_k_cache, full_v_cache,
                cu_q, cu_k, block_table, max_q, max_k, selected_k_cache, selected_v_cache, selected_bt_row,
                AttentionKind::Full, parameters, 0, i as int);
        },
        AttentionConfigRepr::SlidingWindow(window_size) => {

            reveal(GAC::swa_paged_attention_launch_ready);
            reveal(GAC::paged_attention_selected_cache_pages_equal);
            reveal(LAYERS::full_paged_attention_repr);
            reveal(LAYERS::sliding_window_paged_attention_repr);
            let kl = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
            assert forall|pos: nat| #![trigger crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos)]
                pos < kl implies {
                &&& crate::proof::tensor::geometry::cache_at(full_k_cache, crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos))
                    == crate::proof::tensor::geometry::cache_at(selected_k_cache, crate::proof::tensor::geometry::block_table_slot(selected_bt_row, pos))
                &&& crate::proof::tensor::geometry::cache_at(full_v_cache, crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos))
                    == crate::proof::tensor::geometry::cache_at(selected_v_cache, crate::proof::tensor::geometry::block_table_slot(selected_bt_row, pos))
            } by {
                crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, kl);
            }
            AO::lemma_batch_projection_relocated(q, full_k_cache, full_v_cache,
                cu_q, cu_k, block_table, max_q, max_k, selected_k_cache, selected_v_cache, selected_bt_row,
                AttentionKind::SlidingWindow, parameters, window_size, i as int);
        },
    }
}

// Causal-prefix equality for singleton full-prefill launches.  SlidingWindow
// uses the same full causal-prefix premise as Full attention; this theorem does
// not claim that entries outside the window can be evicted.
pub proof fn paged_attention_singleton_prefix_invariance(
    full_q: Tensor2D,
    prefix_q: Tensor2D,
    full_k: KVCacheLayerRepr,
    full_v: KVCacheLayerRepr,
    prefix_k: KVCacheLayerRepr,
    prefix_v: KVCacheLayerRepr,
    full_bt: Seq<BlockId>,
    prefix_bt: Seq<BlockId>,
    full_len: nat,
    prefix_len: nat,
    attention: AttentionConfigRepr,
    parameters: AttentionParametersRepr,
)
    requires
        RT::paged_attention_numeric_domain(),
        layer_attention_config_valid(attention),
        0 < prefix_len <= full_len,
        full_q.len() == full_len,
        prefix_q.len() == prefix_len,
        full_q.subrange(0, prefix_len as int) == prefix_q,
        crate::proof::tensor::geometry::blocks_needed_for(full_len) <= full_bt.len(),
        crate::proof::tensor::geometry::blocks_needed_for(prefix_len) <= prefix_bt.len(),
        forall|pos: nat|
            #![trigger crate::proof::tensor::geometry::block_table_slot(full_bt, pos)]
            #![trigger crate::proof::tensor::geometry::block_table_slot(prefix_bt, pos)]
            pos < prefix_len ==> {
                let full_slot = crate::proof::tensor::geometry::block_table_slot(full_bt, pos);
                let prefix_slot = crate::proof::tensor::geometry::block_table_slot(prefix_bt, pos);
                &&& crate::proof::tensor::geometry::slot_in_cache(full_k, full_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(full_v, full_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(prefix_k, prefix_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(prefix_v, prefix_slot)
                &&& crate::proof::tensor::geometry::cache_at(full_k, full_slot)
                    == crate::proof::tensor::geometry::cache_at(prefix_k, prefix_slot)
                &&& crate::proof::tensor::geometry::cache_at(full_v, full_slot)
                    == crate::proof::tensor::geometry::cache_at(prefix_v, prefix_slot)
            },
    ensures
        MODEL::paged_attention_repr(
            full_q, full_k, full_v,
            seq![0int, full_len as int], seq![0int, full_len as int],
            full_len, full_len, seq![full_bt], attention, parameters,
        ).subrange(0, prefix_len as int)
        == MODEL::paged_attention_repr(
            prefix_q, prefix_k, prefix_v,
            seq![0int, prefix_len as int], seq![0int, prefix_len as int],
            prefix_len, prefix_len, seq![prefix_bt], attention, parameters,
        ),
{
    let full_attn = MODEL::paged_attention_repr(
        full_q, full_k, full_v,
        seq![0int, full_len as int], seq![0int, full_len as int],
        full_len, full_len, seq![full_bt], attention, parameters,
    );
    let prefix_attn = MODEL::paged_attention_repr(
        prefix_q, prefix_k, prefix_v,
        seq![0int, prefix_len as int], seq![0int, prefix_len as int],
        prefix_len, prefix_len, seq![prefix_bt], attention, parameters,
    );
    MODEL::lemma_paged_attention_repr_shape(
        full_q, full_k, full_v,
        seq![0int, full_len as int], seq![0int, full_len as int],
        full_len, full_len, seq![full_bt], attention, parameters,
    );
    MODEL::lemma_paged_attention_repr_shape(
        prefix_q, prefix_k, prefix_v,
        seq![0int, prefix_len as int], seq![0int, prefix_len as int],
        prefix_len, prefix_len, seq![prefix_bt], attention, parameters,
    );
    assert forall|j: int| 0 <= j < prefix_len as int implies
        full_attn[j] == prefix_attn[j]
    by {
        assert(full_q[j] == prefix_q[j]);
        assert forall|pos: nat|
            #![trigger crate::proof::tensor::geometry::block_table_slot(full_bt, pos)]
            #![trigger crate::proof::tensor::geometry::block_table_slot(prefix_bt, pos)]
            pos as int <= j implies {
                let full_slot = crate::proof::tensor::geometry::block_table_slot(full_bt, pos);
                let prefix_slot = crate::proof::tensor::geometry::block_table_slot(prefix_bt, pos);
                &&& (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)
                    < full_bt.len()
                &&& (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)
                    < prefix_bt.len()
                &&& crate::proof::tensor::geometry::slot_in_cache(full_k, full_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(full_v, full_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(prefix_k, prefix_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(prefix_v, prefix_slot)
                &&& crate::proof::tensor::geometry::cache_at(full_k, full_slot)
                    == crate::proof::tensor::geometry::cache_at(prefix_k, prefix_slot)
                &&& crate::proof::tensor::geometry::cache_at(full_v, full_slot)
                    == crate::proof::tensor::geometry::cache_at(prefix_v, prefix_slot)
            }
        by {
            assert(pos < prefix_len);
            crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, full_len);
            crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, prefix_len);
        }
        reveal(MODEL::paged_attention_repr);

        reveal(LAYERS::full_paged_attention_repr);
        reveal(LAYERS::sliding_window_paged_attention_repr);
        reveal(layer_attention_config_valid);
        reveal(attention_config_valid);
        match attention {
            AttentionConfigRepr::Full => {
                AO::lemma_singleton_equivalence(
            full_q, full_k, full_v, full_bt, full_len, j as nat as int,
            prefix_q, prefix_k, prefix_v, prefix_bt, prefix_len, j as nat as int,
            AttentionKind::Full, parameters, 0,
        );
            },
            AttentionConfigRepr::SlidingWindow(window_size) => {
                AO::lemma_singleton_equivalence(
            full_q, full_k, full_v, full_bt, full_len, j as nat as int,
            prefix_q, prefix_k, prefix_v, prefix_bt, prefix_len, j as nat as int,
            AttentionKind::SlidingWindow, parameters, window_size,
        );
            },
        }
    }
    assert(full_attn.subrange(0, prefix_len as int) =~= prefix_attn);
}

// Suffix-only singleton execution at absolute positions.  Both sides retain
// the complete causal cache; SlidingWindow changes the attention value but
// does not weaken this theorem's cache-equality premise.
pub proof fn paged_attention_singleton_suffix_invariance(
    full_q: Tensor2D,
    suffix_q: Tensor2D,
    full_k: KVCacheLayerRepr,
    full_v: KVCacheLayerRepr,
    suffix_k: KVCacheLayerRepr,
    suffix_v: KVCacheLayerRepr,
    full_bt: Seq<BlockId>,
    suffix_bt: Seq<BlockId>,
    full_len: nat,
    prefix_len: nat,
    attention: AttentionConfigRepr,
    parameters: AttentionParametersRepr,
)
    requires
        RT::paged_attention_numeric_domain(),
        layer_attention_config_valid(attention),
        0 < prefix_len < full_len,
        full_q.len() == full_len,
        suffix_q.len() == full_len - prefix_len,
        full_q.subrange(prefix_len as int, full_len as int) == suffix_q,
        crate::proof::tensor::geometry::blocks_needed_for(full_len) <= full_bt.len(),
        crate::proof::tensor::geometry::blocks_needed_for(full_len) <= suffix_bt.len(),
        forall|pos: nat|
            #![trigger crate::proof::tensor::geometry::block_table_slot(full_bt, pos)]
            #![trigger crate::proof::tensor::geometry::block_table_slot(suffix_bt, pos)]
            pos < full_len ==> {
                let full_slot = crate::proof::tensor::geometry::block_table_slot(full_bt, pos);
                let suffix_slot = crate::proof::tensor::geometry::block_table_slot(suffix_bt, pos);
                &&& crate::proof::tensor::geometry::slot_in_cache(full_k, full_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(full_v, full_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(suffix_k, suffix_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(suffix_v, suffix_slot)
                &&& crate::proof::tensor::geometry::cache_at(full_k, full_slot)
                    == crate::proof::tensor::geometry::cache_at(suffix_k, suffix_slot)
                &&& crate::proof::tensor::geometry::cache_at(full_v, full_slot)
                    == crate::proof::tensor::geometry::cache_at(suffix_v, suffix_slot)
            },
    ensures ({
        let suffix_len = (full_len - prefix_len) as nat;
        MODEL::paged_attention_repr(
            full_q, full_k, full_v,
            seq![0int, full_len as int], seq![0int, full_len as int],
            full_len, full_len, seq![full_bt], attention, parameters,
        ).subrange(prefix_len as int, full_len as int)
        == MODEL::paged_attention_repr(
            suffix_q, suffix_k, suffix_v,
            seq![0int, suffix_len as int], seq![0int, full_len as int],
            suffix_len, full_len, seq![suffix_bt], attention, parameters,
        )
    }),
{
    let suffix_len = (full_len - prefix_len) as nat;
    let full_attn = MODEL::paged_attention_repr(
        full_q, full_k, full_v,
        seq![0int, full_len as int], seq![0int, full_len as int],
        full_len, full_len, seq![full_bt], attention, parameters,
    );
    let suffix_attn = MODEL::paged_attention_repr(
        suffix_q, suffix_k, suffix_v,
        seq![0int, suffix_len as int], seq![0int, full_len as int],
        suffix_len, full_len, seq![suffix_bt], attention, parameters,
    );
    MODEL::lemma_paged_attention_repr_shape(
        full_q, full_k, full_v,
        seq![0int, full_len as int], seq![0int, full_len as int],
        full_len, full_len, seq![full_bt], attention, parameters,
    );
    MODEL::lemma_paged_attention_repr_shape(
        suffix_q, suffix_k, suffix_v,
        seq![0int, suffix_len as int], seq![0int, full_len as int],
        suffix_len, full_len, seq![suffix_bt], attention, parameters,
    );
    assert forall|j: int| 0 <= j < suffix_len as int implies
        full_attn[prefix_len as int + j] == suffix_attn[j]
    by {
        let full_j = prefix_len as int + j;
        assert(full_q[full_j] == suffix_q[j]);
        assert forall|pos: nat|
            #![trigger crate::proof::tensor::geometry::block_table_slot(full_bt, pos)]
            #![trigger crate::proof::tensor::geometry::block_table_slot(suffix_bt, pos)]
            pos as int <= full_j implies {
                let full_slot = crate::proof::tensor::geometry::block_table_slot(full_bt, pos);
                let suffix_slot = crate::proof::tensor::geometry::block_table_slot(suffix_bt, pos);
                &&& (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)
                    < full_bt.len()
                &&& (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)
                    < suffix_bt.len()
                &&& crate::proof::tensor::geometry::slot_in_cache(full_k, full_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(full_v, full_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(suffix_k, suffix_slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(suffix_v, suffix_slot)
                &&& crate::proof::tensor::geometry::cache_at(full_k, full_slot)
                    == crate::proof::tensor::geometry::cache_at(suffix_k, suffix_slot)
                &&& crate::proof::tensor::geometry::cache_at(full_v, full_slot)
                    == crate::proof::tensor::geometry::cache_at(suffix_v, suffix_slot)
            }
        by {
            assert(pos < full_len);
            crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, full_len);
        }
        reveal(MODEL::paged_attention_repr);

        reveal(LAYERS::full_paged_attention_repr);
        reveal(LAYERS::sliding_window_paged_attention_repr);
        reveal(layer_attention_config_valid);
        reveal(attention_config_valid);
        match attention {
            AttentionConfigRepr::Full => {
                AO::lemma_singleton_equivalence(
            full_q, full_k, full_v, full_bt, full_len, full_j as nat as int,
            suffix_q, suffix_k, suffix_v, suffix_bt, full_len, j as nat as int,
            AttentionKind::Full, parameters, 0,
        );
            },
            AttentionConfigRepr::SlidingWindow(window_size) => {
                AO::lemma_singleton_equivalence(
            full_q, full_k, full_v, full_bt, full_len, full_j as nat as int,
            suffix_q, suffix_k, suffix_v, suffix_bt, full_len, j as nat as int,
            AttentionKind::SlidingWindow, parameters, window_size,
        );
            },
        }
    }
    assert(full_attn.subrange(prefix_len as int, full_len as int)
        =~= suffix_attn);
}

// Request-major scatter establishes the generated ragged contract's exact
// indexed-cache premise.  Because ContractIR relates complete processed pages,
// this proof rounds k_len up to its full page span rather than silently
// weakening the premise to only logical positions below k_len.
pub proof fn selected_cache_pages_equal_from_request_major_store(
    kr: Tensor2D,
    vr: Tensor2D,
    old_k: KVCacheLayerRepr,
    old_v: KVCacheLayerRepr,
    slots: Seq<int>,
    bt_row: Seq<BlockId>,
    k_len: nat,
    lo: int,
    hi: int,
    attention: AttentionConfigRepr,
)
    requires
        kr.len() == vr.len(),
        vr.len() == slots.len(),
        0 <= lo <= hi <= slots.len(),
        crate::proof::tensor::geometry::blocks_needed_for(k_len) <= bt_row.len(),
        forall|m: int, l: int|
            #![trigger slots.subrange(0, lo)[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), bt_row[l]]
            0 <= m < lo && 0 <= l < bt_row.len() ==>
                slots.subrange(0, lo)[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != bt_row[l] as int,
        forall|m: int, l: int|
            #![trigger slots.subrange(hi, slots.len() as int)[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), bt_row[l]]
            0 <= m < slots.len() - hi && 0 <= l < bt_row.len() ==>
                slots.subrange(hi, slots.len() as int)[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != bt_row[l] as int,
        forall|pos: nat|
            pos < crate::proof::tensor::geometry::blocks_needed_for(k_len)
                * crate::types::BLOCK_SIZE_SPEC ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    old_k, #[trigger] crate::proof::tensor::geometry::block_table_slot(bt_row, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    old_v, crate::proof::tensor::geometry::block_table_slot(bt_row, pos),
                ),
    ensures
        paged_attention_selected_cache_pages_equal(
            RT::store_kv_cache_repr(kr, vr, old_k, old_v, slots).0,
            RT::store_kv_cache_repr(kr, vr, old_k, old_v, slots).1,
            bt_row,
            RT::store_kv_cache_repr(
                kr.subrange(lo, hi), vr.subrange(lo, hi),
                old_k, old_v, slots.subrange(lo, hi),
            ).0,
            RT::store_kv_cache_repr(
                kr.subrange(lo, hi), vr.subrange(lo, hi),
                old_k, old_v, slots.subrange(lo, hi),
            ).1,
            bt_row,
            k_len,
            attention,
        ),
{
    let page_span = crate::proof::tensor::geometry::blocks_needed_for(k_len)
        * crate::types::BLOCK_SIZE_SPEC;
    crate::proof::tensor::geometry::lemma_blocks_needed_for_full_pages(
        crate::proof::tensor::geometry::blocks_needed_for(k_len),
    );
    crate::proof::tensor::seq_flatten::lemma_seq_split3(kr, lo, hi);
    crate::proof::tensor::seq_flatten::lemma_seq_split3(vr, lo, hi);
    crate::proof::tensor::seq_flatten::lemma_seq_split3(slots, lo, hi);
    RT::store_agrees_at_all_block_pos(
        kr.subrange(0, lo), vr.subrange(0, lo),
        kr.subrange(lo, hi), vr.subrange(lo, hi),
        kr.subrange(hi, kr.len() as int), vr.subrange(hi, vr.len() as int),
        old_k, old_v,
        slots.subrange(0, lo), slots.subrange(lo, hi),
        slots.subrange(hi, slots.len() as int),
        bt_row, page_span,
    );
    let full = RT::store_kv_cache_repr(kr, vr, old_k, old_v, slots);
    let own = RT::store_kv_cache_repr(
        kr.subrange(lo, hi), vr.subrange(lo, hi),
        old_k, old_v, slots.subrange(lo, hi),
    );
    reveal(paged_attention_selected_cache_pages_equal);
    match attention {
        AttentionConfigRepr::Full => {
            reveal(GAC::paged_attention_selected_cache_pages_equal);
        },
        AttentionConfigRepr::SlidingWindow(window_size) => {
            reveal(GAC::paged_attention_selected_cache_pages_equal);
        },
    }
    assert forall|pos: nat| #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row, pos)]
        pos < page_span implies {
            let slot = crate::proof::tensor::geometry::block_table_slot(bt_row, pos);
            &&& crate::proof::tensor::geometry::cache_at(full.0, slot)
                == crate::proof::tensor::geometry::cache_at(own.0, slot)
            &&& crate::proof::tensor::geometry::cache_at(full.1, slot)
                == crate::proof::tensor::geometry::cache_at(own.1, slot)
    } by {}
}

// Any two scatters over the same base cache retain the same logical page
// geometry.  This lets the full launch's cache geometry establish the
// independently stored singleton launch without assuming it separately.
pub proof fn alternate_store_preserves_paged_cache_geometry(
    full_kr: Tensor2D,
    full_vr: Tensor2D,
    full_slots: Seq<int>,
    selected_kr: Tensor2D,
    selected_vr: Tensor2D,
    selected_slots: Seq<int>,
    old_k: KVCacheLayerRepr,
    old_v: KVCacheLayerRepr,
    attention: AttentionConfigRepr,
)
    requires
        paged_cache_geometry(
            RT::store_kv_cache_repr(
                full_kr, full_vr, old_k, old_v, full_slots,
            ).0,
            RT::store_kv_cache_repr(
                full_kr, full_vr, old_k, old_v, full_slots,
            ).1,
            attention,
        ),
    ensures
        paged_cache_geometry(
            RT::store_kv_cache_repr(
                selected_kr, selected_vr, old_k, old_v, selected_slots,
            ).0,
            RT::store_kv_cache_repr(
                selected_kr, selected_vr, old_k, old_v, selected_slots,
            ).1,
            attention,
        ),
        RT::store_kv_cache_repr(
            selected_kr, selected_vr, old_k, old_v, selected_slots,
        ).0.len() == RT::store_kv_cache_repr(
            full_kr, full_vr, old_k, old_v, full_slots,
        ).0.len(),
{
    let full = RT::store_kv_cache_repr(
        full_kr, full_vr, old_k, old_v, full_slots,
    );
    let selected = RT::store_kv_cache_repr(
        selected_kr, selected_vr, old_k, old_v, selected_slots,
    );
    RT::lemma_store_kv_cache_repr_shape_unconditional(
        full_kr, full_vr, old_k, old_v, full_slots,
    );
    RT::lemma_store_kv_cache_repr_shape_unconditional(
        selected_kr, selected_vr, old_k, old_v, selected_slots,
    );
    reveal(paged_cache_geometry);
    match attention {
        AttentionConfigRepr::Full => {
            reveal(GAC::paged_cache_geometry);
        },
        AttentionConfigRepr::SlidingWindow(window_size) => {
            reveal(GAC::paged_cache_geometry);
        },
    }
    assert forall|p: int| 0 <= p < selected.0.len() implies
        (#[trigger] selected.0[p]).len()
            == crate::types::BLOCK_SIZE_SPEC as int by {
        assert(selected.0[p].len() == old_k[p].len());
        assert(full.0[p].len() == old_k[p].len());
    }
    assert forall|p: int| 0 <= p < selected.1.len() implies
        (#[trigger] selected.1[p]).len()
            == crate::types::BLOCK_SIZE_SPEC as int by {
        assert(selected.1[p].len() == old_v[p].len());
        assert(full.1[p].len() == old_v[p].len());
    }
}

// Architecture-level launch domain after the exact four-norm pre-store stage and
// full-batch KV scatter.  This predicate intentionally says nothing about a
// separately stored singleton cache.
pub open spec fn decoder_layer_attention_launch_ready(
    common: LayerWeightsRepr,
    extension: FourNormGatedLayerExtensionRepr,
    hidden: Tensor2D,
    positions: IntTensor1D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
) -> bool
    recommends hidden.len() == positions.len(), hidden.len() == slots.len(),
{
    let pre = MODEL::attention_pre_store_repr(
        common, extension, hidden, positions,
    );
    let cache = RT::store_kv_cache_repr(
        pre.1, pre.2, k_cache, v_cache, slots,
    );
    match extension.attention {
        AttentionConfigRepr::Full => GAC::paged_attention_launch_ready(
            pre.0.len(), cache.0, cache.1, cu_q, cu_k,
            max_q, max_k, block_table,
        ),
        AttentionConfigRepr::SlidingWindow(window_size) =>
            GAC::swa_paged_attention_launch_ready(
                pre.0.len(), cache.0, cache.1, cu_q, cu_k,
                max_q, max_k, block_table,
                window_size,
            ),
    }
}

// Exact launch-domain fold for the full-batch four-norm layer chain.  The next
// predicate state follows the same hidden/cache update as `layer_chain_repr`,
// keeping kernel-domain assumptions adjacent to the layer that consumes them.
pub open spec fn layer_chain_attention_launch_ready(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    start: nat,
) -> bool
    recommends
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
    decreases (common_layers.len() - start) as nat,
{
    if start >= common_layers.len() {
        true
    } else {
        let layer = MODEL::decoder_layer_step_repr(
            common_layers[start as int], extension_layers[start as int],
            hidden, positions,
            caches[start as int].0, caches[start as int].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        decoder_layer_attention_launch_ready(
            common_layers[start as int], extension_layers[start as int],
            hidden, positions,
            caches[start as int].0, caches[start as int].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        )
        && layer_chain_attention_launch_ready(
            common_layers, extension_layers, layer.0, positions,
            caches.update(start as int, layer.1), slots,
            cu_q, cu_k, max_q, max_k, block_table, (start + 1) as nat,
        )
    }
}

// The scheduler already establishes the architecture-neutral paged-attention
// launch domain for every cache layer.  The four-norm decoder's pre-attention path preserves
// the query row count, and KV scatter preserves cache page geometry, so that
// common domain is sufficient for both the generated full-attention contract
// and the staged SWA contract.  The latter adds only its configured positive window
// launch parameter here; this lemma deliberately does not claim window-only
// cache dependence or justify SWA cache eviction.
pub proof fn layer_chain_attention_launch_ready_from_common_domain(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    start: nat,
)
    requires
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
        forall|i: int| start <= i < extension_layers.len() ==>
            layer_attention_config_valid(
                #[trigger] extension_layers[i].attention,
            ),
        forall|i: int| start <= i < common_layers.len() ==>
            #[trigger] RT::paged_attention_launch_ready(
                hidden.len(), caches[i].0, caches[i].1,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
    ensures
        layer_chain_attention_launch_ready(
            common_layers, extension_layers, hidden, positions, caches, slots,
            cu_q, cu_k, max_q, max_k, block_table, start,
        ),
    decreases (common_layers.len() - start) as nat,
{
    reveal(layer_chain_attention_launch_ready);
    if start < common_layers.len() {
        let common = common_layers[start as int];
        let extension = extension_layers[start as int];
        let pre = MODEL::attention_pre_store_repr(
            common, extension, hidden, positions,
        );
        MODEL::lemma_attention_pre_store_repr_shape(
            common, extension, hidden, positions,
        );
        assert(pre.1.len() == hidden.len());
        assert(pre.2.len() == hidden.len());
        assert(pre.1.len() == pre.2.len());
        assert(pre.2.len() == slots.len());
        let cache = RT::store_kv_cache_repr(
            pre.1, pre.2, caches[start as int].0,
            caches[start as int].1, slots,
        );
        RT::lemma_paged_attention_launch_ready_after_store(
            hidden.len(), pre.1, pre.2,
            caches[start as int].0, caches[start as int].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        reveal(decoder_layer_attention_launch_ready);
        reveal(RT::paged_attention_launch_ready);
        reveal(GAC::paged_attention_launch_ready);
        reveal(GAC::paged_attention_metadata_ready);
        reveal(GAC::paged_cache_geometry);
        reveal(GAC::swa_paged_attention_launch_ready);
        reveal(GAC::paged_attention_metadata_ready);
        reveal(GAC::paged_cache_geometry);
        match extension.attention {
            AttentionConfigRepr::Full => {
                assert(GAC::paged_attention_metadata_ready(
                    pre.0.len(), cache.0.len(), cu_q, cu_k,
                    max_q, max_k, block_table,
                ));
                assert(GAC::paged_attention_launch_ready(
                    pre.0.len(), cache.0, cache.1, cu_q, cu_k,
                    max_q, max_k, block_table,
                ));
            },
            AttentionConfigRepr::SlidingWindow(window_size) => {
                assert(GAC::paged_attention_metadata_ready(
                    pre.0.len(), cache.0.len(), cu_q, cu_k,
                    max_q, max_k, block_table,
                ));
                assert(window_size > 0);
                assert(GAC::swa_paged_attention_launch_ready(
                    pre.0.len(), cache.0, cache.1, cu_q, cu_k,
                    max_q, max_k, block_table,
                    window_size,
                ));
            },
        }
        assert(decoder_layer_attention_launch_ready(
            common, extension, hidden, positions,
            caches[start as int].0, caches[start as int].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ));

        let layer = MODEL::decoder_layer_step_repr(
            common, extension, hidden, positions,
            caches[start as int].0, caches[start as int].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        MODEL::lemma_decoder_layer_step_repr_shape(
            common, extension, hidden, positions,
            caches[start as int].0, caches[start as int].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        let next_caches = caches.update(start as int, layer.1);
        assert forall|i: int| (start + 1) as int <= i < common_layers.len()
            implies #[trigger] RT::paged_attention_launch_ready(
                layer.0.len(), next_caches[i].0, next_caches[i].1,
                cu_q, cu_k, max_q, max_k, block_table,
            ) by {
            assert(i != start as int);
            assert(0 <= i < next_caches.len());
            assert(next_caches[i] == caches[i]);
            assert(layer.0.len() == hidden.len());
        }
        layer_chain_attention_launch_ready_from_common_domain(
            common_layers, extension_layers, layer.0, positions, next_caches,
            slots, cu_q, cu_k, max_q, max_k, block_table, (start + 1) as nat,
        );
    }
}

// Full one-block request isolation with an independently executed singleton
// KV scatter.  Request-major block disjointness discharges the entire indexed
// cache relation; the singleton launch domain remains explicit so physical
// cache geometry is not smuggled into the architecture proof.
pub proof fn decoder_layer_request_isolation_from_layout(
    common: LayerWeightsRepr,
    extension: FourNormGatedLayerExtensionRepr,
    hidden: Tensor2D,
    positions: IntTensor1D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    i: nat,
)
    requires
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
        i < block_table.len(),
        0 <= cu_q[i as int] < cu_q[i as int + 1]
            <= hidden.len() as int,
        decoder_layer_attention_launch_ready(
            common, extension, hidden, positions, k_cache, v_cache, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
        forall|m: int, l: int|
            #![trigger slots.subrange(0, cu_q[i as int])[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
            0 <= m < cu_q[i as int]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(0, cu_q[i as int])[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
            0 <= m < slots.len() - cu_q[i as int + 1]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|pos: nat|
            pos < crate::proof::tensor::geometry::blocks_needed_for(
                (cu_k[i as int + 1] - cu_k[i as int]) as nat,
            ) * crate::types::BLOCK_SIZE_SPEC ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    k_cache,
                    #[trigger] crate::proof::tensor::geometry::block_table_slot(
                        block_table[i as int], pos,
                    ),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    v_cache,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                ),
    ensures ({
        let full = MODEL::decoder_layer_step_repr(
            common, extension, hidden, positions, k_cache, v_cache, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        let single = MODEL::decoder_layer_step_repr(
            common, extension,
            hidden.subrange(cu_q[i as int], cu_q[i as int + 1]),
            positions.subrange(cu_q[i as int], cu_q[i as int + 1]),
            k_cache, v_cache,
            slots.subrange(cu_q[i as int], cu_q[i as int + 1]),
            seq![0int, cu_q[i as int + 1] - cu_q[i as int]],
            seq![0int, cu_k[i as int + 1] - cu_k[i as int]],
            (cu_q[i as int + 1] - cu_q[i as int]) as nat,
            (cu_k[i as int + 1] - cu_k[i as int]) as nat,
            seq![block_table[i as int]],
        );
        &&& full.0.subrange(cu_q[i as int], cu_q[i as int + 1]) == single.0
        &&& paged_attention_selected_cache_pages_equal(
            full.1.0, full.1.1, block_table[i as int],
            single.1.0, single.1.1, block_table[i as int],
            (cu_k[i as int + 1] - cu_k[i as int]) as nat,
            extension.attention,
        )
    }),
{
    broadcast use {
        MODEL::lemma_attention_pre_store_repr_shape,
        MODEL::lemma_paged_attention_repr_shape,
    };
    let lo = cu_q[i as int];
    let hi = cu_q[i as int + 1];
    let q_len = (hi - lo) as nat;
    let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
    let pre = MODEL::attention_pre_store_repr(
        common, extension, hidden, positions,
    );
    let single_pre = MODEL::attention_pre_store_repr(
        common, extension, hidden.subrange(lo, hi),
        positions.subrange(lo, hi),
    );
    MODEL::lemma_attention_pre_store_repr_shape(
        common, extension, hidden, positions,
    );
    MODEL::lemma_attention_pre_store_repr_shape(
        common, extension, hidden.subrange(lo, hi),
        positions.subrange(lo, hi),
    );
    attention_pre_store_subrange_invariance(
        common, extension, hidden, positions, lo, hi,
    );
    let full_cache = RT::store_kv_cache_repr(
        pre.1, pre.2, k_cache, v_cache, slots,
    );
    let single_cache = RT::store_kv_cache_repr(
        single_pre.1, single_pre.2, k_cache, v_cache,
        slots.subrange(lo, hi),
    );
    selected_cache_pages_equal_from_request_major_store(
        pre.1, pre.2, k_cache, v_cache, slots,
        block_table[i as int], k_len, lo, hi,
        extension.attention,
    );
    assert(paged_attention_selected_cache_pages_equal(
        full_cache.0, full_cache.1, block_table[i as int],
        single_cache.0, single_cache.1, block_table[i as int],
        k_len, extension.attention,
    ));
    reveal(decoder_layer_attention_launch_ready);
    assert(paged_attention_launch_ready(
        pre.0, full_cache.0, full_cache.1,
        cu_q, cu_k, max_q, max_k, block_table,
        extension.attention,
    ));
    paged_attention_launch_ready_cache_geometry(
        pre.0, full_cache.0, full_cache.1,
        cu_q, cu_k, max_q, max_k, block_table,
        extension.attention,
    );
    alternate_store_preserves_paged_cache_geometry(
        pre.1, pre.2, slots,
        single_pre.1, single_pre.2, slots.subrange(lo, hi),
        k_cache, v_cache, extension.attention,
    );
    selected_attention_launch_ready_from_full(
        pre.0, full_cache.0, full_cache.1,
        single_cache.0, single_cache.1,
        cu_q, cu_k, max_q, max_k, block_table,
        extension.attention, i,
    );
    paged_attention_request_segment_relocation(
        pre.0, full_cache.0, full_cache.1,
        single_cache.0, single_cache.1,
        cu_q, cu_k, max_q, max_k, block_table,
        block_table[i as int], extension.attention,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale), i,
    );
    let attended = MODEL::paged_attention_repr(
        pre.0, full_cache.0, full_cache.1,
        cu_q, cu_k, max_q, max_k, block_table,
        extension.attention, crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale),
    );
    MODEL::lemma_paged_attention_repr_shape(
        pre.0, full_cache.0, full_cache.1,
        cu_q, cu_k, max_q, max_k, block_table,
        extension.attention, crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale),
    );
    post_attention_and_mlp_subrange_invariance(
        common, extension, hidden, attended, lo, hi,
    );
    reveal(MODEL::decoder_layer_step_repr);
}

// Relational lift of one-block isolation through the remaining four-norm layer
// chain.  The two runs may carry different cache state for already-processed
// layers, but must agree on the suffix that can still be read.  The result
// deliberately relates only the complete pages selected by this request's
// block-table row; it neither claims whole-pool equality nor a smaller SWA
// dependency window.
pub proof fn layer_chain_request_relocation_from_layout(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    full_caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    selected_caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    i: nat,
    start: nat,
)
    requires
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        full_caches.len() >= common_layers.len(),
        selected_caches.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
        i < block_table.len(),
        0 <= cu_q[i as int] < cu_q[i as int + 1]
            <= hidden.len() as int,
        layer_chain_attention_launch_ready(
            common_layers, extension_layers, hidden, positions, full_caches,
            slots, cu_q, cu_k, max_q, max_k, block_table, start,
        ),
        forall|layer: int| start <= layer < common_layers.len() ==>
            #[trigger] full_caches[layer] == selected_caches[layer],
        forall|m: int, l: int|
            #![trigger slots.subrange(0, cu_q[i as int])[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
            0 <= m < cu_q[i as int]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(0, cu_q[i as int])[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
            0 <= m < slots.len() - cu_q[i as int + 1]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|layer: int, pos: nat|
            #![trigger full_caches[layer].0,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos)]
            start <= layer < common_layers.len()
                && pos < crate::proof::tensor::geometry::blocks_needed_for(
                    (cu_k[i as int + 1] - cu_k[i as int]) as nat,
                ) * crate::types::BLOCK_SIZE_SPEC ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    full_caches[layer].0,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    full_caches[layer].1,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                ),
    ensures ({
        let lo = cu_q[i as int];
        let hi = cu_q[i as int + 1];
        let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
        let full = MODEL::layer_chain_repr(
            common_layers, extension_layers, hidden, positions, full_caches,
            slots, cu_q, cu_k, max_q, max_k, block_table, start,
        );
        let selected = MODEL::layer_chain_repr(
            common_layers, extension_layers, hidden.subrange(lo, hi),
            positions.subrange(lo, hi), selected_caches,
            slots.subrange(lo, hi), seq![0int, hi - lo],
            seq![0int, cu_k[i as int + 1] - cu_k[i as int]],
            (hi - lo) as nat, k_len, seq![block_table[i as int]], start,
        );
        &&& full.0.subrange(lo, hi) == selected.0
        &&& forall|layer: int| start <= layer < common_layers.len() ==>
            #[trigger] paged_attention_selected_cache_pages_equal(
                full.1[layer].0, full.1[layer].1, block_table[i as int],
                selected.1[layer].0, selected.1[layer].1,
                block_table[i as int], k_len,
                extension_layers[layer].attention,
            )
    }),
    decreases (common_layers.len() - start) as nat,
{
    let lo = cu_q[i as int];
    let hi = cu_q[i as int + 1];
    let q_len = (hi - lo) as nat;
    let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
    let single_hidden = hidden.subrange(lo, hi);
    let single_positions = positions.subrange(lo, hi);
    let single_slots = slots.subrange(lo, hi);
    let single_cu_q = seq![0int, q_len as int];
    let single_cu_k = seq![0int, k_len as int];
    let single_bt = seq![block_table[i as int]];
    if start < common_layers.len() {
        let layer = start as int;
        assert(full_caches[layer] == selected_caches[layer]);
        let full_step = MODEL::decoder_layer_step_repr(
            common_layers[layer], extension_layers[layer], hidden, positions,
            full_caches[layer].0, full_caches[layer].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        let single_step = MODEL::decoder_layer_step_repr(
            common_layers[layer], extension_layers[layer],
            single_hidden, single_positions,
            selected_caches[layer].0, selected_caches[layer].1, single_slots,
            single_cu_q, single_cu_k, q_len, k_len, single_bt,
        );
        reveal(layer_chain_attention_launch_ready);
        decoder_layer_request_isolation_from_layout(
            common_layers[layer], extension_layers[layer], hidden, positions,
            full_caches[layer].0, full_caches[layer].1, slots,
            cu_q, cu_k, max_q, max_k, block_table, i,
        );
        assert(full_step.0.subrange(lo, hi) == single_step.0);
        assert(paged_attention_selected_cache_pages_equal(
            full_step.1.0, full_step.1.1, block_table[i as int],
            single_step.1.0, single_step.1.1, block_table[i as int],
            k_len, extension_layers[layer].attention,
        ));
        MODEL::lemma_decoder_layer_step_repr_shape(
            common_layers[layer], extension_layers[layer], hidden, positions,
            full_caches[layer].0, full_caches[layer].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        MODEL::lemma_decoder_layer_step_repr_shape(
            common_layers[layer], extension_layers[layer],
            single_hidden, single_positions,
            selected_caches[layer].0, selected_caches[layer].1, single_slots,
            single_cu_q, single_cu_k, q_len, k_len, single_bt,
        );
        let full_next_caches = full_caches.update(layer, full_step.1);
        let selected_next_caches = selected_caches.update(layer, single_step.1);
        assert forall|later: int|
            (start + 1) as int <= later < common_layers.len() implies
                #[trigger] full_next_caches[later]
                    == selected_next_caches[later] by {
            assert(later != layer);
            assert(full_next_caches[later] == full_caches[later]);
            assert(selected_next_caches[later] == selected_caches[later]);
        }
        assert forall|later: int, pos: nat|
            #![trigger full_next_caches[later].0,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos)]
            (start + 1) as int <= later < common_layers.len()
                && pos < crate::proof::tensor::geometry::blocks_needed_for(k_len)
                    * crate::types::BLOCK_SIZE_SPEC implies
                crate::proof::tensor::geometry::slot_in_cache(
                    full_next_caches[later].0,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    full_next_caches[later].1,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                ) by {
            assert(later != layer);
            assert(full_next_caches[later] == full_caches[later]);
            let read_slot = crate::proof::tensor::geometry::block_table_slot(
                block_table[i as int], pos,
            );
            assert(crate::proof::tensor::geometry::slot_in_cache(
                full_caches[later].0, read_slot,
            ));
            assert(crate::proof::tensor::geometry::slot_in_cache(
                full_caches[later].1, read_slot,
            ));
        }
        layer_chain_request_relocation_from_layout(
            common_layers, extension_layers, full_step.0, positions,
            full_next_caches, selected_next_caches, slots,
            cu_q, cu_k, max_q, max_k, block_table, i,
            (start + 1) as nat,
        );
        let full_tail = MODEL::layer_chain_repr(
            common_layers, extension_layers, full_step.0, positions,
            full_next_caches, slots, cu_q, cu_k, max_q, max_k,
            block_table, (start + 1) as nat,
        );
        let selected_tail = MODEL::layer_chain_repr(
            common_layers, extension_layers, single_step.0, single_positions,
            selected_next_caches, single_slots, single_cu_q, single_cu_k,
            q_len, k_len, single_bt, (start + 1) as nat,
        );
        MODEL::lemma_layer_chain_cache_before_start_unchanged(
            common_layers, extension_layers, full_step.0, positions,
            full_next_caches, slots, cu_q, cu_k, max_q, max_k,
            block_table, (start + 1) as nat, layer,
        );
        MODEL::lemma_layer_chain_cache_before_start_unchanged(
            common_layers, extension_layers, single_step.0, single_positions,
            selected_next_caches, single_slots, single_cu_q, single_cu_k,
            q_len, k_len, single_bt, (start + 1) as nat, layer,
        );
        assert forall|target: int| start <= target < common_layers.len()
            implies #[trigger] paged_attention_selected_cache_pages_equal(
                full_tail.1[target].0, full_tail.1[target].1,
                block_table[i as int],
                selected_tail.1[target].0, selected_tail.1[target].1,
                block_table[i as int], k_len,
                extension_layers[target].attention,
            ) by {
            if target == layer {
                assert(full_tail.1[target] == full_step.1);
                assert(selected_tail.1[target] == single_step.1);
            } else {
                assert((start + 1) as int <= target);
            }
        }
    }
}

// Same-initial-cache corollary used by batched model projection.  Keeping this
// wrapper separate makes the cache relocation premise explicit in the
// induction while preserving a compact public theorem for the common case.
pub proof fn layer_chain_request_isolation_from_layout(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    i: nat,
    start: nat,
)
    requires
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
        i < block_table.len(),
        0 <= cu_q[i as int] < cu_q[i as int + 1]
            <= hidden.len() as int,
        layer_chain_attention_launch_ready(
            common_layers, extension_layers, hidden, positions, caches, slots,
            cu_q, cu_k, max_q, max_k, block_table, start,
        ),
        forall|m: int, l: int|
            #![trigger slots.subrange(0, cu_q[i as int])[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
            0 <= m < cu_q[i as int]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(0, cu_q[i as int])[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
            0 <= m < slots.len() - cu_q[i as int + 1]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|layer: int, pos: nat|
            #![trigger caches[layer].0,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos)]
            start <= layer < common_layers.len()
                && pos < crate::proof::tensor::geometry::blocks_needed_for(
                    (cu_k[i as int + 1] - cu_k[i as int]) as nat,
                ) * crate::types::BLOCK_SIZE_SPEC ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    caches[layer].0,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    caches[layer].1,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                ),
    ensures ({
        let lo = cu_q[i as int];
        let hi = cu_q[i as int + 1];
        let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
        let full = MODEL::layer_chain_repr(
            common_layers, extension_layers, hidden, positions, caches, slots,
            cu_q, cu_k, max_q, max_k, block_table, start,
        );
        let single = MODEL::layer_chain_repr(
            common_layers, extension_layers,
            hidden.subrange(lo, hi), positions.subrange(lo, hi),
            caches, slots.subrange(lo, hi), seq![0int, hi - lo],
            seq![0int, cu_k[i as int + 1] - cu_k[i as int]],
            (hi - lo) as nat, k_len, seq![block_table[i as int]], start,
        );
        &&& full.0.subrange(lo, hi) == single.0
        &&& forall|layer: int| start <= layer < common_layers.len() ==>
            #[trigger] paged_attention_selected_cache_pages_equal(
                full.1[layer].0, full.1[layer].1, block_table[i as int],
                single.1[layer].0, single.1[layer].1,
                block_table[i as int], k_len,
                extension_layers[layer].attention,
            )
    }),
{
    assert forall|layer: int| start <= layer < common_layers.len() implies
        #[trigger] caches[layer] == caches[layer] by {}
    layer_chain_request_relocation_from_layout(
        common_layers, extension_layers, hidden, positions, caches, caches,
        slots, cu_q, cu_k, max_q, max_k, block_table, i, start,
    );
}

// End-to-end cache-state projection for one four-norm request.  Every processed
// layer in the batched forward agrees with an independently executed singleton
// forward on the complete pages selected by that request's block-table row.
// This is intentionally weaker than global cache equality and, for SWA,
// intentionally no stronger than the retained full causal-prefix model.
pub proof fn model_forward_kv_request_isolation_from_layout(
    wr: ModelWeightsRepr,
    config: FourNormGatedDecoderConfigRepr,
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
    requires
        config.layers.len() == wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
        i < block_table.len(),
        0 <= cu_q[i as int] < cu_q[i as int + 1]
            <= input_ids.len() as int,
        layer_chain_attention_launch_ready(
            wr.layers, config.layers,
            MODEL::scaled_embed_repr(
                input_ids, wr.embed_weight, config.geometry.hidden_size,
            ),
            positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, 0,
        ),
        forall|m: int, l: int|
            #![trigger slots.subrange(0, cu_q[i as int])[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
            0 <= m < cu_q[i as int]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(0, cu_q[i as int])[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
            0 <= m < slots.len() - cu_q[i as int + 1]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|layer: int, pos: nat|
            #![trigger pre_kv[layer].0,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos)]
            0 <= layer < wr.layers.len()
                && pos < crate::proof::tensor::geometry::blocks_needed_for(
                    (cu_k[i as int + 1] - cu_k[i as int]) as nat,
                ) * crate::types::BLOCK_SIZE_SPEC ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    pre_kv[layer].0,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    pre_kv[layer].1,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                ),
    ensures ({
        let lo = cu_q[i as int];
        let hi = cu_q[i as int + 1];
        let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
        let full = MODEL::model_forward_kv_reprs(
            wr, config, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        let single = MODEL::model_forward_kv_reprs(
            wr, config, input_ids.subrange(lo, hi),
            positions.subrange(lo, hi), pre_kv, slots.subrange(lo, hi),
            seq![0int, hi - lo],
            seq![0int, cu_k[i as int + 1] - cu_k[i as int]],
            (hi - lo) as nat, k_len, seq![block_table[i as int]],
        );
        forall|layer: int| 0 <= layer < wr.layers.len() ==>
            #[trigger] paged_attention_selected_cache_pages_equal(
                full[layer].0, full[layer].1, block_table[i as int],
                single[layer].0, single[layer].1, block_table[i as int],
                k_len, config.layers[layer].attention,
            )
    }),
{
    broadcast use {
        MODEL::lemma_scaled_embed_repr_shape,
    };
    let lo = cu_q[i as int];
    let hi = cu_q[i as int + 1];
    let embedded = MODEL::scaled_embed_repr(
        input_ids, wr.embed_weight, config.geometry.hidden_size,
    );
    scaled_embed_subrange_invariance(
        input_ids, wr.embed_weight, config.geometry.hidden_size, lo, hi,
    );
    assert(embedded.len() == input_ids.len());
    assert forall|layer: int, pos: nat|
        #![trigger pre_kv[layer].0,
            crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos)]
        0 <= layer < wr.layers.len()
            && pos < crate::proof::tensor::geometry::blocks_needed_for(
                (cu_k[i as int + 1] - cu_k[i as int]) as nat,
            ) * crate::types::BLOCK_SIZE_SPEC implies
            crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].0,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].1,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
            ) by {}
    layer_chain_request_isolation_from_layout(
        wr.layers, config.layers, embedded, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, i, 0,
    );
    reveal(MODEL::model_forward_kv_reprs);
    reveal(MODEL::model_forward_hidden_and_kv_reprs);
}

// End-to-end four-norm text logits projection for one request.  The embedding,
// final norm, and tied LM head are row-local; the layer-chain theorem above is
// the only non-elementwise step.  This remains a semantic theorem and does not
// make a model family engine-reachable.
pub proof fn model_forward_logits_request_isolation_from_layout(
    wr: ModelWeightsRepr,
    config: FourNormGatedDecoderConfigRepr,
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
    requires
        config.layers.len() == wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
        i < block_table.len(),
        0 <= cu_q[i as int] < cu_q[i as int + 1]
            <= input_ids.len() as int,
        layer_chain_attention_launch_ready(
            wr.layers, config.layers,
            MODEL::scaled_embed_repr(
                input_ids, wr.embed_weight, config.geometry.hidden_size,
            ),
            positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, 0,
        ),
        forall|m: int, l: int|
            #![trigger slots.subrange(0, cu_q[i as int])[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
            0 <= m < cu_q[i as int]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(0, cu_q[i as int])[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), block_table[i as int][l]]
            0 <= m < slots.len() - cu_q[i as int + 1]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(cu_q[i as int + 1], slots.len() as int)[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|layer: int, pos: nat|
            #![trigger pre_kv[layer].0,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos)]
            0 <= layer < wr.layers.len()
                && pos < crate::proof::tensor::geometry::blocks_needed_for(
                    (cu_k[i as int + 1] - cu_k[i as int]) as nat,
                ) * crate::types::BLOCK_SIZE_SPEC ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    pre_kv[layer].0,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    pre_kv[layer].1,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                ),
    ensures
        MODEL::model_forward_logits_repr(
            wr, config, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ).subrange(cu_q[i as int], cu_q[i as int + 1])
        == MODEL::model_forward_logits_repr(
            wr, config,
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
    broadcast use {
        MODEL::lemma_scaled_embed_repr_shape,
    };
    let lo = cu_q[i as int];
    let hi = cu_q[i as int + 1];
    let embedded = MODEL::scaled_embed_repr(
        input_ids, wr.embed_weight, config.geometry.hidden_size,
    );
    let full_chain = MODEL::layer_chain_repr(
        wr.layers, config.layers, embedded, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, 0,
    );
    scaled_embed_subrange_invariance(
        input_ids, wr.embed_weight, config.geometry.hidden_size, lo, hi,
    );
    layer_chain_request_isolation_from_layout(
        wr.layers, config.layers, embedded, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, i, 0,
    );
    MODEL::lemma_layer_chain_repr_shape(
        wr.layers, config.layers, embedded, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, 0,
    );
    FOUR_NORM::final_logits_subrange_invariance(
        full_chain.0, wr.final_norm, wr.lm_head,
        config.final_norm_policy, config.final_logit_softcap, lo, hi,
    );
    reveal(MODEL::model_forward_logits_repr);
    reveal(MODEL::model_forward_hidden_and_kv_reprs);
}

pub proof fn merge_attention_heads_subrange_invariance(
    input: Tensor2D,
    a: int,
    b: int,
)
    requires 0 <= a <= b <= input.len(),
    ensures MODEL::merge_attention_heads_repr(input.subrange(a, b))
        == MODEL::merge_attention_heads_repr(input).subrange(a, b),
{
    reveal(MODEL::merge_attention_heads_repr);
    assert(MODEL::merge_attention_heads_repr(input.subrange(a, b))
        =~= MODEL::merge_attention_heads_repr(input).subrange(a, b));
}

pub proof fn gelu_tanh_mul_subrange_invariance(
    gate: Tensor2D,
    up: Tensor2D,
    a: int,
    b: int,
)
    requires 0 <= a <= b <= gate.len(), gate.len() == up.len(),
    ensures MODEL::gelu_tanh_mul_repr(gate.subrange(a, b), up.subrange(a, b))
        == MODEL::gelu_tanh_mul_repr(gate, up).subrange(a, b),
{
    reveal(MODEL::gelu_tanh_mul_repr);
    assert(MODEL::gelu_tanh_mul_repr(gate.subrange(a, b), up.subrange(a, b))
        =~= MODEL::gelu_tanh_mul_repr(gate, up).subrange(a, b));
}

pub proof fn add_subrange_invariance(
    left: Tensor2D,
    right: Tensor2D,
    a: int,
    b: int,
)
    requires 0 <= a <= b <= left.len(), left.len() == right.len(),
    ensures MODEL::add_repr(left.subrange(a, b), right.subrange(a, b))
        == MODEL::add_repr(left, right).subrange(a, b),
{
    reveal(MODEL::add_repr);
    assert(MODEL::add_repr(left.subrange(a, b), right.subrange(a, b))
        =~= MODEL::add_repr(left, right).subrange(a, b));
}

pub open spec fn request_projection_ready(
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
) -> bool {
    &&& family.layers.len() == wr.layers.len()
    &&& forall|layer: int| 0 <= layer < family.layers.len() ==>
        layer_attention_config_valid(
            #[trigger] family.layers[layer].attention,
        )
    &&& pre_kv.len() >= wr.layers.len()
    &&& input_ids.len() == positions.len()
    &&& input_ids.len() == slots.len()
    &&& i < block_table.len()
    &&& 0 <= cu_q[i as int] < cu_q[i as int + 1]
        <= input_ids.len() as int
    &&& layer_chain_attention_launch_ready(
        wr.layers, family.layers,
        MODEL::scaled_embed_repr(
            input_ids, wr.embed_weight, family.geometry.hidden_size,
        ),
        positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, 0,
    )
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
            && pos < crate::proof::tensor::geometry::blocks_needed_for(
                (cu_k[i as int + 1] - cu_k[i as int]) as nat,
            ) * crate::types::BLOCK_SIZE_SPEC ==>
            crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].0,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].1,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
            )
}

} // verus!
