//! Shared four-norm gated decoder relocation proofs.
//!
//! The fold is parameterized by immutable row/attention policies. Attention
//! uses the common generated raw operator and checked projection lemmas;
//! admitting a new profile still requires exact kernel-interface qualification.
//! Full causal-prefix cache retention remains the conservative contract.

use crate::model_config::AttentionKind;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::model as MODEL;
#[cfg(verus_only)]
use crate::boundary::attention_operator as AO;
#[cfg(verus_only)]
use crate::boundary::dense_layer_primitives as LAYERS;
#[cfg(verus_only)]
use crate::{types::{BLOCK_SIZE_SPEC}, proof::tensor::geometry::{block_table_slot, blocks_needed_for, cache_at, slot_in_cache}};
#[cfg(verus_only)]
use crate::proof::model::family_layout as LAYOUT;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// Singleton selected-row equality for both four-norm attention variants.  The SWA
// branch deliberately consumes equality for every logical K/V position through
// the causal endpoint, even though the source mask also enforces the configured
// window.  This is the relocation theorem needed today; it is not an eviction
// certificate.
pub open spec fn paged_attention_singleton_repr(
    q: Tensor2D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    q_len: nat,
    k_len: nat,
    bt_row: Seq<BlockId>,
    attention: AttentionConfigRepr,
    parameters: AttentionParametersRepr,
) -> Tensor2D {
    MODEL::paged_attention_repr(
        q, k_cache, v_cache,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row], attention, parameters,
    )
}

pub proof fn paged_attention_singleton_equivalence(
    q_a: Tensor2D,
    q_b: Tensor2D,
    k_cache_a: KVCacheLayerRepr,
    v_cache_a: KVCacheLayerRepr,
    k_cache_b: KVCacheLayerRepr,
    v_cache_b: KVCacheLayerRepr,
    q_len_a: nat,
    k_len_a: nat,
    q_len_b: nat,
    k_len_b: nat,
    bt_row_a: Seq<BlockId>,
    bt_row_b: Seq<BlockId>,
    j_a: nat,
    j_b: nat,
    attention: AttentionConfigRepr,
    parameters: AttentionParametersRepr,
)
    requires
        RT::paged_attention_numeric_domain(),
        layer_attention_config_valid(attention),
        q_a.len() == q_len_a,
        q_b.len() == q_len_b,
        q_len_a > 0,
        q_len_b > 0,
        j_a < q_len_a,
        j_b < q_len_b,
        q_len_a <= k_len_a,
        q_len_b <= k_len_b,
        crate::proof::tensor::geometry::blocks_needed_for(k_len_a) <= bt_row_a.len(),
        crate::proof::tensor::geometry::blocks_needed_for(k_len_b) <= bt_row_b.len(),
        q_a[j_a as int] == q_b[j_b as int],
        k_len_a as int - q_len_a as int + j_a as int
            == k_len_b as int - q_len_b as int + j_b as int,
        forall|pos: nat| #![auto]
            pos as int <= k_len_a as int - q_len_a as int + j_a as int ==>
                (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)
                    < bt_row_a.len()
                && (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)
                    < bt_row_b.len()
                && crate::proof::tensor::geometry::slot_in_cache(
                    k_cache_a,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    v_cache_a,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    k_cache_b,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    v_cache_b,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::cache_at(
                    k_cache_a,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    k_cache_b,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::cache_at(
                    v_cache_a,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    v_cache_b,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                ),
    ensures
        paged_attention_singleton_repr(
            q_a, k_cache_a, v_cache_a,
            q_len_a, k_len_a, bt_row_a, attention, parameters,
        )[j_a as int] == paged_attention_singleton_repr(
            q_b, k_cache_b, v_cache_b,
            q_len_b, k_len_b, bt_row_b, attention, parameters,
        )[j_b as int],
{
    reveal(paged_attention_singleton_repr);
    reveal(MODEL::paged_attention_repr);

    reveal(LAYERS::full_paged_attention_repr);
    reveal(LAYERS::sliding_window_paged_attention_repr);
    match attention {
        AttentionConfigRepr::Full => {
            AO::lemma_singleton_equivalence(
            q_a, k_cache_a, v_cache_a, bt_row_a, k_len_a, j_a as int,
            q_b, k_cache_b, v_cache_b, bt_row_b, k_len_b, j_b as int,
            AttentionKind::Full, parameters, 0,
        );
        },
        AttentionConfigRepr::SlidingWindow(window_size) => {
            AO::lemma_singleton_equivalence(
            q_a, k_cache_a, v_cache_a, bt_row_a, k_len_a, j_a as int,
            q_b, k_cache_b, v_cache_b, bt_row_b, k_len_b, j_b as int,
            AttentionKind::SlidingWindow, parameters, window_size,
        );
        },
    }
}

pub proof fn paged_attention_physical_relocation(
    q: Tensor2D,
    k_cache_a: KVCacheLayerRepr,
    v_cache_a: KVCacheLayerRepr,
    k_cache_b: KVCacheLayerRepr,
    v_cache_b: KVCacheLayerRepr,
    q_len: nat,
    k_len: nat,
    bt_row_a: Seq<BlockId>,
    bt_row_b: Seq<BlockId>,
    attention: AttentionConfigRepr,
    parameters: AttentionParametersRepr,
)
    requires
        RT::paged_attention_numeric_domain(),
        layer_attention_config_valid(attention),
        q.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        crate::proof::tensor::geometry::blocks_needed_for(k_len) <= bt_row_a.len(),
        crate::proof::tensor::geometry::blocks_needed_for(k_len) <= bt_row_b.len(),
        forall|pos: nat| #![auto] pos < k_len ==>
            (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)
                < bt_row_a.len()
            && (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)
                < bt_row_b.len()
            && crate::proof::tensor::geometry::slot_in_cache(
                k_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                v_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                k_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                v_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
            )
            && crate::proof::tensor::geometry::cache_at(
                k_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
            ) == crate::proof::tensor::geometry::cache_at(
                k_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
            )
            && crate::proof::tensor::geometry::cache_at(
                v_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
            ) == crate::proof::tensor::geometry::cache_at(
                v_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
            ),
    ensures
        paged_attention_singleton_repr(
            q, k_cache_a, v_cache_a,
            q_len, k_len, bt_row_a, attention, parameters,
        ) == paged_attention_singleton_repr(
            q, k_cache_b, v_cache_b,
            q_len, k_len, bt_row_b, attention, parameters,
        ),
{
    MODEL::lemma_paged_attention_repr_shape(
        q, k_cache_a, v_cache_a,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row_a], attention, parameters,
    );
    MODEL::lemma_paged_attention_repr_shape(
        q, k_cache_b, v_cache_b,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row_b], attention, parameters,
    );
    assert forall|j: int| 0 <= j < q_len as int implies
        paged_attention_singleton_repr(
            q, k_cache_a, v_cache_a,
            q_len, k_len, bt_row_a, attention, parameters,
        )[j] == paged_attention_singleton_repr(
            q, k_cache_b, v_cache_b,
            q_len, k_len, bt_row_b, attention, parameters,
        )[j]
    by {
        assert forall|pos: nat| #![auto]
            pos as int <= k_len as int - q_len as int + j implies
                (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)
                    < bt_row_a.len()
                && (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)
                    < bt_row_b.len()
                && crate::proof::tensor::geometry::slot_in_cache(
                    k_cache_a,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    v_cache_a,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    k_cache_b,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    v_cache_b,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::cache_at(
                    k_cache_a,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    k_cache_b,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::cache_at(
                    v_cache_a,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    v_cache_b,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
        by {
            assert(pos < k_len);
        }
        paged_attention_singleton_equivalence(
            q, q,
            k_cache_a, v_cache_a, k_cache_b, v_cache_b,
            q_len, k_len, q_len, k_len,
            bt_row_a, bt_row_b, j as nat, j as nat,
            attention, parameters,
        );
    }
    assert(paged_attention_singleton_repr(
        q, k_cache_a, v_cache_a,
        q_len, k_len, bt_row_a, attention, parameters,
    ) =~= paged_attention_singleton_repr(
        q, k_cache_b, v_cache_b,
        q_len, k_len, bt_row_b, attention, parameters,
    ));
}

// Relocate one complete four-norm block between two physical cache layouts.  The
// row-local stages are identical functions of the shared hidden input; only
// the post-store attention read requires a relational certificate.
pub proof fn decoder_layer_relocation_from_layout(
    common: LayerWeightsRepr,
    extension: FourNormGatedLayerExtensionRepr,
    hidden: Tensor2D,
    positions: IntTensor1D,
    k_cache_a: KVCacheLayerRepr,
    v_cache_a: KVCacheLayerRepr,
    slots_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    k_cache_b: KVCacheLayerRepr,
    v_cache_b: KVCacheLayerRepr,
    slots_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        layer_attention_config_valid(extension.attention),
        hidden.len() == positions.len(),
        slots_a.len() == hidden.len(),
        slots_b.len() == hidden.len(),
        hidden.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        blocks_needed_for(k_len) <= bt_row_a.len(),
        blocks_needed_for(k_len) <= bt_row_b.len(),
        forall|j: int| 0 <= j < q_len as int ==>
            block_table_slot(
                bt_row_a, (k_len - q_len + j as nat) as nat,
            ) == #[trigger] slots_a[j] as nat
            && block_table_slot(
                bt_row_b, (k_len - q_len + j as nat) as nat,
            ) == slots_b[j] as nat,
        forall|j: int| 0 <= j < q_len as int ==>
            #[trigger] slots_a[j] >= 0 && slots_b[j] >= 0,
        forall|j: int, m: int| #![trigger slots_a[m], slots_a[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slots_a[m] != slots_a[j],
        forall|j: int, m: int| #![trigger slots_b[m], slots_b[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slots_b[m] != slots_b[j],
        forall|j: int| 0 <= j < q_len as int ==>
            slot_in_cache(k_cache_a, #[trigger] slots_a[j] as nat)
            && slot_in_cache(v_cache_a, slots_a[j] as nat)
            && slot_in_cache(k_cache_b, #[trigger] slots_b[j] as nat)
            && slot_in_cache(v_cache_b, slots_b[j] as nat),
        forall|pos: nat| #![trigger block_table_slot(bt_row_a, pos)]
            pos < k_len - q_len ==>
                slot_in_cache(k_cache_a, block_table_slot(bt_row_a, pos))
                && slot_in_cache(v_cache_a, block_table_slot(bt_row_a, pos))
                && slot_in_cache(k_cache_b, block_table_slot(bt_row_b, pos))
                && slot_in_cache(v_cache_b, block_table_slot(bt_row_b, pos))
                && cache_at(k_cache_a, block_table_slot(bt_row_a, pos))
                    == cache_at(k_cache_b, block_table_slot(bt_row_b, pos))
                && cache_at(v_cache_a, block_table_slot(bt_row_a, pos))
                    == cache_at(v_cache_b, block_table_slot(bt_row_b, pos)),
        LAYOUT::fresh_writes_miss_cached_prefix(
            slots_a, bt_row_a, slots_b, bt_row_b,
            (k_len - q_len) as int,
        ),
    ensures
        MODEL::decoder_layer_step_repr(
            common, extension, hidden, positions,
            k_cache_a, v_cache_a, slots_a,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a],
        ).0 == MODEL::decoder_layer_step_repr(
            common, extension, hidden, positions,
            k_cache_b, v_cache_b, slots_b,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b],
        ).0,
        LAYOUT::cache_pair_logical_prefix_equal(
            MODEL::decoder_layer_step_repr(
                common, extension, hidden, positions,
                k_cache_a, v_cache_a, slots_a,
                seq![0int, q_len as int], seq![0int, k_len as int],
                q_len, k_len, seq![bt_row_a],
            ).1,
            bt_row_a,
            MODEL::decoder_layer_step_repr(
                common, extension, hidden, positions,
                k_cache_b, v_cache_b, slots_b,
                seq![0int, q_len as int], seq![0int, k_len as int],
                q_len, k_len, seq![bt_row_b],
            ).1,
            bt_row_b,
            k_len,
        ),
{
    let pre = MODEL::attention_pre_store_repr(
        common, extension, hidden, positions,
    );
    MODEL::lemma_attention_pre_store_repr_shape(
        common, extension, hidden, positions,
    );
    let post_a = RT::store_kv_cache_repr(
        pre.1, pre.2, k_cache_a, v_cache_a, slots_a,
    );
    let post_b = RT::store_kv_cache_repr(
        pre.1, pre.2, k_cache_b, v_cache_b, slots_b,
    );
    assert forall|pos: nat| #![trigger block_table_slot(bt_row_a, pos)]
        pos < k_len implies
            (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_a.len()
            && (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_b.len()
            && slot_in_cache(post_a.0, block_table_slot(bt_row_a, pos))
            && slot_in_cache(post_a.1, block_table_slot(bt_row_a, pos))
            && slot_in_cache(post_b.0, block_table_slot(bt_row_b, pos))
            && slot_in_cache(post_b.1, block_table_slot(bt_row_b, pos))
            && cache_at(post_a.0, block_table_slot(bt_row_a, pos))
                == cache_at(post_b.0, block_table_slot(bt_row_b, pos))
            && cache_at(post_a.1, block_table_slot(bt_row_a, pos))
                == cache_at(post_b.1, block_table_slot(bt_row_b, pos))
    by {
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, k_len);
        if pos < k_len - q_len {
            LAYOUT::fresh_writes_miss_cached_prefix_at(
                slots_a, bt_row_a, slots_b, bt_row_b,
                (k_len - q_len) as int, pos,
            );
            RT::prefix_positions_relocation_agree(
                pre.1, pre.2,
                k_cache_a, v_cache_a, slots_a, bt_row_a,
                k_cache_b, v_cache_b, slots_b, bt_row_b, pos,
            );
        } else {
            let j = (pos - (k_len - q_len)) as int;
            assert((k_len - q_len + j as nat) as nat == pos);
            RT::fresh_positions_relocation_agree(
                pre.1, pre.2,
                k_cache_a, v_cache_a, slots_a, bt_row_a,
                k_cache_b, v_cache_b, slots_b, bt_row_b,
                q_len, k_len, j,
            );
        }
    }
    paged_attention_physical_relocation(
        pre.0, post_a.0, post_a.1, post_b.0, post_b.1,
        q_len, k_len, bt_row_a, bt_row_b,
        extension.attention, crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale),
    );
    reveal(MODEL::decoder_layer_step_repr);
    assert(LAYOUT::cache_pair_logical_prefix_equal(
        post_a, bt_row_a, post_b, bt_row_b, k_len,
    )) by {
        reveal(LAYOUT::cache_pair_logical_prefix_equal);
    }
}

// Relational layer fold for one request.  Each side updates only its current
// layer cache; later-layer cache premises therefore pass through Seq::update
// unchanged while the hidden tensors remain equal in lockstep.
pub proof fn layer_chain_relocation_from_layout(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches_a: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    caches_b: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
    start: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        forall|layer: int| start <= layer < extension_layers.len() ==>
            layer_attention_config_valid(
                #[trigger] extension_layers[layer].attention,
            ),
        caches_a.len() >= common_layers.len(),
        caches_b.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        slots_a.len() == hidden.len(),
        slots_b.len() == hidden.len(),
        hidden.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        blocks_needed_for(k_len) <= bt_row_a.len(),
        blocks_needed_for(k_len) <= bt_row_b.len(),
        forall|j: int| 0 <= j < q_len as int ==>
            block_table_slot(
                bt_row_a, (k_len - q_len + j as nat) as nat,
            ) == #[trigger] slots_a[j] as nat
            && block_table_slot(
                bt_row_b, (k_len - q_len + j as nat) as nat,
            ) == slots_b[j] as nat,
        forall|j: int| 0 <= j < q_len as int ==>
            #[trigger] slots_a[j] >= 0 && slots_b[j] >= 0,
        forall|j: int, m: int| #![trigger slots_a[m], slots_a[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slots_a[m] != slots_a[j],
        forall|j: int, m: int| #![trigger slots_b[m], slots_b[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slots_b[m] != slots_b[j],
        LAYOUT::fresh_writes_miss_cached_prefix(
            slots_a, bt_row_a, slots_b, bt_row_b,
            (k_len - q_len) as int,
        ),
        forall|layer: int, j: int| #![trigger caches_a[layer].0, slots_a[j]]
            start <= layer < common_layers.len()
                && 0 <= j < q_len as int ==>
                slot_in_cache(caches_a[layer].0, slots_a[j] as nat)
                && slot_in_cache(caches_a[layer].1, slots_a[j] as nat)
                && slot_in_cache(caches_b[layer].0, slots_b[j] as nat)
                && slot_in_cache(caches_b[layer].1, slots_b[j] as nat),
        forall|layer: int, pos: nat|
            #![trigger caches_a[layer].0, block_table_slot(bt_row_a, pos)]
            start <= layer < common_layers.len()
                && pos < k_len - q_len ==>
                slot_in_cache(
                    caches_a[layer].0, block_table_slot(bt_row_a, pos),
                )
                && slot_in_cache(
                    caches_a[layer].1, block_table_slot(bt_row_a, pos),
                )
                && slot_in_cache(
                    caches_b[layer].0, block_table_slot(bt_row_b, pos),
                )
                && slot_in_cache(
                    caches_b[layer].1, block_table_slot(bt_row_b, pos),
                )
                && cache_at(
                    caches_a[layer].0, block_table_slot(bt_row_a, pos),
                ) == cache_at(
                    caches_b[layer].0, block_table_slot(bt_row_b, pos),
                )
                && cache_at(
                    caches_a[layer].1, block_table_slot(bt_row_a, pos),
                ) == cache_at(
                    caches_b[layer].1, block_table_slot(bt_row_b, pos),
                ),
    ensures ({
        let out_a = MODEL::layer_chain_repr(
            common_layers, extension_layers, hidden, positions,
            caches_a, slots_a,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a], start,
        );
        let out_b = MODEL::layer_chain_repr(
            common_layers, extension_layers, hidden, positions,
            caches_b, slots_b,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b], start,
        );
        &&& out_a.0 == out_b.0
        &&& forall|target: int| start <= target < common_layers.len() ==>
            #[trigger] LAYOUT::cache_pair_logical_prefix_equal(
                out_a.1[target],
                bt_row_a,
                out_b.1[target],
                bt_row_b,
                k_len,
            )
    }),
    decreases (common_layers.len() - start) as nat,
{
    if start < common_layers.len() {
        let layer = start as int;
        decoder_layer_relocation_from_layout(
            common_layers[layer], extension_layers[layer], hidden, positions,
            caches_a[layer].0, caches_a[layer].1, slots_a, bt_row_a,
            caches_b[layer].0, caches_b[layer].1, slots_b, bt_row_b,
            q_len, k_len,
        );
        let step_a = MODEL::decoder_layer_step_repr(
            common_layers[layer], extension_layers[layer], hidden, positions,
            caches_a[layer].0, caches_a[layer].1, slots_a,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a],
        );
        let step_b = MODEL::decoder_layer_step_repr(
            common_layers[layer], extension_layers[layer], hidden, positions,
            caches_b[layer].0, caches_b[layer].1, slots_b,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b],
        );
        assert(step_a.0 == step_b.0);
        MODEL::lemma_decoder_layer_step_repr_shape(
            common_layers[layer], extension_layers[layer], hidden, positions,
            caches_a[layer].0, caches_a[layer].1, slots_a,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a],
        );
        let next_a = caches_a.update(layer, step_a.1);
        let next_b = caches_b.update(layer, step_b.1);
        assert forall|later: int, j: int|
            #![trigger next_a[later].0, slots_a[j]]
            (start + 1) as int <= later < common_layers.len()
                && 0 <= j < q_len as int implies
                slot_in_cache(next_a[later].0, slots_a[j] as nat)
                && slot_in_cache(next_a[later].1, slots_a[j] as nat)
                && slot_in_cache(next_b[later].0, slots_b[j] as nat)
                && slot_in_cache(next_b[later].1, slots_b[j] as nat)
        by {
            assert(later != layer);
            assert(next_a[later] == caches_a[later]);
            assert(next_b[later] == caches_b[later]);
        }
        assert forall|later: int, pos: nat|
            #![trigger next_a[later].0, block_table_slot(bt_row_a, pos)]
            (start + 1) as int <= later < common_layers.len()
                && pos < k_len - q_len implies
                slot_in_cache(next_a[later].0, block_table_slot(bt_row_a, pos))
                && slot_in_cache(next_a[later].1, block_table_slot(bt_row_a, pos))
                && slot_in_cache(next_b[later].0, block_table_slot(bt_row_b, pos))
                && slot_in_cache(next_b[later].1, block_table_slot(bt_row_b, pos))
                && cache_at(next_a[later].0, block_table_slot(bt_row_a, pos))
                    == cache_at(next_b[later].0, block_table_slot(bt_row_b, pos))
                && cache_at(next_a[later].1, block_table_slot(bt_row_a, pos))
                    == cache_at(next_b[later].1, block_table_slot(bt_row_b, pos))
        by {
            assert(later != layer);
            assert(next_a[later] == caches_a[later]);
            assert(next_b[later] == caches_b[later]);
        }
        layer_chain_relocation_from_layout(
            common_layers, extension_layers, step_a.0, positions,
            next_a, slots_a, bt_row_a,
            next_b, slots_b, bt_row_b,
            q_len, k_len, (start + 1) as nat,
        );
        let tail_a = MODEL::layer_chain_repr(
            common_layers, extension_layers, step_a.0, positions,
            next_a, slots_a,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a], (start + 1) as nat,
        );
        let tail_b = MODEL::layer_chain_repr(
            common_layers, extension_layers, step_b.0, positions,
            next_b, slots_b,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b], (start + 1) as nat,
        );
        MODEL::lemma_layer_chain_cache_before_start_unchanged(
            common_layers, extension_layers, step_a.0, positions,
            next_a, slots_a,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a],
            (start + 1) as nat, layer,
        );
        MODEL::lemma_layer_chain_cache_before_start_unchanged(
            common_layers, extension_layers, step_b.0, positions,
            next_b, slots_b,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b],
            (start + 1) as nat, layer,
        );
        assert forall|target: int|
            start <= target < common_layers.len() implies
                #[trigger] LAYOUT::cache_pair_logical_prefix_equal(
                    tail_a.1[target], bt_row_a,
                    tail_b.1[target], bt_row_b, k_len,
                )
        by {
            if target == layer {
                assert(tail_a.1[target] == step_a.1);
                assert(tail_b.1[target] == step_b.1);
            } else {
                assert((start + 1) as int <= target);
            }
        }
        reveal(MODEL::layer_chain_repr);
    }
}

// Whole-model four-norm relocation. The checked forward fold remains opaque here:
// this theorem alone unfolds the four-norm layer fold and discharges every
// attention read from the shared logical-prefix layout contract.
pub proof fn model_forward_relocation_from_layout(
    wr: ModelWeightsRepr,
    config: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    caches_a: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    caches_b: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        config.layers.len() == wr.layers.len(),
        forall|layer: int| 0 <= layer < config.layers.len() ==>
            layer_attention_config_valid(
                #[trigger] config.layers[layer].attention,
            ),
        caches_a.len() >= wr.layers.len(),
        caches_b.len() >= wr.layers.len(),
        input_ids.len() == positions.len(),
        slots_a.len() == input_ids.len(),
        slots_b.len() == input_ids.len(),
        input_ids.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        blocks_needed_for(k_len) <= bt_row_a.len(),
        blocks_needed_for(k_len) <= bt_row_b.len(),
        forall|j: int| 0 <= j < q_len as int ==>
            block_table_slot(
                bt_row_a, (k_len - q_len + j as nat) as nat,
            ) == #[trigger] slots_a[j] as nat
            && block_table_slot(
                bt_row_b, (k_len - q_len + j as nat) as nat,
            ) == slots_b[j] as nat,
        forall|j: int| 0 <= j < q_len as int ==>
            #[trigger] slots_a[j] >= 0 && slots_b[j] >= 0,
        forall|j: int, m: int| #![trigger slots_a[m], slots_a[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slots_a[m] != slots_a[j],
        forall|j: int, m: int| #![trigger slots_b[m], slots_b[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slots_b[m] != slots_b[j],
        LAYOUT::fresh_writes_miss_cached_prefix(
            slots_a, bt_row_a, slots_b, bt_row_b,
            (k_len - q_len) as int,
        ),
        forall|layer: int, j: int| #![trigger caches_a[layer].0, slots_a[j]]
            0 <= layer < wr.layers.len() && 0 <= j < q_len as int ==>
                slot_in_cache(caches_a[layer].0, slots_a[j] as nat)
                && slot_in_cache(caches_a[layer].1, slots_a[j] as nat)
                && slot_in_cache(caches_b[layer].0, slots_b[j] as nat)
                && slot_in_cache(caches_b[layer].1, slots_b[j] as nat),
        forall|layer: int, pos: nat|
            #![trigger caches_a[layer].0, block_table_slot(bt_row_a, pos)]
            0 <= layer < wr.layers.len() && pos < k_len - q_len ==>
                slot_in_cache(
                    caches_a[layer].0, block_table_slot(bt_row_a, pos),
                )
                && slot_in_cache(
                    caches_a[layer].1, block_table_slot(bt_row_a, pos),
                )
                && slot_in_cache(
                    caches_b[layer].0, block_table_slot(bt_row_b, pos),
                )
                && slot_in_cache(
                    caches_b[layer].1, block_table_slot(bt_row_b, pos),
                )
                && cache_at(
                    caches_a[layer].0, block_table_slot(bt_row_a, pos),
                ) == cache_at(
                    caches_b[layer].0, block_table_slot(bt_row_b, pos),
                )
                && cache_at(
                    caches_a[layer].1, block_table_slot(bt_row_a, pos),
                ) == cache_at(
                    caches_b[layer].1, block_table_slot(bt_row_b, pos),
                ),
    ensures
        MODEL::model_forward_logits_repr(
            wr, config, input_ids, positions,
            caches_a, slots_a,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a],
        ) == MODEL::model_forward_logits_repr(
            wr, config, input_ids, positions,
            caches_b, slots_b,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b],
        ),
        LAYOUT::cache_sequence_logical_prefix_equal(
            MODEL::model_forward_kv_reprs(
                wr, config, input_ids, positions,
                caches_a, slots_a,
                seq![0int, q_len as int], seq![0int, k_len as int],
                q_len, k_len, seq![bt_row_a],
            ),
            bt_row_a,
            MODEL::model_forward_kv_reprs(
                wr, config, input_ids, positions,
                caches_b, slots_b,
                seq![0int, q_len as int], seq![0int, k_len as int],
                q_len, k_len, seq![bt_row_b],
            ),
            bt_row_b,
            wr.layers.len(),
            k_len,
        ),
{
    let hidden = MODEL::scaled_embed_repr(
        input_ids, wr.embed_weight, config.geometry.hidden_size,
    );
    reveal(MODEL::scaled_embed_repr);
    assert(hidden.len() == input_ids.len());
    layer_chain_relocation_from_layout(
        wr.layers, config.layers, hidden, positions,
        caches_a, slots_a, bt_row_a,
        caches_b, slots_b, bt_row_b,
        q_len, k_len, 0,
    );
    MODEL::lemma_model_forward_kv_reprs_len(
        wr, config, input_ids, positions, caches_a, slots_a,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row_a],
    );
    MODEL::lemma_model_forward_kv_reprs_len(
        wr, config, input_ids, positions, caches_b, slots_b,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row_b],
    );
    reveal(MODEL::model_forward_logits_repr);
    reveal(MODEL::model_forward_kv_reprs);
    reveal(MODEL::model_forward_hidden_and_kv_reprs);
    let out_a = MODEL::model_forward_kv_reprs(
        wr, config, input_ids, positions,
        caches_a, slots_a,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row_a],
    );
    let out_b = MODEL::model_forward_kv_reprs(
        wr, config, input_ids, positions,
        caches_b, slots_b,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row_b],
    );
    assert(LAYOUT::cache_sequence_logical_prefix_equal(
        out_a,
        bt_row_a,
        out_b,
        bt_row_b,
        wr.layers.len(),
        k_len,
    )) by {
        reveal(LAYOUT::cache_sequence_logical_prefix_equal);
        assert forall|layer: int| 0 <= layer < wr.layers.len()
            implies #[trigger] LAYOUT::cache_pair_logical_prefix_equal(
                out_a[layer],
                bt_row_a,
                out_b[layer],
                bt_row_b,
                k_len,
            ) by {}
    }
}


} // verus!
