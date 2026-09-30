//! Architecture dispatch for engine-to-independent-machine forward relocation.
//!
//! Consumers establish one common physical-layout contract and call one opaque
//! theorem. The closed match delegates through paired family adapters and does
//! not depend on either family proof development directly.

#[cfg(verus_only)]
use crate::proof::tensor::geometry::{block_table_slot, blocks_needed_for, cache_at, slot_in_cache};
#[cfg(verus_only)]
use crate::proof::model::architecture as MA;
#[cfg(verus_only)]
use crate::proof::model::families::dense_swiglu as DENSE;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::layer_witnesses as FOUR;
#[cfg(verus_only)]
use crate::proof::model::family_layout as LAYOUT;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// Common physical-layout domain for relocating one singleton forward between
// two cache geometries.  The family clause contains only the semantic shape
// facts that differ across architectures; all cache-address obligations are
// shared.
pub open spec fn model_forward_singleton_relocation_ready(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
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
) -> bool {
    &&& RT::paged_attention_numeric_domain()
    &&& MA::request_projection_configuration_ready(
        wr,
        architecture_repr,
    )
    &&& caches_a.len() >= wr.layers.len()
    &&& caches_b.len() >= wr.layers.len()
    &&& input_ids.len() == positions.len()
    &&& slots_a.len() == input_ids.len()
    &&& slots_b.len() == input_ids.len()
    &&& input_ids.len() == q_len
    &&& q_len > 0
    &&& q_len <= k_len
    &&& blocks_needed_for(k_len) <= bt_row_a.len()
    &&& blocks_needed_for(k_len) <= bt_row_b.len()
    &&& forall|j: int| 0 <= j < q_len as int ==>
        block_table_slot(
            bt_row_a, (k_len - q_len + j as nat) as nat,
        ) == #[trigger] slots_a[j] as nat
        && block_table_slot(
            bt_row_b, (k_len - q_len + j as nat) as nat,
        ) == slots_b[j] as nat
    &&& forall|j: int| 0 <= j < q_len as int ==>
        #[trigger] slots_a[j] >= 0 && slots_b[j] >= 0
    &&& forall|j: int, m: int| #![trigger slots_a[m], slots_a[j]]
        0 <= j < q_len as int && j < m < q_len as int ==>
            slots_a[m] != slots_a[j]
    &&& forall|j: int, m: int| #![trigger slots_b[m], slots_b[j]]
        0 <= j < q_len as int && j < m < q_len as int ==>
            slots_b[m] != slots_b[j]
    &&& LAYOUT::fresh_writes_miss_cached_prefix(
        slots_a, bt_row_a, slots_b, bt_row_b,
        (k_len - q_len) as int,
    )
    &&& forall|layer: int, j: int|
        #![trigger caches_a[layer].0, slots_a[j]]
        0 <= layer < wr.layers.len() && 0 <= j < q_len as int ==>
            slot_in_cache(caches_a[layer].0, slots_a[j] as nat)
            && slot_in_cache(caches_a[layer].1, slots_a[j] as nat)
            && slot_in_cache(caches_b[layer].0, slots_b[j] as nat)
            && slot_in_cache(caches_b[layer].1, slots_b[j] as nat)
    &&& forall|layer: int, pos: nat|
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
            )
}

pub proof fn lemma_model_forward_singleton_relocation(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
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
    requires model_forward_singleton_relocation_ready(
        wr, architecture_repr, input_ids, positions,
        caches_a, slots_a, bt_row_a,
        caches_b, slots_b, bt_row_b,
        q_len, k_len,
    ),
    ensures
        MA::model_forward_logits_repr(
            wr, architecture_repr, input_ids, positions,
            caches_a, slots_a,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a],
        ) == MA::model_forward_logits_repr(
            wr, architecture_repr, input_ids, positions,
            caches_b, slots_b,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b],
        ),
        LAYOUT::cache_sequence_logical_prefix_equal(
            MA::model_forward_kv_reprs(
                wr, architecture_repr, input_ids, positions,
                caches_a, slots_a,
                seq![0int, q_len as int], seq![0int, k_len as int],
                q_len, k_len, seq![bt_row_a],
            ),
            bt_row_a,
            MA::model_forward_kv_reprs(
                wr, architecture_repr, input_ids, positions,
                caches_b, slots_b,
                seq![0int, q_len as int], seq![0int, k_len as int],
                q_len, k_len, seq![bt_row_b],
            ),
            bt_row_b,
            wr.layers.len(),
            k_len,
        ),
{
    reveal(model_forward_singleton_relocation_ready);
    reveal(MA::request_projection_configuration_ready);
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_forward_relocation(
                wr, family, input_ids, positions,
                caches_a, slots_a, bt_row_a,
                caches_b, slots_b, bt_row_b,
                q_len, k_len,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_forward_relocation(
                wr, family, input_ids, positions,
                caches_a, slots_a, bt_row_a,
                caches_b, slots_b, bt_row_b,
                q_len, k_len,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(config) => {
            FOUR::lemma_forward_relocation(
                wr, config.decoder, input_ids, positions,
                caches_a, slots_a, bt_row_a, caches_b, slots_b, bt_row_b,
                q_len, k_len);
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            FOUR::lemma_forward_relocation(
                wr, family, input_ids, positions,
                caches_a, slots_a, bt_row_a,
                caches_b, slots_b, bt_row_b,
                q_len, k_len,
            );
        },
    }
}

// Common request/layout domain for composing batched request projection with
// singleton physical relocation. SWA retains exactly the same full logical
// causal-prefix premise; no window-only cache fact appears here.
pub open spec fn model_forward_engine_to_machine_ready(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    engine_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    engine_slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    request_slots: Seq<int>,
    request_index: nat,
    machine_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    machine_slots: Seq<int>,
    machine_bt_row: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
) -> bool {
    &&& q_len == cu_q[request_index as int + 1] - cu_q[request_index as int]
    &&& k_len == cu_k[request_index as int + 1] - cu_k[request_index as int]
    &&& request_slots == engine_slots.subrange(
        cu_q[request_index as int], cu_q[request_index as int + 1],
    )
    &&& MA::model_forward_request_projection_domain(
        wr, architecture_repr, input_ids, positions, engine_kv, engine_slots,
        cu_q, cu_k, max_q, max_k, block_table, request_index,
    )
    &&& model_forward_singleton_relocation_ready(
        wr, architecture_repr,
        input_ids.subrange(
            cu_q[request_index as int], cu_q[request_index as int + 1],
        ),
        positions.subrange(
            cu_q[request_index as int], cu_q[request_index as int + 1],
        ),
        engine_kv, request_slots, block_table[request_index as int],
        machine_kv, machine_slots, machine_bt_row,
        q_len, k_len,
    )
}

pub proof fn lemma_model_forward_engine_to_machine(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    engine_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    engine_slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    request_slots: Seq<int>,
    request_index: nat,
    machine_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    machine_slots: Seq<int>,
    machine_bt_row: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
)
    requires model_forward_engine_to_machine_ready(
        wr, architecture_repr, input_ids, positions,
        engine_kv, engine_slots, cu_q, cu_k, max_q, max_k, block_table,
        request_slots, request_index,
        machine_kv, machine_slots, machine_bt_row, q_len, k_len,
    ),
    ensures
        MA::model_forward_logits_repr(
            wr, architecture_repr, input_ids, positions,
            engine_kv, engine_slots, cu_q, cu_k, max_q, max_k, block_table,
        ).subrange(
            cu_q[request_index as int], cu_q[request_index as int + 1],
        ) == MA::model_forward_logits_repr(
            wr, architecture_repr,
            input_ids.subrange(
                cu_q[request_index as int], cu_q[request_index as int + 1],
            ),
            positions.subrange(
                cu_q[request_index as int], cu_q[request_index as int + 1],
            ),
            machine_kv, machine_slots,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![machine_bt_row],
        ),
        LAYOUT::cache_sequence_logical_prefix_equal(
            MA::model_forward_kv_reprs(
                wr, architecture_repr, input_ids, positions,
                engine_kv, engine_slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
            block_table[request_index as int],
            MA::model_forward_kv_reprs(
                wr, architecture_repr,
                input_ids.subrange(
                    cu_q[request_index as int],
                    cu_q[request_index as int + 1],
                ),
                positions.subrange(
                    cu_q[request_index as int],
                    cu_q[request_index as int + 1],
                ),
                machine_kv, machine_slots,
                seq![0int, q_len as int], seq![0int, k_len as int],
                q_len, k_len, seq![machine_bt_row],
            ),
            machine_bt_row,
            wr.layers.len(),
            k_len,
        ),
{
    reveal(model_forward_engine_to_machine_ready);
    let lo = cu_q[request_index as int];
    let hi = cu_q[request_index as int + 1];
    let request_input_ids = input_ids.subrange(lo, hi);
    let request_positions = positions.subrange(lo, hi);
    let engine_bt_row = block_table[request_index as int];
    let single_cu_q = seq![0int, q_len as int];
    let single_cu_k = seq![0int, k_len as int];
    let single_engine_bt = seq![engine_bt_row];
    let single_machine_bt = seq![machine_bt_row];
    let full_post = MA::model_forward_kv_reprs(
        wr, architecture_repr, input_ids, positions,
        engine_kv, engine_slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
    let single_engine_post = MA::model_forward_kv_reprs(
        wr, architecture_repr, request_input_ids, request_positions,
        engine_kv, request_slots,
        single_cu_q, single_cu_k, q_len, k_len, single_engine_bt,
    );
    let machine_post = MA::model_forward_kv_reprs(
        wr, architecture_repr, request_input_ids, request_positions,
        machine_kv, machine_slots,
        single_cu_q, single_cu_k, q_len, k_len, single_machine_bt,
    );
    MA::lemma_model_forward_logits_request_isolation(
        wr, architecture_repr, input_ids, positions,
        engine_kv, engine_slots,
        cu_q, cu_k, max_q, max_k, block_table, request_index,
    );
    MA::lemma_model_forward_kv_request_isolation(
        wr, architecture_repr, input_ids, positions,
        engine_kv, engine_slots,
        cu_q, cu_k, max_q, max_k, block_table, request_index,
    );
    assert(model_forward_singleton_relocation_ready(
        wr, architecture_repr, request_input_ids, request_positions,
        engine_kv, request_slots, engine_bt_row,
        machine_kv, machine_slots, machine_bt_row,
        q_len, k_len,
    )) by {
        reveal(model_forward_singleton_relocation_ready);
        reveal(MA::model_forward_request_projection_domain);
        reveal(MA::request_projection_configuration_ready);
        match architecture_repr {
            ModelWeightsArchitectureRepr::Qwen3(_) => {},
            ModelWeightsArchitectureRepr::Llama3(_) => {},
            ModelWeightsArchitectureRepr::Gemma3Text(_) => {},
            ModelWeightsArchitectureRepr::Gemma4Text(_) => {},
        }
    }
    lemma_model_forward_singleton_relocation(
        wr, architecture_repr, request_input_ids, request_positions,
        engine_kv, request_slots, engine_bt_row,
        machine_kv, machine_slots, machine_bt_row,
        q_len, k_len,
    );
    LAYOUT::lemma_cache_sequence_logical_prefix_equal_transitive(
        full_post, engine_bt_row,
        single_engine_post, engine_bt_row,
        machine_post, machine_bt_row,
        wr.layers.len(), k_len,
    );
}

} // verus!
