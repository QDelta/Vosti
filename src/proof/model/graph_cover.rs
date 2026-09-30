//! Architecture-neutral pure-decode CUDA-graph covering refinement.
//
// A captured decode graph may have more rows than the current decode step.
// The runtime copies the real batch into the prefix, appends one-token dummy
// requests, gives every dummy KV row slot -1, and discards their selected
// logits.  CUDA capture/replay fidelity remains a separate runtime premise;
// this module proves the semantic part of the optimization: the dummy suffix
// cannot change the real logits or the complete post-forward KV cache.

use crate::proof::tensor::geometry::*;
use crate::exec::engine::{Engine, StepReprs};
use crate::proof::model::architecture as MA;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::assert_seqs_equal;
use vstd::prelude::*;

verus! {

pub open spec fn cover_int_rows(rows: Seq<int>, pads: nat, value: int) -> Seq<int> {
    rows + Seq::new(pads, |_i: int| value)
}

pub open spec fn cover_cumulative(rows: Seq<int>, pads: nat) -> Seq<int>
    recommends rows.len() > 0,
{
    let end = rows[rows.len() - 1];
    rows + Seq::new(pads, |i: int| end + i + 1)
}

pub open spec fn cover_block_table(
    rows: Seq<Seq<BlockId>>,
    pads: nat,
) -> Seq<Seq<BlockId>> {
    rows + Seq::new(pads, |_i: int| seq![0u64])
}

pub open spec fn covered_model_forward_logits(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    pads: nat,
) -> Tensor2D {
    MA::model_forward_logits_repr(
        wr, architecture_repr,
        cover_int_rows(input_ids, pads, 0),
        cover_int_rows(positions, pads, 0),
        pre_kv,
        cover_int_rows(slots, pads, -1),
        cover_cumulative(cu_q, pads),
        cover_cumulative(cu_k, pads),
        max_q,
        max_k,
        cover_block_table(bt, pads),
    )
}

pub open spec fn covered_model_forward_kv_reprs(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    pads: nat,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> {
    MA::model_forward_kv_reprs(
        wr, architecture_repr,
        cover_int_rows(input_ids, pads, 0),
        cover_int_rows(positions, pads, 0),
        pre_kv,
        cover_int_rows(slots, pads, -1),
        cover_cumulative(cu_q, pads),
        cover_cumulative(cu_k, pads),
        max_q,
        max_k,
        cover_block_table(bt, pads),
    )
}

pub open spec fn covered_selected_logits(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    pads: nat,
    real_seqs: nat,
) -> Tensor2D {
    let logits = covered_model_forward_logits(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt, pads,
    );
    let cuq2 = cover_cumulative(cu_q, pads);
    Seq::new(real_seqs, |i: int|
        RT::select_sample_logits_repr(logits, cuq2, i as nat))
}

pub proof fn lemma_zero_cover_inputs(
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    bt: Seq<Seq<BlockId>>,
)
    requires cu_q.len() > 0, cu_k.len() > 0,
    ensures
        cover_int_rows(input_ids, 0, 0) == input_ids,
        cover_int_rows(positions, 0, 0) == positions,
        cover_int_rows(slots, 0, -1) == slots,
        cover_cumulative(cu_q, 0) == cu_q,
        cover_cumulative(cu_k, 0) == cu_k,
        cover_block_table(bt, 0) == bt,
{
}

pub proof fn lemma_zero_covered_model_forward(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    real_seqs: nat,
)
    requires cu_q.len() > 0, cu_k.len() > 0,
    ensures
        covered_model_forward_logits(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, 0,
        ) == MA::model_forward_logits_repr(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt,
        ),
        covered_model_forward_kv_reprs(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, 0,
        ) == MA::model_forward_kv_reprs(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt,
        ),
        covered_selected_logits(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, 0, real_seqs,
        ) == Seq::new(real_seqs, |i: int|
            RT::select_sample_logits_repr(
                MA::model_forward_logits_repr(
                    wr, architecture_repr, input_ids, positions, pre_kv, slots,
                    cu_q, cu_k, max_q, max_k, bt,
                ),
                cu_q,
                i as nat,
            )),
{
    lemma_zero_cover_inputs(input_ids, positions, slots, cu_q, cu_k, bt);
}

pub open spec fn decode_cover_row_ready(
    wr: ModelWeightsRepr,
    input_ids: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    i: int,
) -> bool
    recommends
        0 <= i < bt.len(),
        cu_q.len() == bt.len() + 1,
        cu_k.len() == bt.len() + 1,
        slots.len() == input_ids.len(),
{
    let qd = cu_q[i + 1] - cu_q[i];
    let kd = cu_k[i + 1] - cu_k[i];
    &&& 0 <= cu_q[i] < cu_q[i + 1] <= input_ids.len() as int
    &&& 0 <= cu_k[i] < cu_k[i + 1]
    &&& qd <= max_q as int
    &&& kd <= max_k as int
    &&& qd <= kd
    &&& blocks_needed_for(kd as nat) <= bt[i].len()
    &&& (forall|m: int, l: int|
        #![trigger slots.subrange(0, cu_q[i])[m]
            / (BLOCK_SIZE_SPEC as int), bt[i][l]]
        0 <= m < cu_q[i] && 0 <= l < bt[i].len() ==>
            slots.subrange(0, cu_q[i])[m] / (BLOCK_SIZE_SPEC as int)
                != bt[i][l] as int)
    &&& (forall|m: int, l: int|
        #![trigger slots.subrange(cu_q[i + 1], slots.len() as int)[m]
            / (BLOCK_SIZE_SPEC as int), bt[i][l]]
        0 <= m < slots.len() - cu_q[i + 1]
            && 0 <= l < bt[i].len() ==>
            slots.subrange(cu_q[i + 1], slots.len() as int)[m]
                / (BLOCK_SIZE_SPEC as int) != bt[i][l] as int)
    &&& (forall|ell: int, pos: nat|
        #![trigger pre_kv[ell].0, block_table_slot(bt[i], pos)]
        0 <= ell < wr.layers.len() && pos < kd as nat ==>
            slot_in_cache(pre_kv[ell].0, block_table_slot(bt[i], pos))
            && slot_in_cache(pre_kv[ell].1, block_table_slot(bt[i], pos)))
    &&& (forall|ell: int, pos: nat|
        #![trigger pre_kv[ell].0, block_table_slot(bt[i], pos)]
        0 <= ell < wr.layers.len()
            && pos < blocks_needed_for(kd as nat) * BLOCK_SIZE_SPEC ==>
            slot_in_cache(pre_kv[ell].0, block_table_slot(bt[i], pos))
            && slot_in_cache(pre_kv[ell].1, block_table_slot(bt[i], pos)))
}

// Exact proof domain for a scheduler-produced pure-decode plan on which a
// larger captured graph is allowed to cover.  Kept opaque so the quantified
// request-isolation premises do not leak into ordinary engine queries.
#[verifier::opaque]
pub open spec fn decode_cover_ready(
    wr: ModelWeightsRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
) -> bool {
    &&& bt.len() > 0
    &&& wr.layers.len() > 0
    &&& pre_kv.len() == wr.layers.len()
    &&& input_ids.len() == positions.len()
    &&& slots.len() == input_ids.len()
    // With positive cu_q segments, one query row per request characterizes a
    // pure decode independently of any executable policy enum.
    &&& input_ids.len() == bt.len()
    &&& max_q == 1
    &&& cu_q.len() == cu_k.len()
    &&& cu_k.len() == bt.len() + 1
    &&& cu_q[0] == 0
    &&& cu_k[0] == 0
    &&& cu_q[bt.len() as int] == input_ids.len() as int
    &&& (forall|i: int| 0 <= i < bt.len() as int ==>
        #[trigger] decode_cover_row_ready(
            wr, input_ids, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, i,
        ))
    &&& (forall|ell: int| 0 <= ell < wr.layers.len() ==>
        #[trigger] RT::paged_attention_launch_ready(
            input_ids.len(), pre_kv[ell].0, pre_kv[ell].1,
            cu_q, cu_k, max_q, max_k, bt,
        ))
}

pub proof fn lemma_decode_cover_ready_parts(
    wr: ModelWeightsRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
)
    requires decode_cover_ready(
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt,
    ),
    ensures
        bt.len() > 0,
        wr.layers.len() > 0,
        pre_kv.len() == wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == positions.len(),
        slots.len() == input_ids.len(),
        input_ids.len() == bt.len(),
        max_q == 1,
        cu_q.len() == cu_k.len(),
        cu_k.len() == bt.len() + 1,
        cu_q[0] == 0,
        cu_k[0] == 0,
        cu_q[bt.len() as int] == input_ids.len() as int,
        forall|ell: int| 0 <= ell < wr.layers.len() ==>
            #[trigger] RT::paged_attention_launch_ready(
                input_ids.len(), pre_kv[ell].0, pre_kv[ell].1,
                cu_q, cu_k, max_q, max_k, bt,
            ),
{
    reveal(decode_cover_ready);
}

pub proof fn lemma_decode_cover_ready_row(
    wr: ModelWeightsRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    j: int,
)
    requires
        decode_cover_ready(
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt,
        ),
        0 <= j < bt.len(),
    ensures
        decode_cover_row_ready(
            wr, input_ids, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, j,
        ),
        0 <= cu_q[j] < cu_q[j + 1] <= input_ids.len() as int,
        0 <= cu_k[j] < cu_k[j + 1],
        cu_q[j + 1] - cu_q[j] <= max_q as int,
        cu_k[j + 1] - cu_k[j] <= max_k as int,
        cu_q[j + 1] - cu_q[j] <= cu_k[j + 1] - cu_k[j],
        blocks_needed_for((cu_k[j + 1] - cu_k[j]) as nat) <= bt[j].len(),
{
    reveal(decode_cover_ready);
    assert(decode_cover_row_ready(
        wr, input_ids, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt, j,
    ));
    assert(0 <= cu_q[j]);
    assert(0 <= cu_k[j]);
    assert(cu_q[j] < cu_q[j + 1]);
    assert(cu_k[j] < cu_k[j + 1]);
    assert(cu_q[j + 1] <= input_ids.len() as int);
    assert(cu_q[j + 1] - cu_q[j] <= max_q as int);
    assert(cu_k[j + 1] - cu_k[j] <= max_k as int);
    assert(cu_q[j + 1] - cu_q[j] <= cu_k[j + 1] - cu_k[j]);
    assert(blocks_needed_for((cu_k[j + 1] - cu_k[j]) as nat) <= bt[j].len());
}

#[verifier::spinoff_prover]
pub proof fn lemma_cover_row_metadata_ready(
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    pads: nat,
    num_pages: nat,
    j: int,
)
    requires
        bt.len() > 0,
        num_pages > 0,
        RT::paged_attention_metadata_ready(
            bt.len(), num_pages, cu_q, cu_k, max_q, max_k, bt,
        ),
        0 <= j < cover_block_table(bt, pads).len(),
    ensures ({
        let cuq2 = cover_cumulative(cu_q, pads);
        let cuk2 = cover_cumulative(cu_k, pads);
        let bt2 = cover_block_table(bt, pads);
        let q_len = cuq2[j + 1] - cuq2[j];
        let k_len = cuk2[j + 1] - cuk2[j];
        &&& cuq2.len() == bt2.len() + 1
        &&& cuk2.len() == bt2.len() + 1
        &&& cuq2[j] < cuq2[j + 1]
        &&& cuk2[j] < cuk2[j + 1]
        &&& q_len <= max_q as int
        &&& k_len <= max_k as int
        &&& q_len <= k_len
        &&& blocks_needed_for(k_len as nat) <= bt2[j].len()
        &&& (forall|l: int| 0 <= l < bt2[j].len() ==>
            #[trigger] bt2[j][l] < num_pages)
    }),
{
    reveal(RT::paged_attention_metadata_ready);
    assert(cu_q.len() > 0);
    assert(cu_k.len() > 0);
    assert(cu_q.len() == bt.len() + 1);
    assert(cu_k.len() == bt.len() + 1);
    let n = bt.len() as int;
    let cuq2 = cover_cumulative(cu_q, pads);
    let cuk2 = cover_cumulative(cu_k, pads);
    let bt2 = cover_block_table(bt, pads);
    assert(cuq2.len() == bt2.len() + 1);
    assert(cuk2.len() == bt2.len() + 1);
    if j < n {
        assert(0 <= j < cuq2.len());
        assert(0 <= j + 1 < cuq2.len());
        assert(0 <= j < cuk2.len());
        assert(0 <= j + 1 < cuk2.len());
        assert(0 <= j < cu_q.len());
        assert(0 <= j + 1 < cu_q.len());
        assert(0 <= j < cu_k.len());
        assert(0 <= j + 1 < cu_k.len());
        assert(cuq2[j] == cu_q[j]);
        assert(cuq2[j + 1] == cu_q[j + 1]);
        assert(cuk2[j] == cu_k[j]);
        assert(cuk2[j + 1] == cu_k[j + 1]);
        assert(bt2[j] == bt[j]);
    } else {
        let p = j - n;
        assert(0 <= p < pads as int);
        assert(0 <= j < cuq2.len());
        assert(0 <= j + 1 < cuq2.len());
        assert(0 <= j < cuk2.len());
        assert(0 <= j + 1 < cuk2.len());
        assert(0 <= n < cu_q.len());
        assert(0 <= n < cu_k.len());
        assert(cuq2[j] == cu_q[n] + p);
        assert(cuq2[j + 1] == cu_q[n] + p + 1);
        assert(cuk2[j] == cu_k[n] + p);
        assert(cuk2[j + 1] == cu_k[n] + p + 1);
        assert(bt2[j] == seq![0u64]);
        assert(blocks_needed_for(1) == 1);
    }
}

// Extending the ragged launch metadata by one-token dummy requests preserves
// the paged-attention launch domain. Dummy block-table rows name page zero;
// the original launch contract already proves that the cache has at least one
// page, so even the discarded rows remain within the deployed kernel domain.
#[verifier::spinoff_prover]
#[verifier::rlimit(100)]
pub proof fn lemma_decode_cover_paged_attention_launch_ready(
    wr: ModelWeightsRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    pads: nat,
    layer: int,
)
    requires
        decode_cover_ready(
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt,
        ),
        0 <= layer < wr.layers.len(),
    ensures
        RT::paged_attention_launch_ready(
            cover_int_rows(input_ids, pads, 0).len(),
            pre_kv[layer].0, pre_kv[layer].1,
            cover_cumulative(cu_q, pads),
            cover_cumulative(cu_k, pads),
            max_q, max_k,
            cover_block_table(bt, pads),
        ),
{
    lemma_decode_cover_ready_parts(
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt,
    );
    let n = bt.len() as int;
    let ids2 = cover_int_rows(input_ids, pads, 0);
    let cuq2 = cover_cumulative(cu_q, pads);
    let cuk2 = cover_cumulative(cu_k, pads);
    let bt2 = cover_block_table(bt, pads);
    let pages = pre_kv[layer].0.len();
    RT::lemma_paged_attention_launch_ready_parts(
        input_ids.len(), pre_kv[layer].0, pre_kv[layer].1,
        cu_q, cu_k, max_q, max_k, bt,
    );
    assert(RT::paged_cache_geometry(pre_kv[layer].0, pre_kv[layer].1));
    reveal(RT::paged_cache_geometry);
    assert(pre_kv[layer].0.len() > 0);
    reveal(RT::paged_attention_metadata_ready);
    assert(ids2.len() > 0);
    assert(bt2.len() > 0);
    assert(max_q > 0);
    assert(max_k > 0);
    assert(cuq2.len() == bt2.len() + 1);
    assert(cuk2.len() == bt2.len() + 1);
    assert(cuq2[0] == 0);
    assert(cuk2[0] == 0);
    assert(cuq2[bt2.len() as int] == ids2.len() as int);
    assert({
        &&& ids2.len() > 0
        &&& pages > 0
        &&& bt2.len() > 0
        &&& cuq2.len() == bt2.len() + 1
        &&& cuk2.len() == bt2.len() + 1
        &&& cuq2[0] == 0
        &&& cuk2[0] == 0
        &&& cuq2[bt2.len() as int] == ids2.len() as int
        &&& max_q > 0
        &&& max_k > 0
    });
    assert(RT::paged_attention_metadata_header_ready(
        ids2.len(), pages, cuq2, cuk2, max_q, max_k, bt2,
    ));
    assert(RT::paged_attention_metadata_rows_ready(
        pages, cuq2, cuk2, max_q, max_k, bt2,
    )) by {
        reveal(RT::paged_attention_metadata_rows_ready);
        assert_forall_by(|j: int| {
            requires(0 <= j && j < bt2.len() as int);
            ensures({
                let q_len = cuq2[j + 1] - cuq2[j];
                let k_len = cuk2[j + 1] - cuk2[j];
                &&& cuq2[j] < #[trigger] cuq2[j + 1]
                &&& cuk2[j] < cuk2[j + 1]
                &&& q_len <= max_q as int
                &&& k_len <= max_k as int
                &&& q_len <= k_len
                &&& blocks_needed_for(k_len as nat) <= bt2[j].len()
                &&& (forall|l: int| 0 <= l < bt2[j].len() ==>
                    #[trigger] bt2[j][l] < pages)
            });
            lemma_cover_row_metadata_ready(
                cu_q, cu_k, max_q, max_k, bt, pads,
                pages, j,
            );
        });
    }
    RT::lemma_paged_attention_metadata_ready_from_parts(
        ids2.len(), pages,
        cuq2, cuk2, max_q, max_k, bt2,
    );
    RT::lemma_paged_attention_launch_ready_from_parts(
        ids2.len(), pre_kv[layer].0, pre_kv[layer].1,
        cuq2, cuk2, max_q, max_k, bt2,
    );
}

// Every real request keeps the same complete logit slice when dummy decode
// requests are appended. Both executions isolate to the identical singleton
// request; the only new non-interference case is a -1 dummy slot, whose page
// quotient is negative and therefore cannot name a real block-table page.
#[verifier::spinoff_prover]
#[verifier::rlimit(200)]
pub proof fn lemma_decode_cover_real_request(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    pads: nat,
    i: nat,
    layer: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        model_weights_architecture_repr_valid(wr, architecture_repr),
        MA::request_projection_configuration_ready(wr, architecture_repr),
        decode_cover_ready(
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt,
        ),
        i < bt.len(),
        layer < wr.layers.len(),
        cu_q.len() == bt.len() + 1,
        cu_k.len() == bt.len() + 1,
        0 <= cu_q[i as int] < cu_q[i as int + 1]
            <= input_ids.len() as int,
        forall|j: int| 0 <= j < bt.len() as int ==>
            cu_q[j] < #[trigger] cu_q[j + 1],
        forall|j: int| 0 <= j < bt.len() as int ==>
            cu_k[j] < #[trigger] cu_k[j + 1],
        MA::model_forward_logits_repr(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt,
        ).len() == input_ids.len(),
        MA::model_forward_logits_repr(
            wr, architecture_repr,
            cover_int_rows(input_ids, pads, 0),
            cover_int_rows(positions, pads, 0),
            pre_kv,
            cover_int_rows(slots, pads, -1),
            cover_cumulative(cu_q, pads),
            cover_cumulative(cu_k, pads),
            max_q, max_k,
            cover_block_table(bt, pads),
        ).len() == cover_int_rows(input_ids, pads, 0).len(),
        ({
            let real_rows = MA::model_layer_kv_rows(
                wr, architecture_repr, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, bt, layer,
            );
            real_rows.0.len() == input_ids.len()
                && real_rows.1.len() == input_ids.len()
        }),
        ({
            let ids2 = cover_int_rows(input_ids, pads, 0);
            let cover_rows = MA::model_layer_kv_rows(
                wr, architecture_repr,
                ids2,
                cover_int_rows(positions, pads, 0),
                pre_kv,
                cover_int_rows(slots, pads, -1),
                cover_cumulative(cu_q, pads),
                cover_cumulative(cu_k, pads),
                max_q, max_k,
                cover_block_table(bt, pads),
                layer,
            );
            cover_rows.0.len() == ids2.len()
                && cover_rows.1.len() == ids2.len()
        }),
    ensures ({
        let ids2 = cover_int_rows(input_ids, pads, 0);
        let pos2 = cover_int_rows(positions, pads, 0);
        let slots2 = cover_int_rows(slots, pads, -1);
        let cuq2 = cover_cumulative(cu_q, pads);
        let cuk2 = cover_cumulative(cu_k, pads);
        let bt2 = cover_block_table(bt, pads);
        let real_logits = MA::model_forward_logits_repr(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt,
        );
        let cover_logits = MA::model_forward_logits_repr(
            wr, architecture_repr, ids2, pos2, pre_kv, slots2,
            cuq2, cuk2, max_q, max_k, bt2,
        );
        let real_rows = MA::model_layer_kv_rows(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, layer,
        );
        let cover_rows = MA::model_layer_kv_rows(
            wr, architecture_repr, ids2, pos2, pre_kv, slots2,
            cuq2, cuk2, max_q, max_k, bt2, layer,
        );
        &&& cover_logits.subrange(cu_q[i as int], cu_q[i as int + 1])
            == real_logits.subrange(cu_q[i as int], cu_q[i as int + 1])
        &&& cover_rows.0.subrange(cu_q[i as int], cu_q[i as int + 1])
            == real_rows.0.subrange(cu_q[i as int], cu_q[i as int + 1])
        &&& cover_rows.1.subrange(cu_q[i as int], cu_q[i as int + 1])
            == real_rows.1.subrange(cu_q[i as int], cu_q[i as int + 1])
    }),
{
    reveal(decode_cover_ready);
    let n = bt.len() as int;
    let ids2 = cover_int_rows(input_ids, pads, 0);
    let pos2 = cover_int_rows(positions, pads, 0);
    let slots2 = cover_int_rows(slots, pads, -1);
    let cuq2 = cover_cumulative(cu_q, pads);
    let cuk2 = cover_cumulative(cu_k, pads);
    let bt2 = cover_block_table(bt, pads);
    let lo = cu_q[i as int];
    let hi = cu_q[i as int + 1];
    let slot_i = slots.subrange(lo, hi);

    lemma_decode_cover_ready_row(
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt, i as int,
    );

    assert(0 <= i < bt.len());
    assert(0 <= lo < hi <= input_ids.len() as int);
    assert(ids2.len() == pos2.len());
    assert(slots2.len() == ids2.len());
    assert(cuq2.len() == cuk2.len());
    assert(cuk2.len() == bt2.len() + 1);
    assert(cuq2[0] == 0);
    assert(cuk2[0] == 0);
    assert(cuq2[bt2.len() as int] == ids2.len() as int);
    assert(cuq2[i as int] == cu_q[i as int]);
    assert(cuq2[i as int + 1] == cu_q[i as int + 1]);
    assert(cuk2[i as int] == cu_k[i as int]);
    assert(cuk2[i as int + 1] == cu_k[i as int + 1]);
    assert(bt2[i as int] == bt[i as int]);
    assert(ids2.subrange(lo, hi) == input_ids.subrange(lo, hi));
    assert(pos2.subrange(lo, hi) == positions.subrange(lo, hi));
    assert(slots2.subrange(lo, hi) == slot_i);

    // Recover the padded global ragged metadata from one layer; all layers
    // share it, and the per-layer launch proof below supplies cache geometry.
    lemma_decode_cover_paged_attention_launch_ready(
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt, pads, 0,
    );
    RT::lemma_paged_attention_launch_ready_parts(
        ids2.len(), pre_kv[0].0, pre_kv[0].1,
        cuq2, cuk2, max_q, max_k, bt2,
    );
    RT::lemma_paged_attention_launch_ready_parts(
        input_ids.len(), pre_kv[0].0, pre_kv[0].1,
        cu_q, cu_k, max_q, max_k, bt,
    );
    reveal(RT::paged_attention_metadata_ready);

    assert(slots2.subrange(0, lo) == slots.subrange(0, lo));
    assert forall|m: int, l: int|
        #![trigger slots2.subrange(0, lo)[m]
            / (BLOCK_SIZE_SPEC as int), bt2[i as int][l]]
        0 <= m < lo && 0 <= l < bt2[i as int].len() implies
            slots2.subrange(0, lo)[m] / (BLOCK_SIZE_SPEC as int)
                != bt2[i as int][l] as int
    by {
        assert(slots2.subrange(0, lo)[m] == slots.subrange(0, lo)[m]);
        assert(bt2[i as int][l] == bt[i as int][l]);
    }

    assert forall|m: int, l: int|
        #![trigger slots2.subrange(hi, slots2.len() as int)[m]
            / (BLOCK_SIZE_SPEC as int), bt2[i as int][l]]
        0 <= m < slots2.len() - hi && 0 <= l < bt2[i as int].len() implies
            slots2.subrange(hi, slots2.len() as int)[m]
                / (BLOCK_SIZE_SPEC as int) != bt2[i as int][l] as int
    by {
        assert(bt2[i as int][l] == bt[i as int][l]);
        let q = hi + m;
        assert(slots2.subrange(hi, slots2.len() as int)[m] == slots2[q]);
        if q < slots.len() as int {
            assert(m < slots.len() - hi);
            assert(slots2[q] == slots[q]);
            assert(slots.subrange(hi, slots.len() as int)[m] == slots[q]);
        } else {
            assert(slots2[q] == -1);
            let bs = BLOCK_SIZE_SPEC as int;
            assert(bs == 64);
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod(-1, bs);
            vstd::arithmetic::div_mod::lemma_mod_bound(-1, bs);
            assert((-1int) / bs < 0);
            assert(0 <= bt2[i as int][l] as int);
        }
    }

    reveal(decode_cover_row_ready);
    assert forall|ell: int, pos: nat|
        #![trigger pre_kv[ell].0, block_table_slot(bt2[i as int], pos)]
        0 <= ell < wr.layers.len()
            && pos < (cuk2[i as int + 1] - cuk2[i as int]) as nat implies
            slot_in_cache(
                pre_kv[ell].0, block_table_slot(bt2[i as int], pos),
            )
            && slot_in_cache(
                pre_kv[ell].1, block_table_slot(bt2[i as int], pos),
            )
    by {
        assert(bt2[i as int] == bt[i as int]);
        assert(cuk2[i as int + 1] - cuk2[i as int]
            == cu_k[i as int + 1] - cu_k[i as int]);
        assert(block_table_slot(bt2[i as int], pos)
            == block_table_slot(bt[i as int], pos));
        assert(slot_in_cache(
            pre_kv[ell].0, block_table_slot(bt[i as int], pos),
        ));
        assert(slot_in_cache(
            pre_kv[ell].1, block_table_slot(bt[i as int], pos),
        ));
    }
    assert forall|ell: int, pos: nat|
        #![trigger pre_kv[ell].0, block_table_slot(bt2[i as int], pos)]
        0 <= ell < wr.layers.len()
            && pos < blocks_needed_for(
                (cuk2[i as int + 1] - cuk2[i as int]) as nat,
            ) * BLOCK_SIZE_SPEC implies
            slot_in_cache(
                pre_kv[ell].0, block_table_slot(bt2[i as int], pos),
            )
            && slot_in_cache(
                pre_kv[ell].1, block_table_slot(bt2[i as int], pos),
            )
    by {
        assert(bt2[i as int] == bt[i as int]);
        assert(cuk2[i as int + 1] - cuk2[i as int]
            == cu_k[i as int + 1] - cu_k[i as int]);
        assert(block_table_slot(bt2[i as int], pos)
            == block_table_slot(bt[i as int], pos));
        assert(slot_in_cache(
            pre_kv[ell].0, block_table_slot(bt[i as int], pos),
        ));
        assert(slot_in_cache(
            pre_kv[ell].1, block_table_slot(bt[i as int], pos),
        ));
    }

    assert forall|ell: int| 0 <= ell < wr.layers.len() implies
        #[trigger] RT::paged_attention_launch_ready(
            ids2.len(), pre_kv[ell].0, pre_kv[ell].1,
            cuq2, cuk2, max_q, max_k, bt2,
        )
    by {
        lemma_decode_cover_paged_attention_launch_ready(
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, pads, ell,
        );
    }

    assert forall|j: int| 0 <= j < bt2.len() as int implies
        cuq2[j] < #[trigger] cuq2[j + 1]
    by {
        let pages = pre_kv[0].0.len();
        lemma_cover_row_metadata_ready(
            cu_q, cu_k, max_q, max_k, bt, pads, pages, j,
        );
    }
    assert forall|j: int| 0 <= j < bt2.len() as int implies
        cuk2[j] < #[trigger] cuk2[j + 1]
    by {
        let pages = pre_kv[0].0.len();
        lemma_cover_row_metadata_ready(
            cu_q, cu_k, max_q, max_k, bt, pads, pages, j,
        );
    }

    assert(RT::paged_attention_numeric_domain());
    assert(MA::request_projection_configuration_ready(
        wr, architecture_repr,
    ));
    assert(pre_kv.len() >= wr.layers.len());
    assert(0 <= cuq2[i as int]
        < cuq2[i as int + 1]
        <= ids2.len() as int);
    assert(0 <= cuk2[i as int] < cuk2[i as int + 1]);
    assert(max_q > 0);
    assert(cuq2[i as int + 1] - cuq2[i as int] <= max_q as int);
    assert(cuk2[i as int + 1] - cuk2[i as int] <= max_k as int);
    assert(cuq2[i as int + 1] - cuq2[i as int]
        <= cuk2[i as int + 1] - cuk2[i as int]);
    assert(blocks_needed_for(
        (cuk2[i as int + 1] - cuk2[i as int]) as nat,
    ) <= bt2[i as int].len());

    assert(crate::proof::model::family_layout::request_projection_common_domain(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt, i,
    )) by {
        reveal(crate::proof::model::family_layout::request_projection_common_domain);
    }
    assert(crate::proof::model::family_layout::request_projection_common_domain(
        wr, architecture_repr, ids2, pos2, pre_kv, slots2,
        cuq2, cuk2, max_q, max_k, bt2, i,
    )) by {
        reveal(crate::proof::model::family_layout::request_projection_common_domain);
    }
    MA::lemma_model_forward_request_projection_domain_from_common(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt, i,
    );
    MA::lemma_model_forward_request_projection_domain_from_common(
        wr, architecture_repr, ids2, pos2, pre_kv, slots2,
        cuq2, cuk2, max_q, max_k, bt2, i,
    );
    MA::lemma_model_forward_logits_request_isolation(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt, i,
    );
    MA::lemma_model_forward_logits_request_isolation(
        wr, architecture_repr, ids2, pos2, pre_kv, slots2,
        cuq2, cuk2, max_q, max_k, bt2, i,
    );
    MA::lemma_model_layer_kv_rows_request_isolation(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt, i, layer,
    );
    MA::lemma_model_layer_kv_rows_request_isolation(
        wr, architecture_repr, ids2, pos2, pre_kv, slots2,
        cuq2, cuk2, max_q, max_k, bt2, i, layer,
    );
}

pub open spec fn real_logit_prefix_equal(
    real_logits: Tensor2D,
    cover_logits: Tensor2D,
    real_rows: nat,
) -> bool {
    real_logits.len() == real_rows
    && cover_logits.len() >= real_rows
    && (forall|q: int| 0 <= q < real_rows ==>
        #[trigger] cover_logits[q] == real_logits[q])
}

// Capstone for pure-decode covering replay. Appending any finite number of
// dummy requests preserves every real logit row and the entire physical KV
// cache produced by model_forward. No batch-size or M-dependent parameter
// appears: `pads` is universally quantified policy input.
#[verifier::spinoff_prover]
#[verifier::rlimit(300)]
pub proof fn lemma_decode_cover_refines_model_forward(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    pads: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        model_weights_architecture_repr_valid(wr, architecture_repr),
        MA::request_projection_configuration_ready(wr, architecture_repr),
        decode_cover_ready(
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt,
        ),
    ensures ({
        let ids2 = cover_int_rows(input_ids, pads, 0);
        let pos2 = cover_int_rows(positions, pads, 0);
        let slots2 = cover_int_rows(slots, pads, -1);
        let cuq2 = cover_cumulative(cu_q, pads);
        let cuk2 = cover_cumulative(cu_k, pads);
        let bt2 = cover_block_table(bt, pads);
        let real_logits = MA::model_forward_logits_repr(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt,
        );
        let cover_logits = MA::model_forward_logits_repr(
            wr, architecture_repr, ids2, pos2, pre_kv, slots2,
            cuq2, cuk2, max_q, max_k, bt2,
        );
        &&& real_logit_prefix_equal(
            real_logits, cover_logits, input_ids.len(),
        )
        &&& covered_selected_logits(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, pads, bt.len(),
        ) == Seq::new(bt.len(), |i: int|
            RT::select_sample_logits_repr(real_logits, cu_q, i as nat))
        &&& covered_model_forward_kv_reprs(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, pads,
        ) == MA::model_forward_kv_reprs(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt,
        )
    }),
{
    reveal(decode_cover_ready);
    let n = bt.len() as int;
    let ids2 = cover_int_rows(input_ids, pads, 0);
    let pos2 = cover_int_rows(positions, pads, 0);
    let slots2 = cover_int_rows(slots, pads, -1);
    let pad_slots = Seq::new(pads, |_p: int| -1int);
    let cuq2 = cover_cumulative(cu_q, pads);
    let cuk2 = cover_cumulative(cu_k, pads);
    let bt2 = cover_block_table(bt, pads);
    let real_logits = MA::model_forward_logits_repr(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt,
    );
    let cover_logits = MA::model_forward_logits_repr(
        wr, architecture_repr, ids2, pos2, pre_kv, slots2,
        cuq2, cuk2, max_q, max_k, bt2,
    );
    let real_post = MA::model_forward_kv_reprs(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt,
    );
    let cover_post = MA::model_forward_kv_reprs(
        wr, architecture_repr, ids2, pos2, pre_kv, slots2,
        cuq2, cuk2, max_q, max_k, bt2,
    );
    assert forall|j: int| 0 <= j < bt.len() as int implies
        cu_q[j] < #[trigger] cu_q[j + 1]
    by {
        lemma_decode_cover_ready_row(
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, j,
        );
    }
    assert forall|j: int| 0 <= j < bt.len() as int implies
        cu_k[j] < #[trigger] cu_k[j + 1]
    by {
        lemma_decode_cover_ready_row(
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, j,
        );
    }

    MA::lemma_model_forward_logits_repr_shape(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt,
    );
    MA::lemma_model_forward_logits_repr_shape(
        wr, architecture_repr, ids2, pos2, pre_kv, slots2,
        cuq2, cuk2, max_q, max_k, bt2,
    );
    assert(real_logits.len() == input_ids.len());
    assert(cover_logits.len() == ids2.len());
    assert(ids2.len() >= input_ids.len());

    assert forall|q: int| 0 <= q < input_ids.len() implies
        #[trigger] cover_logits[q] == real_logits[q]
    by {
        let k = crate::proof::tensor::geometry::lemma_cu_locate(cu_q, n, q);
        lemma_decode_cover_ready_row(
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, k,
        );
        let lo = cu_q[k];
        let hi = cu_q[k + 1];
        assert(0 <= k < bt.len());
        assert(0 <= lo <= q < hi <= input_ids.len());
        MA::lemma_model_layer_kv_rows_shape(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, 0,
        );
        MA::lemma_model_layer_kv_rows_shape(
            wr, architecture_repr, ids2, pos2, pre_kv, slots2,
            cuq2, cuk2, max_q, max_k, bt2, 0,
        );
        lemma_decode_cover_real_request(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, pads, k as nat, 0,
        );
        assert(cover_logits.subrange(lo, hi)
            == real_logits.subrange(lo, hi));
        assert(cover_logits.subrange(lo, hi)[q - lo] == cover_logits[q]);
        assert(real_logits.subrange(lo, hi)[q - lo] == real_logits[q]);
    }
    assert(real_logit_prefix_equal(
        real_logits, cover_logits, input_ids.len(),
    ));
    let real_selected = Seq::new(bt.len(), |i: int|
        RT::select_sample_logits_repr(real_logits, cu_q, i as nat));
    let cover_selected = covered_selected_logits(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt, pads, bt.len(),
    );
    assert_seqs_equal!(cover_selected, real_selected, i => {
        assert(0 <= i < bt.len());
        lemma_decode_cover_ready_row(
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, i,
        );
        assert(cover_cumulative(cu_q, pads)[i] == cu_q[i]);
        assert(cover_cumulative(cu_q, pads)[i + 1] == cu_q[i + 1]);
        let q = cu_q[i + 1] - 1;
        assert(0 <= q < input_ids.len());
        assert(cover_logits[q] == real_logits[q]);
    });

    MA::lemma_model_forward_kv_reprs_len(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt,
    );
    MA::lemma_model_forward_kv_reprs_len(
        wr, architecture_repr, ids2, pos2, pre_kv, slots2,
        cuq2, cuk2, max_q, max_k, bt2,
    );
    assert(real_post.len() == pre_kv.len());
    assert(cover_post.len() == pre_kv.len());

    assert_seqs_equal!(cover_post, real_post, ell => {
        assert(0 <= ell < wr.layers.len());
        let layer = ell as nat;
        let real_rows = MA::model_layer_kv_rows(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, layer,
        );
        let cover_rows = MA::model_layer_kv_rows(
            wr, architecture_repr, ids2, pos2, pre_kv, slots2,
            cuq2, cuk2, max_q, max_k, bt2, layer,
        );
        MA::lemma_model_layer_kv_rows_shape(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, layer,
        );
        MA::lemma_model_layer_kv_rows_shape(
            wr, architecture_repr, ids2, pos2, pre_kv, slots2,
            cuq2, cuk2, max_q, max_k, bt2, layer,
        );
        assert(real_rows.0.len() == input_ids.len());
        assert(real_rows.1.len() == input_ids.len());
        assert(cover_rows.0.len() == ids2.len());
        assert(cover_rows.1.len() == ids2.len());
        assert forall|q: int| 0 <= q < input_ids.len() implies
            #[trigger] cover_rows.0[q] == real_rows.0[q]
        by {
            let k = crate::proof::tensor::geometry::lemma_cu_locate(cu_q, n, q);
            lemma_decode_cover_ready_row(
                wr, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, bt, k,
            );
            let lo = cu_q[k];
            let hi = cu_q[k + 1];
            assert(0 <= k < bt.len());
            assert(0 <= lo <= q < hi <= input_ids.len());
            lemma_decode_cover_real_request(
                wr, architecture_repr, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, bt, pads, k as nat, layer,
            );
            assert(cover_rows.0.subrange(lo, hi)
                == real_rows.0.subrange(lo, hi));
            assert(cover_rows.1.subrange(lo, hi)
                == real_rows.1.subrange(lo, hi));
            assert(cover_rows.0.subrange(lo, hi)[q - lo]
                == cover_rows.0[q]);
            assert(real_rows.0.subrange(lo, hi)[q - lo]
                == real_rows.0[q]);
        }
        assert forall|q: int| 0 <= q < input_ids.len() implies
            #[trigger] cover_rows.1[q] == real_rows.1[q]
        by {
            let k = crate::proof::tensor::geometry::lemma_cu_locate(cu_q, n, q);
            lemma_decode_cover_ready_row(
                wr, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, bt, k,
            );
            let lo = cu_q[k];
            let hi = cu_q[k + 1];
            assert(0 <= k < bt.len());
            assert(0 <= lo <= q < hi <= input_ids.len());
            lemma_decode_cover_real_request(
                wr, architecture_repr, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, bt, pads, k as nat, layer,
            );
            assert(cover_rows.1.subrange(lo, hi)
                == real_rows.1.subrange(lo, hi));
            assert(cover_rows.1.subrange(lo, hi)[q - lo]
                == cover_rows.1[q]);
            assert(real_rows.1.subrange(lo, hi)[q - lo]
                == real_rows.1[q]);
        }
        assert(real_rows.0 == cover_rows.0.subrange(0, input_ids.len() as int)) by {
            assert_seqs_equal!(real_rows.0,
                cover_rows.0.subrange(0, input_ids.len() as int));
        }
        assert(real_rows.1 == cover_rows.1.subrange(0, input_ids.len() as int)) by {
            assert_seqs_equal!(real_rows.1,
                cover_rows.1.subrange(0, input_ids.len() as int));
        }
        let kr_pad = cover_rows.0.subrange(
            input_ids.len() as int, cover_rows.0.len() as int,
        );
        let vr_pad = cover_rows.1.subrange(
            input_ids.len() as int, cover_rows.1.len() as int,
        );
        cover_rows.0.lemma_split_at(input_ids.len() as int);
        cover_rows.1.lemma_split_at(input_ids.len() as int);
        assert(cover_rows.0 == real_rows.0 + kr_pad);
        assert(cover_rows.1 == real_rows.1 + vr_pad);
        assert(slots2 == slots + pad_slots);
        assert(kr_pad.len() == pads);
        assert(vr_pad.len() == pads);
        assert forall|p: int| 0 <= p < pad_slots.len() implies
            #[trigger] pad_slots[p] == -1
        by {}
        RT::lemma_store_kv_cache_repr_append_no_write_rows(
            real_rows.0, real_rows.1, kr_pad, vr_pad,
            pre_kv[ell].0, pre_kv[ell].1, slots, pad_slots,
        );
        MA::lemma_model_forward_layer_store(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, layer,
        );
        MA::lemma_model_forward_layer_store(
            wr, architecture_repr, ids2, pos2, pre_kv, slots2,
            cuq2, cuk2, max_q, max_k, bt2, layer,
        );
    });
}

// Turn the scheduler's stable row geometry into the exact side condition for
// pure-decode graph covering. Architecture configuration is supplied by the
// family-neutral execution capability; the scheduler discharges only layout.
#[verifier::spinoff_prover]
pub proof fn lemma_reprs_decode_cover_ready(
    old_e: Engine,
    reprs: StepReprs,
)
    requires
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_ok(old_e, reprs),
        crate::exec::engine::reprs_other_writes_miss_plan_rows(reprs),
        reprs.scheduled.len() > 0,
        old_e.kv_caches_repr@.len() == reprs.wr.layers.len(),
        reprs.wr.layers.len() > 0,
        reprs.input_ids.len() == reprs.bt.len(),
        reprs.max_q == 1,
    ensures
        decode_cover_ready(
            reprs.wr, reprs.input_ids, reprs.positions,
            old_e.kv_caches_repr@, reprs.slots,
            reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt,
        ),
{
    reveal(crate::exec::engine::step_reprs_wf);
    reveal(decode_cover_ready);
    assert forall|i: int| 0 <= i < reprs.bt.len() implies
        #[trigger] decode_cover_row_ready(
            reprs.wr, reprs.input_ids, old_e.kv_caches_repr@,
            reprs.slots, reprs.cu_q, reprs.cu_k,
            reprs.max_q, reprs.max_k, reprs.bt, i,
        )
    by {
        assert(crate::exec::engine::reprs_forward_layout_at(old_e, reprs, i));
        reveal(crate::exec::engine::reprs_forward_layout_at);
        reveal(decode_cover_row_ready);
        let lo = reprs.cu_q[i];
        let hi = reprs.cu_q[i + 1];
        let kd = reprs.cu_k[i + 1] - reprs.cu_k[i];
        assert(0 <= lo < hi <= reprs.input_ids.len() as int);
        assert(reprs.cu_k[i] < reprs.cu_k[i + 1]);
        crate::proof::tensor::geometry::lemma_cu_mono(
            reprs.cu_k, reprs.scheduled.len() as int, 0, i,
        );
        assert(0 <= reprs.cu_k[i] < reprs.cu_k[i + 1]);
        assert(hi - lo <= reprs.max_q as int);
        assert(kd <= reprs.max_k as int);
        assert(hi - lo <= kd);
        assert(blocks_needed_for(kd as nat) <= reprs.bt[i].len());
        assert(reprs.slots.len() == reprs.input_ids.len());

        assert forall|m: int, l: int|
            #![trigger reprs.slots.subrange(0, lo)[m]
                / (BLOCK_SIZE_SPEC as int), reprs.bt[i][l]]
            0 <= m < lo && 0 <= l < reprs.bt[i].len() implies
                reprs.slots.subrange(0, lo)[m]
                    / (BLOCK_SIZE_SPEC as int) != reprs.bt[i][l] as int
        by {
            assert(reprs.slots.subrange(0, lo)[m] == reprs.slots[m]);
            crate::proof::cache::coherence::lemma_other_request_rows_miss_plan_row(
                old_e, reprs, i, m, l,
            );
        }
        assert forall|m: int, l: int|
            #![trigger reprs.slots.subrange(hi, reprs.slots.len() as int)[m]
                / (BLOCK_SIZE_SPEC as int), reprs.bt[i][l]]
            0 <= m < reprs.slots.len() - hi
                && 0 <= l < reprs.bt[i].len() implies
                reprs.slots.subrange(hi, reprs.slots.len() as int)[m]
                    / (BLOCK_SIZE_SPEC as int) != reprs.bt[i][l] as int
        by {
            let q = hi + m;
            assert(reprs.slots.subrange(hi, reprs.slots.len() as int)[m]
                == reprs.slots[q]);
            crate::proof::cache::coherence::lemma_other_request_rows_miss_plan_row(
                old_e, reprs, i, q, l,
            );
        }
        assert forall|ell: int, pos: nat|
            #![trigger old_e.kv_caches_repr@[ell].0,
                block_table_slot(reprs.bt[i], pos)]
            0 <= ell < reprs.wr.layers.len() && pos < kd as nat implies
                slot_in_cache(
                    old_e.kv_caches_repr@[ell].0,
                    block_table_slot(reprs.bt[i], pos),
                )
                && slot_in_cache(
                    old_e.kv_caches_repr@[ell].1,
                    block_table_slot(reprs.bt[i], pos),
                )
        by {
            crate::proof::cache::coherence::lemma_plan_slot_in_pre_cache(
                old_e, reprs, i, ell, pos,
            );
        }
        assert forall|ell: int, pos: nat|
            #![trigger old_e.kv_caches_repr@[ell].0,
                block_table_slot(reprs.bt[i], pos)]
            0 <= ell < reprs.wr.layers.len()
                && pos < blocks_needed_for(kd as nat) * BLOCK_SIZE_SPEC
            implies
                slot_in_cache(
                    old_e.kv_caches_repr@[ell].0,
                    block_table_slot(reprs.bt[i], pos),
                )
                && slot_in_cache(
                    old_e.kv_caches_repr@[ell].1,
                    block_table_slot(reprs.bt[i], pos),
                )
        by {
            crate::proof::cache::coherence::lemma_plan_page_slot_in_pre_cache(
                old_e, reprs, i, ell, pos,
            );
        }
        assert(decode_cover_row_ready(
            reprs.wr, reprs.input_ids, old_e.kv_caches_repr@,
            reprs.slots, reprs.cu_q, reprs.cu_k,
            reprs.max_q, reprs.max_k, reprs.bt, i,
        ));
    }
    crate::proof::cache::coherence::lemma_reprs_paged_attention_launch_ready(
        old_e, reprs,
    );
}

}
