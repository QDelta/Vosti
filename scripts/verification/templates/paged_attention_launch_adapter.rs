// Checked whole-launch binding to the architecture-neutral mapped semantics.
verus! {
pub open spec fn mapped_launch_output(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
) -> Tensor2D {
    AP::launch_output(canonical_row_operation(h, hkv, scale, window, fill), h * __D__,
        q, k, v, cu_q, cu_k, table)
}

// These are exactly the generated numeric predicates needed for each
// singleton-to-canonical comparison, not an assumption of output equality.
pub open spec fn launch_row_runtime_assumptions(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
    request: int, token: int,
) -> bool {
        selected_runtime_assumptions(
            adapter_side(q.subrange(cu_q[request], cu_q[request + 1]), k, v, seq![table[request]],
                seq![0int, cu_q[request + 1] - cu_q[request]],
                seq![0int, cu_k[request + 1] - cu_k[request]],
                (cu_q[request + 1] - cu_q[request]) as nat, h, hkv, scale, window, fill),
            canonical_row_side(q[token],
                logical_prefix(k, table[request], (cu_k[request + 1] - cu_k[request]
                    - cu_q[request + 1] + token + 1) as nat),
                logical_prefix(v, table[request], (cu_k[request + 1] - cu_k[request]
                    - cu_q[request + 1] + token + 1) as nat),
                h, hkv, scale, window, fill),
            __SELECTED_FREE__ { __PORT_LEFT_SELECTOR__: token - cu_q[request], __PORT_RIGHT_SELECTOR__: 0 })
}

pub open spec fn launch_runtime_assumptions(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
) -> bool {
    forall|request: int, token: int|
        0 <= request < table.len() && cu_q[request] <= token < cu_q[request + 1] ==>
        #[trigger] launch_row_runtime_assumptions(q, k, v, table, cu_q, cu_k,
            h, hkv, scale, window, fill, request, token)
}

// Keep this composition query independent of unrelated generated declarations.
// This changes solver isolation, not the resource limit or logical premises.
#[verifier::spinoff_prover]
pub proof fn checked_launch_row_equivalence(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat, max_k: nat,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
    token: int,
)
    requires
        SUP::paged_attention_metadata_ready(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table),
        TS::tensor2d_shape(q, q.len(), h * __D__),
        TS::tensor3d_shape(k, k.len(), 64, hkv * __D__),
        TS::tensor3d_shape(v, k.len(), 64, hkv * __D__),
        common::blocks_needed_for(max_k) <= u64::MAX,
        h > 0, hkv > 0, hkv <= h, h % hkv == 0,
        __WINDOW_REQUIREMENT__
        launch_runtime_assumptions(q, k, v, table, cu_q, cu_k, h, hkv, scale, window, fill),
        0 <= token < q.len(),
    ensures
        adapter_output(q, k, v, table, cu_q, cu_k, max_q,
            h, hkv, scale, window, fill)[token]
        == mapped_launch_output(q, k, v, table, cu_q, cu_k, h, hkv, scale, window, fill)[token],
{
    PL::lemma_metadata_page_table(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table);
    // Metadata already supplies adjacent strict offsets. All-pairs ordering
    // is unnecessary here and would cross-match every offset term in scope.
    common::lemma_cu_int_bounds(cu_q, table.len() as int);
    assert(AP::offsets_valid(cu_q, table.len(), q.len()));
    let op = canonical_row_operation(h, hkv, scale, window, fill);
    let actual = adapter_output(q, k, v, table, cu_q, cu_k, max_q, h, hkv, scale, window, fill);
    let mapped = mapped_launch_output(q, k, v, table, cu_q, cu_k, h, hkv, scale, window, fill);
    lemma_adapter_output_shape(q, k, v, table, cu_q, cu_k, max_q, h, hkv, scale, window, fill);
    AP::lemma_launch_shape(op, h * __D__, q, k, v, cu_q, cu_k, table);
    AP::lemma_owner_exists(cu_q, table.len(), token);
    let i = AP::owner(cu_q, table.len(), token);
    assert(cu_q[i] < cu_q[i + 1]);
    let start = cu_q[i];
    let end = cu_q[i + 1];
    let qlen = (end - start) as nat;
    let klen = (cu_k[i + 1] - cu_k[i]) as nat;
    assert(0 <= start < end <= q.len());
    assert(qlen <= klen <= max_k);
    assert(PL::page_table_ids_valid(seq![table[i]], k.len()));
    checked_batch_projection(q, k, v, table, cu_q, cu_k, max_q, max_k,
        h, hkv, scale, window, fill, i, k, v, table[i]);
    TS::lemma_tensor2d_shape_subrange(q, q.len(), h * __D__, start, end);
    let selected = q.subrange(start, end);
    let relative = token - start;
    assert(selected[relative] == q[token]);
    let singleton = adapter_output(selected, k, v, seq![table[i]], seq![0int, qlen as int],
        seq![0int, klen as int], qlen, h, hkv, scale, window, fill);
    assert(actual.subrange(start, end) == singleton);
    assert(actual[token] == singleton[relative]);
    let count = (klen - qlen + relative + 1) as nat;
    assert(launch_row_runtime_assumptions(q, k, v, table, cu_q, cu_k,
        h, hkv, scale, window, fill, i, token));
    assert(selected_runtime_assumptions(
        adapter_side(selected, k, v, seq![table[i]], seq![0int, qlen as int],
            seq![0int, klen as int], qlen, h, hkv, scale, window, fill),
        canonical_row_side(q[token], logical_prefix(k, table[i], count), logical_prefix(v, table[i], count),
            h, hkv, scale, window, fill),
        __SELECTED_FREE__ { __PORT_LEFT_SELECTOR__: relative, __PORT_RIGHT_SELECTOR__: 0 }));
    checked_canonical_selected_projection(selected, k, v, table[i], klen, relative,
        h, hkv, scale, window, fill);
    let kp = logical_prefix(k, table[i], count);
    let vp = logical_prefix(v, table[i], count);
    assert(singleton[relative] == canonical_row_output(q[token], kp, vp, h, hkv, scale, window, fill));
    lemma_canonical_row_shape(q[token], kp, vp, h, hkv, scale, window, fill);
    AP::lemma_row_output_exact(op, h * __D__, q[token], kp, vp);
    AP::lemma_selected_row(op, h * __D__, q, k, v, cu_q, cu_k, table, i, token);
    assert(actual.subrange(start, end)[relative] == actual[token]);
    assert(actual[token] == mapped[token]);
}

pub proof fn checked_launch_equivalence(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat, max_k: nat,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
)
    requires
        SUP::paged_attention_metadata_ready(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table),
        TS::tensor2d_shape(q, q.len(), h * __D__),
        TS::tensor3d_shape(k, k.len(), 64, hkv * __D__),
        TS::tensor3d_shape(v, k.len(), 64, hkv * __D__),
        common::blocks_needed_for(max_k) <= u64::MAX,
        h > 0, hkv > 0, hkv <= h, h % hkv == 0,
        __WINDOW_REQUIREMENT__
        launch_runtime_assumptions(q, k, v, table, cu_q, cu_k, h, hkv, scale, window, fill),
    ensures
        adapter_output(q, k, v, table, cu_q, cu_k, max_q,
            h, hkv, scale, window, fill)
        == mapped_launch_output(q, k, v, table, cu_q, cu_k, h, hkv, scale, window, fill),
{
    let op = canonical_row_operation(h, hkv, scale, window, fill);
    let actual = adapter_output(q, k, v, table, cu_q, cu_k, max_q, h, hkv, scale, window, fill);
    let mapped = mapped_launch_output(q, k, v, table, cu_q, cu_k, h, hkv, scale, window, fill);
    lemma_adapter_output_shape(q, k, v, table, cu_q, cu_k, max_q, h, hkv, scale, window, fill);
    AP::lemma_launch_shape(op, h * __D__, q, k, v, cu_q, cu_k, table);
    assert forall|token: int| 0 <= token < q.len() implies (#[trigger] actual[token]) == mapped[token] by {
        checked_launch_row_equivalence(q, k, v, table, cu_q, cu_k, max_q, max_k,
            h, hkv, scale, window, fill, token);
    }
    assert(actual =~= mapped);
}
}
