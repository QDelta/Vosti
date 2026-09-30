//! Checked multi-layer fold for four-norm gated decoders.
//! Family adapters supply immutable borrowed layer permissions and policies.
//! This module owns the KV take/store/put loop, not a family-specific copy.

use crate::model_config::FloatParameterBits;
use crate::boundary::four_norm_gated_weights as W;
use crate::boundary::four_norm_gated_primitives as P;
use crate::exec::four_norm_gated_decoder as D;
use crate::exec::model_kv_loop as KV_LOOP;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::model as MODEL;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
#[cfg(verus_only)]
use crate::boundary::scalar::{float_parameter_scalar_repr, positive_float_parameter_valid};
use vstd::prelude::*;

verus! {

// Architecture-neutral packed-input domain consumed by the complete forward.
// This retains the full prefix for SWA; it asserts neither window eviction
// safety nor equality between different numerical implementations.
pub open spec fn inputs_ready(
    tokens: IntTensor1D, positions: IntTensor1D,
    pre: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>, slots: Seq<int>,
    cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat, max_k: nat,
    bt: Seq<Seq<BlockId>>, count: nat,
) -> bool {
    &&& tokens.len() == positions.len()
    &&& tokens.len() == slots.len()
    &&& cu_q.len() == count + 1
    &&& forall|j: int| 0 <= j < count ==>
        0 < #[trigger] cu_q[j + 1] <= tokens.len()
    &&& forall|j: int| 0 <= j < pre.len() ==>
        #[trigger] RT::store_kv_cache_launch_ready(tokens.len(), pre[j].0, pre[j].1, slots)
    &&& forall|j: int| 0 <= j < pre.len() ==>
        #[trigger] RT::paged_attention_launch_ready(tokens.len(), pre[j].0, pre[j].1,
            cu_q, cu_k, max_q, max_k, bt)
}

// Complete composition shared by four-norm families: embedding, checked KV
// fold, then last-row projection. Family adapters only establish immutable
// role/policy agreement; this function owns no architecture switch or tiles.
#[verifier::spinoff_prover]
pub fn model_forward(
    runtime: &RT::ModelFamilyRuntime,
    embed_weight: &RT::Tensor, final_norm: &RT::Tensor, lm_head: &RT::Tensor,
    Tracked(ep): Tracked<&RT::TensorPerm>, Tracked(np): Tracked<&RT::TensorPerm>,
    Tracked(lp): Tracked<&RT::TensorPerm>,
    layers: &Vec<W::FourNormGatedLayerWeights>,
    Tracked(layer_perms): Tracked<&Map<int, &W::FourNormGatedLayerWeightsPerms>>,
    hidden_size: usize, sliding_window: usize,
    value_epsilon: Option<FloatParameterBits>, softcap: Option<FloatParameterBits>,
    input_ids: &RT::Tensor, Tracked(ip): Tracked<&RT::TensorPerm>,
    positions: &RT::Tensor, Tracked(pp): Tracked<&RT::TensorPerm>,
    kv_caches: &Vec<(RT::Tensor, RT::Tensor)>,
    Tracked(kv_perms): Tracked<&mut RT::KVCachePerms>,
    block_table: &RT::Tensor, Tracked(btp): Tracked<&RT::TensorPerm>,
    slot_mapping: &RT::Tensor, Tracked(sp): Tracked<&RT::TensorPerm>,
    cu_seqlens_q: &RT::Tensor, Tracked(cuqp): Tracked<&RT::TensorPerm>,
    cu_seqlens_k: &RT::Tensor, Tracked(cukp): Tracked<&RT::TensorPerm>,
    max_q: usize, max_k: usize, count: usize,
    Ghost(wr): Ghost<ModelWeightsRepr>, Ghost(decoder): Ghost<FourNormGatedDecoderConfigRepr>,
    Ghost(tokens): Ghost<IntTensor1D>, Ghost(pos): Ghost<IntTensor1D>,
    Ghost(slots): Ghost<Seq<int>>, Ghost(cu_q): Ghost<Seq<int>>, Ghost(cu_k): Ghost<Seq<int>>,
    Ghost(bt): Ghost<Seq<Seq<BlockId>>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        P::hidden_size_matches(runtime, hidden_size as nat), hidden_size > 0,
        decoder.geometry.hidden_size == hidden_size,
        decoder.sliding_window == sliding_window, sliding_window > 0,
        P::runtime_norm_policy(runtime) == Some(decoder.final_norm_policy),
        P::runtime_policy(runtime).unwrap().logits_softcap == decoder.final_logit_softcap,
        decoder.final_logit_softcap == softcap,
        match softcap {
            Some(cap) => positive_float_parameter_valid(cap),
            None => true,
        },
        layers_ready(runtime, layers, layer_perms, sliding_window as nat,
            value_epsilon, wr.layers, decoder.layers),
        RT::tensor_repr_2d(*ep, *embed_weight, wr.embed_weight),
        RT::tensor_repr_1d(*np, *final_norm, wr.final_norm),
        RT::tensor_repr_2d(*lp, *lm_head, wr.lm_head),
        RT::int_tensor_repr_1d(*ip, *input_ids, tokens),
        RT::int_tensor_repr_1d(*pp, *positions, pos),
        RT::int_tensor_repr_1d(*sp, *slot_mapping, slots),
        RT::int_tensor_repr_1d(*cuqp, *cu_seqlens_q, cu_q),
        RT::int_tensor_repr_1d(*cukp, *cu_seqlens_k, cu_k),
        RT::block_table_repr(*btp, *block_table, bt),
        tokens.len() == pos.len(), tokens.len() == slots.len(),
        cu_q.len() == count as nat + 1,
        forall|j: int| 0 <= j < count ==>
            0 < #[trigger] cu_q[j + 1] <= tokens.len(),
        kv_caches.len() == layers.len(), old(kv_perms).len() == layers.len(),
        RT::kv_perms_ids_distinct(*old(kv_perms)),
        old(kv_perms).extracted() == Set::<int>::empty(),
        forall|j: int| 0 <= j < layers.len() ==>
            #[trigger] kv_caches[j].0.id() == old(kv_perms).k_id(j)
            && kv_caches[j].1.id() == old(kv_perms).v_id(j),
        forall|j: int| 0 <= j < layers.len() ==>
            #[trigger] RT::store_kv_cache_launch_ready(tokens.len(),
                old(kv_perms).k_repr(j), old(kv_perms).v_repr(j), slots),
        forall|j: int| 0 <= j < layers.len() ==>
            #[trigger] RT::paged_attention_launch_ready(tokens.len(),
                old(kv_perms).k_repr(j), old(kv_perms).v_repr(j),
                cu_q, cu_k, max_q as nat, max_k as nat, bt),
    ensures ({
        let pre = Seq::new(layers.len() as nat, |i: int|
            (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let logits = MODEL::model_forward_logits_repr(wr, decoder,
            tokens, pos, pre, slots, cu_q, cu_k, max_q as nat, max_k as nat, bt);
        RT::tensor_repr_2d(out.1@, out.0, Seq::new(count as nat, |i: int|
            RT::select_sample_logits_repr(logits, cu_q, i as nat)))
    }),
    final(kv_perms).len() == old(kv_perms).len(),
    final(kv_perms).extracted() == Set::<int>::empty(),
    RT::kv_perms_ids_distinct(*final(kv_perms)),
    forall|j: int| 0 <= j < layers.len() ==>
        #[trigger] final(kv_perms).k_id(j) == old(kv_perms).k_id(j),
    forall|j: int| 0 <= j < layers.len() ==>
        #[trigger] final(kv_perms).v_id(j) == old(kv_perms).v_id(j),
    ({
        let pre = Seq::new(layers.len() as nat, |i: int|
            (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let post = MODEL::model_forward_kv_reprs(wr, decoder,
            tokens, pos, pre, slots, cu_q, cu_k, max_q as nat, max_k as nat, bt);
        forall|j: int| 0 <= j < layers.len() ==>
            #[trigger] final(kv_perms).k_repr(j) == post[j].0
    }),
    ({
        let pre = Seq::new(layers.len() as nat, |i: int|
            (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let post = MODEL::model_forward_kv_reprs(wr, decoder,
            tokens, pos, pre, slots, cu_q, cu_k, max_q as nat, max_k as nat, bt);
        forall|j: int| 0 <= j < layers.len() ==>
            #[trigger] final(kv_perms).v_repr(j) == post[j].1
    }),
{
    let ghost pre = Seq::new(layers.len() as nat, |i: int|
        (kv_perms.k_repr(i), kv_perms.v_repr(i)));
    let (hidden, hp) = P::scaled_embed(runtime, input_ids, embed_weight,
        Tracked(ip), Tracked(ep), hidden_size, Ghost(tokens), Ghost(wr.embed_weight),
        Ghost(Set::<RT::TensorId>::empty()));
    let ghost initial = MODEL::scaled_embed_repr(tokens, wr.embed_weight, hidden_size as nat);
    proof { MODEL::lemma_scaled_embed_repr_shape(tokens, wr.embed_weight, hidden_size as nat); }
    let (current, cp) = run_layers(runtime, layers, Tracked(layer_perms),
        sliding_window, value_epsilon, hidden, hp, positions, Tracked(pp),
        kv_caches, Tracked(kv_perms), block_table, Tracked(btp), slot_mapping, Tracked(sp),
        cu_seqlens_q, Tracked(cuqp), cu_seqlens_k, Tracked(cukp), max_q, max_k,
        Ghost(initial), Ghost(pos), Ghost(pre), Ghost(slots), Ghost(cu_q), Ghost(cu_k),
        Ghost(bt), Ghost(wr.layers), Ghost(decoder.layers), Ghost(Set::<RT::TensorId>::empty()));
    let out = D::project_last_logits(runtime, &current, final_norm, lm_head,
        cu_seqlens_q, count, softcap, Tracked(cp.borrow()), Tracked(np), Tracked(lp), Tracked(cuqp),
        Ghost(decoder.final_norm_policy), Ghost(cp@.repr_2d()), Ghost(wr.final_norm),
        Ghost(wr.lm_head), Ghost(cu_q), Ghost(Set::<RT::TensorId>::empty().insert(current.id())));
    proof {
        reveal(MODEL::model_forward_logits_repr);
        reveal(MODEL::model_forward_kv_reprs);
        reveal(MODEL::model_forward_hidden_and_kv_reprs);
        assert(pre =~= Seq::new(layers.len() as nat, |i: int|
            (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i))));
        let post = MODEL::model_forward_kv_reprs(wr, decoder,
            tokens, pos, pre, slots, cu_q, cu_k, max_q as nat, max_k as nat, bt);
        assert forall|j: int|
            #![trigger kv_perms.k_id(j)]
            #![trigger kv_perms.v_id(j)]
            #![trigger kv_perms.k_repr(j)]
            #![trigger kv_perms.v_repr(j)]
            0 <= j < layers.len() implies {
            &&& kv_perms.k_id(j) == old(kv_perms).k_id(j)
            &&& kv_perms.v_id(j) == old(kv_perms).v_id(j)
            &&& kv_perms.k_repr(j) == post[j].0
            &&& kv_perms.v_repr(j) == post[j].1
        } by {
            assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
            assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
            assert(kv_perms.k_repr(j) == post[j].0);
            assert(kv_perms.v_repr(j) == post[j].1);
        }
    }
    out
}

pub open spec fn layers_ready(
    runtime: &RT::ModelFamilyRuntime,
    layers: &Vec<W::FourNormGatedLayerWeights>,
    perms: &Map<int, &W::FourNormGatedLayerWeightsPerms>,
    window: nat, value_epsilon: Option<FloatParameterBits>,
    common: Seq<LayerWeightsRepr>, extensions: Seq<FourNormGatedLayerExtensionRepr>,
) -> bool {
    &&& common.len() == layers.len()
    &&& extensions.len() == layers.len()
    &&& forall|i: int| 0 <= i < layers.len() ==> #[trigger] perms.dom().contains(i)
    &&& forall|i: int| 0 <= i < layers.len() ==>
        #[trigger] common[i] == W::layer_weights_common_repr_of(perms[i])
    &&& forall|i: int| 0 <= i < layers.len() ==> {
        &&& #[trigger] D::layer_execution_policy_matches(
            runtime, &layers[i], perms[i], window, value_epsilon, extensions[i])
    }
}

pub fn run_layers(
    runtime: &RT::ModelFamilyRuntime,
    layers: &Vec<W::FourNormGatedLayerWeights>,
    Tracked(layer_perms): Tracked<&Map<int, &W::FourNormGatedLayerWeightsPerms>>,
    sliding_window: usize,
    value_epsilon: Option<FloatParameterBits>,
    hidden: RT::Tensor,
    hp: Tracked<RT::TensorPerm>,
    positions: &RT::Tensor,
    Tracked(pp): Tracked<&RT::TensorPerm>,
    kv_caches: &Vec<(RT::Tensor, RT::Tensor)>,
    Tracked(kv_perms): Tracked<&mut RT::KVCachePerms>,
    block_table: &RT::Tensor,
    Tracked(bt_perm): Tracked<&RT::TensorPerm>,
    slot_mapping: &RT::Tensor,
    Tracked(sp): Tracked<&RT::TensorPerm>,
    cu_seqlens_q: &RT::Tensor,
    Tracked(cuq_perm): Tracked<&RT::TensorPerm>,
    cu_seqlens_k: &RT::Tensor,
    Tracked(cuk_perm): Tracked<&RT::TensorPerm>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    Ghost(initial_hidden): Ghost<Tensor2D>,
    Ghost(positions_repr): Ghost<IntTensor1D>,
    Ghost(pre_kv_reprs): Ghost<Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>>,
    Ghost(slot_repr): Ghost<Seq<int>>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(cu_k_repr): Ghost<Seq<int>>,
    Ghost(bt_repr): Ghost<Seq<Seq<BlockId>>>,
    Ghost(common_layers): Ghost<Seq<LayerWeightsRepr>>,
    Ghost(extension_layers): Ghost<Seq<FourNormGatedLayerExtensionRepr>>,
    Ghost(scope0): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        layers_ready(runtime, layers, layer_perms, sliding_window as nat,
            value_epsilon, common_layers, extension_layers),
        sliding_window > 0,
        RT::tensor_repr_2d(hp@, hidden, initial_hidden),
        RT::int_tensor_repr_1d(*pp, *positions, positions_repr),
        RT::block_table_repr(*bt_perm, *block_table, bt_repr),
        RT::int_tensor_repr_1d(*sp, *slot_mapping, slot_repr),
        RT::int_tensor_repr_1d(*cuq_perm, *cu_seqlens_q, cu_q_repr),
        RT::int_tensor_repr_1d(*cuk_perm, *cu_seqlens_k, cu_k_repr),
        initial_hidden.len() == positions_repr.len(),
        initial_hidden.len() == slot_repr.len(),
        kv_caches.len() == layers.len(),
        old(kv_perms).len() == layers.len(),
        pre_kv_reprs.len() == layers.len(),
        old(kv_perms).extracted() == Set::<int>::empty(),
        RT::kv_perms_ids_distinct(*old(kv_perms)),
        forall|j: int| 0 <= j < layers.len() ==>
            #[trigger] kv_caches[j].0.id() == old(kv_perms).k_id(j)
                && kv_caches[j].1.id() == old(kv_perms).v_id(j),
        forall|j: int| 0 <= j < layers.len() ==>
            #[trigger] pre_kv_reprs[j] == (old(kv_perms).k_repr(j), old(kv_perms).v_repr(j)),
        forall|j: int| 0 <= j < layers.len() ==>
            #[trigger] RT::store_kv_cache_launch_ready(initial_hidden.len(),
                pre_kv_reprs[j].0, pre_kv_reprs[j].1, slot_repr),
        forall|j: int| 0 <= j < layers.len() ==>
            #[trigger] RT::paged_attention_launch_ready(initial_hidden.len(),
                pre_kv_reprs[j].0, pre_kv_reprs[j].1, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr),
    ensures ({
        let folded = MODEL::layer_chain_repr(common_layers, extension_layers,
            initial_hidden, positions_repr, pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, 0);
        &&& RT::tensor_repr_2d(out.1@, out.0, folded.0)
        &&& out.1@.repr_2d().len() == initial_hidden.len()
        &&& final(kv_perms).len() == old(kv_perms).len()
        &&& final(kv_perms).extracted() == Set::<int>::empty()
        &&& RT::kv_perms_ids_distinct(*final(kv_perms))
        &&& folded.1.len() == layers.len()
        &&& forall|j: int| 0 <= j < layers.len() ==> {
            &&& #[trigger] final(kv_perms).k_id(j) == old(kv_perms).k_id(j)
            &&& final(kv_perms).v_id(j) == old(kv_perms).v_id(j)
            &&& final(kv_perms).k_repr(j) == folded.1[j].0
            &&& final(kv_perms).v_repr(j) == folded.1[j].1
        }
    }),
{
    let n = layers.len();
    let ghost full_fold = MODEL::layer_chain_repr(common_layers, extension_layers,
        initial_hidden, positions_repr, pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
        max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, 0);
    let mut current = hidden;
    let mut current_p = hp;
    let mut scope = Ghost(scope0.insert(current.id()));
    proof {
        assert(pre_kv_reprs =~= Seq::new(n as nat, |j: int|
            (kv_perms.k_repr(j), kv_perms.v_repr(j))));
        assert forall|j: int| 0 <= j < n as int implies
            #[trigger] kv_caches[j].0.id() == kv_perms.k_id(j)
                && kv_caches[j].1.id() == kv_perms.v_id(j) by {
            assert(kv_caches[j].0.id() == old(kv_perms).k_id(j));
            assert(kv_caches[j].1.id() == old(kv_perms).v_id(j));
        }
    }
    let mut i: usize = 0;
    while i < n
        invariant
            RT::paged_attention_numeric_domain(),
            i <= n,
            n == layers.len(),
            n == kv_perms.len(),
            n == kv_caches.len(),
            layers_ready(runtime, layers, layer_perms, sliding_window as nat,
                value_epsilon, common_layers, extension_layers),
            sliding_window > 0,
            RT::tensor_repr_2d(current_p@, current, current_p@.repr_2d()),
            current_p@.repr_2d().len() == initial_hidden.len(),
            scope@.contains(current.id()),
            RT::int_tensor_repr_1d(*pp, *positions, positions_repr),
            RT::block_table_repr(*bt_perm, *block_table, bt_repr),
            RT::int_tensor_repr_1d(*sp, *slot_mapping, slot_repr),
            RT::int_tensor_repr_1d(*cuq_perm, *cu_seqlens_q, cu_q_repr),
            RT::int_tensor_repr_1d(*cuk_perm, *cu_seqlens_k, cu_k_repr),
            initial_hidden.len() == positions_repr.len(),
            initial_hidden.len() == slot_repr.len(),
            pre_kv_reprs.len() == n as nat,
            common_layers.len() == n as nat,
            extension_layers.len() == n as nat,
            kv_perms.extracted() == Set::<int>::empty(),
            RT::kv_perms_ids_distinct(*kv_perms),
            forall|j: int| 0 <= j < n as int ==>
                #[trigger] kv_perms.k_id(j) == old(kv_perms).k_id(j)
                    && kv_perms.v_id(j) == old(kv_perms).v_id(j),
            forall|j: int| 0 <= j < n as int ==>
                #[trigger] kv_caches[j].0.id() == kv_perms.k_id(j)
                    && kv_caches[j].1.id() == kv_perms.v_id(j),
            forall|j: int| 0 <= j < n as int ==>
                #[trigger] RT::store_kv_cache_launch_ready(
                    initial_hidden.len(), pre_kv_reprs[j].0,
                    pre_kv_reprs[j].1, slot_repr,
                ),
            forall|j: int| 0 <= j < n as int ==>
                #[trigger] RT::paged_attention_launch_ready(
                    initial_hidden.len(), pre_kv_reprs[j].0,
                    pre_kv_reprs[j].1, cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                ),
            KV_LOOP::suffix_matches(
                kv_perms, pre_kv_reprs, i as int, n as int,
            ),
            full_fold == MODEL::layer_chain_repr(
                common_layers, extension_layers, current_p@.repr_2d(),
                positions_repr,
                Seq::new(n as nat, |j: int|
                    (kv_perms.k_repr(j), kv_perms.v_repr(j))),
                slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, i as nat,
            ),
        decreases n - i,
    {
        let next_i = i + 1;
        let ghost old_hidden = current_p@.repr_2d();
        let ghost caches_before = Seq::new(n as nat, |j: int|
            (kv_perms.k_repr(j), kv_perms.v_repr(j)));
        let ghost pre_take_k_ids = Seq::new(n as nat, |j: int|
            kv_perms.k_id(j));
        let ghost pre_take_v_ids = Seq::new(n as nat, |j: int|
            kv_perms.v_id(j));
        proof {
            KV_LOOP::suffix_matches_at(
                kv_perms, pre_kv_reprs, i as int, n as int, i as int,
            );
            assert(caches_before[i as int] == pre_kv_reprs[i as int]);
            assert forall|j: int| i as int <= j < n as int implies
                #[trigger] caches_before[j].0 == pre_kv_reprs[j].0
                    && caches_before[j].1 == pre_kv_reprs[j].1 by {
                KV_LOOP::suffix_matches_at(
                    kv_perms, pre_kv_reprs, i as int, n as int, j,
                );
            };
        }

        let tracked layer_p = *layer_perms.tracked_borrow(i as int);
        let tracked (k_taken, v_taken) =
            kv_perms.tracked_take_layer(i as int);
        let tracked mut k_perm = k_taken;
        let tracked mut v_perm = v_taken;
        let ghost k_in = k_perm.kv_cache_repr();
        let ghost v_in = v_perm.kv_cache_repr();
        let ghost post_take_k_ids = Seq::new(n as nat, |j: int|
            kv_perms.k_id(j));
        let ghost post_take_v_ids = Seq::new(n as nat, |j: int|
            kv_perms.v_id(j));
        let ghost post_take_k = Seq::new(n as nat, |j: int|
            kv_perms.k_repr(j));
        let ghost post_take_v = Seq::new(n as nat, |j: int|
            kv_perms.v_repr(j));

        proof {
            assert(k_perm.id() == kv_caches[i as int].0.id());
            assert(v_perm.id() == kv_caches[i as int].1.id());
            assert(k_in == pre_kv_reprs[i as int].0);
            assert(v_in == pre_kv_reprs[i as int].1);
            assert(common_layers[i as int] == W::layer_weights_common_repr_of(layer_p));
            assert(D::layer_execution_policy_matches(runtime, &layers[i as int],
                layer_p, sliding_window as nat, value_epsilon,
                extension_layers[i as int]));
            assert forall|j: int| 0 <= j < n as int implies
                #[trigger] post_take_k_ids[j] == pre_take_k_ids[j]
                    && post_take_v_ids[j] == pre_take_v_ids[j] by {
                assert(kv_perms.k_id(j) == pre_take_k_ids[j]);
                assert(kv_perms.v_id(j) == pre_take_v_ids[j]);
            };
            assert forall|j: int| next_i as int <= j < n as int implies
                #[trigger] post_take_k[j] == pre_kv_reprs[j].0
                    && post_take_v[j] == pre_kv_reprs[j].1 by {
                assert(j != i as int);
                assert(post_take_k[j] == caches_before[j].0);
                assert(post_take_v[j] == caches_before[j].1);
                assert(caches_before[j].0 == pre_kv_reprs[j].0);
                assert(caches_before[j].1 == pre_kv_reprs[j].1);
            };
            assert forall|j: int| 0 <= j < n as int implies
                #[trigger] pre_take_k_ids[j] == old(kv_perms).k_id(j)
                    && pre_take_v_ids[j] == old(kv_perms).v_id(j) by {
                assert(pre_take_k_ids[j] == kv_perms.k_id(j));
                assert(pre_take_v_ids[j] == kv_perms.v_id(j));
                assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
                assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
            };
        }

        let (next, next_p) = D::decoder_layer_forward(
            runtime, &layers[i], Tracked(layer_p), sliding_window, value_epsilon,
            Ghost(extension_layers[i as int]),
            &current, Tracked(current_p.borrow()),
            positions, Tracked(pp),
            &kv_caches[i].0, Tracked(&mut k_perm),
            &kv_caches[i].1, Tracked(&mut v_perm),
            block_table, Tracked(bt_perm),
            slot_mapping, Tracked(sp),
            cu_seqlens_q, Tracked(cuq_perm),
            cu_seqlens_k, Tracked(cuk_perm),
            max_seqlen_q, max_seqlen_k,
            Ghost(old_hidden), Ghost(positions_repr),
            Ghost(k_in), Ghost(v_in), Ghost(bt_repr), Ghost(slot_repr),
            Ghost(cu_q_repr), Ghost(cu_k_repr), scope,
        );

        proof {
            assert(k_perm.id() == kv_perms.k_id(i as int));
            assert(v_perm.id() == kv_perms.v_id(i as int));
            kv_perms.tracked_put_layer(i as int, k_perm, v_perm);
            assert forall|j: int| 0 <= j < n as int implies
                #[trigger] kv_perms.k_id(j) == old(kv_perms).k_id(j)
                    && kv_perms.v_id(j) == old(kv_perms).v_id(j) by {
                assert(kv_perms.k_id(j) == post_take_k_ids[j]);
                assert(kv_perms.v_id(j) == post_take_v_ids[j]);
                assert(post_take_k_ids[j] == pre_take_k_ids[j]);
                assert(post_take_v_ids[j] == pre_take_v_ids[j]);
                assert(pre_take_k_ids[j] == old(kv_perms).k_id(j));
                assert(pre_take_v_ids[j] == old(kv_perms).v_id(j));
            };
            assert forall|j: int| 0 <= j < n as int implies
                #[trigger] kv_caches[j].0.id() == kv_perms.k_id(j)
                    && kv_caches[j].1.id() == kv_perms.v_id(j) by {
                assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
                assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
            };
            assert forall|j: int| next_i as int <= j < n as int implies
                #[trigger] kv_perms.k_repr(j) == pre_kv_reprs[j].0
                    && kv_perms.v_repr(j) == pre_kv_reprs[j].1 by {
                assert(kv_perms.k_id(j) == kv_perms.k_id(j));
            };
            assert(KV_LOOP::suffix_matches(
                kv_perms, pre_kv_reprs, next_i as int, n as int,
            ));

            let ghost layer_step = MODEL::decoder_layer_step_repr(
                common_layers[i as int], extension_layers[i as int],
                old_hidden, positions_repr,
                pre_kv_reprs[i as int].0, pre_kv_reprs[i as int].1,
                slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            );
            assert(next_p@.repr_2d() == layer_step.0);
            assert(kv_perms.k_repr(i as int) == layer_step.1.0);
            assert(kv_perms.v_repr(i as int) == layer_step.1.1);
            let ghost caches_after = Seq::new(n as nat, |j: int|
                (kv_perms.k_repr(j), kv_perms.v_repr(j)));
            assert(caches_after =~= caches_before.update(i as int, layer_step.1)) by {
                assert forall|j: int| 0 <= j < n as int && j != i as int implies
                    #[trigger] caches_after[j] == caches_before[j] by {
                    assert(kv_perms.k_id(j) == kv_perms.k_id(j));
                };
            };
            MODEL::lemma_decoder_layer_step_repr_shape(
                common_layers[i as int], extension_layers[i as int],
                old_hidden, positions_repr,
                pre_kv_reprs[i as int].0, pre_kv_reprs[i as int].1,
                slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            );
            reveal_with_fuel(MODEL::layer_chain_repr, 1);
            assert(full_fold == MODEL::layer_chain_repr(
                common_layers, extension_layers, next_p@.repr_2d(),
                positions_repr, caches_after,
                slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                next_i as nat,
            ));
        }

        current = next;
        current_p = next_p;
        scope = Ghost(scope@.insert(current.id()));
        i = next_i;
        proof {
            assert forall|j: int| 0 <= j < n as int implies
                #[trigger] kv_perms.k_id(j) == old(kv_perms).k_id(j)
                    && kv_perms.v_id(j) == old(kv_perms).v_id(j) by {
                assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
                assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
            }
        }
    }

    let ghost final_hidden = current_p@.repr_2d();
    let ghost final_caches = Seq::new(n as nat, |j: int|
        (kv_perms.k_repr(j), kv_perms.v_repr(j)));
    proof {
        reveal_with_fuel(MODEL::layer_chain_repr, 1);
        assert(full_fold == (final_hidden, final_caches));
    }

    (current, current_p)
}

} // verus!
