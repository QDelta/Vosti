//! Family-neutral dense-kernel projection and batch-invariance lemmas.
//!
// Per-kernel batch-invariance lemmas + repr-level decomposition lemmas.
// Mirrors the earlier Dafny batch-invariance proof structure.
//
// All lemmas here operate on repr types only — no `Tensor`/perm params.
// Every body in this module is checked (zero `admit()`). Row-local facts follow
// from structural configuration-free `*_repr` definitions. Per-launch
// ContractIR certificates are offline deployment evidence and do not occur in
// these engine proofs.
// Attention uses the generated canonical raw row operation. Its batch and
// causal projections below follow from checked shared representation lemmas;
// the runtime binding consumes the exact raw kernel theorems and conditions.
//
// Each batch-invariance theorem states: for batch index `i`, the i-th element
// of the batched output equals the output of running the kernel on a
// singleton input drawn from element `i`.  This is the kernel-level fact
// that the engine-vs-machine refinement leans on.

use crate::model_config::{AttentionKind, FloatParameterBits};
#[cfg(verus_only)]
use crate::boundary::attention_operator as AO;
#[cfg(verus_only)]
use crate::boundary::dense_layer_primitives as DLP;
#[cfg(verus_only)]
use crate::{types::{BLOCK_SIZE_SPEC}, proof::tensor::geometry::{block_table_slot, blocks_needed_for, cache_at, slot_in_cache}};
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
#[cfg(verus_only)]
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

verus! {

// ---------------------------------------------------------------------------
// Per-token kernel batch-invariance theorems.
// ---------------------------------------------------------------------------

pub broadcast proof fn embed_batch_invariance(ir: IntTensor1D, wr: Tensor2D, i: nat)
    requires i < ir.len(),
    ensures #[trigger] RT::embed_repr(seq![ir[i as int]], wr)
        == seq![RT::embed_repr(ir, wr)[i as int]],
{
    reveal(RT::embed_repr);
    assert(RT::embed_repr(seq![ir[i as int]], wr)
        =~= seq![RT::embed_repr(ir, wr)[i as int]]);
}

pub broadcast proof fn linear_batch_invariance(xr: Tensor2D, wr: Tensor2D, i: nat)
    requires i < xr.len(),
    ensures #[trigger] RT::linear_repr(seq![xr[i as int]], wr)
        == seq![RT::linear_repr(xr, wr)[i as int]],
{
    reveal(RT::linear_repr);
    assert(RT::linear_repr(seq![xr[i as int]], wr)
        =~= seq![RT::linear_repr(xr, wr)[i as int]]);
}

pub broadcast proof fn qkv_linear_batch_invariance(
    xr: Tensor2D,
    qwr: Tensor2D,
    kwr: Tensor2D,
    vwr: Tensor2D,
    i: nat,
)
    requires i < xr.len(),
    ensures
        #[trigger] RT::qkv_linear_repr(
            seq![xr[i as int]], qwr, kwr, vwr,
        ).0 == seq![RT::qkv_linear_repr(xr, qwr, kwr, vwr).0[i as int]],
        RT::qkv_linear_repr(
            seq![xr[i as int]], qwr, kwr, vwr,
        ).1 == seq![RT::qkv_linear_repr(xr, qwr, kwr, vwr).1[i as int]],
        RT::qkv_linear_repr(
            seq![xr[i as int]], qwr, kwr, vwr,
        ).2 == seq![RT::qkv_linear_repr(xr, qwr, kwr, vwr).2[i as int]],
{
    reveal(RT::qkv_linear_repr);
}

// Selecting input rows commutes with a row-local linear projection.  This is
// the proof step that permits gathering the requests' last hidden rows before
// the vocabulary-width lm_head GEMM while retaining the original full-logits
// specification.
pub proof fn linear_selected_row_invariance(
    xr: Tensor2D,
    wr: Tensor2D,
    selected: Tensor2D,
    i: int,
    source: int,
)
    requires
        0 <= i < selected.len(),
        0 <= source < xr.len(),
        selected[i] == xr[source],
    ensures
        RT::linear_repr(selected, wr)[i] == RT::linear_repr(xr, wr)[source],
{
    linear_batch_invariance(selected, wr, i as nat);
    linear_batch_invariance(xr, wr, source as nat);
    assert(seq![selected[i]] =~= seq![xr[source]]);
    assert(seq![RT::linear_repr(selected, wr)[i]]
        == seq![RT::linear_repr(xr, wr)[source]]);
    assert(seq![RT::linear_repr(selected, wr)[i]][0]
        == seq![RT::linear_repr(xr, wr)[source]][0]);
}

pub broadcast proof fn rms_norm_batch_invariance(
    xr: Tensor2D,
    wr: Tensor1D,
    epsilon: FloatParameterBits,
    i: nat,
)
    requires i < xr.len(),
    ensures #[trigger] RT::rms_norm_repr(seq![xr[i as int]], wr, epsilon)
        == seq![RT::rms_norm_repr(xr, wr, epsilon)[i as int]],
{
    reveal(RT::rms_norm_repr);
    reveal(RT::rms_norm_kernel_repr);
    assert(RT::rms_norm_repr(seq![xr[i as int]], wr, epsilon)
        =~= seq![RT::rms_norm_repr(xr, wr, epsilon)[i as int]]);
}

pub broadcast proof fn add_rms_norm_batch_invariance(
    xr: Tensor2D,
    rr: Tensor2D,
    wr: Tensor1D,
    epsilon: FloatParameterBits,
    i: nat,
)
    requires xr.len() == rr.len(), i < xr.len(),
    ensures (#[trigger] RT::add_rms_norm_repr(
                seq![xr[i as int]], seq![rr[i as int]], wr, epsilon,
            )).0 == seq![RT::add_rms_norm_repr(xr, rr, wr, epsilon).0[i as int]],
            RT::add_rms_norm_repr(
                seq![xr[i as int]], seq![rr[i as int]], wr, epsilon,
            ).1 == seq![RT::add_rms_norm_repr(xr, rr, wr, epsilon).1[i as int]],
{
    reveal(RT::add_rms_norm_repr);
    reveal(RT::add_rms_norm_output_kernel_repr);
    reveal(RT::add_rms_norm_residual_kernel_repr);
    assert(RT::add_rms_norm_repr(
        seq![xr[i as int]], seq![rr[i as int]], wr, epsilon,
    ).0 =~= seq![RT::add_rms_norm_repr(xr, rr, wr, epsilon).0[i as int]]);
    assert(RT::add_rms_norm_repr(
        seq![xr[i as int]], seq![rr[i as int]], wr, epsilon,
    ).1 =~= seq![RT::add_rms_norm_repr(xr, rr, wr, epsilon).1[i as int]]);
}

pub broadcast proof fn qk_norm_batch_invariance(
    qr: Tensor2D,
    kr: Tensor2D,
    qnr: Tensor1D,
    knr: Tensor1D,
    epsilon: FloatParameterBits,
    i: nat,
)
    requires
        qr.len() == kr.len(),
        i < qr.len(),
        TS::rectangular(qr),
        TS::rectangular(kr),
    ensures (#[trigger] RT::qk_norm_repr(
                seq![qr[i as int]], seq![kr[i as int]], qnr, knr, epsilon,
            )).0 == seq![RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).0[i as int]],
            RT::qk_norm_repr(
                seq![qr[i as int]], seq![kr[i as int]], qnr, knr, epsilon,
            ).1 == seq![RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).1[i as int]],
{
    reveal(RT::qk_norm_repr);
    reveal(RT::head_rms_norm_kernel_repr);
    assert(RT::qk_norm_repr(
        seq![qr[i as int]], seq![kr[i as int]], qnr, knr, epsilon,
    ).0 =~= seq![RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).0[i as int]]);
    assert(RT::qk_norm_repr(
        seq![qr[i as int]], seq![kr[i as int]], qnr, knr, epsilon,
    ).1 =~= seq![RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).1[i as int]]);
}

pub broadcast proof fn view_as_kv_batch_invariance(vr: Tensor2D, i: nat)
    requires i < vr.len(),
    ensures #[trigger] RT::view_as_kv_repr(seq![vr[i as int]])
        == seq![RT::view_as_kv_repr(vr)[i as int]],
{
    reveal(RT::view_as_kv_repr);
    assert(RT::view_as_kv_repr(seq![vr[i as int]]) =~= seq![RT::view_as_kv_repr(vr)[i as int]]);
}

pub broadcast proof fn rope_kernel_batch_invariance(
    xr: Tensor2D, cosr: Tensor2D, sinr: Tensor2D, i: nat,
)
    requires
        xr.len() == cosr.len(),
        xr.len() == sinr.len(),
        i < xr.len(),
    ensures
        #[trigger] RT::rope_kernel_repr(
            seq![xr[i as int]], seq![cosr[i as int]], seq![sinr[i as int]],
        ) == seq![RT::rope_kernel_repr(xr, cosr, sinr)[i as int]],
{
    reveal(RT::rope_kernel_repr);
    assert(RT::rope_kernel_repr(
        seq![xr[i as int]], seq![cosr[i as int]], seq![sinr[i as int]],
    ) =~= seq![RT::rope_kernel_repr(xr, cosr, sinr)[i as int]]);
}

pub proof fn rope_kernel_subrange_invariance(
    xr: Tensor2D, cosr: Tensor2D, sinr: Tensor2D, a: int, b: int,
)
    requires
        xr.len() == cosr.len(),
        xr.len() == sinr.len(),
        0 <= a <= b <= xr.len(),
    ensures
        RT::rope_kernel_repr(
            xr.subrange(a, b), cosr.subrange(a, b), sinr.subrange(a, b),
        ) == RT::rope_kernel_repr(xr, cosr, sinr).subrange(a, b),
{
    let xs = xr.subrange(a, b);
    let cs = cosr.subrange(a, b);
    let ss = sinr.subrange(a, b);
    RT::lemma_rope_kernel_repr_shape(xs, cs, ss);
    RT::lemma_rope_kernel_repr_shape(xr, cosr, sinr);
    assert forall|j: int| 0 <= j < xs.len() implies
        #[trigger] RT::rope_kernel_repr(xs, cs, ss)[j]
            == RT::rope_kernel_repr(xr, cosr, sinr)[a + j]
    by {
        rope_kernel_batch_invariance(xs, cs, ss, j as nat);
        rope_kernel_batch_invariance(xr, cosr, sinr, (a + j) as nat);
        assert(xs[j] == xr[a + j]);
        assert(cs[j] == cosr[a + j]);
        assert(ss[j] == sinr[a + j]);
        assert(RT::rope_kernel_repr(seq![xs[j]], seq![cs[j]], seq![ss[j]])
            == RT::rope_kernel_repr(
                seq![xr[a + j]], seq![cosr[a + j]], seq![sinr[a + j]],
            ));
        assert(RT::rope_kernel_repr(seq![xs[j]], seq![cs[j]], seq![ss[j]])
            == seq![RT::rope_kernel_repr(xs, cs, ss)[j]]);
        assert(RT::rope_kernel_repr(
            seq![xr[a + j]], seq![cosr[a + j]], seq![sinr[a + j]],
        ) == seq![RT::rope_kernel_repr(xr, cosr, sinr)[a + j]]);
        assert(seq![RT::rope_kernel_repr(xs, cs, ss)[j]][0]
            == seq![RT::rope_kernel_repr(xr, cosr, sinr)[a + j]][0]);
    }
    assert(RT::rope_kernel_repr(xs, cs, ss)
        =~= RT::rope_kernel_repr(xr, cosr, sinr).subrange(a, b));
}

pub proof fn rotary_component_batch_invariance(
    pir: IntTensor1D,
    xr: Tensor2D,
    head_count: nat,
    head_dim: nat,
    rotary: RotaryConfigRepr,
    i: nat,
)
    requires pir.len() == xr.len(), i < xr.len(),
    ensures
        #[trigger] RT::rotary_component_repr(
            seq![pir[i as int]], seq![xr[i as int]], head_count, head_dim,
            rotary,
        ) == seq![RT::rotary_component_repr(
            pir, xr, head_count, head_dim, rotary,
        )[i as int]],
{
    reveal(RT::rotary_component_repr);
    let split = RT::split_head_rows_repr(
        xr, head_count, head_dim,
    );
    let cos_rows = RT::rope_cos_rows_repr(pir, rotary);
    let sin_rows = RT::rope_sin_rows_repr(pir, rotary);
    let cosr = RT::repeat_rows_repr(cos_rows, head_count);
    let sinr = RT::repeat_rows_repr(sin_rows, head_count);
    let out = RT::rope_kernel_repr(split, cosr, sinr);
    let lo = i as int * head_count as int;
    let hi = (i as int + 1) * head_count as int;

    RT::lemma_split_head_rows_repr_shape(
        xr, head_count, head_dim,
    );
    RT::lemma_rope_table_rows_shape(pir, rotary);
    RT::lemma_repeat_rows_repr_shape(cos_rows, head_count);
    RT::lemma_repeat_rows_repr_shape(sin_rows, head_count);
    assert(split.len() == cosr.len());
    assert(split.len() == sinr.len());
    assert((i as int) < (xr.len() as int));
    assert(0 <= i as int);
    assert(0 <= head_count as int);
    assert(0 <= lo) by (nonlinear_arith)
        requires 0 <= i as int, 0 <= head_count as int,
            lo == i as int * head_count as int,
    {}
    assert(lo <= hi) by (nonlinear_arith)
        requires 0 <= head_count as int,
            lo == i as int * head_count as int,
            hi == (i as int + 1) * head_count as int,
    {}
    assert(hi <= split.len() as int) by (nonlinear_arith)
        requires i as int + 1 <= xr.len() as int,
            0 <= head_count as int,
            hi == (i as int + 1) * head_count as int,
            split.len() as int == xr.len() as int * head_count as int,
    {}
    assert(0 <= lo <= hi <= split.len());
    RT::lemma_split_head_rows_projection(
        xr, head_count, head_dim, i,
    );
    RT::lemma_rope_table_rows_projection(pir, rotary, i);
    RT::lemma_repeat_rows_projection(cos_rows, head_count, i);
    RT::lemma_repeat_rows_projection(sin_rows, head_count, i);
    assert(RT::repeat_rows_repr(
        RT::rope_cos_rows_repr(seq![pir[i as int]], rotary), head_count,
    ) == cosr.subrange(lo, hi));
    assert(RT::repeat_rows_repr(
        RT::rope_sin_rows_repr(seq![pir[i as int]], rotary), head_count,
    ) == sinr.subrange(lo, hi));
    rope_kernel_subrange_invariance(split, cosr, sinr, lo, hi);
    assert(RT::rope_kernel_repr(
        RT::split_head_rows_repr(
            seq![xr[i as int]], head_count, head_dim,
        ),
        RT::repeat_rows_repr(
            RT::rope_cos_rows_repr(seq![pir[i as int]], rotary), head_count,
        ),
        RT::repeat_rows_repr(
            RT::rope_sin_rows_repr(seq![pir[i as int]], rotary), head_count,
        ),
    ) == out.subrange(lo, hi));
    RT::lemma_rope_kernel_repr_shape(split, cosr, sinr);
    RT::lemma_merge_head_rows_projection(out, xr.len(), head_count, i);
}

pub broadcast proof fn rotary_embed_batch_invariance(
    geometry: AttentionGeometryRepr,
    rotary: RotaryConfigRepr,
    pir: IntTensor1D, qr: Tensor2D, kr: Tensor2D, i: nat,
)
    requires pir.len() == qr.len(), qr.len() == kr.len(), i < qr.len(),
    ensures (#[trigger] DLP::rotary_embed_repr(geometry, rotary, seq![pir[i as int]],
                seq![qr[i as int]], seq![kr[i as int]])).0
            == seq![DLP::rotary_embed_repr(geometry, rotary, pir, qr, kr).0[i as int]],
            DLP::rotary_embed_repr(geometry, rotary, seq![pir[i as int]],
                seq![qr[i as int]], seq![kr[i as int]]).1
            == seq![DLP::rotary_embed_repr(geometry, rotary, pir, qr, kr).1[i as int]],
{
    DLP::lemma_rotary_embed_repr_shape(geometry, rotary, pir, qr, kr);
    reveal(DLP::rotary_embed_repr);
    reveal(RT::rotary_embed_repr);
    rotary_component_batch_invariance(
        pir, qr, geometry.num_attention_heads, geometry.head_dim,
        rotary, i,
    );
    rotary_component_batch_invariance(
        pir, kr, geometry.num_key_value_heads, geometry.head_dim,
        rotary, i,
    );
    assert(DLP::rotary_embed_repr(
        geometry, rotary, seq![pir[i as int]], seq![qr[i as int]], seq![kr[i as int]],
    ).0 == RT::rotary_component_repr(
        seq![pir[i as int]], seq![qr[i as int]],
        geometry.num_attention_heads, geometry.head_dim, rotary,
    ));
    assert(DLP::rotary_embed_repr(geometry, rotary, pir, qr, kr).0
        == RT::rotary_component_repr(
            pir, qr, geometry.num_attention_heads, geometry.head_dim, rotary,
        ));
    assert(DLP::rotary_embed_repr(
        geometry, rotary, seq![pir[i as int]], seq![qr[i as int]], seq![kr[i as int]],
    ).1 == RT::rotary_component_repr(
        seq![pir[i as int]], seq![kr[i as int]],
        geometry.num_key_value_heads, geometry.head_dim, rotary,
    ));
    assert(DLP::rotary_embed_repr(geometry, rotary, pir, qr, kr).1
        == RT::rotary_component_repr(
            pir, kr, geometry.num_key_value_heads, geometry.head_dim, rotary,
        ));
}

pub broadcast proof fn silu_and_mul_batch_invariance(xr: Tensor2D, i: nat)
    requires
        i < xr.len(),
        TS::rectangular(xr),
    ensures #[trigger] RT::silu_and_mul_repr(seq![xr[i as int]])
        == seq![RT::silu_and_mul_repr(xr)[i as int]],
{
    let cols = choose|cols: nat| TS::tensor2d_shape(xr, xr.len(), cols);
    assert(TS::tensor2d_shape(xr, xr.len(), cols));
    RT::lemma_silu_and_mul_repr_shape(xr);
    RT::lemma_split_last_axis_half_repr_tensor2d_shape(
        xr, xr.len(), cols, false,
    );
    RT::lemma_split_last_axis_half_repr_tensor2d_shape(
        xr, xr.len(), cols, true,
    );
    RT::lemma_split_last_axis_half_projection(xr, false, i);
    RT::lemma_split_last_axis_half_projection(xr, true, i);
    reveal(RT::silu_and_mul_repr);
    reveal(RT::silu_mul_kernel_repr);
    let left = RT::split_last_axis_half_repr(xr, false);
    let right = RT::split_last_axis_half_repr(xr, true);
    assert(left.len() == xr.len());
    assert(left[0].len() == cols / 2);
    assert(seq![left[i as int]][0].len() == cols / 2);
    assert(RT::silu_mul_kernel_repr(
        seq![left[i as int]], seq![right[i as int]],
    ) =~= seq![RT::silu_mul_kernel_repr(left, right)[i as int]]);
}

// ---------------------------------------------------------------------------
// Subrange (request-segment) batch invariance for the row-wise kernels.
//
// A system-level "batch element" is a request, which spans a contiguous row
// range [a, b), not a single row.  The row-wise kernels commute with subrange:
// running on a row segment equals slicing the segment out of the full output.
// This is the form that composes with `paged_attention`'s subrange invariance.
// Structural models use `reveal` plus sequence extensionality.
// ---------------------------------------------------------------------------

pub proof fn embed_subrange_invariance(ir: IntTensor1D, wr: Tensor2D, a: int, b: int)
    requires 0 <= a <= b <= ir.len(),
    ensures RT::embed_repr(ir.subrange(a, b), wr) == RT::embed_repr(ir, wr).subrange(a, b),
{
    let sub = ir.subrange(a, b);
    RT::lemma_embed_repr_shape(sub, wr);
    RT::lemma_embed_repr_shape(ir, wr);
    assert forall|j: int| 0 <= j < sub.len() implies
        #[trigger] RT::embed_repr(sub, wr)[j]
            == RT::embed_repr(ir, wr)[a + j]
    by {
        embed_batch_invariance(sub, wr, j as nat);
        embed_batch_invariance(ir, wr, (a + j) as nat);
        assert(sub[j] == ir[a + j]);
        assert(RT::embed_repr(seq![sub[j]], wr)
            == RT::embed_repr(seq![ir[a + j]], wr));
        assert(RT::embed_repr(seq![sub[j]], wr)
            == seq![RT::embed_repr(sub, wr)[j]]);
        assert(RT::embed_repr(seq![ir[a + j]], wr)
            == seq![RT::embed_repr(ir, wr)[a + j]]);
        assert(seq![RT::embed_repr(sub, wr)[j]]
            == seq![RT::embed_repr(ir, wr)[a + j]]);
        assert(seq![RT::embed_repr(sub, wr)[j]][0]
            == seq![RT::embed_repr(ir, wr)[a + j]][0]);
        assert(RT::embed_repr(sub, wr)[j]
            == RT::embed_repr(ir, wr)[a + j]);
    }
    assert(RT::embed_repr(sub, wr) =~= RT::embed_repr(ir, wr).subrange(a, b));
}

pub proof fn linear_subrange_invariance(xr: Tensor2D, wr: Tensor2D, a: int, b: int)
    requires 0 <= a <= b <= xr.len(),
    ensures RT::linear_repr(xr.subrange(a, b), wr) == RT::linear_repr(xr, wr).subrange(a, b),
{
    let sub = xr.subrange(a, b);
    RT::lemma_linear_repr_shape(sub, wr);
    RT::lemma_linear_repr_shape(xr, wr);
    assert forall|j: int| 0 <= j < sub.len() implies
        #[trigger] RT::linear_repr(sub, wr)[j]
            == RT::linear_repr(xr, wr)[a + j]
    by {
        linear_batch_invariance(sub, wr, j as nat);
        linear_batch_invariance(xr, wr, (a + j) as nat);
        assert(sub[j] == xr[a + j]);
        assert(RT::linear_repr(seq![sub[j]], wr)
            == RT::linear_repr(seq![xr[a + j]], wr));
        assert(RT::linear_repr(seq![sub[j]], wr)
            == seq![RT::linear_repr(sub, wr)[j]]);
        assert(RT::linear_repr(seq![xr[a + j]], wr)
            == seq![RT::linear_repr(xr, wr)[a + j]]);
        assert(seq![RT::linear_repr(sub, wr)[j]]
            == seq![RT::linear_repr(xr, wr)[a + j]]);
        assert(seq![RT::linear_repr(sub, wr)[j]][0]
            == seq![RT::linear_repr(xr, wr)[a + j]][0]);
        assert(RT::linear_repr(sub, wr)[j]
            == RT::linear_repr(xr, wr)[a + j]);
    }
    assert(RT::linear_repr(sub, wr) =~= RT::linear_repr(xr, wr).subrange(a, b));
}

pub proof fn silu_and_mul_subrange_invariance(xr: Tensor2D, a: int, b: int)
    requires
        0 <= a <= b <= xr.len(),
        TS::rectangular(xr),
    ensures RT::silu_and_mul_repr(xr.subrange(a, b)) == RT::silu_and_mul_repr(xr).subrange(a, b),
{
    let sub = xr.subrange(a, b);
    let cols = choose|cols: nat| TS::tensor2d_shape(xr, xr.len(), cols);
    assert(TS::tensor2d_shape(xr, xr.len(), cols));
    TS::lemma_tensor2d_shape_subrange(xr, xr.len(), cols, a, b);
    assert(TS::rectangular(sub));
    RT::lemma_silu_and_mul_repr_shape(sub);
    RT::lemma_silu_and_mul_repr_shape(xr);
    assert forall|j: int| 0 <= j < sub.len() implies
        #[trigger] RT::silu_and_mul_repr(sub)[j]
            == RT::silu_and_mul_repr(xr)[a + j]
    by {
        silu_and_mul_batch_invariance(sub, j as nat);
        silu_and_mul_batch_invariance(xr, (a + j) as nat);
        assert(sub[j] == xr[a + j]);
        assert(RT::silu_and_mul_repr(seq![sub[j]])
            == RT::silu_and_mul_repr(seq![xr[a + j]]));
        assert(RT::silu_and_mul_repr(seq![sub[j]])
            == seq![RT::silu_and_mul_repr(sub)[j]]);
        assert(RT::silu_and_mul_repr(seq![xr[a + j]])
            == seq![RT::silu_and_mul_repr(xr)[a + j]]);
        assert(seq![RT::silu_and_mul_repr(sub)[j]]
            == seq![RT::silu_and_mul_repr(xr)[a + j]]);
        assert(seq![RT::silu_and_mul_repr(sub)[j]][0]
            == seq![RT::silu_and_mul_repr(xr)[a + j]][0]);
        assert(RT::silu_and_mul_repr(sub)[j]
            == RT::silu_and_mul_repr(xr)[a + j]);
    }
    assert(RT::silu_and_mul_repr(sub) =~= RT::silu_and_mul_repr(xr).subrange(a, b));
}

pub proof fn rms_norm_subrange_invariance(
    xr: Tensor2D,
    wr: Tensor1D,
    epsilon: FloatParameterBits,
    a: int,
    b: int,
)
    requires 0 <= a <= b <= xr.len(),
    ensures RT::rms_norm_repr(xr.subrange(a, b), wr, epsilon)
        == RT::rms_norm_repr(xr, wr, epsilon).subrange(a, b),
{
    let sub = xr.subrange(a, b);
    RT::lemma_rms_norm_repr_len(sub, wr, epsilon);
    RT::lemma_rms_norm_repr_len(xr, wr, epsilon);
    assert forall|j: int| 0 <= j < sub.len() implies
        #[trigger] RT::rms_norm_repr(sub, wr, epsilon)[j]
            == RT::rms_norm_repr(xr, wr, epsilon)[a + j]
    by {
        rms_norm_batch_invariance(sub, wr, epsilon, j as nat);
        rms_norm_batch_invariance(xr, wr, epsilon, (a + j) as nat);
        assert(sub[j] == xr[a + j]);
        assert(RT::rms_norm_repr(seq![sub[j]], wr, epsilon)
            == RT::rms_norm_repr(seq![xr[a + j]], wr, epsilon));
        assert(RT::rms_norm_repr(seq![sub[j]], wr, epsilon)
            == seq![RT::rms_norm_repr(sub, wr, epsilon)[j]]);
        assert(RT::rms_norm_repr(seq![xr[a + j]], wr, epsilon)
            == seq![RT::rms_norm_repr(xr, wr, epsilon)[a + j]]);
        assert(seq![RT::rms_norm_repr(sub, wr, epsilon)[j]]
            == seq![RT::rms_norm_repr(xr, wr, epsilon)[a + j]]);
        assert(seq![RT::rms_norm_repr(sub, wr, epsilon)[j]][0]
            == seq![RT::rms_norm_repr(xr, wr, epsilon)[a + j]][0]);
        assert(RT::rms_norm_repr(sub, wr, epsilon)[j]
            == RT::rms_norm_repr(xr, wr, epsilon)[a + j]);
    }
    assert(RT::rms_norm_repr(sub, wr, epsilon)
        =~= RT::rms_norm_repr(xr, wr, epsilon).subrange(a, b));
}

pub proof fn view_as_kv_subrange_invariance(vr: Tensor2D, a: int, b: int)
    requires 0 <= a <= b <= vr.len(),
    ensures RT::view_as_kv_repr(vr.subrange(a, b)) == RT::view_as_kv_repr(vr).subrange(a, b),
{
    reveal(RT::view_as_kv_repr);
    assert(RT::view_as_kv_repr(vr.subrange(a, b)) =~= RT::view_as_kv_repr(vr).subrange(a, b));
}

pub proof fn qk_norm_subrange_invariance(
    qr: Tensor2D,
    kr: Tensor2D,
    qnr: Tensor1D,
    knr: Tensor1D,
    epsilon: FloatParameterBits,
    a: int,
    b: int,
)
    requires
        0 <= a <= b <= qr.len(),
        qr.len() == kr.len(),
        TS::rectangular(qr),
        TS::rectangular(kr),
    ensures
        RT::qk_norm_repr(
            qr.subrange(a, b), kr.subrange(a, b), qnr, knr, epsilon,
        ).0 == RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).0.subrange(a, b),
        RT::qk_norm_repr(
            qr.subrange(a, b), kr.subrange(a, b), qnr, knr, epsilon,
        ).1 == RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).1.subrange(a, b),
{
    let qs = qr.subrange(a, b);
    let ks = kr.subrange(a, b);
    let q_cols = choose|cols: nat| TS::tensor2d_shape(qr, qr.len(), cols);
    let k_cols = choose|cols: nat| TS::tensor2d_shape(kr, kr.len(), cols);
    assert(TS::tensor2d_shape(qr, qr.len(), q_cols));
    assert(TS::tensor2d_shape(kr, kr.len(), k_cols));
    TS::lemma_tensor2d_shape_subrange(qr, qr.len(), q_cols, a, b);
    TS::lemma_tensor2d_shape_subrange(kr, kr.len(), k_cols, a, b);
    assert(TS::rectangular(qs));
    assert(TS::rectangular(ks));
    RT::lemma_qk_norm_repr_len(
        qs, ks, qnr, knr, epsilon,
    );
    RT::lemma_qk_norm_repr_len(
        qr, kr, qnr, knr, epsilon,
    );
    assert forall|j: int| 0 <= j < qs.len() implies
        #[trigger] RT::qk_norm_repr(qs, ks, qnr, knr, epsilon).0[j]
            == RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).0[a + j]
    by {
        qk_norm_batch_invariance(qs, ks, qnr, knr, epsilon, j as nat);
        qk_norm_batch_invariance(qr, kr, qnr, knr, epsilon, (a + j) as nat);
        assert(qs[j] == qr[a + j]);
        assert(ks[j] == kr[a + j]);
        assert(RT::qk_norm_repr(seq![qs[j]], seq![ks[j]], qnr, knr, epsilon).0
            == RT::qk_norm_repr(
                seq![qr[a + j]], seq![kr[a + j]], qnr, knr, epsilon,
            ).0);
        assert(RT::qk_norm_repr(seq![qs[j]], seq![ks[j]], qnr, knr, epsilon).0
            == seq![RT::qk_norm_repr(qs, ks, qnr, knr, epsilon).0[j]]);
        assert(RT::qk_norm_repr(
            seq![qr[a + j]], seq![kr[a + j]], qnr, knr, epsilon,
        ).0 == seq![RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).0[a + j]]);
        assert(seq![RT::qk_norm_repr(qs, ks, qnr, knr, epsilon).0[j]][0]
            == seq![RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).0[a + j]][0]);
    }
    assert forall|j: int| 0 <= j < qs.len() implies
        #[trigger] RT::qk_norm_repr(qs, ks, qnr, knr, epsilon).1[j]
            == RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).1[a + j]
    by {
        qk_norm_batch_invariance(qs, ks, qnr, knr, epsilon, j as nat);
        qk_norm_batch_invariance(qr, kr, qnr, knr, epsilon, (a + j) as nat);
        assert(qs[j] == qr[a + j]);
        assert(ks[j] == kr[a + j]);
        assert(RT::qk_norm_repr(seq![qs[j]], seq![ks[j]], qnr, knr, epsilon).1
            == RT::qk_norm_repr(
                seq![qr[a + j]], seq![kr[a + j]], qnr, knr, epsilon,
            ).1);
        assert(RT::qk_norm_repr(seq![qs[j]], seq![ks[j]], qnr, knr, epsilon).1
            == seq![RT::qk_norm_repr(qs, ks, qnr, knr, epsilon).1[j]]);
        assert(RT::qk_norm_repr(
            seq![qr[a + j]], seq![kr[a + j]], qnr, knr, epsilon,
        ).1 == seq![RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).1[a + j]]);
        assert(seq![RT::qk_norm_repr(qs, ks, qnr, knr, epsilon).1[j]][0]
            == seq![RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).1[a + j]][0]);
    }
    assert(RT::qk_norm_repr(qs, ks, qnr, knr, epsilon).0
        =~= RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).0.subrange(a, b));
    assert(RT::qk_norm_repr(qs, ks, qnr, knr, epsilon).1
        =~= RT::qk_norm_repr(qr, kr, qnr, knr, epsilon).1.subrange(a, b));
}

// Closed optional Q/K-normalization layer.  The disabled branch is identity;
// the RMS branch delegates to the existing row-local kernel proof.
pub proof fn apply_qk_norm_subrange_invariance(
    qr: Tensor2D,
    kr: Tensor2D,
    weights: QkNormWeightsRepr,
    epsilon: FloatParameterBits,
    a: int,
    b: int,
)
    requires
        qr.len() == kr.len(),
        0 <= a <= b <= qr.len(),
        TS::rectangular(qr),
        TS::rectangular(kr),
    ensures
        RT::apply_qk_norm_repr(
            qr.subrange(a, b), kr.subrange(a, b), weights, epsilon,
        ).0 == RT::apply_qk_norm_repr(qr, kr, weights, epsilon).0.subrange(a, b),
        RT::apply_qk_norm_repr(
            qr.subrange(a, b), kr.subrange(a, b), weights, epsilon,
        ).1 == RT::apply_qk_norm_repr(qr, kr, weights, epsilon).1.subrange(a, b),
{
    RT::lemma_apply_qk_norm_repr_shape(
        qr, kr, weights, epsilon,
    );
    reveal(RT::apply_qk_norm_repr);
    match weights {
        QkNormWeightsRepr::Disabled => {},
        QkNormWeightsRepr::RmsNorm { q_weight, k_weight } => {
            qk_norm_subrange_invariance(
                qr, kr, q_weight, k_weight, epsilon, a, b,
            );
        },
    }
}

pub proof fn rotary_embed_subrange_invariance(
    geometry: AttentionGeometryRepr,
    rotary: RotaryConfigRepr,
    pir: IntTensor1D, qr: Tensor2D, kr: Tensor2D, a: int, b: int,
)
    requires 0 <= a <= b <= qr.len(), pir.len() == qr.len(), qr.len() == kr.len(),
    ensures
        DLP::rotary_embed_repr(
            geometry, rotary, pir.subrange(a, b), qr.subrange(a, b),
            kr.subrange(a, b),
        ).0 == DLP::rotary_embed_repr(
            geometry, rotary, pir, qr, kr,
        ).0.subrange(a, b),
        DLP::rotary_embed_repr(
            geometry, rotary, pir.subrange(a, b), qr.subrange(a, b),
            kr.subrange(a, b),
        ).1 == DLP::rotary_embed_repr(
            geometry, rotary, pir, qr, kr,
        ).1.subrange(a, b),
{
    let ps = pir.subrange(a, b);
    let qs = qr.subrange(a, b);
    let ks = kr.subrange(a, b);
    DLP::lemma_rotary_embed_repr_shape(geometry, rotary, ps, qs, ks);
    DLP::lemma_rotary_embed_repr_shape(geometry, rotary, pir, qr, kr);
    assert forall|j: int| 0 <= j < qs.len() implies
        #[trigger] DLP::rotary_embed_repr(geometry, rotary, ps, qs, ks).0[j]
            == DLP::rotary_embed_repr(geometry, rotary, pir, qr, kr).0[a + j]
    by {
        rotary_embed_batch_invariance(
            geometry, rotary, ps, qs, ks, j as nat,
        );
        rotary_embed_batch_invariance(
            geometry, rotary, pir, qr, kr, (a + j) as nat,
        );
        assert(ps[j] == pir[a + j]);
        assert(qs[j] == qr[a + j]);
        assert(ks[j] == kr[a + j]);
        assert(DLP::rotary_embed_repr(
            geometry, rotary, seq![ps[j]], seq![qs[j]], seq![ks[j]],
        ).0 == DLP::rotary_embed_repr(
                geometry, rotary, seq![pir[a + j]], seq![qr[a + j]],
                seq![kr[a + j]],
            ).0);
        assert(DLP::rotary_embed_repr(
            geometry, rotary, seq![ps[j]], seq![qs[j]], seq![ks[j]],
        ).0 == seq![DLP::rotary_embed_repr(
            geometry, rotary, ps, qs, ks,
        ).0[j]]);
        assert(DLP::rotary_embed_repr(
            geometry, rotary, seq![pir[a + j]], seq![qr[a + j]],
            seq![kr[a + j]],
        ).0 == seq![DLP::rotary_embed_repr(
            geometry, rotary, pir, qr, kr,
        ).0[a + j]]);
        assert(seq![DLP::rotary_embed_repr(
            geometry, rotary, ps, qs, ks,
        ).0[j]][0] == seq![DLP::rotary_embed_repr(
            geometry, rotary, pir, qr, kr,
        ).0[a + j]][0]);
    }
    assert forall|j: int| 0 <= j < qs.len() implies
        #[trigger] DLP::rotary_embed_repr(geometry, rotary, ps, qs, ks).1[j]
            == DLP::rotary_embed_repr(geometry, rotary, pir, qr, kr).1[a + j]
    by {
        rotary_embed_batch_invariance(
            geometry, rotary, ps, qs, ks, j as nat,
        );
        rotary_embed_batch_invariance(
            geometry, rotary, pir, qr, kr, (a + j) as nat,
        );
        assert(ps[j] == pir[a + j]);
        assert(qs[j] == qr[a + j]);
        assert(ks[j] == kr[a + j]);
        assert(DLP::rotary_embed_repr(
            geometry, rotary, seq![ps[j]], seq![qs[j]], seq![ks[j]],
        ).1 == DLP::rotary_embed_repr(
                geometry, rotary, seq![pir[a + j]], seq![qr[a + j]],
                seq![kr[a + j]],
            ).1);
        assert(DLP::rotary_embed_repr(
            geometry, rotary, seq![ps[j]], seq![qs[j]], seq![ks[j]],
        ).1 == seq![DLP::rotary_embed_repr(
            geometry, rotary, ps, qs, ks,
        ).1[j]]);
        assert(DLP::rotary_embed_repr(
            geometry, rotary, seq![pir[a + j]], seq![qr[a + j]],
            seq![kr[a + j]],
        ).1 == seq![DLP::rotary_embed_repr(
            geometry, rotary, pir, qr, kr,
        ).1[a + j]]);
        assert(seq![DLP::rotary_embed_repr(
            geometry, rotary, ps, qs, ks,
        ).1[j]][0] == seq![DLP::rotary_embed_repr(
            geometry, rotary, pir, qr, kr,
        ).1[a + j]][0]);
    }
    assert(DLP::rotary_embed_repr(geometry, rotary, ps, qs, ks).0
        =~= DLP::rotary_embed_repr(
            geometry, rotary, pir, qr, kr,
        ).0.subrange(a, b));
    assert(DLP::rotary_embed_repr(geometry, rotary, ps, qs, ks).1
        =~= DLP::rotary_embed_repr(
            geometry, rotary, pir, qr, kr,
        ).1.subrange(a, b));
}

pub proof fn add_rms_norm_subrange_invariance(
    xr: Tensor2D,
    rr: Tensor2D,
    wr: Tensor1D,
    epsilon: FloatParameterBits,
    a: int,
    b: int,
)
    requires
        0 <= a <= b <= xr.len(),
        xr.len() == rr.len(),
    ensures
        RT::add_rms_norm_repr(
            xr.subrange(a, b), rr.subrange(a, b), wr, epsilon,
        ).0 == RT::add_rms_norm_repr(xr, rr, wr, epsilon).0.subrange(a, b),
        RT::add_rms_norm_repr(
            xr.subrange(a, b), rr.subrange(a, b), wr, epsilon,
        ).1 == RT::add_rms_norm_repr(xr, rr, wr, epsilon).1.subrange(a, b),
{
    let xs = xr.subrange(a, b);
    let rs = rr.subrange(a, b);
    RT::lemma_add_rms_norm_repr_len(
        xs, rs, wr, epsilon,
    );
    RT::lemma_add_rms_norm_repr_len(
        xr, rr, wr, epsilon,
    );
    assert forall|j: int| 0 <= j < xs.len() implies
        #[trigger] RT::add_rms_norm_repr(xs, rs, wr, epsilon).0[j]
            == RT::add_rms_norm_repr(xr, rr, wr, epsilon).0[a + j]
    by {
        add_rms_norm_batch_invariance(xs, rs, wr, epsilon, j as nat);
        add_rms_norm_batch_invariance(xr, rr, wr, epsilon, (a + j) as nat);
        assert(xs[j] == xr[a + j]);
        assert(rs[j] == rr[a + j]);
        assert(RT::add_rms_norm_repr(seq![xs[j]], seq![rs[j]], wr, epsilon).0
            == RT::add_rms_norm_repr(
                seq![xr[a + j]], seq![rr[a + j]], wr, epsilon,
            ).0);
        assert(RT::add_rms_norm_repr(seq![xs[j]], seq![rs[j]], wr, epsilon).0
            == seq![RT::add_rms_norm_repr(xs, rs, wr, epsilon).0[j]]);
        assert(RT::add_rms_norm_repr(
            seq![xr[a + j]], seq![rr[a + j]], wr, epsilon,
        ).0 == seq![RT::add_rms_norm_repr(xr, rr, wr, epsilon).0[a + j]]);
        assert(seq![RT::add_rms_norm_repr(xs, rs, wr, epsilon).0[j]][0]
            == seq![RT::add_rms_norm_repr(xr, rr, wr, epsilon).0[a + j]][0]);
    }
    assert forall|j: int| 0 <= j < xs.len() implies
        #[trigger] RT::add_rms_norm_repr(xs, rs, wr, epsilon).1[j]
            == RT::add_rms_norm_repr(xr, rr, wr, epsilon).1[a + j]
    by {
        add_rms_norm_batch_invariance(xs, rs, wr, epsilon, j as nat);
        add_rms_norm_batch_invariance(xr, rr, wr, epsilon, (a + j) as nat);
        assert(xs[j] == xr[a + j]);
        assert(rs[j] == rr[a + j]);
        assert(RT::add_rms_norm_repr(seq![xs[j]], seq![rs[j]], wr, epsilon).1
            == RT::add_rms_norm_repr(
                seq![xr[a + j]], seq![rr[a + j]], wr, epsilon,
            ).1);
        assert(RT::add_rms_norm_repr(seq![xs[j]], seq![rs[j]], wr, epsilon).1
            == seq![RT::add_rms_norm_repr(xs, rs, wr, epsilon).1[j]]);
        assert(RT::add_rms_norm_repr(
            seq![xr[a + j]], seq![rr[a + j]], wr, epsilon,
        ).1 == seq![RT::add_rms_norm_repr(xr, rr, wr, epsilon).1[a + j]]);
        assert(seq![RT::add_rms_norm_repr(xs, rs, wr, epsilon).1[j]][0]
            == seq![RT::add_rms_norm_repr(xr, rr, wr, epsilon).1[a + j]][0]);
    }
    assert(RT::add_rms_norm_repr(xs, rs, wr, epsilon).0
        =~= RT::add_rms_norm_repr(xr, rr, wr, epsilon).0.subrange(a, b));
    assert(RT::add_rms_norm_repr(xs, rs, wr, epsilon).1
        =~= RT::add_rms_norm_repr(xr, rr, wr, epsilon).1.subrange(a, b));
}

// ---------------------------------------------------------------------------
// PagedAttention: batch-element independence.
//
// For batch element `i`, output rows in [cu_q[i] .. cu_q[i+1]] equal the
// output of running PagedAttention on just element i's Q rows, with element
// i's block-table row, rebased cu_seqlens, and the same KV cache reprs.
//
// Checked projection of the common mapped operation. The runtime wrapper
// binds this operation to the exact raw kernel theorem under the complete
// generated launch and numeric requirements.
// ---------------------------------------------------------------------------
pub broadcast proof fn paged_attention_batch_invariance(
    qr: Tensor2D,
    k_cache_repr: KVCacheLayerRepr, v_cache_repr: KVCacheLayerRepr,
    cu_q_repr: Seq<int>, cu_k_repr: Seq<int>,
    max_seqlen_q: nat, max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
    i: nat,
    parameters: AttentionParametersRepr,
)
    requires
        RT::paged_attention_launch_ready(
            qr.len(), k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
        i < bt_repr.len(),
    ensures
        #[trigger] RT::paged_attention_repr(qr, k_cache_repr, v_cache_repr,
                cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
                parameters,
                )
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == RT::paged_attention_repr(
            qr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]],
            parameters,
            ),
{
    reveal(RT::paged_attention_launch_ready);
    reveal(RT::paged_attention_repr);
    AO::lemma_batch_projection(qr, k_cache_repr, v_cache_repr,
        cu_q_repr, cu_k_repr, bt_repr, max_seqlen_q, max_seqlen_k,
        AttentionKind::Full, parameters, 0, i as int);
}

// ---------------------------------------------------------------------------
// PagedAttentionSingletonEquivalence — general single-batch equivalence
// for causal attention.  Two singleton-batch computations with potentially
// different (q_len, k_len, cache, block_table) produce the same output at
// rows j_a / j_b if:
//   - Same Q row (qr_a[j_a] == qr_b[j_b]),
//   - Same effective causal position k_len - q_len + j,
//   - Same K/V values in the causal window 0..effective_pos.
//
// The equality follows from the canonical raw row operation's checked logical
// prefix projection. Runtime finiteness remains explicit when binding actual
// launches to that operation; no numerical-correctness claim is made.
// ---------------------------------------------------------------------------
pub open spec fn paged_attention_singleton_repr(
    qr: Tensor2D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    q_len: nat,
    k_len: nat,
    bt_row: Seq<BlockId>,
    parameters: AttentionParametersRepr,
) -> Tensor2D {
    RT::paged_attention_repr(qr, k_cache, v_cache,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row],
        parameters,
        )
}

pub broadcast proof fn paged_attention_singleton_equivalence(
    qr_a: Tensor2D, qr_b: Tensor2D,
    k_cache_a: KVCacheLayerRepr, v_cache_a: KVCacheLayerRepr,
    k_cache_b: KVCacheLayerRepr, v_cache_b: KVCacheLayerRepr,
    q_len_a: nat, k_len_a: nat,
    q_len_b: nat, k_len_b: nat,
    bt_row_a: Seq<BlockId>, bt_row_b: Seq<BlockId>,
    j_a: nat, j_b: nat,
    parameters: AttentionParametersRepr,
)
    requires
        RT::paged_attention_numeric_domain(),
        qr_a.len() == q_len_a, q_len_a > 0,
        qr_b.len() == q_len_b, q_len_b > 0,
        j_a < q_len_a, j_b < q_len_b,
        q_len_a <= k_len_a, q_len_b <= k_len_b,
        blocks_needed_for(k_len_a) <= bt_row_a.len(),
        blocks_needed_for(k_len_b) <= bt_row_b.len(),
        qr_a[j_a as int] == qr_b[j_b as int],
        k_len_a as int - q_len_a as int + j_a as int
            == k_len_b as int - q_len_b as int + j_b as int,
        forall|pos: nat| #![auto]
            pos as int <= k_len_a as int - q_len_a as int + j_a as int ==>
                (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_a.len() &&
                (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_b.len() &&
                slot_in_cache(k_cache_a, block_table_slot(bt_row_a, pos)) &&
                slot_in_cache(v_cache_a, block_table_slot(bt_row_a, pos)) &&
                slot_in_cache(k_cache_b, block_table_slot(bt_row_b, pos)) &&
                slot_in_cache(v_cache_b, block_table_slot(bt_row_b, pos)) &&
                cache_at(k_cache_a, block_table_slot(bt_row_a, pos))
                    == cache_at(k_cache_b, block_table_slot(bt_row_b, pos)) &&
                cache_at(v_cache_a, block_table_slot(bt_row_a, pos))
                    == cache_at(v_cache_b, block_table_slot(bt_row_b, pos)),
    ensures
        #![trigger
            paged_attention_singleton_repr(
                qr_a, k_cache_a, v_cache_a, q_len_a, k_len_a, bt_row_a,
                parameters,
                )[j_a as int],
            paged_attention_singleton_repr(
                qr_b, k_cache_b, v_cache_b, q_len_b, k_len_b, bt_row_b,
                parameters,
                )[j_b as int]]
        paged_attention_singleton_repr(
            qr_a, k_cache_a, v_cache_a, q_len_a, k_len_a, bt_row_a,
            parameters,
            )[j_a as int]
        == paged_attention_singleton_repr(
            qr_b, k_cache_b, v_cache_b, q_len_b, k_len_b, bt_row_b,
            parameters,
            )[j_b as int],
{
    reveal(paged_attention_singleton_repr);
    reveal(RT::paged_attention_repr);
    AO::lemma_singleton_equivalence(
            qr_a, k_cache_a, v_cache_a, bt_row_a, k_len_a, j_a as int,
            qr_b, k_cache_b, v_cache_b, bt_row_b, k_len_b, j_b as int,
            AttentionKind::Full, parameters, 0,
        );
}

// Causal prefix stability for singleton full-prefill attention.  Extending a
// request from `prefix_len` to `full_len` cannot change the earlier attention
// rows when the earlier Q and logical K/V values agree.  This is the exact
// attention induction step needed by semantic cache fidelity; it is derived
// from the per-row causal window above, not assumed as a new kernel property.
pub proof fn paged_attention_singleton_prefix_invariance(
    q_full: Tensor2D,
    q_prefix: Tensor2D,
    k_full: KVCacheLayerRepr,
    v_full: KVCacheLayerRepr,
    k_prefix: KVCacheLayerRepr,
    v_prefix: KVCacheLayerRepr,
    bt_full: Seq<BlockId>,
    bt_prefix: Seq<BlockId>,
    full_len: nat,
    prefix_len: nat,
    parameters: AttentionParametersRepr,
)
    requires
        RT::paged_attention_numeric_domain(),
        q_full.len() == full_len,
        q_prefix.len() == prefix_len,
        0 < prefix_len <= full_len,
        q_full.subrange(0, prefix_len as int) == q_prefix,
        blocks_needed_for(full_len) <= bt_full.len(),
        blocks_needed_for(prefix_len) <= bt_prefix.len(),
        forall|pos: nat| #![auto] pos < prefix_len ==>
            (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_full.len()
            && (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_prefix.len()
            && slot_in_cache(k_full, block_table_slot(bt_full, pos))
            && slot_in_cache(v_full, block_table_slot(bt_full, pos))
            && slot_in_cache(k_prefix, block_table_slot(bt_prefix, pos))
            && slot_in_cache(v_prefix, block_table_slot(bt_prefix, pos))
            && cache_at(k_full, block_table_slot(bt_full, pos))
                == cache_at(k_prefix, block_table_slot(bt_prefix, pos))
            && cache_at(v_full, block_table_slot(bt_full, pos))
                == cache_at(v_prefix, block_table_slot(bt_prefix, pos)),
    ensures
        paged_attention_singleton_repr(
            q_full, k_full, v_full, full_len, full_len, bt_full,
            parameters,
        ).subrange(0, prefix_len as int)
        == paged_attention_singleton_repr(
            q_prefix, k_prefix, v_prefix, prefix_len, prefix_len, bt_prefix,
            parameters,
        ),
{
    let out_full = paged_attention_singleton_repr(
        q_full, k_full, v_full, full_len, full_len, bt_full,
        parameters,
    );
    let out_prefix = paged_attention_singleton_repr(
        q_prefix, k_prefix, v_prefix, prefix_len, prefix_len, bt_prefix,
        parameters,
    );
    RT::lemma_paged_attention_repr_shape(
        q_full, k_full, v_full,
        seq![0int, full_len as int], seq![0int, full_len as int],
        full_len, full_len, seq![bt_full],
        parameters,
    );
    RT::lemma_paged_attention_repr_shape(
        q_prefix, k_prefix, v_prefix,
        seq![0int, prefix_len as int], seq![0int, prefix_len as int],
        prefix_len, prefix_len, seq![bt_prefix],
        parameters,
    );
    assert(out_full.len() == full_len);
    assert(out_prefix.len() == prefix_len);
    assert forall|j: int| 0 <= j < prefix_len implies
        out_full[j] == out_prefix[j]
    by {
        assert(q_full[j] == q_prefix[j]);
        assert forall|pos: nat| #![auto] pos as int <= j implies
            (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_full.len()
            && (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_prefix.len()
            && slot_in_cache(k_full, block_table_slot(bt_full, pos))
            && slot_in_cache(v_full, block_table_slot(bt_full, pos))
            && slot_in_cache(k_prefix, block_table_slot(bt_prefix, pos))
            && slot_in_cache(v_prefix, block_table_slot(bt_prefix, pos))
            && cache_at(k_full, block_table_slot(bt_full, pos))
                == cache_at(k_prefix, block_table_slot(bt_prefix, pos))
            && cache_at(v_full, block_table_slot(bt_full, pos))
                == cache_at(v_prefix, block_table_slot(bt_prefix, pos))
        by {
            assert(pos < prefix_len);
        }
        paged_attention_singleton_equivalence(
            q_full, q_prefix,
            k_full, v_full, k_prefix, v_prefix,
            full_len, full_len, prefix_len, prefix_len,
            bt_full, bt_prefix, j as nat, j as nat,
            parameters,
        );
    }
    assert(out_full.subrange(0, prefix_len as int) =~= out_prefix);
}

// ---------------------------------------------------------------------------
// Derived (provable) lemmas — these have actual bodies, not `admit()`.
// ---------------------------------------------------------------------------

// `store_kv_cache_repr_preserves_unwritten_slots` lives in `tensor_runtime`
// (same module as `store_kv_cache_repr`'s `closed spec fn` definition) so
// the proof can `reveal` the body.  Re-exported here as `RT::...`.

// PagedAttention physical relocation: singleton-form output depends only
// on the logical K/V values at positions 0..k_len via the block table.
// Two (cache, block_table) pairs producing the same K/V at every logical
// position give identical outputs.  Proved by per-row PagedAttentionSingletonEquivalence
// + sequence extensionality.
pub proof fn paged_attention_physical_relocation(
    qr: Tensor2D,
    k_cache_a: KVCacheLayerRepr, v_cache_a: KVCacheLayerRepr,
    k_cache_b: KVCacheLayerRepr, v_cache_b: KVCacheLayerRepr,
    q_len: nat, k_len: nat,
    bt_row_a: Seq<BlockId>, bt_row_b: Seq<BlockId>,
    parameters: AttentionParametersRepr,
)
    requires
        RT::paged_attention_numeric_domain(),
        qr.len() == q_len, q_len > 0,
        q_len <= k_len,
        blocks_needed_for(k_len) <= bt_row_a.len(),
        blocks_needed_for(k_len) <= bt_row_b.len(),
        forall|pos: nat| #![auto] pos < k_len ==>
            (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_a.len() &&
            (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_b.len() &&
            slot_in_cache(k_cache_a, block_table_slot(bt_row_a, pos)) &&
            slot_in_cache(v_cache_a, block_table_slot(bt_row_a, pos)) &&
            slot_in_cache(k_cache_b, block_table_slot(bt_row_b, pos)) &&
            slot_in_cache(v_cache_b, block_table_slot(bt_row_b, pos)) &&
            cache_at(k_cache_a, block_table_slot(bt_row_a, pos))
                == cache_at(k_cache_b, block_table_slot(bt_row_b, pos)) &&
            cache_at(v_cache_a, block_table_slot(bt_row_a, pos))
                == cache_at(v_cache_b, block_table_slot(bt_row_b, pos)),
    ensures
        RT::paged_attention_repr(qr, k_cache_a, v_cache_a,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a],
            parameters,
            )
        == RT::paged_attention_repr(qr, k_cache_b, v_cache_b,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b],
            parameters,
            ),
{
    let res_a = RT::paged_attention_repr(qr, k_cache_a, v_cache_a,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row_a],
        parameters,
        );
    let res_b = RT::paged_attention_repr(qr, k_cache_b, v_cache_b,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row_b],
        parameters,
        );

    // Both singletons have output length q_len (from lemma_paged_attention_repr_shape).
    RT::lemma_paged_attention_repr_shape(qr, k_cache_a, v_cache_a,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row_a],
        parameters,
        );
    RT::lemma_paged_attention_repr_shape(qr, k_cache_b, v_cache_b,
        seq![0int, q_len as int], seq![0int, k_len as int],
        q_len, k_len, seq![bt_row_b],
        parameters,
        );

    // Per-row equality via singleton-equivalence at j_a = j_b = j.
    assert forall|j: int| 0 <= j < q_len as int implies
        res_a[j] == res_b[j]
    by {
        // effective_pos = k_len - q_len + j  (same on both sides since q_len, k_len match).
        // K/V matches at positions 0..effective_pos hold by precondition (0..k_len-1).
        paged_attention_singleton_equivalence(
            qr, qr,
            k_cache_a, v_cache_a, k_cache_b, v_cache_b,
            q_len, k_len, q_len, k_len,
            bt_row_a, bt_row_b,
            j as nat, j as nat,
            parameters,
            );
    }

    // Sequence extensionality.
    assert(res_a =~= res_b);
}

} // verus!
