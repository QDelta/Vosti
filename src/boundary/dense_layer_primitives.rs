//! Architecture-neutral primitive semantics for dense decoder layers.
//!
//! These declarations name the residual semantic obligations between the
//! model composition and its kernels/adapters. They do not provide executable
//! contracts or make any family engine-reachable. Keeping every uninterpreted
//! declaration in `boundary` makes the trust inventory explicit, while the
//! family chooses the concrete scalar parameters and attention policy.

use crate::model_config::{AttentionKind, FloatParameterBits, ModelArchitecture};
use crate::boundary::model_families::gemma3::config::GEMMA3_RMS_NORM_EPSILON_F64_BITS;
use crate::boundary::model_families::llama3::config::LLAMA3_RMS_NORM_EPSILON_F64_BITS;
use crate::boundary::model_families::qwen3::config::QWEN3_RMS_NORM_EPSILON_F64_BITS;
use crate::boundary::tensor_runtime as TR;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::proof::tensor::shape as TS;
#[cfg(verus_only)]
use crate::boundary::scalar::{float_parameter_scalar_repr, positive_float_parameter_valid};
use vstd::prelude::*;

#[cfg(not(verus_only))]
use pyo3::prelude::*;

verus! {

// Exact BF16-downcast sqrt(hidden_size) used by a scaled embedding layer. The
// generated relational certificate retains this shared scalar explicitly; its
// physical value remains part of the runtime-to-Scalar bridge.
pub uninterp spec fn sqrt_hidden_size_scale_repr(hidden_size: nat) -> Scalar;

// Family-local admission fixes the executable parameter. This projection is
// deliberately exhaustive rather than treating the equal Qwen/Gemma values
// as one family-wide constant.
pub open spec fn rms_norm_epsilon_repr(
    architecture: ModelArchitecture,
) -> Scalar
    recommends architecture != ModelArchitecture::Gemma4Text,
{
    match architecture {
        // Outside this fixed-epsilon projection: direct four-norm models
        // supply their exact epsilon through the immutable runtime policy.
        ModelArchitecture::Gemma4Text => arbitrary(),
        ModelArchitecture::Qwen3 =>
            float_parameter_scalar_repr(FloatParameterBits {
                bits: QWEN3_RMS_NORM_EPSILON_F64_BITS,
            }),
        ModelArchitecture::Llama3 =>
            float_parameter_scalar_repr(FloatParameterBits {
                bits: LLAMA3_RMS_NORM_EPSILON_F64_BITS,
            }),
        ModelArchitecture::Gemma3Text =>
            float_parameter_scalar_repr(FloatParameterBits {
                bits: GEMMA3_RMS_NORM_EPSILON_F64_BITS,
            }),
    }
}

// Q and K use the same neutral head-normalization layer. Their different head
// counts are represented by the row shapes rather than separate semantic atoms.
#[verifier::opaque]
pub open spec fn qk_norm_row_repr(
    row: Tensor1D,
    weight: Tensor1D,
) -> Tensor1D {
    crate::boundary::head_normalization_operator::row_output(row, weight,
        float_parameter_scalar_repr(unit_offset_norm_epsilon_repr()), true)
}

// The attention partition is recovered from explicit head width and the Q/K
// projection output widths. It deliberately does not depend on the presence
// of Q/K normalization weights: a decoder with disabled Q/K normalization has
// exactly the same attention geometry obligations.
pub open spec fn layer_attention_geometry_repr(
    weights: LayerWeightsRepr,
) -> AttentionGeometryRepr {
    AttentionGeometryRepr {
        num_attention_heads: weights.q_proj.len() / weights.head_dim,
        num_key_value_heads: weights.k_proj.len() / weights.head_dim,
        head_dim: weights.head_dim,
    }
}

#[verifier::opaque]
pub open spec fn layer_attention_parameters_repr(
    weights: LayerWeightsRepr,
    scale: AttentionScaleRepr,
) -> AttentionParametersRepr {
    AttentionParametersRepr { geometry: layer_attention_geometry_repr(weights), scale }
}

pub proof fn lemma_layer_attention_geometry_from_shapes(
    weights: LayerWeightsRepr,
    geometry: DenseGeometryRepr,
)
    requires
        geometry.head_dim > 0,
        TS::tensor2d_shape(
            weights.q_proj,
            geometry.num_attention_heads * geometry.head_dim,
            geometry.hidden_size,
        ),
        TS::tensor2d_shape(
            weights.k_proj,
            geometry.num_key_value_heads * geometry.head_dim,
            geometry.hidden_size,
        ),
        weights.head_dim == geometry.head_dim,
    ensures
        layer_attention_geometry_repr(weights)
            == attention_geometry_repr(geometry),
{
    reveal(layer_attention_geometry_repr);
    reveal(TS::tensor2d_shape);
    vstd::arithmetic::div_mod::lemma_div_by_multiple(
        geometry.num_attention_heads as int,
        geometry.head_dim as int,
    );
    vstd::arithmetic::div_mod::lemma_div_by_multiple(
        geometry.num_key_value_heads as int,
        geometry.head_dim as int,
    );
    assert(weights.q_proj.len()
        == geometry.num_attention_heads * geometry.head_dim);
    assert(weights.k_proj.len()
        == geometry.num_key_value_heads * geometry.head_dim);
    assert(weights.q_proj.len() / weights.head_dim
        == geometry.num_attention_heads);
    assert(weights.k_proj.len() / weights.head_dim
        == geometry.num_key_value_heads);
    assert(layer_attention_geometry_repr(weights).num_attention_heads
        == attention_geometry_repr(geometry).num_attention_heads);
    assert(layer_attention_geometry_repr(weights).num_key_value_heads
        == attention_geometry_repr(geometry).num_key_value_heads);
    assert(layer_attention_geometry_repr(weights).head_dim
        == attention_geometry_repr(geometry).head_dim);
}

// @kernel-bridge-begin boundary::dense_layer_primitives::rotary_embed
pub open spec fn rotary_embed_repr(
    geometry: AttentionGeometryRepr,
    rotary: RotaryConfigRepr,
    positions: IntTensor1D,
    q: Tensor2D,
    k: Tensor2D,
) -> (Tensor2D, Tensor2D) {
    TR::rotary_embed_repr(
        positions,
        q,
        k,
        geometry.num_attention_heads,
        geometry.num_key_value_heads,
        geometry.head_dim,
        rotary,
    )
}

pub broadcast proof fn lemma_rotary_embed_repr_shape(
    geometry: AttentionGeometryRepr,
    rotary: RotaryConfigRepr,
    positions: IntTensor1D,
    q: Tensor2D,
    k: Tensor2D,
)
    requires positions.len() == q.len(), q.len() == k.len(),
    ensures
        (#[trigger] rotary_embed_repr(
            geometry, rotary, positions, q, k,
        )).0.len() == q.len(),
        rotary_embed_repr(geometry, rotary, positions, q, k).1.len()
            == q.len(),
{
    reveal(rotary_embed_repr);
    TR::lemma_rotary_embed_repr_shape(
        positions,
        q,
        k,
        geometry.num_attention_heads,
        geometry.num_key_value_heads,
        geometry.head_dim,
        rotary,
    );
}

pub fn rotary_embed(
    runtime: &TR::ModelFamilyRuntime,
    positions: &TR::Tensor,
    q: &TR::Tensor,
    k: &TR::Tensor,
    Tracked(pp): Tracked<&TR::TensorPerm>,
    Tracked(qp): Tracked<&TR::TensorPerm>,
    Tracked(kp): Tracked<&TR::TensorPerm>,
    Ghost(pir): Ghost<IntTensor1D>,
    Ghost(qr): Ghost<Tensor2D>,
    Ghost(kr): Ghost<Tensor2D>,
    Ghost(geometry): Ghost<AttentionGeometryRepr>,
    Ghost(rotary): Ghost<RotaryConfigRepr>,
    Ghost(scope): Ghost<Set<TR::TensorId>>,
) -> (out: ((TR::Tensor, TR::Tensor),
            (Tracked<TR::TensorPerm>, Tracked<TR::TensorPerm>)))
    requires
        TR::dense_swiglu_runtime_rotary_matches(
            runtime, geometry, rotary,
        ),
        TR::int_tensor_repr_1d(*pp, *positions, pir),
        TR::tensor_repr_2d(*qp, *q, qr),
        TR::tensor_repr_2d(*kp, *k, kr),
        pir.len() == qr.len(),
        qr.len() == kr.len(),
    ensures ({
        let ((rq, rk), (rqp, rkp)) = out;
        let rotated = rotary_embed_repr(
            geometry, rotary, pir, qr, kr,
        );
        &&& rq.id() != rk.id()
        &&& !scope.contains(rq.id())
        &&& !scope.contains(rk.id())
        &&& TR::tensor_repr_2d(rqp@, rq, rotated.0)
        &&& TR::tensor_repr_2d(rkp@, rk, rotated.1)
    }),
{
    let out = rotary_embed_raw(runtime, positions, q, k, Tracked(pp), Tracked(qp), Tracked(kp),
        Ghost(pir), Ghost(qr), Ghost(kr), Ghost(geometry), Ghost(rotary), Ghost(scope));
    proof {
        TR::checked_rotary_component_binding(pir, qr, geometry.num_attention_heads, geometry.head_dim, rotary);
        TR::checked_rotary_component_binding(pir, kr, geometry.num_key_value_heads, geometry.head_dim, rotary);
        reveal(TR::rotary_embed_repr);
    }
    out
}

#[verifier::external_body]
fn rotary_embed_raw(
    runtime: &TR::ModelFamilyRuntime,
    positions: &TR::Tensor,
    q: &TR::Tensor,
    k: &TR::Tensor,
    Tracked(pp): Tracked<&TR::TensorPerm>,
    Tracked(qp): Tracked<&TR::TensorPerm>,
    Tracked(kp): Tracked<&TR::TensorPerm>,
    Ghost(pir): Ghost<IntTensor1D>,
    Ghost(qr): Ghost<Tensor2D>,
    Ghost(kr): Ghost<Tensor2D>,
    Ghost(geometry): Ghost<AttentionGeometryRepr>,
    Ghost(rotary): Ghost<RotaryConfigRepr>,
    Ghost(scope): Ghost<Set<TR::TensorId>>,
) -> (out: ((TR::Tensor, TR::Tensor),
            (Tracked<TR::TensorPerm>, Tracked<TR::TensorPerm>)))
    requires
        TR::dense_swiglu_runtime_rotary_matches(
            runtime, geometry, rotary,
        ),
        TR::int_tensor_repr_1d(*pp, *positions, pir),
        TR::tensor_repr_2d(*qp, *q, qr),
        TR::tensor_repr_2d(*kp, *k, kr),
        pir.len() == qr.len(),
        qr.len() == kr.len(),
    ensures ({
        let ((rq, rk), (rqp, rkp)) = out;
        let rotated = (
            TR::rotary_component_raw_output(pir, qr, geometry.num_attention_heads, geometry.head_dim, rotary),
            TR::rotary_component_raw_output(pir, kr, geometry.num_key_value_heads, geometry.head_dim, rotary));
        &&& TR::rotary_component_layout(pir, qr, geometry.num_attention_heads, geometry.head_dim, rotary)
        &&& TR::rotary_component_layout(pir, kr, geometry.num_key_value_heads, geometry.head_dim, rotary)
        &&& rq.id() != rk.id()
        &&& !scope.contains(rq.id())
        &&& !scope.contains(rk.id())
        &&& TR::tensor_repr_2d(rqp@, rq, rotated.0)
        &&& TR::tensor_repr_2d(rkp@, rk, rotated.1)
    }),
{
    #[cfg(not(verus_only))]
    {
        let (a, b) = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<(
                pyo3::Py<pyo3::PyAny>,
                pyo3::Py<pyo3::PyAny>,
            )> {
                let module = py.import_bound("vosti_kernels")?;
                let result = module.getattr("rotary_embed")?.call1((
                    positions.inner.bind(py),
                    q.inner.bind(py),
                    k.inner.bind(py),
                    TR::primitive_runtime_argument(runtime, py),
                ))?;
                let tuple = result.downcast::<pyo3::types::PyTuple>()?;
                Ok((tuple.get_item(0)?.unbind(), tuple.get_item(1)?.unbind()))
            },
        ).expect("python rotary_embed kernel failed");
        (
            (TR::Tensor { inner: a }, TR::Tensor { inner: b }),
            (Tracked::assume_new(), Tracked::assume_new()),
        )
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}
// @kernel-bridge-end boundary::dense_layer_primitives::rotary_embed

#[verifier::opaque]
pub open spec fn rotary_row_with_config_repr(
    position: int,
    row: Tensor1D,
    heads: nat,
    head_dim: nat,
    config: RotaryConfigRepr,
) -> Tensor1D {
    TR::rotary_component_repr(seq![position], seq![row], heads, head_dim, config)[0]
}

pub proof fn checked_rotary_rows_binding(
    positions: IntTensor1D, input: Tensor2D, heads: nat, head_dim: nat, config: RotaryConfigRepr,
)
    requires positions.len() == input.len(),
        TR::rotary_component_layout(positions, input, heads, head_dim, config),
    ensures TR::rotary_component_raw_output(positions, input, heads, head_dim, config)
        == Seq::new(input.len(), |r: int| rotary_row_with_config_repr(positions[r], input[r], heads, head_dim, config)),
{
    TR::checked_rotary_component_binding(positions, input, heads, head_dim, config);
    reveal(rotary_row_with_config_repr);
    assert forall|r: int| 0 <= r < input.len() implies
        (#[trigger] TR::rotary_component_repr(positions, input, heads, head_dim, config)[r])
        == rotary_row_with_config_repr(positions[r], input[r], heads, head_dim, config) by {
        crate::proof::model::dense_swiglu::batch_invariance::rotary_component_batch_invariance(
            positions, input, heads, head_dim, config, r as nat);
    };
    TR::lemma_rotary_embed_repr_shape(positions, input, input, heads, heads, head_dim, config);
    reveal(TR::rotary_embed_repr);
    assert(TR::rotary_component_raw_output(positions, input, heads, head_dim, config)
        =~= Seq::new(input.len(), |r: int| rotary_row_with_config_repr(positions[r], input[r], heads, head_dim, config)));
}

// The selected-row imports retain the same explicit finite-value premise as
// production paged attention. This alias keeps the numeric obligation shared
// across every model family.
pub open spec fn attention_numeric_domain() -> bool {
    TR::paged_attention_numeric_domain()
}

pub open spec fn full_paged_attention_repr(
    q: Tensor2D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    parameters: AttentionParametersRepr,
) -> Tensor2D {
    crate::boundary::attention_operator::output(q, k_cache, v_cache,
        cu_q, cu_k, block_table, AttentionKind::Full, parameters, 0)
}

pub open spec fn sliding_window_paged_attention_repr(
    q: Tensor2D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    window_size: nat,
    parameters: AttentionParametersRepr,
) -> Tensor2D {
    crate::boundary::attention_operator::output(q, k_cache, v_cache,
        cu_q, cu_k, block_table, AttentionKind::SlidingWindow, parameters, window_size)
}

pub uninterp spec fn merge_attention_heads_row_repr(row: Tensor1D) -> Tensor1D;

pub open spec fn merge_attention_heads_repr(input: Tensor2D) -> Tensor2D {
    Seq::new(input.len(), |i: int| merge_attention_heads_row_repr(input[i]))
}

pub broadcast proof fn lemma_merge_attention_heads_repr_shape(input: Tensor2D)
    ensures #[trigger] merge_attention_heads_repr(input).len() == input.len(),
{
    reveal(merge_attention_heads_repr);
}

} // verus!
