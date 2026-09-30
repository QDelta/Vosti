// Trusted tensor runtime: the bridge between the Verus proof world and the
// Python kernel implementations called via PyO3. Mirrors the earlier Dafny
// tensor runtime, recast for Verus:
//
//   * `Tensor` is an `external_body` exec wrapper around a Python handle.
//     It carries an opaque `TensorId` (analogous to `vstd::cell::CellId`).
//
//   * `TensorPerm` is a `tracked` struct: the proof-only "identity token"
//     that pins a specific physical Python tensor to a ghost repr value.
//     Mutation flows through `&mut TensorPerm` — when a kernel mutates the
//     underlying Python tensor, it consumes a mutable permission and the
//     postcondition advances `perm.repr_2d()` to the new value.  Other
//     permissions (for other tensors) are mechanically untouched: this is
//     how we get framing for free, replacing Dafny's `modifies` clauses.
//
// Soundness obligations (see `docs/architecture.md` "Soundness obligations"):
//   1. `TensorPerm` is unforgeable: no public constructor, no `Clone`.
//   2. Allocator postconditions state every alias separation used by callers;
//      physical-object-to-id correspondence remains an external obligation.
//   3. `tensor_repr_*` predicates openly bind tensor/permission identity and
//      the permission's uninterpreted representation accessor.
//   4. Every `external_body` Python kernel must faithfully implement its
//      `ensures` postcondition.  This is the trust boundary.
//   5. Launched-config determinism (see docs/deployment.md):
//      engine contracts expose only relational operations over visible tensor
//      inputs; launch blocks are not proof inputs.  Offline qualification
//      checks every launch selected by the sealed deployment plan against the
//      pinned selector.  This matters because another launch configuration can
//      reorder non-associative floating-point reductions on real hardware.

#![cfg(any(verus_only, not(verus_only)))] // allow under both verification and build

use crate::model_config::{AttentionKind, FloatParameterBits, ModelArchitecture};
use crate::boundary::model_families::gemma3::config::Gemma3Config;
use crate::boundary::model_families::gemma4::config::Gemma4Config;
use crate::boundary::model_families::llama3::config::Llama3Config;
use crate::boundary::model_families::qwen3::config::{QWEN3_ROPE_THETA_F64_BITS, Qwen3Config};
use core::marker::PhantomData;
#[cfg(verus_only)]
use crate::boundary::scalar::{float_parameter_scalar_repr, positive_float_parameter_valid};
use vstd::prelude::*;
#[cfg(verus_only)]
use crate::boundary::attention_operator as ATTN;
#[cfg(verus_only)]
use crate::boundary::linear_operator as LINEAR;
#[cfg(verus_only)]
use crate::boundary::qkv_operator as QKV;
#[cfg(verus_only)]
use crate::boundary::normalization_operator as NORM;
#[cfg(verus_only)]
use crate::boundary::pointwise_operator as PW;
#[cfg(verus_only)]
use crate::boundary::embedding_operator as EMBED;
#[cfg(verus_only)]
use crate::boundary::head_normalization_operator as HEAD;
#[cfg(verus_only)]
use crate::boundary::rotary_operator as ROT;
#[cfg(verus_only)]
use crate::boundary::kv_store_operator as KV_STORE;
#[cfg(verus_only)]
use crate::boundary::backend_certificates::{attention as RAW_ATTENTION, support as KERNEL_SUPPORT};


#[cfg(verus_only)]
use crate::boundary::backend_certificates::support as GKC;
#[cfg(verus_only)]
use crate::boundary::dense_layer_primitives as DLP;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::proof::tensor::shape as TS;

pub use crate::boundary::model_families::gemma3::{
    Gemma3LayerWeights, Gemma3LayerWeightsPerms, Gemma3ModelWeights,
};
pub use crate::boundary::model_families::gemma4::weights::Gemma4ModelWeights;
pub use crate::boundary::model_families::llama3::{
    Llama3LayerWeights, Llama3LayerWeightsPerms, Llama3ModelWeights,
};
pub use crate::boundary::model_families::qwen3::{
    Qwen3LayerWeights, Qwen3LayerWeightsPerms, Qwen3ModelWeights,
};
use crate::boundary::model_families::{
    gemma3 as GEMMA3, gemma4 as GEMMA4, llama3 as LLAMA3, qwen3 as QWEN3,
};

// Note: `pyo3` is only needed at exec; verification erases the body.
#[cfg(not(verus_only))]
use pyo3::prelude::*;

verus! {

// ---------------------------------------------------------------------------
// Tensor identity.  Uninterpreted opaque type, like `vstd::cell::CellId`.
// ---------------------------------------------------------------------------

#[verifier::external_body]
#[verifier::ext_equal]
pub struct TensorId {
    _private: (),
}

// ---------------------------------------------------------------------------
// Exec-side tensor handle.  Wraps a `pyo3::Py<PyAny>` (a refcounted Python
// reference).  Cheaply cloneable — multiple Rust `Tensor` values may point
// at the same Python object.  All proof reasoning happens via `TensorPerm`,
// not via the handle, so aliasing the handle is safe.
// ---------------------------------------------------------------------------

#[verifier::external_body]
pub struct Tensor {
    // Real exec field, only present at build time.  Marked `pub` so
    // model.rs can call into it via PyO3 without a thin accessor — the
    // type is `external_body` anyway, so visibility doesn't affect the
    // proof world.
    #[cfg(not(verus_only))]
    pub inner: pyo3::Py<pyo3::PyAny>,
    #[cfg(verus_only)]
    _phantom: PhantomData<()>,
}

impl Tensor {
    /// Stable identity used to bind permissions to this tensor.  Two `Tensor`
    /// values produced by the same allocator call (e.g. via `Clone`) share an
    /// id; values produced by distinct allocators have distinct ids.
    pub uninterp spec fn id(&self) -> TensorId;
}

// ---------------------------------------------------------------------------
// CUDA-graph model-forward overlay.
//
// This is opaque runtime policy state, deliberately separate from `Engine` and
// all semantic invariants.  Each instance owns only graphs captured from the
// canonical verified `model_forward` call for one engine.  Capture/replay
// fidelity is assumed by the architecture-neutral raw launcher; checked
// family projection wrappers remain under `exec::model_families`, and Engine
// never reasons about the contents of this object.
// ---------------------------------------------------------------------------

// @kernel-bridge-begin boundary::tensor_runtime::cuda_graph_overlay_runtime
// Named deployment premise for the one semantic fact deliberately left out of
// scope: for an overlay paired with one engine and its stable weight/KV
// addresses, replay faithfully executes the canonical eager launch that was
// captured, including its installed padded metadata.  Equivalence between a
// padded pure-decode launch and the real unpadded step is proved separately in
// `graph_cover`; it is not part of this premise.
pub uninterp spec fn cuda_graph_replay_fidelity() -> bool;

#[verifier::external_body]
pub struct CudaGraphOverlay {
    #[cfg(not(verus_only))]
    pub inner: pyo3::Py<pyo3::PyAny>,
    #[cfg(verus_only)]
    _phantom: PhantomData<()>,
}

#[verifier::external_body]
pub fn init_cuda_graph_overlay() -> (out: CudaGraphOverlay) {
    #[cfg(not(verus_only))]
    {
        let inner = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let module = py.import_bound("vosti_kernels.graph_overlay")?;
                Ok(module.getattr("CudaGraphOverlay")?.call0()?.unbind())
            },
        )
        .expect("initialize per-engine CUDA-graph overlay");
        CudaGraphOverlay { inner }
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

#[verifier::external_body]
pub fn cuda_graph_overlay_stats_json(overlay: &CudaGraphOverlay) -> (out: String) {
    #[cfg(not(verus_only))]
    {
        pyo3::Python::with_gil(|py| -> pyo3::PyResult<String> {
            overlay.inner.bind(py).call_method0("stats_json")?.extract()
        })
        .expect("read CUDA-graph overlay statistics")
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}
// @kernel-bridge-end boundary::tensor_runtime::cuda_graph_overlay_runtime

// ---------------------------------------------------------------------------
// Tracked permission.  This is the *only* way the proof world reasons about
// what a physical Python tensor "contains".  Constructed only by allocator
// `external_body` functions; not `Clone`, not `Copy`.
// ---------------------------------------------------------------------------

#[verifier::external_body]
pub tracked struct TensorPerm {
    _no_copy: NoCopy,
}

// View into the tracked perm.  We expose three flavors of repr corresponding
// to the three Dafny predicates `TensorRepr1D`, `TensorRepr2D`,
// `IntTensorRepr1D`, plus the cache and block-table reprs.  At any point a
// perm has exactly one repr "kind" (set by the allocator that minted it);
// asking for the wrong kind returns an unconstrained value (uninterp), so
// callers must keep track of which kind a perm is.
//
// We intentionally do NOT use a single `repr: SomeEnum` field, because each
// kernel's ensures only refers to the kind it cares about, and Verus's
// equality reasoning is cleaner over individual uninterpreted spec fns.
impl TensorPerm {
    pub uninterp spec fn id(&self) -> TensorId;

    pub uninterp spec fn repr_1d(&self) -> Tensor1D;
    pub uninterp spec fn repr_2d(&self) -> Tensor2D;
    pub uninterp spec fn int_repr_1d(&self) -> IntTensor1D;
    pub uninterp spec fn block_table_repr(&self) -> Seq<Seq<BlockId>>;
    pub uninterp spec fn kv_cache_repr(&self) -> KVCacheLayerRepr;
}

// ---------------------------------------------------------------------------
// Repr-bond predicates.  `tensor_repr_2d(perm, t, r)` says: this perm is the
// proof identity for tensor `t`, and the perm's 2D repr equals `r`.
// Uninterpreted, but the kernel ensures clauses produce these directly.
// ---------------------------------------------------------------------------

pub open spec fn tensor_repr_1d(perm: TensorPerm, t: Tensor, r: Tensor1D) -> bool {
    perm.id() == t.id() && perm.repr_1d() == r
}

pub open spec fn tensor_repr_2d(perm: TensorPerm, t: Tensor, r: Tensor2D) -> bool {
    perm.id() == t.id()
    && perm.repr_2d() == r
    && TS::rectangular(r)
}

pub open spec fn int_tensor_repr_1d(perm: TensorPerm, t: Tensor, r: IntTensor1D) -> bool {
    perm.id() == t.id() && perm.int_repr_1d() == r
}

pub open spec fn block_table_repr(perm: TensorPerm, t: Tensor, r: Seq<Seq<BlockId>>) -> bool {
    perm.id() == t.id() && perm.block_table_repr() == r
}

pub open spec fn kv_cache_tensor_repr(perm: TensorPerm, t: Tensor, r: KVCacheLayerRepr) -> bool {
    perm.id() == t.id() && perm.kv_cache_repr() == r
}

// ---------------------------------------------------------------------------
// Pure repr functions — operate on repr types only. These model what each
// kernel does at the value level; the kernel methods below have postconditions
// such as `perm_out.repr_2d() == linear_repr(xr, wr)`. Engine-reachable
// row-wise kernels expose only configuration-free relational operations.
// Generated per-launch ContractIR certificates qualify the sealed deployment
// beneath this boundary; runtime adapters and numerical leaves are explicit
// uninterpreted/trusted boundaries documented in `docs/verification.md`.
// ---------------------------------------------------------------------------

// The engine reasons about one deployed linear kernel through only its
// configuration-erased relational contract.  Tile sizes and the offline
// (N,K)-to-config policy remain deployment-qualification data and never enter
// this semantics.  The uninterpreted cell denotes the fixed deployed kernel's
// singleton-row result; the row map is exactly the annotation's batch-axis
// projection contract.
// @kernel-bridge-begin boundary::tensor_runtime::linear_repr
#[verifier::opaque]
pub open spec fn linear_kernel_cell_repr(
    row: Tensor1D, wr: Tensor2D, col: int,
) -> Scalar {
    LINEAR::cell(row, wr, col)
}

pub open spec fn linear_kernel_row_repr(row: Tensor1D, wr: Tensor2D)
    -> Tensor1D {
    Seq::new(wr.len(), |col: int| linear_kernel_cell_repr(row, wr, col))
}

pub open spec fn linear_repr(xr: Tensor2D, wr: Tensor2D) -> Tensor2D {
    Seq::new(xr.len(), |i: int| linear_kernel_row_repr(xr[i], wr))
}
// @kernel-bridge-end boundary::tensor_runtime::linear_repr

// One physical launch computes the three projections.  The engine sees one
// configuration-free row relation for that launch; the selected BLOCK_M/N/K
// values and their generated certificates remain offline qualification data.
// @kernel-bridge-begin boundary::tensor_runtime::qkv_linear_repr
#[verifier::opaque]
pub open spec fn qkv_q_kernel_cell_repr(
    row: Tensor1D, qwr: Tensor2D, kwr: Tensor2D, vwr: Tensor2D, col: int,
) -> Scalar {
    QKV::row_output(row, qwr, kwr, vwr).0[col]
}

#[verifier::opaque]
pub open spec fn qkv_k_kernel_cell_repr(
    row: Tensor1D, qwr: Tensor2D, kwr: Tensor2D, vwr: Tensor2D, col: int,
) -> Scalar {
    QKV::row_output(row, qwr, kwr, vwr).1[col]
}

#[verifier::opaque]
pub open spec fn qkv_v_kernel_cell_repr(
    row: Tensor1D, qwr: Tensor2D, kwr: Tensor2D, vwr: Tensor2D, col: int,
) -> Scalar {
    QKV::row_output(row, qwr, kwr, vwr).2[col]
}

pub open spec fn qkv_linear_repr(
    xr: Tensor2D,
    qwr: Tensor2D,
    kwr: Tensor2D,
    vwr: Tensor2D,
) -> (Tensor2D, Tensor2D, Tensor2D) {
    (
        Seq::new(xr.len(), |i: int|
            Seq::new(qwr.len(), |col: int|
                qkv_q_kernel_cell_repr(xr[i], qwr, kwr, vwr, col))),
        Seq::new(xr.len(), |i: int|
            Seq::new(kwr.len(), |col: int|
                qkv_k_kernel_cell_repr(xr[i], qwr, kwr, vwr, col))),
        Seq::new(xr.len(), |i: int|
            Seq::new(kwr.len(), |col: int|
                qkv_v_kernel_cell_repr(xr[i], qwr, kwr, vwr, col))),
    )
}

pub open spec fn qkv_q_kernel_repr(
    xr: Tensor2D, qwr: Tensor2D, kwr: Tensor2D, vwr: Tensor2D,
) -> Tensor2D {
    qkv_linear_repr(xr, qwr, kwr, vwr).0
}

pub open spec fn qkv_k_kernel_repr(
    xr: Tensor2D, qwr: Tensor2D, kwr: Tensor2D, vwr: Tensor2D,
) -> Tensor2D {
    qkv_linear_repr(xr, qwr, kwr, vwr).1
}

pub open spec fn qkv_v_kernel_repr(
    xr: Tensor2D, qwr: Tensor2D, kwr: Tensor2D, vwr: Tensor2D,
) -> Tensor2D {
    qkv_linear_repr(xr, qwr, kwr, vwr).2
}
// @kernel-bridge-end boundary::tensor_runtime::qkv_linear_repr

pub broadcast proof fn lemma_qkv_linear_repr_shape(
    xr: Tensor2D,
    qwr: Tensor2D,
    kwr: Tensor2D,
    vwr: Tensor2D,
)
    ensures
        #[trigger] qkv_linear_repr(xr, qwr, kwr, vwr).0.len() == xr.len(),
        qkv_linear_repr(xr, qwr, kwr, vwr).1.len() == xr.len(),
        qkv_linear_repr(xr, qwr, kwr, vwr).2.len() == xr.len(),
        TS::tensor2d_shape(
            qkv_linear_repr(xr, qwr, kwr, vwr).0, xr.len(), qwr.len(),
        ),
        TS::tensor2d_shape(
            qkv_linear_repr(xr, qwr, kwr, vwr).1, xr.len(), kwr.len(),
        ),
        TS::tensor2d_shape(
            qkv_linear_repr(xr, qwr, kwr, vwr).2, xr.len(), kwr.len(),
        ),
{
    reveal(qkv_linear_repr);
    assert forall|i: int| 0 <= i < xr.len() implies
        (#[trigger] qkv_linear_repr(xr, qwr, kwr, vwr).0[i]).len()
            == qwr.len() by {}
    assert forall|i: int| 0 <= i < xr.len() implies
        (#[trigger] qkv_linear_repr(xr, qwr, kwr, vwr).1[i]).len()
            == kwr.len() by {}
    assert forall|i: int| 0 <= i < xr.len() implies
        (#[trigger] qkv_linear_repr(xr, qwr, kwr, vwr).2[i]).len()
            == kwr.len() by {}
}

pub broadcast proof fn lemma_linear_repr_shape(xr: Tensor2D, wr: Tensor2D)
    ensures
        #[trigger] linear_repr(xr, wr).len() == xr.len(),
        TS::tensor2d_shape(linear_repr(xr, wr), xr.len(), wr.len()),
        TS::rectangular(linear_repr(xr, wr)),
{
    let cols = wr.len();
    reveal(linear_repr);
    reveal(linear_kernel_row_repr);
    assert forall|i: int| 0 <= i < xr.len() implies
        (#[trigger] linear_repr(xr, wr)[i]).len() == cols by {
    }
    assert(TS::tensor2d_shape(linear_repr(xr, wr), xr.len(), cols));
}

// Embedding has one proof-facing relation. Hidden width is ordinary tensor
// geometry, not a selector for a distinct kernel in the engine proof.
// @kernel-bridge-begin boundary::tensor_runtime::embed_repr
pub open spec fn embed_weight_width_repr(wr: Tensor2D) -> nat {
    if wr.len() > 0 { wr[0].len() } else { 0 }
}

#[verifier::opaque]
pub open spec fn embed_kernel_cell_repr(
    token: int, wr: Tensor2D, col: int,
) -> Scalar {
    EMBED::plain_row_output(token, wr)[col]
}

pub open spec fn embed_kernel_row_repr(token: int, wr: Tensor2D) -> Tensor1D {
    Seq::new(embed_weight_width_repr(wr), |col: int|
        embed_kernel_cell_repr(token, wr, col))
}

pub open spec fn embed_repr(ir: IntTensor1D, wr: Tensor2D) -> Tensor2D {
    Seq::new(ir.len(), |i: int| embed_kernel_row_repr(ir[i], wr))
}
// @kernel-bridge-end boundary::tensor_runtime::embed_repr

// Neutral opaque cells used by architecture compositions whose element-wise
// operations differ from the shared dense SwiGLU path. They expose only the
// row relation required for determinism, never a numerical definition or
// launch configuration.
#[verifier::opaque]
pub open spec fn scaled_embed_kernel_cell_repr(
    token: int, weight: Tensor2D, hidden_size: nat, col: int,
) -> Scalar {
    EMBED::scaled_row_output(token, weight, hidden_size)[col]
}

#[verifier::opaque]
pub open spec fn offset_rms_norm_kernel_cell_repr(
    row: Tensor1D, weight: Tensor1D, col: int,
) -> Scalar {
    NORM::offset_row_output(row, weight,
        float_parameter_scalar_repr(unit_offset_norm_epsilon_repr()))[col]
}

#[verifier::opaque]
pub open spec fn gelu_tanh_mul_kernel_cell_repr(
    gate_row: Tensor1D, up_row: Tensor1D, col: int,
) -> Scalar {
    PW::gelu_tanh_mul_row_output(gate_row, up_row)[col]
}

#[verifier::opaque]
pub open spec fn add_kernel_cell_repr(
    left_row: Tensor1D, right_row: Tensor1D, col: int,
) -> Scalar {
    PW::add_row_output(left_row, right_row)[col]
}

// Newly supported neutral pointwise operators. The kernel verifier proves
// their row relations; arithmetic and launch configuration stay opaque here.
#[verifier::opaque]
pub open spec fn scale_kernel_cell_repr(
    row: Tensor1D, scalar: Tensor1D, col: int,
) -> Scalar {
    PW::scale_row_output(row, scalar)[col]
}

#[verifier::opaque]
pub open spec fn softcap_kernel_cell_repr(
    row: Tensor1D, cap: Scalar, col: int,
) -> Scalar {
    PW::softcap_row_output(row, cap)[col]
}

pub broadcast proof fn lemma_embed_repr_shape(ir: IntTensor1D, wr: Tensor2D)
    ensures
        #[trigger] embed_repr(ir, wr).len() == ir.len(),
        TS::rectangular(embed_repr(ir, wr)),
{
    let cols = embed_weight_width_repr(wr);
    reveal(embed_repr);
    reveal(embed_kernel_row_repr);
    assert forall|i: int| 0 <= i < ir.len() implies
        (#[trigger] embed_repr(ir, wr)[i]).len() == cols by {
    }
    assert(TS::tensor2d_shape(embed_repr(ir, wr), ir.len(), cols));
}

// The kernel theorem is universal in epsilon. Family admission retains the
// exact binary64 identity and passes it through this shared semantic boundary.
// @kernel-bridge-begin boundary::tensor_runtime::rms_norm_kernel_repr
#[verifier::opaque]
pub open spec fn rms_norm_kernel_cell_repr(
    row: Tensor1D, wr: Tensor1D, eps: Scalar, col: int,
) -> Scalar {
    NORM::rms_row_output(row, wr, eps)[col]
}

pub open spec fn rms_norm_kernel_row_repr(
    row: Tensor1D, wr: Tensor1D, eps: Scalar,
) -> Tensor1D {
    Seq::new(wr.len(), |col: int|
        rms_norm_kernel_cell_repr(row, wr, eps, col))
}

pub open spec fn rms_norm_kernel_repr(
    xr: Tensor2D, wr: Tensor1D, eps: Scalar,
) -> Tensor2D {
    Seq::new(xr.len(), |i: int| rms_norm_kernel_row_repr(xr[i], wr, eps))
}
// @kernel-bridge-end boundary::tensor_runtime::rms_norm_kernel_repr

// @kernel-bridge-begin boundary::tensor_runtime::rms_norm_repr
#[verifier::opaque]
pub open spec fn rms_norm_repr(
    xr: Tensor2D,
    wr: Tensor1D,
    epsilon: FloatParameterBits,
) -> Tensor2D {
    rms_norm_kernel_repr(
        xr, wr, float_parameter_scalar_repr(epsilon),
    )
}
// @kernel-bridge-end boundary::tensor_runtime::rms_norm_repr

pub broadcast proof fn lemma_rms_norm_repr_len(
    xr: Tensor2D,
    wr: Tensor1D,
    epsilon: FloatParameterBits,
)
    ensures #[trigger] rms_norm_repr(xr, wr, epsilon).len() == xr.len(),
{
    reveal(rms_norm_repr);
    reveal(rms_norm_kernel_repr);
}

pub broadcast proof fn lemma_rms_norm_repr_shape(
    xr: Tensor2D,
    wr: Tensor1D,
    epsilon: FloatParameterBits,
)
    ensures
        #[trigger] rms_norm_repr(xr, wr, epsilon).len() == xr.len(),
        TS::rectangular(rms_norm_repr(xr, wr, epsilon)),
{
    let cols = wr.len();
    reveal(rms_norm_repr);
    reveal(rms_norm_kernel_repr);
    reveal(rms_norm_kernel_row_repr);
    assert(rms_norm_repr(xr, wr, epsilon).len() == xr.len());
    assert forall|i: int| 0 <= i < xr.len() implies
        (#[trigger] rms_norm_repr(xr, wr, epsilon)[i]).len() == cols by {
    }
    assert(TS::tensor2d_shape(
        rms_norm_repr(xr, wr, epsilon), xr.len(), cols,
    ));
}

// Residual RMSNorm has two independently observable output tensors.  The
// engine sees one two-output relation; launch configuration is qualified below
// this boundary and never appears in either semantic function.
// @kernel-bridge-begin boundary::tensor_runtime::add_rms_norm_kernel_repr
#[verifier::opaque]
pub open spec fn add_rms_norm_output_kernel_cell_repr(
    row: Tensor1D, residual_row: Tensor1D, wr: Tensor1D, eps: Scalar, col: int,
) -> Scalar {
    NORM::residual_row_output(row, residual_row, wr, eps).0[col]
}

#[verifier::opaque]
pub open spec fn add_rms_norm_residual_kernel_cell_repr(
    row: Tensor1D, residual_row: Tensor1D, wr: Tensor1D, eps: Scalar, col: int,
) -> Scalar {
    NORM::residual_row_output(row, residual_row, wr, eps).1[col]
}

pub open spec fn add_rms_norm_output_kernel_row_repr(
    row: Tensor1D, residual_row: Tensor1D, wr: Tensor1D, eps: Scalar,
) -> Tensor1D {
    Seq::new(wr.len(), |col: int|
        add_rms_norm_output_kernel_cell_repr(row, residual_row, wr, eps, col))
}

pub open spec fn add_rms_norm_residual_kernel_row_repr(
    row: Tensor1D, residual_row: Tensor1D, wr: Tensor1D, eps: Scalar,
) -> Tensor1D {
    Seq::new(wr.len(), |col: int|
        add_rms_norm_residual_kernel_cell_repr(row, residual_row, wr, eps, col))
}

pub open spec fn add_rms_norm_output_kernel_repr(
    xr: Tensor2D, rr: Tensor2D, wr: Tensor1D, eps: Scalar,
) -> Tensor2D {
    Seq::new(xr.len(), |i: int| {
        let row = xr[i];
        let residual_row = if i < rr.len() { rr[i] } else { Seq::empty() };
        add_rms_norm_output_kernel_row_repr(row, residual_row, wr, eps)
    })
}
pub open spec fn add_rms_norm_residual_kernel_repr(
    xr: Tensor2D, rr: Tensor2D, wr: Tensor1D, eps: Scalar,
) -> Tensor2D {
    Seq::new(xr.len(), |i: int| {
        let row = xr[i];
        let residual_row = if i < rr.len() { rr[i] } else { Seq::empty() };
        add_rms_norm_residual_kernel_row_repr(row, residual_row, wr, eps)
    })
}
// @kernel-bridge-end boundary::tensor_runtime::add_rms_norm_kernel_repr

// @kernel-bridge-begin boundary::tensor_runtime::add_rms_norm_repr
#[verifier::opaque]
pub open spec fn add_rms_norm_repr(
    xr: Tensor2D,
    rr: Tensor2D,
    wr: Tensor1D,
    epsilon: FloatParameterBits,
) -> (Tensor2D, Tensor2D) {
    let eps = float_parameter_scalar_repr(epsilon);
    (add_rms_norm_output_kernel_repr(xr, rr, wr, eps),
     add_rms_norm_residual_kernel_repr(xr, rr, wr, eps))
}
// @kernel-bridge-end boundary::tensor_runtime::add_rms_norm_repr

pub broadcast proof fn lemma_add_rms_norm_repr_len(
    xr: Tensor2D,
    rr: Tensor2D,
    wr: Tensor1D,
    epsilon: FloatParameterBits,
)
    ensures
        (#[trigger] add_rms_norm_repr(xr, rr, wr, epsilon)).0.len() == xr.len(),
        add_rms_norm_repr(xr, rr, wr, epsilon).1.len() == xr.len(),
{
    reveal(add_rms_norm_repr);
    reveal(add_rms_norm_output_kernel_repr);
    reveal(add_rms_norm_residual_kernel_repr);
}

pub broadcast proof fn lemma_add_rms_norm_kernel_repr_shape(
    xr: Tensor2D, rr: Tensor2D, wr: Tensor1D, eps: Scalar,
)
    ensures
        #[trigger] add_rms_norm_output_kernel_repr(xr, rr, wr, eps).len() == xr.len(),
        #[trigger] add_rms_norm_residual_kernel_repr(xr, rr, wr, eps).len() == xr.len(),
        TS::rectangular(add_rms_norm_output_kernel_repr(xr, rr, wr, eps)),
        TS::rectangular(add_rms_norm_residual_kernel_repr(xr, rr, wr, eps)),
{
    let cols = wr.len();
    reveal(add_rms_norm_output_kernel_repr);
    reveal(add_rms_norm_residual_kernel_repr);
    reveal(add_rms_norm_output_kernel_row_repr);
    reveal(add_rms_norm_residual_kernel_row_repr);
    assert forall|i: int| 0 <= i < xr.len() implies
        (#[trigger] add_rms_norm_output_kernel_repr(xr, rr, wr, eps)[i]).len()
            == cols by {
    }
    assert forall|i: int| 0 <= i < xr.len() implies
        (#[trigger] add_rms_norm_residual_kernel_repr(xr, rr, wr, eps)[i]).len()
            == cols by {
    }
    assert(TS::tensor2d_shape(
        add_rms_norm_output_kernel_repr(xr, rr, wr, eps), xr.len(), cols,
    ));
    assert(TS::tensor2d_shape(
        add_rms_norm_residual_kernel_repr(xr, rr, wr, eps), xr.len(), cols,
    ));
}

pub broadcast proof fn lemma_add_rms_norm_repr_shape(
    xr: Tensor2D,
    rr: Tensor2D,
    wr: Tensor1D,
    epsilon: FloatParameterBits,
)
    ensures
        (#[trigger] add_rms_norm_repr(xr, rr, wr, epsilon)).0.len() == xr.len(),
        add_rms_norm_repr(xr, rr, wr, epsilon).1.len() == xr.len(),
        TS::rectangular(add_rms_norm_repr(xr, rr, wr, epsilon).0),
        TS::rectangular(add_rms_norm_repr(xr, rr, wr, epsilon).1),
{
    reveal(add_rms_norm_repr);
    lemma_add_rms_norm_kernel_repr_shape(
        xr, rr, wr, float_parameter_scalar_repr(epsilon),
    );
}

// Q/K normalization has one proof-facing relation for every head geometry.
// Head count and width are tensor geometry, while physical launch constants
// remain sealed deployment data.
// @kernel-bridge-begin boundary::tensor_runtime::head_rms_norm_kernel_repr
pub open spec fn head_rms_norm_output_width_repr(xr: Tensor2D) -> nat {
    if xr.len() > 0 { xr[0].len() } else { 0 }
}

#[verifier::opaque]
pub open spec fn head_rms_norm_kernel_cell_repr(
    row: Tensor1D, wr: Tensor1D, eps: Scalar, col: int,
) -> Scalar {
    HEAD::row_output(row, wr, eps, false)[col]
}

pub open spec fn head_rms_norm_kernel_row_repr(
    row: Tensor1D, wr: Tensor1D, eps: Scalar, cols: nat,
) -> Tensor1D {
    Seq::new(cols, |col: int|
        head_rms_norm_kernel_cell_repr(row, wr, eps, col))
}

pub open spec fn head_rms_norm_kernel_repr(
    xr: Tensor2D, wr: Tensor1D, eps: Scalar,
) -> Tensor2D {
    let cols = head_rms_norm_output_width_repr(xr);
    Seq::new(xr.len(), |i: int|
        head_rms_norm_kernel_row_repr(xr[i], wr, eps, cols))
}
// @kernel-bridge-end boundary::tensor_runtime::head_rms_norm_kernel_repr

// @kernel-bridge-begin boundary::tensor_runtime::qk_norm_repr
#[verifier::opaque]
pub open spec fn qk_norm_repr(
    qr: Tensor2D,
    kr: Tensor2D,
    qnr: Tensor1D,
    knr: Tensor1D,
    epsilon: FloatParameterBits,
) -> (Tensor2D, Tensor2D) {
    let eps = float_parameter_scalar_repr(epsilon);
    (head_rms_norm_kernel_repr(qr, qnr, eps),
     head_rms_norm_kernel_repr(kr, knr, eps))
}
// @kernel-bridge-end boundary::tensor_runtime::qk_norm_repr

// Architecture-neutral composition of the optional Q/K-normalization layer.
// The disabled case is literal identity: it introduces neither semantic
// weights nor an implicit normalization operation.  The closed model
// composition decides which branch is admissible before execution starts.
pub open spec fn apply_qk_norm_repr(
    qr: Tensor2D,
    kr: Tensor2D,
    weights: QkNormWeightsRepr,
    epsilon: FloatParameterBits,
) -> (Tensor2D, Tensor2D) {
    match weights {
        QkNormWeightsRepr::Disabled => (qr, kr),
        QkNormWeightsRepr::RmsNorm { q_weight, k_weight } =>
            qk_norm_repr(qr, kr, q_weight, k_weight, epsilon),
    }
}

pub broadcast proof fn lemma_apply_qk_norm_repr_shape(
    qr: Tensor2D,
    kr: Tensor2D,
    weights: QkNormWeightsRepr,
    epsilon: FloatParameterBits,
)
    requires
        qr.len() == kr.len(),
        TS::rectangular(qr),
        TS::rectangular(kr),
    ensures
        (#[trigger] apply_qk_norm_repr(qr, kr, weights, epsilon)).0.len()
            == qr.len(),
        apply_qk_norm_repr(qr, kr, weights, epsilon).1.len() == qr.len(),
        TS::rectangular(apply_qk_norm_repr(qr, kr, weights, epsilon).0),
        TS::rectangular(apply_qk_norm_repr(qr, kr, weights, epsilon).1),
{
    reveal(apply_qk_norm_repr);
    match weights {
        QkNormWeightsRepr::Disabled => {},
        QkNormWeightsRepr::RmsNorm { q_weight, k_weight } => {
            lemma_qk_norm_repr_shape(
                qr, kr, q_weight, k_weight, epsilon,
            );
        },
    }
}

pub broadcast proof fn lemma_qk_norm_repr_len(
    qr: Tensor2D,
    kr: Tensor2D,
    qnr: Tensor1D,
    knr: Tensor1D,
    epsilon: FloatParameterBits,
)
    ensures
        (#[trigger] qk_norm_repr(qr, kr, qnr, knr, epsilon)).0.len()
            == qr.len(),
        qk_norm_repr(qr, kr, qnr, knr, epsilon).1.len() == kr.len(),
{
    reveal(qk_norm_repr);
    reveal(head_rms_norm_kernel_repr);
}

pub broadcast proof fn lemma_head_rms_norm_kernel_repr_shape(
    xr: Tensor2D, wr: Tensor1D, eps: Scalar,
)
    ensures
        #[trigger] head_rms_norm_kernel_repr(xr, wr, eps).len() == xr.len(),
        TS::rectangular(head_rms_norm_kernel_repr(xr, wr, eps)),
{
    let cols = head_rms_norm_output_width_repr(xr);
    reveal(head_rms_norm_kernel_repr);
    reveal(head_rms_norm_kernel_row_repr);
    assert forall|i: int| 0 <= i < xr.len() implies
        (#[trigger] head_rms_norm_kernel_repr(xr, wr, eps)[i]).len() == cols by {
    }
    assert(TS::tensor2d_shape(
        head_rms_norm_kernel_repr(xr, wr, eps), xr.len(), cols,
    ));
}

pub broadcast proof fn lemma_qk_norm_repr_shape(
    qr: Tensor2D,
    kr: Tensor2D,
    qnr: Tensor1D,
    knr: Tensor1D,
    epsilon: FloatParameterBits,
)
    requires
        qr.len() == kr.len(),
    ensures
        (#[trigger] qk_norm_repr(qr, kr, qnr, knr, epsilon)).0.len()
            == qr.len(),
        qk_norm_repr(qr, kr, qnr, knr, epsilon).1.len() == qr.len(),
        TS::rectangular(qk_norm_repr(qr, kr, qnr, knr, epsilon).0),
        TS::rectangular(qk_norm_repr(qr, kr, qnr, knr, epsilon).1),
{
    reveal(qk_norm_repr);
    let eps = float_parameter_scalar_repr(epsilon);
    lemma_head_rms_norm_kernel_repr_shape(qr, qnr, eps);
    lemma_head_rms_norm_kernel_repr_shape(kr, knr, eps);
}

// @kernel-bridge-begin boundary::tensor_runtime::view_as_kv_repr
pub uninterp spec fn view_as_kv_row_repr(v_row: Tensor1D) -> Tensor1D;

#[verifier::opaque]
pub open spec fn view_as_kv_repr(vr: Tensor2D) -> Tensor2D {
    Seq::new(vr.len(), |i: int| view_as_kv_row_repr(vr[i]))
}

pub broadcast proof fn lemma_view_as_kv_repr_shape(vr: Tensor2D)
    ensures #[trigger] view_as_kv_repr(vr).len() == vr.len(),
{ reveal(view_as_kv_repr); }
// @kernel-bridge-end boundary::tensor_runtime::view_as_kv_repr

// The physical kernel operates on flattened head rows. The engine sees one
// row relation independent of launch tiling; checked adapters below lift it
// back to engine token rows. Table generation remains wrapper semantics
// because it is not a Triton kernel.
// @kernel-bridge-begin boundary::tensor_runtime::rope_kernel_repr
#[verifier::opaque]
pub open spec fn rope_kernel_cell_repr(
    row: Tensor1D, cos_row: Tensor1D, sin_row: Tensor1D, col: int,
) -> Scalar {
    ROT::row_output(row, cos_row, sin_row)[col]
}

pub open spec fn rope_kernel_repr(
    xr: Tensor2D, cosr: Tensor2D, sinr: Tensor2D,
) -> Tensor2D {
    Seq::new(xr.len(), |i: int| {
        let cos_row = if i < cosr.len() { cosr[i] } else { Seq::empty() };
        let sin_row = if i < sinr.len() { sinr[i] } else { Seq::empty() };
        Seq::new(xr[i].len(), |col: int|
            rope_kernel_cell_repr(xr[i], cos_row, sin_row, col))
    })
}
pub uninterp spec fn rope_cos_row_with_config_repr(
    position: int,
    config: RotaryConfigRepr,
) -> Tensor1D;
pub uninterp spec fn rope_sin_row_with_config_repr(
    position: int,
    config: RotaryConfigRepr,
) -> Tensor1D;
pub open spec fn qwen3_rotary_config_repr() -> RotaryConfigRepr {
    RotaryConfigRepr {
        theta: FloatParameterBits {
            bits: QWEN3_ROPE_THETA_F64_BITS,
        },
        scaling: RotaryScalingRepr::None,
    }
}
// @kernel-bridge-end boundary::tensor_runtime::rope_kernel_repr

pub broadcast proof fn lemma_rope_kernel_repr_shape(
    xr: Tensor2D, cosr: Tensor2D, sinr: Tensor2D,
)
    ensures #[trigger] rope_kernel_repr(xr, cosr, sinr).len() == xr.len(),
{
    reveal(rope_kernel_repr);
}

// @kernel-bridge-begin boundary::tensor_runtime::rotary_adapters
#[verifier::opaque]
pub open spec fn split_head_rows_repr(
    xr: Tensor2D, head_count: nat, head_dim: nat,
) -> Tensor2D {
    Seq::new(xr.len(), |i: int|
        Seq::new(head_count, |h: int|
            xr[i].subrange(
                h * head_dim as int, (h + 1) * head_dim as int,
            )
        )
    ).flatten()
}

#[verifier::opaque]
pub open spec fn repeat_rows_repr(rows: Tensor2D, copies: nat) -> Tensor2D {
    Seq::new(rows.len(), |i: int|
        Seq::new(copies, |_copy: int| rows[i])
    ).flatten()
}

#[verifier::opaque]
pub open spec fn merge_head_rows_repr(
    rows: Tensor2D, token_count: nat, head_count: nat,
) -> Tensor2D {
    Seq::new(token_count, |i: int|
        rows.subrange(
            i * head_count as int, (i + 1) * head_count as int,
        ).flatten()
    )
}

#[verifier::opaque]
pub open spec fn rope_cos_rows_repr(
    pir: IntTensor1D,
    config: RotaryConfigRepr,
) -> Tensor2D {
    Seq::new(pir.len(), |i: int|
        rope_cos_row_with_config_repr(pir[i], config))
}

#[verifier::opaque]
pub open spec fn rope_sin_rows_repr(
    pir: IntTensor1D,
    config: RotaryConfigRepr,
) -> Tensor2D {
    Seq::new(pir.len(), |i: int|
        rope_sin_row_with_config_repr(pir[i], config))
}

#[verifier::opaque]
pub open spec fn rotary_component_repr(
    pir: IntTensor1D,
    xr: Tensor2D,
    head_count: nat,
    head_dim: nat,
    config: RotaryConfigRepr,
) -> Tensor2D {
    let split = split_head_rows_repr(xr, head_count, head_dim);
    let cosr = repeat_rows_repr(rope_cos_rows_repr(pir, config), head_count);
    let sinr = repeat_rows_repr(rope_sin_rows_repr(pir, config), head_count);
    merge_head_rows_repr(
        rope_kernel_repr(split, cosr, sinr), xr.len(), head_count,
    )
}
// @kernel-bridge-end boundary::tensor_runtime::rotary_adapters

pub open spec fn rotary_component_layout(
    positions: IntTensor1D, input: Tensor2D, heads: nat, width: nat, config: RotaryConfigRepr,
) -> bool {
    ROT::layout(split_head_rows_repr(input, heads, width),
        repeat_rows_repr(rope_cos_rows_repr(positions, config), heads),
        repeat_rows_repr(rope_sin_rows_repr(positions, config), heads))
}

pub open spec fn rotary_component_raw_output(
    positions: IntTensor1D, input: Tensor2D, heads: nat, width: nat, config: RotaryConfigRepr,
) -> Tensor2D {
    merge_head_rows_repr(ROT::raw_output(split_head_rows_repr(input, heads, width),
        repeat_rows_repr(rope_cos_rows_repr(positions, config), heads),
        repeat_rows_repr(rope_sin_rows_repr(positions, config), heads)), input.len(), heads)
}

pub proof fn checked_rope_binding(input: Tensor2D, cos: Tensor2D, sin: Tensor2D)
    requires ROT::layout(input, cos, sin),
    ensures ROT::raw_output(input, cos, sin) == rope_kernel_repr(input, cos, sin),
{
    ROT::checked_binding(input, cos, sin);
    reveal(rope_kernel_cell_repr);
    assert forall|r: int| 0 <= r < input.len() implies
        (#[trigger] rope_kernel_repr(input, cos, sin)[r]) == ROT::output(input, cos, sin)[r] by {
        ROT::row_shape(input, cos, sin, r);
        assert(rope_kernel_repr(input, cos, sin)[r] =~= ROT::row_output(input[r], cos[r], sin[r]));
    };
    assert(rope_kernel_repr(input, cos, sin) =~= ROT::output(input, cos, sin));
}

pub proof fn checked_rotary_component_binding(
    positions: IntTensor1D, input: Tensor2D, heads: nat, width: nat, config: RotaryConfigRepr,
)
    requires rotary_component_layout(positions, input, heads, width, config),
    ensures rotary_component_raw_output(positions, input, heads, width, config)
        == rotary_component_repr(positions, input, heads, width, config),
{
    checked_rope_binding(split_head_rows_repr(input, heads, width),
        repeat_rows_repr(rope_cos_rows_repr(positions, config), heads),
        repeat_rows_repr(rope_sin_rows_repr(positions, config), heads));
    reveal(rotary_component_repr);
}

// @kernel-bridge-begin boundary::tensor_runtime::rotary_embed_repr
#[verifier::opaque]
pub open spec fn rotary_embed_repr(
    pir: IntTensor1D,
    qr: Tensor2D,
    kr: Tensor2D,
    q_head_count: nat,
    kv_head_count: nat,
    head_dim: nat,
    config: RotaryConfigRepr,
)
    -> (Tensor2D, Tensor2D) {
    (rotary_component_repr(pir, qr, q_head_count, head_dim, config),
     rotary_component_repr(pir, kr, kv_head_count, head_dim, config))
}
// @kernel-bridge-end boundary::tensor_runtime::rotary_embed_repr

pub broadcast proof fn lemma_rotary_embed_repr_shape(
    pir: IntTensor1D,
    qr: Tensor2D,
    kr: Tensor2D,
    q_head_count: nat,
    kv_head_count: nat,
    head_dim: nat,
    config: RotaryConfigRepr,
)
    requires pir.len() == qr.len(), qr.len() == kr.len(),
    ensures
        (#[trigger] rotary_embed_repr(
            pir, qr, kr, q_head_count, kv_head_count, head_dim, config,
        )).0.len() == qr.len(),
        rotary_embed_repr(
            pir, qr, kr, q_head_count, kv_head_count, head_dim, config,
        ).1.len() == qr.len(),
{
    reveal(rotary_embed_repr);
    reveal(rotary_component_repr);
    reveal(merge_head_rows_repr);
}

pub proof fn lemma_split_head_rows_repr_shape(
    xr: Tensor2D, head_count: nat, head_dim: nat,
)
    ensures
        #[trigger] split_head_rows_repr(xr, head_count, head_dim).len() as int
            == xr.len() as int * head_count as int,
{
    reveal(split_head_rows_repr);
    let groups = Seq::new(xr.len(), |i: int|
        Seq::new(head_count, |h: int|
            xr[i].subrange(
                h * head_dim as int, (h + 1) * head_dim as int,
            )
        )
    );
    assert forall|j: int| 0 <= j < groups.len() implies
        #[trigger] groups[j].len() == head_count by {}
    crate::proof::tensor::seq_flatten::lemma_fixed_width_flatten_len(groups, head_count);
}

pub proof fn lemma_split_head_rows_projection(
    xr: Tensor2D, head_count: nat, head_dim: nat, i: nat,
)
    requires i < xr.len(),
    ensures
        split_head_rows_repr(seq![xr[i as int]], head_count, head_dim)
            == split_head_rows_repr(xr, head_count, head_dim).subrange(
                i as int * head_count as int,
                (i as int + 1) * head_count as int,
            ),
{
    reveal(split_head_rows_repr);
    let groups = Seq::new(xr.len(), |j: int|
        Seq::new(head_count, |h: int|
            xr[j].subrange(
                h * head_dim as int, (h + 1) * head_dim as int,
            )
        )
    );
    assert forall|j: int| 0 <= j < groups.len() implies
        #[trigger] groups[j].len() == head_count by {}
    assert(split_head_rows_repr(xr, head_count, head_dim) == groups.flatten());
    crate::proof::tensor::seq_flatten::lemma_fixed_width_flatten_group(
        groups, head_count, i as int,
    );
    let single_groups = Seq::new(1, |_j: int|
        Seq::new(head_count, |h: int|
            xr[i as int].subrange(
                h * head_dim as int, (h + 1) * head_dim as int,
            )
        )
    );
    assert(single_groups =~= seq![groups[i as int]]);
    single_groups.lemma_flatten_one_element();
    assert(single_groups.flatten() == groups[i as int]);
    let singleton = seq![xr[i as int]];
    let actual_single_groups = Seq::new(singleton.len(), |j: int|
        Seq::new(head_count, |h: int|
            singleton[j].subrange(
                h * head_dim as int, (h + 1) * head_dim as int,
            )
        )
    );
    assert(actual_single_groups =~= single_groups);
    assert(split_head_rows_repr(
        seq![xr[i as int]], head_count, head_dim,
    ) == actual_single_groups.flatten());
}

pub proof fn lemma_repeat_rows_repr_shape(rows: Tensor2D, copies: nat)
    ensures
        #[trigger] repeat_rows_repr(rows, copies).len() as int
            == rows.len() as int * copies as int,
{
    reveal(repeat_rows_repr);
    let groups = Seq::new(rows.len(), |i: int|
        Seq::new(copies, |_copy: int| rows[i])
    );
    assert forall|j: int| 0 <= j < groups.len() implies
        #[trigger] groups[j].len() == copies by {}
    crate::proof::tensor::seq_flatten::lemma_fixed_width_flatten_len(groups, copies);
}

pub proof fn lemma_repeat_rows_projection(
    rows: Tensor2D, copies: nat, i: nat,
)
    requires i < rows.len(),
    ensures
        repeat_rows_repr(seq![rows[i as int]], copies)
            == repeat_rows_repr(rows, copies).subrange(
                i as int * copies as int,
                (i as int + 1) * copies as int,
            ),
{
    reveal(repeat_rows_repr);
    let groups = Seq::new(rows.len(), |j: int|
        Seq::new(copies, |_copy: int| rows[j])
    );
    assert forall|j: int| 0 <= j < groups.len() implies
        #[trigger] groups[j].len() == copies by {}
    assert(repeat_rows_repr(rows, copies) == groups.flatten());
    crate::proof::tensor::seq_flatten::lemma_fixed_width_flatten_group(
        groups, copies, i as int,
    );
    let single_groups = Seq::new(1, |_j: int|
        Seq::new(copies, |_copy: int| rows[i as int])
    );
    assert(single_groups =~= seq![groups[i as int]]);
    single_groups.lemma_flatten_one_element();
    assert(single_groups.flatten() == groups[i as int]);
    let singleton = seq![rows[i as int]];
    let actual_single_groups = Seq::new(singleton.len(), |j: int|
        Seq::new(copies, |_copy: int| singleton[j])
    );
    assert(actual_single_groups =~= single_groups);
    assert(repeat_rows_repr(seq![rows[i as int]], copies)
        == actual_single_groups.flatten());
}

pub proof fn lemma_rope_table_rows_shape(
    pir: IntTensor1D,
    config: RotaryConfigRepr,
)
    ensures
        #[trigger] rope_cos_rows_repr(pir, config).len() == pir.len(),
        #[trigger] rope_sin_rows_repr(pir, config).len() == pir.len(),
{
    reveal(rope_cos_rows_repr);
    reveal(rope_sin_rows_repr);
}

pub proof fn lemma_rope_table_rows_projection(
    pir: IntTensor1D,
    config: RotaryConfigRepr,
    i: nat,
)
    requires i < pir.len(),
    ensures
        rope_cos_rows_repr(seq![pir[i as int]], config)
            == seq![rope_cos_rows_repr(pir, config)[i as int]],
        rope_sin_rows_repr(seq![pir[i as int]], config)
            == seq![rope_sin_rows_repr(pir, config)[i as int]],
{
    reveal(rope_cos_rows_repr);
    reveal(rope_sin_rows_repr);
    assert(rope_cos_rows_repr(seq![pir[i as int]], config)
        =~= seq![rope_cos_rows_repr(pir, config)[i as int]]);
    assert(rope_sin_rows_repr(seq![pir[i as int]], config)
        =~= seq![rope_sin_rows_repr(pir, config)[i as int]]);
}

#[verifier::spinoff_prover]
pub proof fn lemma_merge_head_rows_projection(
    rows: Tensor2D, token_count: nat, head_count: nat, i: nat,
)
    requires
        rows.len() as int == token_count as int * head_count as int,
        i < token_count,
    ensures
        merge_head_rows_repr(
            rows.subrange(
                i as int * head_count as int,
                (i as int + 1) * head_count as int,
            ),
            1,
            head_count,
        ) == seq![merge_head_rows_repr(rows, token_count, head_count)[i as int]],
{
    reveal(merge_head_rows_repr);
    let lo = i as int * head_count as int;
    let hi = (i as int + 1) * head_count as int;
    assert((i as int) < (token_count as int));
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
    assert(hi <= rows.len() as int) by (nonlinear_arith)
        requires i as int + 1 <= token_count as int,
            0 <= head_count as int,
            hi == (i as int + 1) * head_count as int,
            rows.len() as int == token_count as int * head_count as int,
    {}
    assert(0 <= lo <= hi <= rows.len());
    assert(rows.subrange(lo, hi).len() as int == hi - lo);
    assert(hi - lo == head_count as int) by (nonlinear_arith)
        requires lo == i as int * head_count as int,
            hi == (i as int + 1) * head_count as int,
    {}
    assert(rows.subrange(lo, hi).len() == head_count);
    assert(rows.subrange(lo, hi).subrange(0, head_count as int)
        =~= rows.subrange(lo, hi));
    let lhs = merge_head_rows_repr(rows.subrange(lo, hi), 1, head_count);
    let rhs = seq![merge_head_rows_repr(rows, token_count, head_count)[i as int]];
    assert(lhs.len() == 1);
    assert(rhs.len() == 1);
    assert(lhs[0] == rows.subrange(lo, hi)
        .subrange(0, head_count as int).flatten());
    assert(merge_head_rows_repr(rows, token_count, head_count)[i as int]
        == rows.subrange(lo, hi).flatten());
    assert(lhs[0] == rhs[0]);
    assert(lhs =~= rhs);
}

// `torch.chunk(x, 2, dim=-1)` is a row-local adapter between the one-input
// framework wrapper and the two-input Triton kernel.  On the wrapper's even-
// width domain this is its exact sequence model.  For total spec semantics an
// odd row drops its final unpaired element; the executable wrapper rejects that
// case.  Row projection is therefore proved in Verus rather than hidden in the
// generated kernel certificate.
// @kernel-bridge-begin boundary::tensor_runtime::split_last_axis_half_repr
#[verifier::opaque]
pub open spec fn split_last_axis_half_repr(xr: Tensor2D, right_half: bool) -> Tensor2D {
    Seq::new(xr.len(), |i: int| {
        let row = xr[i];
        let middle = (row.len() / 2) as int;
        let paired_end = 2 * middle;
        if right_half {
            row.subrange(middle, paired_end)
        } else {
            row.subrange(0, middle)
        }
    })
}
// @kernel-bridge-end boundary::tensor_runtime::split_last_axis_half_repr

pub broadcast proof fn lemma_split_last_axis_half_repr_shape(xr: Tensor2D, right_half: bool)
    ensures #[trigger] split_last_axis_half_repr(xr, right_half).len() == xr.len(),
{ reveal(split_last_axis_half_repr); }

// The framework adapter is a genuine rectangular reshape when its input is a
// dense matrix.  The odd-width total-spec case deliberately drops the final
// unpaired element, so each result row has exactly `cols / 2` elements.
pub proof fn lemma_split_last_axis_half_repr_tensor2d_shape(
    xr: Tensor2D,
    rows: nat,
    cols: nat,
    right_half: bool,
)
    requires
        TS::tensor2d_shape(xr, rows, cols),
    ensures
        TS::tensor2d_shape(
            split_last_axis_half_repr(xr, right_half),
            rows,
            cols / 2,
        ),
{
    reveal(split_last_axis_half_repr);
    assert(split_last_axis_half_repr(xr, right_half).len() == rows);
    assert forall|i: int| 0 <= i < rows implies
        (#[trigger] split_last_axis_half_repr(xr, right_half)[i]).len() == cols / 2 by {
        TS::lemma_tensor2d_shape_row(xr, rows, cols, i);
        let middle = (xr[i].len() / 2) as int;
        let paired_end = 2 * middle;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(cols as int, 2);
        vstd::arithmetic::div_mod::lemma_mod_bound(cols as int, 2);
        assert(0 <= middle);
        assert(paired_end <= cols);
        if right_half {
            assert(xr[i].subrange(middle, paired_end).len() == middle);
        } else {
            assert(xr[i].subrange(0, middle).len() == middle);
        }
    }
}

pub broadcast proof fn lemma_split_last_axis_half_projection(
    xr: Tensor2D, right_half: bool, i: nat,
)
    requires i < xr.len(),
    ensures #[trigger] split_last_axis_half_repr(seq![xr[i as int]], right_half)
        == seq![split_last_axis_half_repr(xr, right_half)[i as int]],
{
    reveal(split_last_axis_half_repr);
    assert(split_last_axis_half_repr(seq![xr[i as int]], right_half)
        =~= seq![split_last_axis_half_repr(xr, right_half)[i as int]]);
}

// Abstract semantics of the concrete two-input `silu_mul_kernel`. Physical
// launch tiling is sealed below this boundary and is not part of the engine
// proof's kernel relation.
// @kernel-bridge-begin boundary::tensor_runtime::silu_mul_kernel_repr
#[verifier::opaque]
pub open spec fn silu_mul_kernel_cell_repr(
    x_row: Tensor1D, y_row: Tensor1D, col: int,
) -> Scalar {
    PW::silu_mul_row_output(x_row, y_row)[col]
}

pub open spec fn silu_mul_kernel_repr(xr: Tensor2D, yr: Tensor2D) -> Tensor2D {
    let cols = if xr.len() == 0 { 0 } else { xr[0].len() };
    Seq::new(xr.len(), |i: int| {
        let y_row = if i < yr.len() { yr[i] } else { Seq::empty() };
        Seq::new(cols, |col: int|
            silu_mul_kernel_cell_repr(xr[i], y_row, col))
    })
}
// @kernel-bridge-end boundary::tensor_runtime::silu_mul_kernel_repr

// Raw generated execution preserves the output allocation's exact rectangular
// shape by construction, so this shape theorem is checked rather than trusted.
pub proof fn lemma_silu_mul_kernel_repr_shape(
    xr: Tensor2D,
    yr: Tensor2D,
    rows: nat,
    cols: nat,
)
    requires
        TS::tensor2d_shape(xr, rows, cols),
        TS::tensor2d_shape(yr, rows, cols),
    ensures
        #[trigger] silu_mul_kernel_repr(xr, yr).len() == xr.len(),
        TS::tensor2d_shape(silu_mul_kernel_repr(xr, yr), rows, cols),
{
    reveal(silu_mul_kernel_repr);
    assert(silu_mul_kernel_repr(xr, yr).len() == rows);
    assert forall|i: int| 0 <= i < rows implies
        (#[trigger] silu_mul_kernel_repr(xr, yr)[i]).len() == cols by {
        TS::lemma_tensor2d_shape_row(xr, rows, cols, i);
    }
}

// @kernel-bridge-begin boundary::tensor_runtime::silu_and_mul_repr
#[verifier::opaque]
pub open spec fn silu_and_mul_repr(xr: Tensor2D) -> Tensor2D {
    silu_mul_kernel_repr(
        split_last_axis_half_repr(xr, false),
        split_last_axis_half_repr(xr, true),
    )
}
// @kernel-bridge-end boundary::tensor_runtime::silu_and_mul_repr

pub broadcast proof fn lemma_silu_and_mul_repr_shape(xr: Tensor2D)
    requires
        TS::rectangular(xr),
    ensures
        #[trigger] silu_and_mul_repr(xr).len() == xr.len(),
        TS::rectangular(silu_and_mul_repr(xr)),
{
    let cols = choose|cols: nat| TS::tensor2d_shape(xr, xr.len(), cols);
    assert(TS::tensor2d_shape(xr, xr.len(), cols));
    reveal(silu_and_mul_repr);
    lemma_split_last_axis_half_repr_tensor2d_shape(
        xr, xr.len(), cols, false,
    );
    lemma_split_last_axis_half_repr_tensor2d_shape(
        xr, xr.len(), cols, true,
    );
    lemma_silu_mul_kernel_repr_shape(
        split_last_axis_half_repr(xr, false),
        split_last_axis_half_repr(xr, true),
        xr.len(),
        cols / 2,
    );
    assert(TS::rectangular(silu_and_mul_repr(xr)));
}

// One generated canonical raw row operation underlies every architecture.
// The checked launch binding retains exact geometry and numeric premises;
// this is not a numerical-correctness specification of attention.
// @kernel-bridge-begin boundary::tensor_runtime::paged_attention_repr
#[verifier::opaque]
pub open spec fn paged_attention_repr(
    qr: Tensor2D,
    k_cache_repr: KVCacheLayerRepr, v_cache_repr: KVCacheLayerRepr,
    cu_q_repr: Seq<int>, cu_k_repr: Seq<int>,
    max_seqlen_q: nat, max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
    parameters: AttentionParametersRepr,
) -> Tensor2D {
    crate::boundary::attention_operator::output(qr, k_cache_repr, v_cache_repr,
        cu_q_repr, cu_k_repr, bt_repr, AttentionKind::Full, parameters, 0)
}
// @kernel-bridge-end boundary::tensor_runtime::paged_attention_repr

pub broadcast proof fn lemma_paged_attention_repr_shape(
    qr: Tensor2D,
    k_cache_repr: KVCacheLayerRepr, v_cache_repr: KVCacheLayerRepr,
    cu_q_repr: Seq<int>, cu_k_repr: Seq<int>,
    max_seqlen_q: nat, max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
    parameters: AttentionParametersRepr,
)
    ensures (#[trigger] paged_attention_repr(qr, k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
            parameters,
            )).len() == qr.len(),
        TS::rectangular(paged_attention_repr(qr, k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
            parameters,
            )),
{
    reveal(paged_attention_repr);
    crate::boundary::attention_operator::lemma_shape(qr, k_cache_repr, v_cache_repr,
        cu_q_repr, cu_k_repr, bt_repr, AttentionKind::Full, parameters, 0);
}

// The selected-row analyzer reports finite masked V lanes as an undischarged
// IEEE side condition.  This explicit deployment premise prevents the
// generated import from silently turning that conditional result into an
// unconditional theorem.  It is global because Scalar intentionally erases
// concrete floating-point values from the framework model.
// @kernel-bridge-begin boundary::tensor_runtime::paged_attention_numeric_domain
// Deployment premise imported from the selected-row certificate: every
// `v_block` lane consumed by a reachable deployed paged-attention dot is
// finite. `Scalar` intentionally has no IEEE representation, so Verus cannot
// derive this fact from cache values; top-level observable theorems require it
// explicitly. It says nothing about unused/unallocated cache-pool storage.
pub uninterp spec fn paged_attention_numeric_domain() -> bool;
// @kernel-bridge-end boundary::tensor_runtime::paged_attention_numeric_domain

// Thin aliases to the launch conditions generated from fattn_paged.py's proved
// ContractIR. The logical block-table rows are ragged; Python's trusted
// materializer pads them with valid page zero. Physical head geometry and that
// representation mapping remain runtime/source-scope obligations because the
// ghost tensor types intentionally abstract physical head axes.
// @kernel-bridge-begin boundary::tensor_runtime::paged_attention_metadata_ready
pub open spec fn paged_attention_metadata_header_ready(
    query_rows: nat,
    num_pages: nat,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool {
    GKC::paged_attention_metadata_header_ready(
        query_rows, num_pages, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr,
    )
}

pub open spec fn paged_attention_metadata_rows_ready(
    num_pages: nat,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool {
    GKC::paged_attention_metadata_rows_ready(
        num_pages, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr,
    )
}

#[verifier::opaque]
pub open spec fn paged_attention_metadata_ready(
    query_rows: nat,
    num_pages: nat,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool {
    GKC::paged_attention_metadata_ready(
        query_rows, num_pages, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr,
    )
}
// @kernel-bridge-end boundary::tensor_runtime::paged_attention_metadata_ready

// Package separately established metadata facts back into the opaque launch
// adapter predicate.  Scheduler and graph-cover proofs can reason per row
// without exporting the predicate's quantifiers into their surrounding SMT
// contexts.
pub proof fn lemma_paged_attention_metadata_ready_from_parts(
    query_rows: nat,
    num_pages: nat,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
)
    requires
        paged_attention_metadata_header_ready(
            query_rows, num_pages, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr,
        ),
        paged_attention_metadata_rows_ready(
            num_pages, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr,
        ),
    ensures
        paged_attention_metadata_ready(
            query_rows, num_pages, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr,
        ),
{
    reveal(paged_attention_metadata_ready);
}

// @kernel-bridge-begin boundary::tensor_runtime::paged_cache_geometry
#[verifier::opaque]
pub open spec fn paged_cache_geometry(
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
) -> bool {
    GKC::paged_cache_geometry(k_cache_repr, v_cache_repr)
}
// @kernel-bridge-end boundary::tensor_runtime::paged_cache_geometry

// Unary metadata contract for the deployed KV scatter.  This is the
// framework-side counterpart of store_kv_cache.py's ordinary engine-domain
// slot bounds and injectivity preconditions. Covering graph replay is outside
// this unary launch predicate: its pad suffix is handled by the proved
// no-write-row lemmas below, while every scheduler-produced slot remains in
// this stricter nonnegative domain. Keeping it separate from physical cache
// geometry lets the scheduler prove it once per plan rather than once per
// model layer.
// @kernel-bridge-begin boundary::tensor_runtime::store_kv_cache_metadata_ready
#[verifier::opaque]
pub open spec fn store_kv_cache_metadata_ready(
    row_count: nat,
    num_pages: nat,
    slot_repr: Seq<int>,
) -> bool {
    row_count > 0
    && num_pages > 0
    && slot_repr.len() == row_count
    && slot_repr.no_duplicates()
    && (forall|i: int| 0 <= i < slot_repr.len() ==> {
        let slot = #[trigger] slot_repr[i];
        &&& 0 <= slot
        &&& slot < num_pages as int
            * crate::types::BLOCK_SIZE_SPEC as int
    })
}
// @kernel-bridge-end boundary::tensor_runtime::store_kv_cache_metadata_ready

// Complete representable launch domain for the Verus KV-scatter wrapper.
// Physical head/feature axes and dtype are fixed by the attested deployment
// configuration; the ghost cache model captures page count and page size.
// @kernel-bridge-begin boundary::tensor_runtime::store_kv_cache_launch_ready
#[verifier::opaque]
pub open spec fn store_kv_cache_launch_ready(
    row_count: nat,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
) -> bool {
    paged_cache_geometry(k_cache_repr, v_cache_repr)
    && store_kv_cache_metadata_ready(
        row_count, k_cache_repr.len(), slot_repr,
    )
}
// @kernel-bridge-end boundary::tensor_runtime::store_kv_cache_launch_ready

pub proof fn lemma_store_kv_cache_launch_ready_from_parts(
    row_count: nat,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
)
    requires
        paged_cache_geometry(k_cache_repr, v_cache_repr),
        store_kv_cache_metadata_ready(
            row_count, k_cache_repr.len(), slot_repr,
        ),
    ensures
        store_kv_cache_launch_ready(
            row_count, k_cache_repr, v_cache_repr, slot_repr,
        ),
{
    reveal(store_kv_cache_launch_ready);
}

// @kernel-bridge-begin boundary::tensor_runtime::paged_attention_launch_ready
#[verifier::opaque]
pub open spec fn paged_attention_launch_ready(
    query_rows: nat,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool {
    GKC::paged_attention_launch_ready(
        query_rows, k_cache_repr, v_cache_repr, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr,
    )
}
// @kernel-bridge-end boundary::tensor_runtime::paged_attention_launch_ready

pub proof fn lemma_paged_attention_launch_ready_from_parts(
    query_rows: nat,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
)
    requires
        paged_cache_geometry(k_cache_repr, v_cache_repr),
        paged_attention_metadata_ready(
            query_rows, k_cache_repr.len(), cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr,
        ),
    ensures
        paged_attention_launch_ready(
            query_rows, k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
{
    reveal(paged_cache_geometry);
    reveal(paged_attention_metadata_ready);
    reveal(paged_attention_launch_ready);
}

pub proof fn lemma_paged_attention_launch_ready_parts(
    query_rows: nat,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
)
    requires
        paged_attention_launch_ready(
            query_rows, k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
    ensures
        paged_cache_geometry(k_cache_repr, v_cache_repr),
        paged_attention_metadata_ready(
            query_rows, k_cache_repr.len(), cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr,
        ),
{
    reveal(paged_cache_geometry);
    reveal(paged_attention_metadata_ready);
    reveal(paged_attention_launch_ready);
}

// @kernel-bridge-begin boundary::tensor_runtime::select_sample_logits_repr
// SelectSampleLogits: extract the last-query-token row for batch element `index`.
// Pure spec function — no axiom; the body is computable.
pub open spec fn select_sample_logits_repr(
    logits_repr: Tensor2D, cu_q_repr: Seq<int>, index: nat,
) -> Tensor1D
    recommends index + 1 < cu_q_repr.len(),
               cu_q_repr[index as int + 1] > 0,
               cu_q_repr[index as int + 1] <= logits_repr.len(),
{
    logits_repr[cu_q_repr[index as int + 1] - 1]
}
// @kernel-bridge-end boundary::tensor_runtime::select_sample_logits_repr

// Sampling: pure spec fn; the exec wrapper's ensures binds it to the kernel.
// @kernel-bridge-begin boundary::tensor_runtime::sample_from_repr
pub uninterp spec fn sample_from_repr(r: Tensor1D, state: crate::exec::request_state::SamplerState)
    -> (crate::exec::request_state::SamplerState, TokenId);
// @kernel-bridge-end boundary::tensor_runtime::sample_from_repr

// ---------------------------------------------------------------------------
// Linear kernel.  First end-to-end kernel for the spike: takes two input
// tensors with their perms, returns a fresh output tensor + fresh perm.
//
// Exec body calls into `vosti_kernels.linear` via PyO3.  Verus erases the
// body and trusts the postcondition.
// ---------------------------------------------------------------------------

// `scope` is the ghost set of tensor ids the caller has already seen at
// this point in execution.  Each tensor-producing kernel ensures its
// output is fresh w.r.t. that set; the caller threads `scope.insert(out.id())`
// into the next call.  This gives O(N) cross-call freshness without a
// global ghost counter — Rule 2 in the framing discipline.
//
// External_body kernels are trusted to bind fresh physical Torch allocations
// to ids satisfying the named-input and caller-supplied scope clauses. There
// is currently no one global injectivity theorem over every issued TensorId.
// @kernel-bridge-begin boundary::tensor_runtime::linear
pub fn linear(
    runtime: &ModelFamilyRuntime,
    x: &Tensor,
    weight: &Tensor,
    Tracked(xp): Tracked<&TensorPerm>,
    Tracked(wp): Tracked<&TensorPerm>,
    Ghost(xr): Ghost<Tensor2D>,
    Ghost(wr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        family_runtime_execution_valid(runtime),
        tensor_repr_2d(*xp, *x, xr),
        tensor_repr_2d(*wp, *weight, wr),
    ensures
        ({ let (t, perm) = out;
           // Fresh output: distinct from inputs and from the caller's scope.
           t.id() != x.id() &&
           t.id() != weight.id() &&
           !scope.contains(t.id()) &&
           tensor_repr_2d(perm@, t, linear_repr(xr, wr))
        }),
{
    let out = linear_raw(runtime, x, weight, Tracked(xp), Tracked(wp),
        Ghost(xr), Ghost(wr), Ghost(scope));
    proof {
        let (t, perm) = out;
        let width = choose|width: nat|
            LINEAR::layout(xr, wr, width)
                && tensor_repr_2d(perm@, t, LINEAR::raw_output(xr, wr, width));
        LINEAR::checked_runtime_binding(xr, wr, width);
        reveal(linear_kernel_cell_repr);
        assert forall|r: int| 0 <= r < xr.len() implies
            (#[trigger] linear_repr(xr, wr)[r]) == LINEAR::output(xr, wr)[r] by {
            assert(linear_kernel_row_repr(xr[r], wr) =~= LINEAR::row_output(xr[r], wr));
        };
        assert(linear_repr(xr, wr) =~= LINEAR::output(xr, wr));
    }
    out
}

// A successful physical matmul has compatible rectangular inputs and executes
// the qualified raw kernel with the transposed weight view. These shape and
// representation facts remain the FFI boundary; row-map equality is checked.
#[verifier::external_body]
fn linear_raw(
    runtime: &ModelFamilyRuntime,
    x: &Tensor,
    weight: &Tensor,
    Tracked(xp): Tracked<&TensorPerm>,
    Tracked(wp): Tracked<&TensorPerm>,
    Ghost(xr): Ghost<Tensor2D>,
    Ghost(wr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        family_runtime_execution_valid(runtime),
        tensor_repr_2d(*xp, *x, xr),
        tensor_repr_2d(*wp, *weight, wr),
    ensures
        ({ let (t, perm) = out;
           t.id() != x.id() && t.id() != weight.id() && !scope.contains(t.id()) &&
           exists|width: nat| LINEAR::layout(xr, wr, width)
               && tensor_repr_2d(perm@, t, LINEAR::raw_output(xr, wr, width))
        }),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(|py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            let m = py.import_bound("vosti_kernels")?;
            let f = m.getattr("linear")?;
            let res = f.call1((
                x.inner.bind(py), weight.inner.bind(py),
                primitive_runtime_argument(runtime, py),
            ))?;
            Ok(res.unbind())
        }).expect("python linear kernel failed");
        let t = Tensor { inner: py_handle };
        let perm = Tracked::assume_new();
        (t, perm)
    }
    #[cfg(verus_only)]
    {
        // Body is erased during verification.
        unreachable!()
    }
}
// @kernel-bridge-end boundary::tensor_runtime::linear

// @kernel-bridge-begin boundary::tensor_runtime::qkv_linear
pub fn qkv_linear(
    runtime: &ModelFamilyRuntime,
    x: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    Tracked(xp): Tracked<&TensorPerm>,
    Tracked(qwp): Tracked<&TensorPerm>,
    Tracked(kwp): Tracked<&TensorPerm>,
    Tracked(vwp): Tracked<&TensorPerm>,
    Ghost(xr): Ghost<Tensor2D>,
    Ghost(qwr): Ghost<Tensor2D>,
    Ghost(kwr): Ghost<Tensor2D>,
    Ghost(vwr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (
    (Tensor, Tensor, Tensor),
    (Tracked<TensorPerm>, Tracked<TensorPerm>, Tracked<TensorPerm>),
))
    requires
        family_runtime_execution_valid(runtime),
        tensor_repr_2d(*xp, *x, xr),
        tensor_repr_2d(*qwp, *q_weight, qwr),
        tensor_repr_2d(*kwp, *k_weight, kwr),
        tensor_repr_2d(*vwp, *v_weight, vwr),
    ensures ({
        let ((q, k, v), (qp, kp, vp)) = out;
        let repr = qkv_linear_repr(xr, qwr, kwr, vwr);
        &&& q.id() != k.id()
        &&& q.id() != v.id()
        &&& k.id() != v.id()
        &&& q.id() != x.id() && k.id() != x.id() && v.id() != x.id()
        &&& q.id() != q_weight.id() && q.id() != k_weight.id()
        &&& q.id() != v_weight.id()
        &&& k.id() != q_weight.id() && k.id() != k_weight.id()
        &&& k.id() != v_weight.id()
        &&& v.id() != q_weight.id() && v.id() != k_weight.id()
        &&& v.id() != v_weight.id()
        &&& !scope.contains(q.id())
        &&& !scope.contains(k.id())
        &&& !scope.contains(v.id())
        &&& tensor_repr_2d(qp@, q, repr.0)
        &&& tensor_repr_2d(kp@, k, repr.1)
        &&& tensor_repr_2d(vp@, v, repr.2)
    }),

{
    let out = qkv_linear_raw(runtime, x, q_weight, k_weight, v_weight,
        Tracked(xp), Tracked(qwp), Tracked(kwp), Tracked(vwp),
        Ghost(xr), Ghost(qwr), Ghost(kwr), Ghost(vwr), Ghost(scope));
    proof {
        let ((q, k, v), (qp, kp, vp)) = out;
        let width = choose|width: nat| {
            let repr = QKV::raw_output(xr, qwr, kwr, vwr, width);
            #[trigger] QKV::layout(xr, qwr, kwr, vwr, width)
                && tensor_repr_2d(qp@, q, repr.0)
                && tensor_repr_2d(kp@, k, repr.1)
                && tensor_repr_2d(vp@, v, repr.2)
        };
        QKV::checked_runtime_binding(xr, qwr, kwr, vwr, width);
        reveal(qkv_q_kernel_cell_repr);
        reveal(qkv_k_kernel_cell_repr);
        reveal(qkv_v_kernel_cell_repr);
        let expected = qkv_linear_repr(xr, qwr, kwr, vwr);
        let mapped = QKV::output(xr, qwr, kwr, vwr);
        assert forall|r: int| 0 <= r < xr.len() implies
            (#[trigger] expected.0[r]) == mapped.0[r] by {
            assert(expected.0[r] =~= QKV::row_output(xr[r], qwr, kwr, vwr).0);
        };
        assert(expected.0 =~= mapped.0);
        assert forall|r: int| 0 <= r < xr.len() implies
            (#[trigger] expected.1[r]) == mapped.1[r] by {
            assert(expected.1[r] =~= QKV::row_output(xr[r], qwr, kwr, vwr).1);
        };
        assert(expected.1 =~= mapped.1);
        assert forall|r: int| 0 <= r < xr.len() implies
            (#[trigger] expected.2[r]) == mapped.2[r] by {
            assert(expected.2[r] =~= QKV::row_output(xr[r], qwr, kwr, vwr).2);
        };
        assert(expected.2 =~= mapped.2);
    }
    out
}

// Successful physical execution binds three fresh outputs to the same raw
// fused launch and its checked rectangular geometry, not to a row-map axiom.
#[verifier::external_body]
fn qkv_linear_raw(
    runtime: &ModelFamilyRuntime,
    x: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    Tracked(xp): Tracked<&TensorPerm>,
    Tracked(qwp): Tracked<&TensorPerm>,
    Tracked(kwp): Tracked<&TensorPerm>,
    Tracked(vwp): Tracked<&TensorPerm>,
    Ghost(xr): Ghost<Tensor2D>,
    Ghost(qwr): Ghost<Tensor2D>,
    Ghost(kwr): Ghost<Tensor2D>,
    Ghost(vwr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (
    (Tensor, Tensor, Tensor),
    (Tracked<TensorPerm>, Tracked<TensorPerm>, Tracked<TensorPerm>),
))
    requires
        family_runtime_execution_valid(runtime),
        tensor_repr_2d(*xp, *x, xr),
        tensor_repr_2d(*qwp, *q_weight, qwr),
        tensor_repr_2d(*kwp, *k_weight, kwr),
        tensor_repr_2d(*vwp, *v_weight, vwr),
    ensures ({
        let ((q, k, v), (qp, kp, vp)) = out;
        &&& q.id() != k.id()
        &&& q.id() != v.id()
        &&& k.id() != v.id()
        &&& q.id() != x.id() && k.id() != x.id() && v.id() != x.id()
        &&& q.id() != q_weight.id() && q.id() != k_weight.id()
        &&& q.id() != v_weight.id()
        &&& k.id() != q_weight.id() && k.id() != k_weight.id()
        &&& k.id() != v_weight.id()
        &&& v.id() != q_weight.id() && v.id() != k_weight.id()
        &&& v.id() != v_weight.id()
        &&& !scope.contains(q.id())
        &&& !scope.contains(k.id())
        &&& !scope.contains(v.id())
        &&& exists|width: nat| {
            let repr = QKV::raw_output(xr, qwr, kwr, vwr, width);
            #[trigger] QKV::layout(xr, qwr, kwr, vwr, width)
                && tensor_repr_2d(qp@, q, repr.0)
                && tensor_repr_2d(kp@, k, repr.1)
                && tensor_repr_2d(vp@, v, repr.2)
        }
    }),
{
    #[cfg(not(verus_only))]
    {
        let (q, k, v) = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<(
                pyo3::Py<pyo3::PyAny>,
                pyo3::Py<pyo3::PyAny>,
                pyo3::Py<pyo3::PyAny>,
            )> {
                let module = py.import_bound("vosti_kernels")?;
                let function = module.getattr("qkv_linear")?;
                let result = function.call1((
                    x.inner.bind(py),
                    q_weight.inner.bind(py),
                    k_weight.inner.bind(py),
                    v_weight.inner.bind(py),
                    primitive_runtime_argument(runtime, py),
                ))?;
                let tuple = result.downcast::<pyo3::types::PyTuple>()?;
                Ok((
                    tuple.get_item(0)?.unbind(),
                    tuple.get_item(1)?.unbind(),
                    tuple.get_item(2)?.unbind(),
                ))
            },
        ).expect("python qkv_linear kernel failed");
        (
            (Tensor { inner: q }, Tensor { inner: k }, Tensor { inner: v }),
            (
                Tracked::assume_new(),
                Tracked::assume_new(),
                Tracked::assume_new(),
            ),
        )
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}
// @kernel-bridge-end boundary::tensor_runtime::qkv_linear

// ---------------------------------------------------------------------------
// Elementwise / non-mutating forward kernels.  Each: PyO3 body calls into
// `vosti_kernels.<name>`; postcondition binds output repr to the matching
// pure repr function.  Every output tensor is fresh.
// ---------------------------------------------------------------------------

// @kernel-bridge-begin boundary::tensor_runtime::embed
pub fn embed(
    runtime: &ModelFamilyRuntime,
    input_ids: &Tensor,
    weight: &Tensor,
    Tracked(ip): Tracked<&TensorPerm>,
    Tracked(wp): Tracked<&TensorPerm>,
    Ghost(ir): Ghost<IntTensor1D>,
    Ghost(wr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        family_runtime_execution_valid(runtime),
        int_tensor_repr_1d(*ip, *input_ids, ir),
        tensor_repr_2d(*wp, *weight, wr),
    ensures ({ let (t, perm) = out;
               t.id() != input_ids.id()
               && t.id() != weight.id()
               && !scope.contains(t.id())
               && tensor_repr_2d(perm@, t, embed_repr(ir, wr)) }),
{
    let out = embed_raw(runtime, input_ids, weight, Tracked(ip), Tracked(wp), Ghost(ir), Ghost(wr), Ghost(scope));
    proof {
        EMBED::checked_plain_binding(ir, wr);
        reveal(embed_kernel_cell_repr);
        assert forall|row: int| 0 <= row < ir.len() implies
            (#[trigger] embed_repr(ir, wr)[row]) == EMBED::plain_output(ir, wr)[row] by {
            EMBED::plain_row_shape(ir[row], wr);
            assert(embed_kernel_row_repr(ir[row], wr) =~= EMBED::plain_row_output(ir[row], wr));
        };
        assert(embed_repr(ir, wr) =~= EMBED::plain_output(ir, wr));
    }
    out
}

#[verifier::external_body]
fn embed_raw(
    runtime: &ModelFamilyRuntime,
    input_ids: &Tensor,
    weight: &Tensor,
    Tracked(ip): Tracked<&TensorPerm>,
    Tracked(wp): Tracked<&TensorPerm>,
    Ghost(ir): Ghost<IntTensor1D>,
    Ghost(wr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        family_runtime_execution_valid(runtime),
        int_tensor_repr_1d(*ip, *input_ids, ir),
        tensor_repr_2d(*wp, *weight, wr),
    ensures ({ let (t, perm) = out;
               t.id() != input_ids.id()
               && t.id() != weight.id()
               && !scope.contains(t.id())
               && EMBED::layout(wr)
               && tensor_repr_2d(perm@, t, EMBED::plain_raw_output(ir, wr)) }),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("embed")?;
                let res = f.call1((
                    input_ids.inner.bind(py), weight.inner.bind(py),
                    primitive_runtime_argument(runtime, py),
                ))?;
                Ok(res.unbind())
            }).expect("python embed kernel failed");
        (Tensor { inner: py_handle }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}
// @kernel-bridge-end boundary::tensor_runtime::embed

// @kernel-bridge-begin boundary::tensor_runtime::rms_norm
pub fn rms_norm(
    runtime: &ModelFamilyRuntime,
    x: &Tensor, weight: &Tensor,
    Tracked(xp): Tracked<&TensorPerm>,
    Tracked(wp): Tracked<&TensorPerm>,
    Ghost(xr): Ghost<Tensor2D>,
    Ghost(wr): Ghost<Tensor1D>,
    Ghost(epsilon): Ghost<FloatParameterBits>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        dense_swiglu_runtime_rms_norm_matches(runtime, epsilon),
        tensor_repr_2d(*xp, *x, xr),
        tensor_repr_1d(*wp, *weight, wr),
    ensures ({ let (t, perm) = out;
               t.id() != x.id() && t.id() != weight.id()
               && !scope.contains(t.id())
               && tensor_repr_2d(perm@, t, rms_norm_repr(xr, wr, epsilon)) }),
{
    let out = rms_norm_raw(runtime, x, weight, Tracked(xp), Tracked(wp),
        Ghost(xr), Ghost(wr), Ghost(epsilon), Ghost(scope));
    proof {
        let eps = float_parameter_scalar_repr(epsilon);
        NORM::checked_rms_binding(xr, wr, eps);
        reveal(rms_norm_repr);
        reveal(rms_norm_kernel_cell_repr);
        assert forall|r: int| 0 <= r < xr.len() implies
            (#[trigger] rms_norm_kernel_repr(xr, wr, eps)[r]) == NORM::rms_output(xr, wr, eps)[r] by {
            assert(rms_norm_kernel_row_repr(xr[r], wr, eps) =~= NORM::rms_row_output(xr[r], wr, eps));
        };
        assert(rms_norm_kernel_repr(xr, wr, eps) =~= NORM::rms_output(xr, wr, eps));
    }
    out
}

// Physical layout, scalar-parameter identity and execution correspondence are
// the FFI boundary. The annotation-to-row-map step above is checked.
#[verifier::external_body]
fn rms_norm_raw(
    runtime: &ModelFamilyRuntime,
    x: &Tensor, weight: &Tensor,
    Tracked(xp): Tracked<&TensorPerm>,
    Tracked(wp): Tracked<&TensorPerm>,
    Ghost(xr): Ghost<Tensor2D>,
    Ghost(wr): Ghost<Tensor1D>,
    Ghost(epsilon): Ghost<FloatParameterBits>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        dense_swiglu_runtime_rms_norm_matches(runtime, epsilon),
        tensor_repr_2d(*xp, *x, xr),
        tensor_repr_1d(*wp, *weight, wr),
    ensures ({ let (t, perm) = out;
               t.id() != x.id() && t.id() != weight.id()
               && !scope.contains(t.id())
               && NORM::layout(xr, wr)
               && tensor_repr_2d(perm@, t, NORM::rms_raw_output(
                   xr, wr, float_parameter_scalar_repr(epsilon))) }),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("rms_norm")?;
                let res = f.call1((
                    x.inner.bind(py), weight.inner.bind(py),
                    primitive_runtime_argument(runtime, py),
                ))?;
                Ok(res.unbind())
            }).expect("python rms_norm kernel failed");
        (Tensor { inner: py_handle }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}
// @kernel-bridge-end boundary::tensor_runtime::rms_norm

// @kernel-bridge-begin boundary::tensor_runtime::add_rms_norm
pub fn add_rms_norm(
    runtime: &ModelFamilyRuntime,
    x: &Tensor, residual: &Tensor, weight: &Tensor,
    Tracked(xp): Tracked<&TensorPerm>,
    Tracked(rp): Tracked<&TensorPerm>,
    Tracked(wp): Tracked<&TensorPerm>,
    Ghost(xr): Ghost<Tensor2D>,
    Ghost(rr): Ghost<Tensor2D>,
    Ghost(wr): Ghost<Tensor1D>,
    Ghost(epsilon): Ghost<FloatParameterBits>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: ((Tensor, Tensor), (Tracked<TensorPerm>, Tracked<TensorPerm>)))
    requires dense_swiglu_runtime_rms_norm_matches(runtime, epsilon),
             tensor_repr_2d(*xp, *x, xr),
             tensor_repr_2d(*rp, *residual, rr),
             tensor_repr_1d(*wp, *weight, wr),
             xr.len() == rr.len(),
    ensures ({ let ((normed, res), (np, rp_new)) = out;
               normed.id() != res.id()
               && !scope.contains(normed.id()) && !scope.contains(res.id())
               && tensor_repr_2d(
                   np@, normed, add_rms_norm_repr(xr, rr, wr, epsilon).0,
               )
               && tensor_repr_2d(
                   rp_new@, res, add_rms_norm_repr(xr, rr, wr, epsilon).1,
               ) }),
{
    let out = add_rms_norm_raw(runtime, x, residual, weight,
        Tracked(xp), Tracked(rp), Tracked(wp), Ghost(xr), Ghost(rr), Ghost(wr),
        Ghost(epsilon), Ghost(scope));
    proof {
        let eps = float_parameter_scalar_repr(epsilon);
        NORM::checked_residual_binding(xr, rr, wr, eps);
        reveal(add_rms_norm_repr);
        reveal(add_rms_norm_output_kernel_cell_repr);
        reveal(add_rms_norm_residual_kernel_cell_repr);
        let expected = add_rms_norm_repr(xr, rr, wr, epsilon);
        let mapped = NORM::residual_output(xr, rr, wr, eps);
        assert forall|r: int| 0 <= r < xr.len() implies
            (#[trigger] expected.0[r]) == mapped.0[r] by {
            assert(expected.0[r] =~= NORM::residual_row_output(xr[r], rr[r], wr, eps).0);
        };
        assert(expected.0 =~= mapped.0);
        assert forall|r: int| 0 <= r < xr.len() implies
            (#[trigger] expected.1[r]) == mapped.1[r] by {
            assert(expected.1[r] =~= NORM::residual_row_output(xr[r], rr[r], wr, eps).1);
        };
        assert(expected.1 =~= mapped.1);
    }
    out
}

// Both outputs belong to the same fused raw execution. Neither output is
// equated to an independently implemented add or normalization kernel.
#[verifier::external_body]
fn add_rms_norm_raw(
    runtime: &ModelFamilyRuntime,
    x: &Tensor, residual: &Tensor, weight: &Tensor,
    Tracked(xp): Tracked<&TensorPerm>,
    Tracked(rp): Tracked<&TensorPerm>,
    Tracked(wp): Tracked<&TensorPerm>,
    Ghost(xr): Ghost<Tensor2D>,
    Ghost(rr): Ghost<Tensor2D>,
    Ghost(wr): Ghost<Tensor1D>,
    Ghost(epsilon): Ghost<FloatParameterBits>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: ((Tensor, Tensor), (Tracked<TensorPerm>, Tracked<TensorPerm>)))
    requires dense_swiglu_runtime_rms_norm_matches(runtime, epsilon),
             tensor_repr_2d(*xp, *x, xr),
             tensor_repr_2d(*rp, *residual, rr),
             tensor_repr_1d(*wp, *weight, wr),
             xr.len() == rr.len(),
    ensures ({ let ((normed, res), (np, rp_new)) = out;
               normed.id() != res.id()
               && !scope.contains(normed.id()) && !scope.contains(res.id())
               && NORM::residual_layout(xr, rr, wr)
               && tensor_repr_2d(np@, normed, NORM::residual_raw_output(
                   xr, rr, wr, float_parameter_scalar_repr(epsilon)).0)
               && tensor_repr_2d(rp_new@, res, NORM::residual_raw_output(
                   xr, rr, wr, float_parameter_scalar_repr(epsilon)).1) }),
{
    #[cfg(not(verus_only))]
    {
        let (k_handle, v_handle) = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("add_rms_norm")?;
                let res = f.call1((
                    x.inner.bind(py), residual.inner.bind(py),
                    weight.inner.bind(py), primitive_runtime_argument(runtime, py),
                ))?;
                let tup = res.downcast::<pyo3::types::PyTuple>()?;
                Ok((tup.get_item(0)?.unbind(), tup.get_item(1)?.unbind()))
            }).expect("python add_rms_norm kernel failed");
        ((Tensor { inner: k_handle }, Tensor { inner: v_handle }),
         (Tracked::assume_new(), Tracked::assume_new()))
    }
    #[cfg(verus_only)]
    { unreachable!() }
}
// @kernel-bridge-end boundary::tensor_runtime::add_rms_norm

// @kernel-bridge-begin boundary::tensor_runtime::qk_norm
pub proof fn checked_head_norm_binding(input: Tensor2D, weight: Tensor1D, eps: Scalar)
    requires HEAD::layout(input, weight, false),
    ensures HEAD::raw_output(input, weight, eps, false) == head_rms_norm_kernel_repr(input, weight, eps),
{
    HEAD::checked_binding(input, weight, eps, false);
    reveal(head_rms_norm_kernel_cell_repr);
    assert forall|r: int| 0 <= r < input.len() implies
        (#[trigger] head_rms_norm_kernel_repr(input, weight, eps)[r]) == HEAD::output(input, weight, eps, false)[r] by {
        HEAD::row_shape(input, weight, eps, false, r);
        assert(input[r].len() == head_rms_norm_output_width_repr(input));
        assert(head_rms_norm_kernel_repr(input, weight, eps)[r] =~= HEAD::row_output(input[r], weight, eps, false));
    };
    assert(head_rms_norm_kernel_repr(input, weight, eps) =~= HEAD::output(input, weight, eps, false));
}

pub fn qk_norm(
    runtime: &ModelFamilyRuntime,
    q: &Tensor, k: &Tensor, q_norm_weight: &Tensor, k_norm_weight: &Tensor,
    Tracked(qp): Tracked<&TensorPerm>,
    Tracked(kp): Tracked<&TensorPerm>,
    Tracked(qnp): Tracked<&TensorPerm>,
    Tracked(knp): Tracked<&TensorPerm>,
    Ghost(qr): Ghost<Tensor2D>,
    Ghost(kr): Ghost<Tensor2D>,
    Ghost(qnr): Ghost<Tensor1D>,
    Ghost(knr): Ghost<Tensor1D>,
    Ghost(epsilon): Ghost<FloatParameterBits>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: ((Tensor, Tensor), (Tracked<TensorPerm>, Tracked<TensorPerm>)))
    requires dense_swiglu_runtime_qk_norm_matches(runtime, QkNormKind::RmsNorm),
             dense_swiglu_runtime_rms_norm_matches(runtime, epsilon),
             tensor_repr_2d(*qp, *q, qr),
             tensor_repr_2d(*kp, *k, kr),
             tensor_repr_1d(*qnp, *q_norm_weight, qnr),
             tensor_repr_1d(*knp, *k_norm_weight, knr),
             qr.len() == kr.len(),
    ensures ({ let ((nq, nk), (nqp, nkp)) = out;
               nq.id() != nk.id()
               && !scope.contains(nq.id()) && !scope.contains(nk.id())
               && tensor_repr_2d(
                   nqp@, nq, qk_norm_repr(qr, kr, qnr, knr, epsilon).0,
               )
               && tensor_repr_2d(
                   nkp@, nk, qk_norm_repr(qr, kr, qnr, knr, epsilon).1,
               ) }),
{
    let out = qk_norm_raw(runtime, q, k, q_norm_weight, k_norm_weight,
        Tracked(qp), Tracked(kp), Tracked(qnp), Tracked(knp),
        Ghost(qr), Ghost(kr), Ghost(qnr), Ghost(knr), Ghost(epsilon), Ghost(scope));
    proof {
        let eps = float_parameter_scalar_repr(epsilon);
        checked_head_norm_binding(qr, qnr, eps);
        checked_head_norm_binding(kr, knr, eps);
        reveal(qk_norm_repr);
    }
    out
}

#[verifier::external_body]
fn qk_norm_raw(
    runtime: &ModelFamilyRuntime,
    q: &Tensor, k: &Tensor, q_norm_weight: &Tensor, k_norm_weight: &Tensor,
    Tracked(qp): Tracked<&TensorPerm>,
    Tracked(kp): Tracked<&TensorPerm>,
    Tracked(qnp): Tracked<&TensorPerm>,
    Tracked(knp): Tracked<&TensorPerm>,
    Ghost(qr): Ghost<Tensor2D>,
    Ghost(kr): Ghost<Tensor2D>,
    Ghost(qnr): Ghost<Tensor1D>,
    Ghost(knr): Ghost<Tensor1D>,
    Ghost(epsilon): Ghost<FloatParameterBits>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: ((Tensor, Tensor), (Tracked<TensorPerm>, Tracked<TensorPerm>)))
    requires dense_swiglu_runtime_qk_norm_matches(runtime, QkNormKind::RmsNorm),
             dense_swiglu_runtime_rms_norm_matches(runtime, epsilon),
             tensor_repr_2d(*qp, *q, qr),
             tensor_repr_2d(*kp, *k, kr),
             tensor_repr_1d(*qnp, *q_norm_weight, qnr),
             tensor_repr_1d(*knp, *k_norm_weight, knr),
             qr.len() == kr.len(),
    ensures ({ let ((nq, nk), (nqp, nkp)) = out;
               nq.id() != nk.id()
               && !scope.contains(nq.id()) && !scope.contains(nk.id())
               && HEAD::layout(qr, qnr, false) && HEAD::layout(kr, knr, false)
               && tensor_repr_2d(
                   nqp@, nq, HEAD::raw_output(qr, qnr, float_parameter_scalar_repr(epsilon), false),
               )
               && tensor_repr_2d(
                   nkp@, nk, HEAD::raw_output(kr, knr, float_parameter_scalar_repr(epsilon), false),
               ) }),
{
    #[cfg(not(verus_only))]
    {
        let (a, b) = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("qk_norm")?;
                let res = f.call1((
                    q.inner.bind(py), k.inner.bind(py),
                    q_norm_weight.inner.bind(py), k_norm_weight.inner.bind(py),
                    primitive_runtime_argument(runtime, py),
                ))?;
                let tup = res.downcast::<pyo3::types::PyTuple>()?;
                Ok((tup.get_item(0)?.unbind(), tup.get_item(1)?.unbind()))
            }).expect("python qk_norm kernel failed");
        ((Tensor { inner: a }, Tensor { inner: b }),
         (Tracked::assume_new(), Tracked::assume_new()))
    }
    #[cfg(verus_only)]
    { unreachable!() }
}
// @kernel-bridge-end boundary::tensor_runtime::qk_norm

// Executable identity branch for a composition with Q/K normalization
// disabled.  It consumes and returns the existing tensor permissions instead
// of allocating output tensors or crossing a kernel boundary.  Consequently
// this path has no launch key and no certificate obligation.
pub fn bypass_qk_norm(
    q: Tensor,
    k: Tensor,
    Tracked(qp): Tracked<TensorPerm>,
    Tracked(kp): Tracked<TensorPerm>,
    Ghost(qr): Ghost<Tensor2D>,
    Ghost(kr): Ghost<Tensor2D>,
    Ghost(epsilon): Ghost<FloatParameterBits>,
) -> (out: ((Tensor, Tensor), (Tracked<TensorPerm>, Tracked<TensorPerm>)))
    requires
        tensor_repr_2d(qp, q, qr),
        tensor_repr_2d(kp, k, kr),
    ensures ({
        let ((nq, nk), (nqp, nkp)) = out;
        &&& nq.id() == q.id()
        &&& nk.id() == k.id()
        &&& tensor_repr_2d(nqp@, nq, apply_qk_norm_repr(
            qr, kr, QkNormWeightsRepr::Disabled, epsilon,
        ).0)
        &&& tensor_repr_2d(nkp@, nk, apply_qk_norm_repr(
            qr, kr, QkNormWeightsRepr::Disabled, epsilon,
        ).1)
    }),
{
    proof { reveal(apply_qk_norm_repr); }
    ((q, k), (Tracked(qp), Tracked(kp)))
}

// @kernel-bridge-begin boundary::tensor_runtime::view_as_kv
#[verifier::external_body]
pub fn view_as_kv(
    runtime: &ModelFamilyRuntime,
    v: &Tensor,
    Tracked(vp): Tracked<&TensorPerm>,
    Ghost(vr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        family_runtime_execution_valid(runtime),
        tensor_repr_2d(*vp, *v, vr),
    ensures ({ let (t, perm) = out;
               t.id() != v.id()
               && !scope.contains(t.id())
               && tensor_repr_2d(perm@, t, view_as_kv_repr(vr)) }),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("view_as_kv")?;
                let res = f.call1((
                    v.inner.bind(py), primitive_runtime_argument(runtime, py),
                ))?;
                Ok(res.unbind())
            }).expect("python view_as_kv kernel failed");
        (Tensor { inner: py_handle }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}
// @kernel-bridge-end boundary::tensor_runtime::view_as_kv

// Pure layout adapters shared by every dense family. Their numerical content
// is unchanged; fresh copies make the tensor-permission framing explicit.
// @kernel-bridge-begin boundary::tensor_runtime::row_layout_adapters
#[verifier::external_body]
pub fn merge_attention_heads(
    input: &Tensor,
    Tracked(ip): Tracked<&TensorPerm>,
    Ghost(input_repr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires tensor_repr_2d(*ip, *input, input_repr),
    ensures ({ let (tensor, perm) = out;
        tensor.id() != input.id()
            && !scope.contains(tensor.id())
            && tensor_repr_2d(
                perm@, tensor, DLP::merge_attention_heads_repr(input_repr),
            )
    }),
{
    #[cfg(not(verus_only))]
    {
        let inner = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let module = py.import_bound("vosti_kernels")?;
                Ok(module
                    .getattr("merge_attention_heads")?
                    .call1((input.inner.bind(py),))?
                    .unbind())
            },
        )
        .expect("python merge_attention_heads adapter failed");
        (Tensor { inner }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

#[verifier::external_body]
pub fn split_last_axis_halves(
    input: &Tensor,
    Tracked(ip): Tracked<&TensorPerm>,
    Ghost(input_repr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: ((Tensor, Tensor), (Tracked<TensorPerm>, Tracked<TensorPerm>)))
    requires
        tensor_repr_2d(*ip, *input, input_repr),
        TS::tensor2d_even_width(input_repr),
    ensures ({ let ((left, right), (left_perm, right_perm)) = out;
        &&& left.id() != right.id()
        &&& left.id() != input.id()
        &&& right.id() != input.id()
        &&& !scope.contains(left.id())
        &&& !scope.contains(right.id())
        &&& tensor_repr_2d(
            left_perm@, left, split_last_axis_half_repr(input_repr, false),
        )
        &&& tensor_repr_2d(
            right_perm@, right, split_last_axis_half_repr(input_repr, true),
        )
    }),
{
    #[cfg(not(verus_only))]
    {
        let (left, right) = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<(
                pyo3::Py<pyo3::PyAny>,
                pyo3::Py<pyo3::PyAny>,
            )> {
                let module = py.import_bound("vosti_kernels")?;
                let result = module
                    .getattr("split_last_axis_halves")?
                    .call1((input.inner.bind(py),))?;
                let tuple = result.downcast::<pyo3::types::PyTuple>()?;
                Ok((tuple.get_item(0)?.unbind(), tuple.get_item(1)?.unbind()))
            },
        )
        .expect("python split_last_axis_halves adapter failed");
        (
            (Tensor { inner: left }, Tensor { inner: right }),
            (Tracked::assume_new(), Tracked::assume_new()),
        )
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}
// @kernel-bridge-end boundary::tensor_runtime::row_layout_adapters

// @kernel-bridge-begin boundary::tensor_runtime::silu_and_mul
pub fn silu_and_mul(
    runtime: &ModelFamilyRuntime,
    x: &Tensor,
    Tracked(xp): Tracked<&TensorPerm>,
    Ghost(xr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        family_runtime_execution_valid(runtime),
        tensor_repr_2d(*xp, *x, xr),
        TS::tensor2d_even_width(xr),
    ensures ({ let (t, perm) = out;
               t.id() != x.id()
               && !scope.contains(t.id())
               && tensor_repr_2d(perm@, t, silu_and_mul_repr(xr)) }),
{
    let out = silu_and_mul_raw(runtime, x, Tracked(xp), Ghost(xr), Ghost(scope));
    proof {
        let a = split_last_axis_half_repr(xr, false);
        let b = split_last_axis_half_repr(xr, true);
        PW::checked_silu_mul_binding(a, b);
        reveal(silu_and_mul_repr);
        reveal(silu_mul_kernel_cell_repr);
        reveal(silu_mul_kernel_repr);
        assert forall|r: int| 0 <= r < a.len() implies
            (#[trigger] silu_mul_kernel_repr(a, b)[r]) == PW::silu_mul_output(a, b)[r] by {
            assert(silu_mul_kernel_repr(a, b)[r] =~= PW::silu_mul_row_output(a[r], b[r]));
        };
        assert(PW::silu_mul_output(a, b) =~= silu_mul_kernel_repr(a, b));
    }
    out
}

#[verifier::external_body]
fn silu_and_mul_raw(
    runtime: &ModelFamilyRuntime,
    x: &Tensor,
    Tracked(xp): Tracked<&TensorPerm>,
    Ghost(xr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        family_runtime_execution_valid(runtime),
        tensor_repr_2d(*xp, *x, xr),
        TS::tensor2d_even_width(xr),
    ensures ({ let (t, perm) = out;
               t.id() != x.id()
               && !scope.contains(t.id())
               && PW::binary_layout(split_last_axis_half_repr(xr, false), split_last_axis_half_repr(xr, true))
               && tensor_repr_2d(perm@, t, PW::silu_mul_raw_output(
                   split_last_axis_half_repr(xr, false), split_last_axis_half_repr(xr, true))) }),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("silu_and_mul")?;
                let res = f.call1((
                    x.inner.bind(py), primitive_runtime_argument(runtime, py),
                ))?;
                Ok(res.unbind())
            }).expect("python silu_and_mul kernel failed");
        (Tensor { inner: py_handle }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}
// @kernel-bridge-end boundary::tensor_runtime::silu_and_mul

// ---------------------------------------------------------------------------
// Tensor materialization helper used by the self-contained engine tests to
// hand a Python tensor in from Rust without going through a kernel.
// `from_pylist_2d(rows)` builds a fresh tensor from a list-of-lists Python
// literal; the perm's repr is uninterpreted (caller asserts a Ghost value).
// ---------------------------------------------------------------------------

#[verifier::external_body]
pub fn from_pylist_2d(
    rows: usize,
    cols: usize,
    flat_data: &[f32],
    Ghost(repr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        flat_data.len() == rows * cols,
        repr.len() == rows,
    ensures
        ({ let (t, perm) = out;
           !scope.contains(t.id())
           && tensor_repr_2d(perm@, t, repr)
        }),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(|py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            let m = py.import_bound("vosti_kernels")?;
            let f = m.getattr("from_flat")?;
            let data: Vec<f32> = flat_data.to_vec();
            let res = f.call1((rows, cols, data))?;
            Ok(res.unbind())
        }).expect("python from_flat failed");
        let t = Tensor { inner: py_handle };
        let perm = Tracked::assume_new();
        (t, perm)
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

// Family-owned physical weight bundles are defined under
// `boundary/model_families/<family>/weights.rs`. This neutral module retains
// only the closed facade and its shared permission authority.
// @kernel-bridge-begin boundary::tensor_runtime::model_weights_contract

// Closed executable model sum.  The engine stores this facade and branches
// only at the model-forward boundary; scheduler and refinement state remain
// architecture-neutral.
pub enum ModelWeights {
    Qwen3(Qwen3ModelWeights),
    Llama3(Llama3ModelWeights),
    Gemma3Text(Gemma3ModelWeights),
    Gemma4Text(Gemma4ModelWeights),
}

pub open spec fn model_weights_architecture(w: &ModelWeights) -> ModelArchitecture {
    match w {
        ModelWeights::Qwen3(_) => ModelArchitecture::Qwen3,
        ModelWeights::Llama3(_) => ModelArchitecture::Llama3,
        ModelWeights::Gemma3Text(_) => ModelArchitecture::Gemma3Text,
        ModelWeights::Gemma4Text(_) => ModelArchitecture::Gemma4Text,
    }
}

pub open spec fn model_weights_num_layers(w: &ModelWeights) -> nat {
    match w {
        ModelWeights::Qwen3(qwen) => qwen.layers.len() as nat,
        ModelWeights::Llama3(llama) => llama.layers.len() as nat,
        ModelWeights::Gemma3Text(gemma) => gemma.layers.len() as nat,
        ModelWeights::Gemma4Text(gemma) => gemma.layers.len() as nat,
    }
}

#[verifier::external_body]
pub tracked struct ModelWeightsPerms {
    _no_copy: NoCopy,
}

impl ModelWeightsPerms {
    pub uninterp spec fn architecture(&self) -> ModelArchitecture;
    pub uninterp spec fn num_layers(&self) -> nat;
    pub uninterp spec fn qwen3_config(&self) -> Qwen3ModelWeightsExtensionRepr;
    pub uninterp spec fn llama3_config(&self) -> Llama3ModelWeightsExtensionRepr;
    pub uninterp spec fn gemma3_config(&self) -> Gemma3ModelWeightsExtensionRepr;
    pub uninterp spec fn gemma4_config(&self) -> Gemma4ModelWeightsExtensionRepr;
    pub uninterp spec fn four_norm_gated_weights(&self)
        -> crate::boundary::four_norm_gated_weights::FourNormGatedModelWeightsPerms;
    // Read-only access to the explicit inner record bound by the checkpoint
    // validator. Per-layer borrows from that record are checked map borrows.
    #[verifier::external_body]
    pub proof fn tracked_borrow_four_norm_gated_weights<'a>(tracked &'a self)
        -> (tracked out: &'a crate::boundary::four_norm_gated_weights::FourNormGatedModelWeightsPerms)
        requires self.architecture() == ModelArchitecture::Gemma4Text,
        ensures *out == self.four_norm_gated_weights(),
    {
        unreachable!()
    }
    // Per-layer perm projection (proof-only).
    pub uninterp spec fn qwen3_layer(&self, i: int) -> Qwen3LayerWeightsPerms;
    pub uninterp spec fn llama3_layer(&self, i: int) -> Llama3LayerWeightsPerms;
    pub uninterp spec fn gemma3_layer(&self, i: int) -> Gemma3LayerWeightsPerms;
    pub uninterp spec fn gemma3_attention_kind(&self, i: int) -> AttentionKind;
    pub uninterp spec fn gemma3_hidden_size(&self) -> nat;
    pub uninterp spec fn gemma3_sliding_window(&self) -> nat;

    // Closed projection for the physical layer representation shared by
    // Qwen3 and Llama3. Four-norm models use a different layer contract and
    // are outside the executable domain of this projection.
    pub open spec fn dense_swiglu_layer(
        &self,
        i: int,
    ) -> crate::boundary::dense_swiglu_decoder::DenseSwiGluLayerWeightsPerms {
        match self.architecture() {
            ModelArchitecture::Qwen3 => self.qwen3_layer(i),
            ModelArchitecture::Llama3 => self.llama3_layer(i),
            ModelArchitecture::Gemma3Text => self.qwen3_layer(i),
            ModelArchitecture::Gemma4Text => self.qwen3_layer(i),
        }
    }
    pub uninterp spec fn embed_weight_id(&self) -> TensorId;
    pub uninterp spec fn final_norm_id(&self) -> TensorId;
    pub uninterp spec fn lm_head_id(&self) -> TensorId;
    pub uninterp spec fn embed_weight_repr(&self) -> Tensor2D;
    pub uninterp spec fn final_norm_repr(&self) -> Tensor1D;
    pub uninterp spec fn lm_head_repr(&self) -> Tensor2D;

    // Borrow a tracked reference to layer i's weight perms.  The borrow is
    // scoped to `&self`'s lifetime — caller can pass `Tracked(&p)` to
    // kernels expecting `Tracked<&Qwen3LayerWeightsPerms>`. Pure read; no
    // extracted-set bookkeeping (immutable).
    #[verifier::external_body]
    pub proof fn tracked_borrow_qwen3_layer<'a>(
        tracked &'a self,
        i: int,
    ) -> (tracked out: &'a Qwen3LayerWeightsPerms)
        requires 0 <= i < self.num_layers() as int,
        ensures *out == self.qwen3_layer(i),
    {
        unreachable!()
    }

    #[verifier::external_body]
    pub proof fn tracked_borrow_llama3_layer<'a>(
        tracked &'a self,
        i: int,
    ) -> (tracked out: &'a Llama3LayerWeightsPerms)
        requires
            self.architecture() == ModelArchitecture::Llama3,
            0 <= i < self.num_layers() as int,
        ensures *out == self.llama3_layer(i),
    {
        unreachable!()
    }

    pub proof fn tracked_borrow_dense_swiglu_layer<'a>(
        tracked &'a self,
        i: int,
    ) -> (tracked out: &'a crate::boundary::dense_swiglu_decoder::DenseSwiGluLayerWeightsPerms)
        requires
            self.architecture() == ModelArchitecture::Qwen3
                || self.architecture() == ModelArchitecture::Llama3,
            0 <= i < self.num_layers() as int,
        ensures *out == self.dense_swiglu_layer(i),
    {
        if self.architecture() == ModelArchitecture::Llama3 {
            self.tracked_borrow_llama3_layer(i)
        } else {
            self.tracked_borrow_qwen3_layer(i)
        }
    }

    // Architecture-specific layer borrows remain disjoint at the type level:
    // A family forward cannot accidentally call a kernel with another
    // family's bundle.
    #[verifier::external_body]
    pub proof fn tracked_borrow_gemma3_layer<'a>(
        tracked &'a self,
        i: int,
    ) -> (tracked out: &'a Gemma3LayerWeightsPerms)
        requires
            self.architecture() == ModelArchitecture::Gemma3Text,
            0 <= i < self.num_layers() as int,
        ensures *out == self.gemma3_layer(i),
    {
        unreachable!()
    }

    // Borrow the token-embedding matrix permission.
    #[verifier::external_body]
    pub proof fn tracked_borrow_embed_weight<'a>(
        tracked &'a self,
    ) -> (tracked out: &'a TensorPerm)
        ensures
            out.id() == self.embed_weight_id(),
            out.repr_2d() == self.embed_weight_repr(),
    {
        unreachable!()
    }

    // Borrow the final_norm weight perm.
    #[verifier::external_body]
    pub proof fn tracked_borrow_final_norm<'a>(
        tracked &'a self,
    ) -> (tracked out: &'a TensorPerm)
        ensures
            out.id() == self.final_norm_id(),
            out.repr_1d() == self.final_norm_repr(),
    {
        unreachable!()
    }

    // Borrow the lm_head weight perm.
    #[verifier::external_body]
    pub proof fn tracked_borrow_lm_head<'a>(
        tracked &'a self,
    ) -> (tracked out: &'a TensorPerm)
        ensures
            out.id() == self.lm_head_id(),
            out.repr_2d() == self.lm_head_repr(),
    {
        unreachable!()
    }
}

pub open spec fn model_weights_bound(w: &ModelWeights, perms: &ModelWeightsPerms) -> bool {
    perms.architecture() == model_weights_architecture(w)
    && match w {
        ModelWeights::Qwen3(qwen) => QWEN3::weights::model_weights_bound(qwen, perms),
        ModelWeights::Llama3(llama) => LLAMA3::weights::model_weights_bound(llama, perms),
        ModelWeights::Gemma3Text(gemma) => GEMMA3::weights::model_weights_bound(gemma, perms),
        ModelWeights::Gemma4Text(gemma) => GEMMA4::weights::model_weights_bound(gemma, perms),
    }
}

pub open spec fn qwen3_model_weights_bound(
    weights: &Qwen3ModelWeights,
    perms: &ModelWeightsPerms,
) -> bool {
    QWEN3::weights::model_weights_bound(weights, perms)
}

pub open spec fn llama3_model_weights_bound(
    weights: &Llama3ModelWeights,
    perms: &ModelWeightsPerms,
) -> bool {
    LLAMA3::weights::model_weights_bound(weights, perms)
}

pub open spec fn gemma3_model_weights_bound(
    weights: &Gemma3ModelWeights,
    perms: &ModelWeightsPerms,
) -> bool {
    GEMMA3::weights::model_weights_bound(weights, perms)
}

// Exact architecture payload associated with a permission bundle.  This is a
// narrow model-forward input, not part of the common scheduler proof state.
pub closed spec fn model_weights_architecture_repr_of(
    perms: &ModelWeightsPerms,
) -> ModelWeightsArchitectureRepr {
    match perms.architecture() {
        ModelArchitecture::Gemma4Text => ModelWeightsArchitectureRepr::Gemma4Text(
            GEMMA4::weights_extension_repr_of(perms)),
        ModelArchitecture::Qwen3 => ModelWeightsArchitectureRepr::Qwen3(
            QWEN3::weights_extension_repr_of(perms),
        ),
        ModelArchitecture::Llama3 => ModelWeightsArchitectureRepr::Llama3(
            LLAMA3::weights_extension_repr_of(perms),
        ),
        ModelArchitecture::Gemma3Text => ModelWeightsArchitectureRepr::Gemma3Text(
            GEMMA3::weights_extension_repr_of(perms),
        ),
    }
}

pub proof fn lemma_model_weights_architecture_repr_projection(
    perms: &ModelWeightsPerms,
)
    ensures
        model_weights_architecture_repr_of(perms) == match perms.architecture() {
            ModelArchitecture::Gemma4Text => ModelWeightsArchitectureRepr::Gemma4Text(
                GEMMA4::weights_extension_repr_of(perms)),
            ModelArchitecture::Qwen3 => ModelWeightsArchitectureRepr::Qwen3(
                QWEN3::weights_extension_repr_of(perms),
            ),
            ModelArchitecture::Llama3 => ModelWeightsArchitectureRepr::Llama3(
                LLAMA3::weights_extension_repr_of(perms),
            ),
            ModelArchitecture::Gemma3Text => {
                ModelWeightsArchitectureRepr::Gemma3Text(
                    GEMMA3::weights_extension_repr_of(perms),
                )
            },
        },
{
    reveal(model_weights_architecture_repr_of);
}

// Keep family layer-permission choices out of architecture-neutral scheduler
// queries. Family adapters expose checked reductions at their forward boundary.
pub closed spec fn model_weights_common_layers_repr_of(
    perms: &ModelWeightsPerms,
) -> Seq<LayerWeightsRepr> {
    match perms.architecture() {
        ModelArchitecture::Gemma4Text => GEMMA4::weights::common_layers_repr_of(perms),
        ModelArchitecture::Qwen3 => QWEN3::weights::common_layers_repr_of(perms),
        ModelArchitecture::Llama3 => LLAMA3::weights::common_layers_repr_of(perms),
        ModelArchitecture::Gemma3Text => {
            GEMMA3::weights::common_layers_repr_of(perms)
        },
    }
}

pub proof fn lemma_model_weights_common_layers_repr_projection(
    perms: &ModelWeightsPerms,
)
    ensures
        model_weights_common_layers_repr_of(perms)
            == match perms.architecture() {
                ModelArchitecture::Gemma4Text => GEMMA4::weights::common_layers_repr_of(perms),
                ModelArchitecture::Qwen3 => {
                    QWEN3::weights::common_layers_repr_of(perms)
                },
                ModelArchitecture::Llama3 => {
                    LLAMA3::weights::common_layers_repr_of(perms)
                },
                ModelArchitecture::Gemma3Text => {
                    GEMMA3::weights::common_layers_repr_of(perms)
                },
            },
{
    reveal(model_weights_common_layers_repr_of);
}

pub open spec fn model_weights_repr_of(perms: &ModelWeightsPerms) -> ModelWeightsRepr {
    ModelWeightsRepr {
        architecture: perms.architecture(),
        embed_weight: perms.embed_weight_repr(),
        layers: model_weights_common_layers_repr_of(perms),
        final_norm: perms.final_norm_repr(),
        lm_head: perms.lm_head_repr(),
    }
}

pub proof fn lemma_model_weights_architecture_repr_valid(
    weights: &ModelWeights,
    runtime: &ModelRuntime,
    perms: &ModelWeightsPerms,
)
    requires model_execution_valid(weights, runtime, perms),
    ensures
        model_weights_architecture_repr_valid(
            model_weights_repr_of(perms), model_weights_architecture_repr_of(perms),
        ),
        model_weights_repr_of(perms).layers.len() == perms.num_layers(),
{
    reveal(model_execution_valid);
    lemma_model_weights_architecture_repr_projection(perms);
    lemma_model_weights_common_layers_repr_projection(perms);
    match (weights, runtime) {
        (ModelWeights::Qwen3(qwen), ModelRuntime::Qwen3(_)) => {
            reveal(qwen3_model_weights_bound);
            QWEN3::lemma_bound_architecture_repr_valid(qwen, perms);
        },
        (ModelWeights::Llama3(llama), ModelRuntime::Llama3(_)) => {
            reveal(llama3_model_weights_bound);
            LLAMA3::lemma_bound_architecture_repr_valid(llama, perms);
        },
        (ModelWeights::Gemma3Text(gemma), ModelRuntime::Gemma3Text(_)) => {
            reveal(gemma3_model_weights_bound);
            GEMMA3::lemma_weights_extension_repr(perms);
        },
        (ModelWeights::Gemma4Text(gemma), ModelRuntime::Gemma4Text(_)) => {
            GEMMA4::lemma_bound_architecture_repr_valid(gemma, perms);
        },
        _ => { assert(false); },
    }
}

// @kernel-bridge-end boundary::tensor_runtime::model_weights_contract

// The scheduler needs no model-family state: every supported family supplies
// its bound embedding tensor as the physical placement anchor for per-step
// metadata.
pub fn model_weights_device_anchor(weights: &ModelWeights) -> &Tensor {
    match weights {
        ModelWeights::Qwen3(qwen) => &qwen.embed_weight,
        ModelWeights::Llama3(llama) => &llama.embed_weight,
        ModelWeights::Gemma3Text(gemma) => &gemma.embed_weight,
        ModelWeights::Gemma4Text(gemma) => &gemma.embed_weight,
    }
}

// Release serving always places step metadata beside the bound embedding.
// Debug builds retain the zero-layer CPU scheduler fixture, so they omit a
// non-CUDA anchor rather than asking the production materializer to accept one.
// Device placement is absent from the logical Tensor representation; the
// Verus body therefore returns the proof-irrelevant fallback directly.
pub fn model_step_plan_device_anchor(weights: &ModelWeights) -> Option<&Tensor> {
    #[cfg(not(verus_only))]
    {
        let anchor = model_weights_device_anchor(weights);
        if !cfg!(debug_assertions) {
            return Some(anchor);
        }
        let is_cuda = pyo3::Python::with_gil(|py| {
            anchor
                .inner
                .bind(py)
                .getattr("is_cuda")
                .and_then(|value| value.extract::<bool>())
        })
        .expect("inspect model step-plan device anchor");
        if is_cuda { Some(anchor) } else { None }
    }
    #[cfg(verus_only)]
    {
        None
    }
}

// Checked architecture dispatcher over the family-local trusted binders.
// `model_weights_bound` is the common representation contract; executable
// qualification is tracked separately by `model_execution_valid`.
pub fn bind_model_weights_perms(
    weights: &ModelWeights,
    expected_num_layers: usize,
) -> (out: Tracked<ModelWeightsPerms>)
    requires
        model_weights_num_layers(weights) == expected_num_layers,
    ensures
        model_weights_bound(weights, &out@),
{
    match weights {
        ModelWeights::Qwen3(qwen) => {
            let out = QWEN3::weights::bind_model_weights_perms(
                qwen, expected_num_layers,
            );
            proof {
                reveal(model_weights_bound);
            }
            out
        },
        ModelWeights::Llama3(llama) => {
            let out = LLAMA3::weights::bind_model_weights_perms(
                llama, expected_num_layers,
            );
            proof {
                reveal(model_weights_bound);
            }
            out
        },
        ModelWeights::Gemma3Text(gemma) => {
            let out = GEMMA3::weights::bind_model_weights_perms(
                gemma, expected_num_layers,
            );
            proof {
                reveal(model_weights_bound);
            }
            out
        },
        ModelWeights::Gemma4Text(gemma) => {
            GEMMA4::weights::bind_model_weights_perms(gemma, expected_num_layers)
        },
    }
}

// ===========================================================================
// Tensor materialization kernels — build fresh tensors from exec values.
// Ghost reprs are open spec functions over the input sequence so the proof
// can compute them precisely; no axiom needed.
// ===========================================================================

// @kernel-bridge-begin boundary::tensor_runtime::step_plan_materializers
pub open spec fn u64_seq_to_int_repr(v: Seq<u64>) -> IntTensor1D {
    Seq::new(v.len(), |i: int| v[i] as int)
}

pub open spec fn nested_u64_seq_to_block_repr(v: Seq<Seq<u64>>) -> Seq<Seq<BlockId>> {
    v
}

#[verifier::external_body]
pub fn token_tensor(
    tokens: Vec<u64>,
    device_anchor: Option<&Tensor>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    ensures !scope.contains(out.0.id())
            && int_tensor_repr_1d(out.1@, out.0, u64_seq_to_int_repr(tokens@)),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("token_tensor")?;
                let v: Vec<u64> = tokens.clone();
                let anchor = device_anchor.map(|tensor| tensor.inner.bind(py));
                let res = f.call1((v, anchor))?;
                Ok(res.unbind())
            }).expect("python token_tensor kernel failed");
        (Tensor { inner: py_handle }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}

#[verifier::external_body]
pub fn position_tensor(
    positions: Vec<u64>,
    device_anchor: Option<&Tensor>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    ensures !scope.contains(out.0.id())
            && int_tensor_repr_1d(out.1@, out.0, u64_seq_to_int_repr(positions@)),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("position_tensor")?;
                let v: Vec<u64> = positions.clone();
                let anchor = device_anchor.map(|tensor| tensor.inner.bind(py));
                let res = f.call1((v, anchor))?;
                Ok(res.unbind())
            }).expect("python position_tensor kernel failed");
        (Tensor { inner: py_handle }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}

#[verifier::external_body]
pub fn slot_tensor(
    slot_mapping: Vec<u64>,
    device_anchor: Option<&Tensor>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    ensures !scope.contains(out.0.id())
            && int_tensor_repr_1d(out.1@, out.0, u64_seq_to_int_repr(slot_mapping@)),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("slot_tensor")?;
                let v: Vec<u64> = slot_mapping.clone();
                let anchor = device_anchor.map(|tensor| tensor.inner.bind(py));
                let res = f.call1((v, anchor))?;
                Ok(res.unbind())
            }).expect("python slot_tensor kernel failed");
        (Tensor { inner: py_handle }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}

#[verifier::external_body]
pub fn block_tables_tensor(
    block_ids: Vec<Vec<u64>>,
    device_anchor: Option<&Tensor>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    ensures !scope.contains(out.0.id())
            && block_table_repr(out.1@, out.0,
                nested_u64_seq_to_block_repr(Seq::new(block_ids@.len(), |i: int| block_ids@[i]@))),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("block_tables_tensor")?;
                let cloned: Vec<Vec<u64>> = block_ids.iter().cloned().collect();
                let anchor = device_anchor.map(|tensor| tensor.inner.bind(py));
                let res = f.call1((cloned, anchor))?;
                Ok(res.unbind())
            }).expect("python block_tables_tensor kernel failed");
        (Tensor { inner: py_handle }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}

#[verifier::external_body]
pub fn seq_lens_tensor(
    lengths: Vec<u64>,
    device_anchor: Option<&Tensor>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    ensures !scope.contains(out.0.id())
            && int_tensor_repr_1d(out.1@, out.0, u64_seq_to_int_repr(lengths@)),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("seq_lens_tensor")?;
                let v: Vec<u64> = lengths.clone();
                let anchor = device_anchor.map(|tensor| tensor.inner.bind(py));
                let res = f.call1((v, anchor))?;
                Ok(res.unbind())
            }).expect("python seq_lens_tensor kernel failed");
        (Tensor { inner: py_handle }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}
// @kernel-bridge-end boundary::tensor_runtime::step_plan_materializers

// ===========================================================================
// KV cache initialization — multi-tensor allocator.
//
// Returns N (k_cache, v_cache) tensor pairs and a tracked `KVCachePerms`
// collection holding the 2N permissions.  Pairwise id-distinctness is
// expressed as a single `kv_perms_ids_distinct(...)` predicate (Rule 3 in
// docs/architecture.md "Framing discipline"), not as O(N²) inline facts.
//
// `KVCachePerms` is opaque to callers — they query it via `k_id(i)`,
// `v_id(i)`, `k_repr(i)`, `v_repr(i)` spec accessors and use the paired
// `tracked_take_layer` / `tracked_put_layer` proof fns to extract a
// per-layer `TensorPerm` for kernel mutation, then return it.  An
// `extracted: Set<int>` ghost prevents double-take.  No public field,
// no `Clone`/`Copy` (it's `tracked`).
// ===========================================================================

#[verifier::external_body]
pub tracked struct KVCachePerms {
    _no_copy: NoCopy,
}

impl KVCachePerms {
    pub uninterp spec fn len(&self) -> nat;
    pub uninterp spec fn k_id(&self, i: int) -> TensorId;
    pub uninterp spec fn v_id(&self, i: int) -> TensorId;
    pub uninterp spec fn k_repr(&self, i: int) -> KVCacheLayerRepr;
    pub uninterp spec fn v_repr(&self, i: int) -> KVCacheLayerRepr;

    // Set of layer indices whose perms have been extracted via
    // `tracked_take_layer` and not yet returned via `tracked_put_layer`.
    // This is the discipline that prevents double-take from minting
    // duplicate `TensorPerm`s for the same id.
    pub uninterp spec fn extracted(&self) -> Set<int>;

    // Take ownership of layer i's (k_perm, v_perm) for a mutating kernel
    // call.  The `!extracted` precondition prevents double-take (which
    // would mint a duplicate `TensorPerm` with the same id — unsound).
    // Caller must `tracked_put_layer(i, k_new, v_new)` before the
    // function exit (postcondition typically `extracted == empty`).
    #[verifier::external_body]
    pub proof fn tracked_take_layer(
        tracked &mut self,
        i: int,
    ) -> (tracked out: (TensorPerm, TensorPerm))
        requires
            0 <= i < old(self).len() as int,
            !old(self).extracted().contains(i),
            kv_perms_ids_distinct(*old(self)),
        ensures
            out.0.id() == old(self).k_id(i),
            out.0.kv_cache_repr() == old(self).k_repr(i),
            out.1.id() == old(self).v_id(i),
            out.1.kv_cache_repr() == old(self).v_repr(i),
            out.0.id() != out.1.id(),
            final(self).len() == old(self).len(),
            final(self).extracted() == old(self).extracted().insert(i),
            forall|j: int| 0 <= j < final(self).len() as int ==>
                #[trigger] final(self).k_id(j) == old(self).k_id(j)
                && final(self).v_id(j) == old(self).v_id(j),
            forall|j: int| 0 <= j < final(self).len() as int && j != i ==>
                #[trigger] final(self).k_repr(j) == old(self).k_repr(j)
                && final(self).v_repr(j) == old(self).v_repr(j),
            kv_perms_ids_distinct(*final(self)),
    {
        unreachable!()
    }

    // Return ownership of layer i's perms after a mutating kernel call.
    // The caller's `k`/`v` perms must still claim the original ids
    // (kernel-mutating signatures preserve `perm.id()` by construction);
    // this is enforced via the precondition.  After put, layer i's
    // reprs reflect the new perms.
    #[verifier::external_body]
    pub proof fn tracked_put_layer(
        tracked &mut self,
        i: int,
        tracked k: TensorPerm,
        tracked v: TensorPerm,
    )
        requires
            0 <= i < old(self).len() as int,
            old(self).extracted().contains(i),
            k.id() == old(self).k_id(i),
            v.id() == old(self).v_id(i),
        ensures
            final(self).len() == old(self).len(),
            final(self).extracted() == old(self).extracted().remove(i),
            final(self).k_id(i) == old(self).k_id(i),
            final(self).v_id(i) == old(self).v_id(i),
            final(self).k_repr(i) == k.kv_cache_repr(),
            final(self).v_repr(i) == v.kv_cache_repr(),
            forall|j: int| 0 <= j < final(self).len() as int && j != i ==>
                #[trigger] final(self).k_id(j) == old(self).k_id(j)
                && final(self).v_id(j) == old(self).v_id(j)
                && final(self).k_repr(j) == old(self).k_repr(j)
                && final(self).v_repr(j) == old(self).v_repr(j),
            kv_perms_ids_distinct(*final(self)),
    {
        unreachable!()
    }
}

// Runtime tensor handles and their owned permissions must identify the same
// K/V cache pair at every layer.  Naming this constructor-boundary condition
// keeps the raw quantified formula out of higher-level initialization proofs.
pub open spec fn kv_cache_tensor_ids_match(
    kv_caches: Seq<(Tensor, Tensor)>,
    kv_perms: KVCachePerms,
    num_layers: nat,
) -> bool {
    forall|i: int| 0 <= i < num_layers as int ==>
        #[trigger] kv_caches[i].0.id() == kv_perms.k_id(i)
        && kv_caches[i].1.id() == kv_perms.v_id(i)
}

pub proof fn lemma_kv_cache_tensor_ids_match_at(
    kv_caches: Seq<(Tensor, Tensor)>,
    kv_perms: KVCachePerms,
    num_layers: nat,
    i: int,
)
    requires
        kv_cache_tensor_ids_match(kv_caches, kv_perms, num_layers),
        num_layers <= kv_caches.len(),
        num_layers <= kv_perms.len(),
        0 <= i < num_layers as int,
    ensures
        kv_caches[i].0.id() == kv_perms.k_id(i),
        kv_caches[i].1.id() == kv_perms.v_id(i),
{
    reveal(kv_cache_tensor_ids_match);
    assert(kv_caches[i].0.id() == kv_perms.k_id(i));
    assert(kv_caches[i].1.id() == kv_perms.v_id(i));
}

pub proof fn lemma_kv_cache_tensor_ids_match_from_pointwise(
    kv_caches: Seq<(Tensor, Tensor)>,
    kv_perms: KVCachePerms,
    num_layers: nat,
)
    requires
        forall|i: int| 0 <= i < num_layers as int ==>
            #[trigger] kv_caches[i].0.id() == kv_perms.k_id(i)
            && kv_caches[i].1.id() == kv_perms.v_id(i),
    ensures
        kv_cache_tensor_ids_match(kv_caches, kv_perms, num_layers),
{
    reveal(kv_cache_tensor_ids_match);
}

// Distinctness of all 2N tensor ids in a `KVCachePerms` collection.  Single
// named predicate; Z3 instantiates pairs only when a caller's proof requires
// a specific inequality (typical: at most one or two per call site).
// @kernel-bridge-begin boundary::tensor_runtime::init_kv_caches_contract
pub open spec fn kv_perms_ids_distinct(perms: KVCachePerms) -> bool {
    forall|i: int, j: int|
        0 <= i < perms.len() as int && 0 <= j < perms.len() as int && i != j
        ==> #[trigger] perms.k_id(i) != #[trigger] perms.k_id(j)
            && perms.v_id(i) != perms.v_id(j)
            && perms.k_id(i) != perms.v_id(j)
}

// Per-layer KV cache shape: paged layout with `blocks_needed_for(capacity)`
// pages of size `BLOCK_SIZE_SPEC`, expressed as one O(N) `forall|i|` fact.
pub open spec fn kv_perms_page_shape(perms: KVCachePerms, num_pages: nat) -> bool {
    forall|i: int| 0 <= i < perms.len() as int ==>
        (#[trigger] perms.k_repr(i)).len() == num_pages
        && perms.v_repr(i).len() == num_pages
        && (forall|p: int| 0 <= p < perms.k_repr(i).len() ==>
            (#[trigger] perms.k_repr(i)[p]).len()
                == crate::types::BLOCK_SIZE_SPEC as int)
        && (forall|p: int| 0 <= p < perms.v_repr(i).len() ==>
            (#[trigger] perms.v_repr(i)[p]).len()
                == crate::types::BLOCK_SIZE_SPEC as int)
}

pub open spec fn kv_perms_initial_shape(perms: KVCachePerms, capacity: nat) -> bool {
    kv_perms_page_shape(perms, crate::proof::tensor::geometry::blocks_needed_for(capacity))
}

// Allocator.
//
// Postcondition is intentionally O(N) (one `forall|i|` for ids, one for
// shape) plus a single `kv_perms_ids_distinct` fact.  No pairwise enumeration.
#[verifier::external_body]
pub fn init_kv_caches(
    num_layers: usize,
    token_capacity: usize,
) -> (out: (Vec<(Tensor, Tensor)>, Tracked<KVCachePerms>))
    ensures
        ({ let (caches, perms) = out;
           caches.len() == num_layers as nat
           && perms@.len() == num_layers as nat
           && kv_perms_ids_distinct(perms@)
           && kv_perms_initial_shape(perms@, token_capacity as nat)
           && perms@.extracted() == Set::<int>::empty()
           && (forall|i: int| 0 <= i < num_layers as int ==>
                 (#[trigger] caches[i].0).id() == perms@.k_id(i)
                 && caches[i].1.id() == perms@.v_id(i)
                 && caches[i].0.id() != caches[i].1.id())
        }),
{
    #[cfg(not(verus_only))]
    {
        let py_caches: Vec<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<Vec<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)>> {
                let m = py.import_bound("vosti_kernels")?;
                let f = m.getattr("init_kv_caches")?;
                let res = f.call1((num_layers, token_capacity))?;
                // Re-run the source-attested validator at the PyO3 boundary.
                // This still runs even if the public package attribute was
                // replaced dynamically with another allocator.
                let implementation = py.import_bound("vosti_kernels.kernels")?;
                implementation
                    .getattr("_validate_init_kv_caches_runtime_contract")?
                    .call1((&res, num_layers, token_capacity))?;
                let list: &pyo3::Bound<pyo3::types::PyList> = res.downcast::<pyo3::types::PyList>()?;
                if list.len() != num_layers {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "init_kv_caches returned the wrong number of layers",
                    ));
                }
                let mut out = Vec::with_capacity(num_layers);
                for item in list.iter() {
                    let tup: &pyo3::Bound<pyo3::types::PyTuple> =
                        item.downcast::<pyo3::types::PyTuple>()?;
                    if tup.len() != 2 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "init_kv_caches returned a non-pair layer",
                        ));
                    }
                    let k = tup.get_item(0)?.unbind();
                    let v = tup.get_item(1)?.unbind();
                    out.push((k, v));
                }
                Ok(out)
            }).expect("python init_kv_caches failed");
        let caches: Vec<(Tensor, Tensor)> = py_caches.into_iter()
            .map(|(k, v)| (Tensor { inner: k }, Tensor { inner: v }))
            .collect();
        let perms: Tracked<KVCachePerms> = Tracked::assume_new();
        (caches, perms)
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

// Nonempty dense models project cache geometry from the weight roles common
// to every admitted family. The Python helper derives KV heads, head width,
// dtype, and device from these tensors and performs the one shared allocation.
#[verifier::external_body]
fn init_layer_model_kv_caches_raw(
    embed_weight: &Tensor,
    k_projections: &Vec<&Tensor>,
    head_dims: &Vec<usize>,
    token_capacity: usize,
) -> (out: (Vec<(Tensor, Tensor)>, Tracked<KVCachePerms>))
    requires k_projections.len() > 0, head_dims.len() == k_projections.len(),
    ensures
        ({ let (caches, perms) = out;
           caches.len() == k_projections.len()
           && perms@.len() == k_projections.len()
           && kv_perms_ids_distinct(perms@)
           && kv_perms_initial_shape(perms@, token_capacity as nat)
           && perms@.extracted() == Set::<int>::empty()
           && (forall|i: int| 0 <= i < k_projections.len() as int ==>
                 (#[trigger] caches[i].0).id() == perms@.k_id(i)
                 && caches[i].1.id() == perms@.v_id(i)
                 && caches[i].0.id() != caches[i].1.id())
        }),
{
    #[cfg(not(verus_only))]
    {
        let py_caches: Vec<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)> =
            pyo3::Python::with_gil(
                |py| -> pyo3::PyResult<Vec<(
                    pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>,
                )>> {
                    let implementation = py.import_bound("vosti_kernels.physical")?;
                    let projections = pyo3::types::PyTuple::new_bound(py,
                        k_projections.iter().map(|tensor| tensor.inner.bind(py)));
                    let dimensions = pyo3::types::PyTuple::new_bound(py, head_dims);
                    let num_layers = k_projections.len();
                    let res = implementation.getattr("init_layer_model_kv_caches")?.call1((
                        embed_weight.inner.bind(py),
                        &projections,
                        &dimensions,
                        token_capacity,
                    ))?;
                    implementation.getattr("validate_layer_model_kv_caches")?.call1((
                        &res,
                        embed_weight.inner.bind(py),
                        &projections,
                        &dimensions,
                        token_capacity,
                    ))?;
                    let list = res.downcast::<pyo3::types::PyList>()?;
                    if list.len() != num_layers {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "model KV-cache allocator returned the wrong number of layers",
                        ));
                    }
                    let mut out = Vec::with_capacity(num_layers);
                    for item in list.iter() {
                        let tuple = item.downcast::<pyo3::types::PyTuple>()?;
                        if tuple.len() != 2 {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "model KV-cache allocator returned a non-pair layer",
                            ));
                        }
                        out.push((
                            tuple.get_item(0)?.unbind(),
                            tuple.get_item(1)?.unbind(),
                        ));
                    }
                    Ok(out)
                },
            )
            .expect("python model KV-cache allocation failed");
        let caches = py_caches
            .into_iter()
            .map(|(k, v)| (Tensor { inner: k }, Tensor { inner: v }))
            .collect();
        let perms: Tracked<KVCachePerms> = Tracked::assume_new();
        (caches, perms)
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

// Verified strengthening of the raw allocator's pointwise ID postcondition to
// the architecture-neutral cache-bundle predicate consumed by the engine.
pub fn init_layer_model_kv_caches(
    embed_weight: &Tensor,
    k_projections: &Vec<&Tensor>,
    head_dims: &Vec<usize>,
    token_capacity: usize,
) -> (out: (Vec<(Tensor, Tensor)>, Tracked<KVCachePerms>))
    requires k_projections.len() > 0, head_dims.len() == k_projections.len(),
    ensures
        ({ let (caches, perms) = out;
           caches.len() == k_projections.len()
           && perms@.len() == k_projections.len()
           && kv_perms_ids_distinct(perms@)
           && kv_perms_initial_shape(perms@, token_capacity as nat)
           && perms@.extracted() == Set::<int>::empty()
           && kv_cache_tensor_ids_match(caches@, perms@, k_projections.len() as nat)
        }),
{
    let out = init_layer_model_kv_caches_raw(
        embed_weight, k_projections, head_dims, token_capacity,
    );
    proof {
        reveal(kv_cache_tensor_ids_match);
    }
    out
}

// Uniform dense models are a checked broadcast of one immutable geometry.
// The allocation and independent return-value validation are shared with
// heterogeneous models; this adds no allocator trust boundary.
fn init_nonempty_model_kv_caches(
    num_layers: usize, embed_weight: &Tensor, k_proj: &Tensor,
    head_dim: usize, token_capacity: usize,
) -> (out: (Vec<(Tensor, Tensor)>, Tracked<KVCachePerms>))
    requires num_layers > 0,
    ensures ({ let (caches, perms) = out;
        &&& caches.len() == num_layers
        &&& perms@.len() == num_layers
        &&& kv_perms_ids_distinct(perms@)
        &&& kv_perms_initial_shape(perms@, token_capacity as nat)
        &&& perms@.extracted() == Set::<int>::empty()
        &&& kv_cache_tensor_ids_match(caches@, perms@, num_layers as nat)
    }),
{
    let mut projections: Vec<&Tensor> = Vec::new();
    let mut dimensions: Vec<usize> = Vec::new();
    let mut i = 0usize;
    while i < num_layers
        invariant i <= num_layers, projections.len() == i, dimensions.len() == i,
        decreases num_layers - i,
    {
        projections.push(k_proj);
        dimensions.push(head_dim);
        i += 1;
    }
    init_layer_model_kv_caches(embed_weight, &projections, &dimensions, token_capacity)
}

// Architecture-dispatched cache allocation.  This is the constructor callers
// should use when they already own a closed model facade.  The zero-layer
// branch remains useful to architecture-neutral tests and needs no physical
// family geometry.
pub fn init_model_kv_caches(
    weights: &ModelWeights,
    token_capacity: usize,
) -> (out: (Vec<(Tensor, Tensor)>, Tracked<KVCachePerms>))
    ensures
        ({ let (caches, perms) = out;
           caches.len() == model_weights_num_layers(weights)
           && perms@.len() == model_weights_num_layers(weights)
           && kv_perms_ids_distinct(perms@)
           && kv_perms_initial_shape(perms@, token_capacity as nat)
           && perms@.extracted() == Set::<int>::empty()
           && kv_cache_tensor_ids_match(
                caches@, perms@, model_weights_num_layers(weights),
              )
        }),
{
    reveal(kv_cache_tensor_ids_match);
    reveal(model_weights_num_layers);
    match weights {
        ModelWeights::Gemma4Text(gemma) => {
            if gemma.layers.len() == 0 {
                init_kv_caches(0, token_capacity)
            } else {
                GEMMA4::deployment::init_model_kv_caches(gemma, token_capacity)
            }
        },
        ModelWeights::Qwen3(qwen) => {
            if qwen.layers.len() == 0 {
                init_kv_caches(0, token_capacity)
            } else {
                init_nonempty_model_kv_caches(
                    qwen.layers.len(),
                    &qwen.embed_weight,
                    &qwen.layers[0].k_proj,
                    qwen.config.geometry.head_dim,
                    token_capacity,
                )
            }
        },
        ModelWeights::Llama3(llama) => {
            if llama.layers.len() == 0 {
                init_kv_caches(0, token_capacity)
            } else {
                init_nonempty_model_kv_caches(
                    llama.layers.len(),
                    &llama.embed_weight,
                    &llama.layers[0].k_proj,
                    llama.config.geometry.head_dim,
                    token_capacity,
                )
            }
        },
        ModelWeights::Gemma3Text(gemma) => {
            if gemma.layers.len() == 0 {
                init_kv_caches(0, token_capacity)
            } else {
                init_nonempty_model_kv_caches(
                    gemma.layers.len(),
                    &gemma.embed_weight,
                    &gemma.layers[0].k_proj,
                    gemma.config.geometry.head_dim,
                    token_capacity,
                )
            }
        },
    }
}
// @kernel-bridge-end boundary::tensor_runtime::init_kv_caches_contract

// ===========================================================================
// store_kv_cache — first mutating kernel.
//
// Logical scatter model: each row of (k, v) is written into
// the cache at `slot_mapping[row]`, and every other slot is unchanged.  The
// total model chooses last-write-wins for repeated slots, but every deployed
// call requires an injective mapping, so repeated-slot behavior is outside the
// trusted engine domain.
//
// The pure functional model `store_kv_cache_repr` is `closed spec fn` —
// callers must `reveal(store_kv_cache_repr)` in focused proof blocks; the
// derived lemmas in `batch_invariance` (e.g.
// `store_kv_cache_repr_preserves_unwritten_slots`) hide the body.
// ===========================================================================

// Storing zero rows is the identity (base case of the fold below).
// Exported for the empty-plan case of `engine.step`'s verification.
pub proof fn lemma_store_kv_cache_repr_empty_rows(
    kr: Tensor2D, vr: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_repr: Seq<int>,
)
    requires kr.len() == 0,
    ensures store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr) == (old_k, old_v),
{ }

// @kernel-bridge-begin boundary::tensor_runtime::store_kv_cache_repr
pub closed spec fn store_kv_cache_repr(
    kr: Tensor2D, vr: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_repr: Seq<int>,
) -> (KVCacheLayerRepr, KVCacheLayerRepr)
    recommends kr.len() == vr.len(), vr.len() == slot_repr.len(),
    decreases kr.len(),
{
    if kr.len() == 0 {
        (old_k, old_v)
    } else {
        let mid = store_kv_cache_repr(
            kr.subrange(0, kr.len() as int - 1),
            vr.subrange(0, vr.len() as int - 1),
            old_k, old_v,
            slot_repr.subrange(0, slot_repr.len() as int - 1));
        let s = slot_repr[slot_repr.len() - 1];
        let page = s / crate::types::BLOCK_SIZE_SPEC as int;
        let offset = s % crate::types::BLOCK_SIZE_SPEC as int;
        if 0 <= page < mid.0.len() && 0 <= offset < mid.0[page].len()
           && 0 <= page < mid.1.len() && 0 <= offset < mid.1[page].len()
        {
            (mid.0.update(page,
                          mid.0[page].update(offset, kr[kr.len() - 1])),
             mid.1.update(page,
                          mid.1[page].update(offset, vr[vr.len() - 1])))
        } else {
            mid
        }
    }
}
// @kernel-bridge-end boundary::tensor_runtime::store_kv_cache_repr

// A suffix of covering-graph pad rows uses slot -1. The total scatter model
// already drops out-of-range stores; this lemma exposes that those rows are an
// exact cache identity, independent of their computed K/V values.
pub proof fn lemma_store_kv_cache_repr_no_write_rows(
    kr: Tensor2D, vr: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_repr: Seq<int>,
)
    requires
        kr.len() == vr.len(),
        vr.len() == slot_repr.len(),
        forall|i: int| 0 <= i < slot_repr.len() ==>
            #[trigger] slot_repr[i] == -1,
    ensures
        store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr)
            == (old_k, old_v),
    decreases kr.len(),
{
    reveal(store_kv_cache_repr);
    if kr.len() > 0 {
        let n = kr.len() as int;
        let kr0 = kr.subrange(0, n - 1);
        let vr0 = vr.subrange(0, n - 1);
        let slots0 = slot_repr.subrange(0, n - 1);
        assert forall|i: int| 0 <= i < slots0.len() implies
            #[trigger] slots0[i] == -1 by {
            assert(slots0[i] == slot_repr[i]);
        }
        lemma_store_kv_cache_repr_no_write_rows(
            kr0, vr0, old_k, old_v, slots0,
        );
        let s = slot_repr[slot_repr.len() - 1];
        assert(s == -1);
        let bs = crate::types::BLOCK_SIZE_SPEC as int;
        assert(bs == 64);
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(s, bs);
        vstd::arithmetic::div_mod::lemma_mod_bound(s, bs);
        assert(s / bs < 0);
    }
}

// Appending no-write rows to a real scatter leaves the complete physical cache
// equal to the real scatter. This is the KV half of covering replay refinement.
pub proof fn lemma_store_kv_cache_repr_append_no_write_rows(
    kr_real: Tensor2D, vr_real: Tensor2D,
    kr_pad: Tensor2D, vr_pad: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slots_real: Seq<int>, slots_pad: Seq<int>,
)
    requires
        kr_real.len() == vr_real.len(),
        vr_real.len() == slots_real.len(),
        kr_pad.len() == vr_pad.len(),
        vr_pad.len() == slots_pad.len(),
        forall|i: int| 0 <= i < slots_pad.len() ==>
            #[trigger] slots_pad[i] == -1,
    ensures
        store_kv_cache_repr(
            kr_real + kr_pad, vr_real + vr_pad,
            old_k, old_v, slots_real + slots_pad,
        ) == store_kv_cache_repr(
            kr_real, vr_real, old_k, old_v, slots_real,
        ),
{
    store_kv_cache_repr_concat(
        kr_real, vr_real, kr_pad, vr_pad,
        old_k, old_v, slots_real, slots_pad,
    );
    let base = store_kv_cache_repr(
        kr_real, vr_real, old_k, old_v, slots_real,
    );
    lemma_store_kv_cache_repr_no_write_rows(
        kr_pad, vr_pad, base.0, base.1, slots_pad,
    );
}

// StoreKVCacheRepr locality: cache rows at slot indices NOT in `slot_repr`
// are preserved by the scatter write.  Lives here (same module as the
// `closed spec fn`) so the proof can `reveal` the body.
pub proof fn store_kv_cache_repr_preserves_unwritten_slots(
    kr: Tensor2D, vr: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    s: nat,
)
    requires
        kr.len() == vr.len(), vr.len() == slot_repr.len(),
        !slot_repr.contains(s as int),
        crate::proof::tensor::geometry::slot_in_cache(old_k, s),
        crate::proof::tensor::geometry::slot_in_cache(old_v, s),
    ensures ({
        let new_caches = store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr);
        crate::proof::tensor::geometry::slot_in_cache(new_caches.0, s) &&
        crate::proof::tensor::geometry::slot_in_cache(new_caches.1, s) &&
        crate::proof::tensor::geometry::cache_at(new_caches.0, s) == crate::proof::tensor::geometry::cache_at(old_k, s) &&
        crate::proof::tensor::geometry::cache_at(new_caches.1, s) == crate::proof::tensor::geometry::cache_at(old_v, s)
    }),
    decreases kr.len(),
{
    reveal(store_kv_cache_repr);
    if kr.len() == 0 {
    } else {
        // Last write at slot_repr[n-1] is not `s`.
        assert(slot_repr[slot_repr.len() - 1] != s as int);
        // Subrange doesn't contain s either.
        assert(!slot_repr.subrange(0, slot_repr.len() as int - 1).contains(s as int)) by {
            if slot_repr.subrange(0, slot_repr.len() as int - 1).contains(s as int) {
                let i = slot_repr.subrange(0, slot_repr.len() as int - 1).index_of(s as int);
                assert(slot_repr[i] == s as int);
            }
        }
        store_kv_cache_repr_preserves_unwritten_slots(
            kr.subrange(0, kr.len() as int - 1),
            vr.subrange(0, vr.len() as int - 1),
            old_k, old_v,
            slot_repr.subrange(0, slot_repr.len() as int - 1),
            s);
    }
}


// Store composition: scattering the concatenation `A ++ B` equals scattering
// `B` on top of the result of scattering `A`.  The algebraic core for KV-cache
// disjointness reasoning (combined with `preserves_unwritten_slots`, it shows
// that appending writes that miss a slot `s` leaves `s` unchanged).  Proved by
// induction peeling the last element of `B`.
pub proof fn store_kv_cache_repr_concat(
    kr_a: Tensor2D, vr_a: Tensor2D,
    kr_b: Tensor2D, vr_b: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_a: Seq<int>, slot_b: Seq<int>,
)
    requires
        kr_a.len() == vr_a.len(), vr_a.len() == slot_a.len(),
        kr_b.len() == vr_b.len(), vr_b.len() == slot_b.len(),
    ensures
        store_kv_cache_repr(kr_a + kr_b, vr_a + vr_b, old_k, old_v, slot_a + slot_b)
        == store_kv_cache_repr(kr_b, vr_b,
              store_kv_cache_repr(kr_a, vr_a, old_k, old_v, slot_a).0,
              store_kv_cache_repr(kr_a, vr_a, old_k, old_v, slot_a).1,
              slot_b),
    decreases kr_b.len(),
{
    reveal(store_kv_cache_repr);
    if kr_b.len() == 0 {
        assert(kr_a + kr_b =~= kr_a);
        assert(vr_a + vr_b =~= vr_a);
        assert(slot_a + slot_b =~= slot_a);
    } else {
        let nb = kr_b.len() as int;
        let kr_b1 = kr_b.subrange(0, nb - 1);
        let vr_b1 = vr_b.subrange(0, nb - 1);
        let slot_b1 = slot_b.subrange(0, nb - 1);
        assert((kr_a + kr_b).subrange(0, (kr_a + kr_b).len() as int - 1) =~= kr_a + kr_b1);
        assert((vr_a + vr_b).subrange(0, (vr_a + vr_b).len() as int - 1) =~= vr_a + vr_b1);
        assert((slot_a + slot_b).subrange(0, (slot_a + slot_b).len() as int - 1)
            =~= slot_a + slot_b1);
        assert((kr_a + kr_b)[(kr_a + kr_b).len() - 1] == kr_b[nb - 1]);
        assert((vr_a + vr_b)[(vr_a + vr_b).len() - 1] == vr_b[nb - 1]);
        assert((slot_a + slot_b)[(slot_a + slot_b).len() - 1] == slot_b[nb - 1]);
        store_kv_cache_repr_concat(kr_a, vr_a, kr_b1, vr_b1, old_k, old_v, slot_a, slot_b1);
    }
}

// Disjointness corollary: appending a block of writes `B` whose slots all miss
// `s` leaves `s` equal to its value after only the `A` writes.  This is the form
// used to discharge cross-request KV-cache agreement: other requests' writes
// (disjoint blocks) don't change the slots a given request reads.
pub proof fn store_kv_cache_repr_append_misses_slot(
    kr_a: Tensor2D, vr_a: Tensor2D,
    kr_b: Tensor2D, vr_b: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_a: Seq<int>, slot_b: Seq<int>,
    s: nat,
)
    requires
        kr_a.len() == vr_a.len(), vr_a.len() == slot_a.len(),
        kr_b.len() == vr_b.len(), vr_b.len() == slot_b.len(),
        !slot_b.contains(s as int),
        crate::proof::tensor::geometry::slot_in_cache(
            store_kv_cache_repr(kr_a, vr_a, old_k, old_v, slot_a).0, s),
        crate::proof::tensor::geometry::slot_in_cache(
            store_kv_cache_repr(kr_a, vr_a, old_k, old_v, slot_a).1, s),
    ensures ({
        let full = store_kv_cache_repr(kr_a + kr_b, vr_a + vr_b, old_k, old_v, slot_a + slot_b);
        let base = store_kv_cache_repr(kr_a, vr_a, old_k, old_v, slot_a);
        &&& crate::proof::tensor::geometry::slot_in_cache(full.0, s)
        &&& crate::proof::tensor::geometry::slot_in_cache(full.1, s)
        &&& crate::proof::tensor::geometry::cache_at(full.0, s) == crate::proof::tensor::geometry::cache_at(base.0, s)
        &&& crate::proof::tensor::geometry::cache_at(full.1, s) == crate::proof::tensor::geometry::cache_at(base.1, s)
    }),
{
    let base = store_kv_cache_repr(kr_a, vr_a, old_k, old_v, slot_a);
    store_kv_cache_repr_concat(kr_a, vr_a, kr_b, vr_b, old_k, old_v, slot_a, slot_b);
    store_kv_cache_repr_preserves_unwritten_slots(kr_b, vr_b, base.0, base.1, slot_b, s);
}

// Capstone: the full store (own writes `A` followed by other requests' writes
// `B`) agrees with the own-only store at this request's read slot
// `block_table_slot(bt_row, pos)`, provided every other write lands in a block
// different from this request's block.  Composes `slots_miss_disjoint_block`
// (block disjointness ⇒ the other slots miss `s`) with
// `store_kv_cache_repr_append_misses_slot` (missing writes preserve `s`).  This
// is precisely the per-position cache agreement consumed by
// `decoder_core_cache_independence` — its only remaining input is the
// engine/scheduler fact that `slot_repr == own ++ others` with disjoint blocks.
pub proof fn store_agrees_at_own_block(
    kr_a: Tensor2D, vr_a: Tensor2D,
    kr_b: Tensor2D, vr_b: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_a: Seq<int>, slot_b: Seq<int>,
    bt_row: Seq<BlockId>, pos: nat,
)
    requires
        kr_a.len() == vr_a.len(), vr_a.len() == slot_a.len(),
        kr_b.len() == vr_b.len(), vr_b.len() == slot_b.len(),
        (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int) < bt_row.len(),
        forall|m: int| 0 <= m < slot_b.len() ==>
            #[trigger] slot_b[m] / (crate::types::BLOCK_SIZE_SPEC as int)
                != bt_row[(pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)] as int,
        crate::proof::tensor::geometry::slot_in_cache(
            store_kv_cache_repr(kr_a, vr_a, old_k, old_v, slot_a).0,
            crate::proof::tensor::geometry::block_table_slot(bt_row, pos)),
        crate::proof::tensor::geometry::slot_in_cache(
            store_kv_cache_repr(kr_a, vr_a, old_k, old_v, slot_a).1,
            crate::proof::tensor::geometry::block_table_slot(bt_row, pos)),
    ensures ({
        let s = crate::proof::tensor::geometry::block_table_slot(bt_row, pos);
        let full = store_kv_cache_repr(kr_a + kr_b, vr_a + vr_b, old_k, old_v, slot_a + slot_b);
        let base = store_kv_cache_repr(kr_a, vr_a, old_k, old_v, slot_a);
        &&& crate::proof::tensor::geometry::slot_in_cache(full.0, s)
        &&& crate::proof::tensor::geometry::slot_in_cache(full.1, s)
        &&& crate::proof::tensor::geometry::cache_at(full.0, s) == crate::proof::tensor::geometry::cache_at(base.0, s)
        &&& crate::proof::tensor::geometry::cache_at(full.1, s) == crate::proof::tensor::geometry::cache_at(base.1, s)
    }),
{
    let s = crate::proof::tensor::geometry::block_table_slot(bt_row, pos);
    crate::proof::tensor::geometry::slots_miss_disjoint_block(slot_b, bt_row, pos);
    store_kv_cache_repr_append_misses_slot(kr_a, vr_a, kr_b, vr_b, old_k, old_v, slot_a, slot_b, s);
}

// Base congruence at a slot: the store's value at slot `s` depends on the base
// caches only through their value at `s` (and the writes).  So two stores with
// the same writes over bases that agree at `s` agree at `s`.  This lets writes
// on EITHER side of a request be discounted (interleaved / request-major slot
// mappings), generalizing `append_misses_slot`.  Proved by induction peeling the
// last write: if it targets `s` both stores write the same value; otherwise it
// hits a different (page, offset) and leaves `s` untouched.
pub proof fn store_kv_cache_repr_base_congruence_at_slot(
    kr: Tensor2D, vr: Tensor2D,
    b1k: KVCacheLayerRepr, b1v: KVCacheLayerRepr,
    b2k: KVCacheLayerRepr, b2v: KVCacheLayerRepr,
    slot: Seq<int>, s: nat,
)
    requires
        kr.len() == vr.len(), vr.len() == slot.len(),
        crate::proof::tensor::geometry::slot_in_cache(b1k, s),
        crate::proof::tensor::geometry::slot_in_cache(b1v, s),
        crate::proof::tensor::geometry::slot_in_cache(b2k, s),
        crate::proof::tensor::geometry::slot_in_cache(b2v, s),
        crate::proof::tensor::geometry::cache_at(b1k, s) == crate::proof::tensor::geometry::cache_at(b2k, s),
        crate::proof::tensor::geometry::cache_at(b1v, s) == crate::proof::tensor::geometry::cache_at(b2v, s),
    ensures ({
        let s1 = store_kv_cache_repr(kr, vr, b1k, b1v, slot);
        let s2 = store_kv_cache_repr(kr, vr, b2k, b2v, slot);
        &&& crate::proof::tensor::geometry::slot_in_cache(s1.0, s)
        &&& crate::proof::tensor::geometry::slot_in_cache(s1.1, s)
        &&& crate::proof::tensor::geometry::slot_in_cache(s2.0, s)
        &&& crate::proof::tensor::geometry::slot_in_cache(s2.1, s)
        &&& crate::proof::tensor::geometry::cache_at(s1.0, s) == crate::proof::tensor::geometry::cache_at(s2.0, s)
        &&& crate::proof::tensor::geometry::cache_at(s1.1, s) == crate::proof::tensor::geometry::cache_at(s2.1, s)
    }),
    decreases kr.len(),
{
    reveal(store_kv_cache_repr);
    let bs = crate::types::BLOCK_SIZE_SPEC as int;
    if kr.len() == 0 {
    } else {
        let n = kr.len() as int;
        store_kv_cache_repr_base_congruence_at_slot(
            kr.subrange(0, n - 1), vr.subrange(0, n - 1),
            b1k, b1v, b2k, b2v, slot.subrange(0, n - 1), s);
        let s_last = slot[n - 1];
        // Distinct slots map to distinct (page, offset) cells.
        if s_last != s as int {
            assert((s as int) / bs * bs + (s as int) % bs == s as int) by {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod(s as int, bs);
            }
            assert(s_last / bs * bs + s_last % bs == s_last) by {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod(s_last, bs);
            }
        }
    }
}

// Store preserves slot validity: the scatter only `update`s existing cells, so
// it never changes any cache dimension; a slot valid in the base stays valid.
pub proof fn store_kv_cache_repr_preserves_slot_in_cache(
    kr: Tensor2D, vr: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot: Seq<int>, s: nat,
)
    requires
        kr.len() == vr.len(), vr.len() == slot.len(),
        crate::proof::tensor::geometry::slot_in_cache(old_k, s),
        crate::proof::tensor::geometry::slot_in_cache(old_v, s),
    ensures ({
        let st = store_kv_cache_repr(kr, vr, old_k, old_v, slot);
        crate::proof::tensor::geometry::slot_in_cache(st.0, s) && crate::proof::tensor::geometry::slot_in_cache(st.1, s)
    }),
    decreases kr.len(),
{
    reveal(store_kv_cache_repr);
    let bs = crate::types::BLOCK_SIZE_SPEC as int;
    if kr.len() == 0 {
    } else {
        let n = kr.len() as int;
        store_kv_cache_repr_preserves_slot_in_cache(
            kr.subrange(0, n - 1), vr.subrange(0, n - 1),
            old_k, old_v, slot.subrange(0, n - 1), s);
        // The last `update` preserves all outer/inner lengths, so slot_in_cache
        // is unchanged from `mid`.
    }
}

// Store reads back its own write: at the slot written by index `j` and not
// overwritten by any later write (`j < m < n ==> slot[m] != slot[j]`), the
// post-store cache holds exactly the stored row `kr[j]`/`vr[j]`.  This is the
// fresh-position primitive for geometry relocation — each side reads back the
// same just-stored K/V regardless of which physical slot it landed in.
pub proof fn store_kv_cache_repr_reads_own_write(
    kr: Tensor2D, vr: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot: Seq<int>, j: int,
)
    requires
        kr.len() == vr.len(), vr.len() == slot.len(),
        0 <= j < slot.len(),
        slot[j] >= 0,
        crate::proof::tensor::geometry::slot_in_cache(old_k, slot[j] as nat),
        crate::proof::tensor::geometry::slot_in_cache(old_v, slot[j] as nat),
        forall|m: int| j < m < slot.len() ==> slot[m] != slot[j],
    ensures ({
        let s = slot[j] as nat;
        let st = store_kv_cache_repr(kr, vr, old_k, old_v, slot);
        &&& crate::proof::tensor::geometry::slot_in_cache(st.0, s)
        &&& crate::proof::tensor::geometry::slot_in_cache(st.1, s)
        &&& crate::proof::tensor::geometry::cache_at(st.0, s) == kr[j]
        &&& crate::proof::tensor::geometry::cache_at(st.1, s) == vr[j]
    }),
    decreases kr.len(),
{
    reveal(store_kv_cache_repr);
    let bs = crate::types::BLOCK_SIZE_SPEC as int;
    let n = kr.len() as int;
    let sj = slot[j] as int;
    let page = sj / bs;
    let offset = sj % bs;
    let mid = store_kv_cache_repr(
        kr.subrange(0, n - 1), vr.subrange(0, n - 1),
        old_k, old_v, slot.subrange(0, n - 1));
    // mid preserves the slot's validity from the base caches.
    store_kv_cache_repr_preserves_slot_in_cache(
        kr.subrange(0, n - 1), vr.subrange(0, n - 1),
        old_k, old_v, slot.subrange(0, n - 1), sj as nat);
    // Decompose the address of `sj`.
    assert(page * bs + offset == sj) by {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(sj, bs);
    }
    if j == n - 1 {
        // The last write targets `sj`; it `update`s exactly (page, offset).
        // Guard is taken because `sj` is valid in `mid`.
    } else {
        // Later writes don't hit `sj`, so the last (at slot[n-1] != sj) leaves it.
        let s_last = slot[n - 1];
        assert(s_last != sj);  // from the no-later-overwrite hypothesis at m = n-1
        store_kv_cache_repr_reads_own_write(
            kr.subrange(0, n - 1), vr.subrange(0, n - 1),
            old_k, old_v, slot.subrange(0, n - 1), j);
        if s_last >= 0 {
            assert((s_last / bs) * bs + s_last % bs == s_last) by {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod(s_last, bs);
            }
        }
    }
}

// Fresh-position relocation agreement: a single request stores the SAME rows
// `(kr, vr)` at its own (distinct) slots under two different geometries — engine
// `slot_a`/`bt_row_a` and machine `slot_b`/`bt_row_b`.  Under the geometry
// *alignment* `block_table_slot(bt_row_*, k−q+j) == slot_*[j]` (the read of the
// j-th fresh position resolves to the slot the store wrote it to), both sides
// read back exactly `kr[j]`/`vr[j]` — so the post-store caches agree at every
// fresh read position.  Pure consequence of `reads_own_write` on each side.
pub proof fn fresh_positions_relocation_agree(
    kr: Tensor2D, vr: Tensor2D,
    old_k_a: KVCacheLayerRepr, old_v_a: KVCacheLayerRepr,
    slot_a: Seq<int>, bt_row_a: Seq<BlockId>,
    old_k_b: KVCacheLayerRepr, old_v_b: KVCacheLayerRepr,
    slot_b: Seq<int>, bt_row_b: Seq<BlockId>,
    q_len: nat, k_len: nat, j: int,
)
    requires
        kr.len() == vr.len(),
        vr.len() == slot_a.len(),
        slot_a.len() == q_len,
        slot_b.len() == q_len,
        q_len <= k_len,
        0 <= j < q_len,
        slot_a[j] >= 0,
        slot_b[j] >= 0,
        crate::proof::tensor::geometry::slot_in_cache(old_k_a, slot_a[j] as nat),
        crate::proof::tensor::geometry::slot_in_cache(old_v_a, slot_a[j] as nat),
        crate::proof::tensor::geometry::slot_in_cache(old_k_b, slot_b[j] as nat),
        crate::proof::tensor::geometry::slot_in_cache(old_v_b, slot_b[j] as nat),
        // The j-th fresh slot is written exactly once on each side.
        forall|m: int| j < m < q_len as int ==> slot_a[m] != slot_a[j],
        forall|m: int| j < m < q_len as int ==> slot_b[m] != slot_b[j],
        // Geometry alignment: reading fresh position k−q+j resolves to slot[j].
        crate::proof::tensor::geometry::block_table_slot(bt_row_a, (k_len - q_len + j as nat) as nat)
            == slot_a[j] as nat,
        crate::proof::tensor::geometry::block_table_slot(bt_row_b, (k_len - q_len + j as nat) as nat)
            == slot_b[j] as nat,
    ensures ({
        let pos = (k_len - q_len + j as nat) as nat;
        let pa = store_kv_cache_repr(kr, vr, old_k_a, old_v_a, slot_a);
        let pb = store_kv_cache_repr(kr, vr, old_k_b, old_v_b, slot_b);
        &&& crate::proof::tensor::geometry::slot_in_cache(pa.0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
        &&& crate::proof::tensor::geometry::slot_in_cache(pa.1, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
        &&& crate::proof::tensor::geometry::slot_in_cache(pb.0, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
        &&& crate::proof::tensor::geometry::slot_in_cache(pb.1, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
        &&& crate::proof::tensor::geometry::cache_at(pa.0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
            == crate::proof::tensor::geometry::cache_at(pb.0, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
        &&& crate::proof::tensor::geometry::cache_at(pa.1, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
            == crate::proof::tensor::geometry::cache_at(pb.1, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
    }),
{
    // Each side reads back its own write: pa reads kr[j]/vr[j] at slot_a[j]; pb at slot_b[j].
    store_kv_cache_repr_reads_own_write(kr, vr, old_k_a, old_v_a, slot_a, j);
    store_kv_cache_repr_reads_own_write(kr, vr, old_k_b, old_v_b, slot_b, j);
    // The alignment equates the read slots with slot_a[j]/slot_b[j], so both
    // post-store reads equal kr[j] (resp. vr[j]).
}

// Prefix-position relocation agreement: at a cached prefix position `pos < k−q`,
// the fresh store (which writes only the request's own new slots) misses the
// read slot on each side, so the post-store cache equals the pre-store cache
// there.  Given pre-store agreement across the two geometries (= the cached-K/V
// half of `engine_kv_coherent`), the post-store caches agree at `pos`.  Pure
// consequence of `preserves_unwritten_slots` on each side.
pub proof fn prefix_positions_relocation_agree(
    kr: Tensor2D, vr: Tensor2D,
    old_k_a: KVCacheLayerRepr, old_v_a: KVCacheLayerRepr,
    slot_a: Seq<int>, bt_row_a: Seq<BlockId>,
    old_k_b: KVCacheLayerRepr, old_v_b: KVCacheLayerRepr,
    slot_b: Seq<int>, bt_row_b: Seq<BlockId>,
    pos: nat,
)
    requires
        kr.len() == vr.len(),
        vr.len() == slot_a.len(),
        slot_b.len() == slot_a.len(),
        crate::proof::tensor::geometry::slot_in_cache(old_k_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)),
        crate::proof::tensor::geometry::slot_in_cache(old_v_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)),
        crate::proof::tensor::geometry::slot_in_cache(old_k_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos)),
        crate::proof::tensor::geometry::slot_in_cache(old_v_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos)),
        // Pre-store K/V agree at this read position (the engine_kv_coherent half).
        crate::proof::tensor::geometry::cache_at(old_k_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
            == crate::proof::tensor::geometry::cache_at(old_k_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos)),
        crate::proof::tensor::geometry::cache_at(old_v_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
            == crate::proof::tensor::geometry::cache_at(old_v_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos)),
        // The fresh writes miss this cached read slot on each side.
        !slot_a.contains(crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos) as int),
        !slot_b.contains(crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos) as int),
    ensures ({
        let pa = store_kv_cache_repr(kr, vr, old_k_a, old_v_a, slot_a);
        let pb = store_kv_cache_repr(kr, vr, old_k_b, old_v_b, slot_b);
        &&& crate::proof::tensor::geometry::slot_in_cache(pa.0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
        &&& crate::proof::tensor::geometry::slot_in_cache(pa.1, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
        &&& crate::proof::tensor::geometry::slot_in_cache(pb.0, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
        &&& crate::proof::tensor::geometry::slot_in_cache(pb.1, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
        &&& crate::proof::tensor::geometry::cache_at(pa.0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
            == crate::proof::tensor::geometry::cache_at(pb.0, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
        &&& crate::proof::tensor::geometry::cache_at(pa.1, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
            == crate::proof::tensor::geometry::cache_at(pb.1, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
    }),
{
    store_kv_cache_repr_preserves_unwritten_slots(kr, vr, old_k_a, old_v_a, slot_a,
        crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos));
    store_kv_cache_repr_preserves_unwritten_slots(kr, vr, old_k_b, old_v_b, slot_b,
        crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos));
}

// Request-major store agreement: for a slot layout `before ++ own ++ after`
// where `before` and `after` (other requests' writes) both miss slot `s`, the
// full store agrees at `s` with the own-only store.  This is the realistic
// engine layout (the StepPlan concatenates per-request slot blocks), so it
// discharges the cache-agreement hypothesis from per-request block uniqueness.
// Composes `append_misses_slot` (after) + `concat` + `preserves_unwritten`
// (before) + `base_congruence` (own).
pub proof fn store_agrees_request_major(
    kr_before: Tensor2D, vr_before: Tensor2D,
    kr_own: Tensor2D, vr_own: Tensor2D,
    kr_after: Tensor2D, vr_after: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_before: Seq<int>, slot_own: Seq<int>, slot_after: Seq<int>,
    s: nat,
)
    requires
        kr_before.len() == vr_before.len(), vr_before.len() == slot_before.len(),
        kr_own.len() == vr_own.len(), vr_own.len() == slot_own.len(),
        kr_after.len() == vr_after.len(), vr_after.len() == slot_after.len(),
        !slot_before.contains(s as int),
        !slot_after.contains(s as int),
        crate::proof::tensor::geometry::slot_in_cache(old_k, s),
        crate::proof::tensor::geometry::slot_in_cache(old_v, s),
    ensures ({
        let full = store_kv_cache_repr(kr_before + kr_own + kr_after,
            vr_before + vr_own + vr_after, old_k, old_v,
            slot_before + slot_own + slot_after);
        let own = store_kv_cache_repr(kr_own, vr_own, old_k, old_v, slot_own);
        &&& crate::proof::tensor::geometry::slot_in_cache(full.0, s)
        &&& crate::proof::tensor::geometry::slot_in_cache(full.1, s)
        &&& crate::proof::tensor::geometry::cache_at(full.0, s) == crate::proof::tensor::geometry::cache_at(own.0, s)
        &&& crate::proof::tensor::geometry::cache_at(full.1, s) == crate::proof::tensor::geometry::cache_at(own.1, s)
    }),
{
    let before = store_kv_cache_repr(kr_before, vr_before, old_k, old_v, slot_before);
    // `before` leaves s at old's value.
    store_kv_cache_repr_preserves_unwritten_slots(kr_before, vr_before, old_k, old_v, slot_before, s);
    // store(before ++ own) == store(own) on top of `before`.
    store_kv_cache_repr_concat(kr_before, vr_before, kr_own, vr_own, old_k, old_v, slot_before, slot_own);
    // store(own) over `before` agrees at s with store(own) over `old`.
    store_kv_cache_repr_base_congruence_at_slot(kr_own, vr_own, before.0, before.1,
        old_k, old_v, slot_own, s);
    // `after` (missing s) doesn't disturb s.
    store_kv_cache_repr_append_misses_slot(kr_before + kr_own, vr_before + vr_own,
        kr_after, vr_after, old_k, old_v, slot_before + slot_own, slot_after, s);
}

// The hinge from block disjointness to store agreement at a read slot: if the
// other requests' writes (`before`/`after`) all land in blocks different from
// `bt_row`'s block at position `pos`, then the full store agrees with the
// own-only store at `block_table_slot(bt_row, pos)`.  Composes
// `slots_miss_disjoint_block` (block disjointness ⟹ those slots miss `s`) with
// `store_agrees_request_major`.  Quantified over `pos < k_len`, this is exactly
// the per-position cache agreement `decoder_core_request_isolation` consumes —
// reduced to the scheduler's per-request block uniqueness.
pub proof fn store_agrees_at_block_pos(
    kr_before: Tensor2D, vr_before: Tensor2D,
    kr_own: Tensor2D, vr_own: Tensor2D,
    kr_after: Tensor2D, vr_after: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_before: Seq<int>, slot_own: Seq<int>, slot_after: Seq<int>,
    bt_row: Seq<BlockId>, pos: nat,
)
    requires
        kr_before.len() == vr_before.len(), vr_before.len() == slot_before.len(),
        kr_own.len() == vr_own.len(), vr_own.len() == slot_own.len(),
        kr_after.len() == vr_after.len(), vr_after.len() == slot_after.len(),
        (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int) < bt_row.len(),
        forall|m: int| 0 <= m < slot_before.len() ==>
            #[trigger] slot_before[m] / (crate::types::BLOCK_SIZE_SPEC as int)
                != bt_row[(pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)] as int,
        forall|m: int| 0 <= m < slot_after.len() ==>
            #[trigger] slot_after[m] / (crate::types::BLOCK_SIZE_SPEC as int)
                != bt_row[(pos as int) / (crate::types::BLOCK_SIZE_SPEC as int)] as int,
        crate::proof::tensor::geometry::slot_in_cache(old_k, crate::proof::tensor::geometry::block_table_slot(bt_row, pos)),
        crate::proof::tensor::geometry::slot_in_cache(old_v, crate::proof::tensor::geometry::block_table_slot(bt_row, pos)),
    ensures ({
        let s = crate::proof::tensor::geometry::block_table_slot(bt_row, pos);
        let full = store_kv_cache_repr(kr_before + kr_own + kr_after,
            vr_before + vr_own + vr_after, old_k, old_v,
            slot_before + slot_own + slot_after);
        let own = store_kv_cache_repr(kr_own, vr_own, old_k, old_v, slot_own);
        &&& crate::proof::tensor::geometry::slot_in_cache(full.0, s)
        &&& crate::proof::tensor::geometry::slot_in_cache(full.1, s)
        &&& crate::proof::tensor::geometry::cache_at(full.0, s) == crate::proof::tensor::geometry::cache_at(own.0, s)
        &&& crate::proof::tensor::geometry::cache_at(full.1, s) == crate::proof::tensor::geometry::cache_at(own.1, s)
    }),
{
    let s = crate::proof::tensor::geometry::block_table_slot(bt_row, pos);
    crate::proof::tensor::geometry::slots_miss_disjoint_block(slot_before, bt_row, pos);
    crate::proof::tensor::geometry::slots_miss_disjoint_block(slot_after, bt_row, pos);
    store_agrees_request_major(kr_before, vr_before, kr_own, vr_own, kr_after, vr_after,
        old_k, old_v, slot_before, slot_own, slot_after, s);
}

// Quantified over all read positions: if the other requests' writes
// (`before`/`after`) land in blocks disjoint from EVERY block of `bt_row`, then
// the full store agrees with the own-only store at `block_table_slot(bt_row, pos)`
// for every `pos < k_len`.  This is exactly the `forall pos` per-position cache
// agreement `decoder_core_request_isolation` consumes; its only remaining input
// is the engine fact that other requests use disjoint blocks (per-request block
// uniqueness from `cs_valid`).
pub proof fn store_agrees_at_all_block_pos(
    kr_before: Tensor2D, vr_before: Tensor2D,
    kr_own: Tensor2D, vr_own: Tensor2D,
    kr_after: Tensor2D, vr_after: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_before: Seq<int>, slot_own: Seq<int>, slot_after: Seq<int>,
    bt_row: Seq<BlockId>, k_len: nat,
)
    requires
        kr_before.len() == vr_before.len(), vr_before.len() == slot_before.len(),
        kr_own.len() == vr_own.len(), vr_own.len() == slot_own.len(),
        kr_after.len() == vr_after.len(), vr_after.len() == slot_after.len(),
        crate::proof::tensor::geometry::blocks_needed_for(k_len) <= bt_row.len(),
        forall|m: int, l: int|
            #![trigger slot_before[m] / (crate::types::BLOCK_SIZE_SPEC as int), bt_row[l]]
            0 <= m < slot_before.len() && 0 <= l < bt_row.len() ==>
                slot_before[m] / (crate::types::BLOCK_SIZE_SPEC as int) != bt_row[l] as int,
        forall|m: int, l: int|
            #![trigger slot_after[m] / (crate::types::BLOCK_SIZE_SPEC as int), bt_row[l]]
            0 <= m < slot_after.len() && 0 <= l < bt_row.len() ==>
                slot_after[m] / (crate::types::BLOCK_SIZE_SPEC as int) != bt_row[l] as int,
        forall|pos: nat| pos < k_len ==>
            crate::proof::tensor::geometry::slot_in_cache(old_k, #[trigger] crate::proof::tensor::geometry::block_table_slot(bt_row, pos))
            && crate::proof::tensor::geometry::slot_in_cache(old_v, crate::proof::tensor::geometry::block_table_slot(bt_row, pos)),
    ensures
        forall|pos: nat| #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row, pos)]
            pos < k_len ==> {
            let s = crate::proof::tensor::geometry::block_table_slot(bt_row, pos);
            let full = store_kv_cache_repr(kr_before + kr_own + kr_after,
                vr_before + vr_own + vr_after, old_k, old_v,
                slot_before + slot_own + slot_after);
            let own = store_kv_cache_repr(kr_own, vr_own, old_k, old_v, slot_own);
            &&& crate::proof::tensor::geometry::slot_in_cache(full.0, s)
            &&& crate::proof::tensor::geometry::slot_in_cache(full.1, s)
            &&& crate::proof::tensor::geometry::cache_at(full.0, s) == crate::proof::tensor::geometry::cache_at(own.0, s)
            &&& crate::proof::tensor::geometry::cache_at(full.1, s) == crate::proof::tensor::geometry::cache_at(own.1, s)
        },
{
    let bs = crate::types::BLOCK_SIZE_SPEC as int;
    assert forall|pos: nat| #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row, pos)]
        pos < k_len implies {
        let s = crate::proof::tensor::geometry::block_table_slot(bt_row, pos);
        let full = store_kv_cache_repr(kr_before + kr_own + kr_after,
            vr_before + vr_own + vr_after, old_k, old_v,
            slot_before + slot_own + slot_after);
        let own = store_kv_cache_repr(kr_own, vr_own, old_k, old_v, slot_own);
        &&& crate::proof::tensor::geometry::slot_in_cache(full.0, s)
        &&& crate::proof::tensor::geometry::slot_in_cache(full.1, s)
        &&& crate::proof::tensor::geometry::cache_at(full.0, s) == crate::proof::tensor::geometry::cache_at(own.0, s)
        &&& crate::proof::tensor::geometry::cache_at(full.1, s) == crate::proof::tensor::geometry::cache_at(own.1, s)
    } by {
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, k_len);
        // pos / bs < blocks_needed_for(k_len) <= bt_row.len(), so bt_row[pos/bs] is in range.
        assert forall|m: int| 0 <= m < slot_before.len() implies
            #[trigger] slot_before[m] / bs != bt_row[(pos as int) / bs] as int by {}
        assert forall|m: int| 0 <= m < slot_after.len() implies
            #[trigger] slot_after[m] / bs != bt_row[(pos as int) / bs] as int by {}
        store_agrees_at_block_pos(kr_before, vr_before, kr_own, vr_own, kr_after, vr_after,
            old_k, old_v, slot_before, slot_own, slot_after, bt_row, pos);
    }
}

// Length-preservation: scatter write doesn't resize the cache pages.  Stated
// as a structural lemma (provable by induction) so callers don't have to
// reveal the body just to know the page count is unchanged.
// Page structure survives ANY store — out-of-range slots are dropped and
// in-range writes are element updates.  (Unconditional variant of
// `lemma_store_kv_cache_repr_lengths`: no row/slot length matching needed;
// used for the machine-side cache-shape companion.)
pub proof fn lemma_store_kv_cache_repr_shape_unconditional(
    kr: Tensor2D, vr: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_repr: Seq<int>,
)
    ensures
        store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr).0.len() == old_k.len(),
        store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr).1.len() == old_v.len(),
        forall|p: int| 0 <= p < old_k.len() ==>
            (#[trigger] store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr).0[p]).len()
                == old_k[p].len(),
        forall|p: int| 0 <= p < old_v.len() ==>
            (#[trigger] store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr).1[p]).len()
                == old_v[p].len(),
    decreases kr.len(),
{
    reveal(store_kv_cache_repr);
    if kr.len() == 0 {
    } else {
        lemma_store_kv_cache_repr_shape_unconditional(
            kr.subrange(0, kr.len() as int - 1),
            vr.subrange(0, vr.len() as int - 1),
            old_k, old_v,
            slot_repr.subrange(0, slot_repr.len() as int - 1));
    }
}

// Scatter updates cache values but not the page geometry used by the paged
// attention launch contract.
pub proof fn lemma_paged_attention_launch_ready_after_store(
    query_rows: nat,
    kr: Tensor2D,
    vr: Tensor2D,
    old_k: KVCacheLayerRepr,
    old_v: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
)
    requires
        kr.len() == vr.len(),
        vr.len() == slot_repr.len(),
        paged_attention_launch_ready(
            query_rows, old_k, old_v, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr,
        ),
    ensures
        paged_attention_launch_ready(
            query_rows,
            store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr).0,
            store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr).1,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
{
    let post = store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr);
    lemma_paged_attention_launch_ready_parts(
        query_rows, old_k, old_v, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr,
    );
    lemma_store_kv_cache_repr_shape_unconditional(
        kr, vr, old_k, old_v, slot_repr,
    );
    reveal(paged_cache_geometry);
    assert(old_k.len() > 0);
    assert(post.0.len() == old_k.len());
    assert(post.0.len() > 0);
    assert(post.1.len() == old_v.len());
    assert(old_v.len() == old_k.len());
    assert(post.1.len() == post.0.len());
    assert forall|p: int| 0 <= p < post.0.len() implies
        (#[trigger] post.0[p]).len()
            == crate::types::BLOCK_SIZE_SPEC as int
    by {
        assert(post.0[p].len() == old_k[p].len());
        assert(old_k[p].len() == crate::types::BLOCK_SIZE_SPEC as int);
    }
    assert forall|p: int| 0 <= p < post.1.len() implies
        (#[trigger] post.1[p]).len()
            == crate::types::BLOCK_SIZE_SPEC as int
    by {
        assert(post.1[p].len() == old_v[p].len());
        assert(old_v[p].len() == crate::types::BLOCK_SIZE_SPEC as int);
    }
    assert(paged_cache_geometry(post.0, post.1));
    assert(paged_attention_metadata_ready(
        query_rows, post.0.len(), cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr,
    ));
    lemma_paged_attention_launch_ready_from_parts(
        query_rows, post.0, post.1, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr,
    );
}

pub proof fn lemma_store_kv_cache_repr_lengths(
    kr: Tensor2D, vr: Tensor2D,
    old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slot_repr: Seq<int>,
)
    requires kr.len() == vr.len(), vr.len() == slot_repr.len(),
    ensures
        store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr).0.len() == old_k.len(),
        store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr).1.len() == old_v.len(),
        forall|p: int| 0 <= p < old_k.len() ==>
            (#[trigger] store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr).0[p]).len()
                == old_k[p].len(),
        forall|p: int| 0 <= p < old_v.len() ==>
            (#[trigger] store_kv_cache_repr(kr, vr, old_k, old_v, slot_repr).1[p]).len()
                == old_v[p].len(),
    decreases kr.len(),
{
    reveal(store_kv_cache_repr);
    if kr.len() == 0 {
    } else {
        lemma_store_kv_cache_repr_lengths(
            kr.subrange(0, kr.len() as int - 1),
            vr.subrange(0, vr.len() as int - 1),
            old_k, old_v,
            slot_repr.subrange(0, slot_repr.len() as int - 1));
    }
}

// @kernel-bridge-begin boundary::tensor_runtime::store_kv_cache
pub fn store_kv_cache(
    runtime: &ModelFamilyRuntime,
    k: &Tensor, v: &Tensor,
    k_cache: &Tensor, v_cache: &Tensor,
    slot_mapping: &Tensor,
    Tracked(kp): Tracked<&TensorPerm>,
    Tracked(vp): Tracked<&TensorPerm>,
    Tracked(kc_perm): Tracked<&mut TensorPerm>,
    Tracked(vc_perm): Tracked<&mut TensorPerm>,
    Tracked(sp): Tracked<&TensorPerm>,
    Ghost(kr): Ghost<Tensor2D>,
    Ghost(vr): Ghost<Tensor2D>,
    Ghost(k_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(v_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(slot_repr): Ghost<Seq<int>>,
)
    requires
        family_runtime_execution_valid(runtime),
        tensor_repr_2d(*kp, *k, kr),
        tensor_repr_2d(*vp, *v, vr),
        kv_cache_tensor_repr(*old(kc_perm), *k_cache, k_cache_repr),
        kv_cache_tensor_repr(*old(vc_perm), *v_cache, v_cache_repr),
        int_tensor_repr_1d(*sp, *slot_mapping, slot_repr),
        kr.len() == vr.len(),
        vr.len() == slot_repr.len(),
        store_kv_cache_launch_ready(
            kr.len(), k_cache_repr, v_cache_repr, slot_repr,
        ),
    ensures
        kv_cache_tensor_repr(*final(kc_perm), *k_cache,
            store_kv_cache_repr(kr, vr, k_cache_repr, v_cache_repr, slot_repr).0),
        kv_cache_tensor_repr(*final(vc_perm), *v_cache,
            store_kv_cache_repr(kr, vr, k_cache_repr, v_cache_repr, slot_repr).1),
        final(kc_perm).id() == old(kc_perm).id(),
        final(vc_perm).id() == old(vc_perm).id(),
{
    store_kv_cache_raw(runtime, k, v, k_cache, v_cache, slot_mapping,
        Tracked(kp), Tracked(vp), Tracked(kc_perm), Tracked(vc_perm), Tracked(sp),
        Ghost(kr), Ghost(vr), Ghost(k_cache_repr), Ghost(v_cache_repr), Ghost(slot_repr));
    proof {
        let width = choose|width: nat|
            KV_STORE::domain(kr, k_cache_repr, slot_repr, width)
            && KV_STORE::domain(vr, v_cache_repr, slot_repr, width)
            && kv_cache_tensor_repr(*kc_perm, *k_cache, KV_STORE::output(kr, k_cache_repr, slot_repr, width))
            && kv_cache_tensor_repr(*vc_perm, *v_cache, KV_STORE::output(vr, v_cache_repr, slot_repr, width));
        KV_STORE::checked_binding(kr, vr, k_cache_repr, v_cache_repr, slot_repr, width);
    }
}

// Physical scatter launch correspondence, not a handwritten copy/frame law.
// The hidden width witnesses the fixed deployment's row geometry. Dtype,
// non-aliasing and analyzed-to-launched correspondence are retained in domain.
#[verifier::external_body]
fn store_kv_cache_raw(
    runtime: &ModelFamilyRuntime,
    k: &Tensor, v: &Tensor,
    k_cache: &Tensor, v_cache: &Tensor,
    slot_mapping: &Tensor,
    Tracked(kp): Tracked<&TensorPerm>,
    Tracked(vp): Tracked<&TensorPerm>,
    Tracked(kc_perm): Tracked<&mut TensorPerm>,
    Tracked(vc_perm): Tracked<&mut TensorPerm>,
    Tracked(sp): Tracked<&TensorPerm>,
    Ghost(kr): Ghost<Tensor2D>,
    Ghost(vr): Ghost<Tensor2D>,
    Ghost(k_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(v_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(slot_repr): Ghost<Seq<int>>,
)
    requires
        family_runtime_execution_valid(runtime),
        tensor_repr_2d(*kp, *k, kr), tensor_repr_2d(*vp, *v, vr),
        kv_cache_tensor_repr(*old(kc_perm), *k_cache, k_cache_repr),
        kv_cache_tensor_repr(*old(vc_perm), *v_cache, v_cache_repr),
        int_tensor_repr_1d(*sp, *slot_mapping, slot_repr),
        kr.len() == vr.len(), vr.len() == slot_repr.len(),
        store_kv_cache_launch_ready(kr.len(), k_cache_repr, v_cache_repr, slot_repr),
    ensures
        exists|width: nat|
            KV_STORE::domain(kr, k_cache_repr, slot_repr, width)
            && KV_STORE::domain(vr, v_cache_repr, slot_repr, width)
            && kv_cache_tensor_repr(*final(kc_perm), *k_cache, KV_STORE::output(kr, k_cache_repr, slot_repr, width))
            && kv_cache_tensor_repr(*final(vc_perm), *v_cache, KV_STORE::output(vr, v_cache_repr, slot_repr, width)),
        final(kc_perm).id() == old(kc_perm).id(),
        final(vc_perm).id() == old(vc_perm).id(),
{
    #[cfg(not(verus_only))]
    {
        pyo3::Python::with_gil(|py| -> pyo3::PyResult<()> {
            let m = py.import_bound("vosti_kernels")?;
            // Logical slot safety is proved by store_kv_cache_launch_ready;
            // the source-attested materializer/cache/kernel chain establishes
            // the physical roles.  The engine entry point therefore avoids
            // synchronizing device metadata back to the host per layer.
            let f = m.getattr("store_kv_cache_from_verified_caller")?;
            f.call1((
                k.inner.bind(py), v.inner.bind(py),
                k_cache.inner.bind(py), v_cache.inner.bind(py),
                slot_mapping.inner.bind(py), primitive_runtime_argument(runtime, py),
            ))?;
            Ok(())
        }).expect("python store_kv_cache kernel failed");
    }
}
// @kernel-bridge-end boundary::tensor_runtime::store_kv_cache

// ===========================================================================
// PagedAttention — non-mutating read of paged KV cache + Q.
// ===========================================================================

// @kernel-bridge-begin boundary::tensor_runtime::paged_attention
pub fn paged_attention(
    runtime: &ModelFamilyRuntime,
    q: &Tensor,
    k_cache: &Tensor, v_cache: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    block_table: &Tensor,
    Tracked(qp): Tracked<&TensorPerm>,
    Tracked(kc_perm): Tracked<&TensorPerm>,
    Tracked(vc_perm): Tracked<&TensorPerm>,
    Tracked(cuq_perm): Tracked<&TensorPerm>,
    Tracked(cuk_perm): Tracked<&TensorPerm>,
    Tracked(bt_perm): Tracked<&TensorPerm>,
    Ghost(qr): Ghost<Tensor2D>,
    Ghost(k_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(v_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(cu_k_repr): Ghost<Seq<int>>,
    Ghost(bt_repr): Ghost<Seq<Seq<BlockId>>>,
    Ghost(scope): Ghost<Set<TensorId>>,
    Ghost(parameters): Ghost<AttentionParametersRepr>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        paged_attention_numeric_domain(),
        family_runtime_execution_valid(runtime),
        dense_swiglu_runtime_attention_matches(runtime, parameters),
        tensor_repr_2d(*qp, *q, qr),
        kv_cache_tensor_repr(*kc_perm, *k_cache, k_cache_repr),
        kv_cache_tensor_repr(*vc_perm, *v_cache, v_cache_repr),
        int_tensor_repr_1d(*cuq_perm, *cu_seqlens_q, cu_q_repr),
        int_tensor_repr_1d(*cuk_perm, *cu_seqlens_k, cu_k_repr),
        block_table_repr(*bt_perm, *block_table, bt_repr),
        paged_attention_launch_ready(
            qr.len(), k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        ),
    ensures ({ let (t, perm) = out;
               t.id() != q.id()
               && !scope.contains(t.id())
               && tensor_repr_2d(perm@, t,
                    paged_attention_repr(qr, k_cache_repr, v_cache_repr,
                        cu_q_repr, cu_k_repr,
                        max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                        parameters,
                        )) }),
{
    let out = paged_attention_raw(runtime, q, k_cache, v_cache,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, block_table,
        Tracked(qp), Tracked(kc_perm), Tracked(vc_perm), Tracked(cuq_perm), Tracked(cuk_perm), Tracked(bt_perm),
        Ghost(qr), Ghost(k_cache_repr), Ghost(v_cache_repr), Ghost(cu_q_repr), Ghost(cu_k_repr), Ghost(bt_repr),
        Ghost(scope), Ghost(parameters));
    proof {
        ATTN::checked_runtime_binding(qr, k_cache_repr, v_cache_repr, cu_q_repr, cu_k_repr, bt_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, AttentionKind::Full, parameters, 0);
        reveal(paged_attention_repr);
    }
    out
}

// Trusted physical boundary only: qualified raw execution, concrete geometry,
// allocation/representation, and the exact generated numeric predicates under
// the explicit deployed finiteness assumption. No mapped-output equality or
// batching/causal relational law is assumed here.
#[verifier::external_body]
fn paged_attention_raw(
    runtime: &ModelFamilyRuntime,
    q: &Tensor,
    k_cache: &Tensor, v_cache: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    block_table: &Tensor,
    Tracked(qp): Tracked<&TensorPerm>,
    Tracked(kc_perm): Tracked<&TensorPerm>,
    Tracked(vc_perm): Tracked<&TensorPerm>,
    Tracked(cuq_perm): Tracked<&TensorPerm>,
    Tracked(cuk_perm): Tracked<&TensorPerm>,
    Tracked(bt_perm): Tracked<&TensorPerm>,
    Ghost(qr): Ghost<Tensor2D>,
    Ghost(k_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(v_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(cu_k_repr): Ghost<Seq<int>>,
    Ghost(bt_repr): Ghost<Seq<Seq<BlockId>>>,
    Ghost(scope): Ghost<Set<TensorId>>,
    Ghost(parameters): Ghost<AttentionParametersRepr>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        paged_attention_numeric_domain(),
        family_runtime_execution_valid(runtime),
        dense_swiglu_runtime_attention_matches(runtime, parameters),
        tensor_repr_2d(*qp, *q, qr),
        kv_cache_tensor_repr(*kc_perm, *k_cache, k_cache_repr),
        kv_cache_tensor_repr(*vc_perm, *v_cache, v_cache_repr),
        int_tensor_repr_1d(*cuq_perm, *cu_seqlens_q, cu_q_repr),
        int_tensor_repr_1d(*cuk_perm, *cu_seqlens_k, cu_k_repr),
        block_table_repr(*bt_perm, *block_table, bt_repr),
        paged_attention_launch_ready(
            qr.len(), k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        ),
    ensures
        RAW_ATTENTION::binding_valid(AttentionKind::Full, parameters.geometry, 0),
        RAW_ATTENTION::layout_ready(qr, k_cache_repr, v_cache_repr, bt_repr, cu_q_repr, cu_k_repr, max_seqlen_q as nat, max_seqlen_k as nat, parameters.geometry),
        RAW_ATTENTION::numeric_requirements(qr, k_cache_repr, v_cache_repr, bt_repr, cu_q_repr, cu_k_repr, AttentionKind::Full, parameters.geometry, ATTN::scale_log2(parameters), 0, KERNEL_SUPPORT::generated_kernel_allocation_cell()),
        ({ let (tensor, perm) = out;
            &&& tensor.id() != q.id()
            &&& !scope.contains(tensor.id())
            &&& tensor_repr_2d(perm@, tensor,
                RAW_ATTENTION::raw_output(qr, k_cache_repr, v_cache_repr, bt_repr, cu_q_repr, cu_k_repr, max_seqlen_q as nat,
                    AttentionKind::Full, parameters.geometry, ATTN::scale_log2(parameters), 0, KERNEL_SUPPORT::generated_kernel_allocation_cell()).unwrap())
        }),

{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let m = py.import_bound("vosti_kernels")?;
                // paged_attention_launch_ready proves the logical metadata
                // contract; the engine-produced representation is separately
                // source-attested, so this dispatch performs no device-value
                // validation or host copy.
                let f = m.getattr("paged_attention_from_verified_caller")?;
                let res = f.call1((q.inner.bind(py),
                                   k_cache.inner.bind(py),
                                   v_cache.inner.bind(py),
                                   cu_seqlens_q.inner.bind(py),
                                   cu_seqlens_k.inner.bind(py),
                                   max_seqlen_q, max_seqlen_k,
                                   block_table.inner.bind(py),
                                   primitive_runtime_argument(runtime, py)))?;
                Ok(res.unbind())
            }).expect("python paged_attention kernel failed");
        (Tensor { inner: py_handle }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}
// @kernel-bridge-end boundary::tensor_runtime::paged_attention

// ===========================================================================
// Batched last-row selection and sampling.  These exact adapters replace the
// per-request Python crossings on the engine path while retaining the same
// pure select/sample composition.
// ===========================================================================

// @kernel-bridge-begin boundary::tensor_runtime::select_last_hidden_rows
#[verifier::external_body]
pub fn select_last_hidden_rows(
    hidden: &Tensor,
    cu_seqlens_q: &Tensor,
    count: usize,
    Tracked(hp): Tracked<&TensorPerm>,
    Tracked(cuq_perm): Tracked<&TensorPerm>,
    Ghost(hidden_repr): Ghost<Tensor2D>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(scope): Ghost<Set<TensorId>>,
) -> (out: (Tensor, Tracked<TensorPerm>))
    requires
        tensor_repr_2d(*hp, *hidden, hidden_repr),
        int_tensor_repr_1d(*cuq_perm, *cu_seqlens_q, cu_q_repr),
        cu_q_repr.len() == count as nat + 1,
        forall|k: int| 0 <= k < count as int ==>
            #[trigger] cu_q_repr[k + 1] > 0
                && cu_q_repr[k + 1] <= hidden_repr.len() as int,
    ensures ({ let (tensor, perm) = out;
        tensor.id() != hidden.id()
            && tensor.id() != cu_seqlens_q.id()
            && !scope.contains(tensor.id())
            && tensor_repr_2d(
                perm@,
                tensor,
                Seq::new(count as nat, |i: int|
                    hidden_repr[cu_q_repr[i + 1] - 1]),
            )
    }),
{
    #[cfg(not(verus_only))]
    {
        let py_handle: pyo3::Py<pyo3::PyAny> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let module = py.import_bound("vosti_kernels")?;
                let function = module.getattr("select_rows_for_sampling")?;
                let result = function.call1((
                    hidden.inner.bind(py),
                    cu_seqlens_q.inner.bind(py),
                ))?;
                Ok(result.unbind())
            },
        )
        .expect("python select_rows_for_sampling failed");
        (Tensor { inner: py_handle }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}
// @kernel-bridge-end boundary::tensor_runtime::select_last_hidden_rows

// @kernel-bridge-begin boundary::tensor_runtime::sample_tokens_rows
#[verifier::external_body]
pub fn sample_tokens_rows(
    rows: &Tensor,
    count: usize,
    Tracked(rp): Tracked<&TensorPerm>,
    Ghost(rows_repr): Ghost<Tensor2D>,
    Ghost(states): Ghost<Seq<crate::exec::request_state::SamplerState>>,
) -> (out: Vec<u64>)
    requires
        tensor_repr_2d(*rp, *rows, rows_repr),
        rows_repr.len() == count as nat,
        states.len() == count as nat,
    ensures
        out.len() == count,
        forall|k: int| 0 <= k < count as int ==>
            (#[trigger] out[k]) as nat
                == sample_from_repr(rows_repr[k], states[k]).1,
        forall|k: int| 0 <= k < count as int ==>
            #[trigger] sample_from_repr(rows_repr[k], states[k]).0 == states[k],
{
    #[cfg(not(verus_only))]
    {
        let tokens: Vec<u64> = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<Vec<u64>> {
                let module = py.import_bound("vosti_kernels")?;
                let function = module.getattr("sample_tokens_rows")?;
                let result = function.call1((rows.inner.bind(py),))?;
                result.extract::<Vec<u64>>()
            },
        )
        .expect("python sample_tokens_rows failed");
        if tokens.len() != count {
            panic!(
                "sample_tokens_rows returned {} tokens, expected {}",
                tokens.len(),
                count,
            );
        }
        tokens
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}
// @kernel-bridge-end boundary::tensor_runtime::sample_tokens_rows

// ---------------------------------------------------------------------------
// Qualified kernel-plan runtime capability.
//
// Engine owns one immutable deployment identity independently of scheduling.
// Both family runtimes retain a profile-qualified Python capability providing
// exact static launch lookup. They enter through the same qualified-plan gate,
// and every forward revalidates the executable identity against the capability
// stored by Engine.
//
// This block intentionally stays at the end of the large boundary module. Its
// declarations are independent of the established Qwen proof surface, and the
// placement preserves stable solver declaration order for those proofs.
// ---------------------------------------------------------------------------

// @kernel-bridge-begin boundary::tensor_runtime::qualified_kernel_plan_capability
pub struct KernelPlanId {
    pub word0: u64,
    pub word1: u64,
    pub word2: u64,
    pub word3: u64,
}

pub enum KernelPlanQualification {
    Staged,
    BackendQualified,
}

#[allow(dead_code)] // architecture/id are consumed by Verus specs, not plain Rust.
pub struct QualifiedKernelPlan {
    pub(crate) architecture: ModelArchitecture,
    pub(crate) qualification: KernelPlanQualification,
    pub(crate) plan_id: KernelPlanId,
    #[cfg(not(verus_only))]
    pub(crate) deployment_sha256: String,
}

#[verifier::external_body]
pub(crate) struct RuntimeCapabilityHandle {
    #[cfg(not(verus_only))]
    pub(crate) inner: pyo3::Py<pyo3::PyAny>,
    #[cfg(verus_only)]
    _phantom: PhantomData<()>,
}

// Every family runtime has the same explicit host capability and immutable
// qualified plan. The closed `ModelRuntime` variants retain architecture
// identity; family modules own construction, host checks, and readiness.
#[allow(dead_code)] // model_config is consumed by Verus specs, not plain Rust.
pub struct ModelFamilyRuntime {
    pub(crate) handle: RuntimeCapabilityHandle,
    pub(crate) kernel_plan: QualifiedKernelPlan,
    pub(crate) model_config: RuntimeModelConfig,
}

// Staged runtimes are deliberately configuration-less and cannot pass the
// execution gate. A qualified capability retains the exact typed checkpoint
// configuration for which its static launch plan was admitted.
pub enum RuntimeModelConfig {
    Staged,
    Qwen3(Qwen3Config),
    Llama3(Llama3Config),
    Gemma3Text(Gemma3Config, Vec<AttentionKind>),
    // Exact deployment identity. Physical ModelWeights/ModelRuntime admission
    // remains separate from retaining a typed checkpoint configuration.
    Gemma4Text(Gemma4Config, Vec<AttentionKind>),
}

// Verified executions can only take the qualified arm. The staged arm exists
// solely for debug-only zero-layer scheduler tests, where no deployed kernel
// may be selected and Python must receive the explicit CPU fallback sentinel.
#[cfg(not(verus_only))]
pub(crate) fn primitive_runtime_argument(
    runtime: &ModelFamilyRuntime,
    py: pyo3::Python<'_>,
) -> pyo3::Py<pyo3::PyAny> {
    match &runtime.kernel_plan.qualification {
        KernelPlanQualification::BackendQualified => {
            runtime.handle.inner.clone_ref(py)
        },
        KernelPlanQualification::Staged => {
            #[cfg(debug_assertions)]
            {
                py.None()
            }
            #[cfg(not(debug_assertions))]
            {
                panic!("staged model runtimes cannot execute primitives")
            }
        },
    }
}

pub closed spec fn family_runtime_kernel_plan_architecture(
    runtime: &ModelFamilyRuntime,
) -> ModelArchitecture {
    runtime.kernel_plan.architecture
}

pub closed spec fn family_runtime_kernel_plan_qualification(
    runtime: &ModelFamilyRuntime,
) -> KernelPlanQualification {
    runtime.kernel_plan.qualification
}

pub closed spec fn family_runtime_deployment_config_repr(
    runtime: &ModelFamilyRuntime,
) -> Option<ModelDeploymentConfigRepr> {
    match &runtime.model_config {
        RuntimeModelConfig::Staged => None,
        RuntimeModelConfig::Qwen3(config) =>
            Some(ModelDeploymentConfigRepr::Qwen3(qwen3_config_repr(*config))),
        RuntimeModelConfig::Llama3(config) =>
            Some(ModelDeploymentConfigRepr::Llama3(llama3_config_repr(*config))),
        RuntimeModelConfig::Gemma3Text(config, layer_attention_kinds) =>
            Some(ModelDeploymentConfigRepr::Gemma3Text(
                gemma3_deployment_config_repr(*config, layer_attention_kinds@),
            )),
        RuntimeModelConfig::Gemma4Text(config, kinds) => Some(
            ModelDeploymentConfigRepr::Gemma4Text(
                gemma4_deployment_config_repr(*config, kinds@))),
    }
}

// A kernel-plan identity alone is insufficient: its retained checkpoint
// configuration must also be a valid member of the same closed family. This
// predicate is the common admission fact required by executable primitives.
pub open spec fn family_runtime_configuration_valid(
    runtime: &ModelFamilyRuntime,
) -> bool {
    match family_runtime_deployment_config_repr(runtime) {
        Some(config) =>
            model_deployment_config_valid(config)
            && model_deployment_config_architecture(config)
                == family_runtime_kernel_plan_architecture(runtime),
        None => false,
    }
}

pub open spec fn family_runtime_execution_valid(
    runtime: &ModelFamilyRuntime,
) -> bool {
    family_runtime_kernel_plan_qualification(runtime)
        == KernelPlanQualification::BackendQualified
    && family_runtime_configuration_valid(runtime)
}

pub open spec fn qwen3_runtime_configuration_valid(
    runtime: &ModelFamilyRuntime,
) -> bool {
    family_runtime_execution_valid(runtime)
    && match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Qwen3(_)) => true,
        _ => false,
    }
}

// Closed runtime/composition agreement for the optional Q/K-normalization
// layer. The predicate is family-neutral at its callers; adding another dense
// family requires an explicit arm here before its physical choice can execute.
pub open spec fn dense_swiglu_runtime_rms_norm_matches(
    runtime: &ModelFamilyRuntime,
    epsilon: FloatParameterBits,
) -> bool {
    family_runtime_execution_valid(runtime)
    && match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Qwen3(config)) =>
            config.rms_norm.epsilon == epsilon,
        Some(ModelDeploymentConfigRepr::Llama3(config)) =>
            config.rms_norm.epsilon == epsilon,
        Some(ModelDeploymentConfigRepr::Gemma3Text(_)) => false,
        Some(ModelDeploymentConfigRepr::Gemma4Text(_)) => false,
        None => false,
    }
}

pub open spec fn dense_swiglu_runtime_qk_norm_matches(
    runtime: &ModelFamilyRuntime,
    kind: QkNormKind,
) -> bool {
    family_runtime_execution_valid(runtime)
    && match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Qwen3(config)) =>
            config.composition.qk_norm == kind,
        Some(ModelDeploymentConfigRepr::Llama3(config)) =>
            config.composition.qk_norm == kind,
        Some(ModelDeploymentConfigRepr::Gemma3Text(_)) => false,
        Some(ModelDeploymentConfigRepr::Gemma4Text(_)) => false,
        None => false,
    }
}

pub proof fn lemma_qwen3_runtime_rms_norm_matches(
    runtime: &ModelFamilyRuntime,
)
    requires qwen3_runtime_configuration_valid(runtime),
    ensures dense_swiglu_runtime_rms_norm_matches(
        runtime, qwen3_rms_norm_epsilon_repr(),
    ),
{
    reveal(qwen3_runtime_configuration_valid);
    reveal(family_runtime_execution_valid);
    reveal(family_runtime_configuration_valid);
    reveal(model_deployment_config_valid);
    reveal(dense_swiglu_runtime_rms_norm_matches);
    match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Qwen3(config)) => {
            lemma_qwen3_config_valid_implies_rms_norm_epsilon_identity(config);
        },
        _ => { assert(false); },
    }
}

pub proof fn lemma_qwen3_runtime_qk_norm_matches(
    runtime: &ModelFamilyRuntime,
)
    requires qwen3_runtime_configuration_valid(runtime),
    ensures dense_swiglu_runtime_qk_norm_matches(
        runtime, QkNormKind::RmsNorm,
    ),
{
    reveal(qwen3_runtime_configuration_valid);
    reveal(family_runtime_execution_valid);
    reveal(family_runtime_configuration_valid);
    reveal(model_deployment_config_valid);
    reveal(dense_swiglu_runtime_qk_norm_matches);
    match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Qwen3(config)) => {
            lemma_qwen3_config_valid_implies_qk_rms_norm(config);
        },
        _ => { assert(false); },
    }
}

pub open spec fn qwen3_runtime_attention_geometry_matches(
    runtime: &ModelFamilyRuntime,
    geometry: AttentionGeometryRepr,
) -> bool {
    qwen3_runtime_configuration_valid(runtime)
    && match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Qwen3(config)) =>
            attention_geometry_repr(config.geometry) == geometry,
        _ => false,
    }
}

pub open spec fn llama3_runtime_configuration_valid(
    runtime: &ModelFamilyRuntime,
) -> bool {
    family_runtime_execution_valid(runtime)
    && match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Llama3(_)) => true,
        _ => false,
    }
}

pub open spec fn llama3_runtime_attention_geometry_matches(
    runtime: &ModelFamilyRuntime,
    geometry: AttentionGeometryRepr,
) -> bool {
    llama3_runtime_configuration_valid(runtime)
    && match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Llama3(config)) =>
            attention_geometry_repr(config.geometry) == geometry,
        _ => false,
    }
}

pub proof fn lemma_llama3_runtime_rms_norm_matches(
    runtime: &ModelFamilyRuntime,
)
    requires llama3_runtime_configuration_valid(runtime),
    ensures dense_swiglu_runtime_rms_norm_matches(
        runtime, llama3_rms_norm_epsilon_repr(),
    ),
{
    reveal(llama3_runtime_configuration_valid);
    reveal(family_runtime_execution_valid);
    reveal(family_runtime_configuration_valid);
    reveal(model_deployment_config_valid);
    reveal(dense_swiglu_runtime_rms_norm_matches);
    match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Llama3(config)) => {
            lemma_llama3_config_valid_implies_rms_norm_epsilon_identity(config);
        },
        _ => { assert(false); },
    }
}

pub proof fn lemma_llama3_runtime_qk_norm_matches(
    runtime: &ModelFamilyRuntime,
)
    requires llama3_runtime_configuration_valid(runtime),
    ensures dense_swiglu_runtime_qk_norm_matches(
        runtime, QkNormKind::Disabled,
    ),
{
    reveal(llama3_runtime_configuration_valid);
    reveal(family_runtime_execution_valid);
    reveal(family_runtime_configuration_valid);
    reveal(model_deployment_config_valid);
    reveal(dense_swiglu_runtime_qk_norm_matches);
    match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Llama3(config)) => {
            lemma_llama3_config_valid_implies_qk_norm_disabled(config);
        },
        _ => { assert(false); },
    }
}

// Shared full-attention SwiGLU decoder primitives receive their complete RoPE
// identity explicitly. The closed runtime sum decides which family payload is
// admissible; callers cannot substitute another profile's table policy.
pub open spec fn dense_swiglu_runtime_attention_matches(
    runtime: &ModelFamilyRuntime,
    parameters: AttentionParametersRepr,
) -> bool {
    family_runtime_execution_valid(runtime)
    && parameters.scale == AttentionScaleRepr::InverseSqrtHeadDim
    && match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Qwen3(config)) =>
            attention_geometry_repr(config.geometry) == parameters.geometry,
        Some(ModelDeploymentConfigRepr::Llama3(config)) =>
            attention_geometry_repr(config.geometry) == parameters.geometry,
        _ => false,
    }
}

pub open spec fn dense_swiglu_runtime_rotary_matches(
    runtime: &ModelFamilyRuntime,
    geometry: AttentionGeometryRepr,
    rotary: RotaryConfigRepr,
) -> bool {
    family_runtime_execution_valid(runtime)
    && match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Qwen3(config)) =>
            attention_geometry_repr(config.geometry) == geometry
            && config.rotary == rotary,
        Some(ModelDeploymentConfigRepr::Llama3(config)) =>
            attention_geometry_repr(config.geometry) == geometry
            && config.rotary == rotary,
        _ => false,
    }
}

pub proof fn lemma_qwen3_runtime_rotary_matches(
    runtime: &ModelFamilyRuntime,
    geometry: AttentionGeometryRepr,
)
    requires qwen3_runtime_attention_geometry_matches(runtime, geometry),
    ensures dense_swiglu_runtime_rotary_matches(
        runtime, geometry, qwen3_rotary_config_repr(),
    ),
{
    reveal(qwen3_runtime_attention_geometry_matches);
    reveal(qwen3_runtime_configuration_valid);
    reveal(family_runtime_execution_valid);
    reveal(family_runtime_configuration_valid);
    reveal(model_deployment_config_valid);
    reveal(dense_swiglu_runtime_rotary_matches);
    reveal(qwen3_rotary_config_repr);
    match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Qwen3(config)) => {
            lemma_qwen3_config_valid_implies_rotary_identity(config);
        },
        _ => { assert(false); },
    }
}

pub proof fn lemma_llama3_runtime_rotary_matches(
    runtime: &ModelFamilyRuntime,
    config: Llama3ModelWeightsExtensionRepr,
)
    requires
        llama3_runtime_configuration_valid(runtime),
        family_runtime_deployment_config_repr(runtime)
            == Some(ModelDeploymentConfigRepr::Llama3(config)),
    ensures dense_swiglu_runtime_rotary_matches(
        runtime, attention_geometry_repr(config.geometry), config.rotary,
    ),
{
    reveal(llama3_runtime_configuration_valid);
    reveal(family_runtime_execution_valid);
    reveal(family_runtime_configuration_valid);
    reveal(model_deployment_config_valid);
    reveal(dense_swiglu_runtime_rotary_matches);
}

pub open spec fn gemma3_runtime_configuration_valid(
    runtime: &ModelFamilyRuntime,
) -> bool {
    family_runtime_execution_valid(runtime)
    && match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Gemma3Text(_)) => true,
        _ => false,
    }
}

pub open spec fn gemma3_runtime_hidden_size_matches(
    runtime: &ModelFamilyRuntime,
    hidden_size: nat,
) -> bool {
    gemma3_runtime_configuration_valid(runtime)
    && match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Gemma3Text(config)) =>
            config.geometry.hidden_size == hidden_size,
        _ => false,
    }
}

pub open spec fn gemma3_runtime_attention_parameters_match(
    runtime: &ModelFamilyRuntime,
    geometry: AttentionGeometryRepr,
    sliding_window: nat,
    attention_scale: AttentionScaleRepr,
) -> bool {
    gemma3_runtime_configuration_valid(runtime)
    && match family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Gemma3Text(config)) =>
            attention_geometry_repr(config.geometry) == geometry
            && config.sliding_window == sliding_window
            && config.attention_scale == attention_scale,
        _ => false,
    }
}

pub(crate) proof fn lemma_family_runtime_deployment_config_projection(
    runtime: &ModelFamilyRuntime,
)
    ensures family_runtime_deployment_config_repr(runtime) == match &runtime.model_config {
        RuntimeModelConfig::Staged => None,
        RuntimeModelConfig::Qwen3(config) =>
            Some(ModelDeploymentConfigRepr::Qwen3(qwen3_config_repr(*config))),
        RuntimeModelConfig::Llama3(config) =>
            Some(ModelDeploymentConfigRepr::Llama3(llama3_config_repr(*config))),
        RuntimeModelConfig::Gemma3Text(config, layer_attention_kinds) =>
            Some(ModelDeploymentConfigRepr::Gemma3Text(
                gemma3_deployment_config_repr(*config, layer_attention_kinds@),
            )),
        RuntimeModelConfig::Gemma4Text(config, kinds) => Some(
            ModelDeploymentConfigRepr::Gemma4Text(
                gemma4_deployment_config_repr(*config, kinds@))),
    },
{
}

pub(crate) proof fn lemma_family_runtime_kernel_plan_projection(
    runtime: &ModelFamilyRuntime,
)
    ensures
        family_runtime_kernel_plan_architecture(runtime)
            == runtime.kernel_plan.architecture,
        family_runtime_kernel_plan_qualification(runtime)
            == runtime.kernel_plan.qualification,
{
    reveal(family_runtime_kernel_plan_architecture);
    reveal(family_runtime_kernel_plan_qualification);
}

// Closed executable runtime sum, parallel to `ModelWeights`. All variants
// carry the same explicit Python capability and qualified-plan identity.
pub enum ModelRuntime {
    Qwen3(ModelFamilyRuntime),
    Llama3(ModelFamilyRuntime),
    Gemma3Text(ModelFamilyRuntime),
    Gemma4Text(ModelFamilyRuntime),
}

pub closed spec fn model_runtime_architecture(
    runtime: &ModelRuntime,
) -> ModelArchitecture {
    match runtime {
        ModelRuntime::Qwen3(_) => ModelArchitecture::Qwen3,
        ModelRuntime::Llama3(_) => ModelArchitecture::Llama3,
        ModelRuntime::Gemma3Text(_) => ModelArchitecture::Gemma3Text,
        ModelRuntime::Gemma4Text(_) => ModelArchitecture::Gemma4Text,
    }
}

pub closed spec fn model_runtime_kernel_plan_id(runtime: &ModelRuntime) -> KernelPlanId {
    match runtime {
        ModelRuntime::Qwen3(qwen) => qwen.kernel_plan.plan_id,
        ModelRuntime::Llama3(llama) => llama.kernel_plan.plan_id,
        ModelRuntime::Gemma3Text(gemma) => gemma.kernel_plan.plan_id,
        ModelRuntime::Gemma4Text(gemma) => gemma.kernel_plan.plan_id,
    }
}

pub closed spec fn model_runtime_kernel_plan_architecture(
    runtime: &ModelRuntime,
) -> ModelArchitecture {
    match runtime {
        ModelRuntime::Qwen3(qwen) => qwen.kernel_plan.architecture,
        ModelRuntime::Llama3(llama) => llama.kernel_plan.architecture,
        ModelRuntime::Gemma3Text(gemma) => gemma.kernel_plan.architecture,
        ModelRuntime::Gemma4Text(gemma) => gemma.kernel_plan.architecture,
    }
}

pub closed spec fn model_runtime_kernel_plan_qualification(
    runtime: &ModelRuntime,
) -> KernelPlanQualification {
    match runtime {
        ModelRuntime::Qwen3(qwen) => qwen.kernel_plan.qualification,
        ModelRuntime::Llama3(llama) => llama.kernel_plan.qualification,
        ModelRuntime::Gemma3Text(gemma) => gemma.kernel_plan.qualification,
        ModelRuntime::Gemma4Text(gemma) => gemma.kernel_plan.qualification,
    }
}

pub closed spec fn model_runtime_deployment_config_repr(
    runtime: &ModelRuntime,
) -> Option<ModelDeploymentConfigRepr> {
    match runtime {
        ModelRuntime::Qwen3(qwen) =>
            family_runtime_deployment_config_repr(qwen),
        ModelRuntime::Llama3(llama) =>
            family_runtime_deployment_config_repr(llama),
        ModelRuntime::Gemma3Text(gemma) =>
            family_runtime_deployment_config_repr(gemma),
        ModelRuntime::Gemma4Text(gemma) =>
            family_runtime_deployment_config_repr(gemma),
    }
}

pub proof fn lemma_model_runtime_deployment_config_projection(
    runtime: &ModelRuntime,
)
    ensures model_runtime_deployment_config_repr(runtime) == match runtime {
        ModelRuntime::Qwen3(qwen) =>
            family_runtime_deployment_config_repr(qwen),
        ModelRuntime::Llama3(llama) =>
            family_runtime_deployment_config_repr(llama),
        ModelRuntime::Gemma3Text(gemma) =>
            family_runtime_deployment_config_repr(gemma),
        ModelRuntime::Gemma4Text(gemma) =>
            family_runtime_deployment_config_repr(gemma),
    },
{
}

// Configuration identity obtained directly from the executable checkpoint
// facade, before permissions are minted. The family weight-binding contracts
// prove that this is the same identity later projected from `perms`.
pub open spec fn physical_model_layer_attention_kinds(
    weights: &ModelWeights,
) -> Seq<AttentionKind> {
    match weights {
        ModelWeights::Qwen3(_) => Seq::empty(),
        ModelWeights::Llama3(_) => Seq::empty(),
        ModelWeights::Gemma3Text(gemma) =>
            gemma3_physical_layer_attention_kinds(gemma),
        ModelWeights::Gemma4Text(gemma) =>
            GEMMA4::weights::physical_attention_kinds(gemma),
    }
}

pub open spec fn qwen3_physical_deployment_config_repr(
    weights: &Qwen3ModelWeights,
) -> ModelDeploymentConfigRepr {
    ModelDeploymentConfigRepr::Qwen3(qwen3_config_repr(weights.config))
}

pub open spec fn llama3_physical_deployment_config_repr(
    weights: &Llama3ModelWeights,
) -> ModelDeploymentConfigRepr {
    ModelDeploymentConfigRepr::Llama3(llama3_config_repr(weights.config))
}

pub open spec fn gemma3_physical_layer_attention_kinds(
    weights: &Gemma3ModelWeights,
) -> Seq<AttentionKind> {
    Seq::new(weights.layers.len() as nat, |i: int|
        weights.layers[i].attention_kind)
}

pub open spec fn gemma3_physical_deployment_config_repr(
    weights: &Gemma3ModelWeights,
) -> ModelDeploymentConfigRepr {
    ModelDeploymentConfigRepr::Gemma3Text(gemma3_deployment_config_repr(
        weights.config,
        gemma3_physical_layer_attention_kinds(weights),
    ))
}

pub open spec fn qwen3_runtime_matches_weights(
    runtime: &ModelFamilyRuntime,
    weights: &Qwen3ModelWeights,
) -> bool {
    qwen3_runtime_configuration_valid(runtime)
    && family_runtime_deployment_config_repr(runtime)
        == Some(qwen3_physical_deployment_config_repr(weights))
}

pub open spec fn llama3_runtime_matches_weights(
    runtime: &ModelFamilyRuntime,
    weights: &Llama3ModelWeights,
) -> bool {
    llama3_runtime_configuration_valid(runtime)
    && family_runtime_deployment_config_repr(runtime)
        == Some(llama3_physical_deployment_config_repr(weights))
}

pub open spec fn gemma3_runtime_matches_weights(
    runtime: &ModelFamilyRuntime,
    weights: &Gemma3ModelWeights,
) -> bool {
    gemma3_runtime_configuration_valid(runtime)
    && family_runtime_deployment_config_repr(runtime)
        == Some(gemma3_physical_deployment_config_repr(weights))
}

pub open spec fn physical_model_deployment_config_repr(
    weights: &ModelWeights,
) -> ModelDeploymentConfigRepr {
    match weights {
        ModelWeights::Qwen3(qwen) => qwen3_physical_deployment_config_repr(qwen),
        ModelWeights::Llama3(llama) =>
            llama3_physical_deployment_config_repr(llama),
        ModelWeights::Gemma3Text(gemma) =>
            gemma3_physical_deployment_config_repr(gemma),
        ModelWeights::Gemma4Text(gemma) => ModelDeploymentConfigRepr::Gemma4Text(
            gemma4_deployment_config_repr(gemma.config,
                GEMMA4::weights::physical_attention_kinds(gemma))),
    }
}

pub open spec fn model_runtime_execution_valid(runtime: &ModelRuntime) -> bool {
    match runtime {
        ModelRuntime::Qwen3(qwen) => family_runtime_execution_valid(qwen),
        ModelRuntime::Llama3(llama) => family_runtime_execution_valid(llama),
        ModelRuntime::Gemma3Text(gemma) => family_runtime_execution_valid(gemma),
        ModelRuntime::Gemma4Text(gemma) => family_runtime_execution_valid(gemma),
    }
}

pub proof fn lemma_runtime_execution_valid_from_bound_configuration(
    weights: &ModelWeights,
    runtime: &ModelRuntime,
    perms: &ModelWeightsPerms,
)
    requires
        model_weights_bound(weights, perms),
        model_runtime_architecture(runtime) == model_weights_architecture(weights),
        model_runtime_kernel_plan_architecture(runtime)
            == model_weights_architecture(weights),
        model_runtime_kernel_plan_qualification(runtime)
            == KernelPlanQualification::BackendQualified,
        model_runtime_deployment_config_repr(runtime)
            == Some(physical_model_deployment_config_repr(weights)),
    ensures model_runtime_execution_valid(runtime),
{
    lemma_model_runtime_architecture_projection(runtime);
    lemma_model_runtime_kernel_plan_architecture_projection(runtime);
    lemma_model_runtime_kernel_plan_qualification_projection(runtime);
    lemma_model_runtime_deployment_config_projection(runtime);
    reveal(model_weights_bound);
    reveal(model_runtime_execution_valid);
    reveal(family_runtime_execution_valid);
    reveal(family_runtime_configuration_valid);
    reveal(physical_model_deployment_config_repr);
    match (weights, runtime) {
        (ModelWeights::Qwen3(qwen), ModelRuntime::Qwen3(qwen_runtime)) => {
            QWEN3::weights::lemma_physical_deployment_config_valid(
                qwen, perms,
            );
            lemma_family_runtime_deployment_config_projection(qwen_runtime);
        },
        (ModelWeights::Llama3(llama), ModelRuntime::Llama3(llama_runtime)) => {
            LLAMA3::weights::lemma_physical_deployment_config_valid(
                llama, perms,
            );
            lemma_family_runtime_deployment_config_projection(llama_runtime);
        },
        (
            ModelWeights::Gemma3Text(gemma),
            ModelRuntime::Gemma3Text(gemma_runtime),
        ) => {
            GEMMA3::weights::lemma_physical_deployment_config_valid(
                gemma, perms,
            );
            lemma_family_runtime_deployment_config_projection(gemma_runtime);
        },
        (ModelWeights::Gemma4Text(gemma), ModelRuntime::Gemma4Text(gemma_runtime)) => {
            GEMMA4::weights::lemma_physical_deployment_config_valid(
                gemma, &perms.four_norm_gated_weights());
            lemma_family_runtime_deployment_config_projection(gemma_runtime);
        },
        _ => {
            assert(false);
        },
    }
}

pub proof fn lemma_model_runtime_architecture_projection(
    runtime: &ModelRuntime,
)
    ensures
        model_runtime_architecture(runtime) == match runtime {
            ModelRuntime::Qwen3(_) => ModelArchitecture::Qwen3,
            ModelRuntime::Llama3(_) => ModelArchitecture::Llama3,
            ModelRuntime::Gemma3Text(_) => ModelArchitecture::Gemma3Text,
            ModelRuntime::Gemma4Text(_) => ModelArchitecture::Gemma4Text,
        },
{
    reveal(model_runtime_architecture);
}

pub proof fn lemma_model_runtime_kernel_plan_architecture_projection(
    runtime: &ModelRuntime,
)
    ensures
        model_runtime_kernel_plan_architecture(runtime) == match runtime {
            ModelRuntime::Qwen3(qwen) =>
                family_runtime_kernel_plan_architecture(qwen),
            ModelRuntime::Llama3(llama) =>
                family_runtime_kernel_plan_architecture(llama),
            ModelRuntime::Gemma3Text(gemma) =>
                family_runtime_kernel_plan_architecture(gemma),
            ModelRuntime::Gemma4Text(gemma) =>
                family_runtime_kernel_plan_architecture(gemma),
        },
{
    reveal(model_runtime_kernel_plan_architecture);
    reveal(family_runtime_kernel_plan_architecture);
}

pub proof fn lemma_model_runtime_kernel_plan_qualification_projection(
    runtime: &ModelRuntime,
)
    ensures
        model_runtime_kernel_plan_qualification(runtime) == match runtime {
            ModelRuntime::Qwen3(qwen) =>
                family_runtime_kernel_plan_qualification(qwen),
            ModelRuntime::Llama3(llama) =>
                family_runtime_kernel_plan_qualification(llama),
            ModelRuntime::Gemma3Text(gemma) =>
                family_runtime_kernel_plan_qualification(gemma),
            ModelRuntime::Gemma4Text(gemma) =>
                family_runtime_kernel_plan_qualification(gemma),
        },
{
    reveal(model_runtime_kernel_plan_qualification);
    reveal(family_runtime_kernel_plan_qualification);
}

// Exact permission/capability gate for the architecture-dispatched executable
// model boundary.
pub open spec fn model_execution_valid(
    weights: &ModelWeights,
    runtime: &ModelRuntime,
    perms: &ModelWeightsPerms,
) -> bool {
    perms.architecture() == model_weights_architecture(weights)
    && model_runtime_architecture(runtime) == model_weights_architecture(weights)
    && model_runtime_kernel_plan_architecture(runtime)
        == model_weights_architecture(weights)
    && model_runtime_kernel_plan_qualification(runtime)
        == KernelPlanQualification::BackendQualified
    && model_runtime_execution_valid(runtime)
    && model_runtime_deployment_config_repr(runtime)
        == Some(physical_model_deployment_config_repr(weights))
    && match (weights, runtime) {
        (ModelWeights::Qwen3(qwen), ModelRuntime::Qwen3(_)) =>
            qwen3_model_weights_bound(qwen, perms),
        (ModelWeights::Llama3(llama), ModelRuntime::Llama3(_)) =>
            llama3_model_weights_bound(llama, perms),
        (ModelWeights::Gemma3Text(gemma), ModelRuntime::Gemma3Text(_)) =>
            gemma3_model_weights_bound(gemma, perms),
        (ModelWeights::Gemma4Text(gemma), ModelRuntime::Gemma4Text(_)) =>
            GEMMA4::weights::model_weights_bound(gemma, perms),
        _ => false,
    }
}

pub proof fn lemma_model_execution_valid_num_layers(
    weights: &ModelWeights,
    runtime: &ModelRuntime,
    perms: &ModelWeightsPerms,
)
    requires model_execution_valid(weights, runtime, perms),
    ensures model_weights_num_layers(weights) == perms.num_layers(),
{
    reveal(model_execution_valid);
    match (weights, runtime) {
        (ModelWeights::Qwen3(qwen), ModelRuntime::Qwen3(_)) => {
            reveal(qwen3_model_weights_bound);
        },
        (ModelWeights::Llama3(llama), ModelRuntime::Llama3(_)) => {
            reveal(llama3_model_weights_bound);
        },
        (ModelWeights::Gemma3Text(gemma), ModelRuntime::Gemma3Text(_)) => {
            reveal(gemma3_model_weights_bound);
        },
        (ModelWeights::Gemma4Text(gemma), ModelRuntime::Gemma4Text(_)) => {},
        _ => { assert(false); },
    }
}

#[verifier::external_body]
pub(crate) fn deployment_sha256_plan_id(
    deployment_sha256: &str,
) -> (out: KernelPlanId) {
    #[cfg(not(verus_only))]
    {
        assert_eq!(deployment_sha256.len(), 64, "deployment identity is not SHA-256");
        assert!(
            deployment_sha256
                .bytes()
                .all(|character| character.is_ascii_digit() || (b'a'..=b'f').contains(&character)),
            "deployment identity is not lowercase hexadecimal",
        );
        let word = |index: usize| {
            u64::from_str_radix(&deployment_sha256[index * 16..(index + 1) * 16], 16)
                .expect("deployment identity is not lowercase hexadecimal")
        };
        KernelPlanId {
            word0: word(0),
            word1: word(1),
            word2: word(2),
            word3: word(3),
        }
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

pub fn model_runtime_reports_backend_qualified(runtime: &ModelRuntime) -> (out: bool)
    ensures
        out == (model_runtime_kernel_plan_qualification(runtime)
            == KernelPlanQualification::BackendQualified),
{
    proof {
        lemma_model_runtime_kernel_plan_qualification_projection(runtime);
    }
    match runtime {
        ModelRuntime::Qwen3(qwen) => {
            QWEN3::deployment::reports_backend_qualified(qwen)
        },
        ModelRuntime::Llama3(llama) => {
            LLAMA3::deployment::reports_backend_qualified(llama)
        },
        ModelRuntime::Gemma3Text(gemma) => {
            GEMMA3::deployment::reports_backend_qualified(gemma)
        },
        ModelRuntime::Gemma4Text(gemma) => {
            GEMMA4::deployment::reports_backend_qualified(gemma)
        },
    }
}

// Ordinary-Rust integration tests use an intentionally impossible zero-layer
// model to exercise scheduler mechanics without a GPU deployment. Keep that
// escape hatch unavailable to release builds and to every nonempty model. It
// is absent from the Verus build, whose Engine constructor always requires the
// backend-qualified branch of `model_execution_valid`.
#[cfg(not(verus_only))]
pub fn model_runtime_admitted_by_runtime_gate(
    runtime: &ModelRuntime,
    num_layers: usize,
) -> bool {
    if model_runtime_reports_backend_qualified(runtime) {
        return true;
    }
    if !cfg!(debug_assertions) || num_layers != 0 {
        return false;
    }
    let plan = match runtime {
        ModelRuntime::Qwen3(qwen) => &qwen.kernel_plan,
        ModelRuntime::Llama3(llama) => &llama.kernel_plan,
        ModelRuntime::Gemma3Text(gemma) => &gemma.kernel_plan,
        ModelRuntime::Gemma4Text(gemma) => &gemma.kernel_plan,
    };
    matches!(plan.qualification, KernelPlanQualification::Staged)
        && plan.deployment_sha256.is_empty()
}
// @kernel-bridge-end boundary::tensor_runtime::qualified_kernel_plan_capability

} // verus!
