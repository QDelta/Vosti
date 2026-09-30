//! Family-neutral physical weights for a full-attention SwiGLU decoder.
//!
//! Architecture adapters own checkpoint loading and exact profile policy.
//! This module owns only the common physical layer vocabulary. In particular,
//! Q/K normalization is represented as a closed choice in both executable and
//! tracked state: the disabled arm contains no tensor and no permission.

use crate::model_config::FloatParameterBits;
use crate::boundary::tensor_runtime as RT;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

verus! {

// @kernel-bridge-begin boundary::dense_swiglu_decoder::model_weights_contract

pub enum DenseQkNormWeights {
    Disabled,
    RmsNorm {
        q_weight: RT::Tensor,
        k_weight: RT::Tensor,
    },
}

pub tracked enum DenseQkNormWeightsPerms {
    Disabled,
    RmsNorm {
        q_weight: Tracked<RT::TensorPerm>,
        k_weight: Tracked<RT::TensorPerm>,
    },
}

impl DenseQkNormWeightsPerms {
    pub proof fn tracked_borrow_rms(
        tracked &self,
    ) -> (tracked out: (&RT::TensorPerm, &RT::TensorPerm))
        requires
            match self {
                DenseQkNormWeightsPerms::RmsNorm { .. } => true,
                DenseQkNormWeightsPerms::Disabled => false,
            },
        ensures
            match self {
                DenseQkNormWeightsPerms::RmsNorm { q_weight, k_weight } => {
                    &&& *out.0 == q_weight@
                    &&& *out.1 == k_weight@
                },
                DenseQkNormWeightsPerms::Disabled => false,
            },
    {
        match self {
            DenseQkNormWeightsPerms::RmsNorm { q_weight, k_weight } => {
                (q_weight.borrow(), k_weight.borrow())
            },
            DenseQkNormWeightsPerms::Disabled => {
                assert(false);
                proof_from_false()
            },
        }
    }
}

pub open spec fn qk_norm_weights_valid(
    weights: &DenseQkNormWeights,
    perms: &DenseQkNormWeightsPerms,
) -> bool {
    match (weights, perms) {
        (DenseQkNormWeights::Disabled, DenseQkNormWeightsPerms::Disabled) => true,
        (
            DenseQkNormWeights::RmsNorm { q_weight, k_weight },
            DenseQkNormWeightsPerms::RmsNorm {
                q_weight: q_perm,
                k_weight: k_perm,
            },
        ) => q_weight.id() == q_perm@.id() && k_weight.id() == k_perm@.id(),
        _ => false,
    }
}

pub open spec fn qk_norm_weights_repr_of(
    perms: &DenseQkNormWeightsPerms,
) -> QkNormWeightsRepr {
    match perms {
        DenseQkNormWeightsPerms::Disabled => QkNormWeightsRepr::Disabled,
        DenseQkNormWeightsPerms::RmsNorm {
            q_weight,
            k_weight,
        } => QkNormWeightsRepr::RmsNorm {
            q_weight: q_weight@.repr_1d(),
            k_weight: k_weight@.repr_1d(),
        },
    }
}

pub struct DenseSwiGluLayerWeights {
    pub input_norm: RT::Tensor,
    pub q_proj: RT::Tensor,
    pub k_proj: RT::Tensor,
    pub v_proj: RT::Tensor,
    pub qk_norm: DenseQkNormWeights,
    pub o_proj: RT::Tensor,
    pub post_attn_norm: RT::Tensor,
    pub gate_up_proj: RT::Tensor,
    pub down_proj: RT::Tensor,
}

pub tracked struct DenseSwiGluLayerWeightsPerms {
    pub tracked input_norm: RT::TensorPerm,
    pub tracked q_proj: RT::TensorPerm,
    pub tracked k_proj: RT::TensorPerm,
    pub tracked v_proj: RT::TensorPerm,
    pub tracked qk_norm: DenseQkNormWeightsPerms,
    pub tracked o_proj: RT::TensorPerm,
    pub tracked post_attn_norm: RT::TensorPerm,
    pub tracked gate_up_proj: RT::TensorPerm,
    pub tracked down_proj: RT::TensorPerm,
}

pub open spec fn layer_weights_valid(
    weights: &DenseSwiGluLayerWeights,
    perms: &DenseSwiGluLayerWeightsPerms,
) -> bool {
    weights.input_norm.id() == perms.input_norm.id()
    && weights.q_proj.id() == perms.q_proj.id()
    && weights.k_proj.id() == perms.k_proj.id()
    && weights.v_proj.id() == perms.v_proj.id()
    && qk_norm_weights_valid(&weights.qk_norm, &perms.qk_norm)
    && weights.o_proj.id() == perms.o_proj.id()
    && weights.post_attn_norm.id() == perms.post_attn_norm.id()
    && weights.gate_up_proj.id() == perms.gate_up_proj.id()
    && weights.down_proj.id() == perms.down_proj.id()
    && TS::rectangular(perms.q_proj.repr_2d())
    && TS::rectangular(perms.k_proj.repr_2d())
    && TS::rectangular(perms.v_proj.repr_2d())
    && TS::rectangular(perms.o_proj.repr_2d())
    && TS::rectangular(perms.gate_up_proj.repr_2d())
    && perms.gate_up_proj.repr_2d().len() % 2 == 0
    && TS::rectangular(perms.down_proj.repr_2d())
}

pub open spec fn layer_weights_repr_of(
    perms: &DenseSwiGluLayerWeightsPerms,
    head_dim: nat,
) -> LayerWeightsRepr {
    LayerWeightsRepr {
        input_norm: perms.input_norm.repr_1d(),
        q_proj: perms.q_proj.repr_2d(),
        k_proj: perms.k_proj.repr_2d(),
        v_proj: perms.v_proj.repr_2d(),
        head_dim,
        qk_norm: qk_norm_weights_repr_of(&perms.qk_norm),
        o_proj: perms.o_proj.repr_2d(),
        post_attn_norm: perms.post_attn_norm.repr_1d(),
        gate_up_proj: perms.gate_up_proj.repr_2d(),
        down_proj: perms.down_proj.repr_2d(),
    }
}

// @kernel-bridge-end boundary::dense_swiglu_decoder::model_weights_contract

// Closed executable composition of the optional Q/K-normalization layer.
// The disabled arm moves the original tensors and permissions through the
// identity adapter. The RMS arm is the only arm that can call the kernel.
pub fn apply_qk_norm(
    runtime: &RT::ModelFamilyRuntime,
    weights: &DenseQkNormWeights,
    Tracked(weight_perms): Tracked<&DenseQkNormWeightsPerms>,
    q: RT::Tensor,
    k: RT::Tensor,
    Tracked(q_perm): Tracked<RT::TensorPerm>,
    Tracked(k_perm): Tracked<RT::TensorPerm>,
    Ghost(q_repr): Ghost<Tensor2D>,
    Ghost(k_repr): Ghost<Tensor2D>,
    Ghost(epsilon): Ghost<FloatParameterBits>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: ((RT::Tensor, RT::Tensor),
            (Tracked<RT::TensorPerm>, Tracked<RT::TensorPerm>)))
    requires
        qk_norm_weights_valid(weights, weight_perms),
        RT::tensor_repr_2d(q_perm, q, q_repr),
        RT::tensor_repr_2d(k_perm, k, k_repr),
        q_repr.len() == k_repr.len(),
        RT::dense_swiglu_runtime_qk_norm_matches(
            runtime,
            qk_norm_weights_kind(qk_norm_weights_repr_of(weight_perms)),
        ),
        RT::dense_swiglu_runtime_rms_norm_matches(runtime, epsilon),
    ensures ({
        let ((normalized_q, normalized_k), (normalized_q_perm, normalized_k_perm)) = out;
        let normalized = RT::apply_qk_norm_repr(
            q_repr, k_repr, qk_norm_weights_repr_of(weight_perms), epsilon,
        );
        &&& RT::tensor_repr_2d(
            normalized_q_perm@, normalized_q, normalized.0,
        )
        &&& RT::tensor_repr_2d(
            normalized_k_perm@, normalized_k, normalized.1,
        )
    }),
{
    match weights {
        DenseQkNormWeights::Disabled => {
            proof {
                reveal(qk_norm_weights_valid);
                reveal(qk_norm_weights_repr_of);
                match weight_perms {
                    DenseQkNormWeightsPerms::Disabled => {},
                    DenseQkNormWeightsPerms::RmsNorm { .. } => {
                        assert(false);
                    },
                }
                reveal(RT::apply_qk_norm_repr);
            }
            RT::bypass_qk_norm(
                q, k, Tracked(q_perm), Tracked(k_perm),
                Ghost(q_repr), Ghost(k_repr), Ghost(epsilon),
            )
        },
        DenseQkNormWeights::RmsNorm { q_weight, k_weight } => {
            proof {
                reveal(qk_norm_weights_valid);
                reveal(RT::dense_swiglu_runtime_qk_norm_matches);
                match weight_perms {
                    DenseQkNormWeightsPerms::RmsNorm { .. } => {},
                    DenseQkNormWeightsPerms::Disabled => { assert(false); },
                }
            }
            let tracked (q_weight_perm, k_weight_perm) =
                weight_perms.tracked_borrow_rms();
            proof {
                reveal(RT::tensor_repr_1d);
                assert(q_weight.id() == q_weight_perm.id());
                assert(k_weight.id() == k_weight_perm.id());
            }
            let result = RT::qk_norm(
                runtime, &q, &k, q_weight, k_weight,
                Tracked(&q_perm), Tracked(&k_perm),
                Tracked(q_weight_perm), Tracked(k_weight_perm),
                Ghost(q_repr), Ghost(k_repr),
                Ghost(q_weight_perm.repr_1d()),
                Ghost(k_weight_perm.repr_1d()),
                Ghost(epsilon),
                Ghost(scope),
            );
            proof {
                reveal(qk_norm_weights_repr_of);
                reveal(RT::apply_qk_norm_repr);
            }
            result
        },
    }
}

} // verus!
