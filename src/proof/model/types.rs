//! Semantic model representations, configuration projections, and validity predicates.

use crate::model_config::{AttentionGeometry, AttentionKind, DenseGeometry, FloatParameterBits, ModelArchitecture};
use crate::boundary::model_families::gemma3::config::{GEMMA3_GLOBAL_ROPE_FACTOR_F64_BITS, GEMMA3_GLOBAL_ROPE_THETA_F64_BITS, GEMMA3_LOCAL_ROPE_THETA_F64_BITS, GEMMA3_RMS_NORM_EPSILON_F64_BITS, Gemma3Config};
use crate::boundary::model_families::gemma4::config::Gemma4Config;
use crate::boundary::model_families::llama3::config::{LLAMA3_RMS_NORM_EPSILON_F64_BITS, LLAMA3_ROPE_FACTOR_32_F64_BITS, LLAMA3_ROPE_FACTOR_8_F64_BITS, LLAMA3_ROPE_HIGH_FREQUENCY_FACTOR_F64_BITS, LLAMA3_ROPE_LOW_FREQUENCY_FACTOR_F64_BITS, LLAMA3_ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS, LLAMA3_ROPE_THETA_F64_BITS, Llama3Config};
use crate::boundary::model_families::qwen3::config::{QWEN3_RMS_NORM_EPSILON_F64_BITS, QWEN3_ROPE_THETA_F64_BITS, Qwen3Config};
use crate::types::*;
use crate::proof::tensor::types::*;
#[cfg(verus_only)]
use crate::boundary::scalar::{float_parameter_scalar_repr, positive_float_parameter_valid};
use vstd::prelude::*;

verus! {

// Proof-facing attention configuration.  Encoding the window in the sliding
// variant makes it impossible for an architecture proof to select SWA while
// silently falling back to a family-global constant.
pub enum AttentionConfigRepr {
    SlidingWindow(nat),
    Full,
}

pub open spec fn qwen3_rms_norm_epsilon_repr() -> FloatParameterBits {
    FloatParameterBits {
        bits: QWEN3_RMS_NORM_EPSILON_F64_BITS,
    }
}

pub open spec fn llama3_rms_norm_epsilon_repr() -> FloatParameterBits {
    FloatParameterBits {
        bits: LLAMA3_RMS_NORM_EPSILON_F64_BITS,
    }
}

// Values below describe one admitted dense checkpoint, not a family-wide
// profile.  Any field that changes tensor interpretation or the numerical
// forward result must survive checkpoint loading into this proof-visible
// representation.  Family admission predicates below constrain the fields
// that would otherwise change the verified composition itself.
pub struct DenseGeometryRepr {
    pub vocab_size: nat,
    pub hidden_size: nat,
    pub intermediate_size: nat,
    pub num_layers: nat,
    pub num_attention_heads: nat,
    pub num_key_value_heads: nat,
    pub head_dim: nat,
    pub max_position_embeddings: nat,
}

pub struct AttentionGeometryRepr {
    pub num_attention_heads: nat,
    pub num_key_value_heads: nat,
    pub head_dim: nat,
}

pub open spec fn physical_attention_geometry_repr(geometry: AttentionGeometry)
    -> AttentionGeometryRepr
{
    AttentionGeometryRepr {
        num_attention_heads: geometry.num_attention_heads as nat,
        num_key_value_heads: geometry.num_key_value_heads as nat,
        head_dim: geometry.head_dim as nat,
    }
}

pub open spec fn attention_geometry_repr(
    geometry: DenseGeometryRepr,
) -> AttentionGeometryRepr {
    AttentionGeometryRepr {
        num_attention_heads: geometry.num_attention_heads,
        num_key_value_heads: geometry.num_key_value_heads,
        head_dim: geometry.head_dim,
    }
}

pub open spec fn dense_geometry_repr(
    geometry: DenseGeometry,
) -> DenseGeometryRepr {
    DenseGeometryRepr {
        vocab_size: geometry.vocab_size as nat,
        hidden_size: geometry.hidden_size as nat,
        intermediate_size: geometry.intermediate_size as nat,
        num_layers: geometry.num_layers as nat,
        num_attention_heads: geometry.num_attention_heads as nat,
        num_key_value_heads: geometry.num_key_value_heads as nat,
        head_dim: geometry.head_dim as nat,
        max_position_embeddings: geometry.max_position_embeddings as nat,
    }
}

pub open spec fn dense_geometry_valid(geometry: DenseGeometryRepr) -> bool {
    geometry.vocab_size > 0
    && geometry.hidden_size > 0
    && geometry.intermediate_size > 0
    && geometry.num_layers > 0
    && geometry.num_attention_heads > 0
    && geometry.num_key_value_heads > 0
    && geometry.head_dim > 0
    && geometry.head_dim % 2 == 0
    && geometry.max_position_embeddings > 0
    && geometry.num_attention_heads % geometry.num_key_value_heads == 0
}

pub enum RotaryScalingRepr {
    None,
    Proportional {
        factor: FloatParameterBits,
        partial_rotary_factor: FloatParameterBits,
    },
    Linear {
        factor: FloatParameterBits,
    },
    Llama3 {
        factor: FloatParameterBits,
        low_frequency_factor: FloatParameterBits,
        high_frequency_factor: FloatParameterBits,
        original_max_position_embeddings: nat,
    },
}

pub struct RotaryConfigRepr {
    pub theta: FloatParameterBits,
    pub scaling: RotaryScalingRepr,
}

pub open spec fn rotary_config_valid(config: RotaryConfigRepr) -> bool {
    positive_float_parameter_valid(config.theta)
    && match config.scaling {
        RotaryScalingRepr::None => true,
        RotaryScalingRepr::Proportional { factor, partial_rotary_factor } =>
            positive_float_parameter_valid(factor)
            && positive_float_parameter_valid(partial_rotary_factor),
        RotaryScalingRepr::Linear { factor } =>
            positive_float_parameter_valid(factor),
        RotaryScalingRepr::Llama3 {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_max_position_embeddings,
        } =>
            positive_float_parameter_valid(factor)
            && positive_float_parameter_valid(low_frequency_factor)
            && positive_float_parameter_valid(high_frequency_factor)
            && original_max_position_embeddings > 0,
    }
}

pub struct RmsNormConfigRepr {
    pub epsilon: FloatParameterBits,
}

pub enum AttentionScaleRepr {
    InverseSqrtHeadDim,
    InverseSqrtParameter(FloatParameterBits),
}

// Immutable model geometry and the kernel's scale policy. Launch tiles and
// scheduling state deliberately do not belong to this semantic interface.
pub struct AttentionParametersRepr {
    pub geometry: AttentionGeometryRepr,
    pub scale: AttentionScaleRepr,
}

pub enum ActivationKind {
    Silu,
    GeluTanh,
}

pub enum QkNormKind {
    RmsNorm,
    Disabled,
}

// This record is intentionally not a generic forward-program selector.  Each
// family admits exactly one composition below; changing one of these values
// requires a new/updated family proof instead of a runtime branch.
pub struct DenseCompositionRepr {
    pub activation: ActivationKind,
    pub qk_norm: QkNormKind,
    pub attention_bias: bool,
    pub attention_dropout_bits: u64,
    pub tie_word_embeddings: bool,
}

// Proof-visible configuration of the dense pre-norm, full-attention, SwiGLU
// decoder shared by Qwen3 and Llama3. Closed architecture variants retain
// family identity; this record contains only the reusable composition data.
pub struct DenseSwiGluDecoderConfigRepr {
    pub geometry: DenseGeometryRepr,
    pub rms_norm: RmsNormConfigRepr,
    pub rotary: RotaryConfigRepr,
    pub attention_scale: AttentionScaleRepr,
    pub attention: AttentionConfigRepr,
    pub composition: DenseCompositionRepr,
}

// The common forward fold consumes only these numerical parameters. Geometry
// is carried by the checked tensor shapes and layer records; composition
// validity separately binds optional Q/K normalization to every layer. Keeping
// this projection small avoids exposing unused admission fields to the solver
// while retaining the complete RoPE policy as part of semantic identity.
pub struct DenseSwiGluForwardConfigRepr {
    pub rms_norm_epsilon: FloatParameterBits,
    pub rotary: RotaryConfigRepr,
}

pub open spec fn dense_swiglu_forward_config_repr(
    config: DenseSwiGluDecoderConfigRepr,
) -> DenseSwiGluForwardConfigRepr {
    DenseSwiGluForwardConfigRepr {
        rms_norm_epsilon: config.rms_norm.epsilon,
        rotary: config.rotary,
    }
}

pub type Qwen3ModelWeightsExtensionRepr = DenseSwiGluDecoderConfigRepr;

pub type Llama3ModelWeightsExtensionRepr = DenseSwiGluDecoderConfigRepr;

pub open spec fn qwen3_config_repr(
    config: Qwen3Config,
) -> Qwen3ModelWeightsExtensionRepr {
    DenseSwiGluDecoderConfigRepr {
        geometry: dense_geometry_repr(config.geometry),
        rms_norm: RmsNormConfigRepr {
            epsilon: config.rms_norm_epsilon,
        },
        rotary: RotaryConfigRepr {
            theta: config.rope_theta,
            scaling: RotaryScalingRepr::None,
        },
        attention_scale: AttentionScaleRepr::InverseSqrtHeadDim,
        attention: AttentionConfigRepr::Full,
        composition: DenseCompositionRepr {
            activation: ActivationKind::Silu,
            qk_norm: QkNormKind::RmsNorm,
            attention_bias: false,
            attention_dropout_bits: 0,
            tie_word_embeddings: config.tie_word_embeddings,
        },
    }
}

pub open spec fn llama3_config_repr(
    config: Llama3Config,
) -> Llama3ModelWeightsExtensionRepr {
    DenseSwiGluDecoderConfigRepr {
        geometry: dense_geometry_repr(config.geometry),
        rms_norm: RmsNormConfigRepr {
            epsilon: config.rms_norm_epsilon,
        },
        rotary: RotaryConfigRepr {
            theta: config.rope_theta,
            scaling: RotaryScalingRepr::Llama3 {
                factor: config.rope_factor,
                low_frequency_factor: config.rope_low_frequency_factor,
                high_frequency_factor: config.rope_high_frequency_factor,
                original_max_position_embeddings:
                    config.rope_original_max_position_embeddings as nat,
            },
        },
        attention_scale: AttentionScaleRepr::InverseSqrtHeadDim,
        attention: AttentionConfigRepr::Full,
        composition: DenseCompositionRepr {
            activation: ActivationKind::Silu,
            qk_norm: QkNormKind::Disabled,
            attention_bias: false,
            attention_dropout_bits: 0,
            tie_word_embeddings: config.tie_word_embeddings,
        },
    }
}

pub closed spec fn dense_swiglu_decoder_config_valid(
    config: DenseSwiGluDecoderConfigRepr,
) -> bool {
    dense_geometry_valid(config.geometry)
    && positive_float_parameter_valid(config.rms_norm.epsilon)
    && rotary_config_valid(config.rotary)
    && match config.attention_scale {
        AttentionScaleRepr::InverseSqrtHeadDim => true,
        _ => false,
    }
    && match config.attention {
        AttentionConfigRepr::Full => true,
        _ => false,
    }
    && match config.composition.activation {
        ActivationKind::Silu => true,
        _ => false,
    }
    && !config.composition.attention_bias
    && config.composition.attention_dropout_bits == 0
}

// Keep the full admission predicate opaque in large model proofs.  Clients
// that need to reason about tensor geometry can request only that projection,
// avoiding all numerical/composition conjuncts in their solver context.
pub proof fn lemma_dense_swiglu_decoder_config_valid_implies_geometry_valid(
    config: DenseSwiGluDecoderConfigRepr,
)
    requires dense_swiglu_decoder_config_valid(config),
    ensures dense_geometry_valid(config.geometry),
{
    reveal(dense_swiglu_decoder_config_valid);
}

pub closed spec fn qwen3_config_valid(
    config: Qwen3ModelWeightsExtensionRepr,
) -> bool {
    dense_swiglu_decoder_config_valid(config)
    && config.rms_norm.epsilon.bits == QWEN3_RMS_NORM_EPSILON_F64_BITS
    && config.rotary.theta.bits == QWEN3_ROPE_THETA_F64_BITS
    && match config.rotary.scaling {
        RotaryScalingRepr::None => true,
        _ => false,
    }
    && match config.composition.qk_norm {
        QkNormKind::RmsNorm => true,
        _ => false,
    }
}

// Llama 3.1/3.3 and Llama 3.2 share one forward composition but use two
// reviewed Llama3-RoPE factors. Geometry and the tied-weight loading choice
// remain explicit; exact checkpoint profiles close the executable scope.
pub closed spec fn llama3_config_valid(
    config: Llama3ModelWeightsExtensionRepr,
) -> bool {
    dense_swiglu_decoder_config_valid(config)
    && config.rms_norm.epsilon.bits == LLAMA3_RMS_NORM_EPSILON_F64_BITS
    && config.rotary.theta.bits == LLAMA3_ROPE_THETA_F64_BITS
    && match config.rotary.scaling {
        RotaryScalingRepr::Llama3 {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_max_position_embeddings,
        } => {
            &&& (factor.bits == LLAMA3_ROPE_FACTOR_8_F64_BITS
                || factor.bits == LLAMA3_ROPE_FACTOR_32_F64_BITS)
            &&& low_frequency_factor.bits
                == LLAMA3_ROPE_LOW_FREQUENCY_FACTOR_F64_BITS
            &&& high_frequency_factor.bits
                == LLAMA3_ROPE_HIGH_FREQUENCY_FACTOR_F64_BITS
            &&& original_max_position_embeddings
                == LLAMA3_ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS as nat
        },
        _ => false,
    }
    && match config.composition.qk_norm {
        QkNormKind::Disabled => true,
        _ => false,
    }
}

pub proof fn lemma_llama3_config_valid_implies_dense_config_valid(
    config: Llama3ModelWeightsExtensionRepr,
)
    requires llama3_config_valid(config),
    ensures dense_swiglu_decoder_config_valid(config),
{
    reveal(llama3_config_valid);
}

pub proof fn lemma_llama3_config_valid_implies_qk_norm_disabled(
    config: Llama3ModelWeightsExtensionRepr,
)
    requires llama3_config_valid(config),
    ensures config.composition.qk_norm == QkNormKind::Disabled,
{
    reveal(llama3_config_valid);
}

pub proof fn lemma_llama3_config_valid_implies_rms_norm_epsilon_identity(
    config: Llama3ModelWeightsExtensionRepr,
)
    requires llama3_config_valid(config),
    ensures config.rms_norm.epsilon == llama3_rms_norm_epsilon_repr(),
{
    reveal(llama3_config_valid);
    reveal(llama3_rms_norm_epsilon_repr);
}

pub proof fn lemma_qwen3_config_valid_implies_dense_config_valid(
    config: Qwen3ModelWeightsExtensionRepr,
)
    requires qwen3_config_valid(config),
    ensures dense_swiglu_decoder_config_valid(config),
{
    reveal(qwen3_config_valid);
}

pub proof fn lemma_qwen3_config_valid_implies_qk_rms_norm(
    config: Qwen3ModelWeightsExtensionRepr,
)
    requires qwen3_config_valid(config),
    ensures config.composition.qk_norm == QkNormKind::RmsNorm,
{
    reveal(qwen3_config_valid);
}

pub proof fn lemma_qwen3_config_valid_implies_rms_norm_epsilon_identity(
    config: Qwen3ModelWeightsExtensionRepr,
)
    requires qwen3_config_valid(config),
    ensures config.rms_norm.epsilon == qwen3_rms_norm_epsilon_repr(),
{
    reveal(qwen3_config_valid);
    reveal(qwen3_rms_norm_epsilon_repr);
}

pub proof fn lemma_qwen3_config_valid_implies_rotary_identity(
    config: Qwen3ModelWeightsExtensionRepr,
)
    requires qwen3_config_valid(config),
    ensures config.rotary == (RotaryConfigRepr {
        theta: FloatParameterBits {
            bits: QWEN3_ROPE_THETA_F64_BITS,
        },
        scaling: RotaryScalingRepr::None,
    }),

{
    reveal(qwen3_config_valid);
}

pub open spec fn attention_config_kind(
    config: AttentionConfigRepr,
) -> AttentionKind {
    match config {
        AttentionConfigRepr::SlidingWindow(_) => AttentionKind::SlidingWindow,
        AttentionConfigRepr::Full => AttentionKind::Full,
    }
}

pub open spec fn attention_config_valid(
    config: AttentionConfigRepr,
) -> bool {
    match config {
        AttentionConfigRepr::SlidingWindow(window_size) => window_size > 0,
        AttentionConfigRepr::Full => true,
    }
}

// Family composition uses these neutral layer-facing adapters rather than
// unfolding the attention policy representation inside whole-model proofs.
// The extra naming boundary is intentional: it keeps solver contexts stable
// as additional families reuse the same attention configuration vocabulary.
pub open spec fn layer_attention_kind(
    config: AttentionConfigRepr,
) -> AttentionKind {
    attention_config_kind(config)
}

pub open spec fn layer_attention_config_valid(
    config: AttentionConfigRepr,
) -> bool {
    attention_config_valid(config)
}

// Q/K normalization is a closed composition choice, not a tensor role that
// every dense decoder is forced to own.  Keeping the disabled case free of
// tensor fields prevents a family without Q/K normalization from fabricating
// semantically meaningful unit weights merely to fit the common record.
pub enum QkNormWeightsRepr {
    Disabled,
    RmsNorm {
        q_weight: Tensor1D,
        k_weight: Tensor1D,
    },
}

pub open spec fn qk_norm_weights_kind(
    weights: QkNormWeightsRepr,
) -> QkNormKind {
    match weights {
        QkNormWeightsRepr::Disabled => QkNormKind::Disabled,
        QkNormWeightsRepr::RmsNorm { .. } => QkNormKind::RmsNorm,
    }
}

pub open spec fn rms_q_norm_weight(
    weights: QkNormWeightsRepr,
) -> Tensor1D
    recommends qk_norm_weights_kind(weights) == QkNormKind::RmsNorm,
{
    match weights {
        QkNormWeightsRepr::Disabled => Seq::empty(),
        QkNormWeightsRepr::RmsNorm { q_weight, .. } => q_weight,
    }
}

pub open spec fn rms_k_norm_weight(
    weights: QkNormWeightsRepr,
) -> Tensor1D
    recommends qk_norm_weights_kind(weights) == QkNormKind::RmsNorm,
{
    match weights {
        QkNormWeightsRepr::Disabled => Seq::empty(),
        QkNormWeightsRepr::RmsNorm { k_weight, .. } => k_weight,
    }
}

pub open spec fn layer_qk_norm_matches_composition(
    weights: LayerWeightsRepr,
    composition: DenseCompositionRepr,
) -> bool {
    qk_norm_weights_kind(weights.qk_norm) == composition.qk_norm
}

// Per-layer weight tensor reprs. These are ghost-only; the matching
// family-owned physical weight bundles live under
// `boundary/model_families/<family>/weights.rs`. `head_dim` is explicit so
// attention geometry does not depend on whether Q/K normalization is enabled.
pub struct LayerWeightsRepr {
    pub input_norm:    Tensor1D,
    pub q_proj:        Tensor2D,
    pub k_proj:        Tensor2D,
    pub v_proj:        Tensor2D,
    pub head_dim:      nat,
    pub qk_norm:       QkNormWeightsRepr,
    pub o_proj:        Tensor2D,
    pub post_attn_norm: Tensor1D,
    pub gate_up_proj:  Tensor2D,
    pub down_proj:     Tensor2D,
}

// Static numerical policies for the shared four-norm gated decoder. These
// describe operator identity, not launch configuration. UnitOffset names the
// existing closed offset-weight normalization operator (epsilon 1e-6); the
// direct-weight operator retains its checkpoint epsilon explicitly.
pub enum NormPolicyRepr {
    UnitOffset,
    Direct(FloatParameterBits),
}

// Scalar identity of the existing fixed-epsilon UnitOffset policy. Family
// admission checks this same binary64 value; this is not a numerical axiom.
pub open spec fn unit_offset_norm_epsilon_repr() -> FloatParameterBits {
    FloatParameterBits { bits: GEMMA3_RMS_NORM_EPSILON_F64_BITS }
}

pub struct FourNormGatedLayerRepr {
    pub common: LayerWeightsRepr,
    pub pre_feedforward_norm: Tensor1D,
    pub post_feedforward_norm: Tensor1D,
    pub norm: NormPolicyRepr,
    pub qk_norm: NormPolicyRepr,
    pub rotary: RotaryConfigRepr,
    pub value_norm_epsilon: Option<FloatParameterBits>,
    pub layer_scale: Option<Tensor1D>,
}

// Row-local policies are part of immutable model identity. Four-norm decoder
// semantics pair the common layer record with this extension at the same
// index; each admitted family binds these policies to its own exact profile.
pub struct FourNormGatedRowParametersRepr {
    pub norm: NormPolicyRepr,
    pub qk_norm: NormPolicyRepr,
    pub rotary: RotaryConfigRepr,
    pub value_norm_epsilon: Option<FloatParameterBits>,
    pub layer_scale: Option<Tensor1D>,
}

pub struct FourNormGatedLayerExtensionRepr {
    pub pre_feedforward_norm:  Tensor1D,
    pub post_feedforward_norm: Tensor1D,
    pub attention:             AttentionConfigRepr,
    pub attention_scale:       AttentionScaleRepr,
    pub row_parameters: FourNormGatedRowParametersRepr,
}

pub type Gemma3LayerWeightsExtensionRepr = FourNormGatedLayerExtensionRepr;

pub open spec fn gemma3_row_parameters_repr(kind: AttentionKind)
    -> FourNormGatedRowParametersRepr
{
    FourNormGatedRowParametersRepr {
        norm: NormPolicyRepr::UnitOffset,
        qk_norm: NormPolicyRepr::UnitOffset,
        rotary: match kind {
            AttentionKind::SlidingWindow => RotaryConfigRepr {
                theta: FloatParameterBits { bits: GEMMA3_LOCAL_ROPE_THETA_F64_BITS },
                scaling: RotaryScalingRepr::None,
            },
            AttentionKind::Full => RotaryConfigRepr {
                theta: FloatParameterBits { bits: GEMMA3_GLOBAL_ROPE_THETA_F64_BITS },
                scaling: RotaryScalingRepr::Linear {
                    factor: FloatParameterBits { bits: GEMMA3_GLOBAL_ROPE_FACTOR_F64_BITS },
                },
            },
        },
        value_norm_epsilon: None,
        layer_scale: None,
    }
}

pub struct FourNormGatedDecoderConfigRepr {
    // Family configuration belongs to the architecture payload rather than
    // the scheduler-visible common weight identity.  Additional qualified
    // Gemma sizes can therefore reuse the same Engine and proof composition.
    pub geometry: DenseGeometryRepr,
    pub rms_norm: RmsNormConfigRepr,
    pub local_rotary: RotaryConfigRepr,
    pub global_rotary: RotaryConfigRepr,
    pub attention_scale: AttentionScaleRepr,
    pub sliding_window: nat,
    pub composition: DenseCompositionRepr,
    pub layers: Seq<FourNormGatedLayerExtensionRepr>,
    pub final_norm_policy: NormPolicyRepr,
    pub final_logit_softcap: Option<FloatParameterBits>,
}

pub type Gemma3ModelWeightsExtensionRepr = FourNormGatedDecoderConfigRepr;

pub open spec fn gemma3_attention_scale_repr(
    config: Gemma3Config,
) -> AttentionScaleRepr {
    AttentionScaleRepr::InverseSqrtParameter(
        config.query_pre_attention_scalar,
    )
}

// Immutable configuration identity shared by checkpoint permissions and the
// retained runtime capability. Gemma's attention schedule belongs here
// because it changes both RoPE selection and attention masking. Weight
// tensors remain in the architecture representation, not this record.
pub struct Gemma3DeploymentConfigRepr {
    pub geometry: DenseGeometryRepr,
    pub rms_norm: RmsNormConfigRepr,
    pub local_rotary: RotaryConfigRepr,
    pub global_rotary: RotaryConfigRepr,
    pub attention_scale: AttentionScaleRepr,
    pub sliding_window: nat,
    pub composition: DenseCompositionRepr,
    pub layer_attention: Seq<AttentionConfigRepr>,
}

pub enum ModelDeploymentConfigRepr {
    Qwen3(Qwen3ModelWeightsExtensionRepr),
    Llama3(Llama3ModelWeightsExtensionRepr),
    Gemma3Text(Gemma3DeploymentConfigRepr),
    Gemma4Text(Gemma4DeploymentConfigRepr),
}

pub open spec fn gemma3_config_repr(
    config: Gemma3Config,
    layers: Seq<Gemma3LayerWeightsExtensionRepr>,
) -> Gemma3ModelWeightsExtensionRepr {
    Gemma3ModelWeightsExtensionRepr {
        geometry: dense_geometry_repr(config.geometry),
        rms_norm: RmsNormConfigRepr {
            epsilon: config.rms_norm_epsilon,
        },
        local_rotary: RotaryConfigRepr {
            theta: config.local_rope_theta,
            scaling: RotaryScalingRepr::None,
        },
        global_rotary: RotaryConfigRepr {
            theta: config.global_rope_theta,
            scaling: RotaryScalingRepr::Linear {
                factor: config.global_rope_factor,
            },
        },
        attention_scale: gemma3_attention_scale_repr(config),
        sliding_window: config.sliding_window as nat,
        composition: DenseCompositionRepr {
            activation: ActivationKind::GeluTanh,
            qk_norm: QkNormKind::RmsNorm,
            attention_bias: false,
            attention_dropout_bits: 0,
            tie_word_embeddings: true,
        },
        layers,
        final_norm_policy: NormPolicyRepr::UnitOffset,
        final_logit_softcap: None,
    }
}

pub open spec fn gemma3_layer_attention_config(
    config: Gemma3Config,
    kind: AttentionKind,
) -> AttentionConfigRepr {
    match kind {
        AttentionKind::SlidingWindow =>
            AttentionConfigRepr::SlidingWindow(config.sliding_window as nat),
        AttentionKind::Full => AttentionConfigRepr::Full,
    }
}

pub open spec fn gemma3_deployment_config_repr(
    config: Gemma3Config,
    layer_attention_kinds: Seq<AttentionKind>,
) -> Gemma3DeploymentConfigRepr {
    Gemma3DeploymentConfigRepr {
        geometry: dense_geometry_repr(config.geometry),
        rms_norm: RmsNormConfigRepr {
            epsilon: config.rms_norm_epsilon,
        },
        local_rotary: RotaryConfigRepr {
            theta: config.local_rope_theta,
            scaling: RotaryScalingRepr::None,
        },
        global_rotary: RotaryConfigRepr {
            theta: config.global_rope_theta,
            scaling: RotaryScalingRepr::Linear {
                factor: config.global_rope_factor,
            },
        },
        attention_scale: gemma3_attention_scale_repr(config),
        sliding_window: config.sliding_window as nat,
        composition: DenseCompositionRepr {
            activation: ActivationKind::GeluTanh,
            qk_norm: QkNormKind::RmsNorm,
            attention_bias: false,
            attention_dropout_bits: 0,
            tie_word_embeddings: true,
        },
        layer_attention: Seq::new(layer_attention_kinds.len(), |i: int|
            gemma3_layer_attention_config(config, layer_attention_kinds[i])),
    }
}

pub open spec fn gemma3_family_config_valid(
    geometry: DenseGeometryRepr,
    rms_norm: RmsNormConfigRepr,
    local_rotary: RotaryConfigRepr,
    global_rotary: RotaryConfigRepr,
    attention_scale: AttentionScaleRepr,
    sliding_window: nat,
    composition: DenseCompositionRepr,
    layer_attention: Seq<AttentionConfigRepr>,
) -> bool {
    dense_geometry_valid(geometry)
    && geometry.num_layers == layer_attention.len()
    && positive_float_parameter_valid(rms_norm.epsilon)
    && rms_norm.epsilon.bits == GEMMA3_RMS_NORM_EPSILON_F64_BITS
    && sliding_window > 0
    && rotary_config_valid(local_rotary)
    && rotary_config_valid(global_rotary)
    && local_rotary.theta.bits == GEMMA3_LOCAL_ROPE_THETA_F64_BITS
    && global_rotary.theta.bits == GEMMA3_GLOBAL_ROPE_THETA_F64_BITS
    && match local_rotary.scaling {
        RotaryScalingRepr::None => true,
        _ => false,
    }
    && match global_rotary.scaling {
        RotaryScalingRepr::Linear { factor } =>
            factor.bits == GEMMA3_GLOBAL_ROPE_FACTOR_F64_BITS,
        _ => false,
    }
    && match attention_scale {
        AttentionScaleRepr::InverseSqrtParameter(parameter) =>
            positive_float_parameter_valid(parameter),
        _ => false,
    }
    && match composition.activation {
        ActivationKind::GeluTanh => true,
        _ => false,
    }
    && match composition.qk_norm {
        QkNormKind::RmsNorm => true,
        _ => false,
    }
    && !composition.attention_bias
    && composition.attention_dropout_bits == 0
    && composition.tie_word_embeddings
    && forall|i: int| 0 <= i < layer_attention.len() ==>
        #[trigger] layer_attention_config_valid(layer_attention[i])
}

pub open spec fn gemma3_deployment_config_valid(
    config: Gemma3DeploymentConfigRepr,
) -> bool {
    gemma3_family_config_valid(
        config.geometry,
        config.rms_norm,
        config.local_rotary,
        config.global_rotary,
        config.attention_scale,
        config.sliding_window,
        config.composition,
        config.layer_attention,
    )
}

pub open spec fn gemma3_config_valid(
    config: Gemma3ModelWeightsExtensionRepr,
) -> bool {
    config.final_norm_policy == NormPolicyRepr::UnitOffset
    && config.final_logit_softcap == None
    && gemma3_family_config_valid(
        config.geometry,
        config.rms_norm,
        config.local_rotary,
        config.global_rotary,
        config.attention_scale,
        config.sliding_window,
        config.composition,
        config.layers.map_values(
            |layer: Gemma3LayerWeightsExtensionRepr| layer.attention,
        ),
    )
    && forall|i: int| 0 <= i < config.layers.len() ==>
        #[trigger] config.layers[i].attention_scale == config.attention_scale
    && forall|i: int| 0 <= i < config.layers.len() ==>
        #[trigger] config.layers[i].row_parameters == gemma3_row_parameters_repr(
            layer_attention_kind(config.layers[i].attention))
}

pub proof fn lemma_gemma3_config_valid_implies_deployment_projection(
    config: Gemma3ModelWeightsExtensionRepr,
)
    requires gemma3_config_valid(config),
    ensures gemma3_deployment_config_valid(Gemma3DeploymentConfigRepr {
        geometry: config.geometry,
        rms_norm: config.rms_norm,
        local_rotary: config.local_rotary,
        global_rotary: config.global_rotary,
        attention_scale: config.attention_scale,
        sliding_window: config.sliding_window,
        composition: config.composition,
        layer_attention: config.layers.map_values(
            |layer: Gemma3LayerWeightsExtensionRepr| layer.attention,
        ),
    }),

{
    reveal(gemma3_config_valid);
    reveal(gemma3_deployment_config_valid);
}

pub open spec fn model_deployment_config_architecture(
    config: ModelDeploymentConfigRepr,
) -> ModelArchitecture {
    match config {
        ModelDeploymentConfigRepr::Qwen3(_) => ModelArchitecture::Qwen3,
        ModelDeploymentConfigRepr::Llama3(_) => ModelArchitecture::Llama3,
        ModelDeploymentConfigRepr::Gemma3Text(_) => ModelArchitecture::Gemma3Text,
        ModelDeploymentConfigRepr::Gemma4Text(_) => ModelArchitecture::Gemma4Text,
    }
}

pub open spec fn model_deployment_config_valid(
    config: ModelDeploymentConfigRepr,
) -> bool {
    match config {
        ModelDeploymentConfigRepr::Qwen3(qwen) => qwen3_config_valid(qwen),
        ModelDeploymentConfigRepr::Llama3(llama) => llama3_config_valid(llama),
        ModelDeploymentConfigRepr::Gemma3Text(gemma) =>
            gemma3_deployment_config_valid(gemma),
        ModelDeploymentConfigRepr::Gemma4Text(gemma) =>
            gemma4_deployment_config_valid(gemma),
    }
}

// Closed architecture payload passed only at the model-forward boundary.
// Keeping it separate from ModelWeightsRepr prevents equality-heavy
// scheduler/refinement proofs from recursively expanding Gemma's per-layer
// extension sequence.
pub enum ModelWeightsArchitectureRepr {
    Qwen3(Qwen3ModelWeightsExtensionRepr),
    Llama3(Llama3ModelWeightsExtensionRepr),
    Gemma3Text(Gemma3ModelWeightsExtensionRepr),
    Gemma4Text(Gemma4ModelWeightsExtensionRepr),
}

pub open spec fn model_weights_deployment_config_repr(
    architecture: ModelWeightsArchitectureRepr,
) -> ModelDeploymentConfigRepr {
    match architecture {
        ModelWeightsArchitectureRepr::Gemma4Text(gemma) =>
            ModelDeploymentConfigRepr::Gemma4Text(gemma4_deployment_projection(gemma)),
        ModelWeightsArchitectureRepr::Qwen3(qwen) =>
            ModelDeploymentConfigRepr::Qwen3(qwen),
        ModelWeightsArchitectureRepr::Llama3(llama) =>
            ModelDeploymentConfigRepr::Llama3(llama),
        ModelWeightsArchitectureRepr::Gemma3Text(gemma) =>
            ModelDeploymentConfigRepr::Gemma3Text(Gemma3DeploymentConfigRepr {
                geometry: gemma.geometry,
                rms_norm: gemma.rms_norm,
                local_rotary: gemma.local_rotary,
                global_rotary: gemma.global_rotary,
                attention_scale: gemma.attention_scale,
                sliding_window: gemma.sliding_window,
                composition: gemma.composition,
                layer_attention: gemma.layers.map_values(
                    |layer: Gemma3LayerWeightsExtensionRepr| layer.attention,
                ),
            }),
    }
}

pub struct ModelWeightsRepr {
    pub architecture: ModelArchitecture,
    pub embed_weight: Tensor2D,
    pub layers:       Seq<LayerWeightsRepr>,
    pub final_norm:   Tensor1D,
    pub lm_head:      Tensor2D,
}

// Complete ghost identity of one semantic model.  `ModelWeightsRepr` retains
// the compact common record used by scheduler/cache proofs, while this value
// pairs it with every architecture-specific role and configuration consumed by
// whole-model semantics.  Trace equality must use this type: equality of the
// common record alone does not identify a Gemma model.
pub struct SemanticModelRepr {
    pub weights: ModelWeightsRepr,
    pub architecture: ModelWeightsArchitectureRepr,
}

pub open spec fn semantic_model_repr_valid(model: SemanticModelRepr) -> bool {
    model_weights_architecture_repr_valid(model.weights, model.architecture)
}

// This small invariant is the only coupling between the stable architecture
// tag used by scheduler state and the architecture payload used by forward
// semantics.  Gemma also requires one extension entry per common layer.
pub open spec fn model_weights_architecture_repr_valid(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
) -> bool {
    match architecture_repr {
        ModelWeightsArchitectureRepr::Gemma4Text(gemma) =>
            wr.architecture == ModelArchitecture::Gemma4Text
            && gemma.decoder.layers.len() == wr.layers.len()
            && forall|i: int| 0 <= i < wr.layers.len() ==>
                #[trigger] layer_qk_norm_matches_composition(
                    wr.layers[i], gemma.decoder.composition),
        ModelWeightsArchitectureRepr::Qwen3(qwen) =>
            wr.architecture == ModelArchitecture::Qwen3
            && qwen.geometry.num_layers == wr.layers.len()
            && forall|i: int| 0 <= i < wr.layers.len() ==>
                #[trigger] layer_qk_norm_matches_composition(
                    wr.layers[i], qwen.composition,
                ),
        ModelWeightsArchitectureRepr::Llama3(llama) =>
            wr.architecture == ModelArchitecture::Llama3
            && llama.geometry.num_layers == wr.layers.len()
            && forall|i: int| 0 <= i < wr.layers.len() ==>
                #[trigger] layer_qk_norm_matches_composition(
                    wr.layers[i], llama.composition,
                ),
        ModelWeightsArchitectureRepr::Gemma3Text(gemma) =>
            wr.architecture == ModelArchitecture::Gemma3Text
            && gemma.layers.len() == wr.layers.len()
            && forall|i: int| 0 <= i < wr.layers.len() ==>
                #[trigger] layer_qk_norm_matches_composition(
                    wr.layers[i], gemma.composition,
                ),
    }
}

// Gemma-4 participates in the semantic and deployment identity sums. Physical
// weight/runtime binding and exact kernel qualification are separate engine
// admission obligations; a semantic tag is not a serving capability.
pub struct Gemma4DeploymentConfigRepr {
    pub geometry: DenseGeometryRepr,
    pub global_attention_geometry: AttentionGeometryRepr,
    pub rms_norm: RmsNormConfigRepr,
    pub local_rotary: RotaryConfigRepr,
    pub global_rotary: RotaryConfigRepr,
    pub sliding_window: nat,
    pub attention_k_eq_v: bool,
    pub final_logit_softcap: Option<FloatParameterBits>,
    pub layer_attention: Seq<AttentionConfigRepr>,
}

pub struct Gemma4ModelWeightsExtensionRepr {
    pub decoder: FourNormGatedDecoderConfigRepr,
    pub global_attention_geometry: AttentionGeometryRepr,
    pub attention_k_eq_v: bool,
}

pub open spec fn unit_attention_scale_repr() -> AttentionScaleRepr {
    AttentionScaleRepr::InverseSqrtParameter(
        FloatParameterBits { bits: 4_607_182_418_800_017_408 })
}

pub open spec fn gemma4_deployment_config_repr(
    config: Gemma4Config, layer_attention_kinds: Seq<AttentionKind>,
) -> Gemma4DeploymentConfigRepr {
    Gemma4DeploymentConfigRepr {
        geometry: dense_geometry_repr(config.geometry),
        global_attention_geometry: AttentionGeometryRepr {
            num_attention_heads: config.geometry.num_attention_heads as nat,
            num_key_value_heads: config.num_global_key_value_heads as nat,
            head_dim: config.global_head_dim as nat,
        },
        rms_norm: RmsNormConfigRepr {
            epsilon: config.rms_norm_epsilon,
        },
        local_rotary: RotaryConfigRepr {
            theta: config.local_rope_theta,
            scaling: RotaryScalingRepr::None,
        },
        global_rotary: RotaryConfigRepr {
            theta: config.global_rope_theta,
            scaling: RotaryScalingRepr::Proportional {
                factor: config.global_rope_factor,
                partial_rotary_factor: config.global_partial_rotary_factor,
            },
        },
        sliding_window: config.sliding_window as nat,
        attention_k_eq_v: config.attention_k_eq_v,
        final_logit_softcap: config.final_logit_softcap,
        layer_attention: layer_attention_kinds.map_values(|kind: AttentionKind|
            match kind {
                AttentionKind::Full => AttentionConfigRepr::Full,
                AttentionKind::SlidingWindow =>
                    AttentionConfigRepr::SlidingWindow(config.sliding_window as nat),
            }),
    }
}

pub open spec fn gemma4_layer_attention_geometry(
    config: Gemma4DeploymentConfigRepr, kind: AttentionKind,
) -> AttentionGeometryRepr {
    match kind {
        AttentionKind::Full => config.global_attention_geometry,
        AttentionKind::SlidingWindow => attention_geometry_repr(config.geometry),
    }
}

pub open spec fn gemma4_row_parameters_repr(
    config: Gemma4DeploymentConfigRepr, kind: AttentionKind, scale: Tensor1D,
) -> FourNormGatedRowParametersRepr {
    FourNormGatedRowParametersRepr {
        norm: NormPolicyRepr::Direct(config.rms_norm.epsilon),
        qk_norm: NormPolicyRepr::Direct(config.rms_norm.epsilon),
        rotary: match kind {
            AttentionKind::Full => config.global_rotary,
            AttentionKind::SlidingWindow => config.local_rotary,
        },
        value_norm_epsilon: Some(config.rms_norm.epsilon),
        layer_scale: Some(scale),
    }
}

pub open spec fn gemma4_deployment_config_valid(config: Gemma4DeploymentConfigRepr) -> bool {
    let global = config.global_attention_geometry;
    dense_geometry_valid(config.geometry)
    && global.num_attention_heads == config.geometry.num_attention_heads
    && global.num_key_value_heads > 0
    && global.num_attention_heads % global.num_key_value_heads == 0
    && global.head_dim > 0 && global.head_dim % 2 == 0
    && positive_float_parameter_valid(config.rms_norm.epsilon)
    && rotary_config_valid(config.local_rotary)
    && config.local_rotary.scaling == RotaryScalingRepr::None
    && rotary_config_valid(config.global_rotary)
    && match config.global_rotary.scaling {
        RotaryScalingRepr::Proportional { factor, partial_rotary_factor } =>
            partial_rotary_factor.bits <= 4_607_182_418_800_017_408,
        _ => false,
    }
    && config.sliding_window > 0
    && config.layer_attention.len() == config.geometry.num_layers
    && (forall|i: int| 0 <= i < config.layer_attention.len() ==>
        match #[trigger] config.layer_attention[i] {
            AttentionConfigRepr::Full => true,
            AttentionConfigRepr::SlidingWindow(window) => window == config.sliding_window,
        })
    && match config.final_logit_softcap {
        Some(value) => positive_float_parameter_valid(value),
        None => true,
    }
}

pub open spec fn gemma4_config_repr(
    config: Gemma4Config, layers: Seq<FourNormGatedLayerExtensionRepr>,
) -> Gemma4ModelWeightsExtensionRepr {
    let deployment = gemma4_deployment_config_repr(config,
        layers.map_values(|layer: FourNormGatedLayerExtensionRepr|
            layer_attention_kind(layer.attention)));
    Gemma4ModelWeightsExtensionRepr {
        decoder: FourNormGatedDecoderConfigRepr {
            geometry: deployment.geometry,
            rms_norm: deployment.rms_norm,
            local_rotary: deployment.local_rotary,
            global_rotary: deployment.global_rotary,
            attention_scale: unit_attention_scale_repr(),
            sliding_window: deployment.sliding_window,
            composition: DenseCompositionRepr {
                activation: ActivationKind::GeluTanh,
                qk_norm: QkNormKind::RmsNorm,
                attention_bias: false,
                attention_dropout_bits: 0,
                tie_word_embeddings: true,
            },
            layers,
            final_norm_policy: NormPolicyRepr::Direct(deployment.rms_norm.epsilon),
            final_logit_softcap: deployment.final_logit_softcap,
        },
        global_attention_geometry: deployment.global_attention_geometry,
        attention_k_eq_v: deployment.attention_k_eq_v,
    }
}

pub open spec fn gemma4_deployment_projection(
    config: Gemma4ModelWeightsExtensionRepr,
) -> Gemma4DeploymentConfigRepr {
    let decoder = config.decoder;
    Gemma4DeploymentConfigRepr {
        geometry: decoder.geometry,
        global_attention_geometry: config.global_attention_geometry,
        rms_norm: decoder.rms_norm,
        local_rotary: decoder.local_rotary,
        global_rotary: decoder.global_rotary,
        sliding_window: decoder.sliding_window,
        attention_k_eq_v: config.attention_k_eq_v,
        final_logit_softcap: decoder.final_logit_softcap,
        layer_attention: decoder.layers.map_values(
            |layer: FourNormGatedLayerExtensionRepr| layer.attention),
    }
}

pub open spec fn gemma4_config_valid(config: Gemma4ModelWeightsExtensionRepr) -> bool {
    let deployment = gemma4_deployment_projection(config);
    let decoder = config.decoder;
    gemma4_deployment_config_valid(deployment)
    && decoder.attention_scale == unit_attention_scale_repr()
    && decoder.final_norm_policy == NormPolicyRepr::Direct(decoder.rms_norm.epsilon)
    && decoder.composition == DenseCompositionRepr {
        activation: ActivationKind::GeluTanh,
        qk_norm: QkNormKind::RmsNorm,
        attention_bias: false,
        attention_dropout_bits: 0,
        tie_word_embeddings: true,
    }
    && forall|i: int| 0 <= i < decoder.layers.len() ==> {
        let layer = #[trigger] decoder.layers[i];
        layer.attention_scale == unit_attention_scale_repr()
        && match layer.row_parameters.layer_scale {
            Some(scale) => scale.len() == 1
                && layer.row_parameters == gemma4_row_parameters_repr(
                    deployment, layer_attention_kind(layer.attention), scale),
            None => false,
        }
    }
}

pub proof fn lemma_gemma4_config_valid_implies_deployment_projection(
    config: Gemma4ModelWeightsExtensionRepr,
)
    requires gemma4_config_valid(config),
    ensures gemma4_deployment_config_valid(gemma4_deployment_projection(config)),
{}

// Batch step input — set of request IDs being stepped.
pub struct BatchStepInput {
    pub request_ids: Set<RequestId>,
}

} // verus!
