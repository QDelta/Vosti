//! Shared four-norm gated composition; architecture payloads are bound by the
//! neutral model dispatcher. No model-specific forward or cache proof lives here.

pub(crate) use crate::proof::model::four_norm_gated::batch_invariance as batch_invariance;
pub(crate) use crate::proof::model::four_norm_gated::relocation as relocation;
pub(crate) use crate::proof::model::four_norm_gated::model as semantics;
#[cfg(verus_only)]
pub use crate::proof::model::four_norm_gated::capstones::{
    forward_logits_repr, forward_kv_reprs, reference_logits_last_row,
    lemma_reference_logits_last_row_is_forward_last,
    request_projection_ready, lemma_request_projection_ready_from_common_domain,
    lemma_logits_request_isolation, lemma_logits_repr_shape,
    lemma_kv_reprs_len, lemma_kv_reprs_empty, lemma_forward_cache_shape_preserved,
    cache_configuration_ready as cache_refinement_supported,
};
#[cfg(verus_only)]
pub use crate::proof::model::four_norm_gated::layer_witnesses::{
    layer_kv_rows, lemma_forward_layer_store, lemma_forward_relocation,
    lemma_kv_request_isolation, lemma_layer_kv_rows_request_isolation,
    lemma_layer_kv_rows_shape,
};
#[cfg(verus_only)]
pub use crate::proof::model::four_norm_gated::cache_refinement::lemma_cache_refinement_laws;

use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use vstd::prelude::*;

verus! {

pub open spec fn semantic_model(
    wr: ModelWeightsRepr, architecture: ModelWeightsArchitectureRepr,
) -> SemanticModelRepr {
    SemanticModelRepr { weights: wr, architecture }
}

} // verus!
