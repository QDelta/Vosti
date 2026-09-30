//! Architecture-neutral semantic certificates for reusable physical prefixes.
//!
//! Scheduler provenance identifies a physical page chain with an exact token
//! prefix.  This module adds the model-dependent statement that the KV cells
//! in that chain are canonical for one complete semantic model.  It contains
//! no family dispatch and remains valid after the request that produced a page
//! has left the live set.

use crate::exec::engine::Engine;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use vstd::prelude::*;

verus! {

// A registry hit is described entirely by persistent scheduler state: exact
// token placement over a collision-checked physical prefix chain.
pub open spec fn registered_prefix_candidate(
    cs: &crate::exec::cache_scheduler::CacheScheduler,
    chain: Seq<BlockId>,
    request_tokens: Seq<TokenId>,
    c_tokens: nat,
) -> bool {
    let pages = c_tokens / crate::types::BLOCK_SIZE_SPEC;
    &&& c_tokens <= request_tokens.len()
    &&& c_tokens % crate::types::BLOCK_SIZE_SPEC == 0
    &&& pages <= chain.len()
    &&& crate::exec::cache_scheduler::registered_prefix_chain(
        cs.blocks@, chain.subrange(0, pages as int),
    )
    &&& crate::exec::cache_scheduler::token_placement_prefix(
        cs.blocks@, chain, request_tokens, c_tokens as int,
    )
    &&& forall|k: int| #![trigger chain[k]] 0 <= k < pages as int ==> {
        let bid = chain[k];
        &&& cs.blocks@.contains_key(bid)
        &&& cs.hash_to_block@.contains_key(cs.blocks@[bid].hash_value)
        &&& cs.hash_to_block@[cs.blocks@[bid].hash_value] == bid
    }
}

// A provenance-bearing chain need not be the registry's current hash target.
// Exact token placement and physical ancestry are the semantic conditions;
// the hash map is only an indexing mechanism.
pub open spec fn provenance_prefix_candidate(
    cs: &crate::exec::cache_scheduler::CacheScheduler,
    chain: Seq<BlockId>,
    request_tokens: Seq<TokenId>,
    c_tokens: nat,
) -> bool {
    let pages = c_tokens / crate::types::BLOCK_SIZE_SPEC;
    &&& c_tokens <= request_tokens.len()
    &&& c_tokens % crate::types::BLOCK_SIZE_SPEC == 0
    &&& pages <= chain.len()
    &&& crate::exec::cache_scheduler::registered_prefix_chain(
        cs.blocks@, chain.subrange(0, pages as int),
    )
    &&& crate::exec::cache_scheduler::token_placement_prefix(
        cs.blocks@, chain, request_tokens, c_tokens as int,
    )
}

pub open spec fn registered_cache_cell_is_canonical(
    engine: &Engine,
    model: SemanticModelRepr,
    chain: Seq<BlockId>,
    request_tokens: Seq<TokenId>,
    layer: int,
    pos: nat,
) -> bool {
    let request = crate::proof::reference::request_machine::token_seq_to_int(request_tokens);
    let slot = crate::proof::tensor::geometry::block_table_slot(chain, pos);
    &&& crate::proof::tensor::geometry::slot_in_cache(
        engine.kv_caches_repr@[layer].0, slot,
    )
    &&& crate::proof::tensor::geometry::slot_in_cache(
        engine.kv_caches_repr@[layer].1, slot,
    )
    &&& crate::proof::model::cache::canonical_kv_defined(
        model, request, layer, pos,
    )
    &&& (
        crate::proof::tensor::geometry::cache_at(
            engine.kv_caches_repr@[layer].0, slot,
        ),
        crate::proof::tensor::geometry::cache_at(
            engine.kv_caches_repr@[layer].1, slot,
        ),
    ) == crate::proof::model::cache::canonical_kv_at(
        model, request, layer, pos,
    )
}

// Registry-facing certificate used when a partial-prefill row reuses an exact
// cached prompt prefix at the current stable boundary.
pub open spec fn registered_cache_fidelity(
    engine: &Engine,
    model: SemanticModelRepr,
) -> bool {
    forall|chain: Seq<BlockId>, request_tokens: Seq<TokenId>,
            c_tokens: nat, layer: int, pos: nat|
        #![trigger
            registered_cache_cell_is_canonical(
                engine, model, chain, request_tokens, layer, pos,
            ),
            registered_prefix_candidate(
                &engine.cs, chain, request_tokens, c_tokens,
            )
        ]
        registered_prefix_candidate(
            &engine.cs, chain, request_tokens, c_tokens,
        )
        && 0 <= layer < model.weights.layers.len()
        && pos < c_tokens
        ==> registered_cache_cell_is_canonical(
            engine, model, chain, request_tokens, layer, pos,
        )
}

// Strong persistent certificate over every provenance-bearing chain,
// including zero-reference and collision-hidden pages.
pub open spec fn provenance_cache_fidelity(
    engine: &Engine,
    model: SemanticModelRepr,
) -> bool {
    forall|chain: Seq<BlockId>, request_tokens: Seq<TokenId>,
            c_tokens: nat, layer: int, pos: nat|
        #![trigger
            registered_cache_cell_is_canonical(
                engine, model, chain, request_tokens, layer, pos,
            ),
            provenance_prefix_candidate(
                &engine.cs, chain, request_tokens, c_tokens,
            )
        ]
        provenance_prefix_candidate(
            &engine.cs, chain, request_tokens, c_tokens,
        )
        && 0 <= layer < model.weights.layers.len()
        && pos < c_tokens
        ==> registered_cache_cell_is_canonical(
            engine, model, chain, request_tokens, layer, pos,
        )
}

pub proof fn lemma_provenance_fidelity_implies_registered_fidelity(
    engine: &Engine,
    model: SemanticModelRepr,
)
    requires provenance_cache_fidelity(engine, model),
    ensures registered_cache_fidelity(engine, model),
{
    assert forall|chain: Seq<BlockId>, request_tokens: Seq<TokenId>,
            c_tokens: nat, layer: int, pos: nat|
        #![trigger
            registered_cache_cell_is_canonical(
                engine, model, chain, request_tokens, layer, pos,
            ),
            registered_prefix_candidate(
                &engine.cs, chain, request_tokens, c_tokens,
            )
        ]
        registered_prefix_candidate(
            &engine.cs, chain, request_tokens, c_tokens,
        )
        && 0 <= layer < model.weights.layers.len()
        && pos < c_tokens
        implies registered_cache_cell_is_canonical(
            engine, model, chain, request_tokens, layer, pos,
        )
    by {
        reveal(registered_prefix_candidate);
        assert(provenance_prefix_candidate(
            &engine.cs, chain, request_tokens, c_tokens,
        )) by {
            reveal(provenance_prefix_candidate);
        }
        assert(provenance_cache_fidelity(engine, model));
    }
}

// No nonempty provenance chain can be rooted in an empty physical block map.
// Keeping this fact beside the generic provenance vocabulary lets every model
// family share the same persistent initialization proof.
pub proof fn lemma_empty_physical_cache_has_provenance_fidelity(
    engine: &Engine,
    model: SemanticModelRepr,
)
    requires
        engine.cs.blocks@.dom().is_empty(),
    ensures
        provenance_cache_fidelity(engine, model),
{
    assert forall|chain: Seq<BlockId>, request_tokens: Seq<TokenId>,
            c_tokens: nat, layer: int, pos: nat|
        #![trigger
            registered_cache_cell_is_canonical(
                engine, model, chain, request_tokens, layer, pos,
            ),
            provenance_prefix_candidate(
                &engine.cs, chain, request_tokens, c_tokens,
            )
        ]
        provenance_prefix_candidate(
            &engine.cs, chain, request_tokens, c_tokens,
        )
        && 0 <= layer < model.weights.layers.len()
        && pos < c_tokens
        implies registered_cache_cell_is_canonical(
            engine, model, chain, request_tokens, layer, pos,
        )
    by {
        reveal(provenance_prefix_candidate);
        let pages = c_tokens / crate::types::BLOCK_SIZE_SPEC;
        assert(pages > 0) by {
            if pages == 0 {
                assert(c_tokens < crate::types::BLOCK_SIZE_SPEC);
                assert(c_tokens % crate::types::BLOCK_SIZE_SPEC == c_tokens);
                assert(c_tokens == 0);
            }
        }
        assert(0 < pages as int);
        let prefix = chain.subrange(0, pages as int);
        assert(prefix.len() > 0);
        assert(crate::exec::cache_scheduler::registered_prefix_chain(
            engine.cs.blocks@, prefix,
        ));
        reveal(crate::exec::cache_scheduler::registered_prefix_chain);
        let bid = prefix[0];
        assert(engine.cs.blocks@[bid].prefix_depth as int == 1);
        assert(engine.cs.blocks@.contains_key(bid));
        assert(false);
    }
}

} // verus!
