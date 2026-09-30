// Scheduler state and public plan/result data types.
//
// This layer contains representation only. Queue primitives live in
// `availability_queue`; invariants and transition proofs live above it.

use super::availability_queue::FreeBlockQueue;
use crate::exec::request_state::{RequestState, SamplerState};
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::hash_map::HashMapWithView;
use vstd::prelude::*;

verus! {

// Step mode for a batched forward.  `Mixed` (2026-08-06, continuous
// batching) marks a plan whose cu-partition carries both decode rows
// (q_len 1) and prefill rows (q_len == k_len == prompt len) in one batch;
// downstream semantics flows entirely through cu_seqlens_q/k, so the mode
// is informational.
pub enum StepMode {
    Prefill,
    Decode,
    Mixed,
}

// ---------------------------------------------------------------------------
// Datatypes (verbatim from Dafny shape).
// ---------------------------------------------------------------------------

// Exec-friendly: `tokens` is `Vec<TokenId>` so the kernel materializers
// can read it; `refcount`, `hash_value`, and prefix provenance are executable
// metadata.  A registered prompt page records its one-based logical depth and
// exact physical predecessor.  Reuse checks this chain, so rolling-hash
// collisions cannot splice a page into a different attention context.
pub struct BlockEntry {
    pub tokens: Vec<TokenId>,
    pub refcount: u64,
    pub hash_value: u64,
    pub prefix_depth: u64,
    pub parent_block: Option<BlockId>,
}

impl Clone for BlockEntry {
    fn clone(&self) -> (out: Self)
        ensures
            out.tokens@ == self.tokens@,
            out.refcount == self.refcount,
            out.hash_value == self.hash_value,
            out.prefix_depth == self.prefix_depth,
            out.parent_block == self.parent_block,
    {
        BlockEntry {
            tokens: self.tokens.clone(),
            refcount: self.refcount,
            hash_value: self.hash_value,
            prefix_depth: self.prefix_depth,
            parent_block: self.parent_block,
        }
    }
}

pub struct RequestResidency {
    pub block_ids: Vec<BlockId>,
    pub cached_prefix_blocks: u64,
    pub slot_mapping: Vec<SlotId>,
}

impl Clone for RequestResidency {
    fn clone(&self) -> (out: Self)
        ensures
            out.block_ids@ == self.block_ids@,
            out.cached_prefix_blocks == self.cached_prefix_blocks,
            out.slot_mapping@ == self.slot_mapping@,
    {
        RequestResidency {
            block_ids: self.block_ids.clone(),
            cached_prefix_blocks: self.cached_prefix_blocks,
            slot_mapping: self.slot_mapping.clone(),
        }
    }
}

pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
}

pub struct SampleResult {
    pub sampler_state: SamplerState,
    pub token: TokenId,
}

pub type SampleResults = HashMapWithView<u64, SampleResult>;
pub type EmittedTokens = HashMapWithView<u64, TokenId>;

// ---------------------------------------------------------------------------
// StepPlan — the per-step bundle of input tensors + ghost reprs that the
// engine hands to model_forward.  All tensors here are FRESH per-step
// allocations; reprs are produced by `Plan` and consumed by `Commit`.
// ---------------------------------------------------------------------------

pub struct StepPlan {
    pub scheduled_ids: Vec<RequestId>,
    // One authoritative row-effect mask for the whole step.  A `true` row
    // samples/emits a token; a `false` row only advances the KV cache.
    // Chunked prefill uses `false` for non-final chunks without duplicating
    // this decision in commit or the proof layer.
    pub sample_mask: Vec<bool>,
    pub mode: StepMode,
    pub input_ids: RT::Tensor,
    pub positions: RT::Tensor,
    pub block_table: RT::Tensor,
    pub slot_mapping: RT::Tensor,
    pub cu_seqlens_q: RT::Tensor,
    pub cu_seqlens_k: RT::Tensor,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    // Ghost reprs.
    pub input_ids_repr: Ghost<IntTensor1D>,
    pub positions_repr: Ghost<IntTensor1D>,
    pub cu_seqlens_q_repr: Ghost<Seq<int>>,
    pub cu_seqlens_k_repr: Ghost<Seq<int>>,
    pub block_table_repr: Ghost<Seq<Seq<BlockId>>>,
    pub slot_mapping_repr: Ghost<Seq<int>>,
}

pub tracked struct StepPlanPerms {
    pub tracked input_ids: RT::TensorPerm,
    pub tracked positions: RT::TensorPerm,
    pub tracked block_table: RT::TensorPerm,
    pub tracked slot_mapping: RT::TensorPerm,
    pub tracked cu_seqlens_q: RT::TensorPerm,
    pub tracked cu_seqlens_k: RT::TensorPerm,
}

// ---------------------------------------------------------------------------
// CacheScheduler — concrete scheduler/cache state.  Heap class in Dafny;
// here it's a struct with exec-mutable HashMapWithView / Vec fields.
// Specs read via the View (`cs.blocks@`, etc.) — same Map<u64, _> shape
// the predicates expect.
// ---------------------------------------------------------------------------

pub struct CacheScheduler {
    pub config: SchedulerConfig,
    // Total number of physical blocks the cache was provisioned with.
    // BlockIds are drawn from `0..num_blocks`.  Set once at `init`.
    pub num_blocks: u64,
    pub free_blocks: u64,
    // Physically vacant block ids.  Cached zero-reference pages live in the
    // separate `cached_queue`; keeping both queues homogeneous avoids coupling
    // allocation correctness to a moving vacant/cached boundary.
    pub free_queue: FreeBlockQueue,
    // Resident zero-reference provenance pages.  Children precede parents, so
    // the head is always a provenance leaf.  The order among simultaneously
    // eligible leaves remains replacement policy only.
    pub cached_queue: FreeBlockQueue,
    pub blocks: HashMapWithView<u64, BlockEntry>,
    pub running: Vec<RequestId>,
    pub waiting: Vec<RequestId>,
    pub request_residency: HashMapWithView<u64, RequestResidency>,
    pub live_requests: HashMapWithView<u64, RequestState>,
    // Permanent request-identity tombstones.  Completed requests leave the
    // live map but remain here, so a runtime id cannot be rebound to a
    // different logical request in a later dynamic-arrival event.
    pub accepted_requests: HashMapWithView<u64, bool>,
    pub hash_to_block: HashMapWithView<u64, u64>,
}

} // verus!
