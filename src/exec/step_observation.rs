//! Read-only scheduling observations, not inputs to serving semantics.
//!
//! Capture after plan selection and before commit: a cache-only chunk or a
//! completed request can lose its residency during commit. Consumers latch
//! the first observation per request, never a later chunk's self-reuse.

use crate::exec::cache_scheduler::CacheScheduler;
use crate::{types::{RequestId}};
use vstd::prelude::*;

verus! {

pub struct PlannedPrefixReuse {
    pub request_id: RequestId,
    // None is deliberately different from an observed cold prefix (Some(0)).
    pub cached_prefix_blocks: Option<u64>,
}

pub fn record_planned_prefix_reuse(
    cs: &CacheScheduler,
    scheduled: &Vec<RequestId>,
    observations: &mut Vec<PlannedPrefixReuse>,
) {
    observations.clear();
    let mut i = 0usize;
    while i < scheduled.len()
        invariant i <= scheduled.len(),
        decreases scheduled.len() - i,
    {
        let rid = scheduled[i];
        let cached = match cs.request_residency.get(&rid) {
            Some(residency) => Some(residency.cached_prefix_blocks),
            None => None,
        };
        observations.push(PlannedPrefixReuse {
            request_id: rid,
            cached_prefix_blocks: cached,
        });
        i += 1;
    }
}

}
