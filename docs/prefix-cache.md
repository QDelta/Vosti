# Verified prefix caching

The engine reuses full KV pages from prompts and generated tokens across
requests. A released page may stay in the cache with a zero reference count
until another request reuses it or the scheduler needs the space.
The proof shows that reused pages contain the KV values for the request's
token prefix. See [verification](verification.md) for the system-level theorem.

## Protocol

### 1. Match before publication

`match_cached_prefix` walks page-aligned prompt prefixes from left to right.
For each page it computes the rolling hash with `chain_hash_span`, uses
`hash_to_block` as an index, and validates the candidate before accepting it.
A match requires:

- every token in the full page to be equal;
- the stored one-based prefix depth to be correct;
- the exact physical predecessor to be the page already selected; and
- the physical page not to occur twice in the request's block table.

Matching stops at the first miss. It also leaves at least one query position
uncached, so an admitted row always performs model work and its final prompt
row can sample.

Planning decides every reuse match before it registers any page produced by
the same step. A request therefore cannot consume KV values that another row
in the same forward has not yet materialized.

### 2. Attach the prefix and allocate the suffix

`allocate_prefill_from_match` increments the matched pages' refcounts,
allocates fresh pages for the remaining prompt, and records
`cached_prefix_blocks`. The request's slot mapping contains only the uncached
suffix. Shared pages are full and immutable; the only in-place append targets
an exclusively owned non-full tail. No copy-on-write transition is needed for
shared prefix pages.

### 3. Plan partial prefill

For a row with cached prefix length `c` tokens and selected endpoint `e`:

- input IDs and positions cover `[c, e)`;
- the query cumulative length advances by `e - c`;
- the key cumulative length advances by `e`;
- the block table covers the complete logical prefix through `e`; and
- the slot mapping covers exactly the newly computed suffix positions.

A non-final endpoint is page-aligned and has a false sample mask. The step
registers its completed full pages; after model execution, commit releases the
transient residency and returns the unchanged request to the waiting queue. A
later step can match those pages and continue. Only the final prompt chunk is
sampled.

Chunking currently changes scheduling granularity, not KV reservation: the
allocator reserves residency for the complete prompt. The batched sampling
tail may also compute a candidate for a non-sampling row and discard it.

### 4. Register completed pages

`publish_full_prefix_pages` stamps every newly completed full prompt page with:

- its rolling content hash;
- its one-based logical prefix depth; and
- its exact physical parent page.

It inserts an absent hash into `hash_to_block`; the first writer remains the
registry target. A zero hash is reserved and is not registered.

The engine also publishes completed decode pages after the model forward and
before commit appends the next sampled token or releases a finished request.
The newly sampled token has no KV yet and is excluded from this publication.

Registration is scheduler metadata. The semantic proof of the model forward
and KV scatter establishes that newly published physical cells contain the
canonical KV values before those cells can be reused by a later step.

### 5. Release and reclaim

`deallocate` walks a request's block table from child to parent. A page whose
refcount reaches zero is handled as follows:

- an unregistered page is removed immediately and its ID enters `free_queue`;
- a provenance-bearing page remains resident and enters `cached_queue`.

`free_queue` therefore contains exactly vacant IDs. `cached_queue` contains
exactly resident zero-reference provenance pages, ordered so children precede
cached parents. Its head is consequently a provenance leaf.

Under pressure, `reclaim_cached_leaves_until` repeatedly calls
`evict_one_cached_leaf`. Eviction removes the resident leaf, conditionally
clears its registry entry, and transfers its physical ID to `free_queue`.
Before prefill, the scheduler calculates the space needed for the uncached
suffix and pending commits. It reclaims pages if needed, then repeats prefix
matching against the remaining cache.

## Why hashes are not trusted

The rolling hash is only an accelerator. Correctness does not assume collision
resistance or injectivity. Exact page tokens prevent a collision within the
page from being accepted, while prefix depth and the exact physical-parent
chain prevent a byte-identical page computed under another causal context from
being spliced into the request. A collision may cause a miss or select which
equal-hash page remains indexed; it cannot by itself justify reuse.

Physical IDs also need no globally unique generation number. The persistent
provenance closure and child-before-parent eviction order ensure that an ID is
not recycled while a resident child can still name its old incarnation. This
is a lifetime property over the current provenance forest.

## Proof structure

The proof separates structural scheduler facts from semantic KV
facts.

1. Scheduler invariants. `cs_valid`, `free_queue_valid`, and
   `persistent_provenance_closed` cover refcounts, unique per-request block
   tables, exact queue membership, immutable shared pages, registered physical
   chains, and safe leaf eviction. `allocate_prefill_reuse_success` exports the
   exact accepted pre-state chain.
2. Physical-prefix semantics. `provenance_cache_fidelity` states that every
   resident provenance chain, including zero-reference and collision-hidden
   pages, contains canonical KV cells for its exact token prefix.
   `registered_cache_fidelity` is the registry-facing consequence used by a
   cache hit.
3. Step preservation. The architecture-neutral cache-discharge proof shows
   that inherited pages are outside current scatter writes, newly completed
   pages receive canonical KV, and reclamation preserves fidelity for all
   surviving pages.
4. Observable refinement. Cold prefill, cached partial prefill, and decode
   all refine the same independent request machine. Finite-trace induction then
   feeds the generic static and dynamic per-request prefix-equality theorems.

The stable code locations are:

| Concern | Location |
|---|---|
| Publication and matching | `src/exec/cache_scheduler/prefix_cache.rs` |
| Release and pressure eviction | `src/exec/cache_scheduler/reclamation.rs` |
| Planning and chunk geometry | `src/exec/cache_scheduler/planning.rs` |
| Structural invariants | `src/proof/scheduler/` |
| Canonical-cache predicates | `src/proof/cache/provenance.rs` |
| Semantic preservation | `src/proof/cache/semantics.rs` |
| Trace refinement | `src/proof/engine/refinement.rs`, `src/proof.rs` |

## Limitations

The proof does not select an access-recency policy. The current ordering is
constrained only where safety requires child-before-parent eviction; policy
among eligible leaves affects hits and admission, not correctness.

Sliding-window attention still retains and reasons about the complete causal
KV prefix. Prefix caching does not establish window-only dependence and does
not license eviction of rows outside the configured attention window.

The cold/warm test in `scripts/determinism_tests/` checks that reuse occurred
and compares the selected logit rows bit-for-bit. This test is separate from
the proof.

The result remains conditional on the trusted runtime representation bridge,
the imported KV-scatter copy/frame theorem, and the attention finite-value premise
listed in [verification](verification.md#open-gaps). Fairness and eventual completion are also
outside the theorem.
