"""Fixed engine/kernel constraints; importing this module needs no GPU libraries."""

# Fixed implementation constraint, NOT a configurable page size. Rust's
# counterpart is BLOCK_SIZE in src/types.rs. Changing both values is insufficient:
# review cache layouts, scheduler arithmetic/proofs, and kernel specializations;
# regenerate certificates and deployment evidence, then reverify and retest.
PAGE_SIZE = 64
