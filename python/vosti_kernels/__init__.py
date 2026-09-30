"""Architecture-neutral primitive surface for the trusted tensor boundary.

Family checkpoint loaders and explicit qualified primitive runtimes live in
their paired family modules. The package root exposes only shared primitive
operations and runtime-neutral tensor helpers used by the Rust boundary.
"""

__all__ = [
    "linear",
    "qkv_linear",
    "embed",
    "rms_norm",
    "add_rms_norm",
    "qk_norm",
    "view_as_kv",
    "merge_attention_heads",
    "split_last_axis_halves",
    "rotary_embed",
    "silu_and_mul",
    "token_tensor",
    "position_tensor",
    "slot_tensor",
    "block_tables_tensor",
    "seq_lens_tensor",
    "store_kv_cache",
    "store_kv_cache_from_verified_caller",
    "paged_attention",
    "paged_attention_from_verified_caller",
    "select_rows_for_sampling",
    "sample_tokens_rows",
    "select_sample_logits",
    "sample",
    "init_kv_caches",
    "from_flat",
]


def __getattr__(name):
    """Load tensor operations on use; metadata audits need only the standard library."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import kernels

    value = getattr(kernels, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
