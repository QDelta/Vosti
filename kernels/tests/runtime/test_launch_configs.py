from triton_kernels.matmul import (
    CONFIGS,
    CONFIG_INDEX_BY_NK,
    DEFAULT_CONFIG_INDEX,
    select_config,
)
from triton_kernels.fattn_paged import (
    CONFIGS as ATTENTION_CONFIGS,
    DEFAULT_CONFIG_INDEX as ATTENTION_DEFAULT_CONFIG_INDEX,
    select_config as select_attention_config,
)
from triton_kernels.fattn_paged_swa import (
    select_config as select_sliding_attention_config,
)
from triton_kernels.qkv_matmul import (
    CONFIGS as QKV_CONFIGS,
    CONFIG_INDEX_BY_QKV_GEOMETRY,
    DEFAULT_CONFIG_INDEX as QKV_DEFAULT_CONFIG_INDEX,
    select_config as select_qkv_config,
)
from triton_kernels.gemma_rmsnorm import (
    select_config as select_gemma_rmsnorm_config,
)
from triton_kernels.rmsnorm import select_config as select_rmsnorm_config
from triton_kernels.rmsnorm_residual import (
    select_config as select_rmsnorm_residual_config,
)
from triton_kernels.gelu_tanh_mul import (
    select_config as select_gelu_tanh_mul_config,
)
from triton_kernels.silu_mul import select_config as select_silu_mul_config


def test_attention_selectors_have_independent_static_head_geometry_policies() -> None:
    tuned = {
        "BLOCK_M": 16,
        "BLOCK_N": 64,
        "num_warps": 4,
        "num_stages": 2,
        "D_HEAD": 256,
    }
    fallback = {
        **ATTENTION_CONFIGS[ATTENTION_DEFAULT_CONFIG_INDEX],
        "D_HEAD": 128,
    }
    assert select_attention_config(256) == tuned
    assert select_sliding_attention_config(256, 1024) == tuned
    assert select_attention_config(128) == {**tuned, "D_HEAD": 128}
    assert select_sliding_attention_config(128, 4096) == fallback
    for head_dim in (64, 192):
        expected = {**ATTENTION_CONFIGS[ATTENTION_DEFAULT_CONFIG_INDEX], "D_HEAD": head_dim}
        assert select_attention_config(head_dim) == expected
        assert select_sliding_attention_config(head_dim, 1024) == expected


def test_sliding_attention_policy_does_not_follow_full_attention_tuning(monkeypatch) -> None:
    from triton_kernels import fattn_paged

    expected = select_sliding_attention_config(128, 1024)
    monkeypatch.setitem(fattn_paged.CONFIG_INDEX_BY_HEAD_DIM, 128, 2)
    assert select_attention_config(128)["BLOCK_M"] == 32
    assert select_sliding_attention_config(128, 1024) == expected


def test_norm_selectors_use_tuned_model_widths_and_keep_fallbacks() -> None:
    for width in (2560, 3840, 5376):
        assert select_gemma_rmsnorm_config(width)["num_warps"] == 8
    assert select_gemma_rmsnorm_config(2048)["num_warps"] == 4

    for selector in (select_rmsnorm_config, select_rmsnorm_residual_config):
        assert selector(3072)["num_warps"] == 8
        assert selector(4096)["num_warps"] == 8
        assert selector(1024)["num_warps"] == 4


def test_activation_selectors_share_bounded_static_policy() -> None:
    expected = {
        "BLOCK_M": 1,
        "BLOCK_N": 1024,
        "num_warps": 8,
        "num_stages": 1,
    }
    for width in (8192, 10240, 14336, 15360, 21504, 28672):
        assert select_silu_mul_config(width) == expected
        assert select_gelu_tanh_mul_config(width) == expected


def test_matmul_selector_accepts_only_static_nk() -> None:
    call_sites = [
        (2048, 1024),
        (1024, 1024),
        (1024, 2048),
        (6144, 1024),
        (1024, 3072),
        (151936, 1024),
        (4096, 4096),
        (1024, 4096),
        (24576, 4096),
        (28672, 4096),
        (4096, 12288),
        (4096, 14336),
        (128256, 4096),
        (151936, 4096),
        (4096, 3840),
        (2048, 3840),
        (3840, 4096),
        (30720, 3840),
        (3840, 15360),
        (262208, 3840),
        (4096, 5376),
        (2048, 5376),
        (5376, 4096),
        (43008, 5376),
        (5376, 21504),
        (262208, 5376),
    ]
    for n, k in call_sites:
        assert select_config(n, k) in CONFIGS


def test_matmul_selector_uses_exact_finite_nk_policy() -> None:
    for (n, k), index in CONFIG_INDEX_BY_NK.items():
        assert select_config(n, k) == CONFIGS[index]

    assert DEFAULT_CONFIG_INDEX == 0
    assert select_config(2048, 1024) == CONFIGS[0]
    assert select_config(151936, 1024) == CONFIGS[0]
    assert select_config(151936, 4096) == CONFIGS[3]


def test_matmul_selector_uses_small_row_tuned_policies() -> None:
    narrow_k128 = {
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "BLOCK_K": 128,
        "num_warps": 2,
        "num_stages": 3,
    }
    balanced_k128 = {
        "BLOCK_M": 16,
        "BLOCK_N": 64,
        "BLOCK_K": 128,
        "num_warps": 2,
        "num_stages": 3,
    }
    assert select_config(1024, 3072) == narrow_k128
    assert select_config(128256, 3072) == balanced_k128
    assert select_config(2048, 2560) == narrow_k128
    assert select_config(1024, 2560) == narrow_k128
    assert select_config(4096, 3840) == narrow_k128
    assert select_config(2048, 3840) == narrow_k128
    assert select_config(1024, 4096) == narrow_k128
    assert select_config(128256, 4096) == balanced_k128
    assert select_config(4096, 5376) == narrow_k128
    assert select_config(2048, 5376) == narrow_k128


def test_matmul_selector_uses_tuned_k256_down_projection_policies() -> None:
    narrow_k256 = {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_K": 256,
        "num_warps": 4,
        "num_stages": 3,
    }
    wide_k256 = {
        "BLOCK_M": 16,
        "BLOCK_N": 64,
        "BLOCK_K": 256,
        "num_warps": 4,
        "num_stages": 3,
    }
    assert select_config(3072, 8192) == narrow_k256
    assert select_config(3840, 15360) == narrow_k256
    assert select_config(3072, 3072) == narrow_k256
    assert select_config(3840, 4096) == narrow_k256
    assert select_config(4096, 4096) == narrow_k256
    assert select_config(5376, 21504) == wide_k256
    assert select_config(4096, 14336) == {**narrow_k256, "BLOCK_M": 64}
    assert select_config(5376, 4096) == {**wide_k256, "BLOCK_M": 64}
    for hidden_size in (2560, 3840, 5376):
        assert select_config(262208, hidden_size) == wide_k256


def test_matmul_selector_balances_small_rows_and_prefill_for_2560_outputs() -> None:
    for k in (2048, 10240):
        assert select_config(2560, k) == {
            "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 512,
            "num_warps": 4, "num_stages": 3,
        }


def test_matmul_selector_balances_prefill_and_decode_for_gate_up() -> None:
    medium = {
        "BLOCK_M": 64,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "num_warps": 4,
        "num_stages": 4,
    }
    for n, k in ((16384, 3072), (20480, 2560)):
        assert select_config(n, k) == medium
    for n, k in ((30720, 3840), (28672, 4096)):
        assert select_config(n, k) == {
            **medium, "BLOCK_M": 128, "BLOCK_N": 256, "num_warps": 8,
        }
    assert select_config(43008, 5376) == {
        "BLOCK_M": 64,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "num_warps": 4,
        "num_stages": 3,
    }


def test_qkv_selector_uses_only_model_geometry_and_has_a_fallback() -> None:
    for geometry, index in CONFIG_INDEX_BY_QKV_GEOMETRY.items():
        assert select_qkv_config(*geometry) == QKV_CONFIGS[index]

    assert QKV_DEFAULT_CONFIG_INDEX == 0
    assert select_qkv_config(640, 128, 512) == QKV_CONFIGS[0]
    for geometry in (
        (4096, 1024, 4096), (4096, 2048, 3840),
        (4096, 2048, 5376), (3072, 1024, 3072),
    ):
        assert select_qkv_config(*geometry) == {
            "BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 256,
            "num_warps": 4, "num_stages": 3,
        }
    assert QKV_CONFIGS[1]["BLOCK_N"] == 32
    assert select_qkv_config(2048, 1024, 2560) == {
        "BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256,
        "num_warps": 4, "num_stages": 3,
    }
