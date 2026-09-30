"""Soundness regressions for the bounded suffix evaluator."""

import z3

from ir import Grid, Kernel, Var, Where
from diagnostics.suffix_independence import Evaluator, JUNK, NINF, TensorVal, _cast, _mul
from scripts.verify_suffix_independence import check_config, deployed_config


def _evaluator(env):
    kernel = Kernel(
        name="where_test",
        params=[],
        grid=Grid(iters=[], decls=[], body=[]),
    )
    evaluator = Evaluator(kernel, {}, {})
    evaluator.env.update(env)
    return evaluator


def test_ninf_multiplication_fails_closed_except_positive_constant():
    assert _mul(NINF, 0.5) is NINF
    assert _mul(0.5, NINF) is NINF
    assert _mul(NINF, 0.0) is JUNK
    assert _mul(NINF, -1.0) is JUNK
    assert _mul(NINF, NINF) is JUNK


def test_float_casts_are_congruent_per_target_but_not_cross_target():
    value = z3.Real("value")
    bf16 = _cast(value, "float", "tl.bfloat16")
    same_bf16 = _cast(value, "float", "tl.bfloat16")
    fp32 = _cast(value, "float", "tl.float32")

    assert z3.is_true(z3.simplify(bf16 == same_bf16))
    solver = z3.Solver()
    solver.add(bf16 != fp32)
    assert solver.check() == z3.sat


def test_uniform_concrete_where_does_not_evaluate_discarded_branch():
    evaluator = _evaluator(
        {
            "cond": TensorVal((2,), [False, False]),
            "old": TensorVal((2,), [1.0, 2.0]),
        }
    )
    # Evaluating the missing true-branch variable would raise KeyError.
    result = evaluator.e(Where(Var("cond"), Var("missing"), Var("old")))
    assert result.lanes == [1.0, 2.0]


def test_mixed_concrete_where_discards_junk_per_lane():
    evaluator = _evaluator(
        {
            "cond": TensorVal((2,), [True, False]),
            "candidate": TensorVal((2,), [7.0, JUNK]),
            "old": TensorVal((2,), [1.0, 2.0]),
        }
    )
    result = evaluator.e(Where(Var("cond"), Var("candidate"), Var("old")))
    assert result.lanes == [7.0, 2.0]


def test_where_never_discards_selected_junk():
    evaluator = _evaluator(
        {
            "cond": TensorVal((1,), [True]),
            "candidate": TensorVal((1,), [JUNK]),
            "old": TensorVal((1,), [2.0]),
        }
    )
    result = evaluator.e(Where(Var("cond"), Var("candidate"), Var("old")))
    assert result.lanes == [JUNK]


def test_deployed_config_covers_all_pages_when_tile_crosses_page():
    config = deployed_config()
    assert config["K_LEN"] == 65
    assert config["MAXP"] == 2
    assert config["BT"] == [[0, 1]]
    assert config["NUM_PAGES"] >= 2


def test_suffix_checker_rejects_undersized_block_table_before_evaluation():
    config = {
        "Q_LEN": 17,
        "K_LEN": 65,
        "MAXP": 1,
        "BT": [[0]],
        "BLOCK_M": 16,
        "BLOCK_N": 64,
        "D_HEAD": 8,
        "PAGE": 64,
        "NUM_PAGES": 2,
    }
    failures = check_config("not parsed for an invalid config", config)
    assert failures == [
        "invalid suffix-checker config: block table width 1 does not cover 2 logical pages"
    ]
