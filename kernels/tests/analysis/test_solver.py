import z3


def test_monotone_cumulative_lengths_force_selected_batch() -> None:
    cu_q = z3.Function("cu_q", z3.IntSort(), z3.IntSort())
    B, x, bi, li = z3.Ints("B x bi li")
    i, j = z3.Ints("i j")

    assumptions = [
        B > 0,
        0 <= x,
        x < B,
        0 <= bi,
        bi < B,
        z3.ForAll(
            [i, j],
            z3.Implies(
                z3.And(0 <= i, i < B, 0 <= j, j < B, i < j),
                cu_q(i) < cu_q(j),
            ),
        ),
        li >= 0,
        cu_q(0) == 0,
        li * 64 < cu_q(bi + 1) - cu_q(bi),
        cu_q(x) < cu_q(x + 1),
        cu_q(x) < cu_q(bi) + (li + 1) * 64,
        cu_q(x) < cu_q(bi + 1),
        cu_q(bi) + li * 64 < cu_q(bi) + (li + 1) * 64,
        cu_q(bi) + li * 64 < cu_q(x + 1),
        cu_q(bi) + li * 64 < cu_q(bi + 1),
    ]

    solver = z3.Solver()
    solver.add(*assumptions)
    solver.add(z3.Not(bi == x))
    assert solver.check() == z3.unsat
