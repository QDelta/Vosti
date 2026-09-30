"""Source names select bindings; they cannot manufacture solver equalities."""

from pathlib import Path

import pytest
import z3

from ir.names import fresh_name
from ir.relational_dataflow import prove_relational_dataflow_from_annotations
from ir.proof_preparation import prepare_annotation_proof


ROOT = Path(__file__).resolve().parents[2] / "triton_kernels"
CONSTANTS = dict(BLOCK_M=1, BLOCK_N=64)


def _source():
    return (ROOT / "add.py").read_text()


def _add_params(source, declarations, names):
    source = source.replace("# @params(\n", "# @params(\n" + declarations)
    return source.replace("    M,\n    N,\n", "    M,\n    N,\n" +
                          "".join(f"    {name},\n" for name in names), 1)


@pytest.mark.parametrize("related_right", [False, True])
def test_right_suffix_cannot_alias_a_distinct_shared_parameter(related_right):
    source = _add_params(_source(), "#   scalar(MR, int),\n", ["MR"])
    source = source.replace("#   same(N),", "#   same(N, MR),")
    source = source.replace("#     right(M) == 1,",
        "#     right(M) > 0, left(M) == MR," +
        (" right(M) == MR," if related_right else ""))
    source = source.replace("(x_block + y_block).to", "(x_block + M + y_block).to")
    # M_left=MR=2 and M_right=1 satisfies the unstrengthened precondition.
    # With x=y=0, the compared output cells are 2 and 1 respectively.
    report = prove_relational_dataflow_from_annotations(source, "add_kernel", CONSTANTS)
    assert (report.verified_contract is not None) == related_right


def test_input_loop_and_shared_names_have_independent_solver_identities():
    names = ["MR", "M_left", "M_right", "_pid_0L", "_pid_0R", "_pid_0_eff",
             "__tensor_guard_x_0", "__relational_discrete_gate_0", "__identity_coordinate_0"]
    source = _add_params(_source(), "".join(f"#   scalar({name}, int),\n" for name in names), names)
    source = source.replace("#   same(N),", f"#   same(N, {', '.join(names)}),")
    source = source.replace("#     right(M) == 1,", "#     right(M) > 0,")
    config = prepare_annotation_proof(source, "add_kernel", CONSTANTS).first_config
    independent = [config.left_env[name] for name in names]
    for name in ["M", "_pid_0", "_pid_1"]:
        independent += [config.left_env[name], config.right_env[name]]
    assert len({value.get_id() for value in independent}) == len(independent)
    for name in ["N", *names]:
        assert config.left_env[name].eq(config.right_env[name])


def test_tensor_side_suffixes_cannot_alias_shared_tensors():
    source = _add_params(_source(),
        "#   tensor(x_left, float, shape(1, N)),\n"
        "#   tensor(x_right, float, shape(1, N)),\n", ["x_left", "x_right"])
    source = source.replace("#   same(N),", "#   same(N, x_left, x_right),")
    config = prepare_annotation_proof(source, "add_kernel", CONSTANTS).first_config
    functions = [config.left_env['x'], config.right_env['x'],
                 config.left_env['x_left'], config.left_env['x_right']]
    assert all(isinstance(f, z3.FuncDeclRef) for f in functions)
    assert len({f.get_id() for f in functions}) == len(functions)
    for name in ['x_left', 'x_right']:
        assert config.left_env[name].eq(config.right_env[name])


def test_separate_proof_preparations_do_not_share_launch_symbols():
    first = prepare_annotation_proof(_source(), "add_kernel", CONSTANTS).first_config
    second = prepare_annotation_proof(_source(), "add_kernel", CONSTANTS).first_config
    for name in ["M", "N", "x", "b", "_pid_0"]:
        assert not first.left_env[name].eq(second.left_env[name])


def test_unrelated_quantifier_cannot_authorize_bare_launch_parameter():
    source = _source().replace("#     left(M) > 0, N > 0,",
        "#     left(M) > 0, N > 0, M >= 0,\n"
        "#     forall(M, implies(and(M >= 0), M >= 0)),")
    with pytest.raises(ValueError, match="appear bare"):
        prepare_annotation_proof(source, "add_kernel", CONSTANTS)


def test_temporary_coordinate_names_skip_all_existing_bindings():
    occupied = {"__tensor_guard_x_0", "__tensor_guard_x_0_"}
    assert fresh_name("__tensor_guard_x_0", occupied) == "__tensor_guard_x_0__"
    assert fresh_name("ordinary", occupied) == "ordinary"
    assert occupied == {"__tensor_guard_x_0", "__tensor_guard_x_0_"}
