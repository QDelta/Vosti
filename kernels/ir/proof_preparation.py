"""Translate, specialize, and prepare named annotation goals for proof."""
from dataclasses import dataclass
from . import BoolType, FloatType, IntLit, IntType, Kernel, TensorType
from .subst import specialize_kernel_constants, expand_let_bindings
from .preprocess import check_variable_names, check_tensorindex_readonly
from .annotations import RelationalProofGoal, parse_verif_goals
from .annotation_lowering import LoweredProofGoal, lower_proof_goal
from .annotation_to_config import build_config_from_annotation
from .typ import infer_types_for_proof
from .regional_obligations import EquivProofConfig


def ensure_tensor_var_sizes_known(kernel: Kernel) -> None:
    unresolved_dims: list[str] = []
    for decl in kernel.grid.decls:
        match decl.type:
            case TensorType(dims=dims):
                for dim_idx, dim in enumerate(dims):
                    if not isinstance(dim, IntLit):
                        unresolved_dims.append(f"{decl.var.name}[{dim_idx}]={dim}")
            case IntType() | FloatType() | BoolType():
                continue
            case _:
                raise AssertionError(f"unhandled type: {decl.type}")
    if unresolved_dims:
        details = ", ".join(unresolved_dims)
        raise ValueError(
            f"After specialization, tensor variable sizes must be concrete integers; unresolved dimensions: {details}"
        )


@dataclass(frozen=True)
class PreparedAnnotationProof:
    """Validated, specialized IR and checked annotation context."""

    source: str
    constants: tuple[tuple[str, int | float | bool], ...]
    kernel: Kernel
    declared_parameters: tuple
    annotation: LoweredProofGoal
    first_config: EquivProofConfig


@dataclass(frozen=True)
class PreparedKernelProofs:
    """One translated kernel shared by all source-declared proof goals."""

    source: str
    constants: tuple[tuple[str, int | float | bool], ...]
    kernel: Kernel
    declared_parameters: tuple
    goals: tuple[RelationalProofGoal, ...]


def prepare_kernel_proofs(
    source: str,
    kernel_name: str,
    constants: dict[str, int | float | bool],
) -> PreparedKernelProofs:
    """Translate and type one kernel once for all named proof goals.

    Args:
        source: Python source code containing the annotated @triton.jit kernel.
        kernel_name: Name of the kernel function.
        constants: Block size constants to specialize (e.g. {"BLOCK_M": 64}).
    """
    from .translate import translate_kernel_source

    goals = parse_verif_goals(source, kernel_name)
    if not goals:
        raise ValueError(f"No @verif proof goal found in source for {kernel_name}")

    # Separate bool constexprs (for translation-time specialization) from numeric constants
    bool_constants = {k: v for k, v in constants.items() if isinstance(v, bool)}
    numeric_constants = {k: v for k, v in constants.items() if not isinstance(v, bool)}

    # Translate and preprocess kernel
    kernel = translate_kernel_source(
        source, kernel_name,
        specialize=bool_constants if bool_constants else None,
    )
    roles = check_variable_names(kernel)
    check_tensorindex_readonly(kernel)
    # Preserve the annotation-declared symbolic parameter types before numeric
    # constexpr specialization.  Contract consumers use these origins to
    # distinguish (for example) an axis declared as D from an unrelated integer
    # that happens to have the same deployed value.
    declared_parameters = tuple(kernel.params)
    kernel = specialize_kernel_constants(kernel, numeric_constants)
    kernel = expand_let_bindings(kernel)
    ensure_tensor_var_sizes_known(kernel)
    kernel = infer_types_for_proof(kernel, roles)

    return PreparedKernelProofs(
        source=source,
        constants=tuple(sorted(constants.items())),
        kernel=kernel,
        declared_parameters=declared_parameters,
        goals=goals,
    )


def prepare_goal_proof(
    prepared: PreparedKernelProofs,
    goal_name: str | None = None,
) -> PreparedAnnotationProof:
    """Select and validate one relational goal over a prepared kernel."""

    if goal_name is None:
        if len(prepared.goals) != 1:
            raise ValueError(
                "Kernel has multiple @verif proof goals; select one by name"
            )
        annotation = prepared.goals[0]
    else:
        matches = [goal for goal in prepared.goals if goal.name == goal_name]
        if not matches:
            raise ValueError(f"No @verif proof goal named {goal_name!r}")
        annotation = matches[0]

    annotation = lower_proof_goal(annotation)
    first_config = build_config_from_annotation(
        prepared.kernel,
        annotation,
        specialized_constants=dict(prepared.constants),
        post_index=0,
    )
    return PreparedAnnotationProof(
        source=prepared.source,
        constants=prepared.constants,
        kernel=prepared.kernel,
        declared_parameters=prepared.declared_parameters,
        annotation=annotation,
        first_config=first_config,
    )


def prepare_annotation_proof(
    source: str,
    kernel_name: str,
    constants: dict[str, int | float | bool],
    goal_name: str | None = None,
) -> PreparedAnnotationProof:
    """Prepare one named proof goal, sharing translation through the suite API."""

    return prepare_goal_proof(
        prepare_kernel_proofs(source, kernel_name, constants), goal_name
    )
