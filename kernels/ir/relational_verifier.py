"""One fail-closed annotation-to-artifact entrypoint for relational goals.

The unified analysis proves equality; this boundary requires qualified typed
evidence bound to that exact prepared theorem. Consumers must explicitly retain
any analyzer-added conditions. Neither operation authenticates the trusted
Python producer or proves its correspondence with physical GPU execution.
"""

from .contract_schema import validate_relational_contract
from .relational_artifact import _build_verified_dataflow_contract
from .relational_dataflow import (
    RelationalDataflowReport, prove_prepared_relational_dataflow,
)
from .proof_preparation import PreparedAnnotationProof, prepare_annotation_proof


def validate_dataflow_artifact(
    prepared: PreparedAnnotationProof, report: RelationalDataflowReport,
) -> dict:
    """Require supported success and exact artifact-to-live-proof binding.

    Rebuilding serializes the existing typed evidence; it does not run another
    analysis or recover proof meaning from diagnostic strings. Qualification
    requires satisfiable typed annotation premises; this does not establish
    physical realizability or discharge analyzer-added numerical conditions.
    """
    if not report.proved or report.unsupported_reason is not None:
        raise ValueError("relational verifier: failed or unsupported dataflow proof")
    if report.verified_contract is None:
        raise ValueError("relational verifier: missing qualified artifact")
    actual = report.verified_contract.to_data()
    expected = _build_verified_dataflow_contract(prepared, report)
    if expected is None or report.verified_contract != expected:
        raise ValueError("relational verifier: artifact differs from prepared proof/report")
    return actual


def verify_prepared_annotations(
    prepared: PreparedAnnotationProof, *, preserve_analyzer_conditions: bool = False,
) -> RelationalDataflowReport:
    """Verify once, reject unqualified evidence, and enforce consumer policy.

    Default consumers cannot import extra assumptions. Opting into preservation
    is not a discharge: a conditional consumer must carry them into its theorem.
    """
    if type(preserve_analyzer_conditions) is not bool:
        raise TypeError("preserve_analyzer_conditions must be boolean")
    dataflow = prove_prepared_relational_dataflow(prepared)
    validate_dataflow_artifact(prepared, dataflow)
    validate_relational_contract(
        dataflow.verified_contract, family="relational verifier",
        preserve_analyzer_conditions=preserve_analyzer_conditions,
    )
    return dataflow


def verify_annotations(
    source: str, kernel_name: str, constants: dict[str, int | float | bool],
    *, goal_name: str = "batch_invariance", preserve_analyzer_conditions: bool = False,
) -> RelationalDataflowReport:
    return verify_prepared_annotations(prepare_annotation_proof(
        source, kernel_name, constants, goal_name=goal_name,
    ), preserve_analyzer_conditions=preserve_analyzer_conditions)
