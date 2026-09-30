"""Qwen3 binding for the generic deployment qualification workflow."""

from functools import partial

from ... import deployment as generic
from . import profile

ARCHITECTURE = generic.DeploymentArchitecture.from_profile(profile)

CANDIDATE_SCHEMA = generic.CANDIDATE_SCHEMA
REPORT_SCHEMA = generic.REPORT_SCHEMA
DEPLOYMENT_SCHEMA = generic.DEPLOYMENT_SCHEMA
scope = partial(generic.scope, ARCHITECTURE)
scope_sha256 = partial(generic.scope_sha256, ARCHITECTURE)
bind_launches_to_proof_cases = partial(generic.bind_launches_to_proof_cases, ARCHITECTURE)
validate_candidate = partial(generic.validate_candidate, ARCHITECTURE)
seal_candidate = partial(generic.seal_candidate, ARCHITECTURE)
load_bundle = partial(generic.load_bundle, ARCHITECTURE)
validate_runtime_binding = partial(generic.validate_runtime_binding, ARCHITECTURE)
