"""gemma3 binding of the shared deployment contract regression suite."""
import unittest

from deployment_contract_cases import DeploymentContractCases
from vosti_kernels.model_families.gemma3 import deployment, runtime


class Gemma3DeploymentSchemaTests(DeploymentContractCases, unittest.TestCase):
    deployment = deployment
    runtime = runtime
    profile_name = "gemma-3-4b-it-text"
    expected_launch_count = 15


if __name__ == "__main__":
    unittest.main()
