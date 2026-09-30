"""gemma4 binding of the shared deployment contract regression suite."""
import unittest

from deployment_contract_cases import DeploymentContractCases
from vosti_kernels.model_families.gemma4 import deployment, runtime


class Gemma4DeploymentSchemaTests(DeploymentContractCases, unittest.TestCase):
    deployment = deployment
    runtime = runtime
    profile_name = "gemma-4-31b-it-text"
    expected_launch_count = 23


class Gemma4UnifiedDeploymentSchemaTests(DeploymentContractCases, unittest.TestCase):
    deployment = deployment
    runtime = runtime
    profile_name = "gemma-4-12b-it-text"
    expected_launch_count = 23


if __name__ == "__main__":
    unittest.main()
