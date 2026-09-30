import importlib
import unittest

from vosti_kernels.kernel_selection import select_static_launches


FAMILIES = ("qwen3", "gemma3", "gemma4", "llama3")


class KernelSelectionTests(unittest.TestCase):
    def test_every_family_uses_colocated_kernel_selectors(self) -> None:
        for family in FAMILIES:
            with self.subTest(family=family):
                profile_module = importlib.import_module(
                    f"vosti_kernels.model_families.{family}.profile"
                )
                scope = profile_module.scope()
                for contract in scope["kernel_contracts"]:
                    kernel = importlib.import_module(
                        f"triton_kernels.{contract['module']}"
                    )
                    self.assertTrue(callable(getattr(kernel, "select_config", None)))
                for profile in profile_module.model_profiles():
                    selected = select_static_launches(
                        scope["kernel_contracts"],
                        profile_module.launch_inventory(profile),
                    )
                    self.assertEqual(selected, profile["launches"])

    def test_page_size_is_not_configuration(self) -> None:
        forbidden = {"page_size", "page_block_size", "PAGE_BLOCK_SIZE"}
        for family in FAMILIES:
            profile_module = importlib.import_module(
                f"vosti_kernels.model_families.{family}.profile"
            )
            scope = profile_module.scope()
            self.assertTrue(forbidden.isdisjoint(scope["runtime"]))
            for profile in profile_module.model_profiles():
                for launch in profile["launches"]:
                    self.assertTrue(forbidden.isdisjoint(launch["key"]))
                    self.assertTrue(forbidden.isdisjoint(launch["config"]))


if __name__ == "__main__":
    unittest.main()
