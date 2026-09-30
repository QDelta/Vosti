"""Backend labels remain stable across baseline engine upgrades."""
import unittest

from scripts.determinism_tests.vllm_worker import _attention_engine_overrides


class AttentionBackendTests(unittest.TestCase):
    def test_flash_arm_pins_fa3(self):
        self.assertEqual(_attention_engine_overrides("FLASH_ATTN"), {
            "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 3},
        })

    def test_triton_arm_does_not_request_flash(self):
        self.assertEqual(_attention_engine_overrides("TRITON_ATTN"), {
            "attention_config": {"backend": "TRITON_ATTN"},
        })

    def test_auto_stays_auto(self):
        self.assertEqual(_attention_engine_overrides(None), {})
