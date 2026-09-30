import unittest

from vosti_kernels.dense_launch_plan import (
    QK_NORM_DISABLED,
    QK_NORM_RMS,
    derive_static_launch_plan,
    required_model_launch_inventory,
    validate_dense_launch,
)


SHAPE = {
    "hidden": 4096,
    "intermediate_half": 14336,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
}


def _config(**_key: int) -> dict:
    return {"TEST_CONFIG": 1}


class DenseLaunchCompositionTests(unittest.TestCase):
    def _derived(self, qk_norm: str) -> list[dict]:
        return derive_static_launch_plan(
            SHAPE,
            ((128256, 4096),),
            composition={"qk_norm": qk_norm},
            selectors={
                wrapper: _config
                for wrapper in {
                    "rms_norm", "add_rms_norm", "silu_and_mul", "embed",
                    "qk_norm", "store_kv_cache", "rotary_embed", "linear",
                    "qkv_linear", "paged_attention",
                }
            },
        )

    def _inventory(self, qk_norm: str) -> list[dict]:
        return required_model_launch_inventory(
            {
                "vocab_size": 128256,
                "geometry": SHAPE,
                "composition": {"qk_norm": qk_norm},
            }
        )

    def test_rms_composition_requires_both_qk_norm_launches(self) -> None:
        expected = [
            (launch["sites"], launch["key"])
            for launch in self._derived(QK_NORM_RMS)
            if launch["wrapper"] == "qk_norm"
        ]
        self.assertEqual(
            expected,
            [
                (["q_norm"], {"heads": 32, "head_dim": 128}),
                (["k_norm"], {"heads": 8, "head_dim": 128}),
            ],
        )
        self.assertEqual(
            [
                item
                for item in self._inventory(QK_NORM_RMS)
                if item["wrapper"] == "qk_norm"
            ],
            [
                {
                    "wrapper": "qk_norm",
                    "sites": ["q_norm"],
                    "key": {"heads": 32, "head_dim": 128},
                },
                {
                    "wrapper": "qk_norm",
                    "sites": ["k_norm"],
                    "key": {"heads": 8, "head_dim": 128},
                },
            ],
        )

    def test_disabled_composition_has_no_qk_norm_launch(self) -> None:
        self.assertFalse(
            any(
                launch["wrapper"] == "qk_norm"
                for launch in self._derived(QK_NORM_DISABLED)
            )
        )

    def test_qkv_projection_is_one_model_geometry_only_launch(self) -> None:
        expected = {
            "wrapper": "qkv_linear",
            "sites": ["qkv_projection"],
            "key": {"q_width": 4096, "kv_width": 1024, "k": 4096},
        }
        for qk_norm in (QK_NORM_RMS, QK_NORM_DISABLED):
            with self.subTest(qk_norm=qk_norm):
                derived = [
                    launch
                    for launch in self._derived(qk_norm)
                    if launch["wrapper"] == "qkv_linear"
                ]
                inventory = [
                    launch
                    for launch in self._inventory(qk_norm)
                    if launch["wrapper"] == "qkv_linear"
                ]
                self.assertEqual(len(derived), 1)
                self.assertEqual(
                    {key: derived[0][key] for key in ("wrapper", "sites", "key")},
                    expected,
                )
                self.assertEqual(inventory, [expected])
                self.assertFalse(
                    any(
                        site in {"q_projection", "kv_projection"}
                        for launch in self._derived(qk_norm)
                        for site in launch["sites"]
                    )
                )
        self.assertFalse(
            any(
                launch["wrapper"] == "qk_norm"
                for launch in self._inventory(QK_NORM_DISABLED)
            )
        )

    def test_missing_or_unknown_qk_norm_choice_is_rejected(self) -> None:
        for composition in ({}, {"qk_norm": "runtime_auto"}, None):
            with self.subTest(composition=composition):
                with self.assertRaisesRegex(ValueError, "qk_norm|composition"):
                    required_model_launch_inventory(
                        {
                            "vocab_size": 128256,
                            "geometry": SHAPE,
                            "composition": composition,
                        }
                    )

    def test_elementwise_activation_may_tile_a_wider_row(self) -> None:
        config = {
            "BLOCK_M": 1,
            "BLOCK_N": 1024,
            "num_warps": 8,
            "num_stages": 1,
        }
        validate_dense_launch(
            {
                "wrapper": "silu_and_mul",
                "key": {"width": 14336},
                "config": config,
            }
        )
        for wrapper in ("rms_norm", "add_rms_norm"):
            with self.subTest(wrapper=wrapper):
                with self.assertRaisesRegex(ValueError, "reduction block width"):
                    validate_dense_launch(
                        {
                            "wrapper": wrapper,
                            "key": {"width": 14336},
                            "config": config,
                        }
                    )


if __name__ == "__main__":
    unittest.main()
