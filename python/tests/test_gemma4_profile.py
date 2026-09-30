"""Static profile/inventory checks, not backend or full engine qualification."""

import copy
import importlib
import inspect
from pathlib import Path
import sys
import unittest

from vosti_kernels.dense_launch_plan import materialize_static_launch_plan, validate_launch_inventory
from vosti_kernels.model_families.gemma4 import profile


class Gemma4ProfileTests(unittest.TestCase):
    def test_exact_31b_profile_and_complete_static_sites(self):
        p = profile.model_profile_for_name("gemma-4-31b-it-text")
        c = profile.model_config(p)
        self.assertEqual(c["num_hidden_layers"], 60)
        self.assertEqual(profile.model_profile_for_config(c), p)
        self.assertEqual(len(p["launches"]), 23)
        by_site = {site: entry for entry in p["launches"] for site in entry["sites"]}
        self.assertEqual(by_site["global.qkv_projection"]["key"]["q_width"], 16384)
        self.assertEqual(by_site["global.v_norm"]["key"], {"heads": 4, "head_dim": 512})
        self.assertEqual(by_site["local.v_norm"]["key"], {"heads": 16, "head_dim": 256})
        self.assertEqual(by_site["full_attention"]["key"], {"head_dim": 512})
        self.assertIn("layer_output_scale", by_site)
        self.assertIn("final_logits_softcap", by_site)
        with self.assertRaisesRegex(ValueError, "outside the exact"):
            profile.model_profile_for_config({**c, "num_global_key_value_heads": 8})

    def test_profile_configs_equal_colocated_selectors_without_runtime_inputs(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))
        s = profile.scope()
        selectors = {c["wrapper"]: importlib.import_module("triton_kernels." + c["module"]).select_config
                     for c in s["kernel_contracts"]}
        for p in profile.model_profiles():
            inventory = profile.launch_inventory(p)
            for entry in inventory:
                params = set(inspect.signature(selectors[entry["wrapper"]]).parameters)
                self.assertEqual(set(entry["key"]), params)
                self.assertTrue(params.isdisjoint({"M", "m", "batch_size", "query_length", "cache_length"}))
            self.assertEqual(materialize_static_launch_plan(inventory, selectors), p["launches"])

    def test_rejects_missing_sites_extra_runtime_keys_and_invalid_configs(self):
        p = profile.model_profile_for_name("gemma-4-31b-it-text")
        inventory = profile.launch_inventory(p)
        changed = copy.deepcopy(p["launches"])
        changed[0]["sites"].append("runtime_selected_path")
        with self.assertRaisesRegex(ValueError, "call sites"):
            validate_launch_inventory(inventory, changed)
        changed = copy.deepcopy(p["launches"][0])
        changed["key"]["M"] = 1
        with self.assertRaises(ValueError):
            profile.validate_launch(changed)
        changed = next(copy.deepcopy(x) for x in p["launches"] if x["wrapper"] == "softcap")
        changed["config"]["BLOCK_M"] = 0
        with self.assertRaises(ValueError):
            profile.validate_launch(changed)

    def test_exact_12b_geometry_and_no_ambiguous_default(self):
        p = profile.model_profile_for_name("gemma-4-12b-it-text")
        c = profile.model_config(p)
        self.assertEqual((c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"]),
                         (48, 3840, 15360))
        self.assertEqual((c["num_attention_heads"], c["num_key_value_heads"],
                          c["num_global_key_value_heads"]), (16, 8, 1))
        self.assertEqual(profile.model_profile_for_config(c), p)
        self.assertEqual(len(p["launches"]), 23)
        with self.assertRaisesRegex(ValueError, "name is required"):
            profile.model_profile_for_name()


if __name__ == "__main__":
    unittest.main()
