"""Shared full/SWA inventory and configuration-free semantic regressions."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "kernels")]
from scripts.deployment.common import _selected_proof_cases
from scripts.verification.kernel_interface_registry import load_registry
from scripts.verification.kernel_qualification_inventory import family_cases
from vosti_kernels.model_families.gemma3.profile import model_profiles, scope


class FourNormContractProfileTests(unittest.TestCase):
    def test_attention_semantics_keep_model_parameters_without_query_transform(self):
        layers = (ROOT / "src/boundary/dense_layer_primitives.rs").read_text()
        model = (ROOT / "src/proof/model/four_norm_gated/model.rs").read_text()
        runtime = (ROOT / "src/boundary/tensor_runtime.rs").read_text()
        self.assertNotIn("attention_scaled_query", layers + model)
        self.assertIn("parameters: AttentionParametersRepr", model)
        self.assertIn("dense_swiglu_runtime_attention_matches(runtime, parameters)", runtime)
        self.assertIn("ATTN::checked_runtime_binding", runtime)

    def test_complete_full_swa_profiles_are_collected_by_shared_inventory(self):
        inventories, _ = load_registry()
        for family in ("gemma3", "gemma4"):
            cases = family_cases(family, inventories[family])
            attention = [(c, constants) for c, constants in cases
                         if c["evidence"] == "conditional_relational_certificate"]
            with self.subTest(family=family):
                self.assertEqual(len(cases), 29)
                self.assertEqual(len(attention), 4 if family == "gemma3" else 2)
                self.assertEqual({c["wrapper"] for c, _ in attention},
                                 {"paged_attention", "paged_attention_swa"})

    def test_handwritten_semantics_does_not_dispatch_profile_widths(self):
        source = "".join((ROOT / "src/proof/model/four_norm_gated" / name).read_text() for name in
                         ("layers.rs", "model.rs"))
        for width in {p["model"]["hidden_size"] for p in model_profiles()}:
            self.assertNotIn(str(width), source)
        self.assertIn("RT::scaled_embed_kernel_cell_repr", source)
        self.assertIn("RT::offset_rms_norm_kernel_cell_repr", source)
        self.assertNotIn("backend_certificates", source)

    def test_new_row_geometry_is_collected_without_new_binding_tables(self):
        future = deepcopy(model_profiles()[0])
        future["model"]["hidden_size"] = 3072
        scaled = next(x for x in future["launches"] if x["wrapper"] == "gemma3_scaled_embed")
        scaled["key"]["width"] = 3072
        scaled["config"]["D"] = 3072
        cases = _selected_proof_cases(scope(), future["launches"])
        constants = next(c for contract, c in cases if contract["wrapper"] == "gemma3_scaled_embed")
        self.assertEqual(constants["D"], 3072)
        # Collection is not qualification: the shared qualifier must prove
        # this exact case before any deployment can admit it.


if __name__ == "__main__":
    unittest.main()
