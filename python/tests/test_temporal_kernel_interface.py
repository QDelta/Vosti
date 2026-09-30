"""Temporal and mixed theories use the existing fixed-deployment interface."""

from functools import cache
from dataclasses import replace
import json
import os
import re
import subprocess

import pytest

from scripts.deployment.common import qualify_proof_case
from scripts.verification.kernel_interface_codegen import render_kernel_interface
from scripts.verification.kernel_qualification_inventory import selected_kernel_cases
from scripts.verification.mutation_interface_codegen import render_mutation_interface, render_mutation_adapter, infer_scatter_binding
from ir.exact_effects import verify_exact_effects
from ir.verus_exact_effect import render_verified_exact_effect_to_verus
from pathlib import Path


@cache
def qualified():
    return tuple(qualify_proof_case(c, config, ("bfloat16", "float32"))
                 for c, config, _ in selected_kernel_cases()
                 if c["evidence"] == "exact_effect_certificate")


def test_all_mutation_geometries_export_one_fixed_execution_theory():
    assert len(qualified()) == 4
    for case in qualified():
        interface = render_kernel_interface((case.contracts,), symbol_prefix="raw")
        reversed_goals = render_kernel_interface((tuple(reversed(case.contracts)),), symbol_prefix="raw")
        assert interface == reversed_goals
        assert "Proved exact-effect contract:" not in interface.body
        assert "before.__element_dtype_x == before.__element_dtype_cache" in interface.body
        assert interface.body.count("pub open spec fn raw_execute(") == 1
        manifest = json.loads(interface.manifest())
        assert manifest["interpretation"] == "one_fixed_deployed_implementation"
        assert len(manifest["implementations"][0]["standalone_contracts"]) == 2


def test_temporal_only_interface_is_not_mistaken_for_a_relational_certificate():
    case = qualified()[0]
    interface = render_kernel_interface(((case.contracts[1],),), symbol_prefix="raw")
    assert "raw_exact_effect_certificate" in interface.body
    assert "raw_batch_invariance_certificate" not in interface.body


def test_different_static_geometries_are_not_equated():
    a, b = qualified()[:2]
    with pytest.raises(ValueError, match="different logical contracts"):
        render_kernel_interface((a.contracts, b.contracts), symbol_prefix="raw")


@cache
def mutation_catalog():
    cases = [(c, config, families, qualify_proof_case(c, config, ("bfloat16", "float32")))
             for c, config, families in selected_kernel_cases() if c["evidence"] == "exact_effect_certificate"]
    return render_mutation_interface(cases)


def test_mutation_geometry_catalog_uses_exact_source_artifacts():
    body, manifest = mutation_catalog()
    data = json.loads(manifest)
    assert {entry["geometry"]["KVD"] for entry in data["interfaces"]} == {512, 1024, 2048, 4096}
    assert len(data["inventory_contributions"]) == 4
    assert "BLOCK_M" not in body
    assert "checked_copy" in body and "checked_frame" in body


def test_mutation_catalog_does_not_depend_on_qualification_goal_order():
    cases = [(c, config, families, replace(q, contracts=tuple(reversed(q.contracts))))
             for (c, config, families), q in zip(
                 [case for case in selected_kernel_cases()
                  if case[0]["evidence"] == "exact_effect_certificate"], qualified())]
    assert render_mutation_interface(cases) == mutation_catalog()


@pytest.mark.skipif(not os.environ.get("VERUS"), reason="set VERUS to check mutation adapters")
def test_mutation_copy_frame_adapters_pass_verus(tmp_path):
    body, _ = mutation_catalog()
    check_mutation_adapter(tmp_path, body)


@pytest.mark.parametrize('change', ['pre_order', 'post_order', 'binder_names'])
def test_mutation_adapter_tracks_typed_facts_not_helper_order(tmp_path, change):
    source = (Path(__file__).resolve().parents[2] / 'kernels/triton_kernels/store_kv_cache.py').read_text()
    start, end = source.index('# @verif(exact_effect,'), source.index('# @kernel-bridge-begin')
    goal = source[start:end]
    if change == 'binder_names':
        goal = re.sub(r'\bi\b', 'token', goal)
        goal = re.sub(r'\bs\b', 'slot', goal)
    else:
        offset = 0 if change == 'pre_order' else goal.index('#   post(')
        first = goal.index('#     forall(i, ', offset)
        second = goal.index('#     forall(i, j,' if change == 'pre_order' else '#     forall(s,', first)
        stop = goal.index('#   ),', second)
        goal = goal[:first] + goal[second:stop] + goal[first:second] + goal[stop:]
    report = verify_exact_effects(source[:start] + goal + source[end:], 'store_cache_kernel',
                                  {'KVD': 512, 'BLOCK_M': 1})
    assert report.proved, report.unsupported_reason
    raw = render_verified_exact_effect_to_verus(report.verified_contract, symbol_prefix='raw_exact_effect')
    # Use the production execution prefix, independently of the goal prefix.
    interface = render_kernel_interface(((report.verified_contract,),), symbol_prefix='raw')
    fragment = interface.representative.contracts[0]
    adapter = render_mutation_adapter(fragment, infer_scatter_binding(report.verified_contract))
    assert raw.helpers and fragment.helpers
    if os.environ.get('VERUS'):
        check_mutation_adapter(tmp_path,
            'use vstd::prelude::*;\nuse crate::{types::*, proof::model::types::*, proof::tensor::types::*};\nverus! {\n'
            + interface.body + adapter + '\n}')


@pytest.mark.parametrize('change', ['missing', 'duplicate'])
def test_mutation_adapter_rejects_ambiguous_helper_metadata(change):
    interface = render_kernel_interface(((qualified()[0].contracts[1],),), symbol_prefix='raw')
    fragment = interface.representative.contracts[0]
    helpers = () if change == 'missing' else fragment.helpers + fragment.helpers
    with pytest.raises(ValueError, match='ambiguous or unsupported'):
        render_mutation_adapter(replace(fragment, helpers=helpers), infer_scatter_binding(qualified()[0].contracts[1]))


@pytest.mark.parametrize('change', ['all_names', 'parameter_order'])
def test_scatter_binding_and_checked_adapter_are_name_independent(tmp_path, change):
    source = (Path(__file__).resolve().parents[2] / 'kernels/triton_kernels/store_kv_cache.py').read_text()
    entry, goal = 'store_cache_kernel', 'exact_effect'
    constants = {'KVD': 512, 'BLOCK_M': 1}
    if change == 'all_names':
        names = dict(store_cache_kernel='write_rows', exact_effect='preserve_and_copy',
                     x='payload', cache='destination', slot_mapping='indices',
                     M='row_count', NUM_SLOTS='capacity', KVD='row_width', BLOCK_M='row_tile',
                     i='token', j='other_token', s='slot')
        source = re.sub(r'\b(?:' + '|'.join(names) + r')\b', lambda m: names[m[0]], source)
        entry, goal = names[entry], names[goal]
        constants = {names[k]: v for k, v in constants.items()}
    else:
        first = '#   tensor(x, float, shape(M, KVD), strides(stride_xm, stride_xd)),\n'
        second = '#   tensor(slot_mapping, int32, shape(M), strides(stride_s)),\n'
        assert first in source
        assert first + second in source
        source = source.replace(first + second, second + first, 1)
        # Reorder the function parameters too, without changing their roles.
        source = source.replace('    x,\n    slot_mapping,\n    cache,',
                                '    cache,\n    x,\n    slot_mapping,', 1)
    report = verify_exact_effects(source, entry, constants, goal_name=goal)
    assert report.proved, report.unsupported_reason
    binding = infer_scatter_binding(report.verified_contract)
    # Exercise a non-production export prefix as well as a renamed goal.
    interface = render_kernel_interface(((report.verified_contract,),), symbol_prefix='changed')
    adapter = render_mutation_adapter(interface.representative.contracts[0], binding)
    assert 'raw_exact_effect' not in adapter
    if change == 'all_names':
        assert binding.source == 'payload' and binding.destination == 'destination'
        assert binding.width_symbol == 'row_width' and binding.width == 512
        assert 'after.destination' in adapter
    if os.environ.get('VERUS'):
        check_mutation_adapter(tmp_path,
            'use vstd::prelude::*;\nuse crate::{types::*, proof::model::types::*, proof::tensor::types::*};\nverus! {\n'
            + interface.body + adapter + '\n}')


@pytest.mark.parametrize('change', ['destination', 'indices', 'source'])
def test_incorrect_scatter_port_bindings_are_rejected(change):
    effect = qualified()[0].contracts[1]
    binding = infer_scatter_binding(effect)
    fragment = render_kernel_interface(((effect,),), symbol_prefix='raw').representative.contracts[0]
    wrong = replace(binding, **{change: 'not_a_port'})
    with pytest.raises(ValueError, match='differs from raw execution ports'):
        render_mutation_adapter(fragment, wrong)


def check_mutation_adapter(tmp_path, body):
    source = '''#![allow(non_snake_case)]
use vstd::prelude::*;
verus! {
#[verifier::external_body]
#[verifier::ext_equal]
pub struct Scalar { _private: () }
}
pub mod fixture_types {
    use vstd::prelude::*;
    pub use crate::Scalar;
    pub type Tensor2D = Seq<Seq<Scalar>>;
    pub type IntTensor1D = Seq<int>;
}
pub mod types { pub use crate::fixture_types::*; }
pub mod proof {
    pub mod model { pub mod types { pub use crate::fixture_types::*; } }
    pub mod tensor { pub mod types { pub use crate::fixture_types::*; } }
}
pub mod boundary { pub mod scalar { pub use crate::Scalar; } }
pub mod generated {
''' + body + '\n}\nfn main() {}\n'
    path = tmp_path / "mutation.rs"
    path.write_text(source)
    result = subprocess.run([os.environ["VERUS"], str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
