"""Checked generated attention adapters, including rejected weakened premises.

Runtime finiteness remains an explicit assumption; these tests check its
retention and binding in the production adapter, not its truth.
"""

import os
from dataclasses import fields, replace
from functools import cache
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from spec_test_support import page_constants_source

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kernels"))
sys.path.insert(0, str(ROOT / "scripts"))
from ir.relational_verifier import verify_annotations
from scripts.verification.paged_attention_adapter_codegen import render_checked_paged_attention
from scripts.verification.kernel_interface_codegen import RenderedKernelInterface
from scripts.verification.engine_kernel_bindings import FULL_ATTENTION_BINDING, SLIDING_ATTENTION_BINDING

VERUS = os.environ.get("VERUS") or shutil.which("verus")


@cache
def proved_goals(swa, head_dim=128, block_m=16):
    filename = "fattn_paged_swa.py" if swa else "fattn_paged.py"
    kernel = "fattn_varlen_paged_swa_kernel" if swa else "fattn_varlen_paged_fwd_block_ptr_kernel"
    source = (ROOT / "kernels/triton_kernels" / filename).read_text()
    constants = dict(BLOCK_M=block_m, BLOCK_N=64, D_HEAD=head_dim, PAGE_BLOCK_SIZE=64)
    contracts = [verify_annotations(source, kernel, constants, goal_name=goal,
                                   preserve_analyzer_conditions=True).verified_contract
                 for goal in ("batch_invariance", "selected_row_prefix_equivalence")]
    return tuple(contracts)


@cache
def raw_bundle(swa, head_dim=128, block_m=16):
    return render_checked_paged_attention(proved_goals(swa, head_dim, block_m), symbol_prefix="raw",
        binding=SLIDING_ATTENTION_BINDING if swa else FULL_ATTENTION_BINDING)


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check raw attention composition")
@pytest.mark.parametrize("swa", [False, True], ids=["full", "swa"])
@pytest.mark.parametrize("case", ["positive", "missing_geometry", "wrong_query",
                                 "missing_k_equality", "missing_v_equality", "partial_last_page",
                                 "missing_numeric_assumption", "missing_numeric_obligation",
                                 "missing_selected_query", "wrong_causal_position",
                                 "short_selected_prefix", "wrong_numeric_selector",
                                 "missing_canonical_numeric", "wrong_canonical_query", "wrong_canonical_scale",
                                 "missing_launch_numeric", "wrong_launch_mapping", "missing_model_geometry",
                                 "missing_launch_row_numeric", "wrong_launch_row_numeric_selector"])
def test_raw_attention_batch_adapter(tmp_path, swa, case, head_dim=128, block_m=16, adapter_override=None,
                                     binding_override=None, smt_seed=None):
    adapter = adapter_override or raw_bundle(swa, head_dim, block_m)
    bundle = adapter.raw
    raw_body = bundle.body
    if isinstance(bundle, RenderedKernelInterface):
        bundle = bundle.representative
    binding = binding_override or (SLIDING_ATTENTION_BINDING if swa else FULL_ATTENTION_BINDING)
    batch = next(c for c in bundle.contracts if c.symbol_prefix == 'raw_' + binding.batch_goal)
    selected = next(c for c in bundle.contracts if c.symbol_prefix == 'raw_' + binding.selected_goal)
    assert not batch.analyzer_conditions
    template = adapter.adapter_body
    assert "external_body" not in template and "assume(" not in template
    if case == "missing_geometry":
        premise = "        TS::tensor2d_shape(q, q.len(), h * 128),\n"
        assert template.count(premise) == 4  # Batch, canonical, and two launch adapters.
        template = template.replace(premise, "", 1)
    elif case == "wrong_query":
        site = "let right = adapter_side(q.subrange(start, end),"
        assert template.count(site) == 1
        template = template.replace(site, "let right = adapter_side(q.subrange(0, qlen as int),")
    elif case in {"missing_k_equality", "missing_v_equality"}:
        cache_name = "k" if case == "missing_k_equality" else "v"
        premise = f'''        forall|pos: nat| pos < common::blocks_needed_for((cu_k[i + 1] - cu_k[i]) as nat) * 64 ==>
            (#[trigger] common::cache_at({cache_name}, common::block_table_slot(table[i], pos)))
                == common::cache_at(selected_{cache_name}, common::block_table_slot(selected_row, pos)),
'''
        assert template.count(premise) == 1
        template = template.replace(premise, "")
    elif case == "partial_last_page":
        bound = "pos < common::blocks_needed_for((cu_k[i + 1] - cu_k[i]) as nat) * 64"
        assert template.count(bound) == 2
        template = template.replace(bound, "pos < (cu_k[i + 1] - cu_k[i]) as nat")
    elif case in {"missing_numeric_assumption", "missing_numeric_obligation"}:
        kind = "used_assumption" if case == "missing_numeric_assumption" else "external_obligation"
        condition, = [c for c in selected.analyzer_conditions if c.kind == kind]
        predicate = f"{condition.predicate_name}(left, right, free)"
        assert template.count(predicate) == 1
        template = template.replace(predicate, "true")
    elif case == "missing_selected_query":
        premise = "        qa[ja] == qb[jb],\n"
        assert template.count(premise) == 1
        template = template.replace(premise, "")
    elif case == "wrong_causal_position":
        premise = "        kl_a - qa.len() + ja == kl_b - qb.len() + jb,\n"
        assert template.count(premise) == 1
        template = template.replace(premise, "")
    elif case == "short_selected_prefix":
        bound = "pos <= kl_a - qa.len() + ja"
        assert template.count(bound) == 2
        template = template.replace(bound, "pos < kl_a - qa.len() + ja")
    elif case == "wrong_numeric_selector":
        premise = f"{selected.free_type} {{ selected_left_row: ja, selected_right_row: jb }}),"
        assert template.count(premise) == 1
        template = template.replace(premise, premise.replace("selected_right_row: jb", "selected_right_row: 0"))
    elif case == "missing_canonical_numeric":
        start = template.index("        selected_runtime_assumptions(\n            adapter_side(q, k, v,")
        end_marker = f"{selected.free_type} {{ selected_left_row: j, selected_right_row: 0 }}),\n"
        end = template.index(end_marker, start) + len(end_marker)
        template = template[:start] + template[end:]
    elif case == "wrong_canonical_query":
        equality = "        == canonical_row_output(q[j], logical_prefix(k, row,"
        assert template.count(equality) == 1
        template = template.replace(equality, equality.replace("q[j]", "q[0]"))
    elif case == "wrong_canonical_scale":
        # Sharing Q/K/V is insufficient when the immutable scale differs.
        # In particular, no unproved query-rescaling equivalence is imported.
        start = template.index("        == canonical_row_output(q[j],")
        end = template.index("\n{", start)
        conclusion = template[start:end]
        assert conclusion.count("h, hkv, scale, window, fill") == 1
        template = template[:start] + conclusion.replace(
            "h, hkv, scale, window, fill", "h, hkv, fill, window, fill"
        ) + template[end:]
    elif case == "missing_launch_row_numeric":
        start = template.index("pub proof fn checked_launch_row_equivalence(")
        end = template.index("{\n", start)
        header = template[start:end]
        premise = "        launch_runtime_assumptions(q, k, v, table, cu_q, cu_k, h, hkv, scale, window, fill),\n"
        assert header.count(premise) == 1
        template = template[:start] + header.replace(premise, "") + template[end:]
    elif case == "wrong_launch_row_numeric_selector":
        # The selected row's exact numeric premise must be transported, not
        # merely an assumption about row zero of the same singleton request.
        selector = "selected_left_row: token - cu_q[request]"
        assert template.count(selector) == 1
        template = template.replace(selector, "selected_left_row: 0")
    elif case in {"missing_launch_numeric", "wrong_launch_mapping"}:
        start = template.index("pub proof fn checked_launch_equivalence(")
        end = template.index("{\n", start)
        caller = template[start:end].replace("checked_launch_equivalence", "negative_launch_caller")
        if case == "missing_launch_numeric":
            premise = "        launch_runtime_assumptions(q, k, v, table, cu_q, cu_k, h, hkv, scale, window, fill),\n"
            assert caller.count(premise) == 1
            caller = caller.replace(premise, "").split("    ensures", 1)[0]
        else:
            site = "mapped_launch_output(q, k, v,"
            assert caller.count(site) == 1
            caller = caller.replace(site, "mapped_launch_output(q, v, k,")
        template += '\nverus! {\n' + caller + '''{
    checked_launch_equivalence(q, k, v, table, cu_q, cu_k, max_q, max_k,
        h, hkv, scale, window, fill);
}
}
'''
    elif case == "missing_model_geometry":
        start = template.index("pub proof fn checked_bound_launch_equivalence(")
        end = template.index("{\n", start)
        header = template[start:end].replace("checked_bound_launch_equivalence", "negative_binding_caller")
        premise = "        model_binding_valid(geometry, window),\n"
        assert header.count(premise) == 1
        header = header.replace(premise, "").split("    ensures", 1)[0]
        template += '\nverus! {\n' + header + '''{
    checked_bound_launch_equivalence(q, k, v, table, cu_q, cu_k, max_q, max_k,
        geometry, scale, window, fill);
}
}
'''
    if case == "positive":
        # Nonvacuous caller: different pool sizes and physical page IDs, then
        # the original shared-cache specialization of the very same theorem.
        template += (ROOT / "python/tests/fixtures/raw_attention_relocation_caller.rs").read_text().replace("__D__", str(head_dim))
    header = f'''
use vstd::prelude::*;
#[path = "{ROOT / 'src/boundary/scalar.rs'}"] pub mod scalar_boundary;
mod fixture_types {{
    use vstd::prelude::*;
    verus! {{
        pub struct FloatParameterBits {{ pub bits: u64 }}
        pub struct AttentionGeometryRepr {{
            pub num_attention_heads: nat,
            pub num_key_value_heads: nat,
            pub head_dim: nat,
        }}
    }}
    pub type Scalar = super::scalar_boundary::Scalar;
    pub type BlockId = u64;
    pub type Tensor1D = Seq<Scalar>;
    pub type Tensor2D = Seq<Tensor1D>;
    pub type Tensor3D = Seq<Tensor2D>;
    pub type Tensor4D = Seq<Tensor3D>;
    pub type KVCacheLayerRepr = Tensor3D;
    pub type IntTensor2D = Seq<Seq<int>>;
}}
#[path = "{ROOT / 'src/proof/tensor/geometry.rs'}"] pub mod common;
#[path = "{ROOT / 'src/proof/tensor/seq_flatten.rs'}"] pub mod seq_flatten;
mod proof {{
pub mod model {{ pub mod types {{ pub use crate::fixture_types::*; }} }}
pub mod tensor {{
pub mod types {{ pub use crate::fixture_types::*; }}
pub use crate::common as geometry;
pub use crate::seq_flatten as seq_flatten;

    #[path = "{ROOT / 'src/proof/tensor/attention_projection.rs'}"] pub mod attention_projection;
    #[path = "{ROOT / 'src/proof/tensor/paged.rs'}"] pub mod paged;
    #[path = "{ROOT / 'src/proof/tensor/shape.rs'}"] pub mod shape;
    #[path = "{ROOT / 'src/proof/tensor/layout.rs'}"] pub mod layout;

}}
}}
mod boundary {{
pub use crate::scalar_boundary as scalar;
 pub mod backend_certificates {{
    #[path = "{ROOT / 'src/boundary/backend_certificates/support.rs'}"] pub mod support;
}} }}

mod model_config {{ pub use crate::fixture_types::*; }}
mod types {{
    pub use crate::fixture_types::*;
    use vstd::prelude::*;
    verus! {{ {page_constants_source()} }}
}}

fn main() {{}}
'''
    path = tmp_path / "attention_adapter.rs"
    path.write_text(header + '\nmod generated { use vstd::prelude::*;\nverus! {\n'
                    + raw_body + '\n}\n' + template + '\n}\n')
    command = [VERUS, str(path)]
    if smt_seed is not None:
        command += ["--smt-option", f"smt.random_seed={smt_seed}"]
    if case != "positive":
        # Each positive case checks EVERY proof body, including dependencies.
        # Negative probes isolate the intentionally broken obligation rather
        # than asking downstream callers to use its mutated contract as well.
        if case in {"missing_geometry", "wrong_query", "missing_k_equality", "missing_v_equality", "partial_last_page", "partial_output_contract"}:
            target = "checked_batch_projection"
        elif case in {"missing_canonical_numeric", "wrong_canonical_query", "wrong_canonical_scale"}:
            target = "checked_canonical_selected_projection"
        elif case in {"missing_launch_numeric", "wrong_launch_mapping"}:
            target = "negative_launch_caller"
        elif case in {"missing_launch_row_numeric", "wrong_launch_row_numeric_selector"}:
            target = "checked_launch_row_equivalence"
        elif case == "missing_model_geometry":
            target = "negative_binding_caller"
        else:
            target = "checked_selected_row_projection"
        command += ["--verify-only-module", "generated", "--verify-function", target]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if case == "positive":
        assert result.returncode == 0 and "0 errors" in result.stdout, result.stdout + result.stderr
    else:
        assert result.returncode != 0, "incomplete adapter unexpectedly verified"
        failures = [line for line in re.findall(r"^error: (.+)$", result.stderr, re.MULTILINE)
                    if not line.startswith("aborting due to")]
        allowed = {"assertion failed", "precondition not satisfied"}
        if case in {"wrong_query", "wrong_canonical_query", "wrong_canonical_scale", "wrong_launch_mapping", "partial_output_contract"}:
            # The wrong input can also invalidate the final segment equality.
            allowed.add("postcondition not satisfied")
        assert failures and set(failures) <= allowed, result
        if case in {"missing_k_equality", "missing_v_equality", "partial_last_page"}:
            assert "PL::lemma_rectangular_cache_page_equality" in result.stderr, result


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check raw attention composition")
@pytest.mark.parametrize("swa,head_dim,block_m", [
    (False, 128, 32), (True, 128, 32),
    (False, 256, 32), (True, 256, 32), (False, 512, 16),
])
def test_checked_attention_static_geometries(tmp_path, swa, head_dim, block_m):
    test_raw_attention_batch_adapter(tmp_path, swa, "positive", head_dim, block_m)


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check attention solver seeds")
@pytest.mark.parametrize("swa,head_dim", [
    (False, 128), (False, 256), (False, 512), (True, 128), (True, 256),
])
@pytest.mark.parametrize("smt_seed", [1, 2])
def test_checked_attention_solver_seeds(tmp_path, swa, head_dim, smt_seed):
    # Check every body in each generated adapter, not just its launch caller.
    test_raw_attention_batch_adapter(tmp_path, swa, "positive", head_dim=head_dim,
                                     smt_seed=smt_seed)


def test_checked_attention_renderer_rejects_missing_goal():
    with pytest.raises(ValueError, match="expected bound batch and selected-row goals"):
        render_checked_paged_attention(proved_goals(False)[:1], symbol_prefix="raw",
                                       binding=FULL_ATTENTION_BINDING)


@pytest.mark.parametrize('swa', [False, True], ids=['full', 'swa'])
def test_renamed_attention_contract_composes_through_checked_adapter(tmp_path, swa):
    from ir.proof_preparation import prepare_kernel_proofs
    from ir.kernel_verifier import verify_kernel_goal
    filename = 'fattn_paged_swa.py' if swa else 'fattn_paged.py'
    entry = 'fattn_varlen_paged_swa_kernel' if swa else 'fattn_varlen_paged_fwd_block_ptr_kernel'
    source = (ROOT / 'kernels/triton_kernels' / filename).read_text()
    binding = SLIDING_ATTENTION_BINDING if swa else FULL_ATTENTION_BINDING
    names = {getattr(binding, f.name): 'port_' + f.name for f in fields(binding)
             if getattr(binding, f.name) is not None}
    names.update({n: 'extent_' + n.lower() for n in
                  ('Tq', 'H', 'Hkv', 'B', 'MAX_NUM_PAGES', 'NUM_PAGES', 'D_HEAD', 'PAGE_BLOCK_SIZE')})
    names.update({entry: 'masked_operator', 'BLOCK_M': 'ROW_TILE', 'BLOCK_N': 'COLUMN_TILE',
                  binding.batch_goal: 'z_batch', binding.selected_goal: 'a_selected'})
    source = re.sub(r'\b(?:' + '|'.join(names) + r')\b', lambda m: names[m[0]], source)
    renamed = replace(binding, **{f.name: names[getattr(binding, f.name)] for f in fields(binding)
                                   if getattr(binding, f.name) is not None})
    constants = {names[k]: v for k, v in dict(BLOCK_M=16, BLOCK_N=64, D_HEAD=128, PAGE_BLOCK_SIZE=64).items()}
    prepared = prepare_kernel_proofs(source, names[entry], constants)
    contracts = tuple(verify_kernel_goal(prepared, g.name, preserve_analyzer_conditions=True)
                      for g in prepared.goals)
    adapter = render_checked_paged_attention(contracts, symbol_prefix='raw', binding=renamed)
    assert '.port_output' in adapter.adapter_body
    if VERUS:
        test_raw_attention_batch_adapter(tmp_path, swa, 'positive', adapter_override=adapter,
                                         binding_override=renamed)


@pytest.mark.parametrize('change', ['aliased_ports', 'missing_port', 'wrong_goal', 'missing_window'])
def test_paged_binding_rejects_incomplete_or_ambiguous_roles(change):
    binding = SLIDING_ATTENTION_BINDING
    wrong = {
        'aliased_ports': replace(binding, values=binding.keys),
        'missing_port': replace(binding, query='absent'),
        'wrong_goal': replace(binding, batch_goal=binding.selected_goal),
        'missing_window': replace(binding, window=None),
    }[change]
    with pytest.raises(ValueError):
        render_checked_paged_attention(proved_goals(True), symbol_prefix='raw', binding=wrong)


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check partial-output rejection")
def test_partial_raw_output_cannot_justify_whole_engine_rows(tmp_path):
    # The source really proves this weaker goal. Raw export is legitimate, but
    # the production adapter must not promote one head component to a full row.
    source = (ROOT / "kernels/triton_kernels/fattn_paged.py").read_text()
    post = next(line for line in source.splitlines() if line.startswith("#     left(o)["))
    assert post.count("0:D_HEAD") == 2
    source = source.replace(post, post.replace("0:D_HEAD", "0:1"), 1)
    constants = dict(BLOCK_M=16, BLOCK_N=64, D_HEAD=128, PAGE_BLOCK_SIZE=64)
    contracts = tuple(verify_annotations(source, "fattn_varlen_paged_fwd_block_ptr_kernel",
        constants, goal_name=goal, preserve_analyzer_conditions=True).verified_contract
        for goal in ("batch_invariance", "selected_row_prefix_equivalence"))
    adapter = render_checked_paged_attention(contracts, symbol_prefix="raw", binding=FULL_ATTENTION_BINDING)
    test_raw_attention_batch_adapter(tmp_path, False, "partial_output_contract", adapter_override=adapter)


def test_missing_singleton_extent_is_rejected_by_source_qualification():
    source = (ROOT / "kernels/triton_kernels/fattn_paged.py").read_text()
    closure = "#     right(cu_seqlens_q)[1] == right(Tq),\n"
    assert source.count(closure) == 1
    with pytest.raises(ValueError):
        verify_annotations(source.replace(closure, ""), "fattn_varlen_paged_fwd_block_ptr_kernel",
            dict(BLOCK_M=16, BLOCK_N=64, D_HEAD=128, PAGE_BLOCK_SIZE=64), goal_name="batch_invariance")
