"""Multiple annotation goals must refer to one kernel, not independent models."""

from functools import cache
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from ir.relational_verifier import verify_annotations
from ir.verus_contract import (
    render_contract_manifest, render_verified_kernel_to_verus, transpile_verified_kernel_goals,
    render_standalone_verus_module, render_verified_contract_to_verus,
    transpile_verified_kernel_source,
)


KERNELS = Path(__file__).resolve().parents[2] / 'triton_kernels'


@cache
def row_contracts():
    source = (KERNELS / 'silu_mul.py').read_text()
    start, end = source.index('# @verif('), source.index('@triton.jit')
    additional = source[start:end].replace('batch_invariance', 'selected_row')
    additional = re.sub(r'\bb\b', 'selected', additional)
    source = source[:end] + additional + source[end:]
    constants = {'BLOCK_M': 1, 'BLOCK_N': 4096}
    contracts = tuple(verify_annotations(source, 'silu_mul_kernel', constants,
        goal_name=goal).verified_contract for goal in ('batch_invariance', 'selected_row'))
    return source, constants, contracts


def test_row_goals_share_one_execution_without_an_equivalence_assumption():
    _, _, contracts = row_contracts()
    bundle = render_verified_kernel_to_verus(contracts, symbol_prefix='row')
    assert bundle.body.count('pub struct RowSide') == 1
    assert bundle.body.count('pub open spec fn row_execute(') == 1
    assert bundle.body.count('pub uninterp spec fn row_o_cell(') == 1
    assert bundle.body.count('#[verifier::external_body]') == 2
    assert bundle.body.count('row_execute(left), row_execute(right), free') == 2
    assert bundle.output_parameters == ('o',)
    assert {c.execute_name for c in bundle.contracts} == {'row_execute'}
    assert len({c.pre_name for c in bundle.contracts}) == 2
    assert len({c.free_type for c in bundle.contracts}) == 2
    assert bundle == render_verified_kernel_to_verus(tuple(reversed(contracts)), symbol_prefix='row')
    assert 'equivalence' not in bundle.body
    # The execution identity is independent of the selected goals and names.
    single = render_verified_kernel_to_verus(contracts[:1], symbol_prefix='another_name')
    assert single.execution_identity == bundle.execution_identity
    manifest = json.loads(render_contract_manifest(bundle))
    assert manifest['kind'] == 'shared_execution'
    assert manifest['execute_name'] == bundle.execute_name
    assert len(manifest['standalone_contracts']) == 2


def test_source_entrypoint_verifies_every_requested_goal():
    source, constants, contracts = row_contracts()
    assert transpile_verified_kernel_goals(source, 'silu_mul_kernel', constants,
        goal_names=('selected_row', 'batch_invariance'), symbol_prefix='row') == (
            render_verified_kernel_to_verus(contracts, symbol_prefix='row'))
    with pytest.raises(ValueError, match='No @verif proof goal named'):
        transpile_verified_kernel_goals(source, 'silu_mul_kernel', constants,
            goal_names=('batch_invariance', 'missing'))
    with pytest.raises(ValueError, match='duplicate requested goal'):
        transpile_verified_kernel_goals(source, 'silu_mul_kernel', constants,
            goal_names=('batch_invariance', 'batch_invariance'))


@cache
def partially_observed_kernel():
    source = (KERNELS / 'rmsnorm_residual.py').read_text().replace(
        '#     left(residual_out)[b:b+1, 0:N] == right(residual_out)[0:1, 0:N]\n', '')
    constants = {'BLOCK_M': 1, 'BLOCK_N': 1024}
    artifact = verify_annotations(source, 'rmsnorm_residual_kernel', constants).verified_contract
    assert artifact.written_tensor_parameters() == ('o', 'residual_out')
    return source, constants, artifact


@pytest.mark.parametrize('entrypoint', ['artifact', 'source', 'bundle'])
def test_every_entrypoint_preserves_unobserved_writes(entrypoint):
    source, constants, artifact = partially_observed_kernel()
    if entrypoint == 'artifact':
        rendered = render_verified_contract_to_verus(artifact, symbol_prefix='probe')
    elif entrypoint == 'source':
        rendered = transpile_verified_kernel_source(source, 'rmsnorm_residual_kernel', constants,
                                                     symbol_prefix='probe')
    else:
        rendered = render_verified_kernel_to_verus((artifact,), symbol_prefix='probe')
    assert 'residual_out: probe_residual_out_after(before)' in rendered.body
    assert 'residual_out: before.residual_out' not in rendered.body


@pytest.mark.parametrize('field,valid', [('x', True), ('residual_out', False)])
def test_single_goal_cannot_prove_unobserved_output_unchanged(tmp_path, field, valid):
    verus = os.environ.get('VERUS') or shutil.which('verus')
    if verus is None:
        pytest.skip('set VERUS to check frame-property falsifiers')
    artifact = partially_observed_kernel()[2]
    rendered = render_verified_contract_to_verus(artifact, symbol_prefix='probe')
    source = tmp_path / 'frame.rs'
    source.write_text(render_standalone_verus_module(rendered) + f'''
verus! {{
pub proof fn frame(before: ProbeSide)
    ensures probe_execute(before).{field} == before.{field},
{{}}
}}
''')
    result = subprocess.run([verus, '--crate-type=lib', str(source)], capture_output=True,
                            text=True, timeout=120)
    if valid:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0 and 'postcondition not satisfied' in result.stderr, result


@pytest.mark.parametrize('kind', ['logical_schema', 'type_collision'])
def test_bundle_rejects_incompatible_goal_interfaces(kind):
    source, constants, _ = row_contracts()
    second = 'selected_row'
    if kind == 'type_collision':
        second = 'batch__invariance'
        source = source.replace('selected_row', second)
        error = 'goal type names collide'
    else:
        at = source.index('# @verif(selected_row,')
        source = source[:at] + source[at:].replace('#   pre(', '#   pre(\n#     left(extra) >= 0,', 1)
        error = 'incompatible logical launch-state schemas'
    contracts = tuple(verify_annotations(source, 'silu_mul_kernel', constants,
        goal_name=goal).verified_contract for goal in ('batch_invariance', second))
    with pytest.raises(ValueError, match=error):
        render_verified_kernel_to_verus(contracts, symbol_prefix='row')


@pytest.mark.parametrize('kind', ['empty', 'duplicate', 'specialization', 'source'])
def test_bundle_rejects_ambiguous_or_mixed_executions(kind):
    source, constants, contracts = row_contracts()
    if kind == 'empty':
        selected, error = (), 'no proved goals'
    elif kind == 'duplicate':
        selected, error = (contracts[0], contracts[0]), 'duplicate goal'
    else:
        if kind == 'source':
            source += '\n# Different source identity.\n'
        else:
            constants = {**constants, 'BLOCK_N': 2048}
        other = verify_annotations(source, 'silu_mul_kernel', constants,
            goal_name='selected_row').verified_contract
        selected, error = (contracts[0], other), 'mixes source, specialization or typed execution'
    with pytest.raises(ValueError, match=error):
        render_verified_kernel_to_verus(selected, symbol_prefix='row')


@cache
def attention_contracts(swa):
    filename = 'fattn_paged_swa.py' if swa else 'fattn_paged.py'
    kernel = 'fattn_varlen_paged_swa_kernel' if swa else 'fattn_varlen_paged_fwd_block_ptr_kernel'
    source = (KERNELS / filename).read_text()
    constants = {'BLOCK_M': 16, 'BLOCK_N': 64, 'D_HEAD': 128, 'PAGE_BLOCK_SIZE': 64}
    return tuple(verify_annotations(source, kernel, constants, goal_name=goal,
        preserve_analyzer_conditions=True).verified_contract
        for goal in ('batch_invariance', 'selected_row_prefix_equivalence'))


@pytest.mark.parametrize('swa', [False, True])
def test_attention_goals_share_outputs_but_not_numeric_premises(swa):
    bundle = render_verified_kernel_to_verus(attention_contracts(swa), symbol_prefix='attention')
    batch, selected = bundle.contracts
    assert batch.output_parameters == ('o', 'lse')
    assert selected.output_parameters == ('o',)
    assert bundle.output_parameters == ('o', 'lse')
    assert batch.side_type == selected.side_type == 'AttentionSide'
    assert batch.execute_name == selected.execute_name == 'attention_execute'
    assert batch.output_functions[0] == selected.output_functions[0] == 'attention_o_after'
    assert bundle.body.count('pub uninterp spec fn attention_o_cell(') == 1
    assert bundle.body.count('pub uninterp spec fn attention_lse_cell(') == 1
    assert batch.analyzer_conditions == ()
    assert {c.label for c in selected.analyzer_conditions} == {'finite(v_block)@masked-backward-dependency'}
    for condition in selected.analyzer_conditions:
        assert condition.predicate_name not in batch.body
        assert f'{condition.predicate_name}(left, right, free)' in selected.body
    # A goal that only proves o must not acquire a postcondition about lse.
    post = selected.body.split(f'pub open spec fn {selected.post_name}(', 1)[1]
    post = post.split('// Trusted import', 1)[0]
    assert 'left.lse' not in post
    selected_only = render_verified_kernel_to_verus(attention_contracts(swa)[1:],
                                                    symbol_prefix='attention')
    assert selected_only.execution_identity == bundle.execution_identity
    assert selected_only.output_parameters == bundle.output_parameters
    assert 'lse: attention_lse_after(before)' in selected_only.body
    direct = render_verified_contract_to_verus(attention_contracts(swa)[1], symbol_prefix='attention')
    assert 'lse: attention_lse_after(before)' in direct.body
    assert 'lse: before.lse' not in direct.body


@pytest.mark.parametrize('kind', ['row', 'full', 'swa'])
def test_verus_shared_execution_callers(tmp_path, kind):
    """Optional compiler check; set VERUS or put verus on PATH to enable."""
    verus = os.environ.get('VERUS') or shutil.which('verus')
    if verus is None:
        pytest.skip('set VERUS to check generated callers with Verus')
    contracts = row_contracts()[2] if kind == 'row' else attention_contracts(kind == 'swa')
    bundle = render_verified_kernel_to_verus(contracts, symbol_prefix='shared')
    batch, selected = bundle.contracts
    module = render_standalone_verus_module(bundle)
    signature = (f'left: {bundle.side_type}, right: {bundle.side_type}, '
                 f'batch: {batch.free_type}, selected: {selected.free_type}')
    for negative in (False, True):
        # A batch certificate does not authorize a selected-row call without
        # its own domain and numeric premises, despite sharing execution.
        selected_pre = '' if negative else f'{selected.pre_name}(left, right, selected),'
        caller = f'''
verus! {{
pub proof fn caller({signature})
    requires {batch.pre_name}(left, right, batch), {selected_pre}
    ensures
        {batch.post_name}({bundle.execute_name}(left), {bundle.execute_name}(right), batch),
        {selected.post_name}({bundle.execute_name}(left), {bundle.execute_name}(right), selected),
{{
    {batch.certificate_name}(left, right, batch);
    {selected.certificate_name}(left, right, selected);
}}
}}
'''
        source = tmp_path / ('negative.rs' if negative else 'positive.rs')
        source.write_text(module + caller)
        result = subprocess.run([verus, str(source)], capture_output=True, text=True, timeout=120)
        if negative:
            assert result.returncode != 0 and 'precondition not satisfied' in result.stderr, result
        else:
            assert result.returncode == 0 and '0 errors' in result.stdout, result
