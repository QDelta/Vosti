"""Re-read report evidence and emit an explicitly partial or complete table.

This independently checks selected-row bytes, identities, coverage and recorded
telemetry hashes. Scheduling/cache witnesses remain the runner's witnesses;
this numerical/artifact audit is not an independent proof of their semantics.
It never writes to the campaign or changes worker state.
"""
import argparse
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from scripts.determinism_tests.call_inputs import resolve_call
from scripts.determinism_tests.protocol import sha256_json
from scripts.determinism_tests.rank_divergence import ranking_divergence
from scripts.determinism_tests.report_plan import make_plan, REPORT_MODELS, SEED, table_matrix
from scripts.determinism_tests.report_suite import prefill_witness
from scripts.determinism_tests.vllm_observation import public_schedule_events


def backend_evidence(result, directory):
    """Bind auto-selected FlashAttention's actual version to its retained log."""
    evidence = dict(result['backend_evidence'])
    backend = evidence.get('selected_attention_backend') or evidence.get('attention_config_backend')
    if result['engine'] != 'vllm' or backend not in ('FLASH_ATTN', 'AttentionBackendEnum.FLASH_ATTN'):
        return evidence
    path = directory/'stdout.log'
    data = path.read_bytes()
    versions = {int(value) for value in re.findall(
        rb'\[flash_attn\.py:\d+\] Using FlashAttention version ([234])\b', data)}
    if len(versions) != 1:
        raise ValueError(f'missing or ambiguous FlashAttention version log: {path}')
    version = versions.pop()
    configured = evidence.get('attention_config_flash_attn_version')
    if configured is not None and configured != version:
        raise ValueError(f'configured and observed FlashAttention versions differ: {path}')
    evidence.update(selected_flash_attn_version=version,
        version_evidence=dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                              source='FlashAttentionImpl initialization log'))
    return evidence


def request_id_source_evidence(stack_root):
    worker = Path(stack_root)/'vllm/.venv/bin/python'
    code = ('import importlib.metadata as m,json; d=m.distribution("vllm"); '
            'print(json.dumps(dict(version=d.version, paths=[str(d.locate_file(p)) for p in '
            '["vllm/v1/engine/input_processor.py","vllm/utils/__init__.py"]])))')
    installation = json.loads(subprocess.check_output([str(worker),'-c',code], text=True))
    if installation['version'] != '0.28.0':
        raise ValueError('ID-namespace correction requires the inspected vLLM 0.28.0 source')
    paths = [Path(path) for path in installation['paths']]
    tree = ast.parse(paths[0].read_text())
    methods = [node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='assign_request_id']
    expected = ast.dump(ast.parse('request.request_id = f"{request.external_req_id}-{random_uuid():.8}"').body[0])
    if len(methods)!=1 or expected not in {ast.dump(node) for node in ast.walk(methods[0])}:
        raise ValueError('installed vLLM request-ID assignment rule differs from inspected source')
    return dict(version=installation['version'], rule='external_id + hyphen + 8 hex UUID characters',
                sources=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths])


def expected_labels(plan):
    return dict(
        batch={f'composition-{i}/prompt-{p}' for i, group in enumerate(plan['batch_groups']) for p in group}
              | {f'fresh-vs-{kind}-{p}' for p in plan['fresh_spots'] for kind in ('isolated','batch')},
        chunk={f'{len(prompt)}/{budget}' for prompt in plan['chunk_prompts'] for budget in plan['chunk_budgets']},
        pd={f'{i}/{j}' for i in range(len(plan['pd_prompts'])) for j in range(plan['output_tokens'])},
        cache={case['label'] for case in plan['cache_cases']} | {'generated-prefix'})


@lru_cache(maxsize=8)
def array(path):
    value = np.load(path, allow_pickle=False)
    if value.ndim != 1 or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f'invalid retained full-vocabulary row: {path}')
    return value


def check_comparison(row):
    left, right = array(row['left']), array(row['right'])
    equal = left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()
    checks = dict(bitwise_equal=equal, shape_equal=left.shape==right.shape,
                  dtype_equal=left.dtype==right.dtype, finite_left=True, finite_right=True)
    if left.shape == right.shape:
        differing = np.flatnonzero(left.view(np.uint32) != right.view(np.uint32))
        checks.update(mismatch_count=int(differing.size),
            first_mismatch_index=int(differing[0]) if differing.size else None,
            argmax_left=int(left.argmax()), argmax_right=int(right.argmax()),
            argmax_equal=int(left.argmax())==int(right.argmax()))
    for key, value in checks.items():
        if row.get(key) != value:
            raise ValueError(f'recomputed {key} differs in {row["label"]}')
    if row.get('ranking') != ranking_divergence(left, right):
        raise ValueError(f'recomputed ranking differs in {row["label"]}')
    return equal


def audit_row(path, plan, gpu_index):
    data = path.read_bytes()  # Atomic progress snapshots may advance after this read.
    report = json.loads(data)
    if report['checkpoint'] != plan['checkpoint'] or report['seed'] != plan['seed']:
        raise ValueError('report checkpoint/seed differs from campaign')
    descriptors, receipts, backends = {}, [], []
    rebuilt_chunks, corrections = {}, []
    for record in report['arms']:
        result_path = Path(record['result'])
        directory = result_path.parent
        arm = json.loads((directory/'arm.json').read_text())
        result_data = result_path.read_bytes()
        result = json.loads(result_data)
        if arm['model_path'] != plan['model_path'] or arm['execution'] != report['execution']:
            raise ValueError('arm model or execution mode differs from report')
        if sha256_json(arm) != record['arm_sha256'] or result['arm_sha256'] != record['arm_sha256']:
            raise ValueError(f'arm/result identity changed: {directory}')
        telemetry_data = Path(record['telemetry']['path']).read_bytes()
        telemetry = json.loads(telemetry_data)
        if (hashlib.sha256(telemetry_data).hexdigest() != record['telemetry']['sha256']
                or telemetry['status'] != 'complete' or str(telemetry['gpu']['index']) != str(gpu_index)):
            raise ValueError(f'invalid or changed selected-GPU telemetry: {directory}')
        if len(arm['calls']) != len(result['calls']):
            raise ValueError('result call count differs from declared arm')
        if result['engine'] == 'vllm':
            for call in result['calls']:
                events = public_schedule_events(call, result['vllm_version'])
                if record['name'].startswith('chunk-'):
                    if len(call['requests']) != 1:
                        raise ValueError('chunk witness expected one request')
                    request = call['requests'][0]
                    length = len(request['input_token_ids'])
                    budget = arm['engine']['max_num_batched_tokens']
                    label = f'{length}/{budget}'
                    if label in rebuilt_chunks:
                        raise ValueError('duplicate chunk witness source')
                    rebuilt_chunks[label] = dict(prefill_witness(
                        [row for event in events for row in event if row['request_id']==request['request_id']],
                        length,budget), label=label)
        for ci, (declared, call) in enumerate(zip(arm['calls'], result['calls'], strict=True)):
            resolved = resolve_call(declared, result['calls'][:ci])
            if len(resolved['prompts']) != len(call['requests']):
                raise ValueError('request count differs from declared call')
            for prompt, request in zip(resolved['prompts'], call['requests'], strict=True):
                if (request['input_token_ids'] != prompt or len(request['output_token_ids']) != resolved['max_tokens']
                        or request['prompt_sha256'] != sha256_json(prompt)):
                    raise ValueError('request input/output geometry differs from declared call')
                if not arm['engine']['prefix_caching'] and request['num_cached_tokens'] != 0:
                    raise ValueError('declared cold request reused cached tokens')
                rows = list(enumerate(request.get('output_rows', [])))
                if request.get('last_row') is not None:
                    position = resolved.get('record_generated_position')
                    rows.append((resolved['max_tokens']-1 if position is None else position, request['last_row']))
                if resolved.get('record_all_rows') and len(request.get('output_rows', [])) != resolved['max_tokens']:
                    raise ValueError('request omitted required prediction rows')
                for position, row in rows:
                    filename = str(directory/'rows'/row['artifact'])
                    if filename in descriptors and descriptors[filename] != row:
                        raise ValueError('conflicting descriptors for selected raw row')
                    descriptors[filename] = row
                    value = array(filename)
                    if (hashlib.sha256(value.tobytes()).hexdigest() != row['sha256']
                            or list(value.shape) != row['shape'] or str(value.dtype) != row['dtype']
                            or not row['finite'] or int(value.argmax()) != row['argmax']):
                        raise ValueError(f'selected raw row differs from descriptor: {filename}')
                    metadata = row['metadata']
                    if metadata.get('request_id', request['request_id']) != request['request_id']:
                        raise ValueError('selected raw row belongs to a different request')
                    observed_position = (metadata['generated_position'] if 'generated_position' in metadata
                                         else metadata['token_position']-len(prompt)+1)
                    if observed_position != position:
                        raise ValueError('row position differs from its declared prediction index')
                    if request['output_token_ids'][position] != row['argmax']:
                        raise ValueError('selected raw row disagrees with actual emitted token')
        backends.append(backend_evidence(result, directory))
        receipts.append(dict(result=str(result_path), sha256=hashlib.sha256(result_data).hexdigest(),
                             telemetry_sha256=record['telemetry']['sha256']))
    cells = {}
    for relation, labels in expected_labels(plan).items():
        rows = report['comparisons'][relation]
        actual = [row['label'] for row in rows]
        if len(set(actual)) != len(actual) or not set(actual) <= labels or report['expected'][relation] != len(labels):
            raise ValueError(f'comparison coverage/uniqueness differs from plan: {relation}')
        for row in rows:
            if row['left'] not in descriptors or row['right'] not in descriptors:
                raise ValueError('comparison does not refer to a declared selected row')
        equal = sum(check_comparison(row) for row in rows)
        witnesses = report['witnesses'][relation]
        valid = all(value['valid'] for value in witnesses)
        status = ('incomplete' if len(actual) != len(labels) else 'invalid' if not valid else
                  'pass' if equal == len(labels) else 'mismatch')
        cells[relation] = dict(status=status, expected=len(labels), compared=len(actual), equal=equal,
                               mismatches=len(actual)-equal, top_token_differences=sum(not row['argmax_equal'] for row in rows))
        if cells[relation] != report['cells'][relation]:
            raise ValueError(f'cell summary disagrees with re-read evidence: {relation}')
        if relation == 'chunk' and rebuilt_chunks:
            corrected = []
            for witness in witnesses:
                fresh = rebuilt_chunks.get(witness['label'], witness)
                if fresh != witness:
                    # The only admissible offline repair is the empty witness
                    # produced by matching public IDs to randomized internal IDs.
                    if (witness.get('valid') is not False or witness.get('pieces') != []
                            or witness.get('consumed') != 0 or witness.get('expected') != fresh['expected']):
                        raise ValueError('chunk witness changed for a reason other than the known ID namespace bug')
                    corrections.append(dict(reason='vllm_0_28_internal_to_public_request_id', original=witness, corrected=fresh))
                corrected.append(fresh)
            if len(rows) == len(labels):
                cells[relation]['status'] = ('invalid' if not all(item['valid'] for item in corrected) else
                                              'pass' if equal==len(labels) else 'mismatch')
    return dict(checkpoint=plan['checkpoint'], execution=report['execution'], cells=cells,
        snapshot=str(path), snapshot_sha256=hashlib.sha256(data).hexdigest(),
        selected_rows=len(descriptors), arms=receipts, backend_evidence=backends,
        analysis_corrections=corrections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True, action='append')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    analysis_source = dict(
        root=str(root), commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root)),
        files={name:hashlib.sha256((root/'scripts/determinism_tests'/name).read_bytes()).hexdigest()
               for name in ('report_audit.py','report_suite.py','report_plan.py','protocol.py',
                            'vllm_observation.py','call_inputs.py','rank_divergence.py')})
    rows, manifests, id_sources = {}, [], []
    for campaign in args.campaign:
        manifest_data = (campaign/'inputs.json').read_bytes()
        manifest = json.loads(manifest_data)
        if (type(manifest['gpu']) is not int or manifest['gpu'] < 0
                or manifest['seed'] != SEED or manifest['matrix'] != table_matrix()
                or manifest['plans'] != {key:make_plan(key) for key in REPORT_MODELS}):
            raise ValueError('campaign differs from the agreed two-model, one-seed full report plan')
        manifests.append(dict(path=str(campaign/'inputs.json'), sha256=hashlib.sha256(manifest_data).hexdigest()))
        for entry in manifest['matrix']:
            key = f'{entry["checkpoint"]}/{entry["execution"]}'
            rows.setdefault(key, dict(status='unrun'))
            directory = campaign/'suites'/entry['checkpoint']/entry['execution']
            path = directory/'summary.json'
            if not path.exists():
                path = directory/'progress.json'
            if not path.exists():
                continue
            if rows[key].get('status') != 'unrun':
                raise ValueError(f'duplicate executed row; select one campaign: {key}')
            checked = audit_row(path, manifest['plans'][entry['checkpoint']], manifest['gpu'])
            if checked['analysis_corrections'] and not id_sources:
                id_sources.append(request_id_source_evidence(manifest['stack_root']))
            rows[key] = dict(checked, status='complete' if all(c['status'] in ('pass','mismatch')
                                                             for c in checked['cells'].values()) else 'partial')
            print(key, {key: value['status'] for key,value in checked['cells'].items()}, flush=True)
    output = dict(status='complete' if len(rows)==14 and all(row['status']=='complete' for row in rows.values()) else 'partial',
                  manifests=manifests, rows=rows, analysis_source=analysis_source, request_id_source_evidence=id_sources,
                  audit_scope='selected raw bytes, identities, coverage, telemetry hashes; retained witnesses except explicitly rebuilt vLLM ID-namespace chunk witnesses')
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
