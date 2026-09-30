"""Reproducible implementation/specification/proof source-effort inventory."""

import argparse
import ast
import hashlib
import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.effort.specification import PARTS, accounting_claims, split_specifications
from scripts.effort.dependencies import load_dependencies
from scripts.effort.purposes import PURPOSES, purpose_partitions
from scripts.audit.check_kernel_sources import resolve_kernel_sources
from vosti_kernels.model_profile import load_scope

CATEGORIES = ('implementation', 'specification', 'proof')
COMPONENTS = ('Engine and model execution', 'Triton kernels', 'Kernel verifier',
              'Bindings and proof integration')
PRESENTATION_COMPONENTS = (
    'Engine and model execution', 'Triton kernels', 'Kernel verifier',
    'Verification integration and deployment', 'Serving interface and support',
)


def command(args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def tracked(root):
    return set(command(['git', 'ls-files'], root).splitlines())


def ranges(lines):
    result = []
    for n in sorted(lines):
        if result and result[-1][1] == n - 1:
            result[-1][1] = n
        else:
            result.append([n, n])
    return result


def python_lines(source):
    tree = ast.parse(source)
    lines = source.splitlines()
    excluded = set()
    for node in ast.walk(tree):
        # Documentation strings (including standalone documentation literals).
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            excluded.update(range(node.lineno, node.end_lineno + 1))
    code, comments = set(), {}
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
               tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comments[token.start[0]] = token.string
        if token.type not in ignored:
            code.update(range(token.start[0], token.end[0] + 1))
    annotations, depth, goals = set(), 0, 0
    for n, line in enumerate(lines, 1):
        if n not in comments or n in excluded:
            if depth:
                raise ValueError(f'unterminated kernel annotation at line {n}')
            continue
        match = re.match(r'\s*#\s*@(params|grid|verif)\(', line)
        if depth or match:
            if not line.lstrip().startswith('#'):
                raise ValueError(f'unterminated kernel annotation at line {n}')
            text = line.lstrip()[1:].strip()
            if match and match[1] == 'verif':
                goals += 1
            if text:
                annotations.add(n)
            depth += text.count('(') - text.count(')')
            if depth < 0:
                raise ValueError('unbalanced annotation')
    if depth:
        raise ValueError('unterminated annotation')
    code = {n for n in code - excluded if n <= len(lines) and lines[n-1].strip()}
    assert not (code & annotations)
    return dict(implementation=sorted(code), specification=sorted(annotations), proof=[],
                annotation_goals=goals,
                jit_kernels=sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                                any(ast.unparse(d) == 'triton.jit' for d in n.decorator_list)
                                for n in ast.walk(tree)))


def script_dependencies(root, files, roots):
    """Resolve tracked script imports without importing proof/GPU modules."""
    pending, seen = list(roots), set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if path not in files:
            raise ValueError(f'untracked script root: {path}')
        seen.add(path)
        tree = ast.parse((root / path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                modules = [base, *(f'{base}.{alias.name}' if base else alias.name
                                   for alias in node.names)]
            else:
                continue
            for module in modules:
                name = module.removeprefix('scripts.').replace('.', '/')
                candidate = f'scripts/{name}.py'
                if candidate in files:
                    pending.append(candidate)
    return seen


def source_inventory(root):
    files = tracked(root)
    kernel_root = root / 'kernels'
    kernel_files = tracked(kernel_root)
    assigned = {}
    def add(path, component):
        if path in assigned and assigned[path] != component:
            raise ValueError(f'duplicate component assignment: {path}')
        if path not in files and not (path.startswith('kernels/') and path.removeprefix('kernels/') in kernel_files):
            raise ValueError(f'untracked scope file: {path}')
        assigned[path] = component
    for path in sorted(files):
        if path.startswith('src/') and path.endswith('.rs'):
            add(path, COMPONENTS[3] if path.startswith('src/boundary/') else COMPONENTS[0])
        if path.startswith('python/vosti_kernels/') and path.endswith('.py'):
            add(path, COMPONENTS[3])
        if path.startswith('examples/verus_server_') and path.endswith('.rs'):
            add(path, COMPONENTS[3])
    for name in ('engine_setup.rs', 'engine_serving.rs', 'engine_calls.rs', 'openai_server.rs', 'output_tokens.rs'):
        add('examples/support/' + name, COMPONENTS[3])
    for path in sorted(kernel_files):
        if path.startswith(('ir/', 'diagnostics/')) and path.endswith('.py') and not Path(path).name.startswith('test_'):
            add('kernels/' + path, COMPONENTS[2])
        if path.startswith('backend/') and path.endswith('.py'):
            add('kernels/' + path, COMPONENTS[3])
        if path.startswith('scripts/') and path.endswith('.py'):
            add('kernels/' + path, COMPONENTS[2])
    scope_files = sorted(p for p in files if re.fullmatch(r'python/vosti_kernels/model_families/[^/]+/scope.json', p))
    for path in scope_files:
        scope = load_scope(root / path)
        for entry in scope['kernel_contracts'] + scope.get('runtime_support_sources', []):
            add('kernels/triton_kernels/' + entry['source'], COMPONENTS[1])
    # Proof/certificate/deployment drivers and their static in-repo imports.
    pending = ['scripts/verification/verify_kernel_contracts.py', 'scripts/prepare_deployment.py',
               'scripts/audit/runtime_bridge_scope.py']
    # The shared preparation entry point loads family builders dynamically.
    pending += [f'scripts/deployment/model_families/{Path(p).parent.name}.py' for p in scope_files]
    # These policy modules are loaded through family discovery/importlib.
    pending += [p for p in files if re.fullmatch(r'scripts/deployment/(?:model_families/\w+_scope|dense_kernel_scope)\.py', p)]
    for path in script_dependencies(root, files, pending):
        add(path, COMPONENTS[3])
    return assigned, scope_files


def collect(root):
    assigned, scopes = source_inventory(root)
    revision = command(['git', 'rev-parse', 'HEAD'], root)
    resolve_kernel_sources(root)
    measured_paths = set(assigned) | set(scopes) | {
        'audit/claim_surface.json', 'python/vosti_kernels/kernel_catalog.json'}
    before = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in measured_paths}
    generated = []
    for path in list(assigned):
        text = (root / path).read_text()
        if re.search(r'(?im)^\s*(?://|#).*(@generated|DO NOT EDIT)', '\n'.join(text.splitlines()[:5])):
            generated.append(path)
            del assigned[path]
    rust_paths = [p for p in assigned if p.endswith('.rs')]
    helper = ROOT / 'target/effort-accounting/debug/paper-effort-rust'
    if not helper.is_file():
        raise RuntimeError('missing Rust classifier; run cargo build --offline --locked --manifest-path scripts/effort/rust/Cargo.toml --target-dir target/effort-accounting first')
    rust = json.loads(command([str(helper), *rust_paths], root))
    claim_surface = json.loads((root / 'audit/claim_surface.json').read_text())
    dependencies = load_dependencies(root)
    split, split_audit = split_specifications(
        rust, accounting_claims(claim_surface), dependencies['declarations'])
    purposes, purpose_audit = purpose_partitions(rust, split_audit)
    rows, totals = [], {c: dict.fromkeys(CATEGORIES, 0) for c in COMPONENTS}
    for path, component in sorted(assigned.items()):
        raw = (root / path).read_bytes()
        result = rust[path] if path.endswith('.rs') else python_lines(raw.decode())
        counts = {c: len(result[c]) for c in CATEGORIES}
        for c in CATEGORIES:
            totals[component][c] += counts[c]
        parts = split[path] if path in split else dict(
            theorem_surface=result['specification'], trusted_assumptions=[], auxiliary=[])
        if path in purposes:
            purpose_lines = purposes[path]
            assert set(purpose_lines['runtime_contracts']) == set(parts['trusted_assumptions']), path
        else:
            if result['specification'] and component != 'Triton kernels':
                raise ValueError(f'unreviewed non-Rust specification purpose: {path}')
            purpose_lines = {purpose: [] for purpose in PURPOSES}
            purpose_lines['runtime_contracts'] = result['specification']
        rows.append(dict(path=path, component=component, sha256=hashlib.sha256(raw).hexdigest(),
                         counts=counts, lines={c: ranges(result[c]) for c in CATEGORIES},
                         specification_parts={c: len(parts[c]) for c in PARTS},
                         specification_part_lines={c: ranges(parts[c]) for c in PARTS},
                         purpose_counts={c: len(purpose_lines[c]) for c in PURPOSES},
                         purpose_lines={c: ranges(purpose_lines[c]) for c in PURPOSES},
                         trusted=result.get('trusted', []), annotation_goals=result.get('annotation_goals', 0),
                         jit_kernels=result.get('jit_kernels', 0)))
    assert sum(sum(r['counts'].values()) for r in rows) == sum(sum(t.values()) for t in totals.values())
    assert revision == command(['git', 'rev-parse', 'HEAD'], root), 'revision changed during accounting'
    if before != {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in measured_paths}:
        raise ValueError('sources changed during accounting')
    return dict(schema='paper.effort.v3',
                framework_commit=revision,
                kernel_sources_sha256=resolve_kernel_sources(root)[1],
                counting='Nonblank physical source lines; original syntax-based partitions retained. Documentation excluded; structured kernel annotation comments included. No macro expansion in line classification. Reviewed purposes subdivide the compiler-resolved trace/trusted closure; auxiliary specification is presented as proof effort.',
                scope='Tracked engine/model/proof source, serving bindings, scoped Triton modules and wrappers, verifier infrastructure, and certificate/deployment integration. Shared sources counted once across all repository model families.',
                exclusions='Tests, benchmarks, ordinary examples, build/package metadata, generated certificates, third-party dependencies, and unrelated development scripts.',
                generated_excluded=sorted(generated), scope_manifests=scopes, totals=totals, files=rows,
                specification_audit=split_audit,
                purpose_audit=purpose_audit,
                annotation_goals=sum(r['annotation_goals'] for r in rows),
                jit_kernels=sum(r['jit_kernels'] for r in rows))


def presentation_component(row):
    path = row['path']
    if path.startswith('src/'):
        return PRESENTATION_COMPONENTS[0]
    if row['component'] in COMPONENTS[1:3]:
        return row['component']
    if path.startswith('examples/') or path.endswith('/serving_workload.py'):
        return PRESENTATION_COMPONENTS[4]
    if path.startswith(('scripts/', 'kernels/backend/')):
        return PRESENTATION_COMPONENTS[3]
    if path.startswith('python/vosti_kernels/'):
        if Path(path).name in {
            'deployment.py', 'backend_evidence.py', 'dense_launch_plan.py',
            'model_profile.py', 'profile.py', 'kernel_selection.py', 'kernel_modules.py',
        }:
            return PRESENTATION_COMPONENTS[3]
        return PRESENTATION_COMPONENTS[0]
    raise ValueError(f'unassigned presentation component: {path}')


def presentation_totals(data):
    totals = {name: dict.fromkeys(('implementation', *PURPOSES[:3], 'proof'), 0)
              for name in PRESENTATION_COMPONENTS}
    for row in data['files']:
        group = totals[presentation_component(row)]
        for category in ('implementation', 'proof'):
            group[category] += row['counts'][category]
        for part in PURPOSES[:3]:
            group[part] += row['purpose_counts'][part]
        group['proof'] += row['purpose_counts']['auxiliary_proof']
    assert sum(sum(group.values()) for group in totals.values()) == sum(
        sum(row['counts'].values()) for row in data['files'])
    return totals


def render_table(data):
    lines = [r'\begin{table}[t]', r'\centering', r'\footnotesize',
             r'\setlength{\tabcolsep}{3.5pt}',
             r'\caption{Source-size estimates (lines). Specification is divided by purpose; proof effort includes auxiliary specification. Runtime contracts count imported runtime declarations and kernel annotations; supporting definitions belong to concrete instantiation.}',
             r'\label{tab:effort}', r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Component & Impl. & Abstract spec. & Concrete inst. & Runtime contracts & Proof \\', r'\midrule']
    groups = presentation_totals(data)
    totals = {c: sum(group[c] for group in groups.values()) for c in next(iter(groups.values()))}
    for component, counts in [*groups.items(), ('Total', totals)]:
        if component == 'Total':
            lines.append(r'\midrule')
        values = (counts['implementation'],
                  counts['abstract_spec'], counts['concrete_instantiation'],
                  counts['runtime_contracts'], counts['proof'])
        lines.append(component + ' & ' + ' & '.join(f'{value:,}' for value in values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def inventory_matches(saved, current):
    # Framework revision is provenance. Compare every counted source hash,
    # line range, category, scope and kernel source digest.
    return ({k: v for k, v in saved.items() if k != 'framework_commit'}
            == {k: v for k, v in current.items() if k != 'framework_commit'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = collect(args.root.resolve())
    output = args.root.resolve() / 'target/effort'
    outputs = {output / 'effort.json': json.dumps(data, indent=2) + '\n',
               output / 'effort.tex': render_table(data)}
    for path, content in outputs.items():
        if args.check:
            matches = path.is_file() and (
                inventory_matches(json.loads(path.read_text()), data)
                if path.suffix == '.json' else path.read_text() == content
            )
            if not matches:
                raise SystemExit(f'stale accounting: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(json.dumps({k: data[k] for k in ('framework_commit', 'kernel_sources_sha256', 'totals',
                                          'generated_excluded', 'jit_kernels', 'annotation_goals')}, indent=2))


if __name__ == '__main__':
    main()
