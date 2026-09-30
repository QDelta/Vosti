import json
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.effort.account import (
    COMPONENTS, PRESENTATION_COMPONENTS, inventory_matches, presentation_component,
    python_lines, ranges, render_table,
)
from scripts.effort.specification import TRACE_CLAIMS, accounting_claims, split_specifications
from scripts.effort.dependencies import load_dependencies, parse_declaration
from scripts.effort.purposes import declaration_purpose, purpose_partitions


class EffortTest(unittest.TestCase):
    def test_annotations_are_counted_but_comments_and_docstrings_are_not(self):
        source = '\n'.join(['"""Documentation."""', '# ordinary comment', '# @params(',
                            '# tensor(x, float, shape(M)),', '# )', '# @grid(M)',
                            '# @verif(batch, same(M))', 'def f():',
                            '    """Function documentation."""', '    return 1'])
        r = python_lines(source)
        self.assertEqual(r['implementation'], [8, 10])
        self.assertEqual(r['specification'], [3, 4, 5, 6, 7])
        self.assertEqual(r['annotation_goals'], 1)

    def test_multiline_code_string_is_source(self):
        r = python_lines('x = """generated code\nmore code\n"""\n')
        self.assertEqual(r['implementation'], [1, 2, 3])

    def test_documented_annotations_are_not_contracts(self):
        r = python_lines('"""Example\n# @verif(batch, same(M))\n"""\nx = 1\n')
        self.assertEqual(r['specification'], [])
        self.assertEqual(r['annotation_goals'], 0)

    def test_ranges(self):
        self.assertEqual(ranges([4, 2, 1, 8]), [[1, 2], [4, 4], [8, 8]])

    def test_freshness_allows_only_framework_revision_drift(self):
        saved = dict(framework_commit='old', kernel_sources_sha256='kernel',
                     files=[{'sha256': 'original'}], totals={'implementation': 2})
        current = {**saved, 'framework_commit': 'new'}
        self.assertTrue(inventory_matches(saved, current))
        for field, value in (
            ('kernel_sources_sha256', 'changed'), ('files', [{'sha256': 'changed'}]),
            ('totals', {'implementation': 3}), ('counting', 'changed policy'),
        ):
            with self.subTest(field=field):
                self.assertFalse(inventory_matches(saved, {**current, field: value}))

    def test_table_moves_auxiliary_to_proof_without_changing_syntax_counts(self):
        data = {'files': [
            dict(path='src/spec.rs', component=COMPONENTS[0],
                 counts=dict(implementation=100, specification=200, proof=300),
                 purpose_counts=dict(abstract_spec=10, concrete_instantiation=20,
                                     runtime_contracts=20, auxiliary_proof=150)),
        ]}
        table = render_table(data)
        self.assertIn('Component & Impl. & Abstract spec. & Concrete inst. & Runtime contracts & Proof', table)
        self.assertIn('Total & 100 & 10 & 20 & 20 & 450', table)
        self.assertEqual(data['files'][0]['counts']['specification'], 200)

    def test_purpose_separates_observation_binding_and_maintenance(self):
        def declaration(name, path, assumed=False):
            return dict(symbol='vosti_verus::'+name,
                        source_unit=dict(path=path, assumption=assumed))
        cases = [
            ('spec::view', 'src/spec.rs', False, 'abstract_spec'),
            ('proof::lemma_serving_deterministic', 'src/proof.rs', False, 'concrete_instantiation'),
            ('proof::serving::interpretation::request_input', 'src/proof/serving/interpretation.rs', False, 'concrete_instantiation'),
            ('proof::engine::refinement::trace_samples_for', 'src/proof/engine/refinement.rs', False, 'concrete_instantiation'),
            ('proof::engine::refinement::inv', 'src/proof/engine/refinement.rs', False, 'concrete_instantiation'),
            ('proof::scheduler::invariants::refcount_valid',
             'src/proof/scheduler/invariants.rs', False, 'auxiliary_proof'),
            ('proof::scheduler::invariants::token_placement_at',
             'src/proof/scheduler/invariants.rs', False, 'concrete_instantiation'),
            ('proof::tensor::shape::rectangular', 'src/proof/tensor/shape.rs', False, 'concrete_instantiation'),
            ('boundary::tensor_runtime::tensor_repr_2d',
             'src/boundary/tensor_runtime.rs', False, 'concrete_instantiation'),
            ('boundary::tensor_runtime::linear',
             'src/boundary/tensor_runtime.rs', True, 'runtime_contracts'),
            ('boundary::attention_operator::output',
             'src/boundary/attention_operator.rs', False, 'concrete_instantiation'),
            ('boundary::attention_operator::scale_log2',
             'src/boundary/attention_operator.rs', True, 'runtime_contracts'),
            ('boundary::linear_operator::cell',
             'src/boundary/linear_operator.rs', False, 'concrete_instantiation'),
            ('proof::tensor::attention_projection::logical_prefix',
             'src/proof/tensor/attention_projection.rs', False, 'concrete_instantiation'),
            ('proof::tensor::paged::rectangular_page_table',
             'src/proof/tensor/paged.rs', False, 'concrete_instantiation'),
            ('proof::tensor::layout::split_vector',
             'src/proof/tensor/layout.rs', False, 'concrete_instantiation'),
            ('external', 'src/boundary/model_deployment.rs', True, 'runtime_contracts'),
        ]
        for name, path, assumed, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(declaration_purpose(declaration(name, path, assumed))[0], expected)
        with self.assertRaisesRegex(ValueError, 'unreviewed'):
            declaration_purpose(declaration('new', 'src/new.rs'))
        with self.assertRaisesRegex(ValueError, 'unreviewed trusted'):
            declaration_purpose(declaration('axiom', 'src/proof/new.rs', True))

    def test_purpose_partition_is_disjoint_and_rejects_conflicting_line_owners(self):
        rust = {'src/proof/engine/refinement.rs': dict(specification=[1, 2, 3])}
        def node(name, line):
            return dict(symbol='vosti_verus::proof::engine::refinement::'+name,
                        source_unit=dict(path='src/proof/engine/refinement.rs', start=line,
                                         end=line, assumption=False))
        nodes = [node('trace_samples_for', 1), node('inv', 2)]
        with (patch('scripts.effort.purposes.ABSTRACT_DECLARATIONS', {'proof::engine::refinement::trace_samples_for'}),
              patch('scripts.effort.purposes.CONCRETE_DECLARATIONS', set())):
            parts, _ = purpose_partitions(rust, dict(selected_declarations=nodes))
            self.assertEqual(parts['src/proof/engine/refinement.rs'], dict(
                abstract_spec=[1], concrete_instantiation=[2], runtime_contracts=[], auxiliary_proof=[3]))
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                purpose_partitions(rust, dict(selected_declarations=[nodes[0], node('inv', 1)]))
            with self.assertRaisesRegex(ValueError, 'stale purpose overrides'):
                purpose_partitions(rust, dict(selected_declarations=[nodes[1]]))

    def test_whole_file_presentation_mapping(self):
        for path, component in (
            ('src/boundary/a.rs', 0), ('python/vosti_kernels/kernels.py', 0),
            ('python/vosti_kernels/physical.py', 0),
            ('python/vosti_kernels/graph_overlay.py', 0),
            ('python/vosti_kernels/deployment.py', 3),
            ('kernels/backend/probes.py', 3), ('scripts/deployment/common.py', 3),
            ('examples/support/openai_server.rs', 4),
        ):
            with self.subTest(path=path):
                self.assertEqual(presentation_component(
                    dict(path=path, component=COMPONENTS[3])), PRESENTATION_COMPONENTS[component])

    def fixture(self, entries):
        rust, declarations = {}, {}
        for symbol, path, line, mode, refs, assumed in entries:
            kind = 'proof_contract' if mode == 'Proof' else 'spec'
            result = rust.setdefault(path, dict(implementation=[], proof=[],
                                                specification=[], trusted=[], surface=[]))
            result['specification'].append(line)
            result['surface'].append(dict(name=symbol.split('::')[-1], start=line,
                                          end=line, kind=kind, assumption=assumed))
            identity = 'vosti_verus::' + symbol
            declarations[identity] = dict(symbol=identity, path=path, line=line,
                                          kind='Function', mode=mode,
                                          references=['vosti_verus::' + ref for ref in refs])
        return rust, declarations

    def test_only_trace_claims_seed_publication_accounting(self):
        entries = [(name, 'root.rs', i, 'Proof', [], False)
                   for i, name in enumerate(TRACE_CLAIMS, 1)]
        entries += [
            ('proof::serving::refinement::lemma_init_establishes_serving_inv',
             'root.rs', 3, 'Proof', ['init_only'], False),
            ('proof::serving::refinement::lemma_step_logits_match_reference',
             'root.rs', 4, 'Proof', [], False),
            ('init_only', 'root.rs', 5, 'Spec', [], False),
            ('trusted', 'root.rs', 6, 'Spec', ['trusted_dependency'], True),
            ('trusted_dependency', 'root.rs', 7, 'Spec', [], False),
            ('proof::serving::continuation_agreement::certified::lemma_observable_serving_traces_request_prefix_equal',
             'root.rs', 8, 'Proof', ['certified_only'], False),
            ('proof::serving::continuation_agreement::certified::lemma_dynamic_request_traces_prefix_equal',
             'root.rs', 9, 'Proof', ['certified_only'], False),
            ('certified_only', 'root.rs', 10, 'Spec', [], False),
        ]
        rust, graph = self.fixture(entries)
        surface = dict(top_level_claims=[
            dict(item=name, source_span=dict(path=path))
            for name, path, line, mode, _, _ in entries if mode == 'Proof' and line < 8],
            supporting_checked_entrypoints=[
                dict(item=name, source_span=dict(path=path))
                for name, path, line, mode, _, _ in entries if mode == 'Proof' and line >= 8])
        roots = accounting_claims(surface)
        self.assertEqual([r['item'] for r in roots], list(TRACE_CLAIMS))
        parts, _ = split_specifications(rust, roots, graph)
        self.assertEqual(parts['root.rs'], dict(
            theorem_surface=[1, 7], trusted_assumptions=[6], auxiliary=[3, 4, 5, 8, 9, 10]))
        for claims in (surface['top_level_claims'][1:], surface['top_level_claims'] + roots[:1]):
            with self.assertRaisesRegex(ValueError, 'uniquely'):
                accounting_claims(dict(top_level_claims=claims))

    def test_specification_closure_and_assumptions_partition(self):
        rust, graph = self.fixture([
            ('root', 'src/example.rs', 1, 'Proof', ['meaning'], False),
            ('meaning', 'src/example.rs', 2, 'Spec', [], False),
            ('helper', 'src/example.rs', 3, 'Proof', [], False),
            ('assumed', 'src/example.rs', 4, 'Spec', [], True),
        ])
        claim = dict(item='root', source_span=dict(path='src/example.rs'))
        parts, audit = split_specifications(rust, [claim], graph)
        self.assertEqual(parts['src/example.rs'], dict(
            theorem_surface=[1, 2], trusted_assumptions=[4], auxiliary=[3]))
        self.assertEqual(len(audit['selected_declarations']), 3)
        with self.assertRaisesRegex(ValueError, 'uniquely'):
            split_specifications(rust, [{**claim, 'item': 'missing'}], graph)

    def test_qualified_functions_and_receiver_methods_stay_distinct(self):
        rust, graph = self.fixture([
            ('root', 'root.rs', 1, 'Proof', ['a::meaning', 'a::impl&%0::id'], False),
            ('a::meaning', 'a.rs', 1, 'Spec', [], False),
            ('b::meaning', 'b.rs', 1, 'Spec', [], False),
            ('a::impl&%0::id', 'a.rs', 2, 'Spec', [], False),
            ('a::impl&%1::id', 'a.rs', 3, 'Spec', [], False),
        ])
        parts, audit = split_specifications(
            rust, [dict(item='root', source_span=dict(path='root.rs'))], graph)
        self.assertEqual(parts['a.rs']['theorem_surface'], [1, 2])
        self.assertEqual(parts['a.rs']['auxiliary'], [3])
        self.assertEqual(parts['b.rs']['auxiliary'], [1])
        self.assertEqual(audit['ambiguous_names'], {})

    def test_missing_compiler_edges_or_counted_declarations_fail_closed(self):
        rust, graph = self.fixture([('root', 'root.rs', 1, 'Proof', ['missing'], False)])
        claims = [dict(item='root', source_span=dict(path='root.rs'))]
        with self.assertRaisesRegex(ValueError, 'unresolved compiler dependency'):
            split_specifications(rust, claims, graph)
        with self.assertRaisesRegex(ValueError, 'no compiler identity'):
            split_specifications(rust, claims, {})

    def test_inline_assume_cannot_silently_reintroduce_name_matching(self):
        rust, graph = self.fixture([('root', 'root.rs', 1, 'Proof', [], False)])
        rust['root.rs']['trusted'] = [dict(kind='assume', line=1, end=1)]
        with self.assertRaisesRegex(ValueError, 'inline assume needs typed'):
            split_specifications(rust, [], graph)

    def test_compiler_parser_excludes_proof_and_exec_bodies_and_strings(self):
        # VIR has already resolved an import alias or a receiver method to its
        # declaration identity. Source spelling is irrelevant to this parser.
        text = """(@ "src/root.rs:1:1: 9:2 (#0)" (Function
          :name (Fun :path vosti_verus::root)
          :mode MODE
          :require ((Call (Fun :path vosti_verus::a::meaning)))
          :ensure ("(Fun :path vosti_verus::not_a_reference)")
          :ret (Typ Datatype (Dt Path vosti_verus::State) () ())
          :body (Call (Fun :path vosti_verus::a::impl&%0::id))
        ))"""
        for mode in ('Proof', 'Exec', 'Spec'):
            node = parse_declaration(text.replace('MODE', mode), 'src/root.rs', 1, 'Function')
            expected = ['vosti_verus::State', 'vosti_verus::a::meaning']
            if mode == 'Spec':
                expected.append('vosti_verus::a::impl&%0::id')
            self.assertEqual(node['references'], sorted(expected))
        with self.assertRaisesRegex(ValueError, 'unrecognized project reference'):
            parse_declaration(text.replace('MODE', 'Spec').replace(
                '(Fun :path vosti_verus::a::meaning)',
                '(NewEncoding vosti_verus::a::meaning)'), 'src/root.rs', 1, 'Function')

    def test_dependency_freshness_checks_source_hashes(self):
        artifact = dict(schema='vosti.effort-dependencies.v1',
                        source_sha256={'src/a.rs': 'old'})
        with (patch('scripts.effort.dependencies.Path.read_text', return_value=json.dumps(artifact)),
              patch('scripts.effort.dependencies.source_hashes', return_value={'src/a.rs': 'new'})):
            with self.assertRaisesRegex(ValueError, 'stale'):
                load_dependencies(Path('/unused'))


if __name__ == '__main__':
    unittest.main()
