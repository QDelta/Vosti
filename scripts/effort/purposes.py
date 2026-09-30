"""Reviewed purpose classification of the serving-spec/trusted declaration closure.

Only the engine-independent specification module has abstract ownership.
Proof-body-only specifications remain auxiliary. Each physical line has one
owner; dependencies are still traversed across purpose boundaries.
"""

PURPOSES = ('abstract_spec', 'concrete_instantiation', 'runtime_contracts', 'auxiliary_proof')

# File-level responsibility defaults, with declaration-level exceptions below.
# New contributing files fail closed rather than receiving a guessed purpose.
AUXILIARY_FILES = {
    'src/exec/cache_scheduler/availability_queue.rs',
    'src/exec/cache_scheduler/mod.rs',
    'src/proof/scheduler/invariants.rs',
    'src/proof/scheduler/transition_invariants.rs',
    'src/proof/scheduler/executed_publication.rs',
}
ABSTRACT_FILES = {'src/spec.rs'}
CONCRETE_FILES = {
    'src/proof/serving/interpretation.rs',
    'src/proof/serving/transitions.rs',
    'src/proof.rs',
    'src/proof/serving/refinement.rs',
    # Defined runtime-interface vocabulary is concrete instantiation; only
    # assumed declarations in these modules remain runtime contracts.
    'src/boundary/tensor_runtime.rs',
    # Raw kernel-to-model mapping and logical page/row representation. These
    # definitions are concrete bindings, not additional trusted kernel laws.
    'src/boundary/attention_operator.rs',
    'src/boundary/linear_operator.rs',
    'src/boundary/qkv_operator.rs',
    'src/boundary/normalization_operator.rs',
    'src/boundary/pointwise_operator.rs',
    'src/boundary/embedding_operator.rs',
    'src/boundary/head_normalization_operator.rs',
    'src/boundary/rotary_operator.rs',
    'src/boundary/kv_store_operator.rs',
    'src/proof/tensor/attention_projection.rs',
    'src/proof/tensor/paged.rs',
    'src/proof/tensor/layout.rs',
    'src/boundary/dense_layer_primitives.rs',
    'src/boundary/four_norm_gated_primitives.rs',
    'src/boundary/model_forward_graph.rs',
    'src/boundary/scalar.rs',
    'src/boundary/backend_certificates/support.rs',
    'src/proof/tensor/shape.rs',
    'src/boundary/dense_swiglu_decoder.rs',
    'src/boundary/four_norm_gated_weights.rs',
    'src/boundary/model_deployment.rs',
    'src/exec/engine.rs',
    'src/exec/model_families/mod.rs',
    'src/proof/tensor/geometry.rs',
    'src/proof/reference/independent_batch_model.rs',
    'src/proof/reference/request_machine.rs',
    'src/exec/request_state.rs',
    'src/proof/model/types.rs',
    'src/types.rs',
    'src/model_config.rs',
    'src/proof/tensor/types.rs',
    'src/proof/cache/provenance.rs',
    'src/proof/model/dense_swiglu/cache_semantics.rs',
    'src/proof/model/dense_swiglu/semantics.rs',
    'src/proof/model/four_norm_gated/capstones.rs',
    'src/proof/model/four_norm_gated/layers.rs',
    'src/proof/model/four_norm_gated/model.rs',
    'src/proof/model/architecture.rs',
    'src/proof/model/cache.rs',
    'src/proof/model/families/dense_swiglu/mod.rs',
    'src/proof/model/graph_cover.rs',
    'src/proof/engine/refinement.rs',
} | {f'src/boundary/model_families/{family}/{module}.rs'
     for family in ('qwen3', 'llama3', 'gemma3', 'gemma4')
     for module in ('mod', 'weights', 'deployment', 'config')}

# Declaration-level exceptions remain explicit and checked for stale entries.
ABSTRACT_DECLARATIONS = set()
CONCRETE_DECLARATIONS = {
    # State correspondence and admission semantics, not queue-maintenance facts.
    'proof::scheduler::invariants::scheduler_admission_relation',
    'proof::scheduler::invariants::token_placement_at',
    'proof::scheduler::invariants::token_placement_prefix',
    'proof::scheduler::transition_invariants::residency_history_aligned',
    'proof::scheduler::transition_invariants::slot_mapping_aligned',
}


def declaration_purpose(declaration):
    unit = declaration['source_unit']
    name = declaration['symbol'].removeprefix('vosti_verus::')
    if unit['assumption']:
        if not (unit['path'].startswith('src/boundary/')
                and unit['path'] in CONCRETE_FILES):
            raise ValueError(f'unreviewed trusted declaration purpose: {declaration["symbol"]}')
        return 'runtime_contracts', 'trusted external/uninterpreted runtime interface'
    if name in ABSTRACT_DECLARATIONS:
        return 'abstract_spec', 'reviewed observation, trace, or equality declaration'
    if name in CONCRETE_DECLARATIONS:
        return 'concrete_instantiation', 'reviewed concrete state correspondence'
    for files, purpose in (
        (ABSTRACT_FILES, 'abstract_spec'),
        (AUXILIARY_FILES, 'auxiliary_proof'),
        (CONCRETE_FILES, 'concrete_instantiation'),
    ):
        if unit['path'] in files:
            return purpose, 'reviewed file responsibility'
    raise ValueError(f'unreviewed specification purpose: {declaration["symbol"]}')


def purpose_partitions(rust, audit):
    specifications = {path: set(result['specification']) for path, result in rust.items()}
    owners = {path: {} for path in rust}
    classified = []
    used = set()
    for declaration in audit['selected_declarations']:
        unit = declaration['source_unit']
        if unit is None:
            continue  # generated dependencies are traversed but not counted
        path = unit['path']
        lines = specifications[path].intersection(range(unit['start'], unit['end'] + 1))
        if not lines:
            continue
        purpose, reason = declaration_purpose(declaration)
        used.add(declaration['symbol'].removeprefix('vosti_verus::'))
        for line in lines:
            prior = owners[path].setdefault(line, purpose)
            if prior != purpose:
                raise ValueError(f'conflicting specification purposes at {path}:{line}')
        classified.append(dict(symbol=declaration['symbol'], purpose=purpose, reason=reason))
    missing = (ABSTRACT_DECLARATIONS | CONCRETE_DECLARATIONS) - used
    if missing:
        raise ValueError(f'stale purpose overrides: {sorted(missing)}')
    result = {}
    for path, lines in specifications.items():
        result[path] = {purpose: [] for purpose in PURPOSES}
        for line in sorted(lines):
            result[path][owners[path].get(line, 'auxiliary_proof')].append(line)
        assert sum(map(len, result[path].values())) == len(lines)
    return result, dict(
        policy='Reviewed declaration/file purposes within the trace/trusted closure; '
               'all remaining specification lines are auxiliary proof effort. '
               'Abstract ownership is restricted to the engine-independent specification.',
        classified_declarations=classified,
    )
