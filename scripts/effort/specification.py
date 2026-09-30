"""Purpose-based subdivision using source-pinned, compiler-resolved identities.

This is a declaration dependency closure, not a minimal semantic interface.
Proof/exec bodies never provide dependency edges. Original LOC buckets stay put.
"""

PARTS = ('theorem_surface', 'trusted_assumptions', 'auxiliary')
TRACE_CLAIMS = (
    'proof::lemma_serving_deterministic',
)


def accounting_claims(claim_surface):
    """The publication root is engine satisfaction, not every audited claim.

    Initialization and one-step results remain audited supporting results;
    their contracts do not independently seed the specification LOC closure.
    """
    claims = claim_surface['top_level_claims']
    selected = []
    for name in TRACE_CLAIMS:
        matches = [claim for claim in claims if claim['item'] == name]
        if len(matches) != 1:
            raise ValueError(f'accounting trace root must resolve uniquely: {name}')
        selected.append(matches[0])
    return selected


def split_specifications(rust, claims, declarations):
    units, mapped = {}, {}
    for path, result in rust.items():
        if any(a['kind'] == 'assume' for a in result['trusted']):
            # No such expressions occur in the current core inventory. Never
            # silently fall back to terminal-name resolution if one is added.
            raise ValueError(f'inline assume needs typed expression dependency extraction: {path}')
        spec = set(result['specification'])
        for unit in result['surface']:
            key = f"{path}:{unit['start']}:{unit['name']}"
            units[key] = {k: v for k, v in unit.items() if k != 'references'}
            units[key]['path'] = path
            matches = [symbol for symbol, node in declarations.items()
                       if node['path'] == path and unit['start'] <= node['line'] <= unit['end']
                       and (node['kind'] == 'Datatype') == (unit['kind'] == 'type')]
            if len(matches) > 1:
                raise ValueError(f'source declaration does not map uniquely: {key}: {matches}')
            if not matches:
                if spec.intersection(range(unit['start'], unit['end'] + 1)):
                    raise ValueError(f'counted specification has no compiler identity: {key}')
                continue
            symbol = matches[0]
            if symbol in mapped:
                raise ValueError(f'compiler identity maps to multiple source units: {symbol}')
            mapped[symbol] = key

    roots = []
    for claim in claims:
        symbol = 'vosti_verus::' + claim['item']
        if (symbol not in mapped or declarations[symbol]['mode'] != 'Proof'
                or declarations[symbol]['path'] != claim['source_span']['path']):
            raise ValueError(f'theorem root must resolve uniquely: {symbol}')
        roots.append(symbol)
    assumptions = [symbol for symbol, key in mapped.items() if units[key]['assumption']]
    reasons = {symbol: 'top-level theorem contract' for symbol in roots}
    reasons.update({symbol: 'trusted declaration' for symbol in assumptions})
    selected, pending = set(), sorted(reasons)
    while pending:
        symbol = pending.pop()
        if symbol in selected:
            continue
        selected.add(symbol)
        for target in declarations[symbol]['references']:
            if target not in declarations:
                raise ValueError(f'unresolved compiler dependency: {symbol} -> {target}')
            reasons.setdefault(target, symbol)
            if target not in selected:
                pending.append(target)

    partitions = {}
    for path, result in rust.items():
        spec, main, trusted = set(result['specification']), set(), set()
        for symbol in selected:
            unit = units.get(mapped.get(symbol))
            if unit is not None and unit['path'] == path:
                lines = spec.intersection(range(unit['start'], unit['end'] + 1))
                main.update(lines)
                if unit['assumption']:
                    trusted.update(lines)
        partitions[path] = dict(theorem_surface=sorted(main - trusted),
                                trusted_assumptions=sorted(trusted),
                                auxiliary=sorted(spec - main))
        assert sum(map(len, partitions[path].values())) == len(spec)

    return partitions, {
        'root_claims': [claim['item'] for claim in claims],
        'resolution': 'Source-pinned Verus VIR function/type identities, mapped to source spans. '
                      'Spec bodies and declaration contracts/types are traversed; proof/exec bodies '
                      'are not. This is a declaration closure, not a minimal semantic interface.',
        'selected_declarations': [dict(declarations[symbol],
                                      source_unit=units.get(mapped.get(symbol)),
                                      reason=reasons[symbol]) for symbol in sorted(selected)],
        'ambiguous_names': {},
    }
