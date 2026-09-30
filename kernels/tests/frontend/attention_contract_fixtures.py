"""Unsound whole-launch export inputs, retained only as negative fixtures."""

import ast
from pathlib import Path


KERNELS = Path(__file__).resolve().parents[2] / 'triton_kernels'


def iteration_scoped_attention_source(filename: str) -> str:
    original = (KERNELS / filename).read_text()
    lines = []
    replaced = 0
    for line in original.splitlines(keepends=True):
        if line.startswith('#     forall(cache_page, '):
            relation = line.split('PAGE_BLOCK_SIZE)), ', 1)[1].removesuffix(')),\n')
            relation = relation.replace('cache_page', 'ki * BLOCK_N // PAGE_BLOCK_SIZE')
            line = f'#     {relation} given ki * BLOCK_N // PAGE_BLOCK_SIZE,\n'
            replaced += 1
        lines.append(line)
    assert replaced == 2
    source = ''.join(lines)
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(original))
    return source
