"""A fixed unroll hint must retain the same ordered loop IR; fail closed otherwise."""
import pytest

from ir.pp import pretty_kernel
from ir.translate import TranslationError, translate_kernel_source
from ir.validate_subset import validate_triton_subset


def source(iterator):
    return f'''import triton
import triton.language as tl
# @params(scalar(N, int))
# @grid(1)
@triton.jit
def kernel(N):
    pid = tl.program_id(0)
    acc = 0
    for i in {iterator}:
        acc = acc + i
'''


@pytest.mark.parametrize('bounds', ['N', '1, N'])
@pytest.mark.parametrize('hint', ['', ', loop_unroll_factor=1', ', loop_unroll_factor=2', ', loop_unroll_factor=4'])
def test_unroll_hint_preserves_ordered_loop_ir(bounds, hint):
    plain = translate_kernel_source(source(f'range({bounds})'), 'kernel')
    annotated = source(f'tl.range({bounds}{hint})')
    assert not validate_triton_subset(annotated, 'kernel')
    assert pretty_kernel(translate_kernel_source(annotated, 'kernel')) == pretty_kernel(plain)


@pytest.mark.parametrize('iterator', [
    'tl.range(N, step=2)', 'tl.range(0, N, 2)', 'tl.range(N, num_stages=2)',
    'tl.range(N, flatten=True)', 'tl.range(N, warp_specialize=True)',
    'tl.range(N, **kwargs)', 'tl.range(N, loop_unroll_factor=N)',
    'tl.range(N, loop_unroll_factor=0)', 'tl.range(N, loop_unroll_factor=-1)',
    'tl.range(N, loop_unroll_factor=True)', 'tl.range(N, loop_unroll_factor=2.0)',
    'tl.range(N, loop_unroll_factor=2, loop_unroll_factor=4)',
    'range(N, step=2)', 'range(N, loop_unroll_factor=2)', 'range(N, **kwargs)',
])
def test_unsupported_range_syntax_is_rejected(iterator):
    with pytest.raises(TranslationError, match='range|unroll'):
        translate_kernel_source(source(iterator), 'kernel')


@pytest.mark.parametrize('hint', [
    '2 if WIDTH == 128 else 1',
    '1 if WIDTH != 128 else 2',
    '2 if WIDTH == 128 else (4 if WIDTH == 64 else 1)',
])
def test_constexpr_selected_positive_hint_preserves_ordered_ir(hint):
    def annotated(iterator):
        return source(iterator).replace('def kernel(N):', 'def kernel(N, WIDTH: tl.constexpr):')
    plain = translate_kernel_source(annotated('range(N)'), 'kernel')
    selected = annotated(f'tl.range(N, loop_unroll_factor={hint})')
    assert not validate_triton_subset(selected, 'kernel')
    assert pretty_kernel(translate_kernel_source(selected, 'kernel')) == pretty_kernel(plain)


@pytest.mark.parametrize('hint', [
    '2 if N == 128 else 1',  # Runtime input, even if a caller specializes it.
    '2 if UNKNOWN == 128 else 1',
    '2 if WIDTH == 128 else 0', '2 if WIDTH == 128 else -1',
    'True if WIDTH == 128 else 1', '2.0 if WIDTH == 128 else 1',
    'WIDTH if WIDTH == 128 else 1',
    '2 if WIDTH == True else 1', '2 if WIDTH == 128.0 else 1',
    '2 if WIDTH == N else 1', '2 if WIDTH < 128 else 1',
    '2 if WIDTH == 128 == N else 1',
    '2 if (WIDTH == 128 and N == 1) else 1',
    '2 if WIDTH + 1 == 128 else 1', '2 if tl.load(N) == 128 else 1',
    '2 if WIDTH == 128 else (2 if N == 1 else 1)',
])
def test_conditional_hint_rejects_runtime_tests_and_nonpositive_branches(hint):
    annotated = source(f'tl.range(N, loop_unroll_factor={hint})').replace(
        'def kernel(N):', 'def kernel(N, WIDTH: tl.constexpr):')
    assert validate_triton_subset(annotated, 'kernel')
    with pytest.raises(TranslationError, match='range|unroll'):
        translate_kernel_source(annotated, 'kernel')


def test_constexpr_hint_parameter_cannot_be_reassigned_from_runtime():
    annotated = source('tl.range(N, loop_unroll_factor=2 if WIDTH == 128 else 1)').replace(
        'def kernel(N):', 'def kernel(N, WIDTH: tl.constexpr):').replace(
        '    acc = 0', '    WIDTH = N\n    acc = 0')
    with pytest.raises(TranslationError, match='kernel parameter'):
        translate_kernel_source(annotated, 'kernel')
