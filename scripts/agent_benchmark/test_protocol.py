import json
from collections import UserDict
from pathlib import Path

import pytest

from .prepare import select, PUBLIC_FIELDS
from .runtime import parse_stream, message_payload, strip_terminal_eos, tree_blobs, generation_eos_strings
from .run import distribution, prompt_token_count
from .campaign import configure_launch


def test_selection_order_independent_and_no_grading_fields():
    rows = [dict(instance_id=str(i), patch='secret') for i in range(20)]
    assert select(rows,42,10) == select(list(reversed(rows)),42,10)
    assert select(rows,42,10) != select(rows,43,10)
    assert not set(PUBLIC_FIELDS) & {'patch','test_patch','hints_text','FAIL_TO_PASS','PASS_TO_PASS'}
    with pytest.raises(ValueError):
        select(rows+rows,42,10)


def stream_event(**kw):
    return 'data: '+json.dumps(kw)


def test_difficulty_selection():
    rows = [dict(instance_id=str(i), difficulty='<15 min fix' if i % 2 else '') for i in range(20)]
    chosen = select(rows,42,5,'<15 min fix')
    assert all(r['difficulty'] == '<15 min fix' for r in chosen)
    assert chosen == select(rows[::-1],42,5,'<15 min fix')
    with pytest.raises(ValueError):
        select(rows,42,11,'<15 min fix')


def test_stream_separate_usage_and_chunk_timing():
    ticks = iter([1.,2.,3.,4.,5.])
    lines = [': heartbeat',stream_event(choices=[dict(delta={'role':'assistant'})]),
        stream_event(choices=[dict(delta={'content':'abc'})]),
        stream_event(choices=[dict(delta={'content':'def'},finish_reason='stop')]),
        stream_event(choices=[],usage=dict(prompt_tokens=10,completion_tokens=3)), 'data: [DONE]']
    out = parse_stream(lines,0,clock=lambda:next(ticks))
    assert out['content']=='abcdef' and out['ttft_s']==2 and out['last_content_s']==3
    assert len(out['events'])==4 and out['usage']['completion_tokens']==3


def test_incomplete_stream_fails():
    with pytest.raises(RuntimeError):
        parse_stream([stream_event(choices=[dict(delta={'content':'x'})])],0,clock=lambda:1)


def test_client_metadata_never_sent_and_only_terminal_eos_removed():
    assert message_payload([dict(role='user',content='hello',timestamp=4)])==[dict(role='user',content='hello')]
    assert strip_terminal_eos('inside <eos> text<end_of_turn>',['<eos>','<end_of_turn>'])=='inside <eos> text'


def test_checkpoint_owned_terminators_are_only_removed_at_end():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            assert kwargs['skip_special_tokens'] is False
            return {1:'<eos>',106:'<turn|>',50:'<|tool_response>'}[ids[0]]
    eos = generation_eos_strings(Tokenizer(), {'eos_token_ids':[1,106,50]})
    for marker in eos:
        assert strip_terminal_eos('inside '+marker+' text'+marker, eos)=='inside '+marker+' text'
    for ids in [[],[-1],[True],['1']]:
        with pytest.raises(ValueError):generation_eos_strings(Tokenizer(), {'eos_token_ids':ids})


def test_tree_check_ignores_modes_not_source_and_metrics():
    assert tree_blobs('100644 blob abc\tf.py')==tree_blobs('100755 blob abc\tf.py')
    assert tree_blobs('100644 blob abc\tf.py')!=tree_blobs('100644 blob def\tf.py')
    assert distribution([1,2,3,None])['p50']==2
    assert distribution([]) is None


@pytest.mark.parametrize('engine,flags', [
    ('vosti', []),
    ('vllm', ['--max-num-seqs','4','--max-model-len','16384']),
    ('sglang', ['--max-running-requests','4','--context-length','16384'])])
def test_qualified_launch_preserves_backend_and_changes_capacity(engine, flags):
    original = dict(command=['python',*flags,'--attention-backend','selected'],
        execution=dict(engine=engine,key='mode',attention_backend='selected'),
        settings=dict(max_sequences=4,context_limit=16384,gpu_index=3),
        environment=dict(CUDA_VISIBLE_DEVICES='3',VLLM_BATCH_INVARIANT='1'))
    before = json.dumps(original,sort_keys=True)
    out = configure_launch(original,dict(concurrency=1,context_limit=40960),Path('/artifact'))
    assert json.dumps(original,sort_keys=True) == before
    assert out['execution'] == original['execution']
    assert out['settings']['max_sequences'] == 1 and out['settings']['context_limit'] == 40960
    assert out['environment']['VLLM_BATCH_INVARIANT'] == '1'
    assert out['environment']['CUDA_VISIBLE_DEVICES'] == '3'
    assert ('--skip-server-warmup' in out['command']) == (engine == 'sglang')
    if flags:
        assert out['command'][out['command'].index(flags[0])+1] == '1'
        assert out['command'][out['command'].index(flags[2])+1] == '40960'
    else:
        assert out['environment']['VOSTI_MAX_MODEL_LEN'] == '40960'


def test_qualified_launch_rejects_unknown_engine_and_missing_capacity_flag():
    spec = dict(command=[],execution=dict(engine='unknown'),settings={},environment={})
    with pytest.raises(ValueError):
        configure_launch(spec,dict(concurrency=1,context_limit=40960),Path('/artifact'))
    spec['execution']['engine'] = 'vllm'
    with pytest.raises(ValueError,match='max-num-seqs'):
        configure_launch(spec,dict(concurrency=1,context_limit=40960),Path('/artifact'))


def test_sglang_skip_builtin_warmup_is_not_duplicated():
    spec = dict(command=['python','--max-running-requests','4','--context-length','16384',
                         '--skip-server-warmup','--enable-deterministic-inference'],
        execution=dict(engine='sglang',key='mode',attention_backend='triton'),
        settings=dict(max_sequences=4,context_limit=16384,gpu_index=3),environment={})
    out = configure_launch(spec,dict(concurrency=1,context_limit=40960),Path('/artifact'))
    assert out['command'].count('--skip-server-warmup') == 1
    assert '--enable-deterministic-inference' in out['command']


@pytest.mark.parametrize('result', [[1,2,3], {'input_ids':[1,2,3]},
    UserDict(input_ids=[1,2,3],attention_mask=[1,1,1])])
def test_warmup_token_count_accepts_batch_encoding_mapping(result):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=True,add_generation_prompt=True)
            return result
    assert prompt_token_count(Tokenizer(), []) == 3


def test_text_content_adapter_preserves_strings_and_arrays():
    from transformers.utils.chat_template_utils import _compile_jinja_template
    prefix = Path(__file__).with_name('text_content.jinja').read_text()
    template = _compile_jinja_template(prefix + "{% for m in messages %}{{m.role}}:{{m.content}};{% endfor %}")
    plain = [dict(role='system', content='  hello\n'),dict(role='user',content='world')]
    arrays = [dict(role=m['role'],content=[dict(type='text',text=m['content'])]) for m in plain]
    assert template.render(messages=plain) == template.render(messages=arrays) == 'system:  hello\n;user:world;'
    arrays[0]['content'] = [dict(type='image',text='not allowed')]
    with pytest.raises(Exception,match='text content only'):
        template.render(messages=arrays)


def test_report_handles_no_successful_requests(tmp_path):
    from .report import export_csv
    export_csv(tmp_path/'empty.csv', [])
    assert (tmp_path/'empty.csv').exists()
