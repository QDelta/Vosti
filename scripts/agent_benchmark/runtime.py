"""Instrument the pinned upstream agent; no model/engine implementation changes."""
from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from minisweagent.agents.default import LimitsExceeded
from minisweagent.environments.docker import DockerEnvironment

from .prepare import save


def message_payload(messages):
    return [{k: m[k] for k in ('role', 'content')} for m in messages]


def parse_stream(lines, start, clock=time.monotonic, on_event=None):
    text, events, usage, finish, done = [], [], None, None, False
    first, last = None, None
    for line in lines:
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            done = True
            break
        event = json.loads(data)
        elapsed = clock() - start
        item = dict(elapsed_s=elapsed, data=event)
        events.append(item)
        if on_event:
            on_event(item)
        if 'error' in event:
            raise RuntimeError(f'stream error: {event["error"]}')
        if event.get('usage'):
            usage = event['usage']
        for choice in event.get('choices', []):
            content = choice.get('delta', {}).get('content') or ''
            if content:
                first = elapsed if first is None else first
                last = elapsed
                text.append(content)
            if choice.get('finish_reason'):
                finish = choice['finish_reason']
    if not done or finish not in ('stop', 'length') or usage is None:
        raise RuntimeError('incomplete stream, unexpected finish reason, or absent usage')
    return dict(content=''.join(text), events=events, usage=usage, finish_reason=finish,
        ttft_s=first, last_content_s=last, request_s=clock()-start)


def strip_terminal_eos(text, eos_strings):
    # Vosti renders EOS text; baselines usually omit it. Normalize only a suffix,
    # never internal special-token text, and preserve the raw response separately.
    for eos in sorted(eos_strings, key=len, reverse=True):
        if eos and text.endswith(eos):
            return text[:-len(eos)]
    return text


def generation_eos_strings(tokenizer, plan):
    """Use the checkpoint's generation terminators, not a family-name table."""
    ids = plan.get('eos_token_ids')
    if ids is None:
        config = Path(tokenizer.name_or_path)/'generation_config.json'
        ids = json.loads(config.read_text())['eos_token_id']
    ids = [ids] if type(ids) is int else ids
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('invalid checkpoint EOS IDs')
    strings = [tokenizer.decode([i], skip_special_tokens=False,
        clean_up_tokenization_spaces=False) for i in ids]
    if any(not text for text in strings):
        raise ValueError('empty decoded EOS marker')
    return strings


class TimedModel:
    def __init__(self, base_url, model_name, tokenizer, plan, output: Path):
        self.config = SimpleNamespace(model_name=model_name)
        self.base_url, self.tokenizer, self.plan, self.output = base_url, tokenizer, plan, output
        self.cost = 0.0
        self.n_calls = self.generated = 0
        self.records = []
        self.deadline = float('inf')
        self.eos_strings = generation_eos_strings(tokenizer, plan)

    def get_template_vars(self):
        return dict(model_name=self.config.model_name)

    def query(self, messages, **kwargs):
        remaining = self.plan['max_generated_tokens_per_task'] - self.generated
        limit = min(self.plan['max_tokens_per_call'], remaining)
        if limit <= 0 or time.monotonic() >= self.deadline:
            raise LimitsExceeded('generated-token budget or safety timeout')
        payload = dict(model=self.config.model_name, messages=message_payload(messages),
            temperature=0.0, top_p=1.0, max_tokens=limit, stream=True,
            stream_options=dict(include_usage=True))
        tokens = self.tokenizer.apply_chat_template(payload['messages'], tokenize=True, add_generation_prompt=True)
        if isinstance(tokens, dict) or hasattr(tokens, 'keys'):
            tokens = tokens['input_ids']
        if len(tokens) + limit > self.plan['context_limit']:
            raise LimitsExceeded('context limit; no truncation')
        self.n_calls += 1
        prefix = self.output/f'call-{self.n_calls:03d}'
        save(prefix.with_suffix('.request.json'), dict(payload=payload, prompt_token_ids=tokens))
        start = time.monotonic()
        try:
            with httpx.Client(timeout=300, trust_env=False) as client, prefix.with_suffix('.events.jsonl').open('x') as event_log:
                def log_event(item):
                    event_log.write(json.dumps(item)+'\n')
                    event_log.flush()
                with client.stream('POST', self.base_url+'/v1/chat/completions', json=payload) as response:
                    response.raise_for_status()
                    record = parse_stream(response.iter_lines(), start, on_event=log_event)
            record['request_started_monotonic_s'] = start
            record['request_finished_monotonic_s'] = start + record['request_s']
            if record['usage']['prompt_tokens'] != len(tokens):
                raise RuntimeError('server/client prompt token counts disagree')
            generated = record['usage']['completion_tokens']
            if not isinstance(generated, int) or not 0 < generated <= limit:
                raise RuntimeError('invalid completion token accounting')
            self.generated += generated
            # Content-chunk TTFT and TPOT estimate, not kernel-token instrumentation.
            record['tpot_estimate_s'] = ((record['last_content_s']-record['ttft_s'])/(generated-1)
                if generated > 1 and record['ttft_s'] is not None else None)
            save(prefix.with_suffix('.response.json'), record)
            self.records.append({k: v for k, v in record.items() if k not in ('content', 'events')})
            print('CALL', self.output.name, self.n_calls, 'in', len(tokens), 'out', generated,
                  'ttft', record['ttft_s'], 'seconds', round(record['request_s'], 2), flush=True)
            return dict(content=strip_terminal_eos(record['content'], self.eos_strings))
        except Exception as error:
            save(prefix.with_suffix('.error.json'), dict(error=repr(error), elapsed_s=time.monotonic()-start))
            raise


def tree_blobs(text):
    # Images may commit chmod changes during setup. File bytes and paths must
    # nevertheless match the original benchmark base, including all test files.
    return {row.split('\t', 1)[1]: row.split('\t', 1)[0].split()[1:]
            for row in text.splitlines()}


class TaskEnvironment(DockerEnvironment):
    def __init__(self, image, plan, base_commit, output):
        self.records = []
        self.output = output
        super().__init__(image=image, cwd='/testbed', timeout=plan['tool_timeout_s'],
            container_timeout='4h', pull_timeout=120,
            env=dict(PAGER='cat', MANPAGER='cat', LESS='-R', PIP_PROGRESS_BAR='off',
                TQDM_DISABLE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
            run_args=['--rm','--network','none','--cap-drop','ALL','--security-opt','no-new-privileges',
                '--cpus',str(plan['container_cpus']),'--memory',plan['container_memory'],
                '--pids-limit','256','--label','vosti.swebench-pilot=true'])
        try:
            inspect = json.loads(subprocess.check_output(['docker','inspect',self.container_id], text=True))[0]
            assert not inspect['Mounts'] and not inspect['HostConfig']['DeviceRequests']
            assert inspect['HostConfig']['NetworkMode'] == 'none'
            base = self.checked('git ls-tree -r '+shlex.quote(base_commit))
            head = self.checked('git ls-tree -r HEAD')
            if tree_blobs(base) != tree_blobs(head) or self.checked('git status --porcelain').strip():
                raise RuntimeError('image does not contain the clean task base source')
            for path in ['/test.patch','/root/test.patch','/eval.sh','/root/eval.sh','/patch.diff']:
                if super().execute('test -e '+shlex.quote(path))['returncode'] == 0:
                    raise RuntimeError(f'possible grading artifact in image: {path}')
            self.base_head = self.checked('git rev-parse HEAD').strip()
            save(output/'environment.json', dict(container_id=self.container_id, image=image,
                base_commit=base_commit, image_head=self.base_head, base_source_bytes_match=True,
                config=inspect['HostConfig']))
        except BaseException:
            self.cleanup()
            raise

    def checked(self, command):
        result = super().execute(command)
        if result['returncode']:
            raise RuntimeError(f'environment setup failed: {result}')
        return result['output']

    def execute(self, command, cwd='', *, timeout=None):
        start = time.monotonic()
        # Host subprocess timeout alone leaves docker-exec children running.
        # GNU timeout inside the container terminates each timed-out tool group.
        seconds = timeout or self.config.timeout
        wrapped = f'timeout --kill-after=5s {seconds}s bash -lc {shlex.quote(command)}'
        result = super().execute(wrapped, cwd, timeout=seconds+15)
        record = dict(command=command, elapsed_s=time.monotonic()-start, **result)
        self.records.append(record)
        save(self.output/f'tool-{len(self.records):03d}.json', record)
        return result

    def patch(self):
        # Capture a patch even when the agent hits a budget before submitting.
        self.checked('git add -A')
        return self.checked('git -c core.fileMode=false diff --cached --binary '+self.base_head)

    def cleanup(self):
        if getattr(self, 'container_id', None):
            owned = self.container_id
            subprocess.run(['docker','rm','-f',owned], check=True, capture_output=True, timeout=60)
            self.container_id = None
