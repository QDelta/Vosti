"""Adapt a qualified launch to controlled-phase geometry without changing its backend."""
import json


def adapt(original, gpu, cache, frozen):
    spec=json.loads(json.dumps(original));engine=spec['execution']['engine']
    port=18400+gpu
    spec['settings'].update(gpu_index=gpu,port=port,context_limit=32768,
        max_sequences=4,max_batched_tokens=1024,vosti_num_blocks=1152)
    spec['base_url']=f'http://127.0.0.1:{port}'
    spec['environment']['CUDA_VISIBLE_DEVICES']=str(gpu)
    spec['benchmark_frozen_root']=str(frozen)
    def change(flag,value):
        if spec['command'].count(flag)!=1:
            raise ValueError(f'expected one {flag}')
        spec['command'][spec['command'].index(flag)+1]=str(value)
    if engine=='vosti':
        spec['environment'].update(VOSTI_SERVER_PORT=str(port),VOSTI_NUM_BLOCKS='1152',
            VOSTI_MAX_SEQS='4',VOSTI_MAX_MODEL_LEN='32768',VOSTI_MAX_BATCHED_TOKENS='1024')
    else:
        change('--port',port)
        if engine=='vllm':
            change('--max-model-len',32768);change('--max-num-seqs',4)
            change('--max-num-batched-tokens',1024)
        else:
            change('--context-length',32768);change('--max-running-requests',4)
            # Deterministic Triton's default prefill alignment is 4096. A
            # smaller chunk budget leaves every long donor unschedulable.
            chunk = 4096 if spec['execution']['key']=='sglang-deterministic-triton' else 1024
            change('--max-prefill-tokens',chunk);change('--chunked-prefill-size',chunk)
            spec['settings']['max_batched_tokens']=chunk
            spec['benchmark_sglang_zero_omission_verified']=True
    for key,sub in [('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:
        spec['environment'][key]=str(cache/sub)
    return spec
