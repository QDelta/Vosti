"""Opt-in import hook propagated to SGLang spawn workers by PYTHONPATH."""
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import PathFinder
import os
import sys


class AlignedLoader(Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self.wrapped, 'create_module', None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        if module.__name__ == 'vllm.v1.engine.core':
            from scripts.serving_benchmark.aligned.vllm_hook import install
        else:
            from scripts.serving_benchmark.aligned.sglang_hook import install
        install(module)


class AlignedFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in ('sglang.srt.managers.scheduler', 'vllm.v1.engine.core'):
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError('cannot install aligned scheduler adapter')
        spec.loader = AlignedLoader(spec.loader)
        return spec


if os.environ.get('VOSTI_ALIGNED_PHASE_RECORDS'):
    sys.meta_path.insert(0, AlignedFinder())
