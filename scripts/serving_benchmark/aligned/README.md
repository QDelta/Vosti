# Aligned controlled-phase benchmarks

This opt-in benchmark adapter is separate from the HTTP serving-wave benchmark.
Four concurrent HTTP requests do not guarantee a physical batch of four. The
adapter instead gates scheduler admission until all four identified requests are
queued, then checks the actual batch's identities, cached lengths and query
lengths. A split batch or cache miss invalidates the trial.

Each prefill request adds 256 tokens and returns one token. Two warmup waves
precede the requested number of measured waves per point.

The measured interval begins after admission, before normal scheduler batch
construction. It ends after forward execution and first-token sampling complete
on the GPU. Synchronization happens only at the interval boundaries. Reported
throughput is 1024 new input tokens divided by median engine interval; also
retain every interval and min/max. Client serialization, admission waiting,
HTTP transport and response processing are outside this pure-prefill metric.
It is not end-to-end serving throughput or a GPU-kernel-only timer.

The overlap scheduler and CUDA graph settings remain enabled. SGLang
may launch an extra decode before CPU result processing notices the one-token
output cap. This normal lookahead is allowed and separately recorded, not
included in pure-prefill timing. The next cohort starts only after preceding
requests finish.

Inputs have fixed seed 42 and distinct first tokens across requests/waves. To
reuse exactly L tokens, an untimed L+1 donor diverges from the measured prompt
at token L. Both scheduler prefix lengths and response cache accounting must
agree with L. The checker accepts neither a minus-one adjustment nor an approximate hit count.

`hooks/sitecustomize.py` installs the adapter only when
`VOSTI_ALIGNED_PHASE_RECORDS` is set. No installed engine package, kernel,
deployment configuration or proof is edited. The adapter fails closed if its
installation receipt or per-wave records are missing.

The supervisor retains sources, inputs, commands, logs, responses, scheduler
records, GPU telemetry, and exit status. It stops on failure; retries use new
directories. Keep these engine-side timings separate from HTTP measurements.

## Matrix runner

`campaign.py` runs a matrix against explicitly supplied qualified launch files,
with five cached contexts (0/4K/8K/12K/16K), two warmup waves and five measured
waves per point. Available modes come from the supplied `MODEL/MODE/launch.json`
files; missing requested modes are errors. Selected GPUs are eligible only when
idle; one trial owns each GPU, with continuous contention telemetry.
Models finish in order and get individual summaries.

Prefill uses Q256/output1. Decode uses Q1/output128: the initial first-token pass
is outside decode timing, which spans the subsequent 127 steps continuously.
Every step must contain exactly the same four requests, with expected growing
context. No per-token GPU synchronization is inserted. Separate per-point
warmup inputs prevent cache contamination; unexpected cache hits or misses fail.

Qualify the adapters on GPU before accepting a new matrix:

- `sglang_hook.py`: admission barrier plus schedule/forward/sample boundaries;
  preserves overlap, and records any unused lookahead separately.
- `vllm_hook.py`: buffers admission until all four requests reach EngineCore,
  validates scheduler geometry, and times normal schedule/execute/sample. It
  requires the existing single-GPU in-process executor; async scheduling stays
  at the mode's default. Unsupported executor layouts fail, not silently change.
- `native.rs`: admits all four through `Engine::try_add_request`, then calls the
  original `Engine::step`. Every timed step must emit for all four requests.
  The step includes its normal sample/commit work; there is no HTTP wrapper in
  this timer. Decode requires graph replay with no new capture/eager execution.
  Exact-size graph replay is valid; padding is not required when unnecessary.

`build_native.py` builds a driver against the frozen Engine crate and its sealed
deployments, without editing the engine. Keep the binary and build record with
the experiment. The runner snapshots the controller separately from the
serving source.

Use `--models`, `--modes`, `--contexts` and `--measured 1` for a small validation
before the full default campaign. Both commands require fresh output paths:

```sh
python -m scripts.serving_benchmark.aligned.build_native \
  --frozen /path/to/qualified-checkout --output /path/to/results/NEW-BUILD
python -m scripts.serving_benchmark.aligned.campaign \
  --native-build /path/to/results/NEW-BUILD \
  --launches /path/to/qualified-launches --gpus 0 \
  --output /path/to/results/NEW-MATRIX --detach
```

Run reports, result tables, validation failures and source snapshots belong in
those artifact directories, not in general `docs/`.
