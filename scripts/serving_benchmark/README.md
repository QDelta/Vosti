# OpenAI serving benchmark

Use small controlled-phase runs before larger multi-session serving campaigns.
For fixed physical batches and engine-side timing, use the
[aligned adapter](aligned/README.md). The HTTP measurements below include serving
and transport overhead. Keep raw timing and contention evidence outside the
source tree. Disable logit-collection hooks during performance runs.

## Preparation commands

Provision the separate engine environments with the
[stack setup recipe](../../docs/setup.md#optional-baseline-environments).

Use `uv run --locked python scripts/serving_benchmark/prepare.py PRESET --help`.
All seven presets only write workload files or trial plans; they never launch a server:

| Preset | Output |
| --- | --- |
| `pilot` | Short multi-turn trial, optionally with small phase screens or saved-job replay |
| `multi-session-replay` | Four-session, six-turn trial plans with ordered warmup and saved request arrivals |
| `multi-workloads` | Shared measured/warmup multi-session workload matrix |
| `phase-workloads` | Shared cold-prefill, decode, and cached-extension workloads |
| `multi-trials` | Multi-session server plans, optionally using calibrated offered rates |
| `phase-trials` | Sequential server plans for the phase workload matrix |
| `capacity-trials` | Separate near-context-limit capacity probes |

Use a fresh output directory; preparation refuses to overwrite existing output.
Keep each run with its frozen sources. Runs cannot resume across source revisions.

### Multi-session replay

```sh
uv run --locked python scripts/serving_benchmark/prepare.py multi-session-replay \
  --checkpoint gemma3-4b --stack-root /path/to/engine-stack --gpu-index 3 \
  --mode vosti-padded-graph --mode vllm-invariant-flash-attn \
  --mode sglang-deterministic-fa3 --seed 42 --output /path/to/new-replay
```

This synthetic workload has four independent sessions, each with six turns.
Initial prompts contain 8,193 shared tokens and 8,192 distinct tokens. Each turn
generates 768 tokens; followups append 256 tokens including their separator.
The request-based Poisson trace offers 2 requests/s with a fixed seed. Each
followup also waits for its preceding response and a 0.5-second think delay;
offered QPS therefore need not equal achieved QPS. Actual generated text is
retained, so retokenized lengths can differ across engines.

In one server lifetime, three excluded warmup stages run in order: populate a
disjoint decode donor, sweep decode shapes with four concurrent sessions, then
populate the shared prefix. Only then does measured replay start. The donor
generates one token; the decode sweep generates 5,952 tokens per session.
The shared-prefix donor has 8,197 input tokens; realized cache hits, including
page-boundary recomputation, remain observations rather than assumed geometry.

The plan uses the general `campaign.py` / `server_trial.py` execution path
described below. Modes are explicit: preparation does not establish that a
backend supports the selected checkpoint. The measured inputs and arrival
trace are identical across modes. Existing output directories are refused.
`multi_turn.py run` accepts repeated `--warmup-workload` arguments for ordered
stages; each stage's raw result and workload digest is retained in
`warmup_stages`, outside measured timings. Any failed stage prevents later
stages and measurement. Single-stage runs use the `warmup` format.
The common-rate calibration matrix uses a separate workload profile.

### Replay against a running server

For an artifact rerun, you can use the prepared workloads directly, without
the campaign supervisor or contention monitor. Use an idle GPU and a fresh
server configured for four sequences and a 32K context limit. In the
[artifact container](../../docs/artifact.md), with Gemma3-4B mounted under
`/models` and the server on port 8000:

```sh
uv run --locked python scripts/serving_benchmark/prepare.py multi-session-replay \
  --checkpoint gemma3-4b --stack-root /results/stack --gpu-index 0 \
  --mode vosti-padded-graph --seed 42 --output /results/replay-inputs
uv run --locked python scripts/serving_benchmark/multi_turn.py run \
  --tokenizer /models/gemma-3-4b-it \
  --workload /results/replay-inputs/measured.json \
  --warmup-workload /results/replay-inputs/decode-donor.json \
  --warmup-workload /results/replay-inputs/decode-warmup.json \
  --warmup-workload /results/replay-inputs/prefix-warmup.json \
  --arrival-trace /results/replay-inputs/arrivals.json --stagger-seconds 0 \
  --base-url http://127.0.0.1:8000 --model gemma3-4b \
  --engine-label vosti-padded-graph --concurrency 4 --context-limit 32768 \
  --output /results/replay-result.json
```

Preparation also writes launch plans; this direct replay uses only the
workloads and arrival trace, so `/results/stack` need not exist. Start and
qualify the server yourself as described in the artifact guide. Expect 24
successful measured requests after the excluded warmups. Repeat against each
baseline server using the same files, a distinct output path, and an accurate
engine label. Record server versions, launch settings, backend, and GPU details
alongside the result. This path records client results without the campaign's
automatic source capture or GPU-contention evidence.

## Broader matrix tooling

`python -m scripts.serving_benchmark.matrix --output /path/to/new-plan.json`
creates a plan for Gemma3 4B/12B/27B and Llama3 3B/8B across seven engine
modes and three repetitions. It includes cold prefill (512/2K/8K),
decode (1K/8K/32K context), and cached extension (8K/32K prefix plus 128/1K
new tokens), at concurrency 1 and 4. Multi-session cases start at 8K or 32K,
generate 512 or 1,024 tokens per turn, append 128 tokens, and use four turns,
eight sessions, and a 0.5-second think delay. This initial session count is
not sufficient for claims about extreme tail latency.

Light/moderate arrival rates remain explicitly unset until a pilot selects
common per-model rates for all engines. Do not independently choose a rate
for each engine. Use separate warmup, confirm token geometry and observed
cache hits, and retain graph-capture/compilation and GPU-contamination evidence.
Performance runs must not enable determinism-suite logit observers.

`calibrate_rates.py --pilots pilots.json --output new-rates.json` selects the
common rates from the completed closed-loop concurrency-four cases: every
performance model, seven modes, both contexts and all three repetitions. It
checks complete warmup/results, observed followup prefix reuse and matching
server-trial telemetry. For each model it takes the lowest mode/context median
throughput, then uses 50% and 80% of that reference for light and moderate load.
These are finite-workload reference rates, not measured saturation capacities.
Missing pilot cells stop calibration.
For unavailable backend cases, supply `--exclusions exclusions.json`, a JSON map
from `checkpoint/mode` keys to reasons. Only those pairs may be absent; every
other pilot is still required. The selection retains the exclusions and uses
common per-model rates across the remaining modes.

For these workloads, pass `multi_turn.py prepare --suffix-tokens 128
--suffix-includes-separator` so that 128 counts the complete appended suffix,
including the separator. Without the flag, the separator is outside that count.
Subsequent full-prompt retokenization can still introduce drift;
the usual per-turn geometry checks remain required.

`python -m scripts.serving_benchmark.prepare multi-workloads --output /path/to/new-workloads`
prepares the five checkpoints' two multi-session profiles and three repetitions,
with distinct measured (eight sessions) and warmup (four sessions) manifests.
The same manifest pair is used for every engine mode. Preparation uses only
the CPU; check GPU memory capacity separately before running it.

## Token-counted phase measurements

`phases.py` prepares and replays cold-prefill, decode, and cached-extension
workloads through streaming `/v1/completions`. Prepare once per tokenizer and
reuse the exact manifest across engine modes. `prepare.py phase-workloads` emits
300 manifests covering the five performance checkpoints, twenty phase settings,
and three repetitions (2,100 measurements across seven modes):

```sh
.venv/bin/python -m scripts.serving_benchmark.prepare phase-workloads --output /path/to/new/phase-workloads
.venv/bin/python -m scripts.serving_benchmark.phases run \
  --tokenizer /path/to/checkpoint --workload /path/to/phase-workload.json \
  --model served-name --engine-label vosti-padded-graph \
  --base-url http://127.0.0.1:8000 --context-limit 40960 \
  --output /path/to/new/result.json
```

Each manifest has one disjoint warmup wave and four measured waves at concurrency
one or four. For cached extensions, each wave first runs prefix donors in the
same server lifetime; donor time is excluded and donor failures stop measurement.
Inputs have exact round-trip token counts and unchanged donor-token prefixes.
Server usage must agree with prompt counts and the fixed ignore-EOS output budget.

Page boundaries, final-logit recomputation, and eviction can reduce prefix
reuse. Compare the recorded cached-token counts and uncached-query lengths
across engines, not just the requested lengths. Missing cache counts remain
unknown; an unexpected hit invalidates a cold measurement.

TTFT includes serving overhead. Decode TPOT excludes initial TTFT; SSE event
intervals are not necessarily token intervals. Throughput uses the sum of measured
wave durations, excluding warmup, donors and between-wave gaps; it is not a
steady-state serving-throughput claim. No logit observer is used. The client does
not launch or qualify servers: the campaign must separately retain backend,
deployment, graph-warmup, GPU-contention and capacity evidence.

## Serving client

`server_config.py` defines shared launch settings and mode-specific switches for
the seven modes. Default modes receive no attention-backend pin. The configuration
uses SGLang's phase-specific decode-graph option; resolved choices
must still be checked from server evidence. Vosti requires a sealed deployment
bundle and enables the padded-cover graph overlay. Inherited engine overrides
and determinism logit-observer hooks are removed from performance environments.

`server_trial.py --spec launch.json --jobs jobs.json --output /path/to/new/trial`
runs a prepared job list against one owned server. Wrap it in
`scripts/common/gpu_monitor.py`; the worker itself checks initial idleness,
but does not replace continuous contention monitoring. Jobs use unique `id`s and
either `kind=phase, workload=...` or
`kind=multi_session, workload=..., warmup=..., concurrency=...`.
`--validate-only` checks all token geometry and cross-job prefix disjointness
without starting a server. Repeated inputs must use a fresh server, not a retry
against a silently warmed cache. Raw `/server_info`, `/metrics`, `/v1/models`,
server logs and client outputs are preserved. These helpers are not evidence that
the complete matrix has run or that the configured KV pool fits every checkpoint.

`prepare.py phase-trials` and `prepare.py multi-trials` accept `--root` to bind
launch plans to a separate frozen serving checkout. Plan generation is distinct
from serving execution. Both baseline launchers explicitly enable per-request
cache reporting (`--enable-prompt-tokens-details` / `--enable-cache-report`).

`prepare.py capacity-trials` makes separate near-limit C4 probes (40,704 input
tokens plus 256 generated) to run before benchmark cells. The probes use one
warmup and one measured wave and do not count as matrix repetitions.

`campaign.py --root FROZEN --stack-root STACK --build BUILD --plan PLAN.json
--output NEW_DIRECTORY` executes prepared plans sequentially. Supply multiple
`--plan` arguments to order capacity probes before phase/multi-session trials.
It records source, binary, workload and package identities, qualifies shared
Vosti bundles under valid GPU telemetry, and continuously monitors every trial.
`--resume` rechecks completed evidence and uses a new server/output
directory for each failed trial retry. Completion applies only to supplied plans;
arrival-rate plans still require the completed common calibration first.

Use `--exclusions exclusions.json` to leave specified checkpoint/mode pairs unrun
in the original matrix. The file is hashed with the other inputs and must match
the exclusions used for rate calibration. The driver emits `partial.json` while
any trials remain unrun; measurement errors still stop it. It does not depend on
a determinism campaign or its coordinator PID. Sequence separate campaigns in
the calling shell or job scheduler. Run the coordinator from a frozen checkout
as well as freezing the serving source, so later main-branch edits do not alter
its resume identity.

### ShareGPT client and timing

The measured request interval begins immediately before the HTTP request and
ends after the `[DONE]` streaming marker. It includes network overhead, server
queueing, tokenization, model execution, and detokenization. Dataset loading,
client tokenization for accounting, and request construction are excluded.
The client reports request and input/output-token throughput plus mean, p50,
p90, and p99 end-to-end latency, time to first token, time per output token,
inter-output-event latency, and client scheduling lag. A streaming output event
may contain zero, one, or multiple generated tokens, so exact output-token
counts come from the final usage record while event timing remains the
client-observable API timing. Counts of token IDs exposed by a server are retained
as supplemental evidence, but are not required. The client retains every raw
request record and refuses to overwrite an existing result.
If a request generates only skipped special tokens, usage still validates its
completion while visible-output timing metrics for that request remain null.
For Vosti, the captured `/metrics` snapshots also include the sealed
deployment digest, CUDA-graph counters, engine-step count, and peak/overlap
request counts, so a run can show that dynamic arrivals actually coexisted in
the Engine.

Generate disjoint measured and warmup workloads from a downloaded ShareGPT
dataset with the checkpoint tokenizer:

```bash
.venv/bin/python scripts/serving_benchmark/workloads/sharegpt.py \
  --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer-path /path/to/models/Llama-3.2-3B \
  --output-dir /path/to/results/serving/workloads/llama3-3b \
  --requests 128
```

Start a Vosti family server with a sealed deployment bundle, for example:

```bash
MODEL_PATH=/path/to/models/Llama-3.2-3B \
VOSTI_DEPLOYMENT_BUNDLE=/path/to/bundle \
VOSTI_SERVED_MODEL_NAME=llama-3.2-3b \
VOSTI_CUDA_GRAPH=1 \
VOSTI_NUM_BLOCKS=4096 \
uv run --locked python scripts/launch.py --kind server --family llama3
```

Then run the same workload against any compatible server:

```bash
.venv/bin/python scripts/serving_benchmark/run.py \
  --base-url http://127.0.0.1:8000 \
  --engine-label vosti-qualified-padded-graph \
  --model llama-3.2-3b \
  --tokenizer /path/to/models/Llama-3.2-3B \
  --workload /path/to/sharegpt.json \
  --warmup-workload /path/to/sharegpt_warmup.json \
  --request-rate 2 \
  --output /path/to/results/serving/result.json
```

For cross-engine comparisons, use the exact same workload file, arrival seed,
arrival process, request rate, endpoint, and model tokenizer. The client sends
greedy fixed-length requests (`temperature=0`, `ignore_eos=true`) so expected
output-token counts are known before timing. `--engine-label` is retained as
an operator-declared configuration label; it is not evidence that an external
server actually selected a claimed backend, so preserve that server's startup
log alongside the benchmark artifact.

## Synthetic multi-turn sessions

`prepare.py pilot` prepares four sessions with three turns each (8K initial
context, 128-token suffix, 256-token output), at a default offered rate of
0.5 QPS. This is a smoke-test workload, not a capacity estimate.
For a longer decode-heavy screen, `--turns 12 --output-tokens 512 --request-rate 0.4`
keeps the same four sessions and grows context to just under the 16K server
limit. Warmup uses the same geometry with a disjoint seed.
By default it prepares vLLM invariant FA3, SGLang deterministic FA3, and Vosti.
Select `--checkpoint gemma4-31b` explicitly for Gemma-4. On H200 its 512-wide
global layers rule out all-FA3 execution. The serving-only mode
`vllm-invariant-fa3-local-triton-global` pins FA3 for `sliding_window` groups
and Triton for `full_attention` groups; it does not expand the deterministic
test matrix. `vllm-fast-auto` and `sglang-fast-auto` leave attention unpinned.
`vllm-invariant-auto` likewise leaves attention unpinned but enables invariant
mode, allowing separate startup diagnostics rather than silently substituting
a working backend. Backend labels must come from full model startup logs:
model-specific configuration updates can override the generic selector.
Use `--phase-screen` to add three small controlled jobs to a new pilot, or
reuse all those jobs unchanged through `--replay-jobs`.
To screen another Vosti candidate without repeating the baselines, use
`--mode vosti-padded-graph --replay-jobs /path/to/previous/plan/jobs.json` with a
new `--output` directory and the same `--checkpoint` and `--stack-root`.
This reuses the previous workloads and saved arrival offsets; it does not
regenerate them from seeds. The campaign still binds the new source and checks
token geometry, cache use, and GPU contention. Compare result files using
`multi_turn compare`; a short pilot is not a substitute for sustained-load
validation or the targeted controlled-phase regression checks.

### Diagnostic decode profiling

`profile_decode.py` launches one owned native server under Nsight Systems,
runs two warmup waves, then records a short repeated-prefix decode wave at
individual CUDA Graph node granularity. It rejects a new graph capture inside
that interval. Use a clean, frozen checkout and a deployment bundle matching
the bound kernel/support sources, model, and environment:

```sh
.venv/bin/python scripts/common/gpu_monitor.py \
  --gpu-index 3 --output /path/to/new-telemetry.json -- \
  .venv/bin/python -m scripts.serving_benchmark.profile_decode \
  --spec /path/to/native-launch.json --workload /path/to/measured.json \
  --deployment-bundle /path/to/fresh-qualified-bundle \
  --concurrency 4 --tokens 256 --output /path/to/new-profile
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum /path/to/new-profile/decode.nsys-rep
```

Graph-node tracing affects timing. CUDA API duration can include waiting for
the GPU, so it cannot be read directly as CPU work. Validate any proposed
improvement with uninstrumented serving runs.
`kernels/benchmarks/attention.py` and `kernels/benchmarks/matmul.py` provide
small fixed-geometry launch screens with decode and prefill cases and sampled
bytewise row-consistency checks. They never update deployed selectors or admit
candidate configurations; those still require the regular proof and deployment
qualification gates.

For reproducible offered-rate comparisons, prepare the workload once, then save
the arrival trace once and replay both artifacts on every engine:

```sh
.venv/bin/python -m scripts.serving_benchmark.multi_turn prepare-trace \
  --workload /path/to/measured-workload.json --request-rate 2 \
  --arrival-process poisson --arrival-seed 42 --output /path/to/arrivals.json
.venv/bin/python -m scripts.serving_benchmark.multi_turn run \
  --tokenizer /path/to/checkpoint --model served-name --engine-label vosti-padded-graph \
  --base-url http://127.0.0.1:8000 --workload /path/to/measured-workload.json \
  --warmup-workload /path/to/disjoint-warmup.json --arrival-trace /path/to/arrivals.json \
  --concurrency 8 --stagger-seconds 0 --context-limit 40960 --output /path/to/result.json
```

The trace contains every `(session, turn, offered offset)` in turn-major order,
the rate, process and seed, a workload digest, and its own digest. The first
arrival is at zero; subsequent Poisson intervals are exponential with mean
`1 / request_rate`. Replay consumes the stored offsets without resampling.
Changed workloads, damaged traces, and conflicting arrival flags are rejected
before contacting the server. Trace creation never overwrites an existing file.
The rate in this example is illustrative, not a calibrated comparison rate.

`prepare.py multi-trials --rates ...` writes one shared arrival artifact per
checkpoint/context/load/repetition, outside the engine-mode loop. Arrival seeds
are `42 + repetition`; workload seeds are separately fixed by `prepare.py multi-workloads`.
The campaign hashes trace files along with workloads and launch plans, and each
result embeds the trace for later comparison. Direct `--request-rate` runs
also embed their generated trace; save it in advance for engine comparisons.

Reproducibility applies to the offered workload and schedule, not wall-clock
dispatch, generated answers, or latency. Previous-turn completion and think time
can delay sends; actual send/finish times and causal delays are recorded separately.
Output-dependent retokenization drift still needs the geometry checks below.

Optional offered-rate mode uses `--request-rate QPS --arrival-process poisson
--arrival-seed SEED --stagger-seconds 0`, with `--concurrency` at least the number
of sessions (eight for the matrix). The server's scheduling limit remains
separate. A fixed turn-major trace offers every session's first turn, then every
second turn, and so on. A request is dispatched no earlier than both its offered
deadline and its previous response completion plus think time. Results preserve
the exact offered trace, causal delay, offered-to-send lag and offered-to-completion
time. They do not claim the target QPS was achieved when session dependencies
delay it. Warmup stays closed-loop and is excluded. Omitting `--request-rate`
preserves the existing fixed-lane closed-loop behavior.

`multi_turn.py` supports a closed-loop workload, inspired by the configurable
prefix/user/output lengths of vLLM's `benchmarks/multi_turn` generator. It uses
our same HTTP sender and timing definitions for all engines; no engine changes,
forced-token generation, or live tools are needed. It currently uses raw text
`/v1/completions`, not chat templates. The synthetic text is a sizing workload,
not a model-quality or actual agent-task evaluation.

Prepare one immutable manifest per tokenizer, and share it across engines:

```bash
.venv/bin/python scripts/serving_benchmark/multi_turn.py prepare \
  --tokenizer /path/to/models/Llama-3.2-3B \
  --sessions 16 --turns 4 --initial-tokens 8192 \
  --suffix-tokens 256 --output-tokens 1024 --think-seconds 0.5 \
  --seed 42 --output /path/to/results/multi-turn/llama3-3b-workload.json
```

Initial prompt length includes tokenizer special tokens. The suffix target
counts only the synthetic text; the two separating newlines are additional
and their actual standalone token count is stored in the manifest. Each
session in this example starts with distinct synthetic text and no deliberate
shared system prefix. The multi-session-replay preset above adds a shared prefix.
Text generation verifies encode/decode lengths
and uses a local RNG; it does not run the model or download a dataset.

The following requires a server actually configured and qualified for the
specified context length. A 2,048-token benchmark deployment is not
sufficient. Validate a prepared workload without contacting a server by adding
`--validate-only`:

```bash
.venv/bin/python scripts/serving_benchmark/multi_turn.py run \
  --tokenizer /path/to/models/Llama-3.2-3B \
  --workload /path/to/results/multi-turn/llama3-3b-workload.json \
  --base-url http://127.0.0.1:8000 --model llama3-3b \
  --engine-label vosti-padded-graph \
  --concurrency 4 --context-limit 16384 --stagger-seconds 0.1 \
  --output /path/to/results/multi-turn/vosti-c4.json
```

Concurrency limits sessions, with at most one outstanding turn per session.
Each lane processes a fixed subset of the manifest's sessions. After a response
and a fixed think/tool delay, it submits the previous prompt, the server's own
generated answer, and the next synthetic suffix. Finished sessions are replaced
until the fixed manifest is exhausted. Different engines may reach later turns
at different wall-clock times; the total planned work is unchanged.

No retries, history truncation, or answer substitution occur. A failed turn or
actual context overflow stops that session; other sessions finish and all
failures/unattempted turns are retained. `ignore_eos=true` plus `max_tokens`
controls output length, but server usage must confirm it. Missing usage,
short output, invalid reported cache counts, or server/client input-count
disagreement fails the turn. Missing cache counters remain unknown, not zero.

The vosti HTTP wrapper reports `usage.prompt_tokens_details.cached_tokens` from
the scheduler's initially reused full KV pages, using the compiled page size.
Engine snapshots each selected request's reuse immediately after scheduling,
before forward/commit can remove its residency. Both the HTTP wrapper and native
benchmark latch the first snapshot per request. This includes non-final prefill
chunks and first-step completion, and does not count later chunks' self-reuse
as an initial cache hit. The snapshot is output-only telemetry: scheduling and
model execution do not consume it. Missing/invalid observations remain unknown;
zero means an observed cold allocation. Streaming responses require
`stream_options.include_usage`.

The result includes generated text (which may contain user-provided data if
manifests are edited), supplemental output token IDs, prompt/token-ID hashes,
per-turn actual and nominal context lengths, re-tokenized output lengths, and
the retained previous-input token-prefix length. Re-tokenization may change
lengths or prefix boundaries; do not assume equal output budgets imply exactly
equal subsequent inputs. Cached-token fraction is token-weighted over only
requests with reported cache counts, with observation coverage alongside it.

Report first/cold and follow-up turns separately. These labels refer to session
position, not verified misses/hits. All throughput denominators are the full
lifecycle duration, including think time, client processing and drain; subset
throughput is not a separately measured steady-state capacity. TTFT observation
coverage is reported, since skipped special tokens can produce no visible text.

Compare actual geometry before interpreting timing differences:

```bash
.venv/bin/python scripts/serving_benchmark/multi_turn.py compare \
  --left /path/to/results/multi-turn/vosti-c4.json \
  --right /path/to/results/multi-turn/vllm-c4.json \
  --output /path/to/results/multi-turn/comparison-c4.json
```

The comparison reports setup and per-turn geometry differences. Save backend
startup logs, deployment identity, hardware/precision, cache capacity, and GPU
contention evidence separately. Optionally pass `run --warmup-workload PATH` to
execute a separate prepared workload in the same server before measurement.
Prepare that workload with a different seed and the intended warmup geometry;
use the same pair of manifests across engines. Both workloads are validated
before HTTP requests, and repeated initial token sequences across the two are
rejected. Partial shared prefixes are still possible: distinct manifests do
not guarantee cold caches.

Warmup requests, timing, workload digest and failure evidence are retained under
`warmup`; measured timing excludes this phase. Failed warmup skips measurement
and writes an incomplete result with unknown measured throughput. Metrics are
captured before warmup, immediately before measurement, and after measurement.
Comparison checks warmup workload identity and actual geometry too. Omitting
the option skips warmup.
No cache flush or explicit compiler/graph prewarm is performed: document cache
lifecycle and audit measured compilation/capture activity, since a finite
warmup need not exercise every shape. Running the same measured manifest twice
against a live server can warm its initial prefixes.

Begin a campaign with a short two-session/two-turn compatibility pilot on an
idle GPU. Choose context limits, concurrency, and duration from observed memory
use and token/cache accounting. The 16-session example above is too small for
reliable tail-latency estimates. Keep prefill/decode microbenchmarks separate.
