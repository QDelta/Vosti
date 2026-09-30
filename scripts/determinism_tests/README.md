# GPU logit-relation experiments

Two suites compare raw logits within each engine. Use the
[expanded report](#expanded-report-suite) for the paper's determinism table,
or the [core matrix](#core-matrix) for smaller tests across more models.
Both record bitwise equality and changes in the highest-ranked token.
Logit collection affects timing, so use the separate serving suite for performance.

## Expanded report suite

The report suite compares execution variations within each engine, not logits
between different engines. It tests batch composition, chunked prefill,
prefill/decode, and prefix reuse. `report_plan.py` defines the seeded inputs;
`report_suite.py` executes them. `report_campaign.py` manages baseline rows,
and `native_report.py prepare` qualifies and runs the native adapter.

The default report uses Llama-3.1-8B and text-only Gemma3-4B, with seven modes
per model: Vosti qualified padded CUDA graphs; vLLM default, invariant FA3,
and invariant Triton; SGLang default, deterministic FA3, and deterministic
Triton. Default modes retain engine-selected backends and record the selection.
Model locations, worker environments, and device selection are execution inputs.

### Relations and evidence

- Batch: vary batch sizes, order, and companions; compare with isolated
  references and fresh-process singleton checks.
- Chunk: vary prefill budgets across page, chunk, long-context, and sliding
  window boundaries. Verify actual chunk execution, not just a requested flag.
- Prefill/decode: compare each prediction with teacher-forced prefill.
  For output `y[i]`, teacher input is `prompt + y[:i]`, excluding `y[i]`.
  Each engine supplies its own generated sequence.
- Prefix reuse: compare cold references with observed partial/full hits,
  shared prefixes followed by different suffixes, and generated-token prefixes.
  Donor and reuse requests must share one engine lifetime.

A pass requires every required comparison to be present, valid, finite, and
byte-identical. Missing or invalid evidence is not a numerical mismatch.
Same-argmax results are recorded separately from bitwise equality. Retain raw
logit rows, prediction positions, scheduler/cache witnesses, source/deployment
identities, actual backend settings, and GPU contention telemetry.

`report_audit.py --campaign RUN_DIR --output NEW_JSON` independently re-reads
raw bytes, descriptor hashes, declared coverage, and telemetry hashes, and
recalculates rank divergence. Multiple campaign directories may combine
separate baseline/native rows; duplicate executed rows are rejected. Cache
and scheduler witnesses remain instrumentation evidence, not formal proof.

### Execution and qualification

Native `engine.multi_call` keeps a qualified engine and graph overlay alive
across calls. Cold calls reclaim unused cache pages through the existing
engine interface; warm calls retain them. `native_report.py smoke` checks row
mapping, capacity, padded-cover replay, and generated-prefix reuse before full
rows run. Small smokes supplement, rather than replace, report coverage.

Correctness workers may share a GPU only under explicit identity-bound cohort
telemetry with fixed per-worker cache budgets and sufficient measured VRAM
headroom. Unregistered jobs and leftover children invalidate the run. Shared
correctness timings are not serving-performance measurements; performance
benchmarks use an exclusive GPU.

Resume requires unchanged inputs and source identities. Valid completed arms may
be reused; incomplete attempts receive fresh retry directories. Original raw
evidence is not overwritten, and backend labels must not be changed after a
run. Keep run-specific reports and corrections with the private run artifacts.

`expanded_suite.py --pilot` provides a smaller SGLang FA3 harness check;
`--request-identity-smoke` checks instrumentation mapping. Neither substitutes
for the full report suite. Passing these finite tests does not establish
determinism on untested models, inputs, or configurations.

### Extending the native report

The native orchestration entrypoint is
`python -m scripts.determinism_tests.native_report COMMAND --help`:

- `prepare`: prepare qualified deployments, smoke-check capacity, and run the
  two default native report rows once the selected GPU is idle.
- `run`: smoke-check and run one catalog checkpoint using an existing qualified
  worker, binary, and deployment.
- `smoke`: run only the adapter/capacity checks for a report checkpoint.

The `run` command executes the fixed-seed expanded report for one catalog
checkpoint, Vosti only, on an explicit GPU. Use
`python -m scripts.determinism_tests.native_report run --help` for arguments.
Execution starts with the native adapter/32K-capacity smoke.
Supply the clean qualified worker checkout, its matching binary and deployment
bundle, and a new output directory. The controller revision is recorded
separately from the worker revision; retained rows are never relabeled as new
results. `inputs.json`, `smoke/`, `suite/`, and `complete.json` retain the plan,
raw logits, schedule/cache witnesses, and per-arm GPU telemetry. The live
terminal prints a heartbeat every 30 seconds and completion/failure events.

The default 768 cache blocks cover this workload's largest eight-prompt batch
and 32K single-request cases; this is allocation capacity, not a kernel-plan
change. This command does not change the original two-model report table or
launch baseline engines.

## Core matrix

This suite observes raw logits under the same execution variations in each
serving system:

- hardware campaign labels: H200 and A100;
- models: Qwen3 dense, text-only Gemma 3/4 dense, and Llama 3 dense;
- execution configurations: Vosti qualified padded CUDA graph, vLLM fast/auto,
  vLLM batch-invariant with FlashAttention and Triton attention, SGLang
  fast/auto, and SGLang deterministic with FA3 and Triton attention;
- relations: batch versus single, chunked-prefill size, prefill versus decode,
  and cold versus warm prefix-cache execution.

`matrix.py` defaults to H200 and materializes 588 logical relations: seven
checkpoints (Gemma3 4B/12B/27B, Gemma4 31B, Llama3 3B/8B, Qwen3 8B), seven execution modes,
three fixed seeds, and four relations. There are 420 strict and 168
observational cells. `--seed 0x5EED2026` selects the first 196-cell round.
The other seeds are that value plus 10,000 and 20,000. Explicit FA modes pin
FA3; default modes let the engine choose. `run.py` also accepts
`sglang-deterministic-fa4`, outside the default matrix. Record the actual
backend and use a new campaign plan when changing it.
`run.py` executes all four relations for one
hardware/model/execution tuple. Every comparison uses the complete selected
vocabulary row represented losslessly as float32, not only the selected token
or a tolerance.

Check backend support before launching a matrix. In particular, Gemma4's
512-wide global heads need a backend that supports that geometry.

`NATIVE_CHECKPOINTS` additionally includes text-only Gemma4-12B for
`native_report.py run`. It uses the same fixed-seed expanded relations;
this native-only addition does not expand or qualify the baseline matrix.

`native_cover_smoke.py` adds a small architecture-neutral padded-cover screen.
It reuses the native multi-call observer and existing cover-witness checker,
requires two actual covered decode steps, and compares six full-vocabulary
logit pairs. Specify a checkpoint, clean worker root, native binary, sealed
deployment, worker Python, GPU index and new output directory. It does not
replace the four core relations or the broader schedule-stress suite.

The SGLang and Vosti observers are installed at process startup through the
test-only `hooks/sitecustomize.py`; they patch runtime modules without editing
installed package files. vLLM uses its public `raw_logits` logprob mode.
Observation copies logits to CPU, so this is a correctness experiment rather
than a performance benchmark.

### Relations and acceptance

Inputs are reproducible random valid token IDs from seed `0x5EED_2026`, with
the model BOS token prepended and added/special IDs excluded. Strict modes
require exact equality of shape, dtype, finiteness, and every float32 bit.
Fast/auto modes retain the same diagnostics but differences are observational.
Pass `run.py --seed INTEGER` to select another reproducible input set. The seed
is retained in the immutable inputs and is checked when resuming a suite.
Each batch-composition singleton gets its own fresh engine. The prefix-reuse
arm enables caching before engine construction, records both its initially
uncached donor and measured reuse request in that same engine, and requires a
positive cached-token count for the latter. All four chunk budgets are smaller
than the chunk-test prompt, so that relation compares genuinely chunked
prefills rather than mixing chunked and unchunked execution.

The exact full-vocabulary comparison is stronger than the
documented guarantees of some baseline modes. A mismatch is evidence about
this test relation, not by itself evidence that an engine violated its stated
contract.

Each arm records its configuration and GPU telemetry without overwriting
earlier results. `--resume` accepts a completed arm only when its configuration
and telemetry completion record match.

The `--hardware` argument labels the retained campaign; it does not discover
or validate the device model. `--gpu-index` selects a physical GPU through
`CUDA_VISIBLE_DEVICES`, after which the worker sees it as `cuda:0` and records
its device name. Confirm that the selected device matches the campaign label
before accepting a result.

### Commands

Run the 147 fresh-engine suites (588 logical relations) sequentially on GPU 3
from a clean, frozen checkout. `STACK` contains separate `vosti/.venv`,
`vllm/.venv`, and `sglang/.venv` environments. The campaign records package
inventories, source commits, all planned/unrun cells, commands, per-suite
statuses, and rank-divergence analysis. It prepares fresh Vosti deployments.
Use the [baseline environment recipe](../../docs/setup.md#optional-baseline-environments)
to provision a new stack.

```bash
.venv/bin/python -m scripts.determinism_tests.campaign \
  --stack-root /path/to/common-stack --gpu-index 3 \
  --output /path/to/results/new-campaign
```

`--resume` requires identical source, package inventories, and configuration.
Changes require a new campaign directory. Failed attempts remain separate artifacts.
Numerical mismatches do not stop other cells. A worker/configuration/telemetry
error stops the campaign for inspection rather than presenting it as a
numerical failure or repeatedly launching a broken configuration. Occupied
GPUs are waited on without signalling other users' jobs.

To continue past an inspected failure, use
`--defer-case CHECKPOINT/MODE --defer-reason 'explanation'
--defer-evidence /path/to/failure.log`. Repeat `--defer-case` for each affected
pair. The runner records the reason and evidence, preserves earlier errors,
and leaves the deferred cases unrun. Unresolved cases produce `partial.json`
instead of `complete.json`.

This driver executes the core matrix. Generated-KV reuse, long-context,
window-boundary, and graph/reclamation stress extensions are separate follow-up
campaigns; a complete core status must not be read as their completion.

`context_relations.py --profile long-context --matrix --output new-plan.json`
lists 8K/32K cases across six checkpoints, seven modes and three seeds (252
cases, two relations each). `--profile window-boundary` derives the three lengths
immediately below/at/above the configured sliding window from each applicable
checkpoint (189 cases for the current checkpoints). These are additional plans,
not completed core cells. Long-context chunk budgets are 512/2K/4K; window-boundary
budgets are 64/256/512. Every arm retains the prediction at the final prompt token.
Prefix donors and reuse share one engine; the cold reference uses a fresh engine,
and positive warm reuse plus observed cold misses are required. Missing/false
cache witnesses are invalid experiments, not numerical mismatches.

To execute a case, supply profile, length, model/path, mode, worker Python, seed,
output and (for Vosti) its engine binary and qualified bundle. `--worker-root`
can point to a frozen serving checkout; both subprocess cwd and import paths use
that checkout. Core behavior is unchanged when this option is omitted. KV pool
capacity is sized for the context without introducing a configurable page size.

For native schedule-stress arms, `engine.trace_steps=true` enables an intrusive
invocation-local step trace in a freshly built shared engine example. The trace
records scheduled request IDs in row order, emitted tokens (or null for a
chunk-only row), and cache observations. The logit observer tags the same step
identity. `step_trace.py` checks the complete mapping and indexes actual emissions
by request, without assuming a rectangular batch across decoding steps. Missing,
duplicate or inconsistent trace/row evidence is an invalid test. Ordinary core
arms keep this instrumentation disabled; enabling it requires a matching new
engine binary and qualified deployment.

A traced Vosti call may provide `arrival_steps`, one nonnegative integer per
prompt. The native prepared format has an optional third TSV column; omitted
values remain zero. Requests are submitted before the corresponding engine-loop
step, including arrivals while older requests decode. The trace must witness the
declared arrivals exactly. These are synthetic step schedules, not wall-clock
arrival rates; OpenAI serving benchmarks retain their separate QPS client.

`schedule_stress.py --matrix --output new-plan.json` lists 18 Vosti cases (six
checkpoints, three seeds), with four relations each. Eight fresh single-request
references are compared with queued batches, staggered arrivals, a smaller KV
pool, and padded-cover replay. Prompt lengths are 17/63/64/65/127/129/257/385;
each request generates 16 tokens. Nonreference arms use four active sequences
and a 128-token step budget. Staggered arrivals occur at steps
0/0/2/4/8/12/16/20; cache capacity is 128 blocks, or 32 for pressure. The padded
case primes a four-request graph before warmup. Equality requires the same
generated IDs and bitwise-equal final prediction logits.

Trace witnesses require arrivals during unfinished generation, an actual
registered prefix-page removal/replacement during the measured pressure run,
and measured graph replay (specifically cover replay for the padded case).
Prefix snapshots record physical IDs and exact provenance/token metadata, not
just hashes. Absent witnesses invalidate the corresponding experiment. Supply
`--checkpoint`, worker Python/root, binary, deployment bundle, seed and output
to execute a case after qualification. These checks test determinism around
window boundaries, not the functional correctness of the attention mask.

Print the logical matrix:

```bash
.venv/bin/python -m scripts.determinism_tests.matrix --pretty
```

Print only the H200 campaign:

```bash
.venv/bin/python -m scripts.determinism_tests.matrix --hardware h200 --pretty
```

Run one vLLM or SGLang suite (use the corresponding provisioned environment):

```bash
.venv/bin/python -m scripts.determinism_tests.run \
  --hardware h200 --gpu-index 2 \
  --model qwen3 --model-path /path/to/qwen3 \
  --execution-config sglang-deterministic-fa3 \
  --worker-python /path/to/sglang/.venv/bin/python \
  --output /path/to/new-result-directory
```

Vosti additionally requires a binary and a sealed deployment bundle created
from the same clean framework commit and hardware environment:

```bash
.venv/bin/python -m scripts.determinism_tests.run \
  --hardware h200 --gpu-index 2 \
  --model qwen3 --model-path /path/to/qwen3 \
  --execution-config vosti-padded-graph \
  --worker-python .venv/bin/python \
  --vosti-binary target/debug/examples/verus_engine_qwen3 \
  --deployment-bundle /path/to/new-qwen3-bundle \
  --output /path/to/new-vosti-result
```

For Gemma3 or Llama3, select `--model gemma3` or `--model llama3`, the matching
text checkpoint, example binary, and deployment bundle. Always pass
`--model-path`; the protocol's developer defaults are not portable inputs.
Output directories must be new unless `--resume` is used. Retained experiment
records should remain outside the repository.

### Generated-prefix qualification

Add `--generated-prefix-only` to the Vosti command above to exercise generated
KV reuse specifically instead of the four standard relation groups. This runs
three native engine lifetimes: a donor generating 128 tokens from a seeded
1,024-token prompt, a cold follow-up, and a repeated donor followed by the same
follow-up in one engine lifetime. The follow-up appends a seeded 65-token
suffix to the complete prompt/output token-ID history. It requests two outputs
but compares the full logits predicting the first output, before that follow-up
has performed decode.

Each lifetime first runs a separate, seeded graph primer: two 1,280-token
prompts generating four tokens each. This captures a larger-batch decode graph
covering both singleton shapes, so the test exercises actual padded-cover
replay. Primer inputs are bound into each arm, its observer rows have their
own phase, and its cache contents stay live; cold-hit checks must still pass.

Qualification requires byte-equal donor and follow-up logits, identical donor
output IDs across lifetimes, zero cold hits, and exactly 1,088 reused tokens.
That count covers generated KV pages while excluding the last unexecuted
sample before rounding to the compiled 64-token page size. The donor must have
used padded-cover graph replay. Missing witnesses are `invalid`, not a pass;
the retained raw rows and telemetry use the same machinery as the standard
suite. This is a correctness experiment, not a latency measurement.

### External model reference

After a strict Vosti suite passes, compare every retained row with the same
checkpoint through Transformers:

```bash
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE=cuda:0 \
  .venv/bin/python scripts/checks/reference_suite.py \
  /path/to/strict-suite/summary.json \
  --model /path/to/model \
  --output /path/to/new-reference-report.json \
  --min-reference-margin 0.1 \
  --max-logit-abs-error 0.5
```

The checker does not trust the summary's pass label alone. It rechecks the
exact required arm inventory, source/deployment agreement, input and artifact
digests, complete telemetry, cache reuse, and the byte equality of each paired
Engine row before evaluating the 11 unique contexts. It requires a clean
checker checkout, hashes every checkpoint shard, and refuses to overwrite its
output.

For a fresh multi-step forward/KV differential rather than retained schedule
rows, use the family-neutral checker directly:

```bash
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE=cuda:0 \
MODEL_PATH=/path/to/model \
VOSTI_DEPLOYMENT_BUNDLE=/path/to/matching-bundle \
  .venv/bin/python scripts/checks/reference.py llama3 \
  --max-tokens 8 --check-kv \
  --min-reference-margin 0.5 \
  --max-logit-abs-error 0.5 --max-kv-abs-error 0.5 \
  --output /path/to/new-reference-report.json
```

Replace `llama3` with another supported family and use its matching checkpoint
and deployment. These reports are bounded empirical evidence, not proof or a
new premise imported by the Engine theorems.

### Ranked-logit divergence

`rank_divergence.py` consumes retained suite summaries and verifies every raw
row against its worker-recorded SHA-256 before analysis. For each comparison,
it orders entries by descending float32 logit with ascending token ID as the
tie-breaker, then reports the first one-based rank where either the token ID or
the corresponding logit differs. The result distinguishes `logit_only`,
`token_only`, and `token_and_logit` divergence and separately flags a changed
top token and tie-affected ranks. Softmax is not materialized because it
preserves this ordering.

```bash
.venv/bin/python -m scripts.determinism_tests.rank_divergence \
  --summary run=/path/to/retained/summary.json \
  --output /path/to/new-ranked-logit-result.json
```
