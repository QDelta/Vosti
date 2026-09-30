# Live SWE-bench pilot

This benchmark runs an agent on SWE-bench tasks and measures success and latency.
Unlike the synthetic serving workloads, outputs and tool actions determine
later prompts, so different engines may perform different amounts of work.

The runner accepts an explicit model, concurrency, and qualified serving mode;
there is no model-specific launch table. Engine modes run sequentially on the
selected GPU. Agent v1.17.5 is pinned for its text-based bash protocol, which
requires no native structured tool calls. The runner uses the existing serving environments.

## Environment

Use a separate Python 3.12 environment and a local Docker daemon. From the
repository root, with a new environment path outside the checkout:

```sh
export VOSTI_AGENT_ENV=/path/to/new/agent-env
uv venv --python 3.12 "$VOSTI_AGENT_ENV"
uv pip install --python "$VOSTI_AGENT_ENV/bin/python" \
  'mini-swe-agent==1.17.5' 'swebench==5.0.2' 'datasets==5.0.1' \
  'docker==7.2.0' 'PyYAML==6.0.3' 'httpx==0.28.1' \
  'transformers==5.8.0' 'pytest==9.1.1'
uv pip freeze --python "$VOSTI_AGENT_ENV/bin/python" > "$VOSTI_AGENT_ENV/requirements.txt"
git clone --branch v1.17.5 --depth 1 https://github.com/SWE-agent/mini-swe-agent.git /path/to/new/agent-source
"$VOSTI_AGENT_ENV/bin/python" -m scripts.agent_benchmark.prepare --help
"$VOSTI_AGENT_ENV/bin/python" -m pytest -q scripts/agent_benchmark/test_protocol.py
```

Pass that checkout as `prepare --agent-source`; preparation reads its upstream
agent configuration and records its commit. Keep the resolved requirements
with the run, including transitive versions. Recreate that package set with
`uv pip sync --python ENV/bin/python requirements.txt`. Model servers use the
separate [baseline/Vosti stack](../../docs/setup.md#optional-baseline-environments),
not this agent environment. Use the agent interpreter for all commands below.
Docker image preparation may download large images; inference and grading
containers retain the isolation restrictions below.

## Workload

`prepare.py` pins the dataset revision, selects issues by seeded SHA256 order,
and separates agent-visible input from host-only grading data. It preserves the
upstream SWE-bench prompts, with 40 calls, 2048 output tokens/call, 16384 total
generated tokens/task, and a 30-minute safety timeout. EOS is enabled.
Context overflow ends the attempt without engine-specific truncation. These are pilot
budgets, not the official agent's default configuration or leaderboard scores.

Agent containers must have no host mounts, GPU access, network access, Docker
socket, or grading artifacts. Pin image digests, prepare clean base checkouts,
and keep the same CPU/memory limits across modes. Only problem statements and
base repositories are inputs: no gold patches, hints, or evaluator test patches.
Do not modify the upstream agent or engine proofs to accommodate this benchmark.

Keep failed and budget-exhausted tasks in the attempted count. Report
resolved/attempted, infrastructure errors, wall time, resolved tasks/hour,
request latency, and token/cache counts. Preserve trajectories, patches,
timing records, and source details. Grade patches separately with a unique
run ID per mode so cached grades cannot be reused for a different patch.

## Detailed performance records

Each model call saves its exact request and client-tokenized prompt, incremental
timestamped SSE events (including partial streams on error), raw output, finish
reason, server token usage/cache fields, client TTFT, request latency, and TPOT
estimate. TTFT is first nonempty text arrival; TPOT is first-to-last text arrival
divided by completion tokens minus one. These are transport-observed metrics,
not per-token GPU kernel times: a text chunk may combine multiple tokens.
Raw events allow other definitions to be computed later. Only checkpoint-owned
terminal EOS strings are removed before parsing agent actions; raw output is preserved.

Per-task records retain tool timings, all model calls, total model-request time,
wall time, and budget/error termination. Mode summaries provide mean/p50/p95/max
latencies, token throughput, and task throughput over the measured campaign.
Server counters before/after and exclusive GPU telemetry are also retained.
Setup, warmup, patch extraction, and grading are not counted as task execution;
campaign elapsed time also includes dispatch gaps and result persistence.

To select an easier subset, use `prepare --difficulty '<15 min fix>' --seed 42 --count 10`
with the same pinned `--revision` and cached `--dataset-arrow`. Filtering precedes
seeded selection; selection never depends on model success. Subsets may overlap.
`images --reuse-from OLD_PILOT` preserves image identities for overlapping cases;
retain the original source and artifacts.
Grading and reference controls cap BLAS/OpenMP threads at two to respect the
container's two-CPU and PID limits.

Inference accounting reports both the union of outstanding request intervals
(active wall time) and summed request latency (overlapping across sessions).
The sum is partitioned into time to first text, first-to-last text, trailing
stream completion, and requests with no visible text. Cached and uncached
prompt tokens are separated; missing cache statistics are not assumed zero.
Subsequent decode token counts estimate completion tokens minus one per call.
Prefill/decode client rate estimates use summed TTFT/generation intervals;
these include scheduling/transport and are not GPU-only phase throughput.
Campaign throughput and output tokens per active request wall second are
reported separately. No synchronizing GPU profiler is enabled in timed runs.

## Qualified serving replay

`python -m scripts.agent_benchmark.campaign --previous-pilot OLD
--serving-campaign QUALIFIED_SERVING --mode MODE --output NEW`
reuses the exact old issues, agent configuration and image identities against
one mode from a completed serving campaign. It checks source, packages,
binaries and telemetry receipts (plus deployment receipts for native serving);
archives the harness and records an execution lock; then runs inference and
official grading sequentially. It preserves prior concurrency unless explicitly
overridden with `--concurrency`. Capacity flags are adjusted consistently for
each engine without changing the qualified attention backend or mode. A chat
warmup includes a synthetic system message and checks client/server prompt
token counts before measured tasks. Baseline launches use a text-only adapter
prepended to the unmodified checkpoint template: string messages pass through;
OpenAI text-part arrays are joined without extra separators before rendering.
This avoids model-visible whitespace changes from content-format autodetection.
Non-text input is rejected. The adapter is checked against retained prompt
token IDs and archived alongside the generated template. Each
invocation requires a fresh output directory; there are no automatic retries.
Infrastructure-error runs remain diagnostic artifacts, not model-quality scores.
For SGLang, `--skip-server-warmup` disables its built-in image request for VLM
checkpoints. Model initialization and CUDA graph capture still run, followed by
the harness's mandatory text-chat warmup and token-count check before timing.
Run once per selected mode using the campaign entry point above.

For a new subset, `prepare.py` requires `--model` and one or more `--mode` flags;
then pin containers with `images.py`.

Reports take the model and concurrency from the plan. Keep the executed harness
archive alongside each run's execution lock. Use the same pinned tasks and
budgets across modes; keep scores and qualification reports outside the source tree.

## Durable background execution

For a comparison that must survive terminal closure, use
`python -m scripts.agent_benchmark.launch --previous-pilot OLD --output NEW
--qualified-mode CAMPAIGN MODE` (repeat the last pair for each mode).
The launcher returns after recording a detached supervisor PID/start identity.
Shared process/timing helpers in `scripts/common/` are included in the harness archive.
The supervisor and each sequential campaign write directly to regular log files;
neither depends on a terminal reader or `tee` pipe. Optional log viewers can be
closed safely. Each mode writes start/exit receipts. `complete.json` requires all
modes to exit successfully with inference, grading and report artifacts present;
`failed.json` records failures and later modes remain unrun. Source hashes are
checked before each mode, and each campaign retains its finer-grained locks.
Failed runs are not automatically retried or resumed.

## Tests

Dependency-free launch and timing regressions run in the normal CPU suite:

```sh
uv run --locked python -m pytest -q python/tests/test_agent_launch.py python/tests/test_benchmark_common.py
```

The remaining `test_protocol.py` checks import the optional `minisweagent` stack.
Run them explicitly in the configured agent environment; the core test suite
does not install or require that optional dependency.
