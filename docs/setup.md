# Setup

Use the [Docker walkthrough](artifact.md) to evaluate the paper artifact,
including [bundle preparation and GPU serving](artifact.md#first-gpu-run).
This page is the optional native-installation and developer command reference. The
container already has the Python environment, Rust toolchain, and Verus;
its `UV_PROJECT_ENVIRONMENT` and `PYO3_PYTHON` select `/opt/vosti-env` instead
of the native installation's `.venv`.

The Rust/Verus crate calls a uv-managed Python/PyO3 runtime for kernels,
deployment qualification, tests, and examples.

Verification and CPU tests do not need a GPU. Inference, backend qualification,
and model-level device checks require CUDA.

## Pinned inputs

| Input | Source |
|---|---|
| Rust `1.97.1` and required components | `rust-toolchain.toml` |
| Rust dependencies | `Cargo.lock` |
| Python `3.12` | `.python-version`, `pyproject.toml` |
| Python dependencies | `uv.lock` |
| Kernel source and verifier | Tracked `kernels/` tree plus family `scope.json` files |

The current Python lock resolves Torch 2.13.0 (CUDA 13.0 on Linux x86-64),
Triton 3.7.1, and Transformers 5.8.0. Treat `uv.lock`, not this prose, as
authoritative after a dependency update. FlashAttention is not a Vosti
dependency: serving attention requires the bound verified kernel and fails
closed if that binding is missing or its launch fails. Attention reference
tests may use Torch math SDPA; serving does not fall back to it.

Keep vLLM and SGLang benchmark environments separate from Vosti. The common
CUDA 13 baseline stack uses vLLM 0.28.0 and SGLang 0.5.19, both with Torch
2.13.0 and Triton 3.7.1. Matching these dependencies does not imply that the
engines compute identical results.

## Prerequisites

Engine and server examples require `MODEL_PATH` to name a local checkpoint.
Benchmark registries use `VOSTI_MODEL_ROOT`, defaulting to the ignored `models/`
directory in this checkout; subdirectories retain checkpoint names such as
`Llama-3.1-8B` and `gemma-3-4b-it`. Individual runners may also accept
`--model-path`. These settings locate weights; they do not change a sealed
kernel plan. Set `TMPDIR` to a writable short path if the system temporary
directory is unsuitable. Select an available GPU explicitly for campaigns.

Keep run outputs outside this source tree. Raw telemetry includes host/device
identities and process command lines for contention checks; review and sanitize
any evidence exported publicly, and never pass credentials in recorded commands.

Use a Linux x86-64 host with:

- `rustup` and the toolchain selected by `rust-toolchain.toml`;
- `uv`;
- a system C/C++ toolchain and `unzip` (for example, `build-essential` and
  `unzip` on Ubuntu);
- a Verus installation compatible with the pinned Rust toolchain and `vstd`;
- Git for source-identity checks; and
- optionally, an NVIDIA driver and CUDA-visible GPU for deployment and model
  tests.

Install the Rust toolchain if needed:

```bash
rustup toolchain install 1.97.1
```

The Makefile looks for Verus at:

```text
~/.local/verus/verus
~/.local/verus/cargo-verus
```

Override both paths when using another installation:

```bash
make verify VERUS=/path/to/verus CARGO_VERUS=/path/to/cargo-verus
```

The currently exercised installation reports Verus
`0.2026.08.23.fbbbbcf` with toolchain
`1.97.1-x86_64-unknown-linux-gnu`. Check yours with:

```bash
/path/to/verus --version
```

Download `verus-0.2026.08.23.fbbbbcf-x86-linux.zip` from the
[pinned Verus release](https://github.com/verus-lang/verus/releases/tag/release%2F0.2026.08.23.fbbbbcf).
Extract the complete archive, including Z3 and the libraries, into a new directory
outside the repository:

```bash
unzip /path/to/verus-0.2026.08.23.fbbbbcf-x86-linux.zip -d /path/to/new/verus-install
export VERUS=/path/to/new/verus-install/verus-x86-linux/verus
export CARGO_VERUS=/path/to/new/verus-install/verus-x86-linux/cargo-verus
"$VERUS" --version
```

Keep these exports in the shell used for the Make commands. Do not substitute
the latest Verus release without updating and validating the pinned inputs.
See the [upstream installation guide](https://github.com/verus-lang/verus/blob/main/INSTALL.md)
for platform requirements and source-build instructions.

## Provision the repository

Create the locked Python environment:

```bash
make setup
```

This runs `uv sync --locked`. Use the Makefile targets for PyO3 tests because
they also expose the uv interpreter's shared-library directory correctly:

```bash
make test
```

The repository has one Python project and one root `.venv/`, shared by the
engine integration and `kernels/`. Runtime dependencies are under
`project.dependencies`; the `verification` group supplies NetworkX and Z3,
and `dev` supplies pytest. Both groups are installed by default, so `make
setup` provisions the complete development environment. The root lockfile
also fixes the versions used by kernel verification. Do not create a separate
environment inside `kernels/`.

The CPU Python suite needs loopback TCP sockets for a local HTTP fixture.
Its preflight fails if sockets are denied. Pytest dumps thread stacks after
120 seconds in a test to help diagnose stalls; this does not stop the test.

Do not use an arbitrary system Python. The project requires Python 3.12, and
the Makefile selects `.venv/bin/python` for PyO3 builds once the environment
exists.

The kernel verifier and Triton kernels are normal tracked files under
`kernels/`; a regular clone includes them. Check their source identity with:

```bash
uv run --locked python scripts/audit/check_kernel_sources.py
```

An alternate source directory can be selected with `KERNELS_DIR`. Its declared
kernel and support files must match the source digests in every model-family
scope. This check does not require Git or reject unrelated documentation edits.
Kernel verification and generated-interface freshness are separate checks;
matching source hashes alone does not establish a proof.

Deployment preparation and runtime startup check the same declared files and
can operate from a source export without Git. Deployment schema
v5 omits repository revisions from compatibility and plan identity; older
bundles must be regenerated. See [deployment qualification](deployment.md).

## CPU-side checks

The root Makefile is the only Make entry point; `make help` lists its targets.
The Python test gate regenerates the compiler-resolved specification dependency
graph in `target/effort/`; generated accounting snapshots are not tracked.

```bash
make verify-engine   # whole-crate Verus and emitted-VIR trust check
make verify-kernels  # all deployed kernel proofs and generated freshness
make verify          # both of the above
make test            # all CPU regression suites
make check           # complete CPU verification, tests, audits, diagnostics
```

### Test suites

| Command | Scope | External inputs |
|---|---|---|
| `make test` | Rust/PyO3 and shared HTTP tests, all example builds, Python and kernel regressions | Locked environment and Verus |
| `uv run --locked python scripts/project.py test --suite rust` | Rust/PyO3 and shared HTTP tests; build all family examples | Locked environment and Verus |
| `uv run --locked python scripts/project.py test --suite python` | Runtime, deployment, and experiment-tool regressions | Verus; local loopback socket permission |
| `uv run --locked python scripts/project.py test --suite kernels` | Kernel verifier and CPU kernel regressions | Verus; CUDA-only tests skip |
| `uv run --locked python scripts/project.py test --suite checkpoint` | Installed Gemma4-12B/31B configurations against admitted profiles | `VOSTI_MODEL_ROOT`; at least one checkpoint required |
| `make test-gpu FAMILY=<family>` | One family's device and causal-confinement checks | Explicit GPU, checkpoint, sealed bundle |

`python/tests/` is the default Python CPU suite; `python/integration/` holds
explicit local-checkpoint checks. The checkpoint command checks every installed
Gemma4 checkpoint among the two named sizes; it does not load weights or run
inference. CPU gates hide CUDA devices. Missing Verus fails compiler-backed
regression gates; missing loopback socket access fails the HTTP fixture's
environment preflight.

The Rust suite enables `openai-server` and tests the shared HTTP wrapper once
through `tests/openai_server.rs`. It also compiles every engine and server
example. These tests do not load checkpoints or start inference servers.

Experiment-tool tests do not launch benchmark campaigns. GPU experiments have
separate commands and retained outputs: see
[determinism reproduction](../scripts/determinism_tests/README.md),
[serving benchmarks](../scripts/serving_benchmark/README.md), and
[agent benchmarks](../scripts/agent_benchmark/README.md).

`make verify-engine` uses a dedicated Cargo target directory, clears this
package's cached artifacts and prior VIR, verifies the whole crate, and checks
the emitted typed VIR against the live source trust manifest. The fixed
`CACHEDIR.TAG` signature marks disposable build data; it is not a proof or
source-identity hash.

`make verify-kernels` checks the scoped kernel sources and runs all families'
deployed certificate cases and generated-artifact freshness checks.
`make check` combines both verification gates with example verification,
CPU tests, claim linkage, the independent Qwen3 logical race pass, the bounded
suffix falsifier, and TCB-inventory freshness. Source-effort accounting is
optional; its regeneration and freshness commands live in
[scripts/effort](../scripts/effort/README.md). The diagnostics are not additional
premises of the relational proof; the suffix pass is not an unbounded proof.
A successful `python3 scripts/audit/claim_ledger.py` alone establishes only
source/digest/consumer linkage; it does not rerun either prover.

### Focused checks and regeneration

```bash
uv run --locked python scripts/project.py examples
uv run --locked python scripts/project.py architecture
```

`VERUS_CLAIM_TARGET_DIR` and `VERUS_CLAIM_VIR_LOG_DIR` may select dedicated
build/log directories. The engine gate clears this package's cached artifacts
because forwarded Verus flags may not participate in Cargo's fingerprint.

To qualify a single family's kernel inventory without regenerating shared files:

```sh
uv run --locked python scripts/audit/check_kernel_sources.py
uv run --locked python scripts/verification/verify_kernel_contracts.py --family gemma4
```

Family-filtered verification is partial coverage. Shared interface regeneration
always uses the full inventory:

```sh
make generate
python3 scripts/audit/claim_ledger.py
python3 scripts/audit/tcb.py
python3 scripts/audit/tcb.py --check
```

`generate` verifies the kernel cases before writing shared interfaces and the
claim ledger. It does not replace whole-crate Verus verification or automatically
approve changed trusted declarations. Review generated changes and audit records,
then run `make check`. Declared source digests and generated interfaces must match
the current files. TCB regeneration is separate, after source changes.

### Independent diagnostics

```sh
uv run --locked python scripts/project.py diagnostics
```

This runs `verify_qwen3_kernel_races.py` for logical write disjointness and
`kernels/scripts/verify_suffix_independence.py` for bounded suffix-independence
checks. They are included in `make check`, but are independent of the relational
proof. Neither proves physical memory safety; the suffix check covers only
its tested shapes.

`python3 scripts/audit/tcb.py --time` measures full-gate wall time, not core
verifier time alone. `make clean` removes Cargo artifacts; it preserves the
shared `.venv`, checkpoints, deployment bundles, and external run results.


## CUDA deployment and model checks

Each supported family needs:

- a local text checkpoint;
- the kernel/support sources declared by the family's scope;
- a CUDA-visible device with enough memory for that checkpoint; and
- a new or empty directory for a device- and software-qualified deployment
  bundle.

Preparation records the model configuration, bound sources, launch plan, and
GPU/software environment. It verifies the selected kernels, runs backend
probes, and seals the result. See [bundle reuse](deployment.md#reusing-a-bundle)
for which changes require preparation again.

All families use the same commands. This example uses `gemma4`; substitute
`qwen3`, `gemma3`, or `llama3` together with that family's checkpoint and bundle:

```bash
CUDA_VISIBLE_DEVICES=3 \
  uv run --locked python scripts/prepare_deployment.py --family gemma4 /path/to/model \
  --output /path/to/new-bundle

CUDA_VISIBLE_DEVICES=3 MODEL_PATH=/path/to/model CUDA_DEVICE=cuda:0 \
VOSTI_DEPLOYMENT_BUNDLE=/path/to/new-bundle \
  uv run --locked python scripts/launch.py --kind engine --family gemma4

CUDA_VISIBLE_DEVICES=3 MODEL_PATH=/path/to/model CUDA_DEVICE=cuda:0 \
VOSTI_DEPLOYMENT_BUNDLE=/path/to/new-bundle \
  make test-gpu FAMILY=gemma4
```

These commands select physical GPU 3 as process-local `cuda:0`; choose an idle
device appropriate to the host. Each family needs its own checkpoint and bundle.
There is no implicit all-family GPU run. Preparation uses the family candidate
builder and shared prove/probe/seal pipeline. Launching requires the existing
bundle and does not select or prepare a new configuration. Example arguments
follow `--`.

Each `test-gpu` invocation runs KV-store effects, engine smoke, reference
comparison, and causal-confinement checks sequentially. For focused checks,
use `scripts/checks/kv_store_effect.py`, `scripts/checks/engine.py`, or
`scripts/checks/reference.py` with the positional family and the same environment.
See [deployment qualification](deployment.md) for what these tests cover.
Performance measurements require an idle GPU; the
commands do not reserve it.

### HTTP serving

The server keeps one qualified Engine alive and admits requests between
verified steps. Start it with the same checkpoint and sealed bundle:

```bash
CUDA_VISIBLE_DEVICES=3 MODEL_PATH=/path/to/model CUDA_DEVICE=cuda:0 \
VOSTI_DEPLOYMENT_BUNDLE=/path/to/new-bundle \
VOSTI_SERVED_MODEL_NAME=gemma4 VOSTI_CUDA_GRAPH=1 \
  uv run --locked python scripts/launch.py --kind server --family gemma4
```

`--kind server` uses a release build; `--kind engine` keeps the example's
development build mode. The server exposes `/health`, `/metrics`, `/v1/models`,
`/v1/completions`, and `/v1/chat/completions`. Completion endpoints support JSON
and streaming SSE. Unsupported sampling features are rejected.

The HTTP wrapper is intended for trusted local experiments. It binds to
`127.0.0.1` by default and provides no authentication or TLS. Keep it on loopback
or behind an access-controlled proxy; do not expose it directly to untrusted
clients. Determinism verification does not establish HTTP-service security.

See [serving benchmarks](../scripts/serving_benchmark/README.md) for the common
cross-engine clients, reproducible workloads, and timing definitions.

## Optional baseline environments

Campaigns expect a stack directory containing `vosti/.venv`, `vllm/.venv`, and
`sglang/.venv`. Create a new stack outside the repository. From the repository
root, reuse the root Vosti environment and install the baselines separately:

```bash
export VOSTI_BENCH_STACK=/path/to/new/engine-stack
mkdir -p "$VOSTI_BENCH_STACK/vosti"
ln -s "$PWD/.venv" "$VOSTI_BENCH_STACK/vosti/.venv"
uv venv --python 3.12 "$VOSTI_BENCH_STACK/vllm/.venv"
uv venv --python 3.12 "$VOSTI_BENCH_STACK/sglang/.venv"
uv pip install --python "$VOSTI_BENCH_STACK/vllm/.venv/bin/python" \
  'vllm==0.28.0' 'torch==2.13.0' 'triton==3.7.1' 'cuda-bindings==13.0.3'
uv pip install --python "$VOSTI_BENCH_STACK/sglang/.venv/bin/python" \
  'sglang==0.5.19' 'torch==2.13.0' 'triton==3.7.1' 'cuda-bindings==13.0.3'
uv pip freeze --python "$VOSTI_BENCH_STACK/vllm/.venv/bin/python" > "$VOSTI_BENCH_STACK/vllm/requirements.txt"
uv pip freeze --python "$VOSTI_BENCH_STACK/sglang/.venv/bin/python" > "$VOSTI_BENCH_STACK/sglang/requirements.txt"
```

These commands pin the main numerical stack, not every transitive dependency.
Retain the resolved requirements with the campaign; use `uv pip sync --python
ENV/bin/python requirements.txt` to recreate that package set. Historical
results require their recorded inventories, not a new resolution of these
commands. Campaigns check worker identities and record package inventories.
Run backend preflight and a small smoke before measurements; installation alone
does not qualify a backend, checkpoint, or GPU.

The [agent benchmark](../scripts/agent_benchmark/README.md#environment) has its
own optional environment. Neither baseline nor agent packages belong in the
root Vosti environment.

## Updating dependencies

Use the lockfiles for normal development. To update dependencies:

- Rust: edit `Cargo.toml`, then use `cargo update`;
- Python: edit `pyproject.toml`, then regenerate `uv.lock`;
- kernel verifier: review changes under `kernels/`, update the shared kernel
  catalog's source digests if kernel/support files changed, and regenerate
  the import records in the same reviewed branch.

After an update, run the complete CPU claim gate. Regenerate and retest each
family deployment on every environment intended for use:

```bash
make check
```

Follow the surrounding formatting when editing Rust/Verus. Whole-tree
`cargo fmt --check` is not an acceptance gate: the tree is not uniformly
rustfmt-formatted. Avoid bulk formatting, especially generated contracts and
source-attested boundaries. A formatting-only change there can require
regenerating digests and audit records, followed by verification.

Update `README.md` when supported features change and `docs/verification.md`
for changes to the claim, trust boundary, or open gaps.
