# Evaluating the paper artifact

This guide assumes you have read *Vosti: Specifying, Implementing, and
Verifying Deterministic LLM Inference* (link forthcoming). Start with the CPU
checks, then try one checkpoint before expanding to the paper's matrices.

| Paper result | Artifact entry point | Needs a GPU? |
| --- | --- | --- |
| Engine determinism theorem | `make verify-engine`; statement in [`src/spec.rs`](../src/spec.rs), instantiation in [`src/proof.rs`](../src/proof.rs) | No |
| Kernel relational properties | `make verify-kernels`; [annotations and verifier](../kernels/README.md) | No |
| Logit comparisons | [Expanded determinism suite](../scripts/determinism_tests/README.md#expanded-report-suite) | Yes |
| Prefill/decode throughput | [Aligned controlled phases](../scripts/serving_benchmark/aligned/README.md) | Yes |
| Session completion time | [Multi-session replay](../scripts/serving_benchmark/README.md#multi-session-replay) | Yes |
| Source effort | [Accounting tools](../scripts/effort/README.md) | No |

The [verification guide](verification.md) explains the assumptions behind the proofs.

## Environment

Use Linux x86-64, Docker, and a fresh Git clone. The [Dockerfile](../Dockerfile)
installs Rust, Verus with Z3, and the locked Python dependencies, including
Torch and Triton. It is an environment image: the commands below mount the
source checkout. Checkpoints and results stay outside the image.
Building the image and downloading Cargo dependencies need Internet access.

From the checkout, as your normal non-root user:

```bash
docker build --build-arg VOSTI_UID="$(id -u)" --build-arg VOSTI_GID="$(id -g)" \
  -t vosti-artifact .
docker run --rm -it --mount type=bind,src="$PWD",dst=/workspace \
  vosti-artifact
```

Commands inside the container run from `/workspace`. Its Python environment
lives at `/opt/vosti-env`, so it does not use a host `.venv`. Build outputs
under `target/` persist in the checkout. Use a dedicated clone to avoid mixing
host and container build products. If you change the lockfiles or toolchain,
rebuild the image. Base-image and OS-package updates can change image contents;
record the image ID with results, as well as the Git revision and package versions.

Allow space for the image and build caches as well as model weights and results.
The CPU checks still install the CUDA-enabled Torch packages from the lockfile,
but do not need a GPU or NVIDIA driver. Native setup is documented in
[setup.md](setup.md).

## Check the proofs

Inside the container:

```bash
make verify-engine
make verify-kernels
make test
```

The engine gate runs whole-crate Verus verification and checks emitted trust
declarations. Expect a verification summary with zero errors and a successful
trust-manifest check. The kernel gate checks every deployed static case and
generated-interface freshness; expect `[QUALIFIED]` entries followed by
`[PASS]`. `make test` runs the CPU regressions and builds the examples; GPU-only
kernel tests may skip. These commands must exit successfully. `make check`
also runs the remaining audits and independent diagnostics.

Verification time depends on CPU resources, solver behavior, and build caches.
The paper's core verifier times exclude compilation, orchestration, and audits;
timing `make verify` measures a broader operation. Source counts can also
change as the artifact is maintained. Preserve the revision and full logs
when comparing either measurement with the paper.

## First GPU run

Generate the bundle and run the model inside the Docker image built above.
You do not need a Python environment, Rust, or Verus installed on the host.

The paper uses an H200 NVL. Another NVIDIA GPU needs sufficient memory for
weights, KV cache, and workspace, plus support for the selected kernels. The
locked CUDA 13 stack needs a compatible host driver; NVIDIA lists driver 580
or newer as the CUDA 13 family minimum, with additional limitations for JIT
code. See [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. The image does not install or replace the host driver.

Obtain a supported checkpoint under its publisher's license. The examples use
text-only Gemma3-4B in a local `gemma-3-4b-it` directory containing the config,
tokenizer, and weights. Weights are not included in this repository or image.
Start on an otherwise idle GPU; the commands do not reserve it.

### Start the GPU container

Exit the CPU container. From the repository root on the host, substitute your
model and results paths:

```bash
mkdir -p /path/to/results
docker run --rm -it --name vosti-artifact-run --gpus device=0 --shm-size=8g \
  --mount type=bind,src="$PWD",dst=/workspace \
  --mount type=bind,src=/path/to/models,dst=/models,readonly \
  --mount type=bind,src=/path/to/results,dst=/results \
  -e VOSTI_MODEL_ROOT=/models vosti-artifact
```

### Generate the deployment bundle

Run the following inside the container, from `/workspace`. The selected GPU
appears as CUDA device 0. Use a new or empty bundle output directory:

```bash
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE=cuda:0
export MODEL_PATH=/models/gemma-3-4b-it
export VOSTI_DEPLOYMENT_BUNDLE=/results/gemma3-4b-deployment
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
uv run --locked python scripts/prepare_deployment.py --family gemma3 "$MODEL_PATH" \
  --output "$VOSTI_DEPLOYMENT_BUNDLE"
make test-gpu FAMILY=gemma3
```

Preparation verifies the selected kernel cases, runs device probes, and writes
a deployment bundle. Serving checks that this bundle matches its model
configuration, kernel sources, GPU, and software environment.
The recorded configuration hash comes from your mounted checkpoint's
`config.json`. The bundle is saved under the mounted `/results` directory,
so it survives container removal.

On a different GPU model, prepare a new bundle on that GPU. Different weights,
such as a fine-tuned checkpoint, can reuse a bundle if the configuration and
weight layout still match; the bundle does not hash weight values. Matching
tensor shapes alone is insufficient. See [bundle reuse](deployment.md#reusing-a-bundle)
for the exact checks. Neither the source release nor the image includes a
prequalified deployment bundle.

Changing the GPU or weight values does not require new engine proofs. The
theorem is parameterized by the model and kernel plan. Preparation verifies
the selected kernel configurations and tests their backend assumptions on
your device; it does not prove compiler or GPU correctness.

The device checks test KV writes, engine execution, reference agreement, and
causal confinement. Run them with your checkpoint even when reusing a bundle.
They are a quick check before the larger experiments below.

### Start the server

Continue in the same container shell so the model and bundle variables remain set:

```bash
VOSTI_SERVED_MODEL_NAME=gemma3-4b VOSTI_CUDA_GRAPH=1 \
VOSTI_MAX_SEQS=4 VOSTI_MAX_BATCHED_TOKENS=4096 \
VOSTI_NUM_BLOCKS=2560 VOSTI_MAX_MODEL_LEN=32768 \
  uv run --locked python scripts/launch.py --kind server --family gemma3
```

These are the replay's capacity settings, not a guarantee that it fits every
GPU. A smaller cache or workload may be needed on a smaller device; report
that change. The server stays on container loopback, with no authentication
or TLS. In a second host terminal, enter the same container:

```bash
docker exec -it vosti-artifact-run bash
curl --fail http://127.0.0.1:8000/health
```

## Try a serving workload

In that second container shell, prepare a short synthetic workload and send
it to the running server:

```bash
uv run --locked python scripts/serving_benchmark/multi_turn.py prepare \
  --tokenizer /models/gemma-3-4b-it --sessions 4 --turns 2 \
  --initial-tokens 1024 --suffix-tokens 256 --suffix-includes-separator \
  --output-tokens 128 --think-seconds 0.5 --seed 42 \
  --output /results/smoke-workload.json
uv run --locked python scripts/serving_benchmark/multi_turn.py run \
  --tokenizer /models/gemma-3-4b-it --workload /results/smoke-workload.json \
  --base-url http://127.0.0.1:8000 --model gemma3-4b \
  --engine-label vosti-padded-graph --concurrency 4 --context-limit 32768 \
  --request-rate 2 --arrival-process poisson --arrival-seed 42 --stagger-seconds 0 \
  --output /results/smoke-result.json
```

Expect eight successful requests with the requested output lengths and a
complete result. The JSON retains per-request timings, generated text, token
counts, cache observations, and aggregate latency/throughput. This short run
includes first-use costs and is only a smoke test. Performance measurements
need separate warmup and a fresh server for each experiment.

## Repeat the paper experiments

Use the suite guides linked below for their exact commands and evidence format.
The optional campaign runners also record source and environment details and
monitor GPU contention. Use native setup for those runners: their monitors
need host-visible process IDs. The quick start and direct replay do not need them.

### Determinism

Use the [expanded report suite](../scripts/determinism_tests/README.md#expanded-report-suite),
not the smaller core matrix. It compares full logit rows within one engine
under changes to batching, chunking, prefill/decode, and prefix reuse. The
fixed report seed is `0x5EED2026`. Each complete model/mode row contains 784
comparisons: 296 batch, 90 chunk, 384 prefill/decode, and 14 prefix-reuse pairs.

The cross-engine table uses Llama3.1-8B and Gemma3-4B. Vosti's extended coverage
adds Llama3.2-3B, Gemma3-12B/27B, and Gemma4-12B/31B. Start with a single
Vosti row using `native_report.py run`; its guide describes the binary,
checkpoint, and deployment inputs. The baseline report runner uses separate
[vLLM/SGLang environments](setup.md#optional-baseline-environments).

Report matched/valid pairs, with missing or invalid comparisons separately.
Passing a finite test suite does not prove determinism; a mismatch may still be
acceptable for an engine's intended use. The tests compare executions within
one engine and configuration, not outputs across engines or GPUs.

### Performance

The paper's performance models are Gemma3-4B, Llama3.1-8B, Gemma3-27B, and
Gemma4-31B. The seven modes are Vosti with padded CUDA graphs; vLLM default,
invariant FA3, and invariant Triton; and SGLang default, deterministic FA3,
and deterministic Triton. Default backends are engine-selected and must be
recorded. Gemma4 uses vLLM's local-FA3/global-Triton invariant variant;
SGLang deterministic FA3 is unsupported for that model.

The [multi-session replay preset](../scripts/serving_benchmark/README.md#multi-session-replay)
creates the main workload: four sessions, six turns, 8,193 shared plus 8,192
distinct initial tokens, 256 appended tokens and 768 output tokens per turn.
It saves seeded Poisson request arrivals at 2 requests/s; session dependencies
and the 0.5-second inter-turn delay can reduce achieved QPS. Ordered cache and
decode-shape warmups are excluded from measurement. Use the same saved inputs
and arrivals for all engines, and retain their actual generated lengths.
The [direct replay recipe](../scripts/serving_benchmark/README.md#replay-against-a-running-server)
runs this workload against an already running server without campaign monitoring.

For prefill/decode curves, use the [aligned phase adapter](../scripts/serving_benchmark/aligned/README.md):
batch size four, cached contexts 0/4K/8K/12K/16K, 256 new prefill tokens per
request, and 128 output tokens for decode. It checks the physical batch and
actual cached length. Four concurrent HTTP requests alone do not establish
that geometry. These engine-side intervals differ from client-observed TTFT
and TPOT; keep them separate.

Use an idle GPU and retain warmup separately. Report actual input/output tokens,
observed cache reuse, TTFT, TPOT, and completion time. Compare engines on the
same machine with the same workloads and numerical dependencies. If memory or
backend support requires a smaller experiment, label it as an adapted run;
do not silently change one engine's workload or mark unsupported cells passed.

## Interpreting a rerun

Docker supplies the software environment; the host still supplies the GPU and
driver. Performance and baseline mismatch counts may differ from the paper.
Keep the model revision, image ID, package versions, GPU/driver details,
deployment, workload seed, and raw results with each run. Record any smaller
workloads needed to fit your GPU.
