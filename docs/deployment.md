# Deployment-time kernel qualification

Serving uses two phases:

1. preparation selects kernel configurations, verifies their contracts, and
   tests the backend on the target GPU;
2. startup checks the resulting bundle and fixes those configurations for
   the lifetime of the engine.

## Phase 1: prepare a sealed bundle

All families use the same deployment compiler. See
[setup](setup.md#cuda-deployment-and-model-checks) for preparation and launch commands.

The model directory supplies the configuration. The compiler observes hardware,
Torch/Triton/CUDA, driver, and visible-device facts directly. The output
directory must be new or empty.

The compiler:

1. resolves `config.json` and requires exact equality with a supported family
   profile;
2. checks the exact kernel and runtime-support source files declared by the
   family scope, without consulting Git or unrelated working files;
3. derives the complete model-reachable launch inventory, including wrapper,
   call sites, static keys, specialization, block configuration, warps, and
   stages;
4. runs the installed kernel verifier in-process for every distinct selected
   `(source, kernel, specialization)` case, including the batch goal and
   selected-row and exact-effect goals where required by the engine interface;
5. binds every launch to the resulting exact proof case and exports the backend
   assumptions used by the qualified relational analysis;
6. writes `deployment-candidate.json`, then runs one GPU probe for every
   distinct required obligation and applicable
   sealed execution context;
7. requires exact passing coverage and an unchanged observed environment; and
8. seals the candidate, report, model configuration, sources, environment, and
   static launch inventory into one deployment identity.

Preparation verifies only the selected specializations. `make verify-kernels`
checks the full family inventory, and `make check` also runs CPU regressions.
The generated interface and its Verus adapters are described in
[kernel contracts](kernel-contracts.md).

Batch size, token count, query length, and KV length are not static launch
selection keys. Matmul selection uses model/call-site `(N, K)` while `M` remains
the runtime token axis. The KV page size is a compiled engine constant, not a user-configurable deployment option.

## Bundle contents

The output directory contains:

| File | Meaning |
| --- | --- |
| `deployment-candidate.json` | Model configuration, sources, environment, kernel proofs, required probes, and proposed launches |
| `backend-qualification-report.json` | Exact empirical probe results for the candidate |
| `deployment.json` | Static plan sealed after all required probes pass; startup must still bind it to the loaded model |

The candidate, report, and deployment schemas are version 5; backend requirements
use version 2. Unknown or older schemas are rejected.

The optional `provenance.base_revision` records the Git revision for reference.
It does not affect bundle compatibility or deployment identity.

The deployment SHA-256 is the formal `KernelPlanId` source. It identifies a
determinism domain; it is not a claim that another legal plan, compiler build,
or hardware environment is equivalent.

## Phase 2: bind a runtime

Serving receives the directory through `VOSTI_DEPLOYMENT_BUNDLE`:

~~~bash
VOSTI_DEPLOYMENT_BUNDLE=/path/to/deployment-bundle \
  uv run --locked python scripts/launch.py --kind engine --family qwen3

VOSTI_DEPLOYMENT_BUNDLE=/path/to/deployment-bundle \
  uv run --locked python scripts/launch.py --kind engine --family gemma3

VOSTI_DEPLOYMENT_BUNDLE=/path/to/deployment-bundle \
  uv run --locked python scripts/launch.py --kind engine --family gemma4

VOSTI_DEPLOYMENT_BUNDLE=/path/to/deployment-bundle \
  uv run --locked python scripts/launch.py --kind engine --family llama3
~~~

At startup the family runtime:

1. reloads all three records and reconstructs the seal exactly;
2. rechecks the complete resolved model config and its digest;
3. rechecks the deployment scope, imported kernel origins and source digests,
   and the observed environment; no repository commit or clean-tree check runs;
4. independently rederives the profile's complete launch inventory and rejects
   missing, extra, duplicate, or changed launches;
5. constructs an immutable launch lookup inside one explicit family runtime
   object; and
6. exposes the sealed deployment identity to the checked Rust assembly path.

There is no process-global model or kernel-plan selection. Every primitive that
needs launch configuration receives the explicit capability stored in its
Engine. Independent Engines may therefore retain different qualified runtime
objects in one process without changing one another.

The family checkpoint loader and Rust permission binder then validate physical
roles and project the semantic model. The common checked deployment assembler
combines matching weights, permissions, runtime architecture, and the
`BackendQualified` plan into `model_execution_valid`. Only that checked result
may enter `Engine::init`. The runtime identity is revalidated at model-forward
boundaries.

Startup fails if the bundle is missing, malformed, incomplete, or does not
match the model configuration, environment, or bound sources.
No environment variable may select an alternate configured implementation
after the runtime object is constructed.

### Reusing a bundle

A bundle can be copied to another machine if startup's checks still pass:

| Change | Reuse the bundle? |
| --- | --- |
| Another physical GPU with the same recorded properties and software | Yes; GPU UUIDs are not checked |
| Different GPU model or memory capacity | No; prepare a new bundle on that GPU |
| Different weights with identical configuration and a supported weight layout | Yes; the bundle does not hash weight values |
| Same tensor shapes but different model configuration | No; shapes alone do not determine compatibility |
| Changed Torch, Triton, CUDA runtime, or driver version | No; prepare a new bundle |
| Moved checkout or documentation-only edits | Yes, if all bound sources are unchanged |

The GPU check compares the name, compute capability, and memory capacity of
every visible device, including the number and order of devices. The software
check compares the recorded Torch, Triton, CUDA runtime, and driver versions.
Model checks compare both the resolved configuration and the hash of
`config.json`, so even an otherwise harmless config-file edit requires
preparation again. Kernel/support sources, contracts, and launch configurations
must also match. Git history and filesystem location need not match.

For a fine-tuned checkpoint, run the model-level device checks below even if
the bundle is reusable. The bundle checks configuration compatibility; it
does not establish numerical behavior for new weights. Determinism compares
runs with the same weights, not the original and fine-tuned models.

For a new GPU or model configuration, preparation is the starting point,
not a guarantee of support. It rejects unsupported profiles and failed kernel
or backend checks. Enough GPU memory is also required. See [models](models.md)
for adding a profile.

Bundle reuse does not replace framework verification or the checks that the
built framework corresponds to its verified sources.

### Model-level device checks

`make test-gpu FAMILY=<family>` runs four empirical checks after deployment
qualification:

1. the KV-store check validates written rows, preservation of unwritten rows,
   cache identities, and batched/singleton agreement;
2. the Engine smoke check exercises checked assembly and multi-step serving;
3. the Transformers reference check captures the Engine's complete selected
   float32 vocabulary rows at the sampling boundary, validates their sources
   and digests, and requires both the same token sequence and bounded numerical
   differences. Its optional K/V mode also observes every model-layer cache
   scatter, requires each stored row to equal its pre-scatter Engine input
   byte-for-byte, and compares those rows with Transformers intermediates; and
4. causal-confinement checks test the attention boundary independently.

The reference checker defaults to a minimum top-logit margin of `1.0` and a
maximum absolute logit error of `1.0`; both can be changed on the command line.
This is a numerical comparison, not a bitwise-equality check. With `--output`,
it requires a clean checkout and matching deployment, records checkpoint
hashes, checks that the files do not change during the run, and writes a report
outside the checkout.

`scripts/checks/reference_suite.py` compares retained logit rows from a complete,
passing strict Vosti determinism suite with Transformers. It rechecks the suite's
inputs, results, and cache-reuse evidence before making the comparison.
Neither reference check supplies assumptions to the Verus proof.

## Static deployment versus dynamic execution

| Sealed during qualification | Dynamic under the Engine proof |
| --- | --- |
| Model profile and complete resolved config | Request arrival and admission |
| Exact kernel/support source catalog and contracts | Scheduler ordering and batching |
| Hardware/compiler/runtime environment | Decode and prefill mixture |
| Kernel wrapper, specialization, and launch meta | Token, query, and KV lengths within proved domains |
| No additional sealed choice | Prefix reuse, page allocation, eviction, and optional CUDA-graph execution under its separate capability and replay premise |

Cross-trace determinism requires the same semantic model and complete plan
identity. Changing a sealed choice creates a different theorem premise and
requires a separately qualified deployment.

## What backend probes establish

The trust chain is conditional:

~~~text
tested primitive obligations
  + trusted verifier derivation
  + exact source/specialization/runtime binding
  => imported kernel property
~~~

The verifier derives structural, regional, and selected-row properties from
primitive semantics. Qualification probes the specialized primitive
obligations and execution contexts used by that derivation; it does not
independently prove whole serving kernels.

The probe registry fails closed on unknown operations, properties, probe
families, or routing. It covers the specialized shapes and dtypes required by
the sealed catalog, including arithmetic and casts, reductions, dot products,
memory operations, control/grid behavior, address arithmetic, mask identities,
and relevant BF16/FP32 conversions. Unified relational-dataflow artifacts supply
both batch and selected-row imports through one qualified verifier. Failed or
unsupported proofs, missing artifacts, and absent positive typed annotation
satisfiability evidence are rejected. This satisfiability check includes
quantified premises but does not discharge analyzer-added numerical conditions
or establish physical realizability on the GPU.

Determinism, locality, and exact-identity probes compare tensor shapes, dtypes,
and element bytes, including signed zero and NaN payloads. Explicit numerical
zero properties permit either zero sign; they do not qualify bitwise additive
identity or erase the sign dependency of elementwise multiplication.

A passing report establishes only bounded observations on the recorded
environment. It does not prove:

- numerical correctness of a full kernel;
- the soundness of the verifier's rule-to-obligation derivation;
- correspondence between the reviewed model-to-launch inventory and every
  reachable Python call;
- Python/Torch tensor representation or physical memory safety;
- correctness of Triton compilation or GPU execution; or
- equivalence with another launch plan or environment.

## Residual qualification gaps

Remaining gaps are:

- the analysis-rule-to-backend-obligation mapping is a reviewed closed
  inventory rather than an automatically emitted derivation trace;
- the model-to-launch inventory is independently checked but not extracted by a
  formally verified call-graph analysis;
- specialized block-pointer records do not retain every source stride/order
  expression;
- probes are bounded tests even when every sealed warp/stage context is covered;
  and
- compiler-binary identity, relevant flags, and compiled artifact hashes are
  not sealed.

The verifier, lowering, runtime representation, compiler, GPU, and correspondence
between analyzed and executed code remain in the TCB. See the generated
[trust inventory](../audit/tcb.md) and [claim ledger](../audit/claims.md).
