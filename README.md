# Vosti

Vosti is a research prototype for deterministic LLM inference. It combines a
Rust serving engine verified with Verus and annotated Triton kernels checked
by a relational kernel verifier.

This repository is the artifact for [*Vosti: Specifying, Implementing, and
Verifying Deterministic LLM Inference*](https://arxiv.org/abs/2609.38981). The guides
below assume familiarity with the paper and explain how to check its proofs
and repeat its experiments on your own hardware.

For a fixed model and qualified kernel plan, requests with the same prompt
and initial sampler state agree on every output index reached by both runs,
despite differences in batching, chunking, cache reuse, or scheduling.
The guarantee concerns determinism. It does not establish numerical correctness,
progress, or equality across kernel plans or hardware.
See the [specification and proof](docs/verification.md) for the exact statement
and trust boundary.

## Supported execution

| Family | Dense text profiles |
| --- | --- |
| Qwen3 | Qwen3 dense text |
| Gemma3 | 4B, 12B, 27B |
| Gemma4 | 12B, 31B |
| Llama3 | Llama 3.1 8B, 3.2 3B, 3.3 70B |

Checkpoints must match a supported profile. Prepare a deployment bundle for
your GPU and software environment, and allow enough memory for weights, KV
cache, and workspace. See [bundle reuse](docs/deployment.md#reusing-a-bundle)
for different GPUs or fine-tuned weights.

All families share continuous batching, decode, page-aligned chunked prefill,
dynamic request admission, pressure reclamation, prompt/generated-prefix reuse,
and exact-signature or padded pure-decode CUDA graphs. Full causal and
sliding-window attention are supported. Sliding-window layers still retain the
complete causal KV prefix; window-local KV eviction is not proved. Multimodal
execution, MoE, and linear attention are not supported.

## Evaluate the artifact

Start with the [artifact guide](docs/artifact.md). It covers the Docker
environment, CPU verification, a first GPU run, and the paper's determinism
and performance experiments. Model weights are downloaded separately.

Build the environment from the repository root on Linux x86-64:

```bash
docker build --build-arg VOSTI_UID="$(id -u)" --build-arg VOSTI_GID="$(id -g)" \
  -t vosti-artifact .
docker run --rm -it --mount type=bind,src="$PWD",dst=/workspace \
  vosti-artifact
```

Inside the container:

```bash
make verify          # engine and deployed kernel proofs, plus freshness checks
make test            # CPU regressions; no checkpoint required
```

These checks need no GPU. Continue with the artifact guide's
[GPU workflow](docs/artifact.md#first-gpu-run) to generate a deployment bundle
and serve a model inside Docker. Native installation is optional and documented
in [setup](docs/setup.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Artifact guide](docs/artifact.md) | Paper experiments, Docker workflow, and interpreting reruns |
| [Setup](docs/setup.md) | Environment, commands, server launch, and validation gates |
| [Architecture](docs/architecture.md) | Runtime components, ownership, and two-phase serving |
| [Verification](docs/verification.md) | Specification, theorem, proof structure, and limitations |
| [Kernel contracts](docs/kernel-contracts.md) | Generated Verus assumptions and checked engine adapters |
| [Deployment](docs/deployment.md) | Offline qualification, sealed bundles, and runtime admission |
| [Models](docs/models.md) | Adding profiles, families, and primitives |
| [Prefix cache](docs/prefix-cache.md) | Reuse protocol, coherence, and cache fidelity |
| [Kernels](kernels/README.md) | Annotation language, analysis, and local tools |
| [Scripts](scripts/README.md) | Tool ownership and benchmark-suite index |
| [Audit](audit/README.md) | Reviewed manifests and generated [claims](audit/claims.md) and [TCB](audit/tcb.md) inventories |

The public definition is [`src/spec.rs`](src/spec.rs); its engine satisfaction
theorem is [`src/proof.rs`](src/proof.rs). See the architecture guide for the
source-module map and [CLAUDE.md](CLAUDE.md) for contributor rules.

The Rust crate is `vosti_verus`, the Python package is `vosti_kernels`, and
environment variables use `VOSTI_`. Benchmark tools select this engine as
`vosti`; the separate vLLM baseline remains `vllm`.

## License

Vosti is distributed under the [MIT License](LICENSE). Dependencies, model
checkpoints, datasets, and container images retain their respective licenses.
