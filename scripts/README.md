# Scripts

Run commands from the repository root. The root Makefile is the public entry
point; `project.py` orchestrates its checks. See [setup](../docs/setup.md) for the
command catalog, preparation, launch, and regeneration procedures.

## Tool ownership

| Path | Responsibility |
| --- | --- |
| `project.py` | Sequential verification, tests, diagnostics, and audit gates |
| `prepare_deployment.py`, `launch.py` | Stable model preparation and engine/server entry points |
| `deployment/` | Shared qualification pipeline and family candidate builders |
| `verification/` | Kernel-to-engine contract generation, verification, and Rust templates |
| `verification/diagnostics/` | Independent logical checks |
| `audit/` | Architecture/trust checks and generated claims/TCB inventories |
| `checks/` | Empirical model, KV-store, and reference checks |
| `common/` | Shared tokenization, model discovery, process/GPU monitoring, telemetry, and timing |
| [effort/](effort/README.md) | Source classification, dependency extraction, and accounting |

Kernel analysis and tuning live in [`kernels/`](../kernels/README.md).
Tuning produces measurements; a selected configuration must still pass
verification and deployment checks.

## Experiment suites

| Suite | Purpose |
| --- | --- |
| [Determinism](determinism_tests/README.md) | Raw-logit comparisons under execution variations |
| [Serving](serving_benchmark/README.md) | ShareGPT and synthetic multi-session replay over HTTP |
| [Aligned phases](serving_benchmark/aligned/README.md) | Controlled, synchronized engine-batch timings |
| [Agent](agent_benchmark/README.md) | Task-driven agent performance and outcome evaluation |

ShareGPT preparation lives in `serving_benchmark/workloads/sharegpt.py`;
synthetic multi-turn preparation uses the common tokenizer directly. Keep
run-specific plans, outputs, and reports with their run artifacts outside the
source tree.
