# One public entry point; orchestration and specialist commands live in scripts/.
.DEFAULT_GOAL := help
UV ?= uv
UV_SYNC_FLAGS ?= --locked
VERUS ?= $(HOME)/.local/verus/verus
CARGO_VERUS ?= $(HOME)/.local/verus/cargo-verus
export VERUS CARGO_VERUS
export KERNELS_DIR VERUS_CLAIM_TARGET_DIR VERUS_CLAIM_VIR_LOG_DIR
export PYO3_PYTHON ?= $(CURDIR)/.venv/bin/python
export FAMILY MODEL_PATH VOSTI_DEPLOYMENT_BUNDLE

.PHONY: help setup build verify verify-engine verify-kernels test test-gpu check generate clean

help:
	@printf '%s\n' \
	  'setup           Install the locked Python environment' \
	  'build           Build the Rust library' \
	  'verify          Verify the engine and all deployed kernel contracts' \
	  'verify-engine   Whole-crate Verus proof and emitted-VIR trust check' \
	  'verify-kernels  All families: kernel proofs and generated freshness' \
	  'test            CPU Rust, Python, and kernel regression suites' \
	  'test-gpu        One FAMILY; requires explicit GPU, checkpoint, bundle' \
	  'check           Complete CPU gate, including examples and diagnostics' \
	  'generate        Regenerate shared kernel interfaces and claim ledger' \
	  'clean           Remove build artifacts; preserve .venv' \
	  'See scripts/README.md for focused tests, audits, and accounting.'

setup:
	$(UV) sync $(UV_SYNC_FLAGS)

build:
	$(UV) run --locked cargo build

verify verify-engine verify-kernels test test-gpu check generate:
	$(UV) run --locked python scripts/project.py $@

clean:
	cargo clean
