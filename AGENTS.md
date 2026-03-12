# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Coordination With CLAUDE.md

- Read [CLAUDE.md](CLAUDE.md) together with this file before making repository changes.
- If there are any differences between `AGENTS.md` and `CLAUDE.md`, reconcile them before continuing.
- Keep both files synchronized when adding or changing guidance.

## Project Overview

GiGL (GIgantic Graph Learning) is an open-source library for billion-scale GNN training and inference. It supports node
classification, link prediction, supervised learning, and unsupervised learning. Primary runtime stack: Python 3.11 with
`uv` package management.

## High-Level Workflow

- Setup: `make install_dev_deps`
- Unit tests: `make unit_test_py` or `make unit_test_py PY_TEST_FILES="specific_test.py"`
- Integration tests: `make integration_test PY_TEST_FILES="specific_test.py"` (run one at a time, slower)
- Formatting: `make format`, or language-specific targets (`make format_py`, `make format_scala`, `make format_md`)
- Format checks: `make check_format`
- Type checks: `make type_check`
- Build/docs: `make compile_protos`, `make build_docs`

## Architecture At A Glance

### Pipeline Stages (`gigl/src/`)

1. ConfigPopulator
2. DataPreprocessor
3. SplitGenerator (deprecated unless explicitly requested)
4. SubgraphSampler (deprecated unless explicitly requested)
5. Trainer
6. Inferencer
7. PostProcessor

### Distributed Training (`gigl/distributed/`)

- Extends GraphLearn-for-PyTorch (GLT) with GiGL-specific distributed dataset/loaders/samplers.
- Core types include `DistDataset`, `DistNeighborLoader`, `DistABLPLoader`, and `DistNeighborSampler`.
- Supports two deployment modes:
  - Colocated (storage + compute on same nodes)
  - Graph Store (separate storage/compute via RPC)
- Handles homogeneous, heterogeneous, and labeled homogeneous graph variants.

### Other Key Areas

- `gigl/common/`: shared utilities, URI types, logging, services, metrics
- `gigl/orchestration/`: Kubeflow pipeline compilation + local orchestration
- `gigl/nn/`: neural network modules
- `gigl/src/common/types/pb_wrappers/`: protobuf wrappers
- `gigl/src/mocking/`: test dataset asset mocking
- `scala/` and `scala_spark35/`: legacy Scala components

## Coding Standards (High-Level)

- Prefer explicit, clear names and reuse/refactor existing code where possible.
- Fail fast on invalid state; use strict access patterns when keys are required.
- Use complete type annotations and modern built-in generic types.
- Add Google-style docstrings for public functions/methods.
- Use `gigl.common.logger.Logger` for logging.
- For protobuf config handling, deserialize early and pass wrappers/data classes (not raw wrappers deep downstream).
- Be performance-aware for large/distributed workloads (generators, CPU parallelism where appropriate, memory cleanup).
- Keep `__init__.py` exports minimal and stable via `__all__`.

## Testing Conventions (High-Level)

- Base test class: `tests.test_assets.test_case.TestCase` (not `unittest.TestCase`)
- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- E2E definitions in `testing/e2e_tests/e2e_tests.yaml`
- Use `unittest.mock` for external dependencies
- Assert error paths with `self.assertRaises`; only assert exact messages when load-bearing

## Pre-Submit And Formatting Notes

From [`.claude/formatting.md`](.claude/formatting.md):

1. Do not suppress errors with `# type: ignore` workarounds.
2. Run `make type_check`.
3. Run relevant unit tests (`make unit_test_py PY_TEST_FILES="relevant_test.py"`).
4. Run integration tests for cross-component changes.
5. Run `make check_format` (or `make format` to auto-fix).
6. Remember `make format` is not a pre-commit hook; run formatting manually.

From [`.claude/development.md`](.claude/development.md):

- Only create a new branch if you are on `main` or explicitly asked.
- Branch naming: `{USER}/snake-case-feature` (`USER` from `GIGL_ALERT_EMAILS` username or `whoami`).
- Keep PRs small and self-contained when asked to split work.
- Use formatters and tests targeted to the files changed.
