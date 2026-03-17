# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coordination With AGENTS.md

- CLAUDE.md is the cannonical source of truth for this repo.
- AGENTS.md should only say to load CLAUDE.md.

## Project Overview

GiGL (GIgantic Graph Learning) is an open-source library for training and inference of Graph Neural Networks at
billion-scale. It supports node classification, link prediction, and both supervised and unsupervised learning. Python
3.11, `uv` for package management.

## Common Commands

```bash
# Setup
make install_dev_deps              # Full dev setup (gcloud auth, uv, pre-commit)

# Testing
make unit_test_py                                    # All Python unit tests (includes type_check)
# NOTE: PY_TEST_FILES should *only* be the filename, *not* the full path.
# e.g. if you want to test `tests/unit/common/foo_test.py` then you should run `make unit_test_py PY_TEST_FILES="foo_test.py"
make unit_test_py PY_TEST_FILES="specific_test.py"   # Single test file
make integration_test PY_TEST_FILES="specific_test.py"  # Integration (run one at a time, slow)

# Formatting & Linting
make format              # Auto-fix Python, Scala, Markdown
make format_py           # Auto-fix Python only
make format_scala        # Auto-fix Scala only
make format_md           # Auto-fix Markdown only
make check_format        # Check without fixing
make type_check          # mypy static type checking

# Build
make compile_protos      # Regenerate protobuf code after .proto changes
make build_docs          # Sphinx documentation
```

## Architecture

### Pipeline Components (`gigl/src/`)

GiGL runs as a multi-stage pipeline. Each stage is a standalone runnable module:

1. **ConfigPopulator** (`config_populator/`) - Deserializes YAML task configs into protobuf
2. **DataPreprocessor** (`data_preprocessor/`) - Preprocesses raw graph data
3. **SplitGenerator** (`split_generator/`) - Creates train/val/test splits - Deprecated, do not consider for planning
   unless explicitly asked.
4. **SubgraphSampler** (`subgraph_sampler/`) - Samples subgraphs for training - Deprecated, do not consider for planning
   unless explicitly asked.
5. **Trainer** (`training/`) - V1 trainer and V2 GLT trainer
6. **Inferencer** (`inference/`) - Model inference
7. **PostProcessor** (`post_process/`) - Post-processing results

### Distributed Training (`gigl/distributed/`)

GiGL extends GraphLearn-for-PyTorch (GLT) for distributed GNN training. Key class hierarchy:

- **`DistDataset`** (extends `graphlearn_torch.distributed.DistDataset`) - Core data container adding link prediction
  labels, split metadata, and feature info
- **`DistNeighborLoader`** (extends GLT `DistLoader`) - Standard node-based sampling loader
- **`DistABLPLoader`** (extends GLT `DistLoader`) - Anchor-Based Link Prediction sampling loader
- **`DistNeighborSampler`** (extends GLT `DistNeighborSampler`) - Unified sampler supporting both standard neighbor
  sampling and ABLP with positive/negative label injection

**Two deployment modes:**

- **Colocated**: Data and compute on same nodes. Each rank has a local partition of the graph/features.
- **Graph Store**: Separate storage and compute clusters. Storage nodes run `DistServer`, compute nodes use
  `RemoteDistDataset` via RPC. Scales to 100+ nodes using sequential per-node initialization to avoid GLT's
  ThreadPoolExecutor bottleneck.

**Data flow:**

```
dataset_factory.build_dataset()  →  DistDataset (partitioned via DistPartitioner)
    → DistNeighborLoader / DistABLPLoader  →  sampled subgraph batches (Data/HeteroData)
        → Model training loop
```

**Key files:**

- `dist_dataset.py` - Core dataset structure with IPC serialization
- `distributed_neighborloader.py` - DistNeighborLoader (both modes)
- `dist_ablp_neighborloader.py` - DistABLPLoader (both modes)
- `dist_neighbor_sampler.py` - ABLP-aware neighbor sampling
- `dataset_factory.py` - Dataset building and partitioning orchestration
- `graph_store/dist_server.py` - Storage server for Graph Store mode
- `graph_store/remote_dist_dataset.py` - Client-side dataset proxy
- `graph_store/compute.py` - RPC utilities (`request_server`, `async_request_server`)
- `utils/neighborloader.py` - `SamplingClusterSetup` enum, `DatasetSchema`

**Graph types:** Supports homogeneous, heterogeneous, and "labeled homogeneous" (heterogeneous with one default node
type + label edge types, treated as homogeneous for sampling).

### Other Key Packages

- **`gigl/common/`** - Shared utilities: `Uri` types (GcsUri, HttpUri, LocalUri), `Logger`, services, metrics
- **`gigl/orchestration/`** - Kubeflow pipeline compilation and local orchestration
- **`gigl/nn/`** - Neural network modules
- **`gigl/src/common/types/pb_wrappers/`** - Protobuf wrapper classes
- **`gigl/src/mocking/`** - Dataset asset mocking for tests

### Scala Components (Legacy)

Two Scala projects under `scala/` and `scala_spark35/`, built with SBT. These are legacy and not the focus of active
development.

### Configuration

- Task configs: YAML files defining the GNN pipeline (examples in `examples/`)
- Resource configs: YAML files for GCP resources (e.g., `deployment/configs/`)
- Test resource config: `deployment/configs/unittest_resource_config.yaml`

## Coding Standards

### Naming and Clarity

- Use explicit, unabbreviated variable names. When in doubt, spell it out. Shortened names are OK only for universally
  understood abbreviations (`i`, `e`, `url`, `id`, `config`) or to avoid shadowing.
- Use OOP for model architectures, functional style for data transforms/pipelines.
- Re-use and refactor existing code as a priority instead of implementing new code.

### Fail Fast on Invalid State

- Use `dict[key]` (bracket access) when the key **must** exist. Only use `.get(key, default)` when absence is a valid,
  expected case with a meaningful default.
- Validate preconditions at function entry. Raise explicit exceptions rather than silently continuing with bad data.

### Type Annotations

- Always use type annotations for function parameters and return values.
- Prefer native types (`dict[str, str]`, `list[int]`) over `typing.Dict`, `typing.List`.
- Use `Final` for constants. Use `@dataclass(frozen=True)` for immutable data containers.
- Always annotate empty containers: `names: list[str] = []` not `names = []`.

### Docstrings

Add Google-style docstrings for all public functions and methods. Include: one-line summary, optional details, Example
with `>>>` for doctests, Args, Returns, and Raises. Docstrings should be Sphinx-compatible.

### Logging

```python
from gigl.common.logger import Logger
logger = Logger()
```

### Protocol Buffers

- Proto definitions: `proto/snapchat/research/gbml/`. Import types from `snapchat.research.gbml`.
- Use wrapper classes for protobuf operations:
  - `GbmlConfigPbWrapper` for `gbml_config_pb2.GbmlConfig` (task config / template task config)
  - `GiglResourceConfigWrapper` for `gigl_resource_config_pb2.GiglResourceConfig`
- Deserialize protos into wrapper objects or explicit data classes **as early as possible** in entry-point files
  (ConfigPopulator, DataPreprocessor, SubgraphSampler, SplitGenerator, Trainer, Inferencer). Downstream code called by
  these entry points should NOT receive `GbmlConfigPbWrapper` or `GiglResourceConfigWrapper` directly.

### Performance

- Use generators for large data processing.
- Use `concurrent.futures.ProcessPoolExecutor` for CPU-bound parallel tasks.
- Use GiGL's timeout utilities: `from gigl.src.common.utils.timeout import timeout`
- Be mindful of memory in distributed settings. Delete intermediate tensors and call `gc.collect()` to prevent OOM.

### `__init__.py` Files

Define a minimal, consistent public API. Only expose stable, user-facing classes/functions through `__all__`. Keep
helpers/internal logic in private modules.

## Testing Conventions

### Base Class

Use `tests.test_assets.test_case.TestCase` as the base class, **NOT** `unittest.TestCase`.

### Test Organization

- **Unit tests**: `tests/unit/` - Fast, isolated tests
- **Integration tests**: `tests/integration/` - Component interaction tests, require cloud resources
- **E2E tests**: Defined in `testing/e2e_tests/e2e_tests.yaml`
- **Test assets**: `tests/test_assets/` (configs in `configs/`, test graphs in `small_graph/`)

### Test Structure

```python
from tests.test_assets.test_case import TestCase

class TestMyComponent(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Shared resources across tests
        ...

    def setUp(self) -> None:
        # Per-test setup
        ...

    def tearDown(self) -> None:
        # Per-test cleanup
        ...
```

### Error Testing

- Test error cases with `self.assertRaises`.
- Avoid asserting on exact error message strings **unless** the message is load-bearing: disambiguating multiple error
  paths in the same function, or structured error reporting used downstream.

### Mocking

Mock external services using `unittest.mock` (`Mock`, `patch`, `MagicMock`). Create minimal test configs in
`tests/test_assets/configs/`.

## Additional instructions

- For a pre-submit checklist and formatting see .claude/formatting.md

- For general development and branch naming conventions see .claude/development.md

- When migrating code, make sure to migrate any doc comments or diagrams over to the new code.
