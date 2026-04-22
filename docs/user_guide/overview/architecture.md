# GiGL Architecture

GiGL now has two execution models:

- The current, recommended path uses in-memory subgraph sampling, where the graph is loaded into memory and sampled live
  during training and inference. If you are looking for the older tabularized pipeline, see
  [Deprecated tabularized docs](deprecated_tabularized/index.md).
- The older tabularized path materializes sampled subgraphs ahead of time through Subgraph Sampler and Split Generator.

This page focuses on the current in-memory subgraph sampling architecture and points to the legacy docs separately.

## Primary Pipeline Flow

The primary GiGL flow is:

```text
Config Populator -> Data Preprocessor -> Trainer? -> Inferencer -> Post Processor
```

`Trainer` is optional. Inference-only pipelines skip training and run inference against a graph using a pre-trained
model.

### Active Components

- [**Config Populator**](components/config_populator.md): Freezes the template task config into a runnable `GbmlConfig`.
- [**Data Preprocessor**](components/data_preprocessor.md): Builds graph metadata, transforms features, and enumerates
  node IDs into compact integer IDs.
- [**Trainer**](components/trainer.md): Launches either legacy training or in-memory subgraph sampling training.
- [**Inferencer**](components/inferencer.md): Launches either legacy inference or in-memory subgraph sampling inference.
- [**Post Processor**](components/post_processor.md): Restores original node IDs for outputs produced by in-memory
  subgraph sampling and runs optional user-defined post-processing logic.

#### Component Diagram

Below is a high-level system overview. Note that both training and inference are backed by the same in-memory graph
sampling engine.

![System overview](../../assets/images/in-memory-sgs-sys-overview.png)

### Source Entry Points

| Component         | Source Code                                                               |
| ----------------- | ------------------------------------------------------------------------- |
| Config Populator  | {py:class}`gigl.src.config_populator.config_populator.ConfigPopulator`    |
| Data Preprocessor | {py:class}`gigl.src.data_preprocessor.data_preprocessor.DataPreprocessor` |
| Trainer           | {py:class}`gigl.src.training.trainer.Trainer`                             |
| Inferencer        | {py:class}`gigl.src.inference.inferencer.Inferencer`                      |
| Post Processor    | {py:class}`gigl.src.post_process.post_processor.PostProcessor`            |

## How In-Memory Sampling Works

The in-memory path is currently enabled by setting `featureFlags.should_run_glt_backend` to `True` in the task config.

At a high level:

1. The Data Preprocessor still performs the heavy lifting of preparing graph data. It writes `PreprocessedMetadata`,
   feature assets, graph/type metadata, and the mapping tables needed to convert between original node IDs and
   enumerated integer IDs.
2. Training and inference build distributed datasets from the preprocessed assets instead of reading precomputed sampled
   TFRecords.
3. Distributed loaders sample neighborhoods online from those datasets during training and inference.

The main abstractions used by the in-memory path are:

- {py:class}`gigl.distributed.dist_dataset.DistDataset` for colocated sampling, where each machine stores its graph
  partition locally.
- {py:class}`gigl.distributed.graph_store.remote_dist_dataset.RemoteDistDataset` for graph store mode, where compute
  nodes sample from a dedicated storage cluster.
- {py:class}`gigl.distributed.distributed_neighborloader.DistNeighborLoader` for standard neighborhood sampling.
- {py:class}`gigl.distributed.dist_ablp_neighborloader.DistABLPLoader` for anchor-based link prediction batches with
  positives and negatives.

## Cost savings

For internal Snap usecases, we have seen up to 90% cost savings when swapping to in-memory subgraph sampling from the
tabularized solution. We also saw major (>50%) improvements on overall inference pipeline run time.

## Data Preprocessor to Trainer to Inferencer to Post Processor

### Data Preprocessor

The Data Preprocessor produces:

- transformed node and edge features,
- graph/type metadata,
- preprocessed asset locations,
- enumerated node ID mapping tables.

Those assets are what the Trainer and Inferencer use to build either a colocated `DistDataset` or a graph-store
`RemoteDistDataset`.

### Trainer

When training is enabled, the Trainer chooses the runtime based on the frozen config:

- In-memory path: launches a user-provided training command and performs on-the-fly subgraph sampling during training.

For link prediction, the example training loops under `examples/link_prediction` use:

- `DistABLPLoader` for anchor/positive/negative batches.
- `DistNeighborLoader` for random negative batches.

### Inferencer

The Inferencer follows the same split:

- In-memory path: launches a user-provided inference command that samples neighborhoods online and writes embeddings
  and/or predictions for enumerated node IDs.

Inference-only pipelines use the same preprocessed assets as training pipelines, but skip the Trainer stage entirely. In
practice, that means setting `sharedConfig.shouldSkipTraining: true` and `sharedConfig.shouldSkipModelEvaluation: true`,
while keeping `sharedConfig.shouldSkipInference: false` so the Inferencer still runs. Inference-only pipelines also need
`sharedConfig.trainedModelMetadata` to point at an existing trained model for the Inferencer to load.

### Post Processor

In the in-memory path, inference outputs are initially keyed by enumerated node IDs. The Post Processor is responsible
for joining those outputs back to the Data Preprocessor's enumeration mapping tables so downstream consumers see
original, user-facing IDs again.

This unenumeration step is part of the system architecture, not just optional user logic.

## Example Implementations

The link prediction examples are the clearest reference implementations for the current architecture:

- Colocated training:
  [`homogeneous_training.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/homogeneous_training.py),
  [`heterogeneous_training.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/heterogeneous_training.py)
- Colocated inference:
  [`homogeneous_inference.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/homogeneous_inference.py),
  [`heterogeneous_inference.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/heterogeneous_inference.py)

You can also start from the [Examples index](../examples/index.md), which includes the end-to-end link prediction
walkthroughs.

## Deprecated Tabularized Architecture

If you are maintaining an older deployment that still relies on precomputed sampled subgraphs, see
[Deprecated tabularized docs](deprecated_tabularized/index.md).

That flow is:

```text
Config Populator -> Data Preprocessor -> Subgraph Sampler -> Split Generator -> Trainer -> Inferencer
```

```{toctree}
:maxdepth: 2
:hidden:

components/config_populator
components/data_preprocessor
components/trainer
components/inferencer
components/post_processor
deprecated_tabularized/index
```
