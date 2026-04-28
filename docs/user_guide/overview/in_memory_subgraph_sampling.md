# In-Memory Subgraph Sampling

In-memory subgraph sampling is the current, recommended GiGL execution model. Instead of materializing sampled subgraphs
ahead of time, GiGL loads the graph into memory and samples neighborhoods live during training and inference.

If you are maintaining an older deployment that still relies on precomputed sampled subgraphs, see
[Deprecated tabularized docs](deprecated_tabularized/index.md).

## How It Works

The in-memory path is currently enabled by setting `featureFlags.should_run_glt_backend` to `True` in the task config.

At a high level:

1. The Data Preprocessor writes `PreprocessedMetadata`, transformed feature assets, graph metadata, and the mapping
   tables needed to convert between original node IDs and enumerated integer IDs.
2. Training and inference build distributed datasets from those preprocessed assets instead of reading precomputed
   sampled TFRecords.
3. Distributed loaders sample neighborhoods online from those datasets during training and inference.
4. The Post Processor restores original node IDs after inference so downstream systems receive user-facing outputs.

For stage-specific behavior, see the component guides for [Data Preprocessor](components/data_preprocessor.md),
[Trainer](components/trainer.md), [Inferencer](components/inferencer.md), and
[Post Processor](components/post_processor.md).

## Runtime Abstractions

The main abstractions used by the in-memory path are:

- {py:class}`gigl.distributed.dist_dataset.DistDataset` for colocated sampling, where each machine stores its graph
  partition locally.
- {py:class}`gigl.distributed.distributed_neighborloader.DistNeighborLoader` for standard neighborhood sampling.
- {py:class}`gigl.distributed.dist_ablp_neighborloader.DistABLPLoader` for anchor-based link prediction batches with
  positives and negatives.

## Example Implementations

The link prediction examples are the clearest reference implementations for the current runtime:

- Colocated training:
  [`homogeneous_training.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/homogeneous_training.py),
  [`heterogeneous_training.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/heterogeneous_training.py)
- Colocated inference:
  [`homogeneous_inference.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/homogeneous_inference.py),
  [`heterogeneous_inference.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/heterogeneous_inference.py)

You can also start from the [Examples index](../examples/index.md), which includes the end-to-end link prediction
walkthroughs.

## Cost Savings

For internal Snap use cases, we have seen up to 90% cost savings when swapping to in-memory subgraph sampling from the
tabularized solution. We also saw major (>50%) improvements in overall inference pipeline run time.
