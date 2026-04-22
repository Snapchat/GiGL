# Inference

The Inferencer component is responsible for running model inference and persisting output embeddings and/or predictions.
It supports both the legacy tabularized path and the newer in-memory subgraph sampling path.

## Input

- **job_name** (AppliedTaskIdentifier): which uniquely identifies an end-to-end task.
- **task_config_uri** (Uri): Path which points to a frozen `GbmlConfig` proto yaml file.
- **resource_config_uri** (Uri): Path which points to a `GiGLResourceConfig` yaml
- **Optional: custom_worker_image_uri**: Path to docker file to be used for dataflow worker harness image

## What does it do?

The Inferencer undertakes the following actions:

- Reads the frozen `GbmlConfig` and resource config.
- Clears staging paths for inference assets so retries do not clobber or mix outputs.
- Chooses the inference backend:
  - Legacy tabularized path when `featureFlags.should_run_glt_backend` is not enabled.
  - In-memory subgraph sampling path when `featureFlags.should_run_glt_backend` is `True`.

### Legacy path

In the legacy path, the Inferencer runs the v1 inference stack over precomputed samples and writes embeddings and/or
predictions to BigQuery.

### In-Memory Subgraph Sampling Path

In the in-memory path, the Inferencer launches the distributed runtime used for live neighborhood sampling. At a high
level, that runtime:

- launches the user-provided inference command from `inferencerConfig.command`,
- loads the trained model,
- builds a `DistDataset` or `RemoteDistDataset` from Data Preprocessor outputs,
- samples neighborhoods online during inference instead of consuming precomputed sampled subgraphs,
- writes embeddings and/or predictions keyed by enumerated node IDs.

For in-memory sampling runs, automatic unenumeration is handled by the Post Processor after inference completes.

## How do I run it?

**Import GiGL**

```python
from gigl.src.inference.inferencer import Inferencer
from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier

inferencer = Inferencer()

inferencer.run(
    applied_task_identifier=AppliedTaskIdentifier("sample_job_name"),
    task_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml"),
    resource_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/resource_config.yaml"),
    custom_worker_image_uri="gcr.io/project/directory/dataflow_image:x.x.x",  # Optional
)
```

**Command Line**

```bash
python -m gigl.src.inference.inferencer \
    --job_name="sample_job_name" \
    --task_config_uri="gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml" \
    --resource_config_uri="gs://MY TEMP ASSETS BUCKET/resource_config.yaml"
```

## Output

The Inferencer outputs embedding and / or prediction assets, based on the `taskMetadata` in the frozen `GbmlConfig`.
Specifically, for Node-anchor Based Link Prediction tasks as we have in the sample MAU config, the embeddings are
written to the BQ table specified at the `embeddingsBqPath` field in the `sharedConfig.inferenceMetadata` section.

For in-memory inference, these assets are initially keyed by enumerated node IDs and are unenumerated in the Post
Processor stage.

## Custom Usage

The customization point depends on the backend:

- Legacy path: implement a custom `BaseInferencer` if you need custom v1 inference behavior.
- In-memory path: provide a custom inference command in `inferencerConfig.command`, such as a script built on top of
  `DistDataset` or `RemoteDistDataset`.

## Examples

Reference in-memory inference implementations:

- [`examples/link_prediction/homogeneous_inference.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/homogeneous_inference.py)
- [`examples/link_prediction/heterogeneous_inference.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/heterogeneous_inference.py)
- [`examples/link_prediction/graph_store/homogeneous_inference.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/graph_store/homogeneous_inference.py)
- [`examples/link_prediction/graph_store/heterogeneous_inference.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/graph_store/heterogeneous_inference.py)

## Other

- **Design:** GiGL now supports both the legacy Dataflow-based inference path and the in-memory subgraph sampling path,
  which can run on single-pool or graph-store Vertex AI jobs depending on the resource config.

- **Debugging:** For the legacy path, the core logic executes in Dataflow. For the in-memory path, the core logic
  executes in the launched training or inference job, so logs should be inspected in the corresponding Vertex AI job
  output.
