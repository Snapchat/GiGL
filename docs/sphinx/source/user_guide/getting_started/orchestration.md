# Orchestration

GiGL is designed to support easy end to end orchestration of your GNN tasks/workflows with minimal setup required. This
page outlines three ways to orchestrate GiGL for after you have set up your configs (See:
[quick start](../config_guides/task_config_guide.md) if you have not done so).

## Local Runner

The local runner provides a simple interface to kick off an end to end GiGL pipeline.

1. Create a pipeline config. The pipeline config takes in:

- applied_task_identifier: your job name (string)
- template_task_config_uri: the URI to your template task config (Uri)
- resource_config_uri: The URI to your resource config (URI)
- custom_cuda_docker_uri: For custom training spec and GPU training on VertexAI (optional, string)
- custom_cpu_docker_uri: For custom training spec and CPU training on VertexAI (optional, string)
- dataflow_docker_uri: For custom datapreprocessor spec that will run in dataflow (optional, string)

Example:

```python
from gigl.orchestration.local.runner import PipelineConfig
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.common import UriFactory, Uri

PipelineConfig(
    applied_task_identifier=AppliedTaskIdentifier("demo-gigl-job"),
    template_task_config_uri=UriFactory.create_uri("gs://project/my_task_config.yaml"),
    resource_config_uri=UriFactory.create_uri("gs://project/my_resource_config.yaml")
)
```

2. Initialize and Run

Now you can create the GiGL runner object and kick off a pipeline.

Example:

```python

runner = Runner.run(pipeline_config=pipeline_config)

```

3. Optional: The runner also supports running individual components as needed

Example:

```python
Runner.run_data_preprocessor(pipeline_config=pipeline_config)
```

## Kubeflow Orchestration

GiGL also supports orchestration of your workflows using
[Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/v2/) using a `KfpOrchestrator` class. We make
use of Vertex AI to run the Kubeflow Pipelines.

### Usage Example

```python

from gigl.orchestration.kubeflow.runner import KfpOrchestrator
from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier

orchestrator = KfpOrchestrator()

task_config_uri = UriFactory.create_uri("gs://path/to/task_config.yaml")
resource_config_uri = UriFactory.create_uri("gs://path/to/resource_config.yaml")
applied_task_identifier = AppliedTaskIdentifier("kfp_demo")

orchestrator.run(
    applied_task_identifier=applied_task_identifier,
    task_config_uri=task_config_uri,
    resource_config_uri=resource_config_uri,
    start_at="config_populator"
)

```

## Importable GiGL

You may want to integrate gigl into your existing workflows or create custom orchestration logic. This can be done by
importing the components and using each of their `.run()` components.

### Trainer Component Example:

```python
from gigl.src.training.trainer import Trainer

trainer = Trainer()
trainer.run(
    applied_task_identifier=job_name,
    task_config_uri="gs://...",
    resource_config_uri=="gs://...",
)
```

For component specific parameters/information, see [Components](../overview/components.md)
