# Quick Start Guide

GiGL is a flexible framework that allows customization for many graph ML tasks in its components like data data
pre-processing, training logic, inference.

This page outlines the steps needed to get up and running an end to end pipeline in different scenarios, starting from a
simple local setup to more complex cloud-based operations.

## Install GiGL

Before proceeding, make sure you have correctly installed `gigl` by following the
[installation guide](./installation.md).

## Quick Start: Local

:::{admonition} Note :class: note

Not yet supported :::

Running an end to end GiGL pipeline on your local machine is not yet supported. This would help provide a quick way to
test functionalities without any cloud dependencies. Stay tuned for future releases!

## Quick Start: Running Distributed on Cloud

This section outlines the steps needed to get up and running an end to end pipeline (via GCP) on an in-built sample
task: training and inference for transductive node-anchor based link prediction task on Cora (homogeneous). Running in
the cloud enables the usage of services like Dataproc, VertexAI, and Dataflow to enable large-scale
distributed/optimized performance.

### Config Setup

To run an end to end pipeline in GiGL, two config files are required. In this guide, some samples/templates are provided
to get started but these would need to be modified as needed for custom tasks.

**Resource Config**:

The resource config contains GCP project specific information (service account, buckets, etc.) as well as GiGL Component
resource allocation. To setup cloud resources for `shared_resource_config.common_compute_config`, see
[cloud setup guide](./cloud_setup_guide.md).

Once you have setup your cloud project, you can populate the `common_compute_config` section of the template resource
config provided in `GiGL/docs/examples/template_resource_config.yaml`. The remainder of the resource config is populated
with some pre-defined values which should be suitable for this task.

For more information, see [resource config guide](../config_guides/resource_config_guide.md).

**Task Config**:

The template task config is for populating custom class paths, custom arguments, and data configuations which will be
passed into config populator. For task config usage/spec creation, see the
[task_config_guide](../config_guides/task_config_guide.md).

In this guide, we will be using a built-in preprocessor config to run on one of GiGL's supported mocked datasets (in
this specific example `cora_homogeneous_node_anchor_edge_features`). The other supported mocked datasets can be seen in
`python/tests/test_assets/dataset_mocking/lib/mocked_dataset_artifact_metadata.json`

The path for the template task config is:
`python/tests/test_assets/dataset_mocking/pipeline_test_assets/configs/e2e_node_anchor_based_link_prediction_template_gbml_config.yaml`

### Running an End To End GiGL Pipeline

Now that we have our two config files setup, we can now kick off an end to end GiGL run.

GiGL supports various ways to orchestrate an end to end run such as Kubeflow Orchestration, GiGL Runner, and manual
component import and running as needed. For more details see [here](./orchestration.md)

Below is an example on how we can use GiGL Runner and use the specs we created above.

```python

from gigl.orchestration.local.runner import Runner, PipelineConfig
from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier

pipeline_config = PipelineConfig(
    applied_task_identifier=AppliedTaskIdentifier("my-first-gigl-job"),
    template_task_config_uri=UriFactory.create_uri("python/tests/test_assets/dataset_mocking/pipeline_test_assets/configs/e2e_node_anchor_based_link_prediction_template_gbml_config.yaml"),
    resource_config_uri=UriFactory.create_uri("path-to-your-resource-config.yaml"),
)

runner = Runner(pipeline_config=pipeline_config)
runner.run()

```

## Quick Start: Digging Deeper and Advanced Usage

Now that you have an idea on how GiGL works, you may want to explore advanced customization options for your specific
tasks. This section directs you to various guides that detail how to create and modify task specifications, use custom
data, and general customization:

- **Task Spec Customization**: For any custom logic needed at the component level, like pulling your own data, writing
  custom training/inference logic, or task specific arguments, see the
  [task_config_guide](../config_guides/task_config_guide.md).

- **Behind the Scenes**: To better understand how each of GiGL's components interact and operate, see the
  [components page](../overview/components.md)

- **Examples**: For easy references and make your next steps easier, various example walkthroughs are available on the
  examples page. See [here](../examples/)
