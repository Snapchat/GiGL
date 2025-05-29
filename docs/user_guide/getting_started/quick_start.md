# Quick Start

GiGL is a flexible framework that allows customization for many graph ML tasks in its components like data data
pre-processing, training logic, inference.

This page outlines the steps needed to get up and running an end to end pipeline in different scenarios, starting from a
simple local setup to more complex cloud-based operations.

## 1. Install GiGL

Before proceeding, make sure you have correctly installed `gigl` by following the
[installation guide](./installation.md).

## 2. Setup your Cloud Environment

For GiGL to function you need to se up a few cloud resources including a Service Account, GCS Buckets, BQ tables and
relevant permissions. Please follow instructions under the [Cloud Setup Guide](./cloud_setup_guide.md)

## 3. Config Setup

To run an end to end pipeline in GiGL, two configs are required: `resource config`, and `task config`. The
`resource config` specifies the resource and environment configurations for each component in the GiGL. Whereas the
`task config` specifies task-related configurations - guiding the behavior of components according to the needs of your
machine learning task.

**Resource Config**:

The resource config contains GCP project specific information (service account, buckets, etc.) as well as GiGL Component
resource allocation. You will find some resource configs already in the repo, but these are either configured to run on
our CI/CD systems, or not completely filled - meaning you will not be able to use them directly.

We will bootstrap a resource config to get you started using the `bootstrap_resource_config.py` script. The script
creates a copy off the `deployment/configs/unittest_resource_config.yaml` config and swaps the compute resources to
point them to resources you created when you [Setup your Cloud Environment](#2-setup-your-cloud-environment) - ensure
you have done this before proceeding.

Run the following command and follow the steps:

```bash
python scripts/bootstrap_resource_config.py
```

You will note that if the script finishes successfully, it will have added three environment variables to your main
shell file i.e. (`~/.zshrc`); mainly `GIGL_TEST_DEFAULT_RESOURCE_CONFIG`, `GIGL_PROJECT`, and
`GIGL_DOCKER_ARTIFACT_REGISTRY`. Ensure vars are available (you may need to restart shell)

```bash
echo $GIGL_TEST_DEFAULT_RESOURCE_CONFIG
echo $GIGL_PROJECT
echo $GIGL_DOCKER_ARTIFACT_REGISTRY
```

For detailed information on `resource config`, see our
[resource config guide](../config_guides/resource_config_guide.md).

**Task Config**:

The template task config is for populating custom class paths, custom arguments, and data configuations which will be
passed into config populator. For task config usage/spec creation, see the
[task_config_guide](../config_guides/task_config_guide.md).

## 4. Running an End To End GiGL Pipeline

```{caution}
Since `.whl`s for GiGL have not been released yet, using GiGL workflows currently follows the same process as you would kick them off if developing GiGL. This is expected to change once wheels are available.
```

GiGL supports various ways to orchestrate an end to end run such as KFP Orchestration, GiGL Runner, and manual component
import and running as needed. For more details see [here](./orchestration.md)

Lets use the following command to run an e2e link prediction example on the Cora dataset:

```bash
export GIGL_CORA_NABLP_TASK_CONFIG="gigl/src/mocking/configs/e2e_node_anchor_based_link_prediction_template_gbml_config.yaml"
make \
  job_name="$(whoami)_gigl_hello_world_cora_nalp" \
  task_config_uri="$GIGL_CORA_NABLP_TASK_CONFIG" \
  resource_config_uri="$GIGL_TEST_DEFAULT_RESOURCE_CONFIG" \
  start_at="config_populator" \
  run_dev_gnn_kubeflow_pipeline
```

If the pipeline ran successfully, you should see a url to Vertex AI where your pipeline is running.

Observe that once you run this command a few things happen:

1. The relevant jars are compiled
2. Docker images are built from the GiGL source code and uploaded to your project
3. A KFP pipeline is compiled with references to the relevant jars, and docker iamges
4. The compiled pipeline runs on Vertex AI on your project w/ the task and resource configs provided.

## Digging Deeper and Advanced Usage

Now that you have an idea on how GiGL works, you may want to explore advanced customization options for your specific
tasks. This section directs you to various guides that detail how to create and modify task specifications, use custom
data, and general customization:

- **Task Spec Customization**: For any custom logic needed at the component level, like pulling your own data, writing
  custom training/inference logic, or task specific arguments, see the
  [task_config_guide](../config_guides/task_config_guide.md).

- **Behind the Scenes**: To better understand how each of GiGL's components interact and operate, see the
  [components page](../overview/architecture.md)

- **Examples**: For easy references and make your next steps easier, various example walkthroughs are available on the
  examples page. See [here](../examples/index.md)
