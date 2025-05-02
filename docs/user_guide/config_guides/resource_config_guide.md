# Resource Config Guide

GiGL Resource Config is a yaml file that is passed in at runtime and specifies the resource and environment
configurations for each component in the GiGL. The proto definition for GiGL Resource Config can be seen
[here](https://github.com/Snapchat/GiGL/blob/main/proto/snapchat/research/gbml/gigl_resource_config.proto)

## Prerequisites

If you don't have cloud assets already setup i.e. a GCP project. See [guide](../getting_started/cloud_setup_guide.md)

## Resource Config Breakdown

## Example: Full Template for Resource Config

<details>
<summary>See Full Template Resource Config</summary>
    <code>
        <pre>
shared_resource_config:
  resource_labels:
    cost_resource_group_tag: ""
    cost_resource_group: ""
  common_compute_config:
    project: "project_id"
    region: "gcp_region_here"
    temp_assets_bucket: "gs://"
    temp_regional_assets_bucket: "gs://"
    perm_assets_bucket: "gs://"
    temp_assets_bq_dataset_name: "bq_dataset_name_here"
    embedding_bq_dataset_name: "bq_dataset_name_here"
    gcp_service_account_email: "service_account_email_here"
    k8_service_account: "service_account_name_here"
    dataflow_worker_harness_image: "gcr.io/..."
    dataflow_runner: "" # DataflowRunner or DirectRunner
preprocessor_config:
  edge_preprocessor_config:
    num_workers: 1
    max_num_workers: 2
    machine_type: "" # e.g. n1-highmem-32
    disk_size_gb: 100
  node_preprocessor_config:
    num_workers: 1
    max_num_workers: 2
    machine_type: "" # e.g. n1-highmem-64
    disk_size_gb: 100
subgraph_sampler_config:
  machine_type: "" # e.g. n1-highmem-32
  num_local_ssds: 1
  num_replicas: 1
split_generator_config:
  machine_type: "" # e.g. n1-highmem-32
  num_local_ssds: 1
  num_replicas: 1
trainer_config:
  vertex_ai_trainer_config:
    machine_type: "" # e.g. n1-highmem-16
    gpu_type: "" # e.g. nvidia-tesla-p100
    gpu_limit: 1
    num_replicas: 1
inferencer_config:
  num_workers: 1
  max_num_workers: 2
  machine_type: "" # e.g. n1-highmem-16
  disk_size_gb: 100
        </pre>
    </code>
</details>

**Shared Resource Config**

The `shared_resource_config` field includes settings that apply across all GiGL components. You need to customize this
section according to your GCP project specifics.

- **Resource Labels**: Resource labels help you manage costs and organzie resources. Modify the `resource_labels`
  section to fit your project's labeling scheme.

- **Common Compute Config**: This section includes important project specifications. Fill out the fields with your
  project ID, region, asset buckets, and service account email.

```yaml
common_compute_config:
  project: "your-gcp-project-id"
  region: "your-region"
  temp_assets_bucket: "gs://your-temp-bucket"
  perm_assets_bucket: "gs://your-permanent-bucket"
  gcp_service_account_email: "your-service-account-email"
```

**Preprocessor Config**

The `preprocessor_config` specifies settings for the Dataflow preprocessor component, includes number of workers,
machine type, and disk size. You must specify both the `node_preprocessor_config` and `edge_preprocessor_config`. See
example:

```yaml
preprocessor_config:
  edge_preprocessor_config:
    num_workers: 1
    max_num_workers: 2
    machine_type: "n1-highmem-32"
    disk_size_gb: 100
  node_preprocessor_config:
    num_workers: 1
    max_num_workers: 2
    machine_type: "n1-highmem-32"
    disk_size_gb: 100
```

**Subgraph Sampler Config**

The `subgraph_sampler_config` specifies settings for the Spark subgraph sampler component, includes machine type, local
SSDs, and number of replicas. See example:

```yaml
subgraph_sampler_config:
  machine_type: "n1-standard-4"
  num_local_ssds: 1
  num_replicas: 2
```

**Split Generator Config**

The `split_generator_config` specifies settings for the Spark split generator component, includes machine type, local
SSDs, and number of replicas

```yaml
split_generator_config:
  machine_type: "n1-standard-4"
  num_local_ssds: 1
  num_replicas: 2
```

**Trainer Config**

The `trainer_config` specifies settings for the trainer config, currently supporting Vertex AI training or Local
Training.

- **Vertex AI Trainer Config**: The `vertex_ai_trainer_config` field of the trainer config requires a machine type, GPU
  type, GPU limit, and number of replicas. See example:

  ```yaml
  trainer_config:
    vertex_ai_trainer_config:
      machine_type: "n1-standard-8"
      gpu_type: "nvidia-tesla-t4"
      gpu_limit: 1
      num_replicas: 1
  ```

- **Local Trainer Config**: The `local_trainer_config` field of the trainer config just requires `num_workers` which can
  be used for data loaders.

**Inferencer Config**

The `inferencer_config` specifies settings for the Dataflow preprocessor component, includes number of workers, machine
type, and disk size. See example:

```yaml
inferencer_config:
  num_workers: 1
  max_num_workers: 256
  machine_type: "c2-standard-16"
  disk_size_gb: 100
```
