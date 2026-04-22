# Trainer

The Trainer component is the entry point for model training in GiGL. It supports both the legacy tabularized pipeline
and the newer in-memory subgraph sampling path.

## Input

- **job_name** (AppliedTaskIdentifier): which uniquely identifies an end-to-end task.
- **task_config_uri** (Uri): Path which points to a "frozen" `GbmlConfig` proto yaml file - Can be either manually
  created, or `config_populator` component (recommended approach) can be used which can generate this frozen config from
  a template config.
- **resource_config_uri** (Uri): Path which points to a `GiGLResourceConfig` yaml

## What does it do?

The Trainer undertakes the following actions:

- Reads the frozen `GbmlConfig` and resource config.
- Cleans existing trainer output paths so retries do not mix old and new assets.
- Chooses the training backend:
  - Legacy tabularized path when `featureFlags.should_run_glt_backend` is not enabled.
  - In-memory subgraph sampling path when `featureFlags.should_run_glt_backend` is `True`.
- Launches the selected training runtime and persists output metadata such as model parameters and offline metrics.

### Legacy path

In the legacy path, the Trainer consumes the outputs of Split Generator and delegates to the v1 trainer stack.

### In-Memory Subgraph Sampling Path

In the in-memory path, the Trainer launches the distributed runtime used for live neighborhood sampling. At a high
level, that runtime:

- launches the user-provided training command from `trainerConfig.command`,
- uses Data Preprocessor outputs to build a `DistDataset` or `RemoteDistDataset`,
- samples neighborhoods online during training instead of consuming precomputed sampled subgraphs.

For link prediction, the reference training loops under `examples/link_prediction` use:

- `DistABLPLoader` for anchor-based link prediction batches,
- `DistNeighborLoader` for random negative batches.

Graph store mode uses the same conceptual flow, but separates storage and compute into different pools and exposes the
graph through `RemoteDistDataset`.

## How do I run it?

**Import GiGL**

```python
from gigl.src.training.trainer import Trainer
from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier

trainer = Trainer()

trainer.run(
    applied_task_identifier=AppliedTaskIdentifier("sample_job_name"),
    task_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml"),
    resource_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/resource_config.yaml"),
)
```

Note: If you are training on VertexAI and using a custom class, you will have to provide a docker image (Either
`cuda_docker_uri` for GPU training or `cpu_docker_uri` for CPU training.)

For in-memory subgraph sampling training, the component currently supports Vertex AI execution. The example training
scripts under `examples/link_prediction` can still be run directly for local experimentation with an already frozen task
config.

**Command Line**

```bash
python -m \
    gigl.src.training.trainer \
    --job_name="sample_job_name" \
    --task_config_uri="gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml" \
    --resource_config_uri="gs://MY TEMP ASSETS BUCKET/resource_config.yaml"
```

## Output

After the training process finishes:

- The Trainer saves the trained model’s `state_dict` at specified location (`trainedModelUri` field of
  `sharedConfig.trainedModelMetadata`).

- The trainer logs training metrics to `trainingLogsUri` field of `sharedConfig.trainedModelMetadata`. To view the
  metrics on your local, you can run the command: `tensorboard --logdir gs://tensorboard_logs_uri_here`

## Examples

Reference in-memory training implementations:

- [`examples/link_prediction/homogeneous_training.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/homogeneous_training.py)
- [`examples/link_prediction/heterogeneous_training.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/heterogeneous_training.py)
- [`examples/link_prediction/graph_store/homogeneous_training.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/graph_store/homogeneous_training.py)
- [`examples/link_prediction/graph_store/heterogeneous_training.py`](https://github.com/Snapchat/GiGL/blob/main/examples/link_prediction/graph_store/heterogeneous_training.py)

## Custom Usage

The customization point depends on the backend:

- Legacy path: training logic is provided through a `BaseTrainer` implementation.
- In-memory path: training logic is provided by the user command referenced in `trainerConfig.command`, such as the
  example scripts under `examples/link_prediction`.

## Other

### Torch Profiler

You can profile trainer performance metrics, such as gpu/cpu utilization by adding below to task_config.yaml

```
profilerConfig:
    should_enable_profiler: true
    profiler_log_dir: gs://path_to_my_bucket  (or a local dir)
    profiler_args:
        wait:'0'
        with_stack: 'True'
```

### Monitoring and logging

Once the trainer component starts, the training process can be monitored via the gcloud console under Vertex AI Custom
Jobs (`https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=<project_name_here>`). You can also view
the job name, status, jobspec, and more using `gcloud ai custom-jobs list --project <project_name_here>`

On the Vertex AI UI, you can see all the information like machine/acceleratior information, CPU Utilization, GPU
utiliization, Network data etc. Here, you will also find the "View logs" tab, which will open the Stackdriver for your
job which logs everything from your modeling task spec as the training progresses in real time.

If you would like to view the logs locally, you can also use:
`gcloud ai custom-jobs stream-logs <custom job ID> --project=<project_name_here> --region=<region here>`.

### Parameters

We provide some base class implementations for training. See:

- `gigl/src/common/modeling_task_specs/graphsage_template_modeling_spec.py`
- `gigl/src/common/modeling_task_specs/node_anchor_based_link_prediction_modeling_task_spec.py`
- `gigl/src/common/modeling_task_specs/node_classification_modeling_task_spec.py`

\*\*\*\* Note: many training/model params require dep on using the right model / training setup i.e. specific
configurations may not be supported - see individual implementations to understand how each param is used. Training
specs are fully customizable - these are only examples

The v1 modeling-task-spec implementations provide runtime arguments similar to below. We present examples of the args
for `node_anchor_based_link_prediction_modeling_task_spec.py` here. These are most relevant to the legacy path;
in-memory training scripts typically define their own runtime arguments in `trainerArgs`.

- Training environment parameters (number of workers for different dataloaders)

  - `train_main_num_workers`
  - `train_random_negative_num_workers`
  - `val_main_num_workers`
  - `val_random_negative_num_workers`
  - `test_main_num_workers`
  - `test_random_negative_num_workers`

  Note that training involves multiple dataloaders simultaneously. Take care to specify these parameters in a way which
  avoids overburdening your machine. It is recommended to specify
  `(train_main_sample_num_workers + train_random_sample_num_workers + val_main_sample_num_workers + val_random_sample_num_workers < num_cpus)`,
  and `(test_main_sample_num_workers + test_random_sample_num_workers < num_cpus)` to avoid training stalling due to
  contention.

- Modifying the GNN model:

  - Specified by arg `gnn_model_class_path`
    - Some Sample GNN models are defined [here](/gigl/src/common/models/pyg/homogeneous.py) and initialized in the
      `init_model` function in ModelingTaskSpec. When trying different GNN models, it is recommended to also include the
      new GNN architectures under the same file and declare them as is currently done. This cannot currently be done
      from the default `GbmlConfig` yaml.

- Non Exhaustive list of Model parameters:

  - `hidden_dim`: dimension of the hidden layers
  - `num_layers`: number of layers in the GNN
  - `out_channels`: dimension of the output embeddings
  - `should_l2_normalize_embedding_layer_output`: whether apply L2 normalization on the output embeddings

- Non Exhaustive list of Training parameters:

  - `num_heads`
  - `val_every_num_batches`: validation frequence per training batches
  - `num_val_batches`: number of validation batches
  - `num_test_batches`: number of testing batches
  - `optim_class_path`: defaults to "torch.optim.Adam"
  - `optim_lr`: learning rate of the optimizer
  - `optim_weight_decay`: weight decay of the optimizer
  - `clip_grad_norm`
  - `lr_scheduler_name`: defaults to "torch.optim.lr_scheduler.ConstantLR"
  - `factor`: param for lr scheduler
  - `total_iters`: param for lr scheduler
  - `main_sample_batch_size`: training batch size
  - `random_negative_sample_batch_size`: random negative sample batch size for training
  - `random_negative_sample_batch_size_for_evaluation`: random negative sample batch size for evaluation
  - `train_main_num_workers`
  - `val_main_num_workers`
  - `test_main_num_workers`
  - `train_random_negative_num_workers`
  - `val_random_negative_num_workers`
  - `test_random_negative_num_workers`
  - `early_stop_criterion`: defaults to "loss"
  - `early_stop_patience`: patience for earlystopping
  - `task_path`: python class path to supported training tasks i.e. Retrieval
    `gigl.src.common.models.layers.task.Retrieval`; see gigl.src.common.models.layers.task.py for more info
  - `softmax_temp`: temperature parameter in the `softmax` loss
  - `should_remove_accidental_hits`

### Background for distributed training

Trainer currently uses PyTorch distributed training abstractions to enable multi-node and multi-GPU training. Some
useful terminology and links to learn about these abstractions below.

- **WORLD**: Group of processes/workers that are used for distributed training.

- **WORLD_SIZE**: The number of processes/workers in the distributed training WORLD.

- **RANK**: The unique id (usually index) of the process/worker in the distributed training WORLD.

- **Data loader worker**: A worker used specifically for loading data; if the dataloader worker is utilizing the same
  thread/process as a worker in distributed training WORLD, then we may incur blocking execution of training, resulting
  in slowdowns.

- **[Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)**: Pytorch's version of
  [Data parallalism](https://en.wikipedia.org/wiki/Data_parallelism) across different **processes** (could even be
  processes on different machines), to speed up traiing on large datasets.

- **[TORCH.DISTRIBUTED package](https://pytorch.org/docs/stable/distributed.html)**: A torch package containing tools
  for distributed communication and trainings.

  - Defines [backends for distributed communication](https://pytorch.org/docs/stable/distributed.html#backends) like
    `gloo` and `nccl` - as a ML practitioner you should not worry about how these work, but important to know what
    **devices** and **collective functions** they support.
  - Contains **"[Collective functions](https://pytorch.org/docs/stable/distributed.html#collective-functions)"** like
    `torch.distributed.broadcast`, `torch.distributed.all_gather`, et al. which allow communication of tensors across
    the **WORLD**.
