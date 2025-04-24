## Split Generator

The Split Generator reads localized subgraph samples produced by Subgraph Sampler, and executes logic to split the data
into training, validation and test sets. The semantics of which nodes and edges end up in which data split depends on
the particular semantics of the splitting strategy.

## Input

- **job_name** (AppliedTaskIdentifier): which uniquely identifies an end-to-end task.
- **task_config_uri** (Uri): Path which points to a "frozen" `GbmlConfig` proto yaml file - Can be either manually
  created, or `config_populator` component (recommended approach) can be used which can generate this frozen config from
  a template config.
- **resource_config_uri** (Uri): Path which points to a `GiGLResourceConfig` yaml

Optional Development Args:

- **cluster_name** (str): Optional param if you want to re-use a cluster for development
- **skip_cluster_delete** (bool): Provide flag to skip automatic cleanup of dataproc cluster
- **debug_cluster_own_alias** (str): Add alias to cluster

## What does it do?

The Split Generator undertakes the following actions:

- Reads frozen `GbmlConfig` proto yaml, which contains a pointer to an instance of a `SplitStrategy` class (see
  `splitStrategyClsPath` field of `datasetConfig.splitGeneratorConfig`), and an instance of an `Assigner` class (see
  `assignerClsPath` field of `datasetConfig.splitGeneratorConfig`). These classes house logic for constructs which
  dictate how to assign nodes and/or edges to different buckets, which are then utilized to assign these objects to
  training, validation and test sets accordingly. See the currently supported strategies in:
  `scala/splitgenerator/src/main/scala/lib/split_strategies/*`

  Custom arguments can also be passed into the `SplitStrategy` class (`Assigner` class) by including them in the
  `splitStrategyArgs` (`assignerArgs`) field(s) inside `datasetConfig.splitGeneratorConfig` section of `GbmlConfig`.
  Several standard configurations of `SplitStrategy` and corresponding `Assigner` classes are implemented already at a
  GiGL platform-level: transductive node classification, inductive node classification, and transductive link prediction
  split routines, as detailed here.

- The component kicks off a Spark job which read samples produced by the Subgraph Sampler component, which are stored at
  URIs referenced inside the `sharedConfig.flattenedGraphMetadata` section of the frozen `GbmlConfig`. Note that
  depending on the `taskMetadata` in the `GbmlConfig`, the URIs will be housed under different keys in this section; for
  example, given the Node-anchor Based Link Prediction setting used in the sample frozen `GbmlConfig` MAU yaml, we can
  find the Subgraph Sampler outputs under the `nodeAnchorBasedLinkPredictionOutput` field. Upon reading the outputs from
  Subgraph Sampler, the Split Generator component executes methods defined in the provided SplitStrategy instance on
  each of the input samples. The pipeline writes out TFRecord samples with appropriate data meant to be visible in
  training, validation and test sets to GCS.

## How do I run it?

Firstly, you can adjust the following parameters in the `GbmlConfig`:

```
  splitGeneratorConfig:
    assignerArgs:
      seed: '42'
      test_split: '0.2'
      train_split: '0.7'
      val_split: '0.1'
    assignerClsPath: splitgenerator.lib.assigners.TransductiveEdgeToLinkSplitHashingAssigner
    splitStrategyClsPath: splitgenerator.lib.split_strategies.TransductiveNodeAnchorBasedLinkPredictionSplitStrategy
```

**Import GiGL**

```python
from gigl.src.split_generator.split_generator import SplitGenerator
from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier

split_generator = SplitGenerator()

split_generator.run(
    applied_task_identifier=AppliedTaskIdentifier("sample_job_name"),
    task_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml"),
    resource_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/resource_config.yaml")
)
```

**Command Line**

```
python -m gigl.src.split_generator.split_generator \
  --job_name"sample_job_name" \
  --task_config_uri="gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml"
  --resource_config_uri="gs://MY TEMP ASSETS BUCKET/resource_config.yaml"
```

The python entry point `split_generator.py` performs the following:

- Create a Dataproc cluster suitable for the scale of the graph at hand,
- Install Spark and Scala dependencies,
- Run the Split Generator Spark job,
- Delete the Dataproc cluster after the job is finished.

**Optional Arguments**: Provide a custom cluster name so you can re-use it instead of having to create a new one every
time.

```
  --cluster_name="unique_name_for_the_cluster"
```

Ensure to skip deleting the cluster so it can be re-used. But, be sure to clean up manually after to prevent $ waste.

```
  --skip_cluster_delete
```

Marks cluster is to be used for debugging/development by the alias provided. i.e. for username some_user, provide
debug_cluster_owner_alias="some_user"

```
  --debug_cluster_owner_alias="your_alias"
```

*Example for when you would want to use cluster for development:*

```
python -m gigl.src.split_generator.split_generator \
  --job_name sample_job_name \
  --task_config_uri "gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml"
  --resource_config_uri="gs://MY TEMP ASSETS BUCKET/resource_config.yaml"
  --cluster_name="unique-name-for-the-cluster"\
  --skip_cluster_delete \
  --debug_cluster_owner_alias="$(whoami)"
```

## Output

Upon completing the Dataflow job referenced in the last bullet point of the [What Does it Do](#what-does-it-do) section,
the Split Generator writes out TFRecord samples belonging to each of the training, validation and test sets to URIs
which are referenced in `sharedConfig.datasetMetadata` section of the `GbmlConfig`. Based on the `taskMetadata` in the
`GbmlConfig`, the outputs will be written to different keys within this section. Given the sample configs for the MAU
task referenced here, they are written to URIs referenced at the `NodeAnchorBasedLinkPredictionDataset` field.

## Custom Usage

- To customize the semantics of the splitting method desired, users can manipulate arguments passed to existing
  `Assigner` and `SplitStrategy` class instances, or even write their own. The instances provided reflect "standard"
  splitting techniques in graph ML literature, which can be tricky to implement, so caution is advised in trying to
  customize or write modified variants, in order to avoid leaking data between training, validation and test sets.

- Currently, all `SplitStrategy` instances leverage `HashingAssigner` (a specialized `Assigner` in which nodes / edges
  are assigned to different buckets randomly, reflecting random splits). In the future, we can consider introducing new
  `Assigner` policies to reflect temporal splitting.

## Other

- **Design**: Graph ML data splitting is tricky. Please see
  [here](http://snap.stanford.edu/class/cs224w-2020/slides/09-theory.pdf) for a good academic reference into how
  splitting is standardly conducted to avoid leakage. We chose to create abstractions around splitting which reflect
  flexible policies around assignment of nodes and/or edges to different buckets, from which defining the visible data
  during training, validation and testing becomes deterministic.

This component runs on Spark. Some info on monitoring this job:

- The list of all jobs/clusters is available on [Dataproc UI](https://cloud.google.com/dataproc?hl=en), and we can
  monitor the overall Spark job statuses and configurations.

- While the cluster is running, we can access Spark UI's WEB INTERFACES tab to monitor each stage of the job in more
  detail.
