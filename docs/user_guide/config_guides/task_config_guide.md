# Task Config Guide

The task config specifies task-related configurations - guiding the behavior of components according to the needs of
your machine learning task.

Whenever we say "task config" we are talking about an instance off
{py:class}`snapchat.research.gbml.gbml_config_pb2.GbmlConfig`

<details>
<summary><bold>This is a protobuff class whose definition can be found in gbml_config.proto:</bold></summary>

```{literalinclude} ../../../proto/snapchat/research/gbml/gbml_config.proto
:language: proto
```

</details>

Just like [resource config](./resource_config_guide.md), the values to instantiate this proto class are usually provided
as a `.yaml` file. Most [components](../overview/architecture.md#components) accept the task config as an argument
`--task_config_uri` - i.e. a {py:class}`gigl.common.Uri` pointing to a `task_config.yaml` file.

## Example

We will use the MAG240M task config to walk you through what a config may look like.

<details>
<summary><bold>Full task config for reference:</bold></summary>

```{literalinclude} ../../../examples/MAG240M/task_config.yaml
:language: yaml
```

</details>

### GraphMetadata

We specify what are all the nodes and edges in the graph. In this case we have one node type: `paper_or_author`. And,
one edge type: `(paper_or_author, references, paper_or_author)`

Note: In this example we have converted the hetrogeneous MAG240M dataset to a homogeneous one with just one edge and one
node; which we will be doing self supervised learning on.

```{literalinclude} ../../../examples/MAG240M/task_config.yaml
:language: yaml
:start-after: GraphMetadata
:end-before: ========
```

### TaskMetadata

Now we specify what type of learning task we want to do. In this case we want to leverage Node Anchor Based Link
Prediction to do self supervised learning on the edge: `(paper_or_author, references, paper_or_author)`. Thus, we are
using the `NodeAnchorBasedLinkPredictionTaskMetadata` task.

```{literalinclude} ../../../examples/MAG240M/task_config.yaml
:language: yaml
:start-after: TaskMetadata
:end-before: ========
```

```{note}
An example of `NodeBasedTaskMetadata` can be found in `python/gigl/src/mocking/configs/e2e_supervised_node_classification_template_gbml_config.yaml`
```

### SharedConfig

Shared config are parameters that are common and may be used across multiple components i.e. Trainer, Inferencer,
SubgraphSampler, etc.

```{literalinclude} ../../../examples/MAG240M/task_config.yaml
:language: yaml
:start-after: SharedConfig
:end-before: ========
```

### DatasetConfig

We create the dataset that we will be using. In this example we will be using the class `dataPreprocessorConfigClsPath`
to read and preprocess the data. See [Preprocessor Guide](../overview/components/data_preprocessor.md).

Once we have the data preprocessed, we will be tabularizing the data with the use of
[Subgraph Sampler](../overview/components/data_preprocessor.md)Specifically, for each node we will be sampling their
`numHops` neighborhood, where each hop will sample `numNeighborsToSample` neighbors. As well, we will be sampling
`numUserDefinedPositiveSamples` positive samples and their respective neighborhood using `numHops` and
`numNeighborsToSample`.

Subsequently, we will be creating test/train/val splits based on the %'s specified, using
[Split Generator](../overview/components/split_generator.md)

```{literalinclude} ../../../examples/MAG240M/task_config.yaml
:language: yaml
:start-after: DatasetConfig
:end-before: ========
```

### TrainerConfig

The class specified by `trainerClsPath` will be initialized and all the arguments specified in `trainerArgs` will be
directly passed as `**kwargs` to your trainer class. Thes only requirement is the trainer class implement the protocol
defined @ {py:class}`gigl.src.training.v1.lib.base_trainer.BaseTrainer`.

Some common sense pre-configured trainer implementations can be found in {py:class}`gigl.src.common.modeling_task_specs`
. Although, you are recommended to implement your own.

```{literalinclude} ../../../examples/MAG240M/task_config.yaml
:language: yaml
:start-after: TrainerConfig
:end-before: ========
```

### InferencerConfig

Similar to Trainer, the class specified by `inferencerClsPath` will be initialized and all arguments specified in
`inferencerArgs` will be directly passed in `**kwargs` to your inferencer class. The only requirement is the inferencer
class implement the protocol defined @ {py:class}`gigl.src.inference.v1.lib.base_inferencer.BaseInferencer`

```{literalinclude} ../../../examples/MAG240M/task_config.yaml
:language: yaml
:start-after: InferencerConfig
:end-before: ========
```
