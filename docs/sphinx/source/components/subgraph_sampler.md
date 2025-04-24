# Subgraph Sampler

The Subgraph Sampler receives node and edge data from Data Preprocessor and mainly generates k-hop localized subgraphs
for each node in the graph. Basically, the Subgraph Sampler enables us to store the computation graph of each node
independently without worrying about maintaining a huge graph in memory for down-stream components. It uses Spark/Scala
and runs on a Dataproc cluster. Based on the predefined sample schema for each task, the output samples are
serialized/saved in TFRecord format.

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

The Subgraph Sampler, supports localized neighborhood sampling for homogeneous and heterogeneous graphs, where subgraph
edges can be sampled with the following strategies: random uniform, top-k, weighted random, or customized sampling
strategies.

The Subgraph Sampler performs the following steps:

- Reads frozen `GbmlConfig` proto yaml to get

  - `preprocessedMetadataUri` to read relevant node and edge metadata such as feature names, node id key and path to
    TFRecords that store node and edge data obtained from the Data Preprocessor.
  - `flattenedGraphMetadata` which includes the URI for storing the Subgraph Sampler outputs
  - `subgraphSamplerConfig`

- Converts node/edge TFRecords to DataFrames

- Samples k-hop neighbors for all nodes according to the `subgraphSamplingStrategy` provided in config

- Hydrates the sampled neighborhoods (with node/edge features)

- If the task is NodeAnchorBasedLinkPrediction, it will sample positive edges and positive node neighborhoods for each
  root node

- Converts final DataFrames to TFRecord format based on the predefined schema in protos.

## How do I run it?

**SubgraphSamplerConfig**

Firstly, you can adjust the `subgraphSamplerConfig` parameters in the `GbmlConfig`.

- `SubgraphSamplingStrategy` allows customization of subgraph sampling operations by the user on a config level.
- Users can specify each step of the `messagePassingPaths` through `samplingOp`.
- Each `samplingOp` has `inputOpNames` where you can specify the parent of the `samplingOp`.
- The `samplingOp` essentially forms a DAG of edge types to sample, indicating how we should construct our sampled k-hop
  message passing graph, one for each root node type.
- (Note: Note: Only node types which exist in `supervision_edge_types` need their own `MessagePassingPaths` define, see
  [task_config_guide](../user_guide/config_guides/task_config_guide.md) for more details)
- We currently support the following sampling methods in `samplingOp`:
  - `randomUniform`: Random sample
  - `topK`: Sample top K, based on `edgeFeatName`
  - `weightedRandom`: Sample nodes based on a specified weight from `edgeFeatName`
  - `custom`: Custom sampling strategy. Users can implement their own custom sampling method.
- New `SubgraphSamplingStrategy` can also be introduced in addition to `MessagePassingPathStrategy`
  `GlobalRandomUniformStrategy`, for example, [Pixie](https://cs.stanford.edu/people/jure/pubs/pixie-www18.pdf) random
  walk sampling.

Example of `SubgraphSamplingStrategy` for heterogeneous graph with 2 edge types (user, to, story) and (story, to, user)
that does 2-hop sampling.

```yaml
subgraphSamplerConfig:
    subgraphSamplingStrategy:
          messagePassingPaths:
            paths:
            - rootNodeType: user
              samplingOps:
              - edgeType:
                  dstNodeType: user
                  relation: to
                  srcNodeType: story
                opName: sample_stories_from_user
                randomUniform:
                  numNodesToSample: 10
              - edgeType:
                  dstNodeType: story
                  relation: to
                  srcNodeType: user
                inputOpNames:
                - sample_stories_from_user
                opName: sample_users_from_story
                randomUniform:
                  numNodesToSample: 10
            - rootNodeType: story
              samplingOps:
              - edgeType:
                  dstNodeType: story
                  relation: to
                  srcNodeType: user
                opName: sample_users_from_story
                randomUniform:
                  numNodesToSample: 10
              - edgeType:
                  dstNodeType: user
                  relation: to
                  srcNodeType: story
                inputOpNames:
                - sample_users_from_story
                opName: sample_stories_from_user
                randomUniform:
                  numNodesToSample: 10
```

Example of `SubgraphSamplingStrategy` for a user - user homogeneous graph that does 2-hop sampling.

```yaml
subgraphSamplerConfig:
    subgraphSamplingStrategy:
          messagePassingPaths:
            paths:
            - rootNodeType: user
              samplingOps:
              - edgeType:
                  dstNodeType: user
                  relation: is_friends_with
                  srcNodeType: user
                opName: sample_first_hop_friends
                randomUniform:
                  numNodesToSample: 10
              - edgeType:
                  dstNodeType: user
                  relation: is_friends_with
                  srcNodeType: user
                inputOpNames:
                - sample_friends
                opName: sample_second_hop_friends
                randomUniform:
                  numNodesToSample: 10
```

(2024 Aug) We support two backends for Subgraph Sampling: GraphDB-based and Pure-Spark. These solutions have different
implications in flexibility, cost-scaling, and relevance for different applications. As of Aug 2024, for heterogeneous
subgraph sampling, a graphDB backend must be used, while for homogeneous subgraph sampling, both backends may be used.
Enabling parity between these two is work-in-progress.

An example of specifying the `subgraphSamplerConfig` to use the graphDB backend with Nebula graph-DB is

```yaml
subgraphSamplerConfig:
    graphDbConfig:
      graphDbArgs:
        port: 9669
        hosts: xxx.xxx.xxx.xxx
        graph_space: MY_GRAPH_SPACE
```

An example of specifying the `subgraphSamplerconfig` to use the Pure-Spark backend:

```yaml
 subgraphSamplerConfig:
   numNeighborsToSample: 10
   numPositiveSamples: 3
```

**Import GiGL**

```python
from gigl.src.split_generator.split_generator import SplitGenerator
from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier

subgraph_sampler = SubgraphSampler()

subgraph_sampler.run(
    applied_task_identifier=AppliedTaskIdentifier("sample_job_name"),
    task_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml"),
    resource_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/resource_config.yaml")
)
```

**Command Line**

```
python -m gigl.src.subgraph_sampler.subgraph_sampler \
  --job_name="sample_job_name" \
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
  --job_name="sample_job_name" \
  --task_config_uri="gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml"
  --resource_config_uri="gs://MY TEMP ASSETS BUCKET/resource_config.yaml"
  --cluster_name="unique-name-for-the-cluster"\
  --skip_cluster_delete \
  --debug_cluster_owner_alias="$(whoami)"
```

## Output

Upon completion of the Spark job, subgraph samples are stored in the URIs defined in `flattenedGraphMetadata` field in
frozen `GbmlConfig`.

For example, for the Node Anchor Based Link Prediction task, we will have two types of samples referenced in
`nodeAnchorBasedLinkPredictionOutput`:

- `tfrecordUriPrefix` which includes main samples in `NodeAnchorBasedLinkPredictionSample` protos which contain an
  anchor node and positive samples with respective neighborhood information.

- `randomNegativeTfrecordUriPrefix` which includes negative samples in `RootedNodeNeighborhood` protos which contain
  anchor node and respective neighborhood information.

## How do I extend business logic?

It is not intended that core Subgraph Sampler logic be extended by end users.

For example, if you want to implement a new sampling strategy, you can add a new `SamplingOp` to the
`subgraph_sampling_strategy.proto` and add implementation of the logic custom query translation class.

## Other

This component runs on Spark. Some info on monitoring this job:

- The list of all jobs/clusters is available on [Dataproc UI](https://console.cloud.google.com/dataproc/), and we can
  monitor the overall Spark job statuses and configurations.

- While the cluster is running, we can access Spark UI's WEB INTERFACES tab to monitor each stage of the job in more
  detail.
