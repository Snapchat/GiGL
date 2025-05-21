## Notes for developers

1. [SGS as of now](#sgs)
1. [Scalability](#scalability)
1. [Resources](#resources)

## SGS

### Set log level

To silence the worker logs

1. Create log4j.properties file from template, under `/scala_spark35` dir, do
   `cp ../tools/scala/spark-3.5.0-bin-hadoop3/conf/log4j2.properties.template ../tools/scala/spark-3.5.0-bin-hadoop3/conf/log4j2.properties`
1. Update the first line in `log4j.properties` to `rootLogger.level = WARN rootLogger.appenderRef.stdout.ref = console`

### Steps:

```
1. Load node/edge TFRecord resources into Spark DataFrame
2. Sample 1hop neighbors (only node ids)
3. Sample 2hop neighbors (only node ids)
4. Hydrate kth hop neighbors with both node and edge features/type
5. Merge 1hop and 2hop hydrated neighbors
6. Merge hydrated root node to step 5, and create subgraphDF
7. Append isolated nodes (if any) to subgraphDF [RootedNodeNeighnorhood]
8. Add task-relevent samples to subgraphDF (such as positive node neighborshoods or node labels) to create trainingSubgraphDF
9. (If specified) append isolated nodes to trainingSubgraphDF [SupervisedNodeClassificationSample,NodeAnchorBasedLinkPredictionSample ]
10. Modify subgraphDF and trainingSubgraphDF schema to compatible structure as defined in `training_samples_schema.proto`.
11. Convert DataFrames from step 10 to DataSet and map DataSet rows to ByteArray
12. Write serialized DataSet to TFRecord
```

### Properties:

- **Isolated nodes**

  are always included in RootedNodeNeighnorhood samples. They can be included in training samples if the assigned flag
  for isolated nodes is set to true.

- **Self loops**

  are removed in preprocessor. They can be added in trainer if users wish to, but SGS is not concerned with self loops.

- **Num hops**

  is hard-coded to k=2. But the code is extendable to k>2, only Spark optimizations are bottle-neck since as the number
  of hops grow neighborhood size grows exponentially.

- **Graph type**

  only supports homogeneous graphs.

  supports both undirected and directed graphs.

- **Neighborhood sampling**

  uniform sampling.

  - note that there are two implementations for uniform sampling:
    1. non-deterministic (using built-in Spark functions), which is the default mode of sampling in SGS
    1. deterministic (using hash based permutation). To enable it, set
    ```
    subgraphSamplerConfigs:
        experimetalFlags:
            permutation_strategy: deterministic
    ```

  When dealing with very large order of data that requires large sets of machines, memory intensive joins, we can run
  into situations where there may be some failures due to OOM, or unexpected instance termination. Usually these
  failures are self resolved i.e. execution plan is automatically optimized, data is recomputed and job eventually
  finishes. Albeit, this requires a recomputation of data (i.e. RDD), if the associated data is lost. This recomputation
  of data is to guarantee Spark being fault-tolerant. As a result, the SGS job will be finished successful and the
  status of subgraph_sampler component shows success on the VAI UI. But, this may yield corrupted data in SGS since
  there are non-deterministic transformations i.e. some data that was recomputed uses a different seed yielding an
  output that is not compatible with other data that was not lost as part of interim failures. Setting permutation
  strategy to `deterministic`, SGS will try to safegaurd against data corruption.

- **numTrainingSamples**

  If users wish to downsample number of nodes used for training (not in inferencer), they can adjut this number as below

  ```
  subgraphSamplerConfigs:
      numTrainingSamples : 10000000
  ```

  NOTE that for NodeAnchorBasedLinkPrediction samples the value set for numTrainingSamples will be further reduced in
  the next component i.e. Split Generator by a factor of `train_split`. Because if in a train split for an anchor node
  if at least a positve edge does not exit, that anchor node is dropped from train split.

- **Valid Samples** (for NodeAnchorBasedLinkPrediction Task)

  1. In random_negative_rooted_neighborhood_samples: every node which appears in the graph should get an embedding.
     Hence, they must appear in the rooted neighborhoods for inferencer to use, regardless of what their
     (in-)neighborhood looks like.

  1. In node_anchor_based_link_prediction_samples: every node which has any outgoing edge could be a valid training
     sample, since in practice we will want to have our trained model robustly perform well at ranking the positive edge
     above negative edges, regardless of what their (in-)neighborhood looks like.

### Implementation:

- Leverages both Spark SQL and DataFrame APIs.

  All SGS task queries use SQL queries. Unittests use DataFrame API.

- Strongly relies on caching both for optimization purposes and accuracy of sampling (pointed out in the code comments).

- Can be run on mac, gCloud VMs and Dataproc VMs. (cmds are available in Makefile)

## Scalability

______________________________________________________________________

### Performance

Time: It is not expected to have SGS task running more than 3hrs. If that's happening the cluster size is too small,
there are spills during some stages or caching steps implemented in code are not taking into effect.

Cost: See Google Cloud [pricing calculator](https://cloud.google.com/products/calculator/#id=)

### Load

As any of below factor increases we should think of strategies to scale the SGS job:

1. Graph size (number of nodes and edges)
1. Number of neighborhood samples and Number of Positive Samples (if any)
1. Node feature Dim
1. Edge feature Dim
1. Number of hops

### Spark Optimization/Scaling strategies

- The easiest way is to scale the cluster horizontally. As of now the number of data partitions (i.e. spark sql
  partitions), is proportional to cluster size, such that cpu usage is at max (>90%) to avoid waste of resources.
  Scaling out number of workers or changing the machine types should be done carefully. If we pick cluster that is too
  big for the task it will be to the job's disadvantage. (unnecessary large number of data partitions leads to slow
  reads, large network communications and long shuffle time)
- Use Local SSDs (NVMe interface) attached to each worker to store cached data. Do not cache data into working memory,
  since it intensifies spills. For final stages of SGS job (pointed out in the code), spills are inevitable but the
  spills are into SSDs and hence won't hurt performance badly).
- Each SGS task firstly creates a SubgraphDF (which includes all hydrated neighborhoods for all root nodes). This
  SubgraphDF must be cached and cache is NOT triggered unless the RootedNodeNeighborhood samples are written. Then
  downstream tasks do not repeat neighborhood sampling/hydration and in turn use cached data which saves at least one hr
  as of now for MAU data.
- Avoid using UDFs, instead maximize leveraging spark.sql
  [functions](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/functions$.html).
- There are stages that require repartitioning as optimization strategy. That is because the size of data in each
  partition grows (we see spills) and by repartitioning we reduce the size of data in each partition. You can find
  stages that need repartition from Spark UI.
- Do NOT change YARN default parameters for Dataproc Cluster.
- If cost become a bottleneck, (and despite trying above strategies we still need to scale) Autoscaling Dataproc with
  customized autoscaling policy should be a potential solution.

## Resources

______________________________________________________________________

Naming Conventions

- [Scala](https://docs.scala-lang.org/style/naming-conventions.html)

- [Spark](https://github.com/databricks/scala-style-guide)

Spark

- [SQL](https://spark.apache.org/docs/latest/sql-ref-functions.html)
- [DataFrame](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/functions$.html)

Protobuf and Spark

- [ScalaPB](https://scalapb.github.io/docs/sparksql/)

Optimization

- The version of Spark matters.
- https://medium.com/@vrba.dave
