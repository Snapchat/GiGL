package libs.task.pureSpark

import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.utils.TFRecordIO
import common.utils.TFRecordIO.RecordTypes
import common.utils.TFRecordIO.readDataframeFromTfrecord
import common.utils.TFRecordIO.writeDataframeToTfrecord
import libs.task.SamplingStrategy.PermutationStrategy
import libs.task.TaskOutputValidator
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.IntegerType
import scalapb.spark.Implicits._
import snapchat.research.gbml.preprocessed_metadata.PreprocessedMetadata
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import snapchat.research.gbml.training_samples_schema.SupervisedNodeClassificationSample

class SupervisedNodeClassificationTask(
  gbmlConfigWrapper: GbmlConfigPbWrapper)
    extends SGSPureSparkV1Task(gbmlConfigWrapper) {

  val graphMetadataPbWrapper = gbmlConfigWrapper.graphMetadataPbWrapper
  // constants used for repartitioning DFs, as their size grows to avoid spills and OOM errors
  // @spark: as numNeighborsToSample and feature dim increases, we need to adjust this (if we're not changing cluster size)
  // However, if we have a cluster scaled to NodeAnchorBasedLinkPred. task, we do not need repartition for this task, as it's a way less expensive task.
  private val RepartitionFactorForRefSubgraphDf = 1

  def run(): Unit = {
    // We get default condensed edge type for now; in the future this might be
    // done in a more iterative fashion to support hetero use cases.
    val defaultCondensedEdgeType: Int =
      gbmlConfigWrapper.preprocessedMetadataWrapper.preprocessedMetadataPb.condensedEdgeTypeToPreprocessedMetadata.keysIterator
        .next()
    val defaultCondensedNodeType: Int =
      gbmlConfigWrapper.preprocessedMetadataWrapper.preprocessedMetadataPb.condensedNodeTypeToPreprocessedMetadata.keysIterator
        .next()
    val experimentalFlags = gbmlConfigWrapper.subgraphSamplerConfigPb.experimentalFlags
    val permutationStrategy =
      experimentalFlags.getOrElse("permutation_strategy", PermutationStrategy.NonDeterministic)
    val numNeighborsToSample = gbmlConfigWrapper.subgraphSamplerConfigPb.numNeighborsToSample
    val includeIsolatedNodesInTrainingSamples =
      false // must be added to subgraphSamplerConfig in gbml config [As of now, SGS v2 with Spark, isolated nodes are NOT included in training samples ]
    // NOTE: we don't have downsampleNumberOfNodes applied to isolated nodes, so if in future users want to
    // include isolated nodes in training samples, all isolated nodes will be included in training samples.
    // If users want to downsample isolated nodes, we need to add a dedicated flag for num of isloated nodes in SGS configs
    val numTrainingSamples = gbmlConfigWrapper.subgraphSamplerConfigPb.numMaxTrainingSamplesToOutput

    val hydratedNodeVIEW: String =
      loadNodeDataframeIntoSparkSql(condensedNodeType = defaultCondensedNodeType)
    val nodeLabelVIEW: String =
      loadNodeLabelDataframeIntoSparkSql(condensedNodeType = defaultCondensedNodeType)
    val hydratedEdgeVIEW: String =
      loadEdgeDataframeIntoSparkSql(condensedEdgeType = defaultCondensedEdgeType)
    val unhydratedEdgeVIEW: String =
      loadUnhydratedEdgeDataframeIntoSparkSql(hydratedEdgeVIEW = hydratedEdgeVIEW)

    val subgraphVIEW = createSubgraph(
      numNeighborsToSample = numNeighborsToSample,
      hydratedNodeVIEW = hydratedNodeVIEW,
      hydratedEdgeVIEW = hydratedEdgeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      permutationStrategy = permutationStrategy,
    ) // does not include isolated nodes and we must cache this DF
    // @spark: caching the DF associated with subgraphVIEW is CRITICAL, substantially reduces job time. Cache is
    // triggered once the action (i.e. *write* RootedNodeNeighborhoodSample to GCS) is finished.
    val cachedSubgraphVIEW = applyCachingToDf(
      dfVIEW = subgraphVIEW,
      forceTriggerCaching = false,
      withRepartition = false,
    ) // set to true if we have both high dim edge/node feats, and we don't wanna scale out cluster nodes. Then use RepartitionFactorForRefSubgraphDf, and colNameForRepartition='_root_node'

    val sncTaskOutputUri =
      gbmlConfigWrapper.flattenedGraphMetadataPb.getSupervisedNodeClassificationOutput

    val rnnVIEW = createRootedNodeNeighborhoodSubgraph(
      hydratedNodeVIEW = hydratedNodeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
      subgraphVIEW = cachedSubgraphVIEW,// NOTE to pass **cached** subgraph
    )                                   // has isolated nodes

    val rnnWithSchemaVIEW = castToRootedNodeNeighborhoodProtoSchema(dfVIEW = rnnVIEW)
    val rnnWithSchemaDF   = spark.table(rnnWithSchemaVIEW)
    // @spark: NOTE 1: for all tasks, we must first write RootedNodeNeighborhood samples.
    // since write is the `action` that triggers caching on subgraphDF. The other sample types, such as
    // SupervisedNodeClassificationSample, use cachedSubgraphDF to avoid repeating neighbor sampling/hydration.

    val rnnWithSchemaDS: Dataset[RootedNodeNeighborhood] =
      rnnWithSchemaDF.as[RootedNodeNeighborhood]
    // validate output before writing to storage
    println("start validating rooted node neighborhood samples")
    val validatedRnnWithSchemaDS = TaskOutputValidator.validateRootedNodeNeighborhoodSamples(
      rootedNodeNeighborhoodSamples = rnnWithSchemaDS,
      graphMetadataPbWrapper = graphMetadataPbWrapper,
    )
    TFRecordIO.writeDatasetToTfrecord[RootedNodeNeighborhood](
      inputDS = validatedRnnWithSchemaDS,
      gcsUri = sncTaskOutputUri.unlabeledTfrecordUriPrefix,
    )
    println("Finished writing RootedNodeNeighborhood samples")
    if (
      gbmlConfigWrapper.sharedConfigPb.shouldSkipTraining && gbmlConfigWrapper.sharedConfigPb.shouldSkipModelEvaluation
    ) {
      println(
        "shouldSkipTraining and shouldSkipModelEvaluation is set to true, thus skipping generating SupervisedNodeClassification samples",
      )
    } else {
      val sncVIEW = createSupervisedNodeClassificationSubgraph(
        includeIsolatedNodesInTrainingSamples = includeIsolatedNodesInTrainingSamples,
        nodeLabelVIEW = nodeLabelVIEW,
        subgraphVIEW = cachedSubgraphVIEW, // NOTE to pass **cached** subgraph
        hydratedNodeVIEW = hydratedNodeVIEW,
        unhydratedEdgeVIEW = unhydratedEdgeVIEW,
        numTrainingSamples = numTrainingSamples,
      )
      val sncWithSchemaVIEW = castToTrainingSampleProtoSchema(dfVIEW = sncVIEW)
      val sncWithSchemaDF   = spark.table(sncWithSchemaVIEW)
      writeDataframeToTfrecord[SupervisedNodeClassificationSample](
        df = sncWithSchemaDF,
        gcsUri = sncTaskOutputUri.labeledTfrecordUriPrefix,
      )
    }

  }

  def loadNodeLabelDataframeIntoSparkSql(condensedNodeType: Int): String = {

    /**
      * loads task specific dataframes, i.e. Node DF including labels used for Supervised node classification task.
      */
    val nodeMetadataOutputPb: PreprocessedMetadata.NodeMetadataOutput =
      gbmlConfigWrapper.preprocessedMetadataWrapper
        .getPreprocessedMetadataForCondensedNodeType(condensedNodeType = condensedNodeType)

    val nodeIdKey: String = nodeMetadataOutputPb.nodeIdKey
    // used for accessing label values from colName = labelKey
    val labelKey: String = nodeMetadataOutputPb.labelKeys(0) // labelKeys is List[String]
    val labelType: String = labelKey // used for creating a String col including labelType
    val nodeDfUri: String = nodeMetadataOutputPb.tfrecordUriPrefix

    val rawNodeLabelDF: DataFrame = readDataframeFromTfrecord(
      uri = nodeDfUri,
      recordType = RecordTypes.ExampleRecordType,
    ).select(nodeIdKey, labelKey) // we don't want node features
    val preprocessedNodeLabelsDF = rawNodeLabelDF.withColumn(
      nodeIdKey,
      rawNodeLabelDF(nodeIdKey) cast (IntegerType),
    )
    val preprocessedNodeLabelsVIEW = "preprocessedNodeLabelsDF" + uniqueTempViewSuffix
    preprocessedNodeLabelsDF.createOrReplaceTempView(preprocessedNodeLabelsVIEW)

    // add `lableType` as a literal STRING column
    val nodeLabelDF = spark.sql(f"""
      SELECT
        ${nodeIdKey} AS _node_id,
        ${labelKey} AS _label_key,
        '${labelType}' AS _label_type
      FROM
        ${preprocessedNodeLabelsVIEW}
    """)
    val nodeLabelVIEW = "nodeLabelDF" + uniqueTempViewSuffix
    nodeLabelDF.createOrReplaceTempView(nodeLabelVIEW)
    nodeLabelVIEW
  }

  def createSupervisedNodeClassificationSubgraph(
    includeIsolatedNodesInTrainingSamples: Boolean,
    nodeLabelVIEW: String,
    subgraphVIEW: String,
    hydratedNodeVIEW: String,
    unhydratedEdgeVIEW: String,
    numTrainingSamples: Int,
  ): String = {

    /**
      * @param nodeLabelVIEW with columns:
      * _node_id: Int
      * _label_key: Int, label value for each node id
      * _label_type: String
      * @param subgraphVIEW with columns:
      * _root_node: Int
      * _neighbor_edges: Array(Struct)
      * _neighbor_nodes: Array(Struct)
      * _node_features: Array
      * _condensed_node_type: Int
      * @param hydratedNodeVIEW with columns:
      * _node_id: Int
      * _node_features: Array
      * _condensed_node_type: Int
      * @param unhydratedEdgeVIEW with columns:
      * _src_node: Int
      * _dst_node: Int
      */
    val retainedNodeLabelVIEW = if (numTrainingSamples > 0) {
      println(f"Downsampling number of training samples to ${numTrainingSamples}")
      val downsampledNodeLabelVIEW = downsampleNumberOfNodes(
        numSamples = numTrainingSamples,
        dfVIEW = nodeLabelVIEW,
      )
      downsampledNodeLabelVIEW
    } else {
      nodeLabelVIEW
    }
    val subgraphWithNodeLabelDF = spark.sql(f"""
        SELECT
          _root_node,
          _neighbor_edges,
          _neighbor_nodes,
          _node_features,
          _condensed_node_type,
          _label_type,
          _label_key AS _label
        FROM
          ${subgraphVIEW}
        INNER JOIN
          ${retainedNodeLabelVIEW}
        ON
          ${subgraphVIEW}._root_node=${retainedNodeLabelVIEW}._node_id
    """)
    val sncVIEW = if (!includeIsolatedNodesInTrainingSamples) {
      val subgraphWithNodeLabelVIEW = "subgraphWithNodeLabelDF" + uniqueTempViewSuffix
      subgraphWithNodeLabelDF.createOrReplaceTempView(subgraphWithNodeLabelVIEW)
      subgraphWithNodeLabelVIEW
    } else {
      val subgraphWithNodeLabelVIEW = "subgraphWithNodeLabelDF" + uniqueTempViewSuffix
      subgraphWithNodeLabelDF.createOrReplaceTempView(subgraphWithNodeLabelVIEW)
      val subgraphWithNodeLabelAndIsolatedNodeVIEW = appendIsolatedNodesToTrainingSamples(
        trainingSubgraphVIEW = subgraphWithNodeLabelVIEW,
        hydratedNodeVIEW = hydratedNodeVIEW,
        unhydratedEdgeVIEW = unhydratedEdgeVIEW,
        nodeLabelVIEW = retainedNodeLabelVIEW,
      )
      subgraphWithNodeLabelAndIsolatedNodeVIEW
    }
    sncVIEW
  }

  def appendIsolatedNodesToTrainingSamples(
    trainingSubgraphVIEW: String,
    hydratedNodeVIEW: String,
    unhydratedEdgeVIEW: String,
    nodeLabelVIEW: String,
  ): String = {

    /**
      * For cases that users want to include neighborless nodes in task specific samples.
      * @param trainingSubgraphVIEW trainingSubgraph without neighborless nodes, with columns:
      * _root_node: Int
      * _neighbor_edges: Array(Struct)
      * _neighbor_nodes: Array(Struct)
      * _node_features: Array
      * _condensed_node_type: Int
      * _label: Int, label value for each node id
      * _label_type: String
      * @param nodeLabelVIEW with columns:
      * _node_id: Int
      * _label_key: Int, label value for each node id
      * _label_type: String
      */
    // TODO: for Supervised Node classification task, verify if we want to include
    // only isolated nodes or all neighborless nodes for training samples
    // corrently, neighborless nodes (isolated + sr only nodes) will be included
    // if we need ONLY isolated nodes, then use `createIsolatedNodesSubgraph()`
    val neighborlessNodesSubgraphVIEW = createNeighborlessNodesSubgraph(
      hydratedNodeVIEW = hydratedNodeVIEW,
      unhydratedEdgeVIEW = unhydratedEdgeVIEW,
    )

    val neighborlessNodesWithLabelsSubgraphDF = spark.sql(f"""
      SELECT
        _root_node,
        _node_features,
        _condensed_node_type,
        _label_key AS _label,
        _label_type
      FROM
        ${neighborlessNodesSubgraphVIEW}
      INNER JOIN
        ${nodeLabelVIEW}
      ON
        ${neighborlessNodesSubgraphVIEW}._root_node=${nodeLabelVIEW}._node_id
        """)
    val neighborlessNodesWithLabelsSubgraphVIEW =
      "neighborlessNodesWithLabelsSubgraphDF" + uniqueTempViewSuffix
    neighborlessNodesWithLabelsSubgraphDF.createOrReplaceTempView(
      neighborlessNodesWithLabelsSubgraphVIEW,
    )
    val trainingSubgraphsWithNeighborlessNodesDF = spark.sql(f"""
        SELECT
          _root_node,
          _neighbor_edges,
          _neighbor_nodes,
          _node_features,
          _condensed_node_type,
          _label,
          _label_type
        FROM
          ${trainingSubgraphVIEW} UNION
        SELECT
          _root_node,
          NULL AS _neighbor_edges,
          ARRAY(STRUCT(_root_node AS _node_id,
              _condensed_node_type,
              _node_features AS _feature_values)) AS _neighbor_nodes,
          _node_features,
          _condensed_node_type,
          _label,
          _label_type
        FROM
          ${neighborlessNodesWithLabelsSubgraphVIEW}
      """)
    val trainingSubgraphsWithNeighborlessNodesVIEW =
      "trainingSubgraphsWithNeighborlessNodesDF" + uniqueTempViewSuffix
    trainingSubgraphsWithNeighborlessNodesDF.createOrReplaceTempView(
      trainingSubgraphsWithNeighborlessNodesVIEW,
    )
    trainingSubgraphsWithNeighborlessNodesVIEW
  }

  def castToTrainingSampleProtoSchema(dfVIEW: String): String = {
    // Forms dfVIEW schema into SupervisedNodeClassification message in training_samples_schema.proto
    val sncDF = spark.sql(f"""
        SELECT
          STRUCT( _root_node AS node_id,
            _condensed_node_type AS condensed_node_type,
            _node_features AS feature_values ) AS root_node,
          STRUCT( _neighbor_nodes AS nodes,
            _neighbor_edges AS edges ) AS neighborhood,
          ARRAY(STRUCT( _label_type AS label_type,
              CAST(_label AS INTEGER) AS label) ) AS root_node_labels
        FROM
          ${dfVIEW}
          """)
    val sncVIEW = "sncDF" + uniqueTempViewSuffix
    sncDF.createOrReplaceTempView(sncVIEW)
    sncVIEW
  }

}
