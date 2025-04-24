package splitgenerator.lib.tasks

import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.utils.TFRecordIO.dataframeToTypedDataset
import scalapb.spark.Implicits._
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood
import splitgenerator.lib.Types.SplitOutputPaths
import splitgenerator.lib.split_strategies.NodeAnchorBasedLinkPredictionSplitStrategy

class NodeAnchorBasedLinkPredictionTask(
  gbmlConfigWrapper: GbmlConfigPbWrapper,
  splitStrategy: NodeAnchorBasedLinkPredictionSplitStrategy)
    extends SplitGeneratorTask(gbmlConfigWrapper, splitStrategy) {

  val randomNegativeCoalesceFactor = 4
  private val __nodeAnchorBasedLinkPredictionDataset =
    gbmlConfigWrapper.sharedConfigPb.datasetMetadata.get.getNodeAnchorBasedLinkPredictionDataset

  override def run(): Unit = {
    runTaskForMainSamples
    runTaskForRandomNegativeSamples
  }

  def runTaskForMainSamples(): Unit = {
    val inputPathForMainSamples =
      gbmlConfigWrapper.flattenedGraphMetadataPb.getNodeAnchorBasedLinkPredictionOutput.tfrecordUriPrefix

    // load, coalesce and cache dataframe from GCS
    val cachedInputDF =
      loadCoalesceCacheDataframe(
        inputPath = inputPathForMainSamples,
        coalesceFactor = mainSamplesCoalesceFactor,
      )
    // Convert raw Dataframe to Dataset[UNodeAnchorBasedLinkPredictionSample]
    val mainSamplesDS =
      dataframeToTypedDataset[NodeAnchorBasedLinkPredictionSample](
        df = cachedInputDF,
      )

    // for each train/test/val, get the split data from the input dataset and write that data back to GCS
    val outputPaths = SplitOutputPaths(
      trainPath = __nodeAnchorBasedLinkPredictionDataset.trainMainDataUri,
      valPath = __nodeAnchorBasedLinkPredictionDataset.valMainDataUri,
      testPath = __nodeAnchorBasedLinkPredictionDataset.testMainDataUri,
    )
    splitSamplesAndWriteToOutputPath(
      outputPaths = outputPaths,
      inputDS = mainSamplesDS,
      splitFn = splitStrategy.splitTrainingSample,
    )

    // Unpersist the cached df
    cachedInputDF.unpersist()
  }

  def runTaskForRandomNegativeSamples(): Unit = {
    // TODO (Tong): The following ~10 line of codes assumes homogeneous graphs, need to update to support heterogeneous graphs in HGS.
    //              They may work for stage 1 of HGS, but needs validation.
    val defaultTargetNodeType =
      gbmlConfigWrapper.taskMetadataPb.getNodeAnchorBasedLinkPredictionTaskMetadata.supervisionEdgeTypes.head.dstNodeType
    val inputPathForRandomNegativeSamples =
      gbmlConfigWrapper.flattenedGraphMetadataPb.getNodeAnchorBasedLinkPredictionOutput.nodeTypeToRandomNegativeTfrecordUriPrefix
        .getOrElse(
          defaultTargetNodeType,
          throw new Exception(
            "If you are seeing this, it means the node_type_to_random_negative_tfrecord_uri_prefix is missing the dstNodeType of the first supervision edge type, please check the frozen config",
          ),
        )

    // load, coalesce and cache dataframe from GCS
    val cachedInputDF =
      loadCoalesceCacheDataframe(
        inputPath = inputPathForRandomNegativeSamples,
        coalesceFactor = randomNegativeCoalesceFactor,
      )
    // Convert raw Dataframe to Dataset[RootedNodeNeighborhood]
    val randomNegativeDS =
      dataframeToTypedDataset[RootedNodeNeighborhood](
        df = cachedInputDF,
      )

    // for each train/test/val, get the split data from the input dataset and write that data back to GCS
    val outputPaths = SplitOutputPaths(
      trainPath =
        __nodeAnchorBasedLinkPredictionDataset.trainNodeTypeToRandomNegativeDataUri.getOrElse(
          defaultTargetNodeType,
          throw new Exception(
            "If you are seeing this, it means the train_node_type_to_random_negative_data_uri is missing the dstNodeType of the first supervision edge type, please check the frozen config",
          ),
        ),
      valPath = __nodeAnchorBasedLinkPredictionDataset.valNodeTypeToRandomNegativeDataUri.getOrElse(
        defaultTargetNodeType,
        throw new Exception(
          "If you are seeing this, it means the val_node_type_to_random_negative_data_uri is missing the dstNodeType of the first supervision edge type, please check the frozen config",
        ),
      ),
      testPath =
        __nodeAnchorBasedLinkPredictionDataset.testNodeTypeToRandomNegativeDataUri.getOrElse(
          defaultTargetNodeType,
          throw new Exception(
            "If you are seeing this, it means the test_node_type_to_random_negative_data_uri is missing the dstNodeType of the first supervision edge type, please check the frozen config",
          ),
        ),
    )
    splitSamplesAndWriteToOutputPath(
      outputPaths = outputPaths,
      inputDS = randomNegativeDS,
      splitFn = splitStrategy.splitRootedNodeNeighborhoodTrainingSample,
    )

    // Unpersist the cached df
    cachedInputDF.unpersist()
  }
}
