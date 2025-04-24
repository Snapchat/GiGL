package splitgenerator.lib.split_strategies

import common.types.pb_wrappers.GraphPbWrappers.EdgePbWrapper
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.SupervisedNodeClassificationSample
import splitgenerator.lib.Types.DatasetSplits.DatasetSplit
import splitgenerator.lib.assigners.AbstractAssigners.Assigner

class InductiveSupervisedNodeClassificationSplitStrategy(
  splitStrategyArgs: Map[String, String],
  nodeToDatasetSplitAssigner: Assigner[Node, DatasetSplit])
    extends SupervisedNodeClassificationSplitStrategy(
      splitStrategyArgs,
      nodeToDatasetSplitAssigner,
    ) {

  /**
    * For inductive node classification splitting, train/val/test graphs are disjoint.
    * First we select the assignments of each node to train/val/test split.  Next, we
    * ensure each labeled sample only sees other nodes which are also in its own split,
    * and only those edges between nodes that are in its own split.
    * Ref: http://snap.stanford.edu/class/cs224w-2020/slides/09-theory.pdf slides 8-9
    *
    * @param sample
    * @param datasetSplit
    * @return
    */
  override def splitTrainingSample(
    sample: SupervisedNodeClassificationSample,
    datasetSplit: DatasetSplit,
  ): Seq[SupervisedNodeClassificationSample] = {

    // root node will always exist. This will fail the pipeline if rootNode is None.
    // Reason for rootNode to be an Option in proto: https://scalapb.github.io/docs/faq/#why-message-fields-are-wrapped-in-an-option-in-proto3
    val rootNodePb: Node =
      sample.rootNode.getOrElse(throw new RuntimeException("Root node does not exist for sample."))

    // split root node and check if it belongs to this particular split
    val splitSample: Seq[SupervisedNodeClassificationSample] =
      if (nodeToDatasetSplitAssigner.assign(rootNodePb) == datasetSplit) {
        // remove all nodes and edges from the sample not belonging to this split
        Seq(
          pruneOutsideNodesAndEdges(
            sample = sample,
            datasetSplit = datasetSplit,
          ),
        )
      } else {
        Seq.empty[SupervisedNodeClassificationSample]
      }

    subsample(
      splitSample = splitSample,
      datasetSplit = datasetSplit,
      subsamplingRatio = mainSamplesSubsamplingRatio,
    )
  }

  private def pruneOutsideNodesAndEdges(
    sample: SupervisedNodeClassificationSample,
    datasetSplit: DatasetSplit,
  ): SupervisedNodeClassificationSample = {
    // filter nodes in neighborhood that belong to same split as root node
    val filteredNodes: Seq[Node] = sample.neighborhood
      .getOrElse(throw new RuntimeException("Neighborhood does not exist in the sample"))
      .nodes
      .filter(nodeToDatasetSplitAssigner.assign(_) == datasetSplit)

    // filter edges where both src and dst belong to the same split as root node.
    val filteredEdges: Seq[Edge] = sample.neighborhood
      .getOrElse(throw new RuntimeException("Neighborhood does not exist in the sample"))
      .edges
      .filter(edge =>
        // get nodePbs from edgePb. Both nodePbs should be assigned to same split
        graphMetadataPbWrapper
          .getFeaturelessNodePbsFromEdge(edgeWrapper = EdgePbWrapper(edge))
          .forall(nodeToDatasetSplitAssigner.assign(_) == datasetSplit),
      )

    // create the output sample
    SupervisedNodeClassificationSample(
      rootNode = sample.rootNode,
      rootNodeLabels = sample.rootNodeLabels,
      neighborhood = Some(Graph(nodes = filteredNodes, edges = filteredEdges)),
    )
  }
}
