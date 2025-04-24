package libs.task

import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.types.pb_wrappers.GraphPbWrappers.EdgePbWrapper
import org.apache.spark.sql.Dataset
import scalapb.spark.Implicits._
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.NodeAnchorBasedLinkPredictionSample
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

object TaskOutputValidator {

  /**
    * Validate that supervsion edges and neighborhood edges (or rather the nodes in those edges)
    * is present in the neighborhood nodes.
    * This method does a dataset.map() on the final output produced by SGS and returns the same dataset
    * if there is no validation failure. Raises and excpetion if there is some error
    * @spark: dataset.map() is not an action (unlike foreach) and does not lead to any 
    * duplication of computation due to this validation code.
    *
    * @param mainSampleDS
    * @param graphMetadataPbWrapper
    * @return
    */

  // TODO (svij-sc) Future candidate for refactor where TaskOutputValidator.__functions__ need not return a new dataset, they just need to throw exception.
  def validateMainSamples(
    mainSampleDS: Dataset[NodeAnchorBasedLinkPredictionSample],
    graphMetadataPbWrapper: GraphMetadataPbWrapper,
  ): Dataset[NodeAnchorBasedLinkPredictionSample] = {

    mainSampleDS.map(sample => {
      val neighborhood: Graph = sample.neighborhood.getOrElse(
        throw new RuntimeException(s"Neighborhood not present in sample ${sample}"),
      )
      validationHelper(
        neighborhoodNodes = neighborhood.nodes,
        edgesToValidate =
          sample.posEdges ++ sample.negEdges ++ sample.hardNegEdges ++ neighborhood.edges,
        graphMetadataPbWrapper = graphMetadataPbWrapper,
      )
      sample
    })
  }

  /**
    * Validate that neighborhood edges (or rather the nodes in those edges)
    * is present in the neighborhood nodes.
    * This method does a dataset.map() on the final output produced by SGS and returns the same dataset
    * if there is no validation failure. Raises and excpetion if there is some error
    * @spark: dataset.map() is not an action (unlike foreach) and does not lead to any 
    * duplication of computation due to this validation code.
    *
    * @param mainSampleDS
    * @param graphMetadataPbWrapper
    * @return
    */
  def validateRootedNodeNeighborhoodSamples(
    rootedNodeNeighborhoodSamples: Dataset[RootedNodeNeighborhood],
    graphMetadataPbWrapper: GraphMetadataPbWrapper,
  ): Dataset[RootedNodeNeighborhood] = {

    rootedNodeNeighborhoodSamples.map(sample => {
      val neighborhood: Graph = sample.neighborhood.getOrElse(
        throw new RuntimeException(s"Neighborhood not present in sample ${sample}"),
      )
      validationHelper(
        neighborhoodNodes = neighborhood.nodes,
        edgesToValidate = neighborhood.edges,
        graphMetadataPbWrapper = graphMetadataPbWrapper,
      )
      sample
    })
  }

  /**
    * Helper method to validate the edges in edgesToValidate has nodes present in the neighborhoodNodes
    *
    * @param neighborhood
    * @param edgesToValidate
    */
  def validationHelper(
    neighborhoodNodes: Seq[Node],
    edgesToValidate: Seq[Edge],
    graphMetadataPbWrapper: GraphMetadataPbWrapper,
  ): Unit = {
    // build featurless nodes set as edges do not contain features
    val setOfFeauturelessNeighborhoodNodes: Set[Node] = neighborhoodNodes
      .map(node => Node(nodeId = node.nodeId, condensedNodeType = node.condensedNodeType))
      .toSet

    // forach edge validate that nodes in the edge present in the neighborhood nodes
    edgesToValidate
      .foreach(edge => {
        val featurelessNodesInEdge =
          graphMetadataPbWrapper.getFeaturelessNodePbsFromEdge(EdgePbWrapper(edge))
        featurelessNodesInEdge.foreach(nodePb => {
          if (!setOfFeauturelessNeighborhoodNodes.contains(nodePb)) {
            throw new RuntimeException(
              s"Output Validation failed: node ${nodePb} of Edge ${edge} not present in neighborhood graph. With neighborhood nodes: ${setOfFeauturelessNeighborhoodNodes}",
            )
          }
        })
      })
  }
}
