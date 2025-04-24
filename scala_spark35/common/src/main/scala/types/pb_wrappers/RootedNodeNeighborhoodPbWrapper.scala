package common.types.pb_wrappers

import common.types.pb_wrappers.GraphPbWrappers.mergeGraphs
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

/**
 * Companion object for the RootedNodeNeighborhoodPbWrapper class that host static methods.
 */
object RootedNodeNeighborhoodPbWrapper {

  /**
   * Merges a sequence of RootedNodeNeighborhood instances into a single RootedNodeNeighborhood.
   *
   * @param rnns The sequence of RootedNodeNeighborhood instances to be merged.
   * @return A RootedNodeNeighborhood instance that is the result of merging the given sequence of RootedNodeNeighborhood instances.
   */
  def mergeRootedNodeNeighborhoods(rnns: Seq[RootedNodeNeighborhood]): RootedNodeNeighborhood = {
    assert(rnns.nonEmpty, "Sequence of RootedNodeNeighborhood is empty")
    assert(
      checkAllRootNodesAreSame(rnns),
      f"All root nodes should be the same in able to merge ${rnns.map(_.rootNode)}",
    )

    val mergedGraph: Graph = mergeGraphs(rnns.flatMap(_.neighborhood))

    RootedNodeNeighborhood(rootNode = rnns.head.rootNode, neighborhood = Some(mergedGraph))
  }

  /**
   * Checks if all RootedNodeNeighborhood instances in the given sequence have root nodes of same node_id and condensed_node_type.
   *
   * @param rnns The sequence of RootedNodeNeighborhood instances to be checked.
   * @return True if all RootedNodeNeighborhood instances in the given sequence have the same root node. False otherwise.
   */
  def checkAllRootNodesAreSame(rnns: Seq[RootedNodeNeighborhood]): Boolean = {
    val firstRootNode = rnns.head.rootNode
    rnns.forall(rnn =>
      rnn.rootNode.get.nodeId == firstRootNode.get.nodeId && rnn.rootNode.get.condensedNodeType == firstRootNode.get.condensedNodeType,
    )
  }
}

class RootedNodeNeighborhoodPbWrapper(rootedNodeNeighborhood: RootedNodeNeighborhood) {
  val getPb: RootedNodeNeighborhood = rootedNodeNeighborhood

  override def equals(obj: Any): Boolean = {
    if (!obj.isInstanceOf[RootedNodeNeighborhoodPbWrapper]) {
      false
    } else {
      val other: RootedNodeNeighborhoodPbWrapper = obj.asInstanceOf[RootedNodeNeighborhoodPbWrapper]
      val otherRootedNodeNeighborhood: RootedNodeNeighborhood = other.getPb
      rootedNodeNeighborhood.rootNode.equals(otherRootedNodeNeighborhood.rootNode)
      rootedNodeNeighborhood.neighborhood.get.nodes.toSet
        .equals(otherRootedNodeNeighborhood.neighborhood.get.nodes.toSet) &&
      rootedNodeNeighborhood.neighborhood.get.edges.toSet
        .equals(otherRootedNodeNeighborhood.neighborhood.get.edges.toSet)
    }
  }

  override def toString(): String = rootedNodeNeighborhood.toString()
}
