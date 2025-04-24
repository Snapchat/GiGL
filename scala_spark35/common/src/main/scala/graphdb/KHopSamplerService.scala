package common.graphdb

import common.types.SamplingOpDAG
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

trait KHopSamplerService {

  def getKHopSubgraphForRootNode(
    rootNode: Node,
    samplingOpDag: SamplingOpDAG,
  ): RootedNodeNeighborhood

  def getKHopSubgraphForRootNodes(
    rootNodes: Seq[Node],
    samplingOpDag: SamplingOpDAG,
  ): Seq[RootedNodeNeighborhood]

  def samplePositiveEdgeNeighborhoods(
    rootNode: Node,
    edgeType: EdgeType,
    numPositives: Int,
    samplingOpDag: SamplingOpDAG,
  ): Tuple2[Seq[Edge], Seq[Graph]]

  def setup(): Unit

  def teardown(): Unit

}
