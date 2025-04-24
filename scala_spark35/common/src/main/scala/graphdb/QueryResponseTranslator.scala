package common.graphdb

import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp

import scala.collection.mutable.HashSet

trait QueryResponseTranslator {

  /**
    * Trait couples with `DBClient` to handle translation of samplingOp --> query string
    * and parsing of query results into edges and nodes. Sits as the "translator" layer
    * between business logic and relevant databases.
    *
    * @param samplingOp
    * @param rootNodes
    * @return
    */
  def translate(
    samplingOp: SamplingOp,
    rootNodes: Seq[
      Node,
    ],
  ): String

  def translate(
    samplingOp: SamplingOp,
    rootNode: Node,
  ): String

  def parseEdgesAndNodesFromResult(
    result: DBResult,
    samplingOp: SamplingOp,
  ): Tuple2[HashSet[Edge], HashSet[Node]]
}
