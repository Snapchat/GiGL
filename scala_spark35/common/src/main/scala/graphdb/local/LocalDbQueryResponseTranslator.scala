package common.graphdb.local

import common.graphdb.DBResult
import common.graphdb.QueryResponseTranslator
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp

import scala.collection.mutable.HashSet

class LocalDbQueryResponseTranslator() extends QueryResponseTranslator with Serializable {

  /**
    * This translator is really simple on purpose and all it does is serialize the samplingOp and rootNodes
    * into a string. This is because we want to have LocalDbClient have all the core logic on what to do for
    * each sampling op.
    *
    * @param samplingOp
    * @param rootNodes
    * @return
    */

  def translate(
    samplingOp: SamplingOp,
    rootNodes: Seq[Node],
  ): String = {
    types
      .LocalDbQuery(
        samplingOp = samplingOp,
        rootNodes = rootNodes,
      )
      .toQueryString
  }

  def translate(
    samplingOp: SamplingOp,
    rootNode: Node,
  ): String = {
    translate(samplingOp, Seq(rootNode))
  }

  def parseEdgesAndNodesFromResult(
    result: DBResult,
    samplingOp: SamplingOp,
  ): Tuple2[HashSet[Edge], HashSet[Node]] = {
    val neighborhoodEdgeSet: HashSet[Edge] = HashSet.empty[Edge]
    val neighborhoodNodeSet: HashSet[Node] = HashSet.empty[Node]
    result // The whole of LocalDbQueryResponse is stored in the column with key QUERY_RESPONSE_KEY
      .colValues(types.QUERY_RESPONSE_KEY)
      .asInstanceOf[List[types.LocalDbQueryResponse]]
      .foreach(queryResponse => {
        neighborhoodEdgeSet ++= queryResponse.neighborhoodEdgeSet
        neighborhoodNodeSet ++= queryResponse.neighborhoodNodeSet
      })
    Tuple2(neighborhoodEdgeSet, neighborhoodNodeSet)
  }

}
