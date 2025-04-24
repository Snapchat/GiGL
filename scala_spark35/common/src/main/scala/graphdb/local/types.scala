package common.graphdb.local

import common.types.GraphTypes.CondensedNodeType
import common.types.GraphTypes.NodeId
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp
import upickle.default.macroRW
import upickle.default.read
import upickle.default.write
import upickle.default.{ReadWriter => RW}

object types {

  type LocalDbQueryString = String

  object LocalDbQuery {

    // We define a serializer; used by upickle to serialize the case class into a JSON object
    // For more info, see: https://com-lihaoyi.github.io/upickle/
    implicit val rw: RW[LocalDbQuery] = macroRW

    def apply(
      query: LocalDbQueryString,
    ): LocalDbQuery = {
      read[LocalDbQuery](query)
    }
    def apply(
      samplingOp: SamplingOp,
      rootNodes: Seq[Node],
    ): LocalDbQuery = {
      val rootNodeIdentifiers: Seq[Tuple2[NodeId, CondensedNodeType]] =
        rootNodes.map(node => (node.nodeId, node.condensedNodeType.get)).toSeq
      val samplingProtoStr = samplingOp.toProtoString
      new LocalDbQuery(
        rootNodeIdentifiers = rootNodeIdentifiers,
        samplingOpProtoStr = samplingProtoStr,
      )
    }

  }
  case class LocalDbQuery(
    /**
      * Case class used to easily searialize and deserialize the query params
      *
      * @param rootNodeIdentifiers
      * @param samplingOpProtoStr
      * @return
      */
    rootNodeIdentifiers: Seq[Tuple2[NodeId, CondensedNodeType]],
    samplingOpProtoStr: String) {
    def toQueryString(): LocalDbQueryString = {
      write(this)
    }
    def getSamplingOp(): SamplingOp = {
      SamplingOp.fromAscii(samplingOpProtoStr) // Reverse of "toProtoString"
    }
    def getRootNodes(): Seq[Node] = {
      rootNodeIdentifiers.map(nodeIdentifier =>
        Node(nodeId = nodeIdentifier._1, condensedNodeType = Some(nodeIdentifier._2)),
      )
    }
  }

  case class LocalDbQueryResponse(
    neighborhoodEdgeSet: Set[Edge],
    neighborhoodNodeSet: Set[Node])

  val QUERY_RESPONSE_KEY =
    "queryResponse" // Key to access LocalDbQueryResponse in the GraphDbResult

}
