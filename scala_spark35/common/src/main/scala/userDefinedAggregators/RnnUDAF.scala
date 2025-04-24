package common.userDefinedAggregators

import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.expressions.Aggregator
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet

object RnnUDAF {

  /**
    * The following type aliases and case classes are used to define the schema of intermediary
    * data structures used in the UDAF. See class RnnUDAF for more details.
    */
  type CondensedEdgeType = Int
  type CondensedNodeType = Int
  type NodeId            = Int
  type NodeType          = Int

  // Wrapper for storing node metadata that does not play a role in aggregation logic
  // Note: You rootNodeType can and should play a role in aggregation logic for hetrogeneous
  // sampling. But you do that by sql GROUP BY rootNodeId, rootNodeType, as such it is outside
  // the scope of this UDAF but at a higher level operation handled by the sql engine.
  case class TmpNodeMetadata(
    featureValues: Seq[Float],
    condensedNodeType: Option[CondensedNodeType])

  // Wrapper for storing edge metadata that does not play a role in aggregation logic
  case class TmpEdgeMetadata(
    featureValues: Seq[Float])

  // The intermediate buffer used to aggregate the 2-hop subgraph information
  case class BufferRNN(
    var rootNodeId: Option[Int],
    var rootNodeType: Option[NodeType],
    var rootNodeFeatures: Seq[Float],
    var edges: HashMap[
      NodeId,
      HashMap[Tuple2[NodeId, CondensedEdgeType], TmpEdgeMetadata],
    ], // (dstNodeId) -> ((srcNodeId, edgeType) -> (edgeFeatures))
    var nodes: HashMap[Int, TmpNodeMetadata])

  // The input into the UDAF expects the following schema
  // rnnUDAF(...): InTwoHopData
  case class InTwoHopData(
    var _root_node_id: NodeId,
    var _root_node_type: NodeType,
    var _root_node_features: Seq[Float],
    var _1_hop_node_id: NodeId,
    var _1_hop_node_type: NodeType,
    var _1_hop_node_features: Seq[Float],
    var _1_hop_edge_features: Seq[Float],
    var _1_hop_edge_type: CondensedEdgeType,
    var _2_hop_node_id: NodeId,
    var _2_hop_node_type: NodeType,
    var _2_hop_node_features: Seq[Float],
    var _2_hop_edge_features: Seq[Float],
    var _2_hop_edge_type: CondensedEdgeType)

}

class RnnUDAF(
  sampleN: Option[Int],
  postProcessFn: Option[RootedNodeNeighborhood => RootedNodeNeighborhood] = None)
    extends Aggregator[RnnUDAF.InTwoHopData, RnnUDAF.BufferRNN, Array[Byte]] {

  /**
    * Introduces a custom user defined aggregation function that 
    * allows for more efficient "GROUP BY" on "root_node_id" when formulating a 2 hop subgraph,
    * as compared to using default Spark aggregate functions like array_append, array_union, array_agg, et al.
    * These functions are quite expensive and not suitable for aggregating all types of columns.
    * 
    * The UDAF is used to aggregate the 2-hop subgraph information into a single RootedNodeNeighborhood
    * protobuf message (byte array).
    * 
    * sampleN: Option[Int] - The number of edges to sample from the 1-hop and 2-hop neighbors of the root node.
    * 
    * Example usage:
    * spark.udf.register("rnnUDAF", F.udaf(new RnnUDAF(sampleN = Some(VAL))))
    * ...
    * ...
    * SELECT
    *   rnnUDAF(
    *       _root_node_id,
    *       _root_node_type,
    *       _root_node_features,
    *       _1_hop_node_id,
    *       _1_hop_node_type,
    *       _1_hop_node_features,
    *       _1_hop_edge_features,
    *       _1_hop_edge_type,
    *       _2_hop_node_id,
    *       _2_hop_node_type,
    *       _2_hop_node_features,
    *       _2_hop_edge_features,
    *       _2_hop_edge_type
    *   ) as result 
    * FROM
    *  ... 
    * GROUP BY
    *  _root_node_id, _root_node_type
    */

  // Buffer initialization
  def zero: RnnUDAF.BufferRNN = RnnUDAF.BufferRNN(
    rootNodeId = None,
    rootNodeType = None,
    rootNodeFeatures = Seq(),
    edges = HashMap(),
    nodes = HashMap(),
  )

  def _assertDataForValidRootNode(
    data: RnnUDAF.InTwoHopData,
    buffer: RnnUDAF.BufferRNN,
  ): Unit = {
    if (buffer.rootNodeId.isDefined && buffer.rootNodeId.get != data._root_node_id) {
      throw new Exception(
        f"Root node id is not consistent, expected ${buffer.rootNodeId.get} " +
          f"but got ${data._root_node_id}",
      )
    }
  }

  def _assertDataForValidRootNode(
    b1: RnnUDAF.BufferRNN,
    b2: RnnUDAF.BufferRNN,
  ): Unit = {
    if (
      b1.rootNodeId.isDefined && b2.rootNodeId.isDefined && b1.rootNodeId.get != b2.rootNodeId.get
    ) {
      throw new Exception(
        f"Root node id is not consistent, expected ${b1.rootNodeId.get} " +
          f"but got ${b2.rootNodeId.get}",
      )
    }
  }

  def reduce(
    buffer: RnnUDAF.BufferRNN,
    data: RnnUDAF.InTwoHopData,
  ): RnnUDAF.BufferRNN = {

    /**
      * We update the buffer with the new data point.
      */
    _assertDataForValidRootNode(data, buffer)

    // fill in root node information
    if (buffer.rootNodeId.isEmpty) {
      buffer.rootNodeId = Some(data._root_node_id)
      buffer.rootNodeType = Some(data._root_node_type)
      buffer.rootNodeFeatures = Option(data._root_node_features).getOrElse(Seq())
      buffer.nodes(data._root_node_id) = RnnUDAF.TmpNodeMetadata(
        featureValues = Option(data._root_node_features).getOrElse(Seq()),
        condensedNodeType = Some(data._root_node_type),
      )
    }

    if (!buffer.nodes.contains(data._1_hop_node_id)) {
      buffer.nodes(data._1_hop_node_id) = RnnUDAF.TmpNodeMetadata(
        featureValues = Option(data._1_hop_node_features).getOrElse(Seq()),
        condensedNodeType = Some(data._1_hop_node_type),
      )
    }
    if (!buffer.nodes.contains(data._2_hop_node_id)) {
      buffer.nodes(data._2_hop_node_id) = RnnUDAF.TmpNodeMetadata(
        featureValues = Option(data._2_hop_node_features).getOrElse(Seq()),
        condensedNodeType = Some(data._2_hop_node_type),
      )
    }

    if (!buffer.edges.contains(data._root_node_id)) {
      buffer.edges(data._root_node_id) = HashMap()
    }
    if (!buffer.edges.contains(data._1_hop_node_id)) {
      buffer.edges(data._1_hop_node_id) = HashMap()
    }

    val rootNodeNeighbors: HashMap[Tuple2[Int, Int], RnnUDAF.TmpEdgeMetadata] =
      buffer.edges(data._root_node_id)
    val _1_hop_node_neighbors_id = (data._1_hop_node_id, data._1_hop_edge_type)
    if (!rootNodeNeighbors.contains(_1_hop_node_neighbors_id)) {
      rootNodeNeighbors(_1_hop_node_neighbors_id) = RnnUDAF.TmpEdgeMetadata(
        featureValues = Option(data._1_hop_edge_features).getOrElse(Seq()),
      )
    }

    val firstHopNeighbors: HashMap[Tuple2[Int, Int], RnnUDAF.TmpEdgeMetadata] =
      buffer.edges(data._1_hop_node_id)
    val _2_hop_node_neighbors_id = (data._2_hop_node_id, data._2_hop_edge_type)
    if (!firstHopNeighbors.contains(_2_hop_node_neighbors_id)) {
      firstHopNeighbors(_2_hop_node_neighbors_id) = RnnUDAF.TmpEdgeMetadata(
        featureValues = Option(data._2_hop_edge_features).getOrElse(Seq()),
      )
    }
    buffer
  }

  def merge(
    b1: RnnUDAF.BufferRNN,
    b2: RnnUDAF.BufferRNN,
  ): RnnUDAF.BufferRNN = {

    /**
      * We merge two buffers together.
      */
    _assertDataForValidRootNode(b1, b2)
    if (b1.rootNodeId.isEmpty) {
      b1.rootNodeId = b2.rootNodeId
      b1.rootNodeType = b2.rootNodeType
      b1.rootNodeFeatures = b2.rootNodeFeatures
    }
    b1.nodes ++= b2.nodes
    for ((dstNodeId, srcNodeDataMap) <- b2.edges) {
      if (!b1.edges.contains(dstNodeId)) {
        b1.edges(dstNodeId) = srcNodeDataMap
      } else {
        b1.edges.getOrElseUpdate(dstNodeId, HashMap()) ++= srcNodeDataMap
      }
    }
    b1
  }

  def finish(reduction: RnnUDAF.BufferRNN): Array[Byte] = {

    /**
      * We use the buffer to construct the final RootedNodeNeighborhood protobuf message.
      * The message is of type RootedNodeNeighborhood and is serialized to a byte array.
      * See outputEncoder for more details.
      */

    val oneHopEdges        = reduction.edges(reduction.rootNodeId.get)
    val sampledOneHopEdges = HashMap[Int, HashMap[Tuple2[Int, Int], RnnUDAF.TmpEdgeMetadata]]()
    val sampledTwoHopEdges = HashMap[Int, HashMap[Tuple2[Int, Int], RnnUDAF.TmpEdgeMetadata]]()
    // TODO: (svij-sc) - Fix for multiple edge types ; this will just sample across all edges, we may want
    // flexibility to sample per edge type
    sampledOneHopEdges(reduction.rootNodeId.get) =
      oneHopEdges.take(sampleN.getOrElse(oneHopEdges.size))

    for ((rootNodeId, edgeMap) <- sampledOneHopEdges) {
      val oneHopNeighborNodeIds = edgeMap.keys
      for ((oneHopNodeId, oneHopEdgeType) <- oneHopNeighborNodeIds) {
        val twoHopEdges = reduction.edges.getOrElse(oneHopNodeId, HashMap())
        if (!twoHopEdges.isEmpty) {
          // TODO: (svij-sc) - Fix for multiple edge types ; this will just sample across all edges, we may want
          // flexibility to sample per edge type
          sampledTwoHopEdges(oneHopNodeId) = twoHopEdges.take(sampleN.getOrElse(twoHopEdges.size))
        }
      }
      // val twoHopEdges = reduction.edges(srcNodeId)
      // val twoHopNodeInfo = twoHopEdges.take(sampleN.getOrElse(twoHopEdges.size))
      // sampledTwoHopEdges(srcNodeId) = twoHopNodeInfo
    }

    val nodeIds: HashSet[Int] = HashSet()
    def recordEdges(
      hopEdgeMap: HashMap[Int, HashMap[Tuple2[Int, Int], RnnUDAF.TmpEdgeMetadata]],
    ): Seq[Edge] = {
      val edges: Seq[Edge] = hopEdgeMap.toSeq
        .map(edgeMap => {
          val (dstNodeId, srcNodeDataMap) = edgeMap
          srcNodeDataMap.map(srcNodeNeighborNodeIdsData => {
            val (srcNodeNeighborNodeIds, featureValues) = srcNodeNeighborNodeIdsData
            val (srcNodeId, condensedEdgeType)          = srcNodeNeighborNodeIds
            nodeIds += dstNodeId
            nodeIds += srcNodeId
            Edge(
              srcNodeId = srcNodeId,
              dstNodeId = dstNodeId,
              condensedEdgeType = Some(condensedEdgeType),
              featureValues = featureValues.featureValues,
            )
          })
        })
        .flatten
      edges
    }

    val edges: Seq[Edge] = recordEdges(sampledOneHopEdges) ++ recordEdges(sampledTwoHopEdges)

    val nodes: Seq[Node] = reduction.nodes.toSeq
      .filter(node => nodeIds.contains(node._1))
      .map(node => {
        Node(
          nodeId = node._1,
          condensedNodeType = node._2.condensedNodeType,
          featureValues = node._2.featureValues,
        )
      })

    var rnn = RootedNodeNeighborhood(
      rootNode = Some(
        Node(
          nodeId = reduction.rootNodeId.get,
          condensedNodeType = reduction.rootNodeType,
          featureValues = reduction.rootNodeFeatures,
        ),
      ),
      neighborhood = Some(
        Graph(
          nodes = nodes,
          edges = edges,
        ),
      ),
    )

    if (postProcessFn.isDefined) {
      rnn = postProcessFn.get(rnn)
    }

    rnn.toByteArray
  }
  // Specifies the Encoder for the intermediate value type i.e. the buffer
  // encodes Product subclasses i.e. tuples, case classes, etc. https://www.scala-lang.org/api/2.12.9/scala/Product.html
  // See for more info: https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/Encoders$.html
  def bufferEncoder: Encoder[RnnUDAF.BufferRNN] = Encoders.product

  // Specifies the Encoder for the final output value type
  // We encode to binary due to the following issue:
  // org.apache.spark.SparkUnsupportedOperationException: [ENCODER_NOT_FOUND] Not found an encoder of the
  //    type com.google.protobuf.ByteString to Spark SQL internal representation.
  // importing `scalapb.spark.Implicits._` does not fix this issue. So we chose to encode to binary instead
  // of immediately spending cycles to resolve this issue.
  //
  // FWIW, this probably also saves on some compute if we are writing directly to a TFRecord file after this UDAF
  def outputEncoder: Encoder[Array[Byte]] = Encoders.BINARY
}
