package common.graphdb.local

import common.graphdb.DBClient
import common.graphdb.DBResult
import common.types.GraphTypes
import common.types.pb_wrappers.GraphMetadataPbWrapper
import common.utils.SparkSessionEntry
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.SparkSession
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp

import scala.collection.Map
import scala.collection.Set
import scala.collection.mutable
import scala.collection.mutable.HashSet

case class LocalGraphDbData(
  dbDstToSrcSet: Map[
    EdgeType,
    Map[GraphTypes.NodeId, Set[GraphTypes.NodeId]],
  ],
  dbSrcToDstSet: Map[
    EdgeType,
    Map[GraphTypes.NodeId, Set[GraphTypes.NodeId]],
  ])
object LocalDbClient {

  // Singleton object to store the broadcasted localGraphDb
  // See: https://spark.apache.org/docs/2.2.0/rdd-programming-guide.html#broadcast-variables
  // To initialize the broadcastedLocalDb, call init() method on the object with the
  // relevant data
  // @transient private var broadcastedLocalDb: Option[Broadcast[LocalGraphDbData]] = null

  @volatile private var instance: LocalDbClient = _

  def init(
    hydratedEdgeVIEW: String,
    graphMetadataPbWrapper: GraphMetadataPbWrapper,
  ): Unit = {

    /**
         * Initialize the singleton instance of LocalDbClient
         *
         * @param spark SparkSession
         * @param edgeDf DataFrame with columns `_from`, `_to`, `_condensed_edge_type`
         * @param numBuckets Number of buckets to partition data for each `_condensed_edge_type`
         *     That is, for each unique `_condensed_edge_type`, the data is partitioned into `numBuckets`
         * @return
         */
    if (instance == null) {
      this.synchronized {
        // Check if null again as by the time thread acquires lock, another thread might have already initialized.
        // We likely dont need this in normal code flow when used inside the driver, but this just makes it thread
        // safe from misuse. Especially because construction of LocalDbClient is expensive and takes time.
        if (instance == null) {
          val localGraphDbData = createLocalGraphDbData(
            graphMetadataPbWrapper = graphMetadataPbWrapper,
            hydratedEdgeVIEW = hydratedEdgeVIEW,
          )
          val spark: SparkSession = SparkSessionEntry.getActiveSparkSession
          val broadcastedData     = spark.sparkContext.broadcast(localGraphDbData)
          instance = new LocalDbClient(
            broadcastedData = broadcastedData,
            graphMetadataPbWrapper = graphMetadataPbWrapper,
          )
        }
      }
    }
  }

  def getInstance(): LocalDbClient = {
    if (instance == null) {
      throw new Exception(
        "LocalDbClient not initialized. Please call init() method first inside the driver.",
      )
    }
    instance
  }

  private def createLocalGraphDbData(
    graphMetadataPbWrapper: GraphMetadataPbWrapper,
    hydratedEdgeVIEW: String,
  ): LocalGraphDbData = {
    val dbDstToSrcSet =
      new mutable.HashMap[EdgeType, mutable.HashMap[GraphTypes.NodeId, mutable.HashSet[
        GraphTypes.NodeId,
      ]]]()
    val dbSrcToDstSet =
      new mutable.HashMap[EdgeType, mutable.HashMap[GraphTypes.NodeId, mutable.HashSet[
        GraphTypes.NodeId,
      ]]]()
    val condensedEdgeTypeMap = graphMetadataPbWrapper.graphMetadataPb.condensedEdgeTypeMap
    val spark: SparkSession  = SparkSessionEntry.getActiveSparkSession()
    spark
      .table(hydratedEdgeVIEW)
      .collect()
      .foreach(row => {
        val srcNodeId: GraphTypes.NodeId = row.getAs[GraphTypes.NodeId]("_from")
        val dstNodeId: GraphTypes.NodeId = row.getAs[GraphTypes.NodeId]("_to")
        val condensedEdgeType: GraphTypes.CondensedEdgeType =
          row.getAs[GraphTypes.CondensedEdgeType]("_condensed_edge_type")
        val edgeType: EdgeType = condensedEdgeTypeMap(condensedEdgeType)

        val dstNodeIdToSrcNodeIdSetMap = dbDstToSrcSet.getOrElseUpdate(
          edgeType,
          mutable.HashMap[GraphTypes.NodeId, mutable.HashSet[GraphTypes.NodeId]](),
        )
        val srcNodeIdToDstNodeIdSetMap = dbSrcToDstSet.getOrElseUpdate(
          edgeType,
          mutable.HashMap[GraphTypes.NodeId, mutable.HashSet[GraphTypes.NodeId]](),
        )
        dstNodeIdToSrcNodeIdSetMap
          .getOrElseUpdate(
            dstNodeId,
            mutable.HashSet[GraphTypes.NodeId](),
          )
          .add(srcNodeId)

        srcNodeIdToDstNodeIdSetMap
          .getOrElseUpdate(
            srcNodeId,
            mutable.HashSet[GraphTypes.NodeId](),
          )
          .add(dstNodeId)
      })
    val localGraphDbData = LocalGraphDbData(
      dbDstToSrcSet = dbDstToSrcSet,
      dbSrcToDstSet = dbSrcToDstSet,
    )
    localGraphDbData
  }

}
class LocalDbClient private (
  broadcastedData: Broadcast[LocalGraphDbData],
  graphMetadataPbWrapper: GraphMetadataPbWrapper)
    extends DBClient[DBResult]
    with Serializable {

  def connect(): Unit = {
    println("LocalTestSnapGraphServiceClient: connect")
  }

  def isConnected(): Boolean = {
    true
  }

  def terminate(): Unit = {
    println("LocalTestSnapGraphServiceClient: terminate")
  }

  def executeQuery(queryStr: types.LocalDbQueryString): DBResult = {
    val localDbQuery: types.LocalDbQuery   = types.LocalDbQuery(queryStr)
    val rootNodes: Seq[Node]               = localDbQuery.getRootNodes()
    val samplingOp: SamplingOp             = localDbQuery.getSamplingOp()
    val neighborhoodEdgeSet: HashSet[Edge] = HashSet.empty[Edge]
    val neighborhoodNodeSet: HashSet[Node] = HashSet.empty[Node]

    val useOutgoingEdges: Boolean = samplingOp.samplingDirection match {
      case SamplingDirection.OUTGOING => true
      case _                          => false
    }

    samplingOp.samplingMethod match {
      case SamplingOp.SamplingMethod.RandomUniform(value) => {
        val numNodesToSample: Int = value.numNodesToSample
        for (rootNode <- rootNodes) {
          val edgeType          = samplingOp.edgeType.get
          val condensedEdgeType = graphMetadataPbWrapper.edgeTypeToCondensedEdgeTypeMap(edgeType)
          val srcNodeType       = edgeType.srcNodeType
          val dstNodeType       = edgeType.dstNodeType
          val condensedSrcNodeType =
            graphMetadataPbWrapper.nodeTypeToCondensedNodeTypeMap(srcNodeType)
          val condensedDstNodeType =
            graphMetadataPbWrapper.nodeTypeToCondensedNodeTypeMap(dstNodeType)
          if (useOutgoingEdges) {
            val dstNodeIds: Set[GraphTypes.NodeId] = getDstNodes(
              edgeType = edgeType,
              srcNodeId = rootNode.nodeId,
            )
            // Sample numNodesToSample nodes from dstNodeIds for each srcNode
            val sampledDstNodeIds: Set[GraphTypes.NodeId] = dstNodeIds.take(numNodesToSample)
            for (dstNodeId <- sampledDstNodeIds) {
              neighborhoodEdgeSet += Edge(
                srcNodeId = rootNode.nodeId,
                dstNodeId = dstNodeId,
                condensedEdgeType = Some(condensedEdgeType),
              )
              neighborhoodNodeSet += Node(
                nodeId = dstNodeId,
                condensedNodeType = Some(condensedDstNodeType),
              )
            }
          } else {
            val srcNodeIds: Set[GraphTypes.NodeId] = getSrcNodes(
              edgeType = edgeType,
              dstNodeId = rootNode.nodeId,
            )
            // Sample numNodesToSample nodes from srcNodeIds for each dstNode
            val sampledSrcNodeIds: Set[GraphTypes.NodeId] = srcNodeIds.take(numNodesToSample)
            for (srcNodeId <- sampledSrcNodeIds) {

              neighborhoodEdgeSet += Edge(
                srcNodeId = srcNodeId,
                dstNodeId = rootNode.nodeId,
                condensedEdgeType = Some(condensedEdgeType),
              )
              neighborhoodNodeSet += Node(
                nodeId = srcNodeId,
                condensedNodeType = Some(condensedSrcNodeType),
              )
            }
          }

        }
      }
      case _ => { // Other cases are not implemented yet
        throw new NotImplementedError("Sampling method is not supported yet")
      }
    }
    val graphDbResult: DBResult = new DBResult()
    graphDbResult.insertRow(
      colNames = List(types.QUERY_RESPONSE_KEY),
      values = List(
        types.LocalDbQueryResponse(
          neighborhoodEdgeSet = neighborhoodEdgeSet.toSet[Edge],
          neighborhoodNodeSet = neighborhoodNodeSet.toSet[Node],
        ),
      ),
    )
    graphDbResult

  }

  private def getSrcNodes(
    edgeType: EdgeType,
    dstNodeId: GraphTypes.NodeId,
  ): Set[GraphTypes.NodeId] = {
    val localGraphDb: LocalGraphDbData = broadcastedData.value
    val dstNodeIdToSrcNodeIdSetMap: Map[GraphTypes.NodeId, Set[GraphTypes.NodeId]] =
      localGraphDb.dbDstToSrcSet.getOrElse(
        edgeType,
        throw new Exception(s"EdgeType: ${edgeType} not found in localGraphDb"),
      )
    dstNodeIdToSrcNodeIdSetMap.getOrElse(
      dstNodeId,
      Set(),
    ) // Return empty set incase no neighbors found
  }

  private def getDstNodes(
    edgeType: EdgeType,
    srcNodeId: GraphTypes.NodeId,
  ): Set[GraphTypes.NodeId] = {
    val localGraphDb: LocalGraphDbData = broadcastedData.value
    val srcNodeIdToDstNodeIdSetMap: Map[GraphTypes.NodeId, Set[GraphTypes.NodeId]] =
      localGraphDb.dbSrcToDstSet.getOrElse(
        edgeType,
        throw new Exception(s"EdgeType: ${edgeType} not found in localGraphDb"),
      )
    srcNodeIdToDstNodeIdSetMap.getOrElse(
      srcNodeId,
      Set(),
    ) // Return empty set incase no neighbors found
  }

}
