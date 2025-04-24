package libs.utils

import com.typesafe.scalalogging.LazyLogging
import common.types.GraphTypes.CondensedEdgeType
import common.types.GraphTypes.CondensedNodeType
import common.utils.SparkSessionEntry.getActiveSparkSession
import common.utils.TFRecordIO.RecordTypes
import common.utils.TFRecordIO.readDataframeFromTfrecord
import libs.utils.Spark.applyCachingToDataFrame
import libs.utils.Spark.uniqueTempViewSuffix
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{functions => F}

import scala.collection.mutable.ListBuffer

object DataLoading extends Serializable with LazyLogging {
  val spark: SparkSession = getActiveSparkSession

  def loadNodeDataframeIntoSparkSql(
    condensedNodeType: CondensedNodeType,
    featureKeys: List[String],
    nodeIdKey: String,
    nodeDfUri: String,
    shouldCacheResult: Boolean = false,
  ): DataFrame = {

    /** For given condensed_node_type, loads the node feature/type dataframe with columns:
     * _node_id: Integer, _node_features: Array(), _condensed_node_type: Integer. Returns VIEW name
     * associated with the node dataframe.
     */
    val rawNodeDF: DataFrame = readDataframeFromTfrecord(
      uri = nodeDfUri,
      recordType = RecordTypes.ExampleRecordType,
    )

    val rawNodeDFVIEW: String = "rawNodeDF" + uniqueTempViewSuffix
    rawNodeDF.createOrReplaceTempView(
      rawNodeDFVIEW,
    )

    var colDtypeMap = collection.mutable.Map[String, String]()
    rawNodeDF.dtypes.foreach { case (columnName, columnType) =>
      colDtypeMap += (columnName -> columnType)
    }

    var columnWithOperations = new ListBuffer[String]()
    featureKeys.foreach { columnName =>
      if (colDtypeMap(columnName).startsWith("ArrayType")) { // cast to array<float>
        columnWithOperations += s"CAST(${columnName} AS array<float>)"
      } else { // convert + cast to array<float>
        columnWithOperations += s"CAST(array(${columnName}) AS array<float>)"
      }
    }
    val hydratedNodeDF: DataFrame = spark.sql(f"""
            SELECT /*+ REPARTITION(_node_id) */
              CAST(${nodeIdKey} AS INTEGER) AS _node_id,
              flatten(array(${columnWithOperations.mkString(", ")})) AS _node_features,
              ${condensedNodeType} AS _condensed_node_type
            FROM ${rawNodeDFVIEW}
              """)

    if (shouldCacheResult) {
      val cachedHydratedNodeDF = applyCachingToDataFrame(
        df = hydratedNodeDF,
        forceTriggerCaching = true,
        withRepartition = false,
      )
      cachedHydratedNodeDF
    } else {
      hydratedNodeDF
    }

  }

  def loadEdgeDataframeIntoSparkSql(
    condensedEdgeType: CondensedEdgeType,
    srcNodeIdKey: String,
    dstNodeIdKey: String,
    edgeDfUri: String,
    edgeFeatureKeys: List[String] = List.empty,
    shouldCacheResult: Boolean = false,
  ): DataFrame = {

    /** For given condensed_edge_type loads edge dataframe with columns: _from: Int, _to: Int,
             * _condensed_edge_type: Int, _edge_features: Array()(if features are provided) (used for
             * hydration)
            */
    val hasEdgeFeatures: Boolean =
      edgeFeatureKeys.nonEmpty // hasEdgeFeatures can be inferred from preprocessed metadata

    println(
      s"loadEdgeDataframeIntoSparkSql - srcNodeIdKey: ${srcNodeIdKey}, dstNodeIdKey: ${dstNodeIdKey}, " +
        s"edgeDfUri: ${edgeDfUri}, hasEdgeFeatures: ${hasEdgeFeatures}, edgeFeatureKeys: ${edgeFeatureKeys}",
    )

    val rawEdgeDF: DataFrame = readDataframeFromTfrecord(
      uri = edgeDfUri,
      recordType = RecordTypes.ExampleRecordType,
    )
    val rawEdgeDFVIEW: String = "rawEdgeDF" + uniqueTempViewSuffix
    rawEdgeDF.createOrReplaceTempView(
      rawEdgeDFVIEW,
    )
    // rename df-edge columns and add edge types and features columns
    val hydratedEdgeDF = if (hasEdgeFeatures == true) {
      // below handles cases where we have array of edge features
      var colDtypeMap = collection.mutable.Map[String, String]()
      rawEdgeDF.dtypes.foreach { case (columnName, columnType) =>
        colDtypeMap += (columnName -> columnType)
      }
      var columnWithOperations = new ListBuffer[String]()
      edgeFeatureKeys.foreach { columnName =>
        if (colDtypeMap(columnName).startsWith("ArrayType")) { // cast to array<float>
          columnWithOperations += s"CAST(${columnName} AS array<float>)"
        } else { // convert + cast to array<float>
          columnWithOperations += s"CAST(array(${columnName}) AS array<float>)"
        }
      }
      spark.sql(f"""
                SELECT
                  CAST($srcNodeIdKey AS INTEGER) AS _from,
                  CAST($dstNodeIdKey AS INTEGER) AS _to,
                  ${condensedEdgeType} AS _condensed_edge_type,
                  flatten(array(${columnWithOperations.mkString(", ")})) AS _edge_features
                FROM
                  ${rawEdgeDFVIEW}
                """)
    } else {
      val emptyEdgeFeature: Seq[Float] =
        Seq.empty[Float] // @spark: explicitly assigns empty list to edge features -- to avoid errors during protobuf conversion
      // NOTE: writing below using SQL leads to conflicts between SQL types and Scala types.
      // That's why below is written in Dataframe API, this way
      // conversions between spark.sql types and scala types are handeled under the hood
      spark
        .table(rawEdgeDFVIEW)
        .select(
          F.col(srcNodeIdKey).cast(IntegerType).alias("_from"),
          F.col(dstNodeIdKey).cast(IntegerType).alias("_to"),
          F.lit(condensedEdgeType).alias("_condensed_edge_type"),
          F.typedLit(emptyEdgeFeature).alias("_edge_features"),
        )
    }

    if (shouldCacheResult) {
      // @spark: we cache hydratedEdgeDF here to trim DAG and avoid running same stages again and again,
      // saves at least 1hr of runtime for MAU
      val cachedHydratedEdgeDF = applyCachingToDataFrame(
        df = hydratedEdgeDF,
        forceTriggerCaching = true,
        withRepartition = true,
        repartitionFactor = 1,
        colNameForRepartition = "_from",
      )
      cachedHydratedEdgeDF
    } else {
      hydratedEdgeDF
    }
  }

  def loadUnhydratedEdgeDataframeIntoSparkSql(
    hydratedEdgeVIEW: String,
  ): String = {

    /** Loads edge dataframe with columns: _src_node: Int, _dst_node: Int (used for sampling)
     *
     * @param hydratedEdgeVIEW with columns:
     *                         _from: Int, src node id
     *                         _to: Int, dst node id
     *                         _condensed_edge_type: Int
     *                         _edge_features: Array, icluding all features
     */
    val unhydratedEdgeVIEW: String = "unhydratedEdgeDF" + uniqueTempViewSuffix
    val unhydratedEdgeDF = spark.sql(f"""
            SELECT
              _from AS _src_node,
              _to AS _dst_node
            FROM
              ${hydratedEdgeVIEW}
            """)

    unhydratedEdgeDF.createOrReplaceTempView(unhydratedEdgeVIEW)
    unhydratedEdgeVIEW
  }
}
