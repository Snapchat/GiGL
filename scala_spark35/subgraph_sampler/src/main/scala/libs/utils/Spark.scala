package libs.utils

import com.typesafe.scalalogging.LazyLogging
import common.utils.SparkSessionEntry.getActiveSparkSession
import common.utils.SparkSessionEntry.getNumCurrentShufflePartitions
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{functions => F}
import org.apache.spark.storage.StorageLevel

import java.util.UUID.randomUUID

object Spark extends Serializable with LazyLogging {
  val spark: SparkSession = getActiveSparkSession
  val samplingSeed: Int   = 42

  private def generateUniqueSuffix: String = {

    /** create a unique suffix for table view names, upon initialization of this calss
     */
    val uniqueTempViewSuffix = randomUUID.toString.replace("-", "_")
    "_" + uniqueTempViewSuffix
  }

  val uniqueTempViewSuffix: String = generateUniqueSuffix

  def downsampleNumberOfNodes(
    numSamples: Int,
    dfVIEW: String,
  ): String = {

    /**
     * Downsamples number of root nodes to numSamples
     * @param numSamples number of root nodes to sample
     * @param dfVIEW
     * @param rootNodeColName
     */
    // @spark: we use RAND to sample uniformely, but this is not deterministic,which is not problematic in this case.
    // in case we wish to enforce determinism, we can use sort and take first numSamples
    val downsampledDF = spark.sql(f"""
          SELECT
            *
          FROM
            ${dfVIEW}
          LIMIT
            ${numSamples}
          """)
    //   (yliu2) using below df.sample can be fast(similar to above query) but we need to do a count which take time
    //    val num_nodes: Long = spark.table(dfVIEW).count() //this might have different effects if the df is cached or not
    //    val ratio: Double = numSamples.toDouble / num_nodes.toDouble
    //    println(s"Downsampling with ratio ${ratio}")
    //    val downsampledDF =
    //      spark.table(dfVIEW).sample(withReplacement = false, fraction = ratio, seed = samplingSeed)

    val downsampledVIEW = "downsampledDF" + uniqueTempViewSuffix
    downsampledDF.createOrReplaceTempView(downsampledVIEW)

    //    if we don 't force sequential execution nor break the parallelization we will have different samples.
    //    So to be extra safe we need another cache here
    val cachedDownSampledVIEW = applyCachingToDataFrameVIEW(
      dfVIEW = downsampledVIEW,
      forceTriggerCaching = true,
      withRepartition = false,
    )
    cachedDownSampledVIEW
  }

  def applyCachingToDataFrameVIEW(
    dfVIEW: String,
    forceTriggerCaching: Boolean,
    withRepartition: Boolean,
    repartitionFactor: Int = -1,
    colNameForRepartition: String = "",
    storageLevel: StorageLevel = StorageLevel.DISK_ONLY,
  ): String = {

    /**
     * Used for performance optimization and more importantly for accuracy of sampling.
     * @param dfVIEW associated with the DataFrame we want to cache
     * @param triggerCachingViaCount whether to apply caching by triggering an action, ie df.count(). If false, another action such as df.write() will trigger cache.
     * @param withRepartition whether to repartition the df associate with subgraph dfVIEW. If there are spills in final stages, set this to true
     * @param repartitionFactor should be determined based on graph size.
     * @param colNameForRepartition name of column (col that has root nodes, or node ids)
     */
    //  @spark: StorageLevel=DISK_ONLY, if data is cached in MEMORY instead of DISK, it leads to faster spill to disk during shuffles. Since both cached data and shuffle data compete for Storage memory of executors. As long as local SSD is attached to each worker, StorageLevel=DISK_ONLY is fast enough.
    // NOTE: repartition MUST be done before caching, since repartitioning clears corresponding cached data
    val df = spark.table(dfVIEW)
    val cachedDF = applyCachingToDataFrame(
      df = df,
      forceTriggerCaching = forceTriggerCaching,
      withRepartition = withRepartition,
      repartitionFactor = repartitionFactor,
      colNameForRepartition = colNameForRepartition,
      storageLevel = storageLevel,
    )
    val cachedDfVIEW = "cached_" + dfVIEW + uniqueTempViewSuffix
    cachedDF.createOrReplaceTempView(cachedDfVIEW)
    cachedDfVIEW
  }

  def applyCachingToDataFrame(
    df: DataFrame,
    forceTriggerCaching: Boolean,
    withRepartition: Boolean,
    repartitionFactor: Int = -1,
    colNameForRepartition: String = "",
    storageLevel: StorageLevel = StorageLevel.DISK_ONLY,
  ): DataFrame = {

    /**
     * Used for performance optimization and more importantly for accuracy of sampling.
     * @param df DataFrame we want to cache
     * @param triggerCachingViaCount whether to apply caching by triggering an action, ie df.count(). If false, another action such as df.write() will trigger cache.
     * @param withRepartition whether to repartition the df associate with subgraph dfVIEW. If there are spills in final stages, set this to true
     * @param repartitionFactor should be determined based on graph size.
     * @param colNameForRepartition name of column (col that has root nodes, or node ids)
     */
    //  @spark: StorageLevel=DISK_ONLY, if data is cached in MEMORY instead of DISK, it leads to faster spill to disk during shuffles. Since both cached data and shuffle data compete for Storage memory of executors. As long as local SSD is attached to each worker, StorageLevel=DISK_ONLY is fast enough.
    // NOTE: repartition MUST be done before caching, since repartitioning clears corresponding cached data
    val cachedDF = if (withRepartition) {
      val repartitionedDF = applyRepartitionToDataFrame(
        df = df,
        colName = colNameForRepartition,
        repartitionFactor = repartitionFactor,
      )
      repartitionedDF.persist(storageLevel)
    } else {
      df.persist(storageLevel)
    }

    if (forceTriggerCaching == true) {
      // use count as action, to trigger cache completely (ie distribute/materialize data on worker nodes).
      // Count is not expensive since it does NOT accumulate data in the driver node, but rather in workers node. [Execution time for count+cache (mau data): 5 min]
      cachedDF.count()
    }
    cachedDF
  }

  def applyRepartitionToDataFrame(
    df: DataFrame,
    colName: String,
    repartitionFactor: Int,
  ): DataFrame = {

    /**
     * @spark: used for repartitioning dataframes in intermediate stages of the Spark job
     * to avoid spills as data size grows.
     * e.g., curNumShufflePartitions = 300 and we have initial partition size 200Mb
     * stages proceed, and data size for each partition grows to 5000Mb, spills or OOM error happen
     * in this case we repartition by a factor eg 10, so numPartitions = 300 * 10 = 3000
     * and data size for each partition will be 5000Mb/10= 500 Mb and spill/OOMs will be mitigated
     * NOTE: repartition clears cached data, first repartition then cache if required
     * @param colName: the column name that joins are performed on
     */
    val numShufflePartitions = getNumCurrentShufflePartitions(spark = spark)
    val repartitionedDF = df.repartition(
      numPartitions = numShufflePartitions * repartitionFactor,
      partitionExprs = F.col(colName),
    )
    repartitionedDF
  }

  def applyRepartitionToDataFrameVIEW(
    dfVIEW: String,
    colName: String,
    repartitionFactor: Int,
  ): String = {

    /**
     * @spark: used for repartitioning dataframes in intermediate stages of the Spark job
     *         to avoid spills as data size grows.
     *         e.g., curNumShufflePartitions = 300 and we have initial partition size 200Mb
     *         stages proceed, and data size for each partition grows to 5000Mb, spills or OOM error happen
     *         in this case we repartition by a factor eg 10, so numPartitions = 300 * 10 = 3000
     *         and data size for each partition will be 5000Mb/10= 500 Mb and spill/OOMs will be mitigated
     *         NOTE: repartition clears cached data, first repartition then cache if required
     * @param colName : the column name that joins are performed on
     */
    val df = spark.table(dfVIEW)
    val repartitionedDF = applyRepartitionToDataFrame(
      df = df,
      colName = colName,
      repartitionFactor = repartitionFactor,
    )
    val repartitionedDfVIEW = "repartitioned_" + dfVIEW + uniqueTempViewSuffix
    repartitionedDF.createOrReplaceTempView(repartitionedDfVIEW)

    repartitionedDfVIEW
  }

  def applyCoalesceToDataFrameVIEW(
    dfVIEW: String,
    colName: String,
    coalesceFactor: Int,
  ): String = {

    /**
     * @spark: used for coalescing dataframes in intermediate stages of the Spark job
     *         to avoid small file problems.
     * @param colName : the column name that joins are performed on
     * @param coalesceFactor : the number of partitions to coalesce to
     *                       ex.10 then new # of partitions will be getCurrentShufflePartitionsNum(spark) / 10
     */
    val df                   = spark.table(dfVIEW)
    val numShufflePartitions = getNumCurrentShufflePartitions(spark = spark)
    val coalescedDF          = df.coalesce(numShufflePartitions / coalesceFactor)
    val coalescedDfVIEW      = "coalesced_" + dfVIEW + uniqueTempViewSuffix
    coalescedDF.createOrReplaceTempView(coalescedDfVIEW)

    coalescedDfVIEW
  }
}
