package common.utils
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.{functions => F}

object Cacher {
  def createDiskPartitionedTable(
    df: DataFrame,
    targetPartitionNum: Int,
    repartitionOnColumn: String,
    outputTableName: String,
  ): Unit = {

    /**
      * Repartition the dataframe on the given column and write it to disk as a bucketed table
      * @param targetPartitionNum: Number of partitions to repartition the dataframe into
      * @param df: DataFrame to be repartitioned + bucketed and written to disk
      * @param repartitionOnColumn: Column to repartition the dataframe on
      * @param outputTableName: Name of the table, which can then be used directly in SQL queries

      * @return: None
    */

    df.repartition(targetPartitionNum, F.col(repartitionOnColumn))
      .write
      .mode("overwrite")
      .format("parquet")
      .bucketBy(targetPartitionNum, repartitionOnColumn)
      .saveAsTable(outputTableName)
  }

}
