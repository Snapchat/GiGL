package common.utils
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{functions => F}

import SparkSessionEntry.getActiveSparkSession

object SlottedJoiner {

  /**
   * This class helps handle OOM and disk space issues in Spark jobs during large table joins.
   * Instead of one big join, it partitions the left table into smaller tables and joins 
   * them iteratively with the right table, ensuring better scalability with commodity hardware.
   *
   * Usage:
   * val leftDf = ...
   * vla rightDf = ...
   * val numSlots = 10
   * val slottedLeftDf = SlottedJoiner.computeSlotsOnDataframe(
   *    df=leftDf, 
   *    columnToComputeSlotOn="joinKey", 
   *    numSlots=numSlots
   * )
   * 
   * // Caching helps us avoid recomputing the tables
   * cacher.createDiskPartitionedTable(
   *    df = leftSlottedDF,
   *    repartitionOnColumn = "joinKey",
   *    outputTableName = "leftSlottedTable",
   * )
   * cacher.createDiskPartitionedTable(
   *    df = rightDf,
   *    repartitionOnColumn = "joinKey",
   *    outputTableName = "rightTable",
   * )
   * val sqlJoin = SELECT * FROM leftTable JOIN rightTable ON joinKey
   * val joinIterator: Iterator[DataFrame] = SlottedJoiner.performSlottedJoinOnLeftSlottedTable(
   *    sql = sqlJoin,
   *    numSlots = numSlots,
   *    leftSlottedTableName = srcPartitionedRawEdgeTableName,
   *    rightTableName = dstPartitionedRawEdgeTable,
   *  )
   *  for (df <- joinIterator) {
   *    // Process the joined dataframe for each slot i.e.:
   *    writeDfToStorage(df)
   *  }
   *
  */

  val SLOT_NUM_COLUMN_NAME = "SLOT_NUM"

  def computeSlotsOnDataframe(
    df: DataFrame,
    columnToComputeSlotOn: String,
    numSlots: Int,
  ): DataFrame = {

    /**
      * Compute the slot number for each row in the dataframe based on the given column. The dataframe can then be
      * used to later to perform slotted joins: @see SlottedJoiner#performSlottedJoin.
      * @param df: DataFrame to compute the slot number on
      * @param columnToComputeSlotOn: Column to compute the slot number on
      * @param numSlots: Number of slots to divide the data into
      * @return: DataFrame with an additional column `SLOT_NUM_COLUMN_NAME` which contains the slot number for each row
    */
    assert(
      df.columns.contains(columnToComputeSlotOn),
      s"Column $columnToComputeSlotOn not found in dataframe",
    )
    assert(
      df.schema(columnToComputeSlotOn)
        .dataType
        .isInstanceOf[org.apache.spark.sql.types.NumericType],
      s"Column $columnToComputeSlotOn is not of type numeric",
    )
    assert(
      df.columns.contains(SLOT_NUM_COLUMN_NAME) == false,
      s"Column $SLOT_NUM_COLUMN_NAME already exists in dataframe",
    )
    df.withColumn(SLOT_NUM_COLUMN_NAME, F.col(columnToComputeSlotOn) % numSlots)
  }

  def performSlottedJoinOnLeftSlottedTable(
    sql: String,
    numSlots: Int,
    leftSlottedTableName: String,
    rightTableName: String,
  ): Iterator[DataFrame] = {

    /**
      * Perform a slotted join on a left slotted table with a right table. The left table should have a column `SLOT_NUM_COLUMN_NAME` which is used to perform the join.
      * The join is performed on each slot separately and the results are returned as a stream of DataFrames.
      * @param sql: SQL query to perform the join
      * @param numSlots: Number of slots in the slottd tables.
      * @param leftSlottedTableName: Name of the left slotted table @see SlottedJoiner#computeSlotsOnDataframe.
      * @param rightTableName: Name of the right table.
      * @return: Stream of DataFrames, each containing the result of the join for a slot number. Use this to iterate over
      * the results of the join and perform further processing.
    */
    val spark: SparkSession = getActiveSparkSession
    assert(
      sql.contains(leftSlottedTableName) && sql.contains(rightTableName),
      "SQL query should contain both left and right table names",
    )
    assert(
      spark.table(leftSlottedTableName).columns.contains(SLOT_NUM_COLUMN_NAME),
      s"Left slotted table should have a column `${SLOT_NUM_COLUMN_NAME}` i.e. it should be a slotted table",
    )

    val joinIterator = new Iterator[DataFrame] {
      var slotNum = 0

      override def hasNext: Boolean = slotNum < numSlots

      override def next(): DataFrame = {
        val leftTableName = f"${leftSlottedTableName}_${slotNum}"
        val currSlotQuery = sql
          .replace(leftSlottedTableName, leftTableName)

        val queryWithSlottedTables = f"""
                  WITH ${leftTableName} as (
                      SELECT * 
                      FROM ${leftSlottedTableName}
                      WHERE ${SLOT_NUM_COLUMN_NAME} = ${slotNum}
                  ) ${currSlotQuery}
              """
        slotNum += 1
        spark.sql(queryWithSlottedTables)
      }
    }
    joinIterator

  }

  def performSlottedJoin(
    sql: String,
    numSlots: Int,
    leftSlottedTableName: String,
    rightSlottedTableName: String,
  ): Stream[DataFrame] = {

    /**
      * Perform a slotted join on two tables. The tables should have a column `SLOT_NUM_COLUMN_NAME` which is used to perform the join.
      * The join is performed on each slot sequentially and the results are returned as a stream of DataFrames.
      * Breaking the join into slots helps in reducing the memory usage and improves the performance of the join.

      * @param sql: SQL query to perform the join
      * @param numSlots: Number of slots in the slottd tables.
      * @param leftSlottedTableName: Name of the left slotted table @see SlottedJoiner#computeSlotsOnDataframe.
      * @param rightSlottedTableName: Name of the right slotted table. @see SlottedJoiner#computeSlotsOnDataframe.
      * @return: Stream of DataFrames, each containing the result of the join for a slot number. Use this to iterate over
      * the results of the join and perform further processing.
    */
    val spark: SparkSession = getActiveSparkSession
    assert(
      sql.contains(leftSlottedTableName) && sql.contains(rightSlottedTableName),
      "SQL query should contain both left and right slotted table names",
    )
    assert(
      spark.table(leftSlottedTableName).columns.contains(SLOT_NUM_COLUMN_NAME) &&
        spark.table(rightSlottedTableName).columns.contains(SLOT_NUM_COLUMN_NAME),
      s"Both left and right slotted tables should have a column ${SLOT_NUM_COLUMN_NAME}",
    )

    // Perform the join for each slot sequentially
    (0 until numSlots).toStream.map { slotNum =>
      val rightTableName = f"${rightSlottedTableName}_$slotNum"
      val leftTableName  = f"${leftSlottedTableName}_$slotNum"
      val currSlotQuery = sql
        .replace(leftSlottedTableName, leftTableName)
        .replace(rightSlottedTableName, rightTableName)
      val queryWithSlottedTables = f"""
                WITH ${leftTableName} as (
                    SELECT * 
                    FROM ${leftSlottedTableName}
                    WHERE ${SLOT_NUM_COLUMN_NAME} = ${slotNum}
                ),
                ${rightTableName} as (
                    SELECT * 
                    FROM ${rightSlottedTableName}
                    WHERE ${SLOT_NUM_COLUMN_NAME} = ${slotNum}
                ) ${currSlotQuery}
            """
      spark.sql(queryWithSlottedTables)
    }
  }

}
