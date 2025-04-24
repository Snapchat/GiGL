import common.test.testLibs.SharedSparkSession
import common.utils.SlottedJoiner
import org.apache.spark.sql.DataFrame
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

class SlottedJoinerTest extends AnyFunSuite with BeforeAndAfterAll with SharedSparkSession {

  test(
    "Slotted Join can break a dataframe into multiple tables and then perform joins for each slot",
  ) {
    import sqlImplicits._
    val edgeData = Seq(
      (0, 1),
      (0, 2),
      (1, 11),
      (2, 22),
      (11, 0),
      (22, 0),
    )
    val edgeDf = edgeData.toDF("src_node", "dst_node")
    val expectedNeighborhoodData = Seq(
      (1, 0, 11),
      (1, 0, 22),
      (2, 0, 11),
      (2, 0, 22),
      (11, 1, 0),
      (22, 2, 0),
      (0, 11, 1),
      (0, 22, 2),
    )
    val expectedDf = expectedNeighborhoodData.toDF("root_node", "1_hop_node", "2_hop_node")

    val numSlots = 3
    val slottedOnSrc = SlottedJoiner.computeSlotsOnDataframe(
      df = edgeDf,
      columnToComputeSlotOn = "src_node",
      numSlots = numSlots,
    )
    val slottedOnDst = SlottedJoiner.computeSlotsOnDataframe(
      df = edgeDf,
      columnToComputeSlotOn = "dst_node",
      numSlots = numSlots,
    )

    // Ensure computed slots are as expected
    assert(slottedOnDst.columns.length == 3) // src_node, dst_node, SLOT_NUM
    assert(slottedOnSrc.columns.length == 3)
    assert(slottedOnDst.columns.contains(SlottedJoiner.SLOT_NUM_COLUMN_NAME))
    assert(slottedOnSrc.columns.contains(SlottedJoiner.SLOT_NUM_COLUMN_NAME))

    slottedOnSrc.createOrReplaceTempView("slottedOnSrc")
    slottedOnDst.createOrReplaceTempView("slottedOnDst")
    assert(
      sparkTest
        .sql(s"SELECT DISTINCT ${SlottedJoiner.SLOT_NUM_COLUMN_NAME} FROM slottedOnSrc")
        .count() == numSlots,
    )
    assert(
      sparkTest
        .sql(s"SELECT DISTINCT ${SlottedJoiner.SLOT_NUM_COLUMN_NAME} FROM slottedOnSrc")
        .count() == numSlots,
    )
    // Ensure same nodes exist in the same slot across both tables
    // This test only works because we have each node listed in both src and dst
    // TODO: (svij-sc) modify so we can also test where same set of nodes are not present in both src and dst columns
    for (i <- (0 until numSlots)) {
      assert(
        sparkTest
          .sql(
            s"SELECT src_node FROM slottedOnSrc WHERE ${SlottedJoiner.SLOT_NUM_COLUMN_NAME} = ${i}",
          )
          .collect()
          .map(_.getInt(0))
          .toSet == sparkTest
          .sql(
            s"SELECT dst_node FROM slottedOnDst WHERE ${SlottedJoiner.SLOT_NUM_COLUMN_NAME} = ${i}",
          )
          .collect()
          .map(_.getInt(0))
          .toSet,
      )
    }

    var countNumJoins = 0
    val joinedTableStream: Stream[DataFrame] = SlottedJoiner.performSlottedJoin(
      sql = """
            SELECT
                slottedOnSrc.dst_node as root_node,
                slottedOnDst.dst_node as 1_hop_node,
                slottedOnDst.src_node as 2_hop_node
            FROM 
                slottedOnSrc 
            JOIN 
                slottedOnDst 
            ON 
                slottedOnSrc.src_node = slottedOnDst.dst_node
        """,
      numSlots = numSlots,
      leftSlottedTableName = "slottedOnSrc",
      rightSlottedTableName = "slottedOnDst",
    )

    var dfUnion: Option[DataFrame] = None
    for (df <- joinedTableStream) {
      countNumJoins += 1

      if (dfUnion.isEmpty) {
        dfUnion = Some(df)
      } else {
        dfUnion = Some(dfUnion.get.union(df))
      }
    }
    assert(countNumJoins == numSlots)
    assert(dfUnion.get.collect().toSet == expectedDf.collect().toSet)
  }

}
