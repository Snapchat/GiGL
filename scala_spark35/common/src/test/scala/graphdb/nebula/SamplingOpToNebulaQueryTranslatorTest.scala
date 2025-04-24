import common.graphdb.nebula.GraphEntityTranslator
import common.graphdb.nebula.NebulaQueryResponseTranslator
import common.graphdb.nebula.types.NebulaEdgeType
import common.graphdb.nebula.types.NebulaVID
import common.types.GraphTypes
import common.types.pb_wrappers.GraphMetadataPbWrapper
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.GraphMetadata
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform
import snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp
import snapchat.research.gbml.subgraph_sampling_strategy.TopK

object NebulaQueryResponseTranslatorTest {

  val edgeType1: EdgeType = EdgeType(srcNodeType = "user", dstNodeType = "story", relation = "to")
  val condensedEdgeType1: GraphTypes.CondensedEdgeType = 1
  val condensedNodeTypeUser                            = 0
  val condensedNodeTypeStory                           = 1
  val graphMetadataPb = new GraphMetadata(
    edgeTypes = Seq(edgeType1),
    nodeTypes = Seq("user", "story"),
    condensedEdgeTypeMap = Map(condensedEdgeType1 -> edgeType1),
    condensedNodeTypeMap = Map(condensedNodeTypeUser -> "user", condensedNodeTypeStory -> "story"),
  )
  val graphMetadataPbWrapper = new GraphMetadataPbWrapper(graphMetadataPb)
  val translator = new NebulaQueryResponseTranslator(
    graphMetadataPbWrapper = graphMetadataPbWrapper,
  )
  val numNodesToSample = 10
  val node: Node       = Node(nodeId = 1, condensedNodeType = Some(condensedNodeTypeStory))
  val nebulaVID: NebulaVID = GraphEntityTranslator.nebulaVIDFromNodeComponents(
    nodeId = node.nodeId,
    condensedNodeType = node.condensedNodeType.get,
  )
  val nebulaEdgeType: NebulaEdgeType = GraphEntityTranslator.nebulaEdgeType(
    edgeType1,
  )
}
class NebulaQueryResponseTranslatorTest extends AnyFunSuite {
  test("translate RandomUniform SamplingOp to Nebula Query") {
    val samplingOp = SamplingOp(
      opName = "SamplingOpRandomUniform",
      edgeType = Some(NebulaQueryResponseTranslatorTest.edgeType1),
      inputOpNames = Seq(),
      samplingMethod = SamplingOp.SamplingMethod.RandomUniform(
        RandomUniform(NebulaQueryResponseTranslatorTest.numNodesToSample),
      ),
    )

    val query = NebulaQueryResponseTranslatorTest.translator.translate(
      samplingOp = samplingOp,
      rootNode = NebulaQueryResponseTranslatorTest.node,
    )
    assert(
      query == "GO 1 STEP " +
        s"FROM ${NebulaQueryResponseTranslatorTest.nebulaVID} " +
        s"OVER ${NebulaQueryResponseTranslatorTest.nebulaEdgeType} REVERSELY " +
        "YIELD src(edge) as src, dst(edge) as dst " +
        s"LIMIT [${NebulaQueryResponseTranslatorTest.numNodesToSample}]",
    )
  }

  test("translate RandomWeighted SamplingOp to Nebula Query") {
    val samplingOp = SamplingOp(
      opName = "SamplingOpRandomWeighted",
      edgeType = Some(NebulaQueryResponseTranslatorTest.edgeType1),
      inputOpNames = Seq(),
      samplingMethod = SamplingOp.SamplingMethod.RandomWeighted(
        RandomWeighted(
          numNodesToSample = NebulaQueryResponseTranslatorTest.numNodesToSample,
          edgeFeatName = "edgeFeatName",
        ),
      ),
    )
    val query = NebulaQueryResponseTranslatorTest.translator.translate(
      samplingOp = samplingOp,
      rootNode = NebulaQueryResponseTranslatorTest.node,
    )
    assert(
      query == "GO 1 STEP " +
        s"FROM ${NebulaQueryResponseTranslatorTest.nebulaVID} " +
        s"OVER ${NebulaQueryResponseTranslatorTest.nebulaEdgeType} REVERSELY " +
        "YIELD src(edge) as src, dst(edge) as dst, " +
        s"${NebulaQueryResponseTranslatorTest.nebulaEdgeType}.edgeFeatName * rand() as edgeFeatName | " +
        "ORDER BY $-.edgeFeatName DESC | " +
        s"LIMIT ${NebulaQueryResponseTranslatorTest.numNodesToSample} | " +
        "YIELD $-.src AS src, $-.dst AS dst",
    )
  }

  test("translate TopK SamplingOp to Nebula Query") {
    val samplingOp = SamplingOp(
      opName = "SamplingOpTopK",
      edgeType = Some(NebulaQueryResponseTranslatorTest.edgeType1),
      inputOpNames = Seq(),
      samplingMethod = SamplingOp.SamplingMethod.TopK(
        TopK(
          numNodesToSample = NebulaQueryResponseTranslatorTest.numNodesToSample,
          edgeFeatName = "edgeFeatName",
        ),
      ),
    )
    val query = NebulaQueryResponseTranslatorTest.translator.translate(
      samplingOp = samplingOp,
      rootNode = NebulaQueryResponseTranslatorTest.node,
    )
    assert(
      query == "GO 1 STEP " +
        s"FROM ${NebulaQueryResponseTranslatorTest.nebulaVID} " +
        s"OVER ${NebulaQueryResponseTranslatorTest.nebulaEdgeType} REVERSELY " +
        "YIELD src(edge) as src, dst(edge) as dst, " +
        s"${NebulaQueryResponseTranslatorTest.nebulaEdgeType}.edgeFeatName as edgeFeatName | " +
        "ORDER BY $-.edgeFeatName DESC | " +
        s"LIMIT ${NebulaQueryResponseTranslatorTest.numNodesToSample} | " +
        "YIELD $-.src AS src, $-.dst AS dst",
    )
  }

}
