import common.graphdb.DBClient
import common.graphdb.DBResult
import common.graphdb.nebula.GraphEntityTranslator
import common.graphdb.nebula.NebulaQueryResponseTranslator
import common.test.testLibs.SharedSparkSession
import common.types.SamplingOpDAG
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.ResourceConfigPbWrapper
import libs.sampler.KHopSamplerServiceFactory
import libs.utils.SGSTask
import org.scalatest.BeforeAndAfterAll
import org.scalatest._
import org.scalatest.funsuite.AnyFunSuite
import snapchat.research.gbml.gbml_config.GbmlConfig
import snapchat.research.gbml.gigl_resource_config.GiglResourceConfig
import snapchat.research.gbml.gigl_resource_config.SharedResourceConfig
import snapchat.research.gbml.graph_schema.Edge
import snapchat.research.gbml.graph_schema.EdgeType
import snapchat.research.gbml.graph_schema.Graph
import snapchat.research.gbml.graph_schema.GraphMetadata
import snapchat.research.gbml.graph_schema.Node
import snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

import scala.collection.AbstractMap

import collection.immutable.HashMap

object TestConstants {
  val SRC_NODE_ID = 6789
  val DST_NODE_ID = 4567

  val PAPER_AUTHOR_EDGE_TYPE = 12340000
  val AUTHOR_PAPER_EDGE_TYPE = 43210000

  val CONDENSED_AUTHOR_NODE_TYPE = 1234
  val CONDENSED_PAPER_NODE_TYPE  = 4321
}

class MockDbClient(args: AbstractMap[String, String]) extends DBClient[DBResult] {
  def connect(): Unit        = {}
  def isConnected(): Boolean = { true }
  def terminate(): Unit      = {}
  def executeQuery(queryStr: String): DBResult = {
    val res = new DBResult()
    res.insertRow(
      List(
        NebulaQueryResponseTranslator.RESULT_DST_NODE_ID_COL_NAME,
        NebulaQueryResponseTranslator.RESULT_SRC_NODE_ID_COL_NAME,
      ),
      List(
        GraphEntityTranslator.nebulaVIDFromNodeComponents(
          TestConstants.DST_NODE_ID,
          TestConstants.CONDENSED_AUTHOR_NODE_TYPE,
        ),
        GraphEntityTranslator.nebulaVIDFromNodeComponents(
          TestConstants.SRC_NODE_ID,
          TestConstants.CONDENSED_PAPER_NODE_TYPE,
        ),
      ),
    )
    res
  }
}

class KHopSamplerServiceFactoryTest
    extends AnyFunSuite
    with BeforeAndAfterAll
    with SharedSparkSession {
  test("Tries to Create Local DB before initialization") {
    val configPb = GbmlConfig(datasetConfig =
      Some(
        GbmlConfig.DatasetConfig(subgraphSamplerConfig =
          Some(
            GbmlConfig.DatasetConfig.SubgraphSamplerConfig(graphDbConfig =
              Some(
                GbmlConfig.GraphDBConfig(
                  graphDbArgs = HashMap("use_local_sampler" -> "true"),
                ),
              ),
            ),
          ),
        ),
      ),
    )
    val configPbWrapper = GbmlConfigPbWrapper(configPb)
    assertThrows[java.lang.Exception] {
      KHopSamplerServiceFactory.createKHopServiceSampler(
        configPbWrapper,
        ResourceConfigPbWrapper(GiglResourceConfig()),
      )
    }
  }
  test("Creates local DB") {
    // Based on scala/common/src/test/assets/subgraph_sampler/heterogeneous/node_anchor_based_link_prediction/frozen_gbml_config_graphdb_dblp_local.yaml
    val configPb = GbmlConfig(
      sharedConfig = Some(
        GbmlConfig.SharedConfig(
          preprocessedMetadataUri =
            "common/src/test/assets/subgraph_sampler/heterogeneous/node_anchor_based_link_prediction/preprocessed_metadata.yaml",
        ),
      ),
      datasetConfig = Some(
        GbmlConfig.DatasetConfig(subgraphSamplerConfig =
          Some(
            GbmlConfig.DatasetConfig.SubgraphSamplerConfig(graphDbConfig =
              Some(
                GbmlConfig.GraphDBConfig(
                  graphDbArgs = HashMap("use_local_sampler" -> "true"),
                ),
              ),
            ),
          ),
        ),
      ),
      graphMetadata = Some(
        GraphMetadata(
          condensedEdgeTypeMap = HashMap(
            0 -> EdgeType(
              dstNodeType = "paper",
              relation = "author_to_paper",
              srcNodeType = "author",
            ),
            1 -> EdgeType(
              dstNodeType = "author",
              relation = "paper_to_author",
              srcNodeType = "paper",
            ),
          ),
          condensedNodeTypeMap = HashMap(0 -> "author", 1 -> "paper"),
        ),
      ),
    )
    val configPbWrapper = GbmlConfigPbWrapper(configPb)
    val hydratedEdgeView = SGSTask.loadHydratedEdgeDataFrame(
      condensedEdgeTypes = Seq(0, 1),
      gbmlConfigWrapper = configPbWrapper,
    )
    KHopSamplerServiceFactory.initializeLocalDbIfNeeded(configPbWrapper, hydratedEdgeView)
    KHopSamplerServiceFactory.createKHopServiceSampler(
      configPbWrapper,
      ResourceConfigPbWrapper(GiglResourceConfig()),
    )
  }
  test("GraphDBClient must be provided through config.") {
    val configPb = GbmlConfig(
      graphMetadata = Some(GraphMetadata()),
      datasetConfig = Some(
        GbmlConfig.DatasetConfig(subgraphSamplerConfig =
          Some(
            GbmlConfig.DatasetConfig.SubgraphSamplerConfig(graphDbConfig =
              Some(
                GbmlConfig.GraphDBConfig(
                  graphDbArgs = HashMap(
                    "use_local_sampler" -> "false",
                    "route_tag"         -> "TAG",
                    "graph_space"       -> "GRAPH_SPACE_GRAPHML_DBLP",
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    )
    val configPbWrapper = GbmlConfigPbWrapper(configPb)
    val resourceConfigPb = GiglResourceConfig(
      sharedResource = GiglResourceConfig.SharedResource.SharedResourceConfig(
        SharedResourceConfig(
          commonComputeConfig = Some(
            SharedResourceConfig.CommonComputeConfig(
              gcpServiceAccountEmail = "foo.service.account@gcp.com",
            ),
          ),
        ),
      ),
    )
    val resourceConfigWrapper = ResourceConfigPbWrapper(resourceConfigPb)
    assertThrows[java.lang.Exception] {
      KHopSamplerServiceFactory.createKHopServiceSampler(
        configPbWrapper,
        resourceConfigWrapper,
      )
    }
  }

  test("Creates GraphDBClient through reflection from config") {
    val configPb = GbmlConfig(
      graphMetadata = Some(
        GraphMetadata(
          condensedEdgeTypeMap = HashMap(
            TestConstants.AUTHOR_PAPER_EDGE_TYPE -> EdgeType(
              dstNodeType = "paper",
              relation = "author_to_paper",
              srcNodeType = "author",
            ),
            TestConstants.PAPER_AUTHOR_EDGE_TYPE -> EdgeType(
              dstNodeType = "author",
              relation = "paper_to_author",
              srcNodeType = "paper",
            ),
          ),
          condensedNodeTypeMap = HashMap(
            TestConstants.CONDENSED_AUTHOR_NODE_TYPE -> "author",
            TestConstants.CONDENSED_PAPER_NODE_TYPE  -> "paper",
          ),
        ),
      ),
      datasetConfig = Some(
        GbmlConfig.DatasetConfig(subgraphSamplerConfig =
          Some(
            GbmlConfig.DatasetConfig.SubgraphSamplerConfig(graphDbConfig =
              Some(
                GbmlConfig.GraphDBConfig(
                  graphDbArgs = HashMap(
                    "use_local_sampler" -> "false",
                  ),
                  graphDbSamplerConfig = Some(
                    GbmlConfig.GraphDBConfig.GraphDBServiceConfig(
                      graphDbClientClassPath = "MockDbClient",
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    )
    val configPbWrapper = GbmlConfigPbWrapper(configPb)
    val resourceConfigPb = GiglResourceConfig(
      sharedResource = GiglResourceConfig.SharedResource.SharedResourceConfig(
        SharedResourceConfig(
          commonComputeConfig = Some(
            SharedResourceConfig.CommonComputeConfig(
              gcpServiceAccountEmail = "foo.service.account@gcp.com",
            ),
          ),
        ),
      ),
    )
    val resourceConfigWrapper = ResourceConfigPbWrapper(resourceConfigPb)
    val samplerService = KHopSamplerServiceFactory.createKHopServiceSampler(
      configPbWrapper,
      resourceConfigWrapper,
    )

    val subgraph: RootedNodeNeighborhood = samplerService.getKHopSubgraphForRootNode(
      Node(
        nodeId = TestConstants.DST_NODE_ID,
        condensedNodeType = Some(TestConstants.CONDENSED_AUTHOR_NODE_TYPE),
      ),
      SamplingOpDAG.from(
        Seq(
          SamplingOp(
            opName = "test_op",
            edgeType = Some(
              EdgeType(
                dstNodeType = "author",
                relation = "paper_to_author",
                srcNodeType = "paper",
              ),
            ),
            samplingMethod = SamplingOp.SamplingMethod.RandomUniform(
              value = RandomUniform(numNodesToSample = 1),
            ),
          ),
        ),
      ),
    )
    val expectedSubgraph = RootedNodeNeighborhood(
      rootNode = Some(
        Node(
          nodeId = TestConstants.DST_NODE_ID,
          condensedNodeType = Some(TestConstants.CONDENSED_AUTHOR_NODE_TYPE),
        ),
      ),
      neighborhood = Some(
        Graph(
          edges = Seq(
            Edge(
              srcNodeId = TestConstants.SRC_NODE_ID,
              dstNodeId = TestConstants.DST_NODE_ID,
              condensedEdgeType = Some(TestConstants.PAPER_AUTHOR_EDGE_TYPE),
            ),
          ),
          nodes = Seq(
            Node(
              nodeId = TestConstants.DST_NODE_ID,
              condensedNodeType = Some(TestConstants.CONDENSED_AUTHOR_NODE_TYPE),
            ),
            Node(
              nodeId = TestConstants.SRC_NODE_ID,
              condensedNodeType = Some(TestConstants.CONDENSED_PAPER_NODE_TYPE),
            ),
          ),
        ),
      ),
    )
    assert(
      subgraph.equals(expectedSubgraph),
      "Subgraph should be equal.\nGot " + subgraph + "\nExpected: " + expectedSubgraph,
    )
  }
}
