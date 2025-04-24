package libs.sampler

import common.graphdb.DBClient
import common.graphdb.DBResult
import common.graphdb.KHopSamplerService
import common.graphdb.local.LocalDbClient
import common.graphdb.local.LocalDbQueryResponseTranslator
import common.graphdb.nebula.NebulaQueryResponseTranslator
import common.types.pb_wrappers.GbmlConfigPbWrapper
import common.types.pb_wrappers.ResourceConfigPbWrapper
import common.utils.Env

import java.lang.reflect.InvocationTargetException
import scala.collection.AbstractMap
import scala.collection.mutable.HashMap
import scala.util.Failure
import scala.util.Success
import scala.util.Try

object KHopSamplerServiceFactory {

  def initializeLocalDbIfNeeded(
    gbmlConfigWrapper: GbmlConfigPbWrapper,
    hydratedEdgeVIEW: String,
  ): Unit = {
    println("Will Initialize Local Db If Needed...")
    if (shouldUseLocalServiceSampler(gbmlConfigWrapper)) {
      println("Initializing Local Db")
      initializeLocalDb(
        gbmlConfigWrapper = gbmlConfigWrapper,
        hydratedEdgeVIEW = hydratedEdgeVIEW,
      )
    }
  }

  def createKHopServiceSampler(
    gbmlConfigWrapper: GbmlConfigPbWrapper,
    giglResourceConfigWrapper: ResourceConfigPbWrapper,
  ): KHopSamplerService = {

    if (shouldUseLocalServiceSampler(gbmlConfigWrapper)) {
      println("Using Local Service Sampler")
      createLocalServiceSampler(
        gbmlConfigWrapper = gbmlConfigWrapper,
      )
    } else { // Default to GraphDBSampler
      println("Using GraphDBSampler")
      createGraphDBSampler(
        gbmlConfigWrapper = gbmlConfigWrapper,
        giglResourceConfigWrapper = giglResourceConfigWrapper,
      )
    }
  }

  private def createGraphDBSampler(
    gbmlConfigWrapper: GbmlConfigPbWrapper,
    giglResourceConfigWrapper: ResourceConfigPbWrapper,
  ): KHopSamplerService = {

    val graphDBClientArgs = new HashMap[String, String]()
    if (!Env.isRunningOnGCP) {
      println("Using Personal Issuer")
    } else {
      val serviceAccount =
        giglResourceConfigWrapper.sharedResourceConfigPb.commonComputeConfig.get.gcpServiceAccountEmail
      println("Using Service Account for auth token issuer: " + serviceAccount)
      graphDBClientArgs.put("service_account", serviceAccount)

      // We copy over all args, it's up to the client class to determine which to use.
      val graphDbArgs = gbmlConfigWrapper.subgraphSamplerConfigPb.graphDbConfig.get.graphDbArgs
      graphDbArgs.foreach { case (k, v) => graphDBClientArgs.put(k, v) }
    }
    // TODO(kmonte): Remove these branches and throw once migration is done.
    val graphDBClientClassPath: String =
      if (
        gbmlConfigWrapper.subgraphSamplerConfigPb.graphDbConfig.get.graphDbSamplerConfig.isEmpty
      ) {
        throw new IllegalArgumentException(
          "No reflection class path provided with gmblconfig.dataset_config.subgraph_sampler_config.graph_db_config.graph_db_sampler_config.graph_db_client_class_path.",
        )
      } else {
        gbmlConfigWrapper.subgraphSamplerConfigPb.graphDbConfig.get.graphDbSamplerConfig.get.graphDbClientClassPath
      }
    println("Class path for graph db client: " + graphDBClientClassPath)

    // We do this try/cath so that we can more easily propgagate error messages
    // from the underlying DBClient class up.
    // Without this, the InvocationTargetException just points to the `newInstance()` call
    // Which is not very useful.
    val dbClient: DBClient[DBResult] = Try {
      val graphDBClientClass = Class.forName(graphDBClientClassPath)
      graphDBClientClass
        .getConstructor(classOf[AbstractMap[String, String]])
        .newInstance(graphDBClientArgs)
        .asInstanceOf[DBClient[DBResult]]
    } match {
      case Success(client) =>
        println("Successfully created graphDBServiceClient!")
        client
      case Failure(e: InvocationTargetException) =>
        println(s"Unable to create GraphDB client for $graphDBClientClassPath!")
        println(s"Underlying error: ${e.getCause}")
        throw e
      case Failure(e) =>
        println(s"An unexpected error occurred: ${e.getMessage}")
        throw e
    }

    new GraphDBSampler(
      graphMetadataPbWrapper = gbmlConfigWrapper.graphMetadataPbWrapper,
      dbClient = dbClient,
      queryResponseTranslator = new NebulaQueryResponseTranslator(
        graphMetadataPbWrapper = gbmlConfigWrapper.graphMetadataPbWrapper,
      ),
    )
  }

  private def createLocalServiceSampler(
    gbmlConfigWrapper: GbmlConfigPbWrapper,
  ): KHopSamplerService = {
    println("Creating LocalServiceSampler")
    val localDbClient: LocalDbClient = LocalDbClient.getInstance()
    val localQueryResponseTranslator = new LocalDbQueryResponseTranslator()
    new GraphDBSampler(
      graphMetadataPbWrapper = gbmlConfigWrapper.graphMetadataPbWrapper,
      dbClient = localDbClient,
      queryResponseTranslator = localQueryResponseTranslator,
    )
  }

  private def shouldUseLocalServiceSampler(
    gbmlConfigWrapper: GbmlConfigPbWrapper,
  ): Boolean = {
    println(
      "shouldUseLocalServiceSampler graphDbArgs: " + gbmlConfigWrapper.subgraphSamplerConfigPb.graphDbConfig.get.graphDbArgs,
    )
    gbmlConfigWrapper.subgraphSamplerConfigPb.graphDbConfig.get.graphDbArgs
      .getOrElse(
        "use_local_sampler",
        "false",
      )
      .toBoolean
  }

  private def initializeLocalDb(
    gbmlConfigWrapper: GbmlConfigPbWrapper,
    hydratedEdgeVIEW: String,
  ): Unit = {
    LocalDbClient.init(
      hydratedEdgeVIEW = hydratedEdgeVIEW,
      graphMetadataPbWrapper = gbmlConfigWrapper.graphMetadataPbWrapper,
    )

  }

}
