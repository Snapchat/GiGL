package common.graphdb

import com.vesoft.nebula.client.graph.SessionPool
import com.vesoft.nebula.client.graph.SessionPoolConfig
import com.vesoft.nebula.client.graph.data.HostAddress
import com.vesoft.nebula.client.graph.data.ResultSet
import common.types.pb_wrappers.GbmlConfigPbWrapper

import scala.collection.JavaConversions._

// TODO (yliu2-sc) Need to update Proto parameters to be able to pass in graphDB client parameters
//    Holding off on this since the required parameters could change after atlas update interface
//    We hardcode for now while testing since the clusters are fixed too

// Nebula Java client example https://github.com/vesoft-inc/nebula-java/blob/release-3.3/examples/src/main/java/com/vesoft/nebula/examples/GraphClientExample.java
// Nebula return type ResultSet definition https://github.com/vesoft-inc/nebula-java/blob/master/client/src/main/java/com/vesoft/nebula/client/graph/data/ResultSet.java
// ValueWrapper https://github.com/vesoft-inc/nebula-java/blob/master/client/src/main/java/com/vesoft/nebula/client/graph/data/ValueWrapper.java#L371

//class NebulaGraphDBClient(gbmlConfigWrapper: GbmlConfigPbWrapper)
//    extends GraphDBClient[ResultSet](gbmlConfigWrapper) {
//
//  private var session: Session =
//    _ // getSession is the action that establishes connection. Nebula session manager.in graphd manages the session connections.
//  private var pool: NebulaPool = _
//
//  val graphdbArgs = gbmlConfigWrapper.subgraphSamplerConfigPb.graphdbArgs
//  val space = graphdbArgs.getOrElse(
//    "space",
//    throw new Exception("NebulaClient requires 'space' field in graphdbArgs."),
//  )
//
//  def connect(): Unit = {
//    pool = new NebulaPool with Serializable
//    val nebulaPoolConfig: NebulaPoolConfig = new NebulaPoolConfig with Serializable
//    nebulaPoolConfig.setMaxConnSize(1) // each client object is tied to a single session
//
//    val hosts = graphdbArgs
//      .getOrElse(
//        "hosts",
//        throw new Exception("NebulaClient requires 'hosts' field in graphdbArgs."),
//      )
//      .split(",")
//
//    // all hosts should be using the same port
//    val port = graphdbArgs
//      .getOrElse(
//        "port",
//        throw new Exception("NebulaClient requires 'port' field in graphdbArgs."),
//      )
//      .toInt
//
//    val addresses = hosts.map(host => new HostAddress(host, port)).toList
//
//    // init connection pool
//    val initResult = pool.init(addresses, nebulaPoolConfig)
//    if (!initResult) {
//      logger.error("Connection pool init failed.")
//    } else {
//      logger.info("OK. Connection pool init success.")
//    }
//
//    // TODO (yliu2-sc) do we need to add retry logic?
//    try {
//      // getSession is the action that establishes connection
//      session = pool.getSession(
//        "root",
//        "nebula",
//        true,
//      ) // Setting to true will auto reconnect if connection is lost
//      logger.info("OK. Get session success. Connected to nebula graphd.")
//    } catch {
//      case e: Exception =>
//        logger.error(f"Could not get session connection to nebula. ${e.getMessage}")
//    }
//
//    executeQuery(
//      f"USE ${space}",
//    )
//  }
//
//  def isConnected(): Boolean = {
//    pool != null && session != null
//  }
//
//  def terminate(): Unit = {
//    if (pool != null && session != null) {
//      session.release()
//      session.close()
//      session = null
//
//      pool.close()
//      pool = null
//    } else {
//      logger.warn(
//        f"No Nebula connection initialized when calling terminate(). pool = ${pool}, session = ${session}",
//      )
//    }
//  }
//
//  def executeQuery(query_string: String): ResultSet = {
//    // TODO (yliu2-sc) add retry logic, return failed status if retries fails
//    assert(
//      isConnected(),
//      f"Nebula client session is not connected to graphd.  pool = ${pool}, session = ${session}",
//    )
//
//    var resultSet: ResultSet = null
//    try {
//      resultSet = session.execute(
//        query_string,
//      )
//    } catch {
//      case e: Exception =>
//        e.printStackTrace()
//        logger.error(f"Could not execute query. ${e.getMessage}")
//        throw e
//    }
//    if (resultSet != null && !resultSet.isSucceeded()) {
//      logger.error(f"Error in Execute Query: ${resultSet.getErrorMessage()}")
//    }
//    resultSet
//  }
//
//  def executeQueryBatch(query_strings: Seq[String]): Seq[ResultSet] = {
//    // Implementation goes here
//    // Perhaps nebula team can provide batch query support
//    throw new UnsupportedOperationException("executeQueryBatch is not implemented yet.")
//    Seq.empty[ResultSet]
//  }
//}

// SessionPool seems to create a new NebulaSession or fetch an idle session in execute call,
// SessionPool has thread-safe and retry built-in
// Example https://github.com/vesoft-inc/nebula-java/blob/master/README.md
// Source code https://github.com/vesoft-inc/nebula-java/blob/master/client/src/main/java/com/vesoft/nebula/client/graph/SessionPool
//
//There are some limitations while using the session pool:
//
// 1. There MUST be an existing space in the DB before initializing the session pool.
// 2. Each session pool is corresponding to a single USER and a single Space. This is to ensure that the user's access control is consistent. i.g. The same user may have different access privileges in different spaces. If you need to run queries in different spaces, you may have multiple session pools.
// 3. Every time when sessinPool.execute() is called, the session will execute the query in the space set in the session pool config.
// 4. Commands that alter passwords or drop users should NOT be executed via session pool.
//
//To use SessionPool, you must config the space to connect for SessionPool.
// The SessionPool is thread-safe, and support retry(release old session and get available session from SessionPool) for both connection error, s
// ession error and execution error(caused by bad storaged server), and the retry mechanism needs users to config retryTimes and intervalTime between retrys.
//
//And SessionPool maintains servers' status, can isolation broken graphd server and auto routing restarted graphd server when you need to execute with new session，
// meaning your parallel is larger than the idle session number in the session pool.

class NebulaGraphDBClient(gbmlConfigWrapper: GbmlConfigPbWrapper)
    extends GraphDBClient[ResultSet](gbmlConfigWrapper) {
  private var sessionPool: SessionPool = _

  val graphdbArgs = gbmlConfigWrapper.subgraphSamplerConfigPb.graphDbConfig.get.graphDbArgs
  val space: String = graphdbArgs.getOrElse(
    "space",
    throw new Exception("NebulaClient requires 'space' field in graphdbArgs."),
  )
  def connect(): Unit = {
    val user     = graphdbArgs.getOrElse("user", "root")
    val password = graphdbArgs.getOrElse("password", "nebula")

    val hosts = graphdbArgs
      .getOrElse(
        "hosts",
        throw new Exception("NebulaClient requires 'hosts' field in graphdbArgs."),
      )
      .split(";")

    // all hosts should be using the same port
    val port = graphdbArgs
      .getOrElse(
        "port",
        throw new Exception("NebulaClient requires 'port' field in graphdbArgs."),
      )
      .toInt

    val addresses = hosts.map(host => new HostAddress(host, port)).toList

    val sessionPoolConfig: SessionPoolConfig =
      new SessionPoolConfig(addresses, space, user, password) with Serializable
    val retryTimes = graphdbArgs.getOrElse("retries", "3").toInt
    val intervalTime =
      graphdbArgs
        .getOrElse("retry_interval", "1000")
        .toInt // interval between retries, set 1000ms = 1s
    val maxSessionSize = graphdbArgs.getOrElse("max_session_size", "3").toInt

    sessionPoolConfig.setRetryTimes(retryTimes)
    sessionPoolConfig.setIntervalTime(intervalTime)
    sessionPoolConfig.setMaxSessionSize(maxSessionSize)

    sessionPool = new SessionPool(sessionPoolConfig) with Serializable

    if (!sessionPool.init()) {
      logger.error("session pool init failed.")
      return
    } else {
      logger.debug("OK. session pool init success.")
    }
  }

  def isConnected(): Boolean = {
    sessionPool != null && sessionPool.isActive()
  }

  def terminate(): Unit = {
    if (isConnected()) {
      sessionPool.close()
      assert(sessionPool.isClosed())
      sessionPool = null
      logger.debug("Session pool connection closed.")
    } else {
      logger.warn(
        f"No Nebula sessionPool initialized or connected when calling terminate(). sessionPool = ${sessionPool}",
      )
    }
  }

  def executeQuery(query_string: String): ResultSet = {
    var resultSet: ResultSet = null
    try {
      resultSet = sessionPool.execute(
        query_string,
      ) // sessionPool.execute() internal implementation will check if session is avaialble
    } catch {
      case e: Exception =>
        e.printStackTrace()
        logger.error(f"Could not execute query. ${e.getMessage}")
        println(f"Could not execute query. ${e.getMessage}")
        throw e
    }
    if (resultSet != null && !resultSet.isSucceeded()) {
      logger.error(resultSet.getErrorMessage())
      println(resultSet.getErrorMessage())
    }
    resultSet
  }

  def executeQueryBatch(query_strings: Seq[String]): Seq[ResultSet] = {
    // Implementation goes here
    // Perhaps nebula team can provide batch query support
    throw new UnsupportedOperationException("executeQueryBatch is not implemented yet.")
    Seq.empty[ResultSet]
  }

}
