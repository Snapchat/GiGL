package common.graphdb.nebula

import com.typesafe.scalalogging.LazyLogging
import com.vesoft.nebula.client.graph.SessionPool
import com.vesoft.nebula.client.graph.SessionPoolConfig
import com.vesoft.nebula.client.graph.data.HostAddress
import com.vesoft.nebula.client.graph.data.ResultSet
import common.graphdb.DBClient

import scala.collection.JavaConversions._

/**
 For Nebula Graph docs, see https://docs.nebula-graph.io/3.6.0/#getting_started
  Nebula Java client https://docs.nebula-graph.io/3.6.0/14.client/4.nebula-java-client/

  SessionPool creates new NebulaSessions or fetch an idle session in execute call,
  SessionPool has thread-safe and retry built-in
  Example https://github.com/vesoft-inc/nebula-java/blob/master/README.md
  Source code https://github.com/vesoft-inc/nebula-java/blob/master/client/src/main/java/com/vesoft/nebula/client/graph/SessionPool

  There are some limitations while using the session pool:
  1. There MUST be an existing graph space in the DB before initializing the session pool.
  2. Each session pool is corresponding to a single USER and a single graph Space. This is to ensure that the user's access control is consistent. i.g. The same user may have different access privileges in different spaces. If you need to run queries in different spaces, you may have multiple session pools.
  3. Every time when sessinPool.execute() is called, the session will execute the query in the space set in the session pool config.
  4. Commands that alter passwords or drop users should NOT be executed via session pool.

  To use SessionPool, you must config the graph space to connect for SessionPool.
  The SessionPool is thread-safe, and support retry(release old session and get available session from SessionPool) for both connection error,
  session error and execution error(caused by bad storaged server), and the retry mechanism needs users to config retryTimes and intervalTime between retrys.
 
 
  This class needs to be serializable if defined outside of mapPartitions, nebula client however is has underlying classes that are not serializable
  ConnectionPool + getSession - java.io.NotSerializableException: com.vesoft.nebula.client.graph.net.RoundRobinLoadBalancer
  SessionPool - Task not serializable: java.io.NotSerializableException: java.util.concurrent.ScheduledThreadPoolExecutor
  each partition has it's own graphDBClient instance/connection, since it's on different executors and threads
  each graphDBClient instance is tied to otne connection, ie. Nebula session
  Nebula sessions are not thread-safe, each partition has it's own connection.
  SessionPool implementation however is threadsafe
  there is always a worry that we're creating and destroying too many objects, the partition size plays a factor here yet to see if it's an issue

 **/

/**
 * NebulaDBClient is a client for connecting to a Nebula Graph database.
 *
 * @param graphSpace The name of the graph space in Nebula Graph to connect to.
 * @param hosts A sequence of hostnames or IP addresses of the Nebula Graph services.
 * @param port The port number of the Nebula Graph services.
 * @param user The username to authenticate with Nebula Graph.
 * @param password The password to authenticate with Nebula Graph.
 * @param numRetries The number of times to retry the connection in case of failure. Default is 3.
 * @param retryInterval The interval time (in milliseconds) between retries. Default is 1000.
 * @param maxSessions The maximum number of sessions that can be created in the session pool. Default is 3.
 */
class NebulaDBClient(
  graphSpace: String,
  hosts: Seq[String],
  port: Int,
  user: Option[String] = None,
  password: Option[String] = None,
  numRetries: Int = 3,
  retryInterval: Int = 1000,
  maxSessions: Int = 3)
    extends DBClient[ResultSet]
    with Serializable
    with LazyLogging {
  private var sessionPool: SessionPool = _

  val nebulaUser: String     = user.getOrElse("root")
  val nebulaPassword: String = password.getOrElse("nebula")

  def connect(): Unit = {
    val addresses = hosts.map(host => new HostAddress(host, port)).toList

    val sessionPoolConfig: SessionPoolConfig =
      new SessionPoolConfig(addresses, graphSpace, nebulaUser, nebulaPassword) with Serializable

    sessionPoolConfig.setRetryTimes(numRetries)
    sessionPoolConfig.setIntervalTime(retryInterval)
    sessionPoolConfig.setMaxSessionSize(maxSessions)

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
      ) // sessionPool.execute() internal implementation will check if session is available
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

}
