package common.graphdb

import com.typesafe.scalalogging.LazyLogging
import common.types.pb_wrappers.GbmlConfigPbWrapper

// Factory class for GraphDB clients
abstract class GraphDBClient[T](gbmlConfigWrapper: GbmlConfigPbWrapper)
    extends Serializable
    with LazyLogging {
  def connect(): Unit

  def terminate(): Unit

  def executeQuery(query_string: String): T

  def executeQueryBatch(query_strings: Seq[String]): Seq[T]

  def isConnected(): Boolean
}
