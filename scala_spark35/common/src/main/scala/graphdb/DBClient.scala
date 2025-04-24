package common.graphdb

// trait for GraphDB clients
trait DBClient[T] {
  def connect(): Unit

  def terminate(): Unit

  // TODO: Can add a strict interface for the return type
  def executeQuery(query_string: String): T

  def isConnected(): Boolean
}
