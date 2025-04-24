package common.graphdb

import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer

class DBResult {

  val _colValues: HashMap[String, ListBuffer[Any]] = new HashMap[String, ListBuffer[Any]]()

  def colValues(colName: String): List[Any] = _colValues.get(colName).get.toList
  def rowValues(rowNum: Int): List[Any]     = _colValues.values.map(_.apply(rowNum)).toList

  def insertRow(
    colNames: List[String],
    values: List[Any],
  ): Unit = {
    assert(colNames.size == values.size, "alias and value should have the same size")
    if (numColumns() == 0) {
      for (i <- 0 until colNames.size) {
        val colName = colNames(i)
        _colValues.put(colName, new ListBuffer())
      }
    }

    assert(
      colNames.size == numColumns(),
      s"Inserting row ${colNames} with different number of columns than expected: ${columnNames()}",
    )
    for (i <- 0 until colNames.size) {
      val colName = colNames(i)
      _colValues.get(colName).get.append(values(i))
    }
  }

  def numColumns(): Int = {
    return _colValues.keySet.size
  }

  def isEmpty(): Boolean = numColumns() == 0

  def columnNames(): List[String] = {
    return _colValues.keySet.toList
  }

  def numRows(): Int = {
    return _colValues.get(_colValues.keySet.head).get.size
  }

  override def toString(): String = {
    if (numColumns() == 0) {
      return "Empty result"
    }
    val sb        = new StringBuilder()
    val delimiter = ", "
    val colNames  = _colValues.keySet.toList
    for (colName <- colNames) {
      sb.append(s"${colName}${delimiter}")
    }
    sb.delete(sb.length - delimiter.length, sb.length)
    val currLen = sb.length
    sb.append("\n" + "=" * currLen + "\n")
    val maxColSize = _colValues.values.map(_.size).max
    for (i <- 0 until maxColSize) {
      for (colName <- colNames) {
        val _colValuesList = _colValues.get(colName).get
        if (i < _colValuesList.size) {
          sb.append(s"${_colValuesList(i)}${delimiter}")
        } else {
          sb.append(delimiter)
        }
      }
      sb.delete(sb.length - delimiter.length, sb.length)
      sb.append("\n")
    }
    return sb.toString()
  }
}
