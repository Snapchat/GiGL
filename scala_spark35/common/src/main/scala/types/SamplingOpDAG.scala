package common.types
import snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp

import scala.collection.mutable.HashMap

import collection.mutable.Buffer
import collection.mutable.Set

class SamplingOpNode(
  val samplingOp: SamplingOp,
  val parentSamplingOpNodes: Set[SamplingOpNode],
  val childSamplingOpNodes: Set[SamplingOpNode])
    extends Serializable {
  def opName: String             = samplingOp.opName
  def parentOpNames: Set[String] = parentSamplingOpNodes.map(_.opName)
  def childOpNames: Set[String]  = childSamplingOpNodes.map(_.opName)
}

object SamplingOpDAG {
  def from(samplingOps: Seq[SamplingOp]): SamplingOpDAG = {
    // Firstly create the raw SamplingOpNodes indexed by the op name
    val opNameToSamplingOpNodeMap: HashMap[String, SamplingOpNode] =
      HashMap.empty[String, SamplingOpNode]
    val rootSamplingOpNodes: Buffer[SamplingOpNode] = Buffer.empty[SamplingOpNode]
    samplingOps.foreach(op => {
      val samplingOpNode = new SamplingOpNode(
        samplingOp = op,
        parentSamplingOpNodes = Set(),
        childSamplingOpNodes = Set(),
      )
      opNameToSamplingOpNodeMap += (op.opName -> samplingOpNode)

      if (op.inputOpNames.isEmpty) {
        rootSamplingOpNodes += samplingOpNode
      }

    })
    // Now create the actual DAG and identify the root nodes
    for (samplingOp <- samplingOps) {
      val currSamplingOpNode = opNameToSamplingOpNodeMap(samplingOp.opName)
      val parentinputOpNames = samplingOp.inputOpNames
      val parentSamplingOpNodes =
        opNameToSamplingOpNodeMap.filterKeys(parentinputOpNames.contains).values.toSet
      parentSamplingOpNodes.foreach(parentSamplingOpNode => {
        parentSamplingOpNode.childSamplingOpNodes.add(currSamplingOpNode)
        currSamplingOpNode.parentSamplingOpNodes.add(parentSamplingOpNode)
      })
    }
    SamplingOpDAG(rootSamplingOpNodes)
  }
}
case class SamplingOpDAG(
  rootSamplingOpNodes: Seq[SamplingOpNode])
