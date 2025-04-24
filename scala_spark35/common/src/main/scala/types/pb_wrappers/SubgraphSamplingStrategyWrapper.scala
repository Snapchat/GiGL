package common.types.pb_wrappers

import common.types.GraphTypes
import common.types.SamplingOpDAG
import snapchat.research.gbml.subgraph_sampling_strategy.SubgraphSamplingStrategy

case class SubgraphSamplingStrategyWrapper(
  val subgraphSamplingStrategyPb: SubgraphSamplingStrategy) {

  def getNodeTypeToSamplingOpDagMap: Map[GraphTypes.NodeType, SamplingOpDAG] = {
    if (!subgraphSamplingStrategyPb.strategy.isMessagePassingPaths) {
      throw new IllegalArgumentException(
        "Strategy is not supported. Provided: " + subgraphSamplingStrategyPb.strategy,
      )
    }
    // TODO: (svij) Validate Sampling Ops are logically correct
    subgraphSamplingStrategyPb.getMessagePassingPaths.paths
      .map(path => (path.rootNodeType, SamplingOpDAG.from(path.samplingOps)))
      .toMap
  }

}
