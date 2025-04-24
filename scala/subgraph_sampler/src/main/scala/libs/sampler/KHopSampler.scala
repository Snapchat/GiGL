package libs.sampler

import com.typesafe.scalalogging.LazyLogging
import common.types.Metapath
import snapchat.research.gbml.training_samples_schema.RootedNodeNeighborhood

trait KHopSampler extends Serializable with LazyLogging {
  def getKHopSubgraphForRootNodes(
    rootNodeIds: Seq[Int],
    metapaths: Seq[Metapath],
    numNeighborsToSample: Int,
  ): Seq[RootedNodeNeighborhood]

}
