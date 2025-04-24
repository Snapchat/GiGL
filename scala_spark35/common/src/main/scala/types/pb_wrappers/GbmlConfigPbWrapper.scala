package common.types.pb_wrappers

import common.utils.ProtoLoader
import snapchat.research.gbml.flattened_graph_metadata.FlattenedGraphMetadata
import snapchat.research.gbml.gbml_config.GbmlConfig
import snapchat.research.gbml.graph_schema.GraphMetadata
import snapchat.research.gbml.preprocessed_metadata.PreprocessedMetadata

case class GbmlConfigPbWrapper(val gbmlConfigPb: GbmlConfig) {
  def sharedConfigPb = gbmlConfigPb.sharedConfig.get
  def preprocessedMetadataWrapper: PreprocessedMetadataPbWrapper = PreprocessedMetadataPbWrapper(
    preprocessedMetadataPb = ProtoLoader.populateProtoFromYaml[PreprocessedMetadata](
      sharedConfigPb.preprocessedMetadataUri,
    ),
  )
  def datasetConfigPb: GbmlConfig.DatasetConfig = gbmlConfigPb.datasetConfig.get
  def subgraphSamplerConfigPb: GbmlConfig.DatasetConfig.SubgraphSamplerConfig =
    datasetConfigPb.subgraphSamplerConfig.get
  def flattenedGraphMetadataPb: FlattenedGraphMetadata = sharedConfigPb.flattenedGraphMetadata.get
  def taskMetadataPb: GbmlConfig.TaskMetadata          = gbmlConfigPb.taskMetadata.get
  def graphMetadataPb: GraphMetadata                   = gbmlConfigPb.graphMetadata.get
  def graphMetadataPbWrapper: GraphMetadataPbWrapper   = GraphMetadataPbWrapper(graphMetadataPb)
  def subgraphSamplingStrategyPbWrapper: SubgraphSamplingStrategyWrapper =
    SubgraphSamplingStrategyWrapper(
      subgraphSamplingStrategyPb = subgraphSamplerConfigPb.subgraphSamplingStrategy.get,
    )

}
