package common.types.pb_wrappers

import snapchat.research.gbml.gigl_resource_config.GiglResourceConfig
import snapchat.research.gbml.gigl_resource_config.SparkResourceConfig

case class ResourceConfigPbWrapper(val resourceConfigPb: GiglResourceConfig) {

  def subgraphSamplerConfig: SparkResourceConfig = resourceConfigPb.subgraphSamplerConfig.get
  def splitGeneratorConfig: SparkResourceConfig  = resourceConfigPb.splitGeneratorConfig.get

}
