package common.types.pb_wrappers

import common.utils.ProtoLoader
import snapchat.research.gbml.gigl_resource_config.GiglResourceConfig
import snapchat.research.gbml.gigl_resource_config.SharedResourceConfig
import snapchat.research.gbml.gigl_resource_config.SparkResourceConfig

case class ResourceConfigPbWrapper(val resourceConfigPb: GiglResourceConfig) {

  def subgraphSamplerConfig: SparkResourceConfig = resourceConfigPb.subgraphSamplerConfig.get
  def splitGeneratorConfig: SparkResourceConfig  = resourceConfigPb.splitGeneratorConfig.get

  def sharedResourceConfigPb: SharedResourceConfig = if (
    resourceConfigPb.sharedResource.isSharedResourceConfigUri
  ) {
    ProtoLoader.populateProtoFromYaml[SharedResourceConfig](
      resourceConfigPb.getSharedResourceConfigUri,
    )
  } else {
    resourceConfigPb.getSharedResourceConfig
  }

}
