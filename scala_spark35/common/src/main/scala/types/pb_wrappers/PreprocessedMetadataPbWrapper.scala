package common.types.pb_wrappers

import common.types.GraphTypes.CondensedEdgeType
import common.types.GraphTypes.CondensedNodeType
import snapchat.research.gbml.preprocessed_metadata.PreprocessedMetadata

case class PreprocessedMetadataPbWrapper(val preprocessedMetadataPb: PreprocessedMetadata) {
  def getPreprocessedMetadataForCondensedNodeType(
    condensedNodeType: CondensedNodeType,
  ): PreprocessedMetadata.NodeMetadataOutput =
    preprocessedMetadataPb.condensedNodeTypeToPreprocessedMetadata.get(condensedNodeType).get
  def getPreprocessedMetadataForCondensedEdgeType(
    condensedEdgeType: CondensedEdgeType,
  ): PreprocessedMetadata.EdgeMetadataOutput =
    preprocessedMetadataPb.condensedEdgeTypeToPreprocessedMetadata.get(condensedEdgeType).get
  def isPosUserDefinedForCondensedEdgeType(condensedEdgeType: CondensedEdgeType): Boolean =
    preprocessedMetadataPb.condensedEdgeTypeToPreprocessedMetadata
      .get(condensedEdgeType)
      .get
      .positiveEdgeInfo
      .isEmpty
  def isNegUserDefinedForCondensedEdgeType(condensedEdgeType: CondensedEdgeType): Boolean =
    preprocessedMetadataPb.condensedEdgeTypeToPreprocessedMetadata
      .get(condensedEdgeType)
      .get
      .negativeEdgeInfo
      .isEmpty

  def hasEdgeFeatures(condensedEdgeType: CondensedEdgeType): Boolean = {
    val edgeMetadataOutputPb: PreprocessedMetadata.EdgeMetadataOutput =
      getPreprocessedMetadataForCondensedEdgeType(condensedEdgeType = condensedEdgeType)
    edgeMetadataOutputPb.mainEdgeInfo.get.featureKeys.nonEmpty
  }
}
