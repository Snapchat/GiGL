from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)


def cache_mappings(gbml_config_pb_wrapper: GbmlConfigPbWrapper):
    """
    Initialize (call) frequently used mappings to cache them.
    """
    graph_metadata_pb_wrapper: GraphMetadataPbWrapper = (
        gbml_config_pb_wrapper.graph_metadata_pb_wrapper
    )
    preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper = (
        gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper
    )
    condensed_node_type_to_node_type_map = (
        graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map
    )
    condensed_edge_type_to_edge_type_map = (
        graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map
    )
    condensed_node_type_map = (
        graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map
    )
