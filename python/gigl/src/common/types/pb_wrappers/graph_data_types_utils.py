from typing import Tuple

from gigl.src.common.types.pb_wrappers.graph_data_types import (
    EdgePbWrapper,
    NodePbWrapper,
)
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from snapchat.research.gbml import graph_schema_pb2


def get_dehydrated_node_pb_wrappers_from_edge_wrapper(
    edge_pb_wrapper: EdgePbWrapper, graph_metadata_wrapper: GraphMetadataPbWrapper
) -> Tuple[NodePbWrapper, NodePbWrapper]:
    """
    Using graph metadata, returns the source and destination NodePb instances
    corresponding to an EdgePb.  This is used for data splitting.

    :param edge_pb_wrapper:
    :param graph_metadata_wrapper:
    :return:
    """

    (
        src_condensed_node_type,
        dst_condensed_node_type,
    ) = graph_metadata_wrapper.condensed_edge_type_to_condensed_node_types[
        edge_pb_wrapper.condensed_edge_type
    ]
    src_node_pb_wrapper = NodePbWrapper(
        pb=graph_schema_pb2.Node(
            node_id=edge_pb_wrapper.src_node_id,
            condensed_node_type=int(src_condensed_node_type),
        )
    )
    dst_node_pb_wrapper = NodePbWrapper(
        pb=graph_schema_pb2.Node(
            node_id=edge_pb_wrapper.dst_node_id,
            condensed_node_type=int(dst_condensed_node_type),
        )
    )
    return src_node_pb_wrapper, dst_node_pb_wrapper
