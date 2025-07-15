from typing import List, Optional, Tuple

import torch

from gigl.common.logger import Logger
from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.graph_builder.gbml_graph_protocol import GbmlGraphDataProtocol
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    Edge,
    EdgeType,
    Node,
    NodeId,
    NodeType,
    Relation,
)
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.utils.data.feature_serialization import FeatureSerializationUtils
from snapchat.research.gbml import graph_schema_pb2

logger = Logger()


class GbmlProtosTranslator:
    @staticmethod
    def node_from_NodePb(
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        node_pb: graph_schema_pb2.Node,
    ) -> Tuple[Node, Optional[torch.Tensor]]:
        """
        Args:
            graph_metadata (GraphMetadataPbWrapper)
            node_pb (graph_schema_pb2.Node)

        Returns:
            Tuple[Node, torch.tensor]: Tuple of Node and related Node features
        """
        node = Node(
            type=graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                CondensedNodeType(node_pb.condensed_node_type)
            ],
            id=NodeId(node_pb.node_id),
        )
        feature_values = (
            torch.tensor(
                FeatureSerializationUtils.deserialize_node_features(
                    node_pb.feature_values
                )
            )
            if node_pb.feature_values
            else None
        )
        return (node, feature_values)

    @staticmethod
    def edge_from_EdgePb(
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        edge_pb: graph_schema_pb2.Edge,
    ) -> Tuple[Edge, Optional[torch.Tensor]]:
        edge_type: EdgeType = (
            graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map[
                CondensedEdgeType(edge_pb.condensed_edge_type)
            ]
        )
        src_node_id = NodeId(edge_pb.src_node_id)
        dst_node_id = NodeId(edge_pb.dst_node_id)
        edge = Edge(
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            edge_type=edge_type,
        )
        feature_values = (
            torch.tensor(
                FeatureSerializationUtils.deserialize_edge_features(
                    edge_pb.feature_values
                )
            )
            if edge_pb.feature_values
            else None
        )
        return (edge, feature_values)

    @staticmethod
    def edge_type_from_EdgeTypePb(edge_type_pb: graph_schema_pb2.EdgeType) -> EdgeType:
        return EdgeType(
            src_node_type=NodeType(edge_type_pb.src_node_type),
            relation=Relation(edge_type_pb.relation),
            dst_node_type=NodeType(edge_type_pb.dst_node_type),
        )

    @staticmethod
    def EdgeTypePb_from_edge_type(edge_type: EdgeType) -> graph_schema_pb2.EdgeType:
        return graph_schema_pb2.EdgeType(
            src_node_type=edge_type.src_node_type,
            relation=edge_type.relation,
            dst_node_type=edge_type.dst_node_type,
        )

    @staticmethod
    def graph_data_from_GraphPb(
        samples: List[graph_schema_pb2.Graph],
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        builder: GraphBuilder,
    ) -> GbmlGraphDataProtocol:
        for sample in samples:
            for node_pb in sample.nodes:
                node, node_features = GbmlProtosTranslator.node_from_NodePb(
                    graph_metadata_pb_wrapper=graph_metadata_pb_wrapper, node_pb=node_pb
                )
                builder.add_node(node=node, feature_values=node_features)

            for edge_pb in sample.edges:
                edge, edge_features = GbmlProtosTranslator.edge_from_EdgePb(
                    graph_metadata_pb_wrapper=graph_metadata_pb_wrapper, edge_pb=edge_pb
                )
                builder.add_edge(edge=edge, feature_values=edge_features)

        builder.register_edge_types(edge_types=graph_metadata_pb_wrapper.edge_types)
        graph_data: GbmlGraphDataProtocol = builder.build()
        return graph_data
