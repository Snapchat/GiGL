from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    EdgeType,
    EdgeUsageType,
    NodeType,
)
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from snapchat.research.gbml import graph_schema_pb2


@dataclass
class MockedDatasetInfo:
    @property
    def node_types(self) -> List[NodeType]:
        return list(self.node_feats.keys())

    @property
    def edge_types(self) -> List[EdgeType]:
        return list(self.edge_index.keys())

    @property
    def num_nodes(self) -> Dict[NodeType, int]:
        return {
            node_type: node_feat.shape[0]
            for node_type, node_feat in self.node_feats.items()
        }

    def get_num_edges(self, edge_type: EdgeType, edge_usage_type: EdgeUsageType) -> int:
        num_edges = 0
        if edge_usage_type == EdgeUsageType.MAIN:
            main_edge_size_dict = {
                edge_type: edge_index.shape[1]
                for edge_type, edge_index in self.edge_index.items()
            }
            num_edges = main_edge_size_dict[edge_type]
        elif (
            self.user_defined_edge_index is not None
            and edge_type in self.user_defined_edge_index
        ):
            # We ignore the edge_type as currently UDL mocking does not have edge type
            if (
                edge_usage_type == EdgeUsageType.POSITIVE
                and EdgeUsageType.POSITIVE in self.user_defined_edge_index[edge_type]
            ):
                num_edges = self.user_defined_edge_index[edge_type][
                    EdgeUsageType.POSITIVE
                ].shape[1]
            elif (
                edge_usage_type == EdgeUsageType.NEGATIVE
                and EdgeUsageType.NEGATIVE in self.user_defined_edge_index[edge_type]
            ):
                num_edges = self.user_defined_edge_index[edge_type][
                    EdgeUsageType.NEGATIVE
                ].shape[1]
        return num_edges

    @property
    def num_node_features(self) -> Dict[NodeType, int]:
        return {
            node_type: feats.shape[1] for node_type, feats in self.node_feats.items()
        }

    @property
    def num_node_distinct_labels(self) -> Dict[NodeType, int]:
        if not self.node_labels:
            return {}

        return {
            node_type: labels.unique().numel()
            for node_type, labels in self.node_labels.items()
        }

    @property
    def num_edge_features(self) -> Dict[EdgeType, int]:
        if self.edge_feats:
            return {
                edge_type: feats.shape[1]
                for edge_type, feats in self.edge_feats.items()
            }
        else:
            return {edge_type: 0 for edge_type in self.edge_types}

    @property
    def num_user_def_edge_features(self) -> Dict[EdgeType, Dict[EdgeUsageType, int]]:
        num_user_def_edge_feats = {}
        if self.user_defined_edge_feats:
            for edge_type, udl_edge_feats in self.user_defined_edge_feats.items():
                num_user_def_edge_feats[edge_type] = {
                    edge_usage_type: feats.shape[1]
                    for edge_usage_type, feats in udl_edge_feats.items()
                }
        else:
            for edge_type in self.edge_types:
                num_user_def_edge_feats[edge_type] = {
                    edge_usage_type: 0
                    for edge_usage_type in [
                        EdgeUsageType.POSITIVE,
                        EdgeUsageType.NEGATIVE,
                    ]
                }
        return num_user_def_edge_feats

    @property
    def graph_metadata_pb_wrapper(self) -> GraphMetadataPbWrapper:
        graph_metadata_pb = graph_schema_pb2.GraphMetadata(
            node_types=[str(node_type) for node_type in self.node_types],
            edge_types=[
                GbmlProtosTranslator.EdgeTypePb_from_edge_type(edge_type=edge_type)
                for edge_type in self.edge_types
            ],
            condensed_node_type_map={
                CondensedNodeType(i): node_type
                for i, node_type in enumerate(self.node_types)
            },
            condensed_edge_type_map={
                CondensedEdgeType(i): GbmlProtosTranslator.EdgeTypePb_from_edge_type(
                    edge_type=edge_type
                )
                for i, edge_type in enumerate(self.edge_types)
            },
        )
        return GraphMetadataPbWrapper(graph_metadata_pb=graph_metadata_pb)

    @property
    def default_node_type(self) -> NodeType:
        return self.node_types[0]

    @property
    def default_edge_type(self) -> EdgeType:
        return self.edge_types[0]

    name: str
    task_metadata_type: TaskMetadataType
    edge_index: Dict[EdgeType, torch.Tensor]
    node_feats: Dict[NodeType, torch.Tensor]
    edge_feats: Optional[Dict[EdgeType, torch.Tensor]] = None
    node_labels: Optional[Dict[NodeType, torch.Tensor]] = None
    sample_node_type: Optional[NodeType] = None
    # TODO (tzhao-sc): currently only supporting 1 supervision edge type, we would need
    #      to extend this to support multiple supervision edge types for HGS stage 2
    sample_edge_type: Optional[EdgeType] = None
    edge_src_column_name: str = "src"
    edge_dst_column_name: str = "dst"
    node_id_column_name: str = "node_id"
    node_label_column_name: str = "node_label"
    user_defined_edge_index: Optional[
        Dict[EdgeType, Dict[EdgeUsageType, torch.Tensor]]
    ] = None
    user_defined_edge_feats: Optional[
        Dict[EdgeType, Dict[EdgeUsageType, torch.Tensor]]
    ] = None
    version: Optional[str] = None
