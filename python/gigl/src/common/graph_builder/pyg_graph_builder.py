from __future__ import annotations

from typing import List

import torch

from gigl.common.collections.frozen_dict import FrozenDict
from gigl.common.logger import Logger
from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.graph_builder.pyg_graph_data import PygGraphData
from gigl.src.common.types.graph_data import Node, NodeId

logger = Logger()


class PygGraphBuilder(GraphBuilder[PygGraphData]):
    def __init__(self) -> None:
        self.reset()

    def build(self) -> PygGraphData:
        data = PygGraphData()
        # Register Node Features
        for node_type, num_nodes in self.subgraph_node_id_counter.items():
            logger.debug(f"Registering {num_nodes} nodes of type {node_type}")
            data[node_type].x = torch.stack(
                [
                    # This needs to default to 1 if no node features are provided
                    # This is a restriction of PyG, that is it expectes node features of atleast size 1
                    (
                        self.subgraph_node_features_dict[
                            Node(type=node_type, id=NodeId(node_id))
                        ]
                        if self.should_register_node_features
                        else torch.ones(1)
                    )
                    for node_id in range(num_nodes)
                ]
            )

        # Register Edge Features
        for edge_type, ordered_edges in self.ordered_edges.items():
            logger.debug(f"Registering {len(ordered_edges)} edges of type {edge_type}")
            src_node_list: List[int] = []
            dst_node_list: List[int] = []

            edge_features_list: List[torch.Tensor] = []
            for edge in ordered_edges:
                src_node_list.append(int(edge.src_node.id))
                dst_node_list.append(int(edge.dst_node.id))
                if self.should_register_edge_features:
                    edge_feature = self.subgraph_edge_feature_dict[edge]
                    assert edge_feature is not None
                    edge_features_list.append(edge_feature)
            if self.should_register_edge_features and len(edge_features_list) > 0:
                data[tuple(edge_type)].edge_attr = torch.stack(edge_features_list)
            if len(src_node_list) > 0 and len(dst_node_list) > 0:
                data[tuple(edge_type)].edge_index = torch.LongTensor(
                    [
                        src_node_list,
                        dst_node_list,
                    ],
                )

        data.global_node_to_subgraph_node_mapping = FrozenDict(
            self.global_node_to_subgraph_node_map.copy()
        )
        self.reset()  # reset before returning

        return data
