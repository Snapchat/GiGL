from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.data.storage import EdgeStorage

from gigl.common.collections.frozen_dict import FrozenDict
from gigl.src.common.graph_builder.gbml_graph_protocol import GbmlGraphDataProtocol
from gigl.src.common.types.graph_data import Edge, EdgeType, Node, NodeId, NodeType


class PygGraphData(HeteroData, GbmlGraphDataProtocol):
    """
    Extends pytorch geometric graph data objects to provide support for more functionality.
    i.e. providing functionality to do equality checks
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            **kwargs,
        )
        self.__global_node_to_subgraph_node_mapping: FrozenDict[
            Node, Node
        ] = FrozenDict({})
        self.__subgraph_node_to_global_node_mapping: Optional[
            FrozenDict[Node, Node]
        ] = None

    @property
    def edge_types_to_be_registered(
        self,
    ) -> List[EdgeType]:
        edge_types_to_be_registered = []
        if hasattr(self, "_edge_store_dict"):
            edge_types_to_be_registered = [
                EdgeType(
                    src_node_type=src_node_type,
                    relation=relation,
                    dst_node_type=dst_node_type,
                )
                for src_node_type, relation, dst_node_type in self._edge_store_dict.keys()
            ]
        return edge_types_to_be_registered

    @property
    def global_node_to_subgraph_node_mapping(
        self,
    ) -> FrozenDict[Node, Node]:
        return self.__global_node_to_subgraph_node_mapping

    @global_node_to_subgraph_node_mapping.setter
    def global_node_to_subgraph_node_mapping(
        self, global_node_to_subgraph_node_mapping: FrozenDict[Node, Node]
    ) -> None:
        self.__global_node_to_subgraph_node_mapping = FrozenDict(
            global_node_to_subgraph_node_mapping
        )
        self.__subgraph_node_to_global_node_mapping = None

    @property
    def subgraph_node_to_global_node_mapping(self) -> FrozenDict[Node, Node]:
        if self.__subgraph_node_to_global_node_mapping is None:
            self.__subgraph_node_to_global_node_mapping = FrozenDict(
                {v: k for k, v in self.global_node_to_subgraph_node_mapping.items()}
            )
        return self.__subgraph_node_to_global_node_mapping

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PygGraphData):
            return False

        if (
            self.global_node_to_subgraph_node_mapping
            != other.global_node_to_subgraph_node_mapping
        ):
            return False

        if not (
            hasattr(self, "x_dict") == hasattr(other, "x_dict")
            and hasattr(self, "_edge_store_dict") == hasattr(other, "_edge_store_dict")
        ):
            return False

        if hasattr(self, "x_dict"):
            if len(self.x_dict) != len(other.x_dict):
                return False

            for self_x_key, self_x_val in self.x_dict.items():
                if self_x_key not in other.x_dict:
                    return False
                other_x_val = other.x_dict[self_x_key]
                if not torch.equal(self_x_val, other_x_val):
                    return False

        if hasattr(self, "_edge_store_dict"):
            if len(self._edge_store_dict) != len(other._edge_store_dict):
                return False

            self_edge_store: EdgeStorage
            for (
                self_edge_type,
                self_edge_store,
            ) in self._edge_store_dict.items():
                if self_edge_type not in other._edge_store_dict:
                    return False
                other_edge_store = other._edge_store_dict[self_edge_type]
                for (
                    key,
                    tensor,
                ) in self_edge_store.items():  # edge_attr, edge_index (keys)
                    if key not in other_edge_store:
                        return False
                    if not torch.equal(tensor, other_edge_store[key]):
                        return False

        return True

    def get_global_node_features_dict(self) -> FrozenDict[Node, torch.Tensor]:
        if not hasattr(self, "x_dict"):
            return FrozenDict({})

        global_node_to_features_map: Dict[Node, torch.Tensor] = {}
        for self_node_type, all_node_features_for_node_type in self.x_dict.items():
            for subgraph_node_id, node_features in enumerate(
                all_node_features_for_node_type
            ):
                subgraph_node = Node(
                    type=NodeType(self_node_type), id=NodeId(subgraph_node_id)
                )
                global_node = (
                    self.subgraph_node_to_global_node_mapping[subgraph_node]
                    if subgraph_node in self.subgraph_node_to_global_node_mapping
                    else subgraph_node
                )
                global_node_to_features_map[global_node] = node_features

        return FrozenDict(global_node_to_features_map)

    def get_global_edge_features_dict(self) -> FrozenDict[Edge, torch.Tensor]:
        global_edge_to_features_map: Dict[Edge, torch.Tensor] = {}

        is_graph_data_in_global_space: bool = (
            not self.subgraph_node_to_global_node_mapping
        )

        if hasattr(self, "_edge_store_dict"):
            # Below, example of edge_index =
            #        [[10, 20], [20, 30]]
            # meaning the following edges exist 10 --> 20, and 20 --> 30
            for (
                edge_type,
                edge_store,
            ) in self._edge_store_dict.items():
                edge_index = edge_store["edge_index"]
                edge_attr = edge_store.get("edge_attr", None)

                src_node_type, relation, dst_node_type = edge_type

                for edge_number, (
                    subgraph_src_node_id_tensor,
                    subgraph_dst_node_id_tensor,
                ) in enumerate(zip(edge_index[0], edge_index[1])):
                    subgraph_src_node_id = subgraph_src_node_id_tensor.item()
                    subgraph_dst_node_id = subgraph_dst_node_id_tensor.item()
                    subgraph_src_node = Node(
                        type=NodeType(src_node_type), id=NodeId(subgraph_src_node_id)
                    )
                    subgraph_dst_node = Node(
                        type=NodeType(dst_node_type), id=NodeId(subgraph_dst_node_id)
                    )
                    global_src_node = (
                        subgraph_src_node
                        if is_graph_data_in_global_space
                        else self.subgraph_node_to_global_node_mapping[
                            subgraph_src_node
                        ]
                    )
                    global_dst_node = (
                        subgraph_dst_node
                        if is_graph_data_in_global_space
                        else self.subgraph_node_to_global_node_mapping[
                            subgraph_dst_node
                        ]
                    )
                    edge = Edge(
                        src_node_id=global_src_node.id,
                        dst_node_id=global_dst_node.id,
                        edge_type=EdgeType(src_node_type, relation, dst_node_type),
                    )

                    edge_feature = (
                        edge_attr[edge_number] if edge_attr is not None else None
                    )
                    global_edge_to_features_map[edge] = edge_feature

        return FrozenDict(global_edge_to_features_map)

    def to_hetero_data(self) -> HeteroData:
        """
        Convert the PygGraphData object back to a PyG HeteroData object

        returns:
            HeteroData: The converted HeteroData object
        """
        hetero_data = HeteroData()
        hetero_data.update(data=self)
        return hetero_data

    @classmethod
    def from_hetero_data(cls, data: HeteroData) -> PygGraphData:
        pyg_graph_data = cls()

        if hasattr(data, "x_dict"):
            for x_key, x_val in data.x_dict.items():
                pyg_graph_data[x_key].x = x_val

        if hasattr(data, "_edge_store_dict"):
            for (
                edge_type,
                edge_store,
            ) in data._edge_store_dict.items():
                pyg_graph_data[edge_type].edge_index = edge_store.edge_index
                if hasattr(edge_store, "edge_attr"):
                    pyg_graph_data[edge_type].edge_attr = edge_store.edge_attr

        return pyg_graph_data

    def __repr__(self) -> str:
        return f"""PygGraphData(
        global_node_to_subgraph_node_mapping={self.global_node_to_subgraph_node_mapping}
        x_dict={self.x_dict if hasattr(self, "x_dict") else {}}
        _edge_store_dict={self._edge_store_dict if hasattr(self, "_edge_store_dict") else {}}
        )
        """

    def __setattr__(self, key: str, value: Any):
        """Need to override functionality cause HeteroData does some weird logic with
        its `__setattr__` function making @property.setter un-usable
        """
        if key in self.__class__.__dict__:
            return object.__setattr__(self, key, value)
        return super().__setattr__(key, value)
