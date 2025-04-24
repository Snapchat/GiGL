from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Generic, List, Optional, TypeVar

import torch

from gigl.src.common.graph_builder.gbml_graph_protocol import GbmlGraphDataProtocol
from gigl.src.common.types.graph_data import Edge, EdgeType, Node, NodeId, NodeType

TGraph = TypeVar("TGraph", bound=GbmlGraphDataProtocol)


class GraphBuilder(Generic[TGraph]):
    def __remap_node(self, node: Node) -> Node:
        if node not in self.global_node_to_subgraph_node_map:
            subgraph_node = Node(
                type=node.type, id=NodeId(self.subgraph_node_id_counter[node.type])
            )
            self.global_node_to_subgraph_node_map[node] = subgraph_node
            self.subgraph_node_id_counter[node.type] += 1

        return self.global_node_to_subgraph_node_map[node]

    def __assert_remapping_exists(self, node: Node):
        if node not in self.global_node_to_subgraph_node_map:
            raise TypeError(
                f"Tried to fetch a node {node} which we have no information on"
            )

    def reset(
        self,
    ) -> None:
        """
        Unregisters / resets all registered nodes and edges.
        "Reinitialized the class to a clean state"
        """
        self.subgraph_node_id_counter: Dict[NodeType, int] = defaultdict(int)
        self.global_node_to_subgraph_node_map: Dict[Node, Node] = {}
        self.ordered_edges: Dict[EdgeType, List[Edge]] = defaultdict(list)
        self.ordered_nodes: Dict[NodeType, List[Node]] = defaultdict(list)
        self.subgraph_node_features_dict: Dict[Node, torch.Tensor] = {}
        self.subgraph_edge_feature_dict: Dict[Edge, Optional[torch.Tensor]] = {}

        self.should_register_node_features: Optional[bool] = None
        self.should_register_edge_features: Optional[bool] = None

    def add_graph_data(self, graph_data: TGraph) -> GraphBuilder:
        """Register pre-built graph data to be built into the new Tgraph data object

        Args:
            graph_data (TGraph): pre-built TGraph data

        Returns:
            GraphBuilder: returns self
        """
        for (
            global_node,
            node_features,
        ) in graph_data.get_global_node_features_dict().items():
            if global_node in self.global_node_to_subgraph_node_map:
                # If original node already exists then lets just sanity check its the same node
                # i.e. features match and skip insertion
                current_subgraph_node = self.__remap_node(global_node)

                def __generate_assertion_fail_msg():
                    return f"""
                    Trying to add global node: {global_node} from new graph data, which was
                    found as duplicate of existing node, but with different features.
                    Existing node has: {self.subgraph_node_features_dict[current_subgraph_node]}
                    but new node has: {node_features}
                    Unsupported operation!
                    """

                if node_features is not None:
                    assert torch.allclose(
                        node_features,
                        self.subgraph_node_features_dict[current_subgraph_node],
                    ), __generate_assertion_fail_msg()
                else:  # ensure that both existing features and features to be added == None
                    assert (
                        node_features
                        == self.subgraph_node_features_dict[current_subgraph_node]
                    ), __generate_assertion_fail_msg()

            else:
                self.add_node(
                    node=global_node,
                    feature_values=node_features,
                )

        # Ensure we register all EdgeTypes from the graph.  Without this registration,
        # the builder will miss EdgeTypes from samples if those EdgeTypes have no edges.
        # This causes problems when calling `build` as the resulting TGraph would result
        # in potentially no EdgeTypes and no edge index for convolution.
        self.register_edge_types(edge_types=graph_data.edge_types_to_be_registered)
        for edge, edge_features in graph_data.get_global_edge_features_dict().items():
            self.add_edge(edge=edge, feature_values=edge_features, skip_if_exists=True)
        return self

    def add_edge(
        self,
        edge: Edge,
        feature_values: Optional[torch.Tensor] = None,
        skip_if_exists: bool = True,
    ) -> GraphBuilder:
        """Registers a given edge to the TGraph data object that will be built.
        Both nodes for the edge must be registered before registering the edge

        Args:
            edge (Edge)
            feature_values (Optional[torch.Tensor]): The feature for the edge

        Returns:
            GraphBuilder: returns self
        """
        self.__assert_remapping_exists(edge.src_node)
        self.__assert_remapping_exists(edge.dst_node)

        if self.should_register_edge_features is None:
            if feature_values is None:
                self.should_register_edge_features = False
            else:
                self.should_register_edge_features = True

        if self.should_register_edge_features and feature_values is None:
            raise TypeError(
                f"Previously registered an edge feature, but now found none being registered for edge: {edge}"
            )
        if not self.should_register_edge_features and feature_values is not None:
            raise TypeError(
                f"Previously didnt register edge feature, but now trying to register one for edge: {edge}"
            )

        subgraph_src_node = self.__remap_node(edge.src_node)
        subgraph_dst_node = self.__remap_node(edge.dst_node)

        subgraph_edge: Edge = Edge(
            src_node_id=subgraph_src_node.id,
            dst_node_id=subgraph_dst_node.id,
            edge_type=edge.edge_type,
        )
        if skip_if_exists and subgraph_edge in self.subgraph_edge_feature_dict:
            return self  # Edge already exists, so we skip adding it

        self.ordered_edges[subgraph_edge.edge_type].append(subgraph_edge)
        self.subgraph_edge_feature_dict[subgraph_edge] = feature_values

        return self

    def register_edge_types(self, edge_types: List[EdgeType]) -> GraphBuilder:
        """Registers edge types

        Args:
            edge_types (List[EdgeType])

        Returns:
            GraphBuilder: returns self
        """
        for edge_type in edge_types:
            if edge_type not in self.ordered_edges:
                self.ordered_edges[edge_type] = []
        return self

    def add_node(
        self, node: Node, feature_values: Optional[torch.Tensor] = None
    ) -> GraphBuilder:
        """Registers the given node to the TGraph data object that will be built.

        Args:
            node (Node): [description]
            feature_values (Optional[torch.Tensor]): [The feature for the node

        Returns:
            GraphBuilder: returns self
        """
        subgraph_node = self.__remap_node(node)

        if self.should_register_node_features is None:
            if feature_values is None:
                self.should_register_node_features = False
            else:
                self.should_register_node_features = True

        if self.should_register_node_features and feature_values is None:
            raise TypeError(
                f"Previously registered a node feature, but now found none being registered for node: {node}"
            )
        if not self.should_register_node_features and feature_values is not None:
            raise TypeError(
                f"Previously didnt register a node feature, but now trying to register one for node: {node}"
            )
        if feature_values is not None:
            feature_values = feature_values
            self.subgraph_node_features_dict[subgraph_node] = feature_values
        return self

    @abstractmethod
    def build(self) -> TGraph:
        """Builds the actual TGraph data object and returns it with all of the registered
        nodes and edges. Will also reset i.e. unregister all existed nodes and edges such that
        GraphBuilder can be used again to build a new graph


        Returns:
            TGraph: The data object with all of the nodes/edges
        """
        raise NotImplementedError
