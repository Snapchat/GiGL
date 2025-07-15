from __future__ import annotations

from typing import Dict, List, Protocol, Set

import torch

from gigl.common.collections.frozen_dict import FrozenDict
from gigl.src.common.types.graph_data import Edge, EdgeType, Node


class GbmlGraphDataProtocol(Protocol):
    @property
    def edge_types_to_be_registered(
        self,
    ) -> List[EdgeType]:
        """Maintains a list of EdgeTypes associated with this graph data.

        Used in conjunction with GraphBuilder, to preserve EdgeTypes when combining
        multiple GbmlGraphDataProtocol objects together.

        Returns:
            List[EdgeType]

        """
        ...

    @property
    def global_node_to_subgraph_node_mapping(
        self,
    ) -> FrozenDict[Node, Node]:
        """Maintains Mapping from original Node to Mapped Node that is used in the underlying graph data format

        During creation of GBML Data representations using graph libraries such as DGL and Pytorch geometric,
        there may be occasions where nodes will need to be remapped to contiguous node ids 0, 1, 2 .... either
        as a requirement from the graph library or to maintain simpler logic for formulating and working with
        these graphs data formats.

        Returns:
            Dict[Node, Node]
        """
        ...

    @global_node_to_subgraph_node_mapping.setter
    def global_node_to_subgraph_node_mapping(
        self, global_node_to_subgraph_node_mapping: Dict[Node, Node]
    ) -> None:
        """See global_node_to_subgraph_node_mapping

        Args:
            global_node_to_subgraph_node_mapping (Dict[Node, Node])
        """
        ...

    @property
    def subgraph_node_to_global_node_mapping(self) -> FrozenDict[Node, Node]:
        """Inverse mapping of global_node_to_subgraph_node_mapping

        Returns:
            FrozenDict[Node, Node]:
        """
        ...

    def get_global_node_features_dict(self) -> FrozenDict[Node, torch.Tensor]:
        """Computes and fetches a dictionary mapping the global node to its
        relevant node features

        Returns:
            FrozenDict[Node, torch.Tensor]
        """
        ...

    def get_global_edge_features_dict(self) -> FrozenDict[Edge, torch.Tensor]:
        """Computes and fetches a dictionary mapping the global edge to its
        relevant edge features

        Returns:
            FrozenDict[Edge, torch.Tensor]
        """
        ...

    @staticmethod
    def are_same_graph(a: GbmlGraphDataProtocol, b: GbmlGraphDataProtocol) -> bool:
        """
        Args:
            a (GbmlGraphDataProtocol)
            b (GbmlGraphDataProtocol)

        Returns:
            bool: Returns True if both a and b objects that implement GbmlGraphDataProtocol
            represent the same graph in the global space. i.e. both have same nodes + related features,
            and edges + related features. i.e. a form of loose equality.

            For example for a: PygGraphData and b: PygGraphData, may both have same 3 nodes and 3 edges with
            the same features. But, because they are built in a specific way i.e order of edges and nodes,
            they may not be strictly equal: a != b. But really, the two represent the same "graph" in
            different ways. This function fills that gap.
        """
        a_global_node_features = a.get_global_node_features_dict()
        b_global_node_features = b.get_global_node_features_dict()
        if len(a_global_node_features) != len(b_global_node_features):
            return False

        a_global_edge_features = a.get_global_edge_features_dict()
        b_global_edge_features = b.get_global_edge_features_dict()
        if len(a_global_edge_features) != len(b_global_edge_features):
            return False

        for global_node, a_node_features in a_global_node_features.items():
            if global_node not in b_global_node_features:
                return False
            b_node_features = b_global_node_features[global_node]
            if type(a_node_features) != type(
                b_node_features
            ):  # Implictly also checks None == None
                return False

            if isinstance(a_node_features, torch.Tensor):
                if not torch.equal(a_node_features, b_node_features):
                    return False

        for global_edge, a_edge_features in a_global_edge_features.items():
            if global_edge not in b_global_edge_features:
                return False
            b_edge_features = b_global_edge_features[global_edge]
            if type(a_edge_features) != type(
                b_edge_features
            ):  # Implictly also checks None == None
                return False

            if isinstance(a_edge_features, torch.Tensor):
                if not torch.equal(a_edge_features, b_edge_features):
                    return False

        return True

    @staticmethod
    def are_disjoint(a: GbmlGraphDataProtocol, b: GbmlGraphDataProtocol) -> bool:
        """
        Returns True if the two GbmlGraphDataProtocol objects do not share any edges.
        :param a:
        :param b:
        :return:
        """
        a_edges = set([edge for edge in a.get_global_edge_features_dict().keys()])
        b_edges = set([edge for edge in b.get_global_edge_features_dict().keys()])

        smaller_edge_set: Set[Edge]
        larger_edge_set: Set[Edge]

        if len(a_edges) < len(b_edges):
            smaller_edge_set = a_edges
            larger_edge_set = b_edges
        else:
            smaller_edge_set = b_edges
            larger_edge_set = a_edges

        for edge in smaller_edge_set:
            if edge in larger_edge_set:
                return False

        return True
