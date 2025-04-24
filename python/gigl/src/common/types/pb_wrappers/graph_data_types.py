from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Union, cast

from gigl.common.utils.func_tools import lru_cache
from gigl.src.common.types.graph_data import CondensedEdgeType, CondensedNodeType
from snapchat.research.gbml import graph_schema_pb2

_HASH_CACHE_KEY = "_hash"


"""
We cache hashes in the below wrapper classes for speedups.
This provides speed benefits in many instances, by avoiding work duplication.
object.__setattr__ is required to cache properties which are not explicitly declared as dataclass fields.

We avoid the explicit field declaration mainly to avoid Apache Beam coder behavior for dataclasses, which
is to (undesirably) encode all declared fields:
https://github.com/apache/beam/blob/release-2.41.0/sdks/python/apache_beam/coders/coder_impl.py#L473
This can cause problems, e.g. if we have an instance A with _hash cached, and an instance B without, the
two instances appear unequal to Beam.
"""


@dataclass(frozen=True)
class EdgePbWrapper:
    pb: graph_schema_pb2.Edge

    @property
    def unique_id(self) -> bytes:
        st = f"{self.src_node_id}-{self.condensed_edge_type}-{self.dst_node_id}"
        return st.encode(encoding="utf-8")

    @property
    def condensed_edge_type(self) -> CondensedEdgeType:
        return CondensedEdgeType(self.pb.condensed_edge_type)

    @property
    def src_node_id(self) -> int:
        return self.pb.src_node_id

    @property
    def dst_node_id(self) -> int:
        return self.pb.dst_node_id

    @property
    def feature_values(self) -> Sequence[float]:
        return self.pb.feature_values

    def flip_edge(self) -> EdgePbWrapper:
        flipped_edge_pb = graph_schema_pb2.Edge(
            src_node_id=self.pb.dst_node_id,
            dst_node_id=self.pb.src_node_id,
            condensed_edge_type=self.pb.condensed_edge_type,
            feature_values=self.pb.feature_values,
        )
        return EdgePbWrapper(pb=flipped_edge_pb)

    def hydrate(self, feature_values: Sequence[float]) -> EdgePbWrapper:
        hydrated_pb = graph_schema_pb2.Edge()
        hydrated_pb.MergeFrom(self.pb)
        del hydrated_pb.feature_values[:]
        hydrated_pb.feature_values.extend(feature_values)
        return EdgePbWrapper(hydrated_pb)

    def dehydrate(self) -> EdgePbWrapper:
        if not self.feature_values:
            return self
        dehydrated_pb = graph_schema_pb2.Edge(
            src_node_id=self.src_node_id,
            dst_node_id=self.dst_node_id,
            condensed_edge_type=self.condensed_edge_type,
        )
        return EdgePbWrapper(pb=dehydrated_pb)

    def __hash__(self) -> int:
        cached_hash = getattr(self, _HASH_CACHE_KEY, None)
        if cached_hash:
            return cast(int, cached_hash)
        else:
            h = hash(
                (
                    self.pb.src_node_id,
                    self.pb.dst_node_id,
                    self.pb.condensed_edge_type,
                )
            )
            object.__setattr__(self, _HASH_CACHE_KEY, h)
            return cast(int, h)

    def __getstate__(self):
        state = self.__dict__.copy()
        if _HASH_CACHE_KEY in state:
            del state[_HASH_CACHE_KEY]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __eq__(self, other: Union[object, EdgePbWrapper]) -> bool:
        if not isinstance(other, EdgePbWrapper):
            return False
        elif hash(self) == hash(other):
            return True
        else:
            return False

    def __repr__(self):
        return f"{self.pb}"


@dataclass(frozen=True)
class NodePbWrapper:
    pb: graph_schema_pb2.Node

    @property
    def unique_id(self) -> bytes:
        st = f"{self.node_id}-{self.condensed_node_type}"
        return st.encode(encoding="utf-8")

    @property
    def condensed_node_type(self) -> CondensedNodeType:
        return CondensedNodeType(self.pb.condensed_node_type)

    @property
    def node_id(self) -> int:
        return self.pb.node_id

    @property
    def feature_values(self) -> Sequence[float]:
        return self.pb.feature_values

    def hydrate(self, feature_values: Sequence[float]) -> NodePbWrapper:
        hydrated_pb = graph_schema_pb2.Node()
        hydrated_pb.MergeFrom(self.pb)
        del hydrated_pb.feature_values[:]
        hydrated_pb.feature_values.extend(feature_values)
        return NodePbWrapper(hydrated_pb)

    def dehydrate(self) -> NodePbWrapper:
        if not self.feature_values:
            return self
        dehydrated_pb = graph_schema_pb2.Node(
            node_id=self.node_id, condensed_node_type=self.condensed_node_type
        )
        return NodePbWrapper(pb=dehydrated_pb)

    def __hash__(self) -> int:
        cached_hash = getattr(self, _HASH_CACHE_KEY, None)
        if cached_hash:
            return cast(int, cached_hash)
        else:
            h = hash(
                (
                    self.pb.node_id,
                    self.pb.condensed_node_type,
                    tuple(self.pb.feature_values),
                )
            )
            object.__setattr__(self, _HASH_CACHE_KEY, h)
            return cast(int, h)

    def __getstate__(self):
        state = self.__dict__.copy()
        if _HASH_CACHE_KEY in state:
            del state[_HASH_CACHE_KEY]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __eq__(self, other: Union[object, NodePbWrapper]) -> bool:
        if not isinstance(other, NodePbWrapper):
            return False
        elif hash(self) == hash(other):
            return True
        else:
            return False

    def __repr__(self):
        return f"{self.pb}"


@dataclass(frozen=True)
class GraphPbWrapper:
    pb: graph_schema_pb2.Graph

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def nodes_wrapper(self) -> List[NodePbWrapper]:
        # TODO: rename to nodes_pb_wrapper for clarity
        return [NodePbWrapper(pb=node_pb2) for node_pb2 in self.pb.nodes]

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def nodes_pb(self) -> List[graph_schema_pb2.Node]:
        return list(self.pb.nodes)

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def edges_wrapper(self) -> List[EdgePbWrapper]:
        # TODO: rename to edges_pb_wrapper for clarity
        return [EdgePbWrapper(pb=edge_pb2) for edge_pb2 in self.pb.edges]

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def edges_pb(self) -> List[graph_schema_pb2.Edge]:
        return list(self.pb.edges)

    @classmethod
    def merge_subgraphs(cls, subgraphs: List[GraphPbWrapper]) -> GraphPbWrapper:
        # proto object types are un-hashable using set operation,
        # and mergeFrom function does not deduplicate,
        # so unfold and rebuild below
        node_set: Set[NodePbWrapper] = set([])
        edge_set: Set[EdgePbWrapper] = set([])

        for subgraph in subgraphs:
            node_set.update(subgraph.nodes_wrapper)
            edge_set.update(subgraph.edges_wrapper)

        merged_graph_pb = graph_schema_pb2.Graph(
            nodes=[node.pb for node in node_set],
            edges=[edge.pb for edge in edge_set],
        )

        # get all information out, find unique,then build new proto
        return cls(pb=merged_graph_pb)

    @classmethod
    def build_dry_subgraph(
        cls,
        node_wrappers: List[NodePbWrapper],
        edge_wrappers: Optional[List[EdgePbWrapper]] = None,
    ) -> GraphPbWrapper:
        if edge_wrappers:
            graph_pb = graph_schema_pb2.Graph(
                nodes=[node.pb for node in node_wrappers],
                edges=[edge.pb for edge in edge_wrappers],
            )
        else:
            graph_pb = graph_schema_pb2.Graph(nodes=[node.pb for node in node_wrappers])

        return cls(pb=graph_pb)

    @staticmethod
    def hydrate_subgraph_features(
        dry_graph_proto: GraphPbWrapper,
        node_features_dict: Optional[Dict[NodePbWrapper, Sequence[float]]] = None,
        edge_features_dict: Optional[Dict[EdgePbWrapper, Sequence[float]]] = None,
    ) -> GraphPbWrapper:
        """
        Hydrate subgraph function is currently used in graphflat,
        to hydrate the resulting k-hop subgraph of graphflat with the associated features,
        since graphflat operates on featureless subgraphs for speed and memory efficiency
        """

        # iterate through repeated field and update
        # if nodes in the proto doesn't have features throw error

        if node_features_dict:
            nodes_pb: List[graph_schema_pb2.Node] = []
            for wrapped_node in dry_graph_proto.nodes_wrapper:
                try:
                    node_feature_values = node_features_dict[wrapped_node]
                except KeyError:
                    raise Exception(f"Features not propagated for node {wrapped_node}")
                nodes_pb.append(
                    wrapped_node.hydrate(node_feature_values).pb
                    if node_feature_values
                    else wrapped_node.pb
                )
        else:
            nodes_pb = dry_graph_proto.nodes_pb

        if edge_features_dict:
            edges_pb: List[graph_schema_pb2.Edge] = []
            for wrapped_edge in dry_graph_proto.edges_wrapper:
                try:
                    edge_feature_values = edge_features_dict[wrapped_edge]
                except KeyError:
                    raise KeyError(f"Features not propagated for edge {wrapped_edge}")
                edges_pb.append(
                    wrapped_edge.hydrate(edge_feature_values).pb
                    if edge_feature_values
                    else wrapped_edge.pb
                )
        else:
            edges_pb = dry_graph_proto.edges_pb

        return GraphPbWrapper(graph_schema_pb2.Graph(nodes=nodes_pb, edges=edges_pb))

    def __hash__(self) -> int:
        """
        We sort the nodes, edges for __eq__ purposes according to below proto behavior where
        Graph protos aren't considered equal when order in repeated fields nodes is switched

        > from snapchat.research.gbml.graph_schema_pb2 import Node, Edge, Graph
        > n1 = Node(node_id=1)
        > n2 = Node(node_id=2)
        > e12 = Edge(src_node_id=1, dst_node_id=2)
        > e21 = Edge(dst_node_id=2, src_node_id=1)
        > g1 = Graph(nodes=[n1, n2], edges=[e12, e21])
        > g2 = Graph(nodes=[n1, n2], edges=[e21, e12])
        > g3 = Graph(nodes=[n2, n1], edges=[e21, e12])
        > g1 == g2
        True
        > g1 == g3
        False
        > g2 == g3
        False

        """

        cached_hash = getattr(self, _HASH_CACHE_KEY, None)
        if cached_hash:
            return cast(int, cached_hash)
        else:
            sorted_graph_pb_repr = tuple(
                sorted(
                    [(node.node_id, node.condensed_node_type) for node in self.pb.nodes]
                )
            ) + tuple(
                sorted(
                    [
                        (
                            edge.src_node_id,
                            edge.dst_node_id,
                            edge.condensed_edge_type,
                        )
                        for edge in self.pb.edges
                    ]
                )
            )
            h = hash(sorted_graph_pb_repr)
            object.__setattr__(self, _HASH_CACHE_KEY, h)
            return cast(int, h)

    def __getstate__(self):
        state = self.__dict__.copy()
        if _HASH_CACHE_KEY in state:
            del state[_HASH_CACHE_KEY]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __eq__(self, other: Union[object, GraphPbWrapper]) -> bool:
        if not isinstance(other, GraphPbWrapper):
            return False
        elif hash(self) == hash(other):
            return True
        else:
            return False

    def __repr__(self):
        return f"{self.pb}"
