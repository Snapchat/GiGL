from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, NewType, Tuple

from gigl.common.utils.func_tools import lru_cache

# Unique identifier for the node for a specific NodeType
NodeId = NewType("NodeId", int)
NodeType = NewType("NodeType", str)
Relation = NewType("Relation", str)


class EdgeUsageType(str, Enum):
    MAIN = "main"
    POSITIVE = "positive"
    NEGATIVE = "negative"

    def __str__(self) -> str:
        return str(self.value)


class EdgeType(NamedTuple):
    src_node_type: NodeType
    relation: Relation
    dst_node_type: NodeType

    def __repr__(self):
        return f"{self.src_node_type}-{self.relation}-{self.dst_node_type}"

    def tuple_repr(self) -> Tuple[NodeType, Relation, NodeType]:
        return (self.src_node_type, self.relation, self.dst_node_type)


CondensedNodeType = NewType("CondensedNodeType", int)
CondensedEdgeType = NewType("CondensedEdgeType", int)


# TODO: (svij-sc): replace with NodePbWrapper
@dataclass(frozen=True)
class Node:
    type: NodeType  # note, this is not "condensed" node type
    id: NodeId

    # Implementing less than and greater than functionality so that list of
    # nodes can be sorted easier i.e. for use in generating hash for use in
    # FrozenDict
    def __lt__(self, other: Node):
        return f"{self.type}_{str(self.id)}" < f"{other.type}_{str(other.id)}"

    def __gt__(self, other: Node):
        return f"{self.type}_{str(self.id)}" > f"{other.type}_{str(other.id)}"


@dataclass(frozen=True)
class Edge:
    src_node_id: NodeId  # EdgeType below houses the node types
    dst_node_id: NodeId
    edge_type: EdgeType

    @classmethod
    def from_nodes(cls, src_node: Node, dst_node: Node, relation: Relation) -> Edge:
        edge = cls(
            src_node_id=src_node.id,
            dst_node_id=dst_node.id,
            edge_type=EdgeType(
                src_node_type=src_node.type,
                relation=relation,
                dst_node_type=dst_node.type,
            ),
        )
        return edge

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def src_node(self) -> Node:
        return Node(id=self.src_node_id, type=self.edge_type.src_node_type)

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def dst_node(self) -> Node:
        return Node(id=self.dst_node_id, type=self.edge_type.dst_node_type)
