from dataclasses import dataclass
from typing import Dict, Final, List

import torch

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation

MOCKED_NUM_PARTITIONS: Final[int] = 2

# Homogeneous case assumes user node type and user to user edge type, heterogeneous case additionally adds in item node type and user_to_item edge type

USER_NODE_TYPE: Final[NodeType] = NodeType("user")
ITEM_NODE_TYPE: Final[NodeType] = NodeType("item")

# u2u edge
USER_TO_USER_EDGE_TYPE: Final[EdgeType] = EdgeType(
    src_node_type=USER_NODE_TYPE,
    relation=Relation("user_to_item"),
    dst_node_type=USER_NODE_TYPE,
)

# u2i edge
USER_TO_ITEM_EDGE_TYPE: Final[EdgeType] = EdgeType(
    src_node_type=USER_NODE_TYPE,
    relation=Relation("user_to_item"),
    dst_node_type=ITEM_NODE_TYPE,
)

MOCKED_HETEROGENEOUS_NODE_TYPES: Final[List[NodeType]] = sorted(
    [USER_NODE_TYPE, ITEM_NODE_TYPE]
)
MOCKED_HETEROGENEOUS_EDGE_TYPES: Final[List[EdgeType]] = sorted(
    [USER_TO_USER_EDGE_TYPE, USER_TO_ITEM_EDGE_TYPE]
)

NODE_TYPE_TO_FEATURE_DIMENSION_MAP: Final[Dict[NodeType, int]] = {
    USER_NODE_TYPE: 2,
    ITEM_NODE_TYPE: 1,
}
EDGE_TYPE_TO_FEATURE_DIMENSION_MAP: Final[Dict[EdgeType, int]] = {
    USER_TO_USER_EDGE_TYPE: 2,
    USER_TO_ITEM_EDGE_TYPE: 1,
}


## Node IDs
# Each rank contains 4 user nodes and 2 item nodes. Each node id tensor is of shape [num_nodes_on_rank].

MOCKED_USER_NODES_IDS_ON_RANK_ZERO: Final[torch.Tensor] = torch.Tensor([0, 1, 2, 3]).to(
    torch.int64
)
MOCKED_USER_NODES_IDS_ON_RANK_ONE: Final[torch.Tensor] = torch.Tensor([4, 5, 6, 7]).to(
    torch.int64
)

MOCKED_ITEM_NODES_IDS_ON_RANK_ZERO: Final[torch.Tensor] = torch.Tensor([0, 1]).to(
    torch.int64
)
MOCKED_ITEM_NODES_IDS_ON_RANK_ONE: Final[torch.Tensor] = torch.Tensor([2, 3]).to(
    torch.int64
)

## Node Features
# Node features are set to be the corresponding node index divided by 10, repeated twice for user nodes and once for item nodes.
# Each node feature tensor is of shape [num_nodes_on_rank, node_feat_dim]

MOCKED_USER_NODE_FEATURES_ON_RANK_ZERO: Final[torch.Tensor] = torch.Tensor(
    [[0, 0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
)
MOCKED_USER_NODE_FEATURES_ON_RANK_ONE: Final[torch.Tensor] = torch.Tensor(
    [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6], [0.7, 0.7]]
)

MOCKED_ITEM_NODE_FEATURES_ON_RANK_ZERO: Final[torch.Tensor] = torch.Tensor([[0], [0.1]])
MOCKED_ITEM_NODE_FEATURES_ON_RANK_ONE: Final[torch.Tensor] = torch.Tensor(
    [[0.2], [0.3]]
)

## Edge Index

# Each rank has 4 u2u edges and 4 u2i edges. When we partition edges, there will be an equal number of edges assigned to each rank in the partition book.
# Each edge index tensor is of shape [2, num_edges_on_rank], where the 0th and 1st rows correspond to the source and destination nodes for an outgoing graph, respectively.

MOCKED_U2U_EDGE_INDEX_ON_RANK_ZERO: Final[torch.Tensor] = torch.Tensor(
    [[0, 1, 2, 3], [1, 2, 3, 0]]
).to(torch.int64)
MOCKED_U2U_EDGE_INDEX_ON_RANK_ONE: Final[torch.Tensor] = torch.Tensor(
    [[4, 5, 6, 7], [5, 6, 7, 4]]
).to(torch.int64)

MOCKED_U2I_EDGE_INDEX_ON_RANK_ZERO: Final[torch.Tensor] = torch.Tensor(
    [[0, 1, 2, 3], [0, 1, 2, 3]]
).to(torch.int64)
MOCKED_U2I_EDGE_INDEX_ON_RANK_ONE: Final[torch.Tensor] = torch.Tensor(
    [[4, 5, 6, 7], [0, 1, 2, 3]]
).to(torch.int64)

## Edge features
# Edge features for U2U edge are set to be the corresponding source node ids value divided by 10.
# U2I edge has no edge features.
# Each U2U edge feature tensor is of shape [num_edges_on_rank, edge_feat_dim].
MOCKED_U2U_EDGE_FEATURES_ON_RANK_ZERO: Final[torch.Tensor] = torch.Tensor(
    [[0, 0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
)
MOCKED_U2U_EDGE_FEATURES_ON_RANK_ONE: Final[torch.Tensor] = torch.Tensor(
    [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6], [0.7, 0.7]]
)

## Labeled Edges
# There are 2 positive labels and 2 negative labels per rank for the u2u edge. For u2i edge, we have 2 labels on rank 0 and 0 labels on rank 1.
# Each label tensor is of shape [2, num_labels_on_rank], where, for an outgoing graph, the 0th and 1st rows correspond to source and destination nodes, respectfully.

MOCKED_U2U_POS_EDGE_INDEX_ON_RANK_ZERO: Final[torch.Tensor] = torch.Tensor(
    [[0, 4], [1, 1]]
).to(torch.int64)
MOCKED_U2U_POS_EDGE_INDEX_ON_RANK_ONE: Final[torch.Tensor] = torch.Tensor(
    [[2, 6], [3, 3]]
).to(torch.int64)

MOCKED_U2U_NEG_EDGE_INDEX_ON_RANK_ZERO: Final[torch.Tensor] = torch.Tensor(
    [[0, 4], [3, 3]]
).to(torch.int64)
MOCKED_U2U_NEG_EDGE_INDEX_ON_RANK_ONE: Final[torch.Tensor] = torch.Tensor(
    [[2, 6], [0, 0]]
).to(torch.int64)

MOCKED_U2I_POS_EDGE_INDEX_ON_RANK_ZERO: Final[torch.Tensor] = torch.Tensor(
    [[0, 4], [0, 0]]
).to(torch.int64)
MOCKED_U2I_POS_EDGE_INDEX_ON_RANK_ONE: Final[torch.Tensor] = torch.Tensor(
    [[2, 6], [2, 2]]
).to(torch.int64)

MOCKED_U2I_NEG_EDGE_INDEX_ON_RANK_ZERO: Final[torch.Tensor] = torch.Tensor(
    [[0, 4], [1, 1]]
).to(torch.int64)
MOCKED_U2I_NEG_EDGE_INDEX_ON_RANK_ONE: Final[torch.Tensor] = torch.empty((2, 0)).to(
    torch.int64
)


@dataclass(frozen=True)
class TestGraphData:
    """
    This class exists as a convenience to hold inputs to the Partitioner class for smaller graphs for testing. This class should not be used for partitioning on
    large graphs, as this can lead to the reference count of tensors and resulting gc.collect() calls more complex. The homogeneous graph uses only node type
    `USER_NODE_TYPE` and edge type `USER_TO_USER_EDGE_TYPE`, while the heterogeneous graph additionally uses node type `ITEM_NODE_TYPE` and edge type `USER_TO_ITEM_EDGE_TYPE`.
    """

    # Node id tensor for mocked data
    node_ids: Dict[NodeType, torch.Tensor]

    # Edge index tensor for mocked data
    edge_index: Dict[EdgeType, torch.Tensor]

    # Node feature tensor for mocked data
    node_features: Dict[NodeType, torch.Tensor]

    # Edge feature tensor for mocked data
    edge_features: Dict[EdgeType, torch.Tensor]

    # Positive edge label tensor for mocked data
    positive_labels: Dict[EdgeType, torch.Tensor]

    # Input negative edge label tensor to partitioner for mocked data
    negative_labels: Dict[EdgeType, torch.Tensor]


RANK_TO_MOCKED_GRAPH: Final[Dict[int, TestGraphData]] = {
    0: TestGraphData(
        node_ids={
            USER_NODE_TYPE: MOCKED_USER_NODES_IDS_ON_RANK_ZERO,
            ITEM_NODE_TYPE: MOCKED_ITEM_NODES_IDS_ON_RANK_ZERO,
        },
        edge_index={
            USER_TO_USER_EDGE_TYPE: MOCKED_U2U_EDGE_INDEX_ON_RANK_ZERO,
            USER_TO_ITEM_EDGE_TYPE: MOCKED_U2I_EDGE_INDEX_ON_RANK_ZERO,
        },
        node_features={
            USER_NODE_TYPE: MOCKED_USER_NODE_FEATURES_ON_RANK_ZERO,
            ITEM_NODE_TYPE: MOCKED_ITEM_NODE_FEATURES_ON_RANK_ZERO,
        },
        edge_features={
            USER_TO_USER_EDGE_TYPE: MOCKED_U2U_EDGE_FEATURES_ON_RANK_ZERO,
        },
        positive_labels={
            USER_TO_USER_EDGE_TYPE: MOCKED_U2U_POS_EDGE_INDEX_ON_RANK_ZERO,
            USER_TO_ITEM_EDGE_TYPE: MOCKED_U2I_POS_EDGE_INDEX_ON_RANK_ZERO,
        },
        negative_labels={
            USER_TO_USER_EDGE_TYPE: MOCKED_U2U_NEG_EDGE_INDEX_ON_RANK_ZERO,
            USER_TO_ITEM_EDGE_TYPE: MOCKED_U2I_NEG_EDGE_INDEX_ON_RANK_ZERO,
        },
    ),
    1: TestGraphData(
        node_ids={
            USER_NODE_TYPE: MOCKED_USER_NODES_IDS_ON_RANK_ONE,
            ITEM_NODE_TYPE: MOCKED_ITEM_NODES_IDS_ON_RANK_ONE,
        },
        edge_index={
            USER_TO_USER_EDGE_TYPE: MOCKED_U2U_EDGE_INDEX_ON_RANK_ONE,
            USER_TO_ITEM_EDGE_TYPE: MOCKED_U2I_EDGE_INDEX_ON_RANK_ONE,
        },
        node_features={
            USER_NODE_TYPE: MOCKED_USER_NODE_FEATURES_ON_RANK_ONE,
            ITEM_NODE_TYPE: MOCKED_ITEM_NODE_FEATURES_ON_RANK_ONE,
        },
        edge_features={
            USER_TO_USER_EDGE_TYPE: MOCKED_U2U_EDGE_FEATURES_ON_RANK_ONE,
        },
        positive_labels={
            USER_TO_USER_EDGE_TYPE: MOCKED_U2U_POS_EDGE_INDEX_ON_RANK_ONE,
            USER_TO_ITEM_EDGE_TYPE: MOCKED_U2I_POS_EDGE_INDEX_ON_RANK_ONE,
        },
        negative_labels={
            USER_TO_USER_EDGE_TYPE: MOCKED_U2U_NEG_EDGE_INDEX_ON_RANK_ONE,
            USER_TO_ITEM_EDGE_TYPE: MOCKED_U2I_NEG_EDGE_INDEX_ON_RANK_ONE,
        },
    ),
}

RANK_TO_NODE_TYPE_TYPE_TO_NUM_NODES: Final[Dict[int, Dict[NodeType, int]]] = {
    rank: {
        USER_NODE_TYPE: test_graph_data.node_ids[USER_NODE_TYPE].size(0),
        ITEM_NODE_TYPE: test_graph_data.node_ids[ITEM_NODE_TYPE].size(0),
    }
    for rank, test_graph_data in RANK_TO_MOCKED_GRAPH.items()
}

MOCKED_UNIFIED_GRAPH: Final[TestGraphData] = TestGraphData(
    node_ids={
        USER_NODE_TYPE: torch.cat(
            (MOCKED_USER_NODES_IDS_ON_RANK_ZERO, MOCKED_USER_NODES_IDS_ON_RANK_ONE),
            dim=0,
        ),
        ITEM_NODE_TYPE: torch.cat(
            (MOCKED_ITEM_NODES_IDS_ON_RANK_ZERO, MOCKED_ITEM_NODES_IDS_ON_RANK_ONE),
            dim=0,
        ),
    },
    edge_index={
        USER_TO_USER_EDGE_TYPE: torch.cat(
            (MOCKED_U2U_EDGE_INDEX_ON_RANK_ZERO, MOCKED_U2U_EDGE_INDEX_ON_RANK_ONE),
            dim=1,
        ),
        USER_TO_ITEM_EDGE_TYPE: torch.cat(
            (MOCKED_U2I_EDGE_INDEX_ON_RANK_ZERO, MOCKED_U2I_EDGE_INDEX_ON_RANK_ONE),
            dim=1,
        ),
    },
    node_features={
        USER_NODE_TYPE: torch.cat(
            (
                MOCKED_USER_NODE_FEATURES_ON_RANK_ZERO,
                MOCKED_USER_NODE_FEATURES_ON_RANK_ONE,
            ),
            dim=0,
        ),
        ITEM_NODE_TYPE: torch.cat(
            (
                MOCKED_ITEM_NODE_FEATURES_ON_RANK_ZERO,
                MOCKED_ITEM_NODE_FEATURES_ON_RANK_ONE,
            ),
            dim=0,
        ),
    },
    edge_features={
        USER_TO_USER_EDGE_TYPE: torch.cat(
            (
                MOCKED_U2U_EDGE_FEATURES_ON_RANK_ZERO,
                MOCKED_U2U_EDGE_FEATURES_ON_RANK_ONE,
            ),
            dim=0,
        ),
    },
    positive_labels={
        USER_TO_USER_EDGE_TYPE: torch.cat(
            (
                MOCKED_U2U_POS_EDGE_INDEX_ON_RANK_ZERO,
                MOCKED_U2U_POS_EDGE_INDEX_ON_RANK_ONE,
            ),
            dim=1,
        ),
        USER_TO_ITEM_EDGE_TYPE: torch.cat(
            (
                MOCKED_U2I_POS_EDGE_INDEX_ON_RANK_ZERO,
                MOCKED_U2I_POS_EDGE_INDEX_ON_RANK_ONE,
            ),
            dim=1,
        ),
    },
    negative_labels={
        USER_TO_USER_EDGE_TYPE: torch.cat(
            (
                MOCKED_U2U_NEG_EDGE_INDEX_ON_RANK_ZERO,
                MOCKED_U2U_NEG_EDGE_INDEX_ON_RANK_ONE,
            ),
            dim=1,
        ),
        USER_TO_ITEM_EDGE_TYPE: torch.cat(
            (
                MOCKED_U2I_NEG_EDGE_INDEX_ON_RANK_ZERO,
                MOCKED_U2I_NEG_EDGE_INDEX_ON_RANK_ONE,
            ),
            dim=1,
        ),
    },
)
