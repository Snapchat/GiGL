"""Shared bipartite graph builders for weighted-sampling tests.

Used by both distributed_weighted_sampling_test and
distributed_ppr_weighted_sampling_test.  Each builder returns a single-rank
PartitionOutput that encodes node type as a feature (hub/user=2.0, good=1.0,
bad=0.0) so tests can assert that weight=0 edges never appear in any sampled
subgraph.
"""

import torch

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
)

USER = NodeType("user")
ITEM = NodeType("item")
USER_TO_ITEM = EdgeType(USER, Relation("to"), ITEM)
ITEM_TO_USER = EdgeType(ITEM, Relation("to"), USER)


def build_homogeneous_bipartite_weight_graph() -> tuple[PartitionOutput, int]:
    """Build a homogeneous graph with hub, good, and bad nodes.

    Graph structure:
      - 10 hub nodes (0..9): used as seed nodes; feature value = 2.0
      - 50 good nodes (10..59): reachable from hubs via weight=1 edges; feature = 1.0
      - 40 bad nodes (60..99): reachable from hubs via weight=0 edges; feature = 0.0
      - Each good node also has 5 outgoing weight=1 edges to nearby good nodes
        (ring topology, for 2nd-hop sampling).

    With weighted sampling only good nodes should ever appear as sampled
    neighbors — weight=0 edges to bad nodes must never be traversed.

    Returns:
        (partition_output, n_hub)
    """
    n_hub = 10
    n_good = 50
    n_bad = 40
    n = n_hub + n_good + n_bad  # 100

    hub_ids = torch.arange(n_hub)
    good_ids = torch.arange(n_hub, n_hub + n_good)
    bad_ids = torch.arange(n_hub + n_good, n)

    # Hub → Good: weight=1
    hub_good_src = hub_ids.repeat_interleave(n_good)
    hub_good_dst = good_ids.repeat(n_hub)
    hub_good_w = torch.ones(n_hub * n_good)

    # Hub → Bad: weight=0
    hub_bad_src = hub_ids.repeat_interleave(n_bad)
    hub_bad_dst = bad_ids.repeat(n_hub)
    hub_bad_w = torch.zeros(n_hub * n_bad)

    # Good → Good: ring with 5 outgoing edges per node, weight=1 (2nd-hop targets)
    connections_per_good = 5
    good_src = good_ids.repeat_interleave(connections_per_good)
    # Row i of [connections_per_good, n_good].T gives neighbors of good_ids[i]
    good_dst = torch.stack(
        [torch.roll(good_ids, -j) for j in range(1, connections_per_good + 1)]
    ).T.reshape(-1)
    good_w = torch.ones(n_good * connections_per_good)

    edge_src = torch.cat([hub_good_src, hub_bad_src, good_src])
    edge_dst = torch.cat([hub_good_dst, hub_bad_dst, good_dst])
    weights = torch.cat([hub_good_w, hub_bad_w, good_w])
    edge_index = torch.stack([edge_src, edge_dst])
    n_edges = edge_src.shape[0]

    # Feature encodes node type: hub=2.0, good=1.0, bad=0.0
    node_feats = torch.cat(
        [
            torch.full((n_hub, 1), 2.0),
            torch.full((n_good, 1), 1.0),
            torch.full((n_bad, 1), 0.0),
        ]
    )

    partition_output = PartitionOutput(
        node_partition_book=torch.zeros(n),
        edge_partition_book=torch.zeros(n_edges),
        partitioned_edge_index=GraphPartitionData(
            edge_index=edge_index,
            edge_ids=torch.arange(n_edges) + 1_000,
            weights=weights,
        ),
        partitioned_node_features=FeaturePartitionData(
            feats=node_feats,
            ids=torch.arange(n),
        ),
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    return partition_output, n_hub


def build_heterogeneous_bipartite_weight_graph() -> tuple[PartitionOutput, int]:
    """Build a heterogeneous (user/item) graph with good and bad item nodes.

    Graph structure:
      - 10 user nodes (0..9): seed nodes; user feature = 2.0
      - 60 item nodes total:
          - Items 0..39: good, reachable from users via weight=1 edges; feature = 1.0
          - Items 40..59: bad, reachable from users via weight=0 edges; feature = 0.0
      - Good items also have weight=1 edges back to all users (for 2nd-hop).

    With weighted sampling only good item nodes should ever appear as sampled
    item neighbors.

    Returns:
        (partition_output, n_user)
    """
    n_user = 10
    n_good_item = 40
    n_bad_item = 20
    n_item = n_good_item + n_bad_item  # 60

    user_ids = torch.arange(n_user)
    good_item_ids = torch.arange(n_good_item)
    bad_item_ids = torch.arange(n_good_item, n_item)

    # User → Good Item: weight=1
    u2gi_src = user_ids.repeat_interleave(n_good_item)
    u2gi_dst = good_item_ids.repeat(n_user)
    u2gi_w = torch.ones(n_user * n_good_item)

    # User → Bad Item: weight=0
    u2bi_src = user_ids.repeat_interleave(n_bad_item)
    u2bi_dst = bad_item_ids.repeat(n_user)
    u2bi_w = torch.zeros(n_user * n_bad_item)

    # Good Item → User: weight=1 (2nd-hop back to users)
    gi2u_src = good_item_ids.repeat_interleave(n_user)
    gi2u_dst = user_ids.repeat(n_good_item)
    gi2u_w = torch.ones(n_good_item * n_user)

    u2i_src = torch.cat([u2gi_src, u2bi_src])
    u2i_dst = torch.cat([u2gi_dst, u2bi_dst])
    u2i_w = torch.cat([u2gi_w, u2bi_w])
    n_u2i_edges = u2i_src.shape[0]

    user_feats = torch.full((n_user, 1), 2.0)
    # Item feature encodes type: good=1.0, bad=0.0
    item_feats = torch.cat(
        [
            torch.full((n_good_item, 1), 1.0),
            torch.full((n_bad_item, 1), 0.0),
        ]
    )

    partition_output = PartitionOutput(
        node_partition_book={
            USER: torch.zeros(n_user),
            ITEM: torch.zeros(n_item),
        },
        edge_partition_book={
            USER_TO_ITEM: torch.zeros(n_u2i_edges),
            ITEM_TO_USER: torch.zeros(gi2u_src.shape[0]),
        },
        partitioned_edge_index={
            USER_TO_ITEM: GraphPartitionData(
                edge_index=torch.stack([u2i_src, u2i_dst]),
                edge_ids=torch.arange(n_u2i_edges) + 1_000,
                weights=u2i_w,
            ),
            ITEM_TO_USER: GraphPartitionData(
                edge_index=torch.stack([gi2u_src, gi2u_dst]),
                edge_ids=torch.arange(gi2u_src.shape[0]) + 10_000,
                weights=gi2u_w,
            ),
        },
        partitioned_node_features={
            USER: FeaturePartitionData(feats=user_feats, ids=torch.arange(n_user)),
            ITEM: FeaturePartitionData(feats=item_feats, ids=torch.arange(n_item)),
        },
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    return partition_output, n_user
