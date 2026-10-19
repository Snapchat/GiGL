"""The fixed-shape k-hop layout equals the sparse k-hop layout."""

from typing import Literal

import torch
from absl.testing import absltest

from gigl.transforms.graph_transformer import (
    _build_sequence_layout_from_sparse_neighbors,
    _get_k_hop_neighbors_sparse,
)
from gigl.transforms.static_khop import build_neighbor_table, khop_layout
from tests.test_assets.test_case import TestCase

_CPU = torch.device("cpu")


def _random_graph(num_nodes: int, num_edges: int, seed: int) -> torch.Tensor:
    """Random edges plus duplicates, self loops (on a seed and a non-seed) and a 2-cycle."""
    generator = torch.Generator().manual_seed(seed)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=generator)
    extra = torch.tensor([[0, 7, 1, 2], [0, 7, 2, 1]])
    return torch.cat([edge_index, edge_index[:, :5], extra], dim=1)


def _sparse_layout(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_anchors: int,
    max_seq_len: int,
    direction: Literal["in", "out"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Node ids, validity and first-reach hops from the sparse path."""
    adjacency = edge_index.flip(0) if direction == "in" else edge_index
    anchors = torch.arange(num_anchors)
    node_index, valid = _build_sequence_layout_from_sparse_neighbors(
        reachable=_get_k_hop_neighbors_sparse(anchors, adjacency, num_nodes, 2, _CPU),
        anchor_indices=anchors,
        max_seq_len=max_seq_len,
        include_anchor_first=True,
        device=_CPU,
    )
    # A node's first-reach hop is 1 if it is a one-hop neighbour, else 2.
    one_hop = _get_k_hop_neighbors_sparse(anchors, adjacency, num_nodes, 1, _CPU)
    one_hop_dense = one_hop.to_dense() > 0
    in_one_hop = one_hop_dense[anchors.unsqueeze(1), node_index.clamp(min=0)] & (
        node_index >= 0
    )
    hops = torch.where(in_one_hop, 1, 2)
    hops = torch.where(valid, hops, 0)
    hops[:, 0] = 0
    return node_index, valid, hops


class StaticKhopLayoutTest(TestCase):
    def _assert_layouts_equal(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_anchors: int,
        max_seq_len: int,
        direction: Literal["in", "out"],
    ) -> None:
        expected_nodes, expected_valid, expected_hops = _sparse_layout(
            edge_index, num_nodes, num_anchors, max_seq_len, direction
        )
        nodes, hops, lengths = khop_layout(
            edge_index,
            num_nodes,
            torch.arange(num_anchors),
            hop_distance=2,
            max_seq_len=max_seq_len,
            sampling_direction=direction,
        )
        valid = torch.arange(max_seq_len).unsqueeze(0) < lengths.unsqueeze(1)
        self.assertTrue(torch.equal(valid, expected_valid))
        self.assertTrue(torch.equal(nodes, expected_nodes))
        self.assertTrue(torch.equal(hops, expected_hops))

    def test_matches_sparse_layout_on_adversarial_graphs(self) -> None:
        for seed in range(4):
            for direction in ("in", "out"):
                for max_seq_len in (4, 16, 64):
                    with self.subTest(seed=seed, direction=direction, L=max_seq_len):
                        self._assert_layouts_equal(
                            _random_graph(40, 90, seed),
                            num_nodes=40,
                            num_anchors=10,
                            max_seq_len=max_seq_len,
                            direction=direction,
                        )

    def test_hop_is_first_reach_and_isolated_anchors_have_length_one(self) -> None:
        # Node 3 is a hop-1 and a hop-2 neighbour of anchor 0; anchor 1 is isolated
        # and node 4 is shared between anchors 0 and 2.
        edge_index = torch.tensor([[3, 5, 3, 4, 4], [0, 0, 5, 3, 2]])
        nodes, hops, lengths = khop_layout(
            edge_index,
            6,
            torch.arange(3),
            hop_distance=2,
            max_seq_len=5,
            sampling_direction="in",
        )
        self.assertEqual(
            nodes.tolist(), [[0, 3, 4, 5, -1], [1, -1, -1, -1, -1], [2, 4, -1, -1, -1]]
        )
        self.assertEqual(
            hops.tolist(), [[0, 1, 2, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        )
        self.assertEqual(lengths.tolist(), [4, 1, 2])
        self._assert_layouts_equal(edge_index, 6, 3, 5, direction="in")

    def test_edgeless_batch(self) -> None:
        empty = torch.empty((2, 0), dtype=torch.long)
        self.assertEqual(build_neighbor_table(empty, 5).shape, (6, 1))
        self._assert_layouts_equal(empty, 5, 5, 8, direction="in")

    def test_table_width_is_the_largest_degree(self) -> None:
        edge_index = torch.tensor([[1, 2, 3, 3], [0, 0, 0, 1]])
        table = build_neighbor_table(edge_index, 4, sampling_direction="in")
        self.assertEqual(
            table.tolist(),
            [[1, 2, 3], [3, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        )

    def test_two_node_types_via_offsets(self) -> None:
        # Nodes 0-2 are one type and 3-5 another; edges cross the boundary.
        edge_index = torch.tensor([[3, 4, 0, 5], [0, 1, 4, 2]])
        self._assert_layouts_equal(edge_index, 6, 3, 6, direction="in")
        self._assert_layouts_equal(edge_index, 6, 3, 6, direction="out")


if __name__ == "__main__":
    absltest.main()
