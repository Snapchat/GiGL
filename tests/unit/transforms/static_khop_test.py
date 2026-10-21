"""The fixed-shape k-hop layout equals the sparse k-hop layout."""

from typing import Literal

import torch
from absl.testing import absltest

from gigl.transforms.graph_transformer import (
    _build_hop_ordered_sequence_layout_from_sparse_edges,
    _build_sequence_layout_from_sparse_neighbors,
    _get_k_hop_neighbors_sparse,
)
from gigl.transforms.static_khop import KhopLayoutTooLargeError, khop_layout
from tests.test_assets.test_case import TestCase

_CPU = torch.device("cpu")


def _random_graph(num_nodes: int, num_edges: int, seed: int) -> torch.Tensor:
    """Random edges plus duplicates, self loops (on a seed and a non-seed) and a 2-cycle."""
    generator = torch.Generator().manual_seed(seed)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=generator)
    extra = torch.tensor([[0, 7, 1, 2], [0, 7, 2, 1]])
    return torch.cat([edge_index, edge_index[:, :5], extra], dim=1)


def _first_reach_hops(
    node_index: torch.Tensor,
    valid: torch.Tensor,
    adjacency: torch.Tensor,
    num_nodes: int,
    hop_distance: int,
) -> torch.Tensor:
    """The smallest hop at which each valid token is reachable; 0 at the anchor and padding."""
    anchors = torch.arange(node_index.size(0))
    hops = torch.zeros_like(node_index)
    # Go from the farthest hop to the nearest so the nearest reach wins.
    for hop in range(hop_distance, 0, -1):
        reach = _get_k_hop_neighbors_sparse(anchors, adjacency, num_nodes, hop, _CPU)
        in_reach = reach.to_dense()[anchors.unsqueeze(1), node_index.clamp(min=0)] > 0
        hops = torch.where(in_reach & (node_index >= 0), hop, hops)
    hops = torch.where(valid, hops, 0)
    hops[:, 0] = 0
    return hops


def _sparse_layout(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_anchors: int,
    max_seq_len: int,
    direction: Literal["in", "out"],
    hop_distance: int = 2,
    prioritize_hop_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Node ids, validity and first-reach hops from the sparse path."""
    adjacency = edge_index.flip(0) if direction == "in" else edge_index
    anchors = torch.arange(num_anchors)
    if prioritize_hop_order:
        node_index, valid = _build_hop_ordered_sequence_layout_from_sparse_edges(
            anchor_indices=anchors,
            edge_index=adjacency,
            num_nodes=num_nodes,
            hop_distance=hop_distance,
            max_seq_len=max_seq_len,
            include_anchor_first=True,
            device=_CPU,
        )
    else:
        node_index, valid = _build_sequence_layout_from_sparse_neighbors(
            reachable=_get_k_hop_neighbors_sparse(
                anchors, adjacency, num_nodes, hop_distance, _CPU
            ),
            anchor_indices=anchors,
            max_seq_len=max_seq_len,
            include_anchor_first=True,
            device=_CPU,
        )
    hops = _first_reach_hops(node_index, valid, adjacency, num_nodes, hop_distance)
    return node_index, valid, hops


class StaticKhopLayoutTest(TestCase):
    def _assert_layouts_equal(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_anchors: int,
        max_seq_len: int,
        direction: Literal["in", "out"],
        hop_distance: int = 2,
        prioritize_hop_order: bool = False,
    ) -> None:
        expected_nodes, expected_valid, expected_hops = _sparse_layout(
            edge_index,
            num_nodes,
            num_anchors,
            max_seq_len,
            direction,
            hop_distance,
            prioritize_hop_order,
        )
        nodes, hops, lengths = khop_layout(
            edge_index,
            num_nodes,
            torch.arange(num_anchors),
            hop_distance=hop_distance,
            max_seq_len=max_seq_len,
            sampling_direction=direction,
            prioritize_hop_order=prioritize_hop_order,
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

    def test_matches_sparse_layout_at_three_and_four_hops(self) -> None:
        for hop_distance in (3, 4):
            for direction in ("in", "out"):
                for max_seq_len in (4, 16, 64):
                    with self.subTest(
                        hop=hop_distance, direction=direction, L=max_seq_len
                    ):
                        self._assert_layouts_equal(
                            _random_graph(40, 70, hop_distance),
                            num_nodes=40,
                            num_anchors=10,
                            max_seq_len=max_seq_len,
                            direction=direction,
                            hop_distance=hop_distance,
                        )

    def test_hop_order_matches_the_hop_ordered_sparse_layout(self) -> None:
        for hop_distance in (1, 2, 3):
            for direction in ("in", "out"):
                for max_seq_len in (3, 8, 64):
                    with self.subTest(
                        hop=hop_distance, direction=direction, L=max_seq_len
                    ):
                        self._assert_layouts_equal(
                            _random_graph(40, 70, hop_distance),
                            num_nodes=40,
                            num_anchors=10,
                            max_seq_len=max_seq_len,
                            direction=direction,
                            hop_distance=hop_distance,
                            prioritize_hop_order=True,
                        )

    def test_hop_order_truncation_drops_the_farthest_hops_first(self) -> None:
        # Anchor 0 has hop-1 neighbours 5 and 6 and hop-2 neighbours 1 and 2.
        edge_index = torch.tensor([[0, 0, 5, 6], [5, 6, 1, 2]])
        for prioritize_hop_order, expected in ((False, [0, 1, 2]), (True, [0, 5, 6])):
            nodes, hops, _ = khop_layout(
                edge_index,
                7,
                torch.tensor([0]),
                hop_distance=2,
                max_seq_len=3,
                sampling_direction="out",
                prioritize_hop_order=prioritize_hop_order,
            )
            self.assertEqual(nodes[0].tolist(), expected)

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

    def test_accepts_int32_edge_index_and_anchors(self) -> None:
        edge_index = _random_graph(40, 90, 0)
        anchors = torch.tensor([5, 0, 17])
        for hop_distance in (0, 1, 2):
            with self.subTest(hop_distance=hop_distance):
                expected = khop_layout(edge_index, 40, anchors, hop_distance, 8)
                actual = khop_layout(
                    edge_index.int(), 40, anchors.int(), hop_distance, 8
                )
                for expected_tensor, actual_tensor in zip(expected, actual):
                    self.assertEqual(actual_tensor.dtype, torch.long)
                    self.assertTrue(torch.equal(actual_tensor, expected_tensor))

    def test_rejects_a_layout_over_the_candidate_budget(self) -> None:
        # Width 3 at two hops is 1 + 3 + 9 = 13 candidates per anchor.
        edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]])
        anchors = torch.arange(2)
        khop_layout(edge_index, 4, anchors, 2, 8, max_candidates=26)
        with self.assertRaises(KhopLayoutTooLargeError):
            khop_layout(edge_index, 4, anchors, 2, 8, max_candidates=25)


if __name__ == "__main__":
    absltest.main()
