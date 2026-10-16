"""The fixed-shape k-hop layout equals the sparse k-hop layout."""

from typing import Literal

import torch
from absl.testing import absltest

from gigl.transforms.graph_transformer import (
    _build_sequence_layout_from_sparse_neighbors,
    _get_k_hop_neighbors_with_hops,
)
from gigl.transforms.static_khop import (
    build_neighbor_table,
    gather_rows,
    khop_sequence_layout,
)
from tests.test_assets.test_case import TestCase


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
    adjacency = edge_index.flip(0) if direction == "in" else edge_index
    anchors = torch.arange(num_anchors)
    reachable, hops = _get_k_hop_neighbors_with_hops(
        anchor_indices=anchors,
        edge_index=adjacency,
        num_nodes=num_nodes,
        k=2,
        device=torch.device("cpu"),
    )
    node_index, valid, hop_sequences = _build_sequence_layout_from_sparse_neighbors(
        reachable=reachable,
        anchor_indices=anchors,
        max_seq_len=max_seq_len,
        include_anchor_first=True,
        device=torch.device("cpu"),
        hop_values=hops,
    )
    assert hop_sequences is not None
    return node_index, valid, hop_sequences.long()


class StaticKhopLayoutTest(TestCase):
    def _assert_layouts_equal(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_anchors: int,
        max_seq_len: int,
        width: int,
        direction: Literal["in", "out"],
    ) -> None:
        expected_nodes, expected_valid, expected_hops = _sparse_layout(
            edge_index, num_nodes, num_anchors, max_seq_len, direction
        )
        table, overflow = build_neighbor_table(edge_index, num_nodes, width, direction)
        nodes, hops, lengths = khop_sequence_layout(
            table, torch.arange(num_anchors), hop_distance=2, max_seq_len=max_seq_len
        )
        valid = torch.arange(max_seq_len).unsqueeze(0) < lengths.unsqueeze(1)
        self.assertFalse(bool(overflow))
        self.assertTrue(torch.equal(valid, expected_valid))
        self.assertTrue(torch.equal(nodes, expected_nodes))
        self.assertTrue(torch.equal(hops, expected_hops))

    def test_matches_sparse_layout_on_adversarial_graphs(self) -> None:
        for seed in range(4):
            for direction in ("in", "out"):
                for max_seq_len in (4, 16, 64):
                    with self.subTest(seed=seed, direction=direction, L=max_seq_len):
                        edge_index = _random_graph(40, 90, seed)
                        in_degree = torch.bincount(
                            edge_index[1 if direction == "in" else 0], minlength=40
                        )
                        self._assert_layouts_equal(
                            edge_index,
                            num_nodes=40,
                            num_anchors=10,
                            max_seq_len=max_seq_len,
                            width=int(in_degree.max()),
                            direction=direction,
                        )

    def test_hop_is_first_reach_and_isolated_anchors_have_length_one(self) -> None:
        # Node 3 is a hop-1 and a hop-2 neighbour of anchor 0; anchor 1 is isolated
        # and node 4 is shared between anchors 0 and 2.
        edge_index = torch.tensor([[3, 5, 3, 4, 4], [0, 0, 5, 3, 2]])
        table, _ = build_neighbor_table(edge_index, 6, width=2, sampling_direction="in")
        nodes, hops, lengths = khop_sequence_layout(
            table, torch.arange(3), hop_distance=2, max_seq_len=5
        )
        self.assertEqual(
            nodes.tolist(), [[0, 3, 4, 5, -1], [1, -1, -1, -1, -1], [2, 4, -1, -1, -1]]
        )
        self.assertEqual(
            hops.tolist(), [[0, 1, 2, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        )
        self.assertEqual(lengths.tolist(), [4, 1, 2])
        self._assert_layouts_equal(edge_index, 6, 3, 5, width=2, direction="in")

    def test_edgeless_batch_and_padding_edges(self) -> None:
        empty = torch.empty((2, 0), dtype=torch.long)
        self._assert_layouts_equal(empty, 5, 5, 8, width=3, direction="in")
        edge_index = torch.tensor([[2, 3], [0, 0]])
        padded = torch.cat([edge_index, torch.full((2, 3), -1)], dim=1)
        table, overflow = build_neighbor_table(
            padded, 4, width=2, sampling_direction="in"
        )
        self.assertFalse(bool(overflow))
        self.assertEqual(
            table.tolist(), build_neighbor_table(edge_index, 4, 2, "in")[0].tolist()
        )
        self.assertEqual(table[-1].tolist(), [-1, -1])

    def test_overflow_flags_rows_wider_than_the_table(self) -> None:
        edge_index = torch.tensor([[1, 2, 3], [0, 0, 0]])
        _, fits = build_neighbor_table(edge_index, 4, width=3, sampling_direction="in")
        _, overflows = build_neighbor_table(
            edge_index, 4, width=2, sampling_direction="in"
        )
        self.assertFalse(bool(fits))
        self.assertTrue(bool(overflows))

    def test_gather_rows_zeroes_invalid_positions(self) -> None:
        features = torch.arange(6.0).view(3, 2)
        index = torch.tensor([[2, -1], [0, 1]])
        valid = torch.tensor([[True, False], [True, False]])
        self.assertEqual(
            gather_rows(features, index, valid).tolist(),
            [[[4.0, 5.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
        )


if __name__ == "__main__":
    absltest.main()
