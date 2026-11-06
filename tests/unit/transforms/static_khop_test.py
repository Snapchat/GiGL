"""The fixed-shape k-hop layout equals the sparse k-hop layout."""

import doctest
import itertools
from typing import Any, Literal
from unittest import mock

import torch
from absl.testing import absltest

from gigl.transforms import sequence_layout, static_khop
from gigl.transforms.graph_transformer import (
    _build_hop_ordered_sequence_layout_from_sparse_edges,
    _build_sequence_layout_from_sparse_neighbors,
    _get_k_hop_neighbors_sparse,
)
from gigl.transforms.sequence_layout import pack_sequences, select_sequences
from gigl.transforms.static_khop import KhopLayoutTooLargeError, pack_khop_layout
from gigl.transforms.static_khop import khop_layout as _khop_layout
from tests.test_assets.test_case import TestCase

_CPU = torch.device("cpu")


def khop_layout(
    *args: Any, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``khop_layout`` with the cells decoded: node ids, first-reach hops (0 at padding), lengths."""
    layout, lengths, hop_bits = _khop_layout(*args, **kwargs)
    valid = layout >= 0
    nodes = torch.where(valid, layout >> hop_bits, -1)
    hops = torch.where(valid, layout & ((1 << hop_bits) - 1), 0)
    return nodes, hops, lengths


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
    ordering: Literal["node_id", "hop"] = "node_id",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Node ids, validity and first-reach hops from the sparse path."""
    adjacency = edge_index.flip(0) if direction == "in" else edge_index
    anchors = torch.arange(num_anchors)
    if ordering == "hop":
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
        ordering: Literal["node_id", "hop"] = "node_id",
    ) -> None:
        expected_nodes, expected_valid, expected_hops = _sparse_layout(
            edge_index,
            num_nodes,
            num_anchors,
            max_seq_len,
            direction,
            hop_distance,
            ordering,
        )
        nodes, hops, lengths = khop_layout(
            edge_index,
            num_nodes,
            torch.arange(num_anchors),
            hop_distance=hop_distance,
            max_seq_len=max_seq_len,
            sampling_direction=direction,
            ordering=ordering,
        )
        valid = torch.arange(max_seq_len).unsqueeze(0) < lengths.unsqueeze(1)
        self.assertTrue(torch.equal(valid, expected_valid))
        self.assertTrue(torch.equal(nodes, expected_nodes))
        self.assertTrue(torch.equal(hops, expected_hops))

    def test_matches_sparse_layout(self) -> None:
        for (
            hop_distance,
            ordering,
            direction,
            max_seq_len,
            seed,
        ) in itertools.product(
            (1, 2, 3, 4), ("node_id", "hop"), ("in", "out"), (3, 4, 16, 64), range(2)
        ):
            with self.subTest(
                hop=hop_distance,
                ordering=ordering,
                direction=direction,
                L=max_seq_len,
                seed=seed,
            ):
                self._assert_layouts_equal(
                    _random_graph(40, 80, seed),
                    num_nodes=40,
                    num_anchors=10,
                    max_seq_len=max_seq_len,
                    direction=direction,
                    hop_distance=hop_distance,
                    ordering=ordering,
                )

    def test_hop_order_truncation_drops_the_farthest_hops_first(self) -> None:
        # Anchor 0 has hop-1 neighbours 5 and 6 and hop-2 neighbours 1 and 2.
        edge_index = torch.tensor([[0, 0, 5, 6], [5, 6, 1, 2]])
        for ordering, expected in (("node_id", [0, 1, 2]), ("hop", [0, 5, 6])):
            nodes, hops, _ = khop_layout(
                edge_index,
                7,
                torch.tensor([0]),
                hop_distance=2,
                max_seq_len=3,
                sampling_direction="out",
                ordering=ordering,
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

    def test_rejects_an_unknown_sampling_direction(self) -> None:
        edge_index = torch.tensor([[0, 1], [1, 2]])
        with self.assertRaises(ValueError):
            khop_layout(edge_index, 3, torch.arange(1), 2, 4, sampling_direction="both")

    def test_rejects_a_layout_over_the_candidate_budget(self) -> None:
        # Width 3 at two hops is 1 + 3 + 9 = 13 candidates per anchor, plus a
        # 5 x 3 neighbour table.
        edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]])
        anchors = torch.arange(2)
        khop_layout(edge_index, 4, anchors, 2, 8, max_candidates=41)
        with self.assertRaises(KhopLayoutTooLargeError):
            khop_layout(edge_index, 4, anchors, 2, 8, max_candidates=40)

    def test_rejects_a_huge_hub_table_before_allocating_it(self) -> None:
        # One anchor passes the candidate check on its own (1 + 3000 + 3000 ** 2
        # is about 9M), but a 200k-node table of that width would be 4.8 GB.
        num_nodes = 200_000
        edge_index = torch.stack(
            [torch.zeros(3000, dtype=torch.long), torch.arange(1, 3001)]
        )
        with mock.patch.object(torch, "full", side_effect=AssertionError("allocated")):
            with self.assertRaises(KhopLayoutTooLargeError):
                khop_layout(edge_index, num_nodes, torch.tensor([5]), 2, 8)

    def test_orderings_pick_different_tokens_when_truncated(self) -> None:
        # Anchor 0 reaches 5 then 3 in slot order. 5 leads to 8 and 1; 3 leads to
        # 7 and 1. Node 1 is first reached through 5, at hop 2.
        edge_index = torch.tensor([[0, 0, 5, 5, 3, 3], [5, 3, 8, 1, 7, 1]])
        expected = {
            "node_id": ([0, 1, 3, 5, 7, 8], [0, 2, 1, 1, 2, 2]),
            "hop": ([0, 3, 5, 1, 7, 8], [0, 1, 1, 2, 2, 2]),
            "input": ([0, 5, 3, 8, 1, 7], [0, 1, 1, 2, 2, 2]),
        }
        for ordering, (expected_nodes, expected_hops) in expected.items():
            for max_seq_len in (6, 4):
                with self.subTest(ordering=ordering, L=max_seq_len):
                    nodes, hops, lengths = khop_layout(
                        edge_index,
                        10,
                        torch.tensor([0]),
                        hop_distance=2,
                        max_seq_len=max_seq_len,
                        ordering=ordering,
                    )
                    self.assertEqual(nodes[0].tolist(), expected_nodes[:max_seq_len])
                    self.assertEqual(hops[0].tolist(), expected_hops[:max_seq_len])
                    self.assertEqual(lengths.tolist(), [max_seq_len])

    def test_rejects_an_unknown_ordering(self) -> None:
        edge_index = torch.tensor([[0, 1], [1, 2]])
        with self.assertRaises(ValueError):
            khop_layout(edge_index, 3, torch.arange(1), 2, 4, ordering="degree")

    def test_select_sequences_orders_by_priority_and_keeps_the_first_copy(self) -> None:
        # Anchor 4 is among its candidates and is dropped from them. Node 9 is in
        # columns 0 and 2; the first copy (priority 0.5) is the one kept. Nodes 6
        # and 9 tie on priority and fall back to ascending node id.
        anchors = torch.tensor([4, 1])
        candidates = torch.tensor([[9, 2, 9, 4, -1, 6], [-1] * 6])
        priority = torch.tensor([[0.5, 0.9, 0.1, 0.0, 0.0, 0.5], [0.0] * 6])
        nodes, source, lengths = select_sequences(anchors, candidates, 5, priority)
        self.assertEqual(nodes.tolist(), [[4, 6, 9, 2, -1], [1, -1, -1, -1, -1]])
        self.assertEqual(source.tolist(), [[-1, 5, 0, 1, -1], [-1] * 5])
        self.assertEqual(lengths.tolist(), [4, 1])

        nodes, source, lengths = select_sequences(anchors, candidates, 3, priority)
        self.assertEqual(nodes.tolist(), [[4, 6, 9], [1, -1, -1]])
        self.assertEqual(lengths.tolist(), [3, 1])

        nodes, source, lengths = select_sequences(anchors, candidates, 5)
        self.assertEqual(nodes[0].tolist(), [4, 2, 6, 9, -1])
        self.assertEqual(source[0].tolist(), [-1, 1, 5, 0, -1])

    def test_select_sequences_can_tie_by_column_and_leave_out_the_anchor(self) -> None:
        anchors = torch.tensor([4, 1])
        candidates = torch.tensor([[9, 2, 9, 4, -1, 6], [-1] * 6])
        priority = torch.tensor([[0.5, 0.9, 0.1, 0.0, 0.0, 0.5], [0.0] * 6])
        # Nodes 9 and 6 tie on priority; 9 comes first because its column does.
        nodes, source, lengths = select_sequences(
            anchors, candidates, 5, priority, ties_by_column=True
        )
        self.assertEqual(nodes.tolist(), [[4, 9, 6, 2, -1], [1, -1, -1, -1, -1]])
        self.assertEqual(source[0].tolist(), [-1, 0, 5, 1, -1])
        self.assertEqual(lengths.tolist(), [4, 1])

        # Without the anchor slot, anchor 4 is an ordinary candidate (column 3).
        nodes, source, lengths = select_sequences(
            anchors, candidates, 5, priority, include_anchor=False
        )
        self.assertEqual(nodes.tolist(), [[4, 6, 9, 2, -1], [-1] * 5])
        self.assertEqual(source.tolist(), [[3, 5, 0, 1, -1], [-1] * 5])
        self.assertEqual(lengths.tolist(), [4, 0])

        nodes, _, lengths = select_sequences(
            anchors, candidates, 3, priority, include_anchor=False
        )
        self.assertEqual(nodes.tolist(), [[4, 6, 9], [-1] * 3])
        self.assertEqual(lengths.tolist(), [3, 0])

    def test_select_sequences_does_not_recompile_across_batch_shapes(self) -> None:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        self.addCleanup(torch._dynamo.reset)
        compiled = torch.compile(select_sequences, backend="eager", dynamic=True)
        generator = torch.Generator().manual_seed(0)
        for num_anchors, num_candidates in ((6, 9), (11, 30), (4, 17)):
            candidates = torch.randint(
                -1, 50, (num_anchors, num_candidates), generator=generator
            )
            priority = torch.rand(num_anchors, num_candidates, generator=generator)
            anchors = torch.arange(num_anchors)
            for args in ((anchors, candidates, 8, priority), (anchors, candidates, 8)):
                expected = select_sequences(*args)
                for expected_tensor, actual_tensor in zip(expected, compiled(*args)):
                    self.assertTrue(torch.equal(expected_tensor, actual_tensor))
        # One graph for each of the two call signatures.
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 2)

    def test_docstring_examples(self) -> None:
        for module in (sequence_layout, static_khop):
            with self.subTest(module=module.__name__):
                result = doctest.testmod(module)
                self.assertGreater(result.attempted, 0)
                self.assertEqual(result.failed, 0)

    def test_pack_sequences_layout(self) -> None:
        layout = pack_sequences(torch.tensor([2, 1, 3]), 3)
        self.assertEqual(layout.flat_index.tolist(), [0, 1, 3, 6, 7, 8])
        self.assertEqual(layout.cu_seqlens.tolist(), [0, 2, 3, 6])
        self.assertEqual(layout.cu_seqlens.dtype, torch.int32)
        self.assertEqual(layout.max_seq_len, 3)

        # A length above the grid width is clamped, so the offsets match the indices.
        layout = pack_sequences(torch.tensor([5, 1]), 3)
        self.assertEqual(layout.flat_index.tolist(), [0, 1, 2, 3])
        self.assertEqual(layout.cu_seqlens.tolist(), [0, 3, 4])

    def test_pack_sequences_layout_fields(self) -> None:
        lengths = torch.tensor([3, 1, 2])
        node_grid = torch.arange(100, 109).view(3, 3)
        layout = pack_sequences(
            lengths,
            max_seq_len=3,
            node_grid=node_grid,
            attr_grids={"hop": node_grid % 3},
        )
        self.assertEqual(layout.flat_index.tolist(), [0, 1, 2, 3, 6, 7])
        self.assertEqual(layout.sequence_id.tolist(), [0, 0, 0, 1, 2, 2])
        self.assertEqual(layout.position.tolist(), [0, 1, 2, 0, 0, 1])
        self.assertEqual(layout.node_index.tolist(), [100, 101, 102, 103, 106, 107])
        self.assertEqual(layout.token_attrs["hop"].tolist(), [1, 2, 0, 1, 1, 2])
        self.assertEqual(layout.cu_seqlens.tolist(), [0, 3, 4, 6])
        self.assertEqual(layout.cu_seqlens.dtype, torch.int32)
        self.assertIsNone(layout.relation_edges)

    def test_pack_sequences_without_a_node_grid_uses_the_flat_index(self) -> None:
        layout = pack_sequences(torch.tensor([2, 1]), max_seq_len=2)
        self.assertEqual(layout.node_index.tolist(), layout.flat_index.tolist())

    def test_pack_khop_layout_matches_khop_layout(self) -> None:
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        layout = pack_khop_layout(
            edge_index,
            num_nodes=5,
            anchor_indices=torch.arange(2),
            hop_distance=2,
            max_seq_len=4,
        )
        self.assertEqual(layout.node_index.tolist(), [0, 1, 2, 1, 2, 3])
        self.assertEqual(
            layout.token_attrs["hop_distance"].tolist(), [0, 1, 2, 0, 1, 2]
        )
        self.assertEqual(layout.cu_seqlens.tolist(), [0, 3, 6])


if __name__ == "__main__":
    absltest.main()
