import collections
from typing import Literal, Optional

import torch
from absl.testing import absltest
from graphlearn_torch.data import Topology
from parameterized import param, parameterized

from gigl.distributed.utils.topology import OffsetTopology, contains_offset_topology
from tests.test_assets.test_case import TestCase

# A partition owning global compressed-dimension ids [100, 105) — the offset is
# deliberately far from zero so globally-sized allocations are distinguishable
# from locally-sized ones.
_OFFSET = 100
_NUM_NODES = 5

# Compressed-dimension ids (all in [100, 105)); the adjacent dimension holds
# arbitrary global neighbor ids, including ids far outside the partition range.
_COMPRESSED = torch.tensor([104, 100, 102, 100, 103, 102, 100])
_ADJACENT = torch.tensor([7, 900, 15, 3, 250, 11, 42])
_EDGE_IDS = torch.tensor([20, 21, 22, 23, 24, 25, 26])
_EDGE_WEIGHTS = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])


def _edge_index_for_layout(layout: str) -> torch.Tensor:
    # For CSR the compressed dimension is the row; for CSC it is the column.
    if layout == "CSR":
        return torch.stack([_COMPRESSED, _ADJACENT])
    else:
        return torch.stack([_ADJACENT, _COMPRESSED])


def _rows_to_edges(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    edge_ids: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    row_id_base: int,
) -> dict[int, collections.Counter]:
    """Collects each row's (neighbor, edge_id, weight) multiset, keyed by global row id."""
    edges_by_row: dict[int, collections.Counter] = {}
    for local_row in range(indptr.numel() - 1):
        start, end = int(indptr[local_row].item()), int(indptr[local_row + 1].item())
        edges = collections.Counter(
            (
                int(indices[position].item()),
                int(edge_ids[position].item()),
                float(edge_weights[position].item())
                if edge_weights is not None
                else None,
            )
            for position in range(start, end)
        )
        edges_by_row[row_id_base + local_row] = edges
    return edges_by_row


class OffsetTopologyTest(TestCase):
    def test_indptr_is_sized_to_the_local_node_range(self):
        """The core fix: stock Topology allocates indptr from the global id
        ceiling, OffsetTopology from the local partition node range."""
        offset_topology = OffsetTopology(
            edge_index=_edge_index_for_layout("CSR"),
            edge_ids=_EDGE_IDS,
            layout="CSR",
            offset=_OFFSET,
            num_nodes=_NUM_NODES,
        )
        self.assertEqual(offset_topology.indptr.numel(), _NUM_NODES + 1)

        stock_topology = Topology(
            edge_index=_edge_index_for_layout("CSR"),
            edge_ids=_EDGE_IDS,
            layout="CSR",
        )
        # Stock GLT infers the compressed-dimension size as rows.max() + 1, so
        # its indptr spans the whole global prefix [0, rows.max()].
        self.assertEqual(
            stock_topology.indptr.numel(), int(_COMPRESSED.max().item()) + 2
        )
        self.assertGreater(
            stock_topology.indptr.numel(), offset_topology.indptr.numel()
        )

    @parameterized.expand(
        [
            param("CSR weighted", layout="CSR", edge_weights=_EDGE_WEIGHTS),
            param("CSR unweighted", layout="CSR", edge_weights=None),
            param("CSC weighted", layout="CSC", edge_weights=_EDGE_WEIGHTS),
            param("CSC unweighted", layout="CSC", edge_weights=None),
        ]
    )
    def test_per_row_neighbor_multisets_match_stock_topology(
        self, _, layout: Literal["CSR", "CSC"], edge_weights: Optional[torch.Tensor]
    ):
        """Each row's (neighbor, edge_id, weight) multiset matches stock GLT's,
        offset-adjusted; intra-row order may differ."""
        edge_index = _edge_index_for_layout(layout)
        offset_topology = OffsetTopology(
            edge_index=edge_index,
            edge_ids=_EDGE_IDS,
            edge_weights=edge_weights,
            layout=layout,
            offset=_OFFSET,
            num_nodes=_NUM_NODES,
        )
        stock_topology = Topology(
            edge_index=edge_index,
            edge_ids=_EDGE_IDS,
            edge_weights=edge_weights,
            layout=layout,
        )

        offset_edges = _rows_to_edges(
            offset_topology.indptr,
            offset_topology.indices,
            offset_topology.edge_ids,
            offset_topology.edge_weights,
            row_id_base=_OFFSET,
        )
        stock_edges = _rows_to_edges(
            stock_topology.indptr,
            stock_topology.indices,
            stock_topology.edge_ids,
            stock_topology.edge_weights,
            row_id_base=0,
        )
        for global_row, edges in offset_edges.items():
            self.assertEqual(edges, stock_edges.get(global_row, collections.Counter()))
        # Rows outside the partition range hold no edges in the stock topology
        # either (the whole graph lives in [offset, offset + num_nodes)).
        for global_row, edges in stock_edges.items():
            if global_row not in offset_edges:
                self.assertEqual(edges, collections.Counter())

    @parameterized.expand(
        [
            param("CSR", layout="CSR"),
            param("CSC", layout="CSC"),
        ]
    )
    def test_to_coo_returns_global_ids(self, _, layout: Literal["CSR", "CSC"]):
        edge_index = _edge_index_for_layout(layout)
        offset_topology = OffsetTopology(
            edge_index=edge_index,
            edge_ids=_EDGE_IDS,
            edge_weights=_EDGE_WEIGHTS,
            layout=layout,
            offset=_OFFSET,
            num_nodes=_NUM_NODES,
        )
        row, col, edge_ids, edge_weights = offset_topology.to_coo()
        assert edge_weights is not None
        actual_edges = collections.Counter(
            zip(row.tolist(), col.tolist(), edge_ids.tolist(), edge_weights.tolist())
        )
        expected_edges = collections.Counter(
            zip(
                edge_index[0].tolist(),
                edge_index[1].tolist(),
                _EDGE_IDS.tolist(),
                _EDGE_WEIGHTS.tolist(),
            )
        )
        self.assertEqual(actual_edges, expected_edges)

    @parameterized.expand(
        [
            param("CSR weighted", layout="CSR", with_weights=True),
            param("CSR unweighted", layout="CSR", with_weights=False),
            param("CSC weighted", layout="CSC", with_weights=True),
            param("CSC unweighted", layout="CSC", with_weights=False),
        ]
    )
    def test_empty_coo_builds_zero_indptr_of_the_partition_size(
        self, _, layout: Literal["CSR", "CSC"], with_weights: bool
    ):
        offset_topology = OffsetTopology(
            edge_index=torch.empty(2, 0, dtype=torch.int64),
            edge_ids=torch.empty(0, dtype=torch.int64),
            edge_weights=torch.empty(0) if with_weights else None,
            layout=layout,
            offset=_OFFSET,
            num_nodes=_NUM_NODES,
        )
        self.assert_tensor_equality(
            offset_topology.indptr, torch.zeros(_NUM_NODES + 1, dtype=torch.int64)
        )
        self.assertEqual(offset_topology.indices.numel(), 0)
        self.assertEqual(offset_topology.edge_ids.numel(), 0)
        if with_weights:
            assert offset_topology.edge_weights is not None
            self.assertEqual(offset_topology.edge_weights.numel(), 0)
        else:
            self.assertIsNone(offset_topology.edge_weights)
        row, col, _, _ = offset_topology.to_coo()
        self.assertEqual(row.numel(), 0)
        self.assertEqual(col.numel(), 0)

    def test_to_csr_and_to_csc_raise(self):
        offset_topology = OffsetTopology(
            edge_index=_edge_index_for_layout("CSR"),
            layout="CSR",
            offset=_OFFSET,
            num_nodes=_NUM_NODES,
        )
        with self.assertRaises(NotImplementedError):
            offset_topology.to_csr()
        with self.assertRaises(NotImplementedError):
            offset_topology.to_csc()

    def test_rejects_non_coo_input_layout(self):
        with self.assertRaises(ValueError):
            OffsetTopology(
                edge_index=_edge_index_for_layout("CSR"),
                input_layout="CSR",
                layout="CSR",
                offset=_OFFSET,
                num_nodes=_NUM_NODES,
            )

    @parameterized.expand(
        [
            param(
                "id below offset",
                compressed=torch.tensor([99, 100, 101]),
            ),
            param(
                "id at or above offset + num_nodes",
                compressed=torch.tensor([100, 101, 105]),
            ),
        ]
    )
    def test_rejects_out_of_range_compressed_ids(self, _, compressed: torch.Tensor):
        with self.assertRaises(ValueError):
            OffsetTopology(
                edge_index=torch.stack([compressed, torch.zeros_like(compressed)]),
                layout="CSR",
                offset=_OFFSET,
                num_nodes=_NUM_NODES,
            )

    def test_default_edge_ids_are_aranged_then_permuted_with_their_edges(self):
        compressed = torch.tensor([102, 100, 101])
        adjacent = torch.tensor([1, 2, 3])
        offset_topology = OffsetTopology(
            edge_index=torch.stack([compressed, adjacent]),
            layout="CSR",
            offset=_OFFSET,
            num_nodes=_NUM_NODES,
        )
        # Input edge k gets default edge id k; after the compressed-dimension
        # sort each neighbor keeps its original edge id.
        neighbor_to_edge_id = dict(
            zip(offset_topology.indices.tolist(), offset_topology.edge_ids.tolist())
        )
        self.assertEqual(neighbor_to_edge_id, {1: 0, 2: 1, 3: 2})


class ContainsOffsetTopologyTest(TestCase):
    def test_none_graph(self):
        self.assertFalse(contains_offset_topology(None))

    def test_homogeneous_graph(self):
        from graphlearn_torch.data import Graph

        stock_graph = Graph(
            Topology(edge_index=_edge_index_for_layout("CSR"), layout="CSR"), "CPU"
        )
        self.assertFalse(contains_offset_topology(stock_graph))

        offset_graph = Graph(
            OffsetTopology(
                edge_index=_edge_index_for_layout("CSR"),
                layout="CSR",
                offset=_OFFSET,
                num_nodes=_NUM_NODES,
            ),
            "CPU",
        )
        self.assertTrue(contains_offset_topology(offset_graph))

    def test_heterogeneous_graph(self):
        from graphlearn_torch.data import Graph

        stock_graph = Graph(
            Topology(edge_index=_edge_index_for_layout("CSR"), layout="CSR"), "CPU"
        )
        offset_graph = Graph(
            OffsetTopology(
                edge_index=_edge_index_for_layout("CSR"),
                layout="CSR",
                offset=_OFFSET,
                num_nodes=_NUM_NODES,
            ),
            "CPU",
        )
        self.assertFalse(contains_offset_topology({"a": stock_graph}))
        self.assertTrue(contains_offset_topology({"a": stock_graph, "b": offset_graph}))


if __name__ == "__main__":
    absltest.main()
