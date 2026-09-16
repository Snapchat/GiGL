"""The lean graph build must produce a graph indistinguishable from ``glt.Dataset.init_graph``.

The consumer is a compiled extension that segfaults rather than raises on a malformed topology,
so these compare sampled neighbourhoods against GLT's own build rather than asserting
hand-written expectations.
"""

import torch
from graphlearn_torch.data import Graph, Topology
from graphlearn_torch.partition import RangePartitionBook
from parameterized import param, parameterized

from gigl.distributed.dist_dataset import (
    DistDataset,
    _build_topology_without_edge_ids,
    _can_build_topology_directly,
    _num_nodes_for_row_dimension,
    _reference_topology_attributes,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import GraphPartitionData
from tests.test_assets.test_case import TestCase

_EDGE_TYPE = EdgeType(NodeType("user"), Relation("to"), NodeType("item"))


def _random_coo(num_nodes: int, num_edges: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, num_nodes, (2, num_edges), generator=generator, dtype=torch.int64
    )


class TopologyBypassTest(TestCase):
    def test_it_populates_exactly_what_a_real_topology_has(self) -> None:
        topology = _build_topology_without_edge_ids(
            indptr=torch.tensor([0, 2, 3], dtype=torch.int64),
            indices=torch.tensor([1, 0, 1], dtype=torch.int64),
            layout="CSR",
        )
        self.assertEqual(set(topology.__dict__), set(_reference_topology_attributes()))

    def test_no_edge_ids_are_fabricated(self) -> None:
        """``Topology.__init__`` would allocate ``arange(num_edges)`` here; that is the point."""
        topology = _build_topology_without_edge_ids(
            indptr=torch.tensor([0, 2, 3], dtype=torch.int64),
            indices=torch.tensor([1, 0, 1], dtype=torch.int64),
            layout="CSR",
        )
        self.assertIsNone(topology._edge_ids)

    def test_a_missing_attribute_is_reported_rather_than_segfaulting(self) -> None:
        import gigl.distributed.dist_dataset as dist_dataset

        original = dist_dataset._reference_topology_attributes
        dist_dataset._reference_topology_attributes = lambda: frozenset(
            {"_layout", "_indptr", "_indices", "_edge_ids", "_edge_weights", "_new"}
        )
        try:
            with self.assertRaises(AttributeError):
                _build_topology_without_edge_ids(
                    indptr=torch.tensor([0, 1], dtype=torch.int64),
                    indices=torch.tensor([0], dtype=torch.int64),
                    layout="CSR",
                )
        finally:
            dist_dataset._reference_topology_attributes = original


class LeanBuildAppliesTest(TestCase):
    @parameterized.expand(
        [
            param(
                "no_ids_no_weights",
                edge_ids=None,
                edge_weights=None,
                edge_features_registered=False,
                expected=True,
            ),
            param(
                "dict_of_none_ids",
                edge_ids={_EDGE_TYPE: None},
                edge_weights=None,
                edge_features_registered=False,
                expected=True,
            ),
            param(
                "materialized_ids",
                edge_ids={_EDGE_TYPE: torch.tensor([0, 1])},
                edge_weights=None,
                edge_features_registered=False,
                expected=False,
            ),
            param(
                "edge_weights",
                edge_ids=None,
                edge_weights=torch.tensor([1.0, 1.0]),
                edge_features_registered=False,
                expected=False,
            ),
            param(
                "edge_features",
                edge_ids=None,
                edge_weights=None,
                edge_features_registered=True,
                expected=False,
            ),
        ]
    )
    def test_gating(
        self,
        _name: str,
        edge_ids,
        edge_weights,
        edge_features_registered: bool,
        expected: bool,
    ) -> None:
        self.assertEqual(
            _can_build_topology_directly(
                edge_ids=edge_ids,
                edge_weights=edge_weights,
                edge_features_registered=edge_features_registered,
            ),
            expected,
        )


class RowDimensionTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.books = {
            NodeType("user"): RangePartitionBook(
                partition_ranges=[(0, 50), (50, 100)], partition_idx=0
            ),
            NodeType("item"): RangePartitionBook(
                partition_ranges=[(0, 10), (10, 20)], partition_idx=0
            ),
        }

    def test_csr_sizes_from_the_source_node_type(self) -> None:
        self.assertEqual(
            _num_nodes_for_row_dimension(_EDGE_TYPE, "CSR", self.books), 100
        )

    def test_csc_sizes_from_the_destination_node_type(self) -> None:
        """CSC compresses columns, which are destination nodes -- the layout GiGL uses."""
        self.assertEqual(
            _num_nodes_for_row_dimension(_EDGE_TYPE, "CSC", self.books), 20
        )

    def test_a_missing_partition_book_is_named(self) -> None:
        with self.assertRaises(ValueError):
            _num_nodes_for_row_dimension(
                EdgeType(NodeType("absent"), Relation("to"), NodeType("item")),
                "CSR",
                self.books,
            )


class InitializeGraphTest(TestCase):
    """Drives the real ``_initialize_graph``, which helper-level tests do not reach.

    ``GraphPartitionData`` is a frozen dataclass, so anything this path tries to write back to it
    raises only when the method itself runs.
    """

    def _dataset(self) -> DistDataset:
        dataset = DistDataset.__new__(DistDataset)
        dataset.edge_dir = "in"
        return dataset

    def _partition_book(self, num_nodes: int) -> RangePartitionBook:
        return RangePartitionBook(
            partition_ranges=[(0, num_nodes // 2), (num_nodes // 2, num_nodes)],
            partition_idx=0,
        )

    def test_a_homogeneous_partition_builds_a_bare_graph(self) -> None:
        num_nodes = 40
        dataset = self._dataset()

        dataset._initialize_graph(
            partitioned_edge_index=GraphPartitionData(
                edge_index=_random_coo(num_nodes, 200, seed=3), edge_ids=None
            ),
            node_partition_book=self._partition_book(num_nodes),
            edge_features_registered=False,
        )

        self.assertIsInstance(dataset.graph, Graph)

    def test_registered_edge_features_keep_the_edge_ids_the_sampler_reads(self) -> None:
        """Edge features are looked up by edge id, and the lean path does not materialize any.

        GLT hands ``torch.empty(0)`` to the compiled graph when ``Topology.edge_ids`` is unset, so
        taking the lean path here segfaults the sampler rather than raising.
        """
        num_nodes = 40
        dataset = self._dataset()

        dataset._initialize_graph(
            partitioned_edge_index=GraphPartitionData(
                edge_index=_random_coo(num_nodes, 200, seed=7), edge_ids=None
            ),
            node_partition_book=self._partition_book(num_nodes),
            edge_features_registered=True,
        )

        self.assertIsNotNone(dataset.graph.topo.edge_ids)

    def test_a_heterogeneous_partition_builds_one_graph_per_edge_type(self) -> None:
        num_nodes = 40
        dataset = self._dataset()

        dataset._initialize_graph(
            partitioned_edge_index={
                _EDGE_TYPE: GraphPartitionData(
                    edge_index=_random_coo(num_nodes, 200, seed=5), edge_ids=None
                )
            },
            node_partition_book={
                NodeType("user"): self._partition_book(num_nodes),
                NodeType("item"): self._partition_book(num_nodes),
            },
            edge_features_registered=False,
        )

        self.assertEqual(set(dataset.graph.keys()), {_EDGE_TYPE})
        self.assertIsInstance(dataset.graph[_EDGE_TYPE], Graph)


class SamplingParityTest(TestCase):
    """A graph built the lean way must sample identically to one built by GLT."""

    @parameterized.expand(
        [
            param("csr_out", layout="CSR"),
            param("csc_in", layout="CSC"),
        ]
    )
    def test_full_fanout_sampling_matches_glt(self, _name: str, layout: str) -> None:
        from graphlearn_torch import py_graphlearn_torch as pywrap

        from gigl.utils.csr import build_csr_from_coo

        num_nodes = 500
        coo = _random_coo(num_nodes, 4_000, seed=11)
        group_by, other = (0, 1) if layout == "CSR" else (1, 0)

        indptr, indices = build_csr_from_coo(
            row=coo[group_by], col=coo[other], num_rows=num_nodes
        )
        lean = Graph(
            _build_topology_without_edge_ids(
                indptr=indptr, indices=indices, layout=layout
            ),
            "CPU",
            None,
        )
        lean.lazy_init()

        reference = Graph(Topology(edge_index=coo, layout=layout), "CPU", None)
        reference.lazy_init()

        seeds = torch.arange(num_nodes, dtype=torch.int64)
        # Full fanout makes the sampler copy every neighbour instead of drawing, so this is exact.
        lean_neighbors, lean_counts = pywrap.CPURandomSampler(
            lean.graph_handler
        ).sample(seeds, 1_000)
        reference_neighbors, reference_counts = pywrap.CPURandomSampler(
            reference.graph_handler
        ).sample(seeds, 1_000)

        torch.testing.assert_close(lean_counts, reference_counts, rtol=0, atol=0)
        torch.testing.assert_close(lean_neighbors, reference_neighbors, rtol=0, atol=0)
        self.assertGreater(lean_neighbors.numel(), 0, "fixture sampled nothing")


if __name__ == "__main__":
    from absl.testing import absltest

    absltest.main()
