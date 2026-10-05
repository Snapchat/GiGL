"""The compact build must sample the same neighbors as GLT for graphs without edge metadata.

The consumer is a compiled extension, so these compare sampled neighbourhoods against GLT's
own build rather than asserting hand-written expectations.
"""

import weakref
from typing import Literal, Union
from unittest import mock

import torch
from graphlearn_torch.data import Graph, Topology
from graphlearn_torch.partition import RangePartitionBook
from parameterized import param, parameterized

from gigl.distributed.dist_dataset import DistDataset, _has_per_edge_metadata
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import GraphPartitionData, PartitionOutput
from gigl.utils.csr import CompactTopology
from gigl.utils.data_splitters import _get_padded_labels
from tests.test_assets.test_case import TestCase

_EDGE_TYPE = EdgeType(NodeType("user"), Relation("to"), NodeType("item"))
_OTHER_EDGE_TYPE = EdgeType(NodeType("item"), Relation("to"), NodeType("user"))
_ONE_PARTITION = RangePartitionBook(partition_ranges=[(0, 40)], partition_idx=0)
_ONE_PARTITION_BY_TYPE = {
    NodeType("user"): _ONE_PARTITION,
    NodeType("item"): _ONE_PARTITION,
}


def _random_coo(num_nodes: int, num_edges: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, num_nodes, (2, num_edges), generator=generator, dtype=torch.int64
    )


class CompactTopologyTest(TestCase):
    def test_it_has_exactly_the_attributes_of_a_real_topology(self) -> None:
        """``__init__`` skips GLT's, so a field GLT adds later would otherwise be missed."""
        coo = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        self.assertEqual(
            set(CompactTopology(coo, layout="CSR").__dict__),
            set(Topology(edge_index=coo, layout="CSR").__dict__),
        )

    def test_no_edge_ids_are_fabricated(self) -> None:
        """``Topology.__init__`` would allocate ``arange(num_edges)`` here; that is the point."""
        coo = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        self.assertIsNone(CompactTopology(coo, layout="CSR").edge_ids)

    def test_a_float_edge_index_is_cast_as_glt_does(self) -> None:
        coo = torch.tensor([[0.0, 2.0, 2.0], [1.0, 0.0, 1.0]])
        lean = CompactTopology(coo, layout="CSR")
        reference = Topology(edge_index=coo, layout="CSR")
        torch.testing.assert_close(lean.indptr, reference.indptr, rtol=0, atol=0)
        torch.testing.assert_close(lean.indices, reference.indices, rtol=0, atol=0)

    def test_an_edge_type_with_no_local_edges_builds_and_samples_nothing(self) -> None:
        """The hash partitioner hands such an edge type over as a float ``(2, 0)`` placeholder."""
        from graphlearn_torch import py_graphlearn_torch as pywrap

        graph = Graph(CompactTopology(torch.empty((2, 0)), layout="CSC"), "CPU", None)
        graph.lazy_init()

        neighbors, counts = pywrap.CPURandomSampler(graph.graph_handler).sample(
            torch.tensor([0, 7], dtype=torch.int64), 4
        )
        self.assertEqual(counts.tolist(), [0, 0])
        self.assertEqual(neighbors.numel(), 0)


class PerEdgeMetadataTest(TestCase):
    @parameterized.expand(
        [
            param(
                "nothing_per_edge",
                edge_ids=None,
                edge_weights=None,
                edge_features_registered=False,
                expected=False,
            ),
            param(
                "dict_of_none_ids",
                edge_ids={_EDGE_TYPE: None},
                edge_weights=None,
                edge_features_registered=False,
                expected=False,
            ),
            param(
                "empty_placeholder_ids",
                edge_ids={_EDGE_TYPE: torch.empty(0), _OTHER_EDGE_TYPE: None},
                edge_weights=None,
                edge_features_registered=False,
                expected=False,
            ),
            param(
                "homogeneous_empty_placeholder_ids",
                edge_ids=torch.empty(0),
                edge_weights=None,
                edge_features_registered=False,
                expected=False,
            ),
            param(
                "materialized_ids",
                edge_ids={_EDGE_TYPE: torch.tensor([0, 1])},
                edge_weights=None,
                edge_features_registered=False,
                expected=True,
            ),
            param(
                "edge_weights",
                edge_ids=None,
                edge_weights=torch.tensor([1.0, 1.0]),
                edge_features_registered=False,
                expected=True,
            ),
            param(
                "edge_features",
                edge_ids=None,
                edge_weights=None,
                edge_features_registered=True,
                expected=True,
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
            _has_per_edge_metadata(
                edge_ids=edge_ids,
                edge_weights=edge_weights,
                edge_features_registered=edge_features_registered,
            ),
            expected,
        )


class InitializeGraphTest(TestCase):
    """Drives the real ``_initialize_graph``, which helper-level tests do not reach.

    ``GraphPartitionData`` is a frozen dataclass, so anything this path tries to write back to it
    raises only when the method itself runs.
    """

    def _dataset(self) -> DistDataset:
        return DistDataset(rank=0, world_size=1, edge_dir="in")

    def test_a_homogeneous_partition_builds_a_bare_graph(self) -> None:
        num_nodes = 40
        dataset = self._dataset()

        dataset._initialize_graph(
            partitioned_edge_index=GraphPartitionData(
                edge_index=_random_coo(num_nodes, 200, seed=3), edge_ids=None
            ),
            node_partition_book=_ONE_PARTITION,
            edge_features_registered=False,
        )

        self.assertIsInstance(dataset.graph, Graph)

    def test_registered_edge_features_keep_the_edge_ids_the_sampler_reads(self) -> None:
        """Edge-feature lookups need ids, which the compact path does not retain."""
        num_nodes = 40
        dataset = self._dataset()

        dataset._initialize_graph(
            partitioned_edge_index=GraphPartitionData(
                edge_index=_random_coo(num_nodes, 200, seed=7), edge_ids=None
            ),
            node_partition_book=_ONE_PARTITION,
            edge_features_registered=True,
        )

        assert isinstance(dataset.graph, Graph)
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
            node_partition_book=_ONE_PARTITION_BY_TYPE,
            edge_features_registered=False,
        )

        assert isinstance(dataset.graph, dict)
        self.assertEqual(set(dataset.graph.keys()), {_EDGE_TYPE})
        self.assertIsInstance(dataset.graph[_EDGE_TYPE], Graph)

    def test_an_edge_type_with_no_local_edges_keeps_the_lean_path(self) -> None:
        """The hash partitioner's placeholder for it: a float ``(2, 0)`` COO and empty edge ids."""
        dataset = self._dataset()

        dataset._initialize_graph(
            partitioned_edge_index={
                _EDGE_TYPE: GraphPartitionData(
                    edge_index=_random_coo(40, 200, seed=9), edge_ids=None
                ),
                _OTHER_EDGE_TYPE: GraphPartitionData(
                    edge_index=torch.empty((2, 0)), edge_ids=torch.empty(0)
                ),
            },
            node_partition_book=_ONE_PARTITION_BY_TYPE,
            edge_features_registered=False,
        )

        assert isinstance(dataset.graph, dict)
        self.assertIsInstance(dataset.graph[_OTHER_EDGE_TYPE].topo, CompactTopology)

    @parameterized.expand(
        [
            param("range_out", edge_dir="out", tensor_book=False, expected_rows=50),
            param("range_in", edge_dir="in", tensor_book=False, expected_rows=10),
            param("tensor_out", edge_dir="out", tensor_book=True, expected_rows=100),
            param("tensor_in", edge_dir="in", tensor_book=True, expected_rows=20),
        ]
    )
    def test_indptr_covers_every_id_this_rank_is_asked_about(
        self,
        _name: str,
        edge_dir: Literal["in", "out"],
        tensor_book: bool,
        expected_rows: int,
    ) -> None:
        """Rank 0's range end for a range book, the global count for a tensor book.

        ``out`` compresses sources (user, 100 nodes), ``in`` destinations (item, 20).
        """
        books: dict[NodeType, Union[torch.Tensor, RangePartitionBook]] = (
            {NodeType("user"): torch.zeros(100), NodeType("item"): torch.zeros(20)}
            if tensor_book
            else {
                NodeType("user"): RangePartitionBook(
                    partition_ranges=[(0, 50), (50, 100)], partition_idx=0
                ),
                NodeType("item"): RangePartitionBook(
                    partition_ranges=[(0, 10), (10, 20)], partition_idx=0
                ),
            }
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir=edge_dir)

        dataset._initialize_graph(
            partitioned_edge_index={
                _EDGE_TYPE: GraphPartitionData(
                    edge_index=torch.tensor([[0, 3], [1, 2]]), edge_ids=None
                )
            },
            node_partition_book=books,
            edge_features_registered=False,
        )

        assert isinstance(dataset.graph, dict)
        self.assertEqual(
            dataset.graph[_EDGE_TYPE].topo.indptr.numel(), expected_rows + 1
        )

    def test_label_lookups_stay_aligned_for_anchors_past_the_last_label(self) -> None:
        """GLT's max(row) + 1 sizing put anchor 0's label on anchor 9 here."""
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        book = RangePartitionBook(partition_ranges=[(0, 10), (10, 20)], partition_idx=0)

        dataset._initialize_graph(
            partitioned_edge_index={
                _EDGE_TYPE: GraphPartitionData(
                    edge_index=torch.tensor([[0], [3]]), edge_ids=None
                )
            },
            node_partition_book={NodeType("user"): book, NodeType("item"): book},
            edge_features_registered=False,
        )

        assert isinstance(dataset.graph, dict)
        labels = _get_padded_labels(
            torch.tensor([9, 0]),
            dataset.graph[_EDGE_TYPE].topo,
            allow_non_existant_node_ids=True,
        )
        self.assertEqual(labels.tolist(), [[-1], [3]])

    def test_degrees_cover_trailing_nodes_with_no_edges(self) -> None:
        """PPR looks up the degree of every node it visits, including sinks past the last row."""
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")

        dataset._initialize_graph(
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[0], [9]]), edge_ids=None
            ),
            node_partition_book=torch.zeros(10),
            edge_features_registered=False,
        )

        assert isinstance(dataset.graph, Graph)
        self.assertEqual(dataset.graph.topo.degrees.tolist(), [1] + [0] * 9)


class BuildReleasesEachCooTest(TestCase):
    def test_each_coo_is_freed_before_the_next_edge_type_is_built(self) -> None:
        """Through ``build()`` with an edge splitter, whose local once kept every COO alive."""
        user, item = NodeType("user"), NodeType("item")
        larger = EdgeType(user, Relation("to"), item)
        smaller = EdgeType(item, Relation("to"), user)
        book = RangePartitionBook(partition_ranges=[(0, 40)], partition_idx=0)
        coos = {
            larger: _random_coo(40, 200, seed=1),
            smaller: _random_coo(40, 50, seed=2),
        }
        refs = {edge_type: weakref.ref(coo) for edge_type, coo in coos.items()}
        partition_output = PartitionOutput(
            node_partition_book={user: book, item: book},
            edge_partition_book=None,
            partitioned_edge_index={
                edge_type: GraphPartitionData(edge_index=coo, edge_ids=None)
                for edge_type, coo in coos.items()
            },
            partitioned_node_features=None,
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        del coos

        alive_at_each_build: list[set[EdgeType]] = []

        def recording_topology(edge_index, layout, num_rows):
            alive_at_each_build.append(
                {edge_type for edge_type, ref in refs.items() if ref() is not None}
            )
            return CompactTopology(edge_index, layout=layout, num_rows=num_rows)

        class _EdgeSplitter:
            should_convert_labels_to_edges = False

            def __call__(self, edge_index):
                ids = torch.arange(40)
                return {
                    node_type: (ids, ids[:0], ids[:0]) for node_type in (user, item)
                }

        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        # A plain function, not a Mock: a Mock records its call args and would keep each COO alive.
        with mock.patch(
            "gigl.distributed.dist_dataset.CompactTopology", new=recording_topology
        ):
            dataset.build(partition_output, splitter=_EdgeSplitter())

        # Largest first, so the smaller COO is the only one left when its turn comes.
        self.assertEqual(alive_at_each_build, [{larger, smaller}, {smaller}])


class SamplingParityTest(TestCase):
    """A graph built the lean way must sample identically to one built by GLT."""

    @parameterized.expand(
        [
            param("csr_out", layout="CSR"),
            param("csc_in", layout="CSC"),
        ]
    )
    def test_full_fanout_sampling_matches_glt(
        self, _name: str, layout: Literal["CSR", "CSC"]
    ) -> None:
        from graphlearn_torch import py_graphlearn_torch as pywrap

        num_nodes = 500
        coo = _random_coo(num_nodes, 4_000, seed=11)

        lean = Graph(CompactTopology(coo, layout=layout), "CPU", None)
        lean.lazy_init()

        reference = Graph(Topology(edge_index=coo, layout=layout), "CPU", None)
        reference.lazy_init()

        torch.testing.assert_close(
            lean.topo.indptr, reference.topo.indptr, rtol=0, atol=0
        )
        torch.testing.assert_close(
            lean.topo.indices, reference.topo.indices, rtol=0, atol=0
        )

        # Seeds past the last row get no neighbours rather than reading out of bounds.
        seeds = torch.arange(num_nodes + 10, dtype=torch.int64)
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
