import unittest
from collections.abc import Mapping

import torch
from graphlearn_torch.data import Dataset, Topology
from parameterized import param, parameterized
from torch.testing import assert_close

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import DEFAULT_HOMOGENEOUS_EDGE_TYPE, to_heterogeneous_edge
from gigl.utils.data_splitters import (
    HashedNodeAnchorLinkSplitter,
    _check_edge_index,
    _check_val_test_percentage,
    _fast_hash,
    _get_padded_labels,
    get_labels_for_anchor_nodes,
    select_ssl_positive_label_edges,
)

# For TestDataSplitters
_NODE_A = NodeType("A")
_NODE_B = NodeType("B")
_NODE_C = NodeType("C")
_TO = Relation("to")

# For SelectSSLPositiveLabelEdgesTest
_NUM_EDGES = 1_000_000
_TEST_EDGE_INDEX = torch.arange(0, _NUM_EDGES * 2).reshape((2, _NUM_EDGES))
_INVALID_TEST_EDGE_INDEX = torch.arange(0, _NUM_EDGES * 10).reshape((10, _NUM_EDGES))


class TestDataSplitters(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Fast hash with int32",
                input_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
                expected_output=torch.tensor(
                    [[0, 1753845952], [697948427, 1408362973]], dtype=torch.int32
                ),
            ),
            param(
                "Fast hash with int64",
                input_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
                expected_output=torch.tensor(
                    [
                        [0, 6350654354804651301],
                        [2606959014078780554, 2185194620014831856],
                    ]
                ),
            ),
        ]
    )
    def test_fast_hash(
        self, _, input_tensor: torch.Tensor, expected_output: torch.Tensor
    ):
        actual = _fast_hash(input_tensor)
        assert_close(actual=actual, expected=expected_output)

    @parameterized.expand(
        [
            param(
                "Using src nodes",
                edges=torch.stack(
                    [
                        torch.arange(10, dtype=torch.int64),
                        torch.zeros(10, dtype=torch.int64),
                    ]
                ),
                sampling_direction="out",
                hash_function=lambda x: x,
                val_num=0.1,
                test_num=0.1,
                expected_train=torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64
                ),
                expected_val=torch.tensor([8], dtype=torch.int64),
                expected_test=torch.tensor([9], dtype=torch.int64),
            ),
            param(
                "Using dst nodes",
                edges=torch.stack(
                    [
                        torch.zeros(10, dtype=torch.int64),
                        torch.arange(10, dtype=torch.int64),
                    ]
                ),
                sampling_direction="in",
                hash_function=lambda x: x,
                val_num=0.1,
                test_num=0.1,
                expected_train=torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64
                ),
                expected_val=torch.tensor([8], dtype=torch.int64),
                expected_test=torch.tensor([9], dtype=torch.int64),
            ),
            param(
                "With dups",
                edges=torch.stack(
                    [
                        torch.cat(
                            [
                                torch.arange(10, dtype=torch.int64),
                                torch.arange(10, dtype=torch.int64),
                            ]
                        ),
                        torch.zeros(20, dtype=torch.int64),
                    ]
                ),
                sampling_direction="out",
                hash_function=lambda x: x,
                val_num=0.1,
                test_num=0.1,
                expected_train=torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64
                ),
                expected_val=torch.tensor([8], dtype=torch.int64),
                expected_test=torch.tensor([9], dtype=torch.int64),
            ),
            param(
                "Real hash fn",
                edges=torch.stack(
                    [
                        torch.zeros(20, dtype=torch.int64),
                        torch.arange(20, dtype=torch.int64),
                    ]
                ),
                sampling_direction="in",
                hash_function=_fast_hash,
                val_num=0.1,
                test_num=0.1,
                expected_train=torch.tensor(
                    [0, 18, 15, 10, 7, 19, 17, 3, 4, 2, 16, 14, 6, 11, 5, 13],
                    dtype=torch.int64,
                ),
                expected_val=torch.tensor([8, 1], dtype=torch.int64),
                expected_test=torch.tensor([9, 12], dtype=torch.int64),
            ),
            param(
                "With explicit val num",
                edges=torch.stack(
                    [
                        torch.arange(10, dtype=torch.int64),
                        torch.zeros(10, dtype=torch.int64),
                    ]
                ),
                sampling_direction="out",
                hash_function=lambda x: x,
                val_num=2,
                test_num=0.1,
                expected_train=torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64),
                expected_val=torch.tensor([7, 8], dtype=torch.int64),
                expected_test=torch.tensor([9], dtype=torch.int64),
            ),
            param(
                "With explicit test num",
                edges=torch.stack(
                    [
                        torch.arange(10, dtype=torch.int64),
                        torch.zeros(10, dtype=torch.int64),
                    ]
                ),
                sampling_direction="out",
                hash_function=lambda x: x,
                val_num=0.1,
                test_num=3,
                expected_train=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64),
                expected_val=torch.tensor([6], dtype=torch.int64),
                expected_test=torch.tensor([7, 8, 9], dtype=torch.int64),
            ),
            param(
                "With explicit val and test num",
                edges=torch.stack(
                    [
                        torch.arange(10, dtype=torch.int64),
                        torch.zeros(10, dtype=torch.int64),
                    ]
                ),
                sampling_direction="out",
                hash_function=lambda x: x,
                val_num=2,
                test_num=3,
                expected_train=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
                expected_val=torch.tensor([5, 6], dtype=torch.int64),
                expected_test=torch.tensor([7, 8, 9], dtype=torch.int64),
            ),
            param(
                "Start from non-zero",
                edges=torch.stack(
                    [
                        torch.arange(2, 22, 2, dtype=torch.int64),
                        torch.zeros(10, dtype=torch.int64),
                    ]
                ),
                sampling_direction="out",
                hash_function=lambda x: x,
                val_num=0.1,
                test_num=0.1,
                expected_train=torch.tensor(
                    [2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.int64
                ),
                expected_val=torch.tensor([18], dtype=torch.int64),
                expected_test=torch.tensor([20], dtype=torch.int64),
            ),
        ]
    )
    def test_node_based_link_splitter(
        self,
        _,
        edges,
        sampling_direction,
        hash_function,
        val_num,
        test_num,
        expected_train,
        expected_val,
        expected_test,
    ):
        splitter = HashedNodeAnchorLinkSplitter(
            sampling_direction=sampling_direction,
            hash_function=hash_function,
            num_val=val_num,
            num_test=test_num,
        )
        train, val, test = splitter(edges)

        assert_close(train, expected_train, rtol=0, atol=0)
        assert_close(val, expected_val, rtol=0, atol=0)
        assert_close(test, expected_test, rtol=0, atol=0)

    @parameterized.expand(
        [
            param(
                "One supervision edge type",
                edges={
                    EdgeType(_NODE_A, _TO, _NODE_B): torch.stack(
                        [
                            torch.zeros(10, dtype=torch.int64),
                            torch.arange(10, dtype=torch.int64),
                        ]
                    )
                },
                edge_types_to_split=[EdgeType(_NODE_A, _TO, _NODE_B)],
                hash_function=lambda x: x,
                val_num=1,
                test_num=1,
                expected={
                    _NODE_B: (
                        torch.arange(8, dtype=torch.int64),
                        torch.tensor([8], dtype=torch.int64),
                        torch.tensor([9], dtype=torch.int64),
                    )
                },
            ),
            param(
                "One supervision edge type, multiple edge types in graph",
                edges={
                    EdgeType(_NODE_A, _TO, _NODE_B): torch.stack(
                        [
                            torch.zeros(10, dtype=torch.int64),
                            torch.arange(10, dtype=torch.int64),
                        ]
                    ),
                    EdgeType(_NODE_A, _TO, _NODE_C): torch.stack(
                        [
                            torch.zeros(10, dtype=torch.int64),
                            torch.arange(10, 20, dtype=torch.int64),
                        ]
                    ),
                },
                edge_types_to_split=[
                    EdgeType(_NODE_A, _TO, _NODE_B),
                ],
                hash_function=lambda x: x,
                val_num=1,
                test_num=1,
                expected={
                    _NODE_B: (
                        torch.arange(8, dtype=torch.int64),
                        torch.tensor([8], dtype=torch.int64),
                        torch.tensor([9], dtype=torch.int64),
                    ),
                },
            ),
            param(
                "Multiple supervision edge types, mutliple target node types",
                edges={
                    EdgeType(_NODE_A, _TO, _NODE_B): torch.stack(
                        [
                            torch.zeros(10, dtype=torch.int64),
                            torch.arange(10, dtype=torch.int64),
                        ]
                    ),
                    EdgeType(_NODE_A, _TO, _NODE_C): torch.stack(
                        [
                            torch.zeros(10, dtype=torch.int64),
                            torch.arange(10, 20, dtype=torch.int64),
                        ]
                    ),
                },
                edge_types_to_split=[
                    EdgeType(_NODE_A, _TO, _NODE_B),
                    EdgeType(_NODE_A, _TO, _NODE_C),
                ],
                hash_function=lambda x: x,
                val_num=1,
                test_num=1,
                expected={
                    _NODE_B: (
                        torch.arange(8, dtype=torch.int64),
                        torch.tensor([8], dtype=torch.int64),
                        torch.tensor([9], dtype=torch.int64),
                    ),
                    _NODE_C: (
                        torch.arange(10, 18, dtype=torch.int64),
                        torch.tensor([18], dtype=torch.int64),
                        torch.tensor([19], dtype=torch.int64),
                    ),
                },
            ),
            param(
                "Multiple supervision edge types, one target node type",
                edges={
                    EdgeType(_NODE_B, _TO, _NODE_A): torch.stack(
                        [
                            torch.zeros(10, dtype=torch.int64),
                            torch.arange(10, dtype=torch.int64),
                        ]
                    ),
                    EdgeType(_NODE_C, _TO, _NODE_A): torch.stack(
                        [
                            torch.zeros(10, dtype=torch.int64),
                            torch.arange(10, 20, dtype=torch.int64),
                        ]
                    ),
                },
                edge_types_to_split=[
                    EdgeType(_NODE_B, _TO, _NODE_A),
                    EdgeType(_NODE_C, _TO, _NODE_A),
                ],
                hash_function=lambda x: x,
                val_num=0.1,
                test_num=0.1,
                expected={
                    _NODE_A: (
                        torch.arange(16, dtype=torch.int64),
                        torch.tensor([16, 17], dtype=torch.int64),
                        torch.tensor([18, 19], dtype=torch.int64),
                    ),
                },
            ),
            param(
                "Multiple supervision edge types, one target node type, dup nodes",
                edges={
                    EdgeType(_NODE_B, _TO, _NODE_A): torch.stack(
                        [
                            torch.zeros(10, dtype=torch.int64),
                            torch.arange(10, dtype=torch.int64),
                        ]
                    ),
                    EdgeType(_NODE_C, _TO, _NODE_A): torch.stack(
                        [
                            torch.zeros(10, dtype=torch.int64),
                            torch.arange(10, dtype=torch.int64),
                        ]
                    ),
                },
                edge_types_to_split=[
                    EdgeType(_NODE_B, _TO, _NODE_A),
                    EdgeType(_NODE_C, _TO, _NODE_A),
                ],
                hash_function=lambda x: x,
                val_num=0.1,
                test_num=0.1,
                expected={
                    _NODE_A: (
                        torch.arange(8, dtype=torch.int64),
                        torch.tensor([8], dtype=torch.int64),
                        torch.tensor([9], dtype=torch.int64),
                    ),
                },
            ),
            param(
                "Multiple supervision edge types, one target node type, different input shapes",
                edges={
                    EdgeType(_NODE_B, _TO, _NODE_A): torch.stack(
                        [
                            torch.zeros(10, dtype=torch.int64),
                            torch.arange(10, dtype=torch.int64),
                        ]
                    ),
                    EdgeType(_NODE_C, _TO, _NODE_A): torch.stack(
                        [
                            torch.zeros(2, dtype=torch.int64),
                            torch.arange(30, 32, dtype=torch.int64),
                        ]
                    ),
                },
                edge_types_to_split=[
                    EdgeType(_NODE_B, _TO, _NODE_A),
                    EdgeType(_NODE_C, _TO, _NODE_A),
                ],
                hash_function=lambda x: x,
                val_num=0.1,
                test_num=0.1,
                expected={
                    _NODE_A: (
                        torch.arange(10, dtype=torch.int64),
                        torch.tensor([30], dtype=torch.int64),
                        torch.tensor([31], dtype=torch.int64),
                    ),
                },
            ),
        ]
    )
    def test_node_based_link_splitter_heterogenous(
        self,
        _,
        edges,
        edge_types_to_split,
        hash_function,
        val_num,
        test_num,
        expected,
    ):
        splitter = HashedNodeAnchorLinkSplitter(
            sampling_direction="in",
            hash_function=hash_function,
            num_val=val_num,
            num_test=test_num,
            edge_types=edge_types_to_split,
        )
        split = splitter(edges)

        assert isinstance(split, Mapping)
        self.assertEqual(split.keys(), expected.keys())
        for node_type, (
            expected_train,
            expected_val,
            expected_test,
        ) in expected.items():
            train, val, test = split[node_type]
            assert_close(train, expected_train, rtol=0, atol=0)
            assert_close(val, expected_val, rtol=0, atol=0)
            assert_close(test, expected_test, rtol=0, atol=0)

    @parameterized.expand(
        [
            param(
                "No edges to split - empty",
                {EdgeType(_NODE_A, _TO, _NODE_B): torch.zeros(10, 2)},
                edge_types_to_split=[],
            ),
            param(
                "No edges to split - None",
                {EdgeType(_NODE_A, _TO, _NODE_B): torch.zeros(10, 2)},
                edge_types_to_split=None,
            ),
            param(
                "Edges not in map",
                {EdgeType(_NODE_A, _TO, _NODE_B): torch.zeros(10, 2)},
                edge_types_to_split=[EdgeType(_NODE_C, _TO, _NODE_A)],
            ),
        ]
    )
    def test_node_based_link_splitter_heterogenous_invalid(
        self,
        _,
        edges,
        edge_types_to_split,
    ):
        with self.assertRaises(ValueError):
            HashedNodeAnchorLinkSplitter(
                sampling_direction="in", edge_types=edge_types_to_split
            )(
                edge_index=edges,
            )

    def test_node_based_link_splitter_no_train_nodes(self):
        edges = torch.stack(
            [
                torch.zeros(10, dtype=torch.int64),
                torch.arange(10, dtype=torch.int64),
            ]
        )
        with self.assertRaises(ValueError):
            HashedNodeAnchorLinkSplitter(
                sampling_direction="in", num_val=5, num_test=5
            )(edges)

    @parameterized.expand(
        [
            param(
                "Too high train percentage", train_percentage=2.0, val_percentage=0.9
            ),
            param(
                "Too low train percentage", train_percentage=-0.2, val_percentage=0.9
            ),
            param("Too high val percentage", train_percentage=0.8, val_percentage=2.3),
            param("Negative val percentage", train_percentage=0.8, val_percentage=-1.0),
        ]
    )
    def test_check_val_test_percentage(self, _, train_percentage, val_percentage):
        with self.assertRaises(ValueError):
            _check_val_test_percentage(train_percentage, val_percentage)

    @parameterized.expand(
        [
            param("First dimension is not 2", edges=torch.zeros(3, 3)),
            param("Not two dimmensions", edges=torch.zeros(2)),
            param("Sparse tensor", edges=torch.zeros(2, 2).to_sparse()),
        ]
    )
    def test_check_edge_index(self, _, edges):
        with self.assertRaises(ValueError):
            _check_edge_index(edges)

    def test_get_labels_for_anchor_nodes(self):
        edges = torch.tensor(
            [
                [9, 10, 10, 10, 11, 11, 12, 12, 10],
                [8, 10, 11, 15, 12, 13, 10, 11, 7],
            ]
        )
        ds = Dataset()
        ds.init_graph(
            edge_index=to_heterogeneous_edge(edges),
            edge_ids=to_heterogeneous_edge(torch.arange(edges.shape[1])),
            graph_mode="CPU",
        )

        # TODO(kmonte): Update to use a splitter once we've migrated splitter API.
        node_ids = torch.tensor([9, 10, 11, 12])
        positive, negative = get_labels_for_anchor_nodes(
            dataset=ds,
            node_ids=node_ids,
            supervision_edge_types=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
        )
        expected_positive = torch.tensor(
            [
                [8, -1, -1, -1],  # node 9 labels
                [7, 10, 11, 15],  # node 10 labels
                [12, 13, -1, -1],  # node 11 labels
                [10, 11, -1, -1],  # node 12 labels
            ]
        )

        assert_close(positive, expected_positive, rtol=0, atol=0)
        self.assertIsNone(negative)

    def test_get_labels_for_anchor_nodes_heterogeneous(self):
        a_to_b = EdgeType(_NODE_A, _TO, _NODE_B)
        a_to_c = EdgeType(_NODE_A, _TO, _NODE_C)
        edges = {
            a_to_b: torch.tensor(
                [
                    [10, 11, 12, 13, 10],
                    [10, 10, 11, 11, 12],
                ]
            ),
            a_to_c: torch.tensor(
                [
                    [11, 12, 13],
                    [10, 10, 11],
                ]
            ),
        }
        ds = Dataset(edge_dir="in")
        ds.init_graph(
            edge_index=edges,
            edge_ids={e: torch.arange(edges[e].shape[1]) for e in edges},
            graph_mode="CPU",
        )
        # TODO(kmonte): Update to use a splitter once we've migrated splitter API.
        node_ids = torch.tensor([10])
        positive, negative = get_labels_for_anchor_nodes(
            dataset=ds, node_ids=node_ids, supervision_edge_types=a_to_b
        )
        expected = torch.tensor([[10, 11]])
        assert_close(positive, expected, rtol=0, atol=0)
        self.assertIsNone(negative)

    def test_get_labels_for_anchor_nodes_heterogeneous_positive_and_negative(self):
        a_to_b = EdgeType(_NODE_A, _TO, _NODE_B)
        a_to_c = EdgeType(_NODE_A, _TO, _NODE_C)
        edges = {
            a_to_b: torch.tensor(
                [
                    [10, 10, 11, 11, 12],
                    [10, 11, 12, 13, 10],
                ]
            ),
            a_to_c: torch.tensor(
                [
                    [10, 11, 12, 11],
                    [20, 30, 40, 50],
                ]
            ),
        }
        ds = Dataset()
        ds.init_graph(
            edge_index=edges,
            edge_ids={e: torch.arange(edges[e].shape[1]) for e in edges},
            graph_mode="CPU",
        )

        # TODO(kmonte): Update to use a splitter once we've migrated splitter API.
        node_ids = torch.tensor([10, 11])
        positive, negative = get_labels_for_anchor_nodes(
            dataset=ds,
            node_ids=node_ids,
            supervision_edge_types=(a_to_b, a_to_c),
        )

        expected_positive = torch.tensor([[10, 11], [12, 13]])
        expected_negative = torch.tensor([[20, -1], [30, 50]])
        assert_close(positive, expected_positive, rtol=0, atol=0)
        assert_close(negative, expected_negative, rtol=0, atol=0)

    @parameterized.expand(
        [
            param(
                "CSR",
                node_ids=torch.tensor([0, 1]),
                topo=Topology(
                    edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]), layout="CSR"
                ),
                expected=torch.tensor([[1, 2], [2, -1]]),
            ),
            param(
                "CSC",
                node_ids=torch.tensor([0, 1]),
                # Note: the edge index needs to be reversed for CSC...
                # Maybe this is a bug? Unclear how/if GLT uses this.
                topo=Topology(
                    edge_index=torch.tensor(
                        [
                            [1, 2, 0],
                            [0, 1, 1],
                        ]
                    ),
                    layout="CSC",
                ),
                expected=torch.tensor([[1, -1], [0, 2]]),
            ),
        ]
    )
    def test_get_padded_labels(self, _, node_ids, topo, expected):
        labels = _get_padded_labels(node_ids, topo)
        assert_close(labels, expected, rtol=0, atol=0)


class SelectSSLPositiveLabelEdgesTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Test positive label selection",
                positive_label_percentage=0.1,
                expected_num_labels=100_000,
            ),
            param(
                "Test zero positive label selection",
                positive_label_percentage=0,
                expected_num_labels=0,
            ),
            param(
                "Test all positive label selection",
                positive_label_percentage=1,
                expected_num_labels=1_000_000,
            ),
        ]
    )
    def test_valid_label_selection(
        self, _, positive_label_percentage: float, expected_num_labels: int
    ):
        labels = select_ssl_positive_label_edges(
            edge_index=_TEST_EDGE_INDEX,
            positive_label_percentage=positive_label_percentage,
        )
        self.assertEqual(labels.size(1), expected_num_labels)

    @parameterized.expand(
        [
            param(
                "Test invalid edge index",
                edge_index=_INVALID_TEST_EDGE_INDEX,
                positive_label_percentage=0.1,
            ),
            param(
                "Test negative positive label percentage",
                edge_index=_TEST_EDGE_INDEX,
                positive_label_percentage=-0.1,
            ),
            param(
                "Test positive label percentage greater than 1",
                edge_index=_TEST_EDGE_INDEX,
                positive_label_percentage=1.1,
            ),
        ]
    )
    def test_invalid_label_selection(
        self, _, edge_index: torch.Tensor, positive_label_percentage: float
    ):
        with self.assertRaises(ValueError):
            select_ssl_positive_label_edges(
                edge_index=edge_index,
                positive_label_percentage=positive_label_percentage,
            )


if __name__ == "__main__":
    unittest.main()
