import unittest
from collections.abc import Mapping

import torch
import torch.multiprocessing as mp
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
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    get_process_group_init_method,
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


def _run_splitter_distributed(
    process_num: int,
    world_size: int,
    init_method: str,
    edges: list[torch.Tensor],
    expected: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
):
    """Run the splitter in a distributed setting and check the results.
    Args:
        process_num (int): The rank of the current process.
        world_size (int): Total number of processes.
        init_method (str): The method to initialize the process group.
        edges (list[torch.Tensor]): List of edge tensors for each process.
        expected (list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Expected train, val, test splits for each process.
    """
    torch.distributed.init_process_group(
        rank=process_num, world_size=world_size, init_method=init_method
    )
    splitter = HashedNodeAnchorLinkSplitter(
        sampling_direction="out",
        should_convert_labels_to_edges=False,
        hash_function=lambda x: x,  # Using identity hash function for simplicity
    )

    train, val, test = splitter(edges[process_num])
    expected_train, expected_val, expected_test = expected[process_num]
    assert_tensor_equality(train, expected_train)
    assert_tensor_equality(val, expected_val)
    assert_tensor_equality(test, expected_test)


class TestDataSplitters(unittest.TestCase):
    def tearDown(self):
        if torch.distributed.is_initialized():
            print("Destroying process group")
            # Ensure the process group is destroyed after each test
            # to avoid interference with subsequent tests
            torch.distributed.destroy_process_group()
        super().tearDown()

    @parameterized.expand(
        [
            param(
                "Fast hash with int32",
                input_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
                expected_output=torch.tensor(
                    [[1492470133, 1071609072], [81290992, 325464930]], dtype=torch.int32
                ),
            ),
            param(
                "Fast hash with int64",
                input_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
                expected_output=torch.tensor(
                    [
                        [1622107259858988186, 3834912982681189024],
                        [2886753494712499930, 5597559336455034305],
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
                    [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 16, 17, 19],
                    dtype=torch.int64,
                ),
                expected_val=torch.tensor([9, 14], dtype=torch.int64),
                expected_test=torch.tensor([7, 15, 18], dtype=torch.int64),
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
        torch.distributed.init_process_group(
            rank=0, world_size=1, init_method=get_process_group_init_method()
        )
        splitter = HashedNodeAnchorLinkSplitter(
            sampling_direction=sampling_direction,
            hash_function=hash_function,
            num_val=val_num,
            num_test=test_num,
            should_convert_labels_to_edges=False,
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
                val_num=0.1,
                test_num=0.1,
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
                val_num=0.1,
                test_num=0.1,
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
                            torch.zeros(20, dtype=torch.int64),
                            torch.arange(20, dtype=torch.int64),
                        ]
                    ),
                },
                edge_types_to_split=[
                    EdgeType(_NODE_A, _TO, _NODE_B),
                    EdgeType(_NODE_A, _TO, _NODE_C),
                ],
                hash_function=lambda x: x,
                val_num=0.1,
                test_num=0.1,
                expected={
                    _NODE_B: (
                        torch.arange(8, dtype=torch.int64),
                        torch.tensor([8], dtype=torch.int64),
                        torch.tensor([9], dtype=torch.int64),
                    ),
                    _NODE_C: (
                        torch.arange(16, dtype=torch.int64),
                        torch.tensor([16, 17], dtype=torch.int64),
                        torch.tensor([18, 19], dtype=torch.int64),
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
                            torch.arange(10, 12, dtype=torch.int64),
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
                        torch.arange(9, dtype=torch.int64),
                        torch.tensor([9], dtype=torch.int64),
                        torch.tensor([10, 11], dtype=torch.int64),
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
        torch.distributed.init_process_group(
            rank=0, world_size=1, init_method=get_process_group_init_method()
        )

        splitter = HashedNodeAnchorLinkSplitter(
            sampling_direction="in",
            hash_function=hash_function,
            num_val=val_num,
            num_test=test_num,
            supervision_edge_types=edge_types_to_split,
            should_convert_labels_to_edges=False,
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

    def test_node_based_link_splitter_parallelized(self):
        init_method = get_process_group_init_method()
        edges = [
            torch.stack(
                [
                    torch.arange(0, 20, dtype=torch.int64),
                    torch.ones(20, dtype=torch.int64),
                ]
            ),
            torch.stack(
                [
                    torch.arange(0, 10, dtype=torch.int64),
                    torch.zeros(10, dtype=torch.int64),
                ]
            ),
            torch.stack(
                [
                    torch.arange(0, 20, 2, dtype=torch.int64),
                    torch.zeros(10, dtype=torch.int64),
                ]
            ),
        ]
        # We need to guarantee that a given node id is always selected into the same split, on every rank.
        # _run_splitter_distributed uses the identity hash function, so we can reason about hash values easily.
        # The way HashedNodeAnchorLinkSplitter works is that it hashes the node ids, and then splits them into train, val, test based on the hash value.
        # The splitting is done by first normalizing the hash values to [0, 1], and then selecting the train, val, test splits based on the provided percentages.
        # e.g. test = normalized_hash_value >= (1 - test_percentage)
        expected_splits = [
            (
                torch.arange(16, dtype=torch.int64),
                torch.tensor([16, 17], dtype=torch.int64),
                torch.tensor([18, 19], dtype=torch.int64),
            ),
            (
                # For process 1, all of the nodes would be selected into train for this example,
                # As the identity hash does not mix, and on process 0 [0, 9] are all train too.
                torch.arange(10, dtype=torch.int64),
                torch.tensor([], dtype=torch.int64),
                torch.tensor([], dtype=torch.int64),
            ),
            (
                torch.arange(0, 16, 2, dtype=torch.int64),
                torch.tensor([16], dtype=torch.int64),
                torch.tensor([18], dtype=torch.int64),
            ),
        ]
        # Run the splitter in parallel
        mp.spawn(
            _run_splitter_distributed,
            args=(
                3,  # world_size
                init_method,  # init_method
                edges,  # edges
                expected_splits,  # expected
            ),
            nprocs=3,
            join=True,
        )

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
            splitter = HashedNodeAnchorLinkSplitter(
                sampling_direction="in",
                supervision_edge_types=edge_types_to_split,
                should_convert_labels_to_edges=False,
            )
            splitter(edge_index=edges)

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

    def test_hashed_node_anchor_link_splitter_requires_process_group(self):
        edges = torch.stack(
            [
                torch.arange(0, 40, 2, dtype=torch.int64),
                torch.zeros(20, dtype=torch.int64),
            ]
        )
        splitter = HashedNodeAnchorLinkSplitter(
            sampling_direction="out",
            should_convert_labels_to_edges=False,
        )
        with self.assertRaises(RuntimeError):
            splitter(edges)

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
            positive_label_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
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
            dataset=ds, node_ids=node_ids, positive_label_edge_type=a_to_b
        )
        expected = torch.tensor([[10, 11]], dtype=torch.int64)
        assert_close(positive, expected, rtol=0, atol=0)
        self.assertIsNone(negative)

    def test_get_labels_for_anchor_nodes_heterogeneous_positive_and_negative(self):
        a_to_b = EdgeType(_NODE_A, _TO, _NODE_B)
        a_to_c = EdgeType(_NODE_A, _TO, _NODE_C)
        edges = {
            a_to_b: torch.tensor(
                [
                    [10, 10, 11, 11, 12, 13],
                    [10, 11, 12, 13, 10, 10],
                ],
                dtype=torch.int64,
            ),
            a_to_c: torch.tensor(
                [
                    [10, 11, 12, 11],
                    [20, 30, 40, 50],
                ],
                dtype=torch.int64,
            ),
        }
        ds = Dataset()
        ds.init_graph(
            edge_index=edges,
            edge_ids={e: torch.arange(edges[e].shape[1]) for e in edges},
            graph_mode="CPU",
        )

        # TODO(kmonte): Update to use a splitter once we've migrated splitter API.
        node_ids = torch.tensor([10, 11, 13])
        positive, negative = get_labels_for_anchor_nodes(
            dataset=ds,
            node_ids=node_ids,
            positive_label_edge_type=a_to_b,
            negative_label_edge_type=a_to_c,
        )
        # "DST" nodes for our anchor nodes (10, 11, 13).
        expected_positive = torch.tensor(
            [[10, 11], [12, 13], [10, -1]], dtype=torch.int64
        )
        expected_negative = torch.tensor(
            [[20, -1], [30, 50], [-1, -1]], dtype=torch.int64
        )
        assert_close(positive, expected_positive, rtol=0, atol=0)
        assert_close(negative, expected_negative, rtol=0, atol=0)

    @parameterized.expand(
        [
            param(
                "CSR",
                node_ids=torch.tensor([0, 1]),
                topo=Topology(
                    edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.int64),
                    layout="CSR",
                ),
                expected=torch.tensor([[1, 2], [2, -1]], dtype=torch.int64),
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
                        ],
                        dtype=torch.int64,
                    ),
                    layout="CSC",
                ),
                expected=torch.tensor([[1, -1], [0, 2]], dtype=torch.int64),
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
