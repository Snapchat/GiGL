import os
import unittest
from collections import defaultdict
from typing import Literal, Optional, Union

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from graphlearn_torch.utils import reverse_edge_type
from parameterized import param, parameterized
from torch_geometric.data import Data, HeteroData

from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_ablp_neighborloader import AnchorLabels, DistABLPLoader
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_partitioner import DistPartitioner
from gigl.distributed.dist_range_partitioner import DistRangePartitioner
from gigl.distributed.utils.neighborloader import (
    ABLP_LABEL_FORMAT_ENV_VAR,
    COLLATE_IMPL_ENV_VAR,
)
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
    DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
    HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    GraphPartitionData,
    PartitionOutput,
    is_label_edge_type,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    to_heterogeneous_node,
    to_homogeneous,
)
from gigl.utils.data_splitters import DistNodeAnchorLinkSplitter
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    create_test_process_group,
)
from tests.test_assets.test_case import TestCase

_POSITIVE_EDGE_TYPE = message_passing_to_positive_label(DEFAULT_HOMOGENEOUS_EDGE_TYPE)
_NEGATIVE_EDGE_TYPE = message_passing_to_negative_label(DEFAULT_HOMOGENEOUS_EDGE_TYPE)

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)
_STORY_TO_USER = EdgeType(_STORY, Relation("to"), _USER)

_A = NodeType("a")
_B = NodeType("b")
_C = NodeType("c")
_TO = Relation("to")
_LINK = Relation("link")
_A_TO_B = EdgeType(_A, _TO, _B)
_A_TO_C = EdgeType(_A, _TO, _C)
_A_LINK_B = EdgeType(_A, _LINK, _B)
_B_TO_A = EdgeType(_B, _TO, _A)
_C_TO_A = EdgeType(_C, _TO, _A)


# GLT requires subclasses of DistNeighborLoader to be run in a separate process. Otherwise, we may run into segmentation fault
# or other memory issues. Calling these functions in separate proceses also allows us to use shutdown_rpc() to ensure cleanup of
# ports, providing stronger guarantees of isolation between tests.


# We require each of these functions to accept local_rank as the first argument since we use mp.spawn with `nprocs=1`


def _assert_labels(
    anchor_nodes: torch.Tensor,
    supervision_nodes: torch.Tensor,
    y: dict[int, torch.Tensor],
    expected: dict[int, torch.Tensor],
):
    """
    Asserts that the given labels (y) match the expected labels (expected).
    The labels are in the *local* node space, but the expected labels are in the *global* node space.
    E.g expected_positive_labels = {10: torch.tensor([15])}
    But datum.y_positive = {0: torch.tensor([1])}
    So we need to convert, using `node`, the nodes in a batch.
    The local IDs are the index of a node in `node`, and the global IDs are the values of `node`.
    For example:
    node = torch.tensor([10, 11])
    y = {0: torch.tensor([1])}
    # y in global space is {10: torch.tensor([11])}
    expected = {10: torch.tensor([11])}
    _assert_labels(node, y, expected)

    Args:
        anchor_nodes (torch.Tensor): Tensor of nodes in the graph with the same type as the anchor node,
            shape [N] where N is the number of nodes in the batch with this node type
        supervision_nodes (torch.Tensor): Tensor of nodes in the graph with the same type as the supervision node,
            shape [M] where M is the number of nodes in the batch with this node type
        y (dict[int, torch.Tensor]): The labels in the local node space.
            The tensors are of shape [X], where X is the number of labels for the current anchor node.
        expected (dict[int, torch.Tensor]): The labels in the global node space.
            The tensors are of shape [X], where X is the number of labels for the current anchor node.
    Raises if:
    - The keys in `y` do not match the keys in `expected`
    - The values in `y` do not match the values in `expected`
    """
    supplied_global_nodes = anchor_nodes[list(y.keys())]
    assert set(supplied_global_nodes.tolist()) == set(expected.keys()), (
        f"Expected keys {expected.keys()} != {supplied_global_nodes.tolist()}"
    )
    for local_anchor in y:
        global_id = int(anchor_nodes[local_anchor].item())
        global_nodes = supervision_nodes[y[local_anchor]]
        expected_nodes = expected[global_id]
        assert_tensor_equality(global_nodes, expected_nodes, dim=0)


def _run_distributed_ablp_neighbor_loader(
    _,
    dataset: DistDataset,
    expected_node: torch.Tensor,
    expected_srcs: torch.Tensor,
    expected_dsts: torch.Tensor,
    expected_positive_labels: dict[int, torch.Tensor],
    expected_negative_labels: Optional[dict[int, torch.Tensor]],
):
    input_nodes = torch.tensor([10, 15])
    batch_size = 2

    create_test_process_group()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=input_nodes,
        batch_size=batch_size,
        pin_memory_device=torch.device("cpu"),
    )

    count = 0
    for datum in loader:
        assert isinstance(datum, Data)
        count += 1

    assert count == 1
    dsts, srcs, *_ = datum.coo()
    assert_tensor_equality(
        datum.node,
        expected_node,
        dim=0,
    )
    _assert_labels(datum.node, datum.node, datum.y_positive, expected_positive_labels)
    if expected_negative_labels is not None:
        # Pass is `datum.node` twice as this a homogenous object
        # and the anchor and supervision nodes are the same type.
        _assert_labels(
            datum.node, datum.node, datum.y_negative, expected_negative_labels
        )
    else:
        assert not hasattr(datum, "y_negative")
    dsts, srcs, *_ = datum.coo()
    assert_tensor_equality(datum.node[srcs], expected_srcs)
    assert_tensor_equality(datum.node[dsts], expected_dsts)

    # Check that the batch and batch_size attributes of the class are correct
    assert_tensor_equality(datum.batch, input_nodes)
    assert datum.batch_size == batch_size

    # This call is not strictly required to pass tests, since each test here uses the `run_in_separate_process` decorator,
    # but rather is good practice to ensure that we cleanup the rpc after we finish dataloading
    shutdown_rpc()


def _run_cora_supervised(
    _,
    dataset: DistDataset,
    expected_data_count: int,
):
    create_test_process_group()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=to_homogeneous(dataset.train_node_ids),
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in loader:
        assert isinstance(datum, Data)
        assert hasattr(datum, "y_positive")
        assert isinstance(datum.y_positive, dict)
        assert hasattr(datum, "y_negative")
        assert isinstance(datum.y_negative, dict)
        assert datum.y_positive.keys() == datum.y_negative.keys()
        count += 1
    assert count == expected_data_count

    shutdown_rpc()


def _run_dblp_supervised(
    _,
    dataset: DistDataset,
    supervision_edge_types: list[EdgeType],
):
    assert len(supervision_edge_types) == 1, (
        "TODO (mkolodner-sc): Support multiple supervision edge types in dataloading"
    )
    supervision_edge_type = supervision_edge_types[0]
    anchor_node_type = supervision_edge_type.src_node_type
    supervision_node_type = supervision_edge_type.dst_node_type
    assert isinstance(dataset.train_node_ids, dict)
    assert isinstance(dataset.graph, dict)
    fanout = [2, 2]
    # Label edge types must not be specified in the fanout (they are injected
    # internally and never sampled), so build num_neighbors over message-passing
    # edges only.
    num_neighbors = {
        edge_type: fanout
        for edge_type in dataset.graph.keys()
        if not is_label_edge_type(edge_type)
    }
    create_test_process_group()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=num_neighbors,
        input_nodes=(anchor_node_type, dataset.train_node_ids[anchor_node_type]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        supervision_edge_type=supervision_edge_type,
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in loader:
        assert isinstance(datum, HeteroData)
        assert hasattr(datum, "y_positive")
        assert isinstance(datum.y_positive, dict)
        assert not hasattr(datum, "y_negative")
        for local_anchor_node_id, local_positive_nodes in datum.y_positive.items():
            assert local_anchor_node_id < len(datum[anchor_node_type].batch)
            assert torch.all(
                local_positive_nodes < len(datum[supervision_node_type].node)
            )
        count += 1
    assert count == dataset.train_node_ids[anchor_node_type].size(0)  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.

    shutdown_rpc()


def _run_toy_heterogeneous_ablp(
    _,
    dataset: DistDataset,
    supervision_edge_types: list[EdgeType],
    fanout: Union[list[int], dict[EdgeType, list[int]]],
):
    anchor_node_type = NodeType("user")
    supervision_node_type = NodeType("story")
    assert len(supervision_edge_types) == 1, (
        "TODO (mkolodner-sc): Support multiple supervision edge types in dataloading"
    )
    supervision_edge_type = supervision_edge_types[0]
    assert isinstance(dataset.train_node_ids, dict)
    assert isinstance(dataset.graph, dict)
    labeled_edge_type = EdgeType(
        supervision_node_type, Relation("to_gigl_positive"), anchor_node_type
    )
    all_positive_supervision_nodes, all_anchor_nodes, _, _ = dataset.graph[
        labeled_edge_type
    ].topo.to_coo()
    create_test_process_group()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=fanout,
        input_nodes=(anchor_node_type, dataset.train_node_ids[anchor_node_type]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        supervision_edge_type=supervision_edge_type,
        # We set the batch size to the number of "user" nodes in the heterogeneous toy graph to guarantee that the dataloader completes an epoch in 1 batch
        batch_size=15,
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in loader:
        count += 1
    assert count == 1
    assert isinstance(datum, HeteroData)
    assert hasattr(datum, "y_positive")
    assert isinstance(datum.y_positive, dict)

    # Ensure that the node ids we should be fanout from are all found in the batch
    assert_tensor_equality(
        dataset.train_node_ids[anchor_node_type],  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        datum[anchor_node_type].batch,
    )
    assert (
        dataset.train_node_ids[anchor_node_type].size(0)  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        == datum[anchor_node_type].batch_size
    )

    global_anchor_nodes = []
    for local_anchor_node, local_positive_supervision_nodes in datum.y_positive.items():
        global_anchor_node = datum[anchor_node_type].node[local_anchor_node]
        global_positive_supervision_nodes = datum[supervision_node_type].node[
            local_positive_supervision_nodes
        ]
        global_anchor_nodes.append(global_anchor_node)

        # Check that the current anchor node from y_positive is found in the expected anchor tensor
        assert global_anchor_node.item() in all_anchor_nodes
        # Check that all positive supervision nodes from y_positive are found in the expected positive supervision tensor
        assert torch.isin(
            global_positive_supervision_nodes, all_positive_supervision_nodes
        ).all()
        # Check that we have also fanned out around the supervision node type
        assert datum.num_sampled_nodes[supervision_node_type][0] > 0

    # Check that the current anchor node from y_positive is found in the batch
    assert_tensor_equality(
        torch.tensor(global_anchor_nodes), datum[anchor_node_type].batch, dim=0
    )

    shutdown_rpc()


def _run_distributed_ablp_neighbor_loader_multiple_supervision_edge_types(
    _,
    input_nodes: tuple[NodeType, torch.Tensor],
    dataset: DistDataset,
    supervision_edge_types: list[EdgeType],
    expected_node: dict[NodeType, torch.Tensor],
    expected_batch: dict[NodeType, torch.Tensor],
    expected_edges: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]],
    expected_positive_labels: dict[EdgeType, dict[int, torch.Tensor]],
    expected_negative_labels: Optional[dict[EdgeType, dict[int, torch.Tensor]]],
):
    batch_size = 1

    create_test_process_group()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=input_nodes,
        batch_size=batch_size,
        pin_memory_device=torch.device("cpu"),
        supervision_edge_type=supervision_edge_types,
    )

    count = 0
    for datum in loader:
        assert isinstance(datum, HeteroData)
        count += 1

    assert count == 1
    assert set(datum.node_types) == set(expected_node.keys())
    for node_type in datum.node_types:
        assert_tensor_equality(
            datum[node_type].node,
            expected_node[node_type],
            dim=0,
        )
    assert hasattr(datum, "y_positive")
    assert set(datum.y_positive.keys()) == set(expected_positive_labels.keys()), (
        f"{datum.y_positive.keys()} != {expected_positive_labels.keys()}"
    )
    anchor_index = 0
    supervision_index = 2
    for edge_type in datum.y_positive.keys():
        _assert_labels(
            anchor_nodes=datum[edge_type[anchor_index]].node,
            supervision_nodes=datum[edge_type[supervision_index]].node,
            y=datum.y_positive[edge_type],
            expected=expected_positive_labels[edge_type],
        )
    if expected_negative_labels is not None:
        _assert_labels(
            anchor_nodes=datum[edge_type[anchor_index]].node,
            supervision_nodes=datum[edge_type[supervision_index]].node,
            y=datum.y_negative[edge_type],
            expected=expected_negative_labels[edge_type],
        )
    else:
        assert not hasattr(datum, "y_negative")

    # Reverse as the dataset edge dir is "out" so GLT reverses under the hood.
    if dataset.edge_dir == "out":
        expected_edges = {
            reverse_edge_type(edge_type): edges
            for edge_type, edges in expected_edges.items()
        }
    dsts, srcs, *_ = datum.coo()
    assert set(expected_edges.keys()) == set(dsts.keys()), (
        f"{expected_edges.keys()} != {dsts.keys()}"
    )
    assert set(expected_edges.keys()) == set(srcs.keys()), (
        f"{expected_edges.keys()} != {srcs.keys()}"
    )
    for edge_type in expected_edges.keys():
        assert_tensor_equality(
            datum[edge_type[0]].node[dsts[edge_type]],
            expected_edges[edge_type][1],
            dim=0,
        )
        assert_tensor_equality(
            datum[edge_type[2]].node[srcs[edge_type]],
            expected_edges[edge_type][0],
            dim=0,
        )

    # Check that the batch and batch_size attributes of the class are correct
    assert set(datum.node_types) == set(expected_node.keys())
    for node_type in datum.node_types:
        assert_tensor_equality(
            datum[node_type].node,
            expected_node[node_type],
            dim=0,
        )
    assert set(datum.node_types) == set(expected_batch.keys())
    for node_type in datum.node_types:
        if expected_batch[node_type] is not None:
            assert_tensor_equality(
                datum[node_type].batch,
                expected_batch[node_type],
                dim=0,
            )
        else:
            assert not hasattr(datum[node_type], "batch")

    # This call is not strictly required to pass tests, since each test here uses the `run_in_separate_process` decorator,
    # but rather is good practice to ensure that we cleanup the rpc after we finish dataloading
    shutdown_rpc()


def _collect_homogeneous_labels(
    _: int,
    return_dict,
    collate_impl: str,
    dataset: DistDataset,
    input_nodes: torch.Tensor,
    batch_size: int,
    has_negatives: bool,
):
    """Child-side: run the loader under one collate impl, return labels in GLOBAL ids.

    Local node indices differ run-to-run, so we translate y_* tensors back to
    global ids via ``datum.node`` before returning, giving the parent a
    representation that is invariant to local-index assignment.
    """
    os.environ[COLLATE_IMPL_ENV_VAR] = collate_impl
    create_test_process_group()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=input_nodes,
        batch_size=batch_size,
        pin_memory_device=torch.device("cpu"),
    )
    collected_positive: dict[int, list[int]] = {}
    collected_negative: dict[int, list[int]] = {}
    for datum in loader:
        assert isinstance(datum, Data)
        node = datum.node
        for local_anchor, local_nodes in datum.y_positive.items():
            global_anchor = int(node[local_anchor].item())
            collected_positive[global_anchor] = sorted(
                int(g.item()) for g in node[local_nodes]
            )
        if has_negatives:
            for local_anchor, local_nodes in datum.y_negative.items():
                global_anchor = int(node[local_anchor].item())
                collected_negative[global_anchor] = sorted(
                    int(g.item()) for g in node[local_nodes]
                )
        else:
            # No negative-label edge type: y_negative must be absent or empty
            # regardless of impl. Catches a spurious-negatives regression that a
            # vacuous {} == {} comparison in the parent would miss.
            assert getattr(datum, "y_negative", {}) == {}, (
                f"{collate_impl}: expected no negatives, got {datum.y_negative}"
            )
    return_dict[collate_impl] = (collected_positive, collected_negative)
    shutdown_rpc()


def _collect_hetero_labels(
    _: int,
    return_dict,
    collate_impl: str,
    input_nodes: tuple[NodeType, torch.Tensor],
    dataset: DistDataset,
    supervision_edge_types: list[EdgeType],
    has_negatives: bool,
):
    """Child-side: run the hetero loader under one collate impl, return labels in GLOBAL ids."""
    os.environ[COLLATE_IMPL_ENV_VAR] = collate_impl
    create_test_process_group()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=input_nodes,
        batch_size=1,
        pin_memory_device=torch.device("cpu"),
        supervision_edge_type=supervision_edge_types,
    )
    anchor_index = 0
    supervision_index = 2
    positive: dict[str, dict[int, list[int]]] = {}
    negative: dict[str, dict[int, list[int]]] = {}
    for datum in loader:
        assert isinstance(datum, HeteroData)
        for edge_type, inner in datum.y_positive.items():
            anchor_node = datum[edge_type[anchor_index]].node
            supervision_node = datum[edge_type[supervision_index]].node
            positive.setdefault(str(edge_type), {})
            for local_anchor, local_nodes in inner.items():
                global_anchor = int(anchor_node[local_anchor].item())
                positive[str(edge_type)][global_anchor] = sorted(
                    int(g.item()) for g in supervision_node[local_nodes]
                )
        if has_negatives:
            for edge_type, inner in datum.y_negative.items():
                anchor_node = datum[edge_type[anchor_index]].node
                supervision_node = datum[edge_type[supervision_index]].node
                negative.setdefault(str(edge_type), {})
                for local_anchor, local_nodes in inner.items():
                    global_anchor = int(anchor_node[local_anchor].item())
                    negative[str(edge_type)][global_anchor] = sorted(
                        int(g.item()) for g in supervision_node[local_nodes]
                    )
        else:
            # No negative-label edge type: y_negative must be absent or empty
            # regardless of impl. Catches a spurious-negatives regression that a
            # vacuous {} == {} comparison in the parent would miss.
            assert getattr(datum, "y_negative", {}) == {}, (
                f"{collate_impl}: expected no negatives, got {datum.y_negative}"
            )
    return_dict[collate_impl] = (positive, negative)
    shutdown_rpc()


def _collect_homogeneous_labels_edge_list(
    _: int,
    return_dict,
    dataset: DistDataset,
    input_nodes: torch.Tensor,
    batch_size: int,
    has_negatives: bool,
):
    """Child-side: run the loader under GIGL_ABLP_LABEL_FORMAT=edge_list.

    Translates AnchorLabels back to global ids using datum.node so the result
    is directly comparable to the dict-format output from
    _collect_homogeneous_labels.
    """
    os.environ[ABLP_LABEL_FORMAT_ENV_VAR] = "edge_list"
    create_test_process_group()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=input_nodes,
        batch_size=batch_size,
        pin_memory_device=torch.device("cpu"),
    )
    collected_positive: dict[int, list[int]] = {}
    collected_negative: dict[int, list[int]] = {}
    for datum in loader:
        assert isinstance(datum, Data)
        node = datum.node
        assert isinstance(datum.y_positive, AnchorLabels), (
            f"Expected AnchorLabels under edge_list format, got {type(datum.y_positive)}"
        )
        y_positive_dict = datum.y_positive.to_dict()
        for local_anchor, local_nodes in y_positive_dict.items():
            global_anchor = int(node[local_anchor].item())
            collected_positive[global_anchor] = sorted(
                int(g.item()) for g in node[local_nodes]
            )
        if has_negatives:
            assert isinstance(datum.y_negative, AnchorLabels), (
                f"Expected AnchorLabels under edge_list format, got {type(datum.y_negative)}"
            )
            y_negative_dict = datum.y_negative.to_dict()
            for local_anchor, local_nodes in y_negative_dict.items():
                global_anchor = int(node[local_anchor].item())
                collected_negative[global_anchor] = sorted(
                    int(g.item()) for g in node[local_nodes]
                )
        else:
            # No negative-label edge type: y_negative must be absent.
            assert not hasattr(datum, "y_negative"), (
                f"edge_list: expected no negatives, got {getattr(datum, 'y_negative', None)}"
            )
    return_dict["edge_list"] = (collected_positive, collected_negative)
    shutdown_rpc()


class DistABLPLoaderTest(TestCase):
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
                "Positive and Negative edges",
                labeled_edges={
                    _POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]]),
                    _NEGATIVE_EDGE_TYPE: torch.tensor(
                        [[10, 10, 11, 15], [13, 16, 14, 17]]
                    ),
                },
                expected_node=torch.tensor([10, 11, 12, 13, 14, 15, 16, 17]),
                expected_srcs=torch.tensor([10, 10, 15, 15, 16, 16, 11, 11]),
                expected_dsts=torch.tensor([11, 12, 13, 14, 12, 14, 13, 17]),
                expected_positive_labels={
                    10: torch.tensor([15]),
                    15: torch.tensor([16]),
                },
                expected_negative_labels={
                    10: torch.tensor([13, 16]),
                    15: torch.tensor([17]),
                },
                max_labels_per_anchor_node=None,
            ),
            param(
                "Positive edges",
                labeled_edges={_POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]])},
                expected_node=torch.tensor([10, 11, 12, 13, 14, 15, 16, 17]),
                expected_srcs=torch.tensor([10, 10, 15, 15, 16, 16, 11, 11]),
                expected_dsts=torch.tensor([11, 12, 13, 14, 12, 14, 13, 17]),
                expected_positive_labels={
                    10: torch.tensor([15]),
                    15: torch.tensor([16]),
                },
                expected_negative_labels=None,
                max_labels_per_anchor_node=None,
            ),
            param(
                "Positive and Negative edges with label cap",
                labeled_edges={
                    _POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]]),
                    _NEGATIVE_EDGE_TYPE: torch.tensor(
                        [[10, 10, 11, 15], [13, 16, 14, 17]]
                    ),
                },
                expected_node=torch.tensor([10, 11, 12, 13, 14, 15, 16, 17]),
                expected_srcs=torch.tensor([10, 10, 15, 15, 16, 16, 11, 11]),
                expected_dsts=torch.tensor([11, 12, 13, 14, 12, 14, 13, 17]),
                expected_positive_labels={
                    10: torch.tensor([15]),
                    15: torch.tensor([16]),
                },
                expected_negative_labels={
                    10: torch.tensor([13]),
                    15: torch.tensor([17]),
                },
                max_labels_per_anchor_node=1,
            ),
        ]
    )
    def test_ablp_dataloader(
        self,
        _,
        labeled_edges,
        expected_node,
        expected_srcs,
        expected_dsts,
        expected_positive_labels,
        expected_negative_labels,
        max_labels_per_anchor_node,
    ):
        # Graph looks like https://is.gd/w2oEVp:
        # Message passing
        # 10 -> {11, 12}
        # 11 -> {13, 17}
        # 15 -> {13, 14}
        # 16 -> {12, 14}
        # Positive labels
        # 10 -> 15
        # 15 -> 16
        # Negative labels
        # 10 -> {13, 16}
        # 11 -> 14

        edge_index = {
            DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor(
                [
                    [10, 10, 11, 11, 15, 15, 16, 16],
                    [11, 12, 13, 17, 13, 14, 12, 14],
                ]
            ),
        }
        edge_index.update(labeled_edges)

        partition_output = PartitionOutput(
            node_partition_book=to_heterogeneous_node(torch.zeros(18)),
            edge_partition_book={
                e_type: torch.zeros(int(e_idx.max().item() + 1))
                for e_type, e_idx in edge_index.items()
            },
            partitioned_edge_index={
                etype: GraphPartitionData(
                    edge_index=idx, edge_ids=torch.arange(idx.size(1))
                )
                for etype, idx in edge_index.items()
            },
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(
            rank=0,
            world_size=1,
            edge_dir="out",
            max_labels_per_anchor_node=max_labels_per_anchor_node,
        )
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_distributed_ablp_neighbor_loader,
            args=(
                dataset,
                expected_node,
                expected_srcs,
                expected_dsts,
                expected_positive_labels,
                expected_negative_labels,
            ),
        )

    def test_cora_supervised(self):
        create_test_process_group()
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=cora_supervised_info.frozen_gbml_config_uri
            )
        )

        serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
            preprocessed_metadata_pb_wrapper=gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
            graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            tfrecord_uri_pattern=".*.tfrecord(.gz)?$",
        )

        splitter = DistNodeAnchorLinkSplitter(
            sampling_direction="in", should_convert_labels_to_edges=True
        )

        dataset = build_dataset(
            serialized_graph_metadata=serialized_graph_metadata,
            sample_edge_direction="in",
            splitter=splitter,
        )

        assert dataset.train_node_ids is not None, "Train node ids must exist."

        mp.spawn(
            fn=_run_cora_supervised,
            args=(
                dataset,
                to_homogeneous(
                    dataset.train_node_ids
                ).numel(),  # Use to_homogeneous to make MyPy happy since dataset.train_node_ids is a dict.
            ),
        )

    # TODO: (mkolodner-sc) - Figure out why this test is failing on Google Cloud Build
    @unittest.skip("Failing on Google Cloud Build - skiping for now")
    def test_dblp_supervised(self):
        create_test_process_group()
        dblp_supervised_info = get_mocked_dataset_artifact_metadata()[
            DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=dblp_supervised_info.frozen_gbml_config_uri
            )
        )

        serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
            preprocessed_metadata_pb_wrapper=gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
            graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            tfrecord_uri_pattern=".*.tfrecord(.gz)?$",
        )

        supervision_edge_types = (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_types()
        )

        splitter = DistNodeAnchorLinkSplitter(
            sampling_direction="in",
            supervision_edge_types=supervision_edge_types,
            should_convert_labels_to_edges=True,
        )

        dataset = build_dataset(
            serialized_graph_metadata=serialized_graph_metadata,
            sample_edge_direction="in",
            _ssl_positive_label_percentage=0.1,
            splitter=splitter,
        )

        mp.spawn(
            fn=_run_dblp_supervised,
            args=(dataset, supervision_edge_types),
        )

    @parameterized.expand(
        [
            param(
                "Tensor-based partitioning, list fanout",
                partitioner_class=DistPartitioner,
                fanout=[2, 2],
            ),
            param(
                "Range-based partitioning, list fanout",
                partitioner_class=DistRangePartitioner,
                fanout=[2, 2],
            ),
            param(
                "Range-based partitioning, dict fanout",
                partitioner_class=DistRangePartitioner,
                fanout={
                    EdgeType(NodeType("user"), Relation("to"), NodeType("story")): [
                        2,
                        2,
                    ],
                    EdgeType(NodeType("story"), Relation("to"), NodeType("user")): [
                        2,
                        2,
                    ],
                },
            ),
        ]
    )
    def test_toy_heterogeneous_ablp(
        self,
        _,
        partitioner_class: type[DistPartitioner],
        fanout: Union[list[int], dict[EdgeType, list[int]]],
    ):
        create_test_process_group()
        toy_heterogeneous_supervised_info = get_mocked_dataset_artifact_metadata()[
            HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=toy_heterogeneous_supervised_info.frozen_gbml_config_uri
            )
        )
        serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
            preprocessed_metadata_pb_wrapper=gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
            graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            tfrecord_uri_pattern=".*.tfrecord(.gz)?$",
        )

        supervision_edge_types = (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_types()
        )

        splitter = DistNodeAnchorLinkSplitter(
            sampling_direction="in",
            supervision_edge_types=supervision_edge_types,
            should_convert_labels_to_edges=True,
        )

        dataset = build_dataset(
            serialized_graph_metadata=serialized_graph_metadata,
            sample_edge_direction="in",
            _ssl_positive_label_percentage=0.1,
            splitter=splitter,
            partitioner_class=partitioner_class,
        )

        mp.spawn(
            fn=_run_toy_heterogeneous_ablp,
            args=(dataset, supervision_edge_types, fanout),
        )

    @parameterized.expand(
        [
            # https://is.gd/zG8xIn
            param(
                "positive edges",
                edge_dir="out",
                edge_index={
                    _A_TO_B: torch.tensor([[10, 10], [11, 12]]),
                    message_passing_to_positive_label(_A_TO_B): torch.tensor(
                        [[10, 10], [13, 14]]
                    ),
                    _A_TO_C: torch.tensor([[10, 10], [20, 21]]),
                    message_passing_to_positive_label(_A_TO_C): torch.tensor(
                        [[10, 10], [22, 23]]
                    ),
                    # Add an edge that isn't a supervision edge type
                    _A_LINK_B: torch.tensor([[10, 10], [20, 21]]),
                },
                supervision_edge_types=[_A_TO_B, _A_TO_C],
                expected_node={
                    _A: torch.tensor([10]),
                    _B: torch.tensor(
                        [
                            11,
                            12,
                            13,
                            14,
                            20,
                            21,
                        ]
                    ),
                    _C: torch.tensor(
                        [
                            20,
                            21,
                            22,
                            23,
                        ]
                    ),
                },
                expected_batch={
                    _A: torch.tensor([10]),
                    _B: None,
                    _C: None,
                },
                expected_edges={
                    _A_TO_B: (torch.tensor([10, 10]), torch.tensor([11, 12])),
                    _A_TO_C: (torch.tensor([10, 10]), torch.tensor([20, 21])),
                    _A_LINK_B: (torch.tensor([10, 10]), torch.tensor([20, 21])),
                },
                expected_positive_labels={
                    _A_TO_B: {10: torch.tensor([13, 14])},
                    _A_TO_C: {10: torch.tensor([22, 23])},
                },
                expected_negative_labels=None,
            ),
            # https://is.gd/nTVBll
            param(
                "positive and negative edges",
                edge_dir="out",
                edge_index={
                    _A_TO_B: torch.tensor([[10, 10], [11, 12]]),
                    message_passing_to_positive_label(_A_TO_B): torch.tensor(
                        [[10, 10], [13, 14]]
                    ),
                    message_passing_to_negative_label(_A_TO_B): torch.tensor(
                        [[10, 10], [15, 16]]
                    ),
                    _A_TO_C: torch.tensor([[10, 10], [20, 21]]),
                    message_passing_to_positive_label(_A_TO_C): torch.tensor(
                        [[10, 10], [22, 23]]
                    ),
                    message_passing_to_negative_label(_A_TO_C): torch.tensor(
                        [[10, 10], [24, 25]]
                    ),
                },
                supervision_edge_types=[_A_TO_B, _A_TO_C],
                expected_node={
                    _A: torch.tensor([10]),
                    _B: torch.tensor([11, 12, 13, 14, 15, 16]),
                    _C: torch.tensor([20, 21, 22, 23, 24, 25]),
                },
                expected_batch={
                    _A: torch.tensor([10]),
                    _B: None,
                    _C: None,
                },
                expected_edges={
                    _A_TO_B: (torch.tensor([10, 10]), torch.tensor([11, 12])),
                    _A_TO_C: (torch.tensor([10, 10]), torch.tensor([20, 21])),
                },
                expected_positive_labels={
                    _A_TO_B: {10: torch.tensor([13, 14])},
                    _A_TO_C: {10: torch.tensor([22, 23])},
                },
                expected_negative_labels={
                    _A_TO_B: {10: torch.tensor([15, 16])},
                    _A_TO_C: {10: torch.tensor([24, 25])},
                },
            ),
            # https://is.gd/mO5cpW
            param(
                "same nodes, different relation",
                edge_dir="out",
                edge_index={
                    _A_TO_B: torch.tensor([[10, 10], [11, 12]]),
                    message_passing_to_positive_label(_A_TO_B): torch.tensor(
                        [[10, 10], [13, 14]]
                    ),
                    _A_LINK_B: torch.tensor([[10, 10], [20, 21]]),
                    message_passing_to_positive_label(_A_LINK_B): torch.tensor(
                        [[10, 10], [22, 23]]
                    ),
                },
                supervision_edge_types=[_A_TO_B, _A_LINK_B],
                expected_node={
                    _A: torch.tensor([10]),
                    _B: torch.tensor([11, 12, 13, 14, 20, 21, 22, 23]),
                },
                expected_batch={
                    _A: torch.tensor([10]),
                    _B: None,
                },
                expected_edges={
                    _A_TO_B: (torch.tensor([10, 10]), torch.tensor([11, 12])),
                    _A_LINK_B: (torch.tensor([10, 10]), torch.tensor([20, 21])),
                },
                expected_positive_labels={
                    _A_TO_B: {10: torch.tensor([13, 14])},
                    _A_LINK_B: {10: torch.tensor([22, 23])},
                },
                expected_negative_labels=None,
            ),
            # https://is.gd/oxDB6C
            param(
                "positive edges, edge_dir=in",
                edge_dir="in",
                edge_index={
                    _B_TO_A: torch.tensor([[11, 12], [10, 10]]),
                    message_passing_to_positive_label(_B_TO_A): torch.tensor(
                        [[13, 14], [10, 10]]
                    ),
                    _C_TO_A: torch.tensor([[20, 21], [10, 10]]),
                    message_passing_to_positive_label(_C_TO_A): torch.tensor(
                        [[22, 23], [10, 10]]
                    ),
                },
                supervision_edge_types=[_A_TO_B, _A_TO_C],
                expected_node={
                    _A: torch.tensor([10]),
                    _B: torch.tensor(
                        [
                            11,
                            12,
                            13,
                            14,
                        ]
                    ),
                    _C: torch.tensor(
                        [
                            20,
                            21,
                            22,
                            23,
                        ]
                    ),
                },
                expected_batch={
                    _A: torch.tensor([10]),
                    _B: None,
                    _C: None,
                },
                expected_edges={
                    _B_TO_A: (torch.tensor([10, 10]), torch.tensor([11, 12])),
                    _C_TO_A: (torch.tensor([10, 10]), torch.tensor([20, 21])),
                },
                expected_positive_labels={
                    _A_TO_B: {10: torch.tensor([13, 14])},
                    _A_TO_C: {10: torch.tensor([22, 23])},
                },
                expected_negative_labels=None,
            ),
        ]
    )
    def test_ablp_dataloder_multiple_supervision_edge_types(
        self,
        _,
        edge_dir: Literal["in", "out"],
        edge_index: dict[EdgeType, torch.Tensor],
        supervision_edge_types: list[EdgeType],
        expected_node: dict[NodeType, torch.Tensor],
        expected_batch: dict[NodeType, Optional[torch.Tensor]],
        expected_edges: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]],
        expected_positive_labels: dict[EdgeType, dict[int, torch.Tensor]],
        expected_negative_labels: Optional[dict[EdgeType, dict[int, torch.Tensor]]],
    ):
        nodes: dict[NodeType, list[torch.Tensor]] = defaultdict(list)
        for edge_type, edge_idx in edge_index.items():
            nodes[edge_type[0]].append(edge_idx[0])
            nodes[edge_type[2]].append(edge_idx[1])
        partition_output = PartitionOutput(
            node_partition_book={
                node_type: torch.zeros(int(torch.cat(node_ids).max().item() + 1))
                for node_type, node_ids in nodes.items()
            },
            edge_partition_book={
                e_type: torch.zeros(int(e_idx.max().item() + 1))
                for e_type, e_idx in edge_index.items()
            },
            partitioned_edge_index={
                etype: GraphPartitionData(
                    edge_index=idx, edge_ids=torch.arange(idx.size(1))
                )
                for etype, idx in edge_index.items()
            },
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir=edge_dir)
        dataset.build(partition_output=partition_output)
        (
            mp.spawn(
                fn=_run_distributed_ablp_neighbor_loader_multiple_supervision_edge_types,
                args=(
                    (NodeType("a"), torch.tensor([10])),  # input_nodes
                    dataset,  # dataset
                    supervision_edge_types,  # supervision_edge_types
                    expected_node,  # expected_node
                    expected_batch,  # expected_batch
                    expected_edges,  # expected_edges
                    expected_positive_labels,  # expected_positive_labels
                    expected_negative_labels,  # expected_negative_labels
                ),
            ),
        )

    @parameterized.expand(
        [
            param(
                "positive and negative",
                labeled_edges={
                    _POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]]),
                    _NEGATIVE_EDGE_TYPE: torch.tensor(
                        [[10, 10, 11, 15], [13, 16, 14, 17]]
                    ),
                },
                input_nodes=torch.tensor([10, 15]),
                batch_size=2,
                has_negatives=True,
                empty_positive_anchor=None,
            ),
            param(
                "positive only",
                labeled_edges={_POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]])},
                input_nodes=torch.tensor([10, 15]),
                batch_size=2,
                has_negatives=False,
                empty_positive_anchor=None,
            ),
            # Anchor 11 has message-passing edges (11 -> {13, 17}) but is the
            # source of NO positive-label edge, so its positive-label CSR row is
            # all-padding and y_positive[11] is a guaranteed-empty tensor. This
            # exercises the empty-anchor branch of both label-remap impls at the
            # loader level.
            param(
                "guaranteed empty positive anchor",
                labeled_edges={
                    _POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]]),
                    _NEGATIVE_EDGE_TYPE: torch.tensor(
                        [[10, 10, 11, 15], [13, 16, 14, 17]]
                    ),
                },
                input_nodes=torch.tensor([10, 11, 15]),
                batch_size=3,
                has_negatives=True,
                empty_positive_anchor=11,
            ),
        ]
    )
    def test_collate_impl_equivalence_homogeneous(
        self,
        _,
        labeled_edges,
        input_nodes,
        batch_size,
        has_negatives,
        empty_positive_anchor,
    ):
        edge_index = {
            DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor(
                [[10, 10, 11, 11, 15, 15, 16, 16], [11, 12, 13, 17, 13, 14, 12, 14]]
            ),
        }
        edge_index.update(labeled_edges)
        partition_output = PartitionOutput(
            node_partition_book=to_heterogeneous_node(torch.zeros(18)),
            edge_partition_book={
                e_type: torch.zeros(int(e_idx.max().item() + 1))
                for e_type, e_idx in edge_index.items()
            },
            partitioned_edge_index={
                etype: GraphPartitionData(
                    edge_index=idx, edge_ids=torch.arange(idx.size(1))
                )
                for etype, idx in edge_index.items()
            },
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        manager = mp.Manager()
        return_dict = manager.dict()
        for collate_impl in ("python", "vectorized"):
            mp.spawn(
                fn=_collect_homogeneous_labels,
                args=(
                    return_dict,
                    collate_impl,
                    dataset,
                    input_nodes,
                    batch_size,
                    has_negatives,
                ),
            )
        self.assertEqual(return_dict["python"][0], return_dict["vectorized"][0])
        self.assertEqual(return_dict["python"][1], return_dict["vectorized"][1])
        if empty_positive_anchor is not None:
            # Both impls must emit the empty anchor's key with an empty list.
            for collate_impl in ("python", "vectorized"):
                positive = return_dict[collate_impl][0]
                self.assertIn(empty_positive_anchor, positive)
                self.assertEqual(positive[empty_positive_anchor], [])

    @parameterized.expand(
        [
            param(
                "positive and negative",
                labeled_edges={
                    _POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]]),
                    _NEGATIVE_EDGE_TYPE: torch.tensor(
                        [[10, 10, 11, 15], [13, 16, 14, 17]]
                    ),
                },
                input_nodes=torch.tensor([10, 15]),
                batch_size=2,
                has_negatives=True,
            ),
            param(
                "positive only",
                labeled_edges={_POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]])},
                input_nodes=torch.tensor([10, 15]),
                batch_size=2,
                has_negatives=False,
            ),
        ]
    )
    def test_label_format_edge_list_equivalence(
        self,
        _,
        labeled_edges,
        input_nodes,
        batch_size,
        has_negatives,
    ):
        """GIGL_ABLP_LABEL_FORMAT=edge_list produces AnchorLabels whose .to_dict()
        matches the dict-format (python collate impl) output exactly.
        """
        edge_index = {
            DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor(
                [[10, 10, 11, 11, 15, 15, 16, 16], [11, 12, 13, 17, 13, 14, 12, 14]]
            ),
        }
        edge_index.update(labeled_edges)
        partition_output = PartitionOutput(
            node_partition_book=to_heterogeneous_node(torch.zeros(18)),
            edge_partition_book={
                e_type: torch.zeros(int(e_idx.max().item() + 1))
                for e_type, e_idx in edge_index.items()
            },
            partitioned_edge_index={
                etype: GraphPartitionData(
                    edge_index=idx, edge_ids=torch.arange(idx.size(1))
                )
                for etype, idx in edge_index.items()
            },
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        manager = mp.Manager()
        return_dict = manager.dict()

        # Collect dict-format baseline with the python collate impl.
        mp.spawn(
            fn=_collect_homogeneous_labels,
            args=(
                return_dict,
                "python",
                dataset,
                input_nodes,
                batch_size,
                has_negatives,
            ),
        )

        # Collect edge_list format (AnchorLabels), expanded back to dicts.
        mp.spawn(
            fn=_collect_homogeneous_labels_edge_list,
            args=(
                return_dict,
                dataset,
                input_nodes,
                batch_size,
                has_negatives,
            ),
        )

        self.assertEqual(return_dict["python"][0], return_dict["edge_list"][0])
        self.assertEqual(return_dict["python"][1], return_dict["edge_list"][1])

    @parameterized.expand(
        [
            param(
                "out, positive and negative",
                edge_dir="out",
                edge_index={
                    _A_TO_B: torch.tensor([[10, 10], [11, 12]]),
                    message_passing_to_positive_label(_A_TO_B): torch.tensor(
                        [[10, 10], [13, 14]]
                    ),
                    message_passing_to_negative_label(_A_TO_B): torch.tensor(
                        [[10, 10], [15, 16]]
                    ),
                    _A_TO_C: torch.tensor([[10, 10], [20, 21]]),
                    message_passing_to_positive_label(_A_TO_C): torch.tensor(
                        [[10, 10], [22, 23]]
                    ),
                    message_passing_to_negative_label(_A_TO_C): torch.tensor(
                        [[10, 10], [24, 25]]
                    ),
                },
                supervision_edge_types=[_A_TO_B, _A_TO_C],
                has_negatives=True,
            ),
            param(
                "in, positive only",
                edge_dir="in",
                edge_index={
                    _B_TO_A: torch.tensor([[11, 12], [10, 10]]),
                    message_passing_to_positive_label(_B_TO_A): torch.tensor(
                        [[13, 14], [10, 10]]
                    ),
                    _C_TO_A: torch.tensor([[20, 21], [10, 10]]),
                    message_passing_to_positive_label(_C_TO_A): torch.tensor(
                        [[22, 23], [10, 10]]
                    ),
                },
                supervision_edge_types=[_A_TO_B, _A_TO_C],
                has_negatives=False,
            ),
        ]
    )
    def test_collate_impl_equivalence_heterogeneous(
        self, _, edge_dir, edge_index, supervision_edge_types, has_negatives
    ):
        nodes: dict[NodeType, list[torch.Tensor]] = defaultdict(list)
        for edge_type, edge_idx in edge_index.items():
            nodes[edge_type[0]].append(edge_idx[0])
            nodes[edge_type[2]].append(edge_idx[1])
        partition_output = PartitionOutput(
            node_partition_book={
                node_type: torch.zeros(int(torch.cat(node_ids).max().item() + 1))
                for node_type, node_ids in nodes.items()
            },
            edge_partition_book={
                e_type: torch.zeros(int(e_idx.max().item() + 1))
                for e_type, e_idx in edge_index.items()
            },
            partitioned_edge_index={
                etype: GraphPartitionData(
                    edge_index=idx, edge_ids=torch.arange(idx.size(1))
                )
                for etype, idx in edge_index.items()
            },
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir=edge_dir)
        dataset.build(partition_output=partition_output)

        manager = mp.Manager()
        return_dict = manager.dict()
        for collate_impl in ("python", "vectorized"):
            mp.spawn(
                fn=_collect_hetero_labels,
                args=(
                    return_dict,
                    collate_impl,
                    (NodeType("a"), torch.tensor([10])),
                    dataset,
                    supervision_edge_types,
                    has_negatives,
                ),
            )
        self.assertEqual(return_dict["python"][0], return_dict["vectorized"][0])
        self.assertEqual(return_dict["python"][1], return_dict["vectorized"][1])

    @parameterized.expand(
        [
            param(
                "Empty list of supervision edge types",
                expected_error=ValueError,
                expected_error_message="supervision_edge_type must be a non-empty list when providing multiple supervision edge types.",
                dataset=DistDataset(
                    rank=0,
                    world_size=1,
                    edge_dir="out",
                    graph_partition={},
                    node_partition_book={},
                ),
                num_neighbors=[2, 2],
                input_nodes=(NodeType("a"), torch.tensor([10])),
                supervision_edge_type=[],
            ),
            param(
                "Homogenous dataset",
                expected_error=ValueError,
                expected_error_message="The dataset must be heterogeneous for ABLP",
                dataset=DistDataset(rank=0, world_size=1, edge_dir="out"),
                num_neighbors=[2, 2],
                input_nodes=(NodeType("a"), torch.tensor([10])),
                supervision_edge_type=[_A_TO_B],
            ),
            param(
                "No supervision edge type, heterogenous sampling",
                expected_error=ValueError,
                expected_error_message="When using heterogeneous ABLP, you must provide supervision_edge_types",
                dataset=DistDataset(
                    rank=0,
                    world_size=1,
                    edge_dir="out",
                    graph_partition={},
                    node_partition_book={},
                ),
                num_neighbors=[2, 2],
                input_nodes=(NodeType("a"), torch.tensor([10])),
                supervision_edge_type=None,
            ),
            param(
                "Mutiple supervision edge types, homogeneous sampling",
                expected_error=ValueError,
                expected_error_message="Expected supervision edge type to be None for homogeneous input nodes",
                dataset=DistDataset(
                    rank=0,
                    world_size=1,
                    edge_dir="out",
                    graph_partition={},
                    node_partition_book={},
                ),
                num_neighbors=[2, 2],
                input_nodes=torch.tensor([10]),
                supervision_edge_type=[_A_TO_B, _A_TO_C],
            ),
        ]
    )
    def test_ablp_dataloader_invalid_inputs(
        self,
        _: str,
        expected_error: type[BaseException],
        expected_error_message: str,
        **kwargs,
    ):
        create_test_process_group()
        with self.assertRaises(expected_error, msg=expected_error_message):
            DistABLPLoader(**kwargs)


if __name__ == "__main__":
    absltest.main()
