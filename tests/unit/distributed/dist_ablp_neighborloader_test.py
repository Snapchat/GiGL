import unittest
from collections import defaultdict
from typing import Any, Literal, Optional, Union
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from graphlearn_torch.utils import reverse_edge_type
from parameterized import param, parameterized
from torch_geometric.data import Data, HeteroData

from gigl.distributed.base_dist_loader import BaseDistLoader
from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_ablp_neighborloader import (
    DistABLPLoader,
    _convert_graph_store_ablp_inputs,
    _is_integral_tensor,
)
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_partitioner import DistPartitioner
from gigl.distributed.dist_range_partitioner import DistRangePartitioner
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
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    to_heterogeneous_node,
    to_homogeneous,
)
from gigl.utils.data_splitters import DistNodeAnchorLinkSplitter
from gigl.utils.sampling import ABLPInputNodes
from tests.test_assets.distributed.utils import (
    MockRemoteDistDataset,
    assert_tensor_equality,
    create_test_process_group,
    get_process_group_init_method,
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
    num_neighbors = {edge_type: fanout for edge_type in dataset.graph.keys()}
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
        for edge_type, expected_labels in expected_negative_labels.items():
            _assert_labels(
                anchor_nodes=datum[edge_type[anchor_index]].node,
                supervision_nodes=datum[edge_type[supervision_index]].node,
                y=datum.y_negative[edge_type],
                expected=expected_labels,
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


def _global_pair_set(
    anchor_node: torch.Tensor,
    label_node: torch.Tensor,
    label_dict: dict[int, torch.Tensor],
) -> list[tuple[int, int]]:
    """Convert a label dictionary to sorted global-id pairs.

    Anchors and labels are looked up in separate node maps so this works for
    heterogeneous graphs, where the two live in different node stores. Pass the
    same tensor twice for a homogeneous graph.
    """
    pairs: list[tuple[int, int]] = []
    for local_anchor, local_labels in label_dict.items():
        global_anchor = int(anchor_node[local_anchor].item())
        for local_label in local_labels.tolist():
            pairs.append((global_anchor, int(label_node[local_label].item())))
    return sorted(pairs)


def _global_pair_set_from_edge_index(
    anchor_node: torch.Tensor,
    label_node: torch.Tensor,
    label_edge_index: torch.Tensor,
) -> list[tuple[int, int]]:
    """Convert a label edge index to sorted global-id pairs.

    Takes separate anchor and label node maps for the same reason as
    :func:`_global_pair_set`.
    """
    return sorted(
        (
            int(anchor_node[local_anchor].item()),
            int(label_node[local_label].item()),
        )
        for local_anchor, local_label in label_edge_index.t().tolist()
    )


def _collect_homogeneous_labels(
    _,
    return_dict,
    use_edge_index_output: bool,
    dataset: DistDataset,
    input_nodes: torch.Tensor,
    batch_size: int,
    has_negatives: bool,
):
    """Run one loader format and return its sorted global-id label pairs."""
    create_test_process_group()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=input_nodes,
        batch_size=batch_size,
        pin_memory_device=torch.device("cpu"),
        use_edge_index_output=use_edge_index_output,
    )
    positive_pairs: list[tuple[int, int]] = []
    negative_pairs: list[tuple[int, int]] = []
    for datum in loader:
        assert isinstance(datum, Data)
        node = datum.node
        if use_edge_index_output:
            assert isinstance(datum.y_positive, torch.Tensor)
            assert datum.y_positive.size(0) == 2
            positive_pairs.extend(
                _global_pair_set_from_edge_index(node, node, datum.y_positive)
            )
        else:
            positive_pairs.extend(_global_pair_set(node, node, datum.y_positive))
        if has_negatives:
            if use_edge_index_output:
                assert isinstance(datum.y_negative, torch.Tensor)
                assert datum.y_negative.size(0) == 2
                negative_pairs.extend(
                    _global_pair_set_from_edge_index(node, node, datum.y_negative)
                )
            else:
                negative_pairs.extend(_global_pair_set(node, node, datum.y_negative))
        else:
            assert not hasattr(datum, "y_negative"), (
                f"expected no negatives, got {getattr(datum, 'y_negative', None)}"
            )
    return_dict[use_edge_index_output] = (
        sorted(positive_pairs),
        sorted(negative_pairs),
    )
    shutdown_rpc()


def _edge_type_key(edge_type: EdgeType) -> tuple[str, ...]:
    """Canonical dict key for an edge type.

    Label edge types reach collation as plain tuples on some paths and as
    ``EdgeType`` on others, and the two stringify differently (``"('a', 'to',
    'b')"`` vs ``'a-to-b'``). Normalizing to plain strings keeps keys comparable
    across the process boundary regardless of which arrives.
    """
    return tuple(str(part) for part in edge_type)


def _accumulate_heterogeneous_pairs(
    data: HeteroData,
    labels_by_edge_type: dict[EdgeType, Union[torch.Tensor, dict[int, torch.Tensor]]],
    use_edge_index_output: bool,
    into: dict[tuple[str, ...], list[tuple[int, int]]],
) -> None:
    """Accumulate one batch's labels as global (anchor, label) id pairs.

    Anchors and supervision nodes are resolved through the node maps of the edge
    type's src and dst stores respectively, so a label remapped against the wrong
    node type surfaces as a wrong global id rather than passing silently.
    """
    anchor_index = 0
    supervision_index = 2
    for edge_type, labels in labels_by_edge_type.items():
        anchor_node = data[edge_type[anchor_index]].node
        label_node = data[edge_type[supervision_index]].node
        if use_edge_index_output:
            assert isinstance(labels, torch.Tensor), f"{edge_type}: {type(labels)}"
            assert labels.size(0) == 2
            pairs = _global_pair_set_from_edge_index(anchor_node, label_node, labels)
        else:
            assert not isinstance(labels, torch.Tensor), f"{edge_type}: {type(labels)}"
            pairs = _global_pair_set(anchor_node, label_node, labels)
        into[_edge_type_key(edge_type)].extend(pairs)


def _collect_heterogeneous_labels(
    _,
    return_dict,
    use_edge_index_output: bool,
    dataset: DistDataset,
    input_nodes: tuple[NodeType, torch.Tensor],
    supervision_edge_types: list[EdgeType],
    batch_size: int,
):
    """Run one loader format on a heterogeneous graph and return its label pairs.

    Results are keyed by ``str(edge_type)`` so they cross the process boundary as
    plain data.
    """
    create_test_process_group()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=input_nodes,
        batch_size=batch_size,
        pin_memory_device=torch.device("cpu"),
        supervision_edge_type=supervision_edge_types,
        use_edge_index_output=use_edge_index_output,
    )
    positive_pairs: dict[tuple[str, ...], list[tuple[int, int]]] = defaultdict(list)
    negative_pairs: dict[tuple[str, ...], list[tuple[int, int]]] = defaultdict(list)
    for datum in loader:
        assert isinstance(datum, HeteroData)
        # Several supervision edge types keep y_positive / y_negative in their
        # dict-of-edge-type form rather than collapsing to a bare value.
        _accumulate_heterogeneous_pairs(
            datum, datum.y_positive, use_edge_index_output, positive_pairs
        )
        _accumulate_heterogeneous_pairs(
            datum, datum.y_negative, use_edge_index_output, negative_pairs
        )
    return_dict[use_edge_index_output] = (
        {edge_type: sorted(pairs) for edge_type, pairs in positive_pairs.items()},
        {edge_type: sorted(pairs) for edge_type, pairs in negative_pairs.items()},
    )
    shutdown_rpc()


def _run_graph_store_validation_protocol(
    rank: int,
    init_method: str,
    scenario: str,
    results: Any,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
    )
    try:
        supervision_edge_type = (
            _A_TO_C if scenario == "schema_mismatch" and rank == 1 else _A_TO_B
        )
        ablp_input = ABLPInputNodes(
            anchor_nodes=torch.tensor([1]),
            anchor_node_type=_A,
            labels={supervision_edge_type: (torch.tensor([[2]]), None)},
        )
        if scenario == "malformed_payload" and rank == 1:
            object.__setattr__(ablp_input, "anchor_nodes", [1])
        dataset = MockRemoteDistDataset(
            num_storage_nodes=1,
            num_compute_nodes=2,
            edge_types=[message_passing_to_positive_label(supervision_edge_type)],
            edge_dir=(
                "sideways" if scenario == "invalid_direction" and rank == 1 else "out"
            ),
        )
        loader = DistABLPLoader.__new__(DistABLPLoader)
        loader._instance_count = 0
        loader._shutdowned = True
        with patch.object(
            BaseDistLoader, "create_graph_store_worker_options"
        ) as create_worker_options:
            try:
                loader._setup_for_graph_store(
                    input_nodes={0: ablp_input},
                    dataset=dataset,
                    num_workers=1,
                    worker_concurrency=1,
                    channel_size="1MB",
                    prefetch_size=1,
                )
            except ValueError as error:
                results[rank] = (str(error), create_worker_options.call_count)
            else:
                results[rank] = (
                    "no validation error",
                    create_worker_options.call_count,
                )
    finally:
        torch.distributed.destroy_process_group()


class DistABLPLoaderTest(TestCase):
    def tearDown(self):
        if torch.distributed.is_initialized():
            print("Destroying process group")
            # Ensure the process group is destroyed after each test
            # to avoid interference with subsequent tests
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_is_integral_tensor(self):
        for dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            with self.subTest(dtype=dtype):
                self.assertTrue(_is_integral_tensor(torch.empty(0, dtype=dtype)))
        for dtype in (torch.bool, torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.assertFalse(_is_integral_tensor(torch.empty(0, dtype=dtype)))

    def test_convert_graph_store_ablp_inputs_outward(self):
        positive_a_to_b = message_passing_to_positive_label(_A_TO_B)
        negative_a_to_b = message_passing_to_negative_label(_A_TO_B)
        positive_a_to_c = message_passing_to_positive_label(_A_TO_C)
        anchors = torch.tensor([1, 2])
        positive_b = torch.tensor([[10], [11]])
        negative_b = torch.tensor([[20, 21], [22, 23]])
        positive_c = torch.tensor([[30], [31]])
        input_nodes = {
            0: ABLPInputNodes(
                anchor_nodes=anchors,
                anchor_node_type=_A,
                labels={
                    _A_TO_B: (positive_b, negative_b),
                    _A_TO_C: (positive_c, None),
                },
            )
        }

        (
            sampler_inputs,
            input_type,
            supervision_edge_types,
            positive_label_edge_types,
            negative_label_edge_types,
        ) = _convert_graph_store_ablp_inputs(
            input_nodes=input_nodes,
            num_storage_nodes=2,
            edge_types=[
                positive_a_to_b,
                negative_a_to_b,
                positive_a_to_c,
            ],
            edge_dir="out",
        )

        self.assertEqual(input_type, _A)
        self.assertEqual(supervision_edge_types, [_A_TO_B, _A_TO_C])
        self.assertEqual(positive_label_edge_types, [positive_a_to_b, positive_a_to_c])
        self.assertEqual(negative_label_edge_types, [negative_a_to_b])
        self.assertLen(sampler_inputs, 2)

        supplied_input = sampler_inputs[0]
        self.assertIs(supplied_input.node, anchors)
        self.assertEqual(supplied_input.input_type, _A)
        self.assertEqual(
            set(supplied_input.positive_label_by_edge_types),
            {positive_a_to_b, positive_a_to_c},
        )
        self.assertIs(
            supplied_input.positive_label_by_edge_types[positive_a_to_b],
            positive_b,
        )
        self.assertIs(
            supplied_input.positive_label_by_edge_types[positive_a_to_c],
            positive_c,
        )
        self.assertEqual(
            set(supplied_input.negative_label_by_edge_types), {negative_a_to_b}
        )
        self.assertIs(
            supplied_input.negative_label_by_edge_types[negative_a_to_b],
            negative_b,
        )

        missing_input = sampler_inputs[1]
        self.assertEqual(missing_input.input_type, _A)
        self.assertEqual(missing_input.node.dtype, torch.long)
        self.assertEqual(tuple(missing_input.node.shape), (0,))
        self.assertEqual(
            set(missing_input.positive_label_by_edge_types),
            {positive_a_to_b, positive_a_to_c},
        )
        for labels in missing_input.positive_label_by_edge_types.values():
            self.assertEqual(labels.dtype, torch.long)
            self.assertEqual(tuple(labels.shape), (0, 0))
        self.assertEqual(
            set(missing_input.negative_label_by_edge_types), {negative_a_to_b}
        )
        self.assertEqual(
            missing_input.negative_label_by_edge_types[negative_a_to_b].dtype,
            torch.long,
        )
        self.assertEqual(
            tuple(missing_input.negative_label_by_edge_types[negative_a_to_b].shape),
            (0, 0),
        )

    def test_convert_graph_store_ablp_inputs_normalizes_base_placeholder(self):
        positive_edge_type = message_passing_to_positive_label(_A_TO_B)
        negative_edge_type = message_passing_to_negative_label(_A_TO_B)
        positive_labels = torch.empty((0, 0), dtype=torch.long)
        topology_complete_negative = torch.empty((0, 0), dtype=torch.long)
        sampler_inputs, *_ = _convert_graph_store_ablp_inputs(
            input_nodes={
                0: ABLPInputNodes(
                    anchor_nodes=torch.empty(0, dtype=torch.long),
                    anchor_node_type=_A,
                    labels={_A_TO_B: (positive_labels, None)},
                ),
                1: ABLPInputNodes(
                    anchor_nodes=torch.empty(0, dtype=torch.long),
                    anchor_node_type=_A,
                    labels={
                        _A_TO_B: (
                            torch.empty((0, 0), dtype=torch.long),
                            topology_complete_negative,
                        )
                    },
                ),
            },
            num_storage_nodes=2,
            edge_types=[positive_edge_type, negative_edge_type],
            edge_dir="out",
        )

        self.assertIs(
            sampler_inputs[0].positive_label_by_edge_types[positive_edge_type],
            positive_labels,
        )
        normalized_negative = sampler_inputs[0].negative_label_by_edge_types[
            negative_edge_type
        ]
        self.assertEqual(normalized_negative.dtype, torch.long)
        self.assertEqual(tuple(normalized_negative.shape), (0, 0))
        self.assertIs(
            sampler_inputs[1].negative_label_by_edge_types[negative_edge_type],
            topology_complete_negative,
        )

    def test_convert_graph_store_ablp_inputs_incoming(self):
        positive_b_to_a = message_passing_to_positive_label(_B_TO_A)
        negative_b_to_a = message_passing_to_negative_label(_B_TO_A)
        positive_c_to_a = message_passing_to_positive_label(_C_TO_A)
        (
            sampler_inputs,
            input_type,
            supervision_types,
            positive_types,
            negative_types,
        ) = _convert_graph_store_ablp_inputs(
            input_nodes={
                0: ABLPInputNodes(
                    anchor_nodes=torch.tensor([1]),
                    anchor_node_type=_A,
                    labels={
                        _B_TO_A: (torch.tensor([[10]]), torch.tensor([[20]])),
                        _C_TO_A: (torch.tensor([[30]]), None),
                    },
                )
            },
            num_storage_nodes=1,
            edge_types=[
                positive_b_to_a,
                negative_b_to_a,
                positive_c_to_a,
            ],
            edge_dir="in",
        )

        self.assertEqual(input_type, _A)
        self.assertEqual(supervision_types, [_B_TO_A, _C_TO_A])
        self.assertEqual(positive_types, [positive_b_to_a, positive_c_to_a])
        self.assertEqual(negative_types, [negative_b_to_a])
        self.assertEqual(
            list(sampler_inputs[0].positive_label_by_edge_types),
            [positive_b_to_a, positive_c_to_a],
        )
        self.assertEqual(
            list(sampler_inputs[0].negative_label_by_edge_types),
            [negative_b_to_a],
        )

    def test_convert_graph_store_ablp_inputs_rejects_invalid_inputs(self):
        positive_edge_type = message_passing_to_positive_label(_A_TO_B)
        negative_edge_type = message_passing_to_negative_label(_A_TO_B)
        positive_c_edge_type = message_passing_to_positive_label(_A_TO_C)
        topology = [positive_edge_type, negative_edge_type, positive_c_edge_type]

        def make_input(
            anchor_nodes: Any,
            labels: Any,
            anchor_node_type: NodeType = _A,
        ) -> ABLPInputNodes:
            return ABLPInputNodes(
                anchor_nodes=anchor_nodes,
                anchor_node_type=anchor_node_type,
                labels=labels,
            )

        valid_input = make_input(
            torch.tensor([1]),
            {_A_TO_B: (torch.tensor([[2]]), torch.tensor([[3]]))},
        )
        cases = [
            ("empty input", {}, 1, topology, "out"),
            (
                "empty labels",
                {0: make_input(torch.tensor([], dtype=torch.long), {})},
                1,
                topology,
                "out",
            ),
            ("negative rank", {-1: valid_input}, 1, topology, "out"),
            ("out of range rank", {1: valid_input}, 1, topology, "out"),
            ("missing edge types", {0: valid_input}, 1, None, "out"),
            (
                "missing positive topology",
                {0: valid_input},
                1,
                [negative_edge_type],
                "out",
            ),
            (
                "wrong outward anchor endpoint",
                {
                    0: make_input(
                        torch.tensor([1]),
                        {_B_TO_A: (torch.tensor([[2]]), None)},
                    )
                },
                1,
                [message_passing_to_positive_label(_B_TO_A)],
                "out",
            ),
            (
                "wrong incoming anchor endpoint",
                {
                    0: make_input(
                        torch.tensor([1]),
                        {_A_TO_B: (torch.tensor([[2]]), None)},
                    )
                },
                1,
                [positive_edge_type],
                "in",
            ),
            (
                "inconsistent anchor types",
                {
                    0: valid_input,
                    1: make_input(
                        torch.tensor([2]),
                        {_A_TO_B: (torch.tensor([[3]]), torch.tensor([[4]]))},
                        _B,
                    ),
                },
                2,
                topology,
                "out",
            ),
            (
                "inconsistent supervision keys",
                {
                    0: valid_input,
                    1: make_input(
                        torch.tensor([2]),
                        {_A_TO_C: (torch.tensor([[3]]), None)},
                    ),
                },
                2,
                topology,
                "out",
            ),
            (
                "non tensor anchors",
                {0: make_input([1], valid_input.labels)},
                1,
                topology,
                "out",
            ),
            (
                "non 1-D anchors",
                {0: make_input(torch.tensor([[1]]), valid_input.labels)},
                1,
                topology,
                "out",
            ),
            (
                "floating anchors",
                {0: make_input(torch.tensor([1.0]), valid_input.labels)},
                1,
                topology,
                "out",
            ),
            (
                "boolean anchors",
                {0: make_input(torch.tensor([True]), valid_input.labels)},
                1,
                topology,
                "out",
            ),
            (
                "None positive labels",
                {
                    0: make_input(
                        torch.tensor([1]), {_A_TO_B: (None, torch.tensor([[3]]))}
                    )
                },
                1,
                topology,
                "out",
            ),
            (
                "non tensor positive labels",
                {
                    0: make_input(
                        torch.tensor([1]), {_A_TO_B: ([2], torch.tensor([[3]]))}
                    )
                },
                1,
                topology,
                "out",
            ),
            (
                "non tensor negative labels",
                {
                    0: make_input(
                        torch.tensor([1]), {_A_TO_B: (torch.tensor([[2]]), [3])}
                    )
                },
                1,
                topology,
                "out",
            ),
            (
                "non 2-D positive labels",
                {
                    0: make_input(
                        torch.tensor([1]),
                        {_A_TO_B: (torch.tensor([2]), torch.tensor([[3]]))},
                    )
                },
                1,
                topology,
                "out",
            ),
            (
                "non 2-D negative labels",
                {
                    0: make_input(
                        torch.tensor([1]),
                        {_A_TO_B: (torch.tensor([[2]]), torch.tensor([3]))},
                    )
                },
                1,
                topology,
                "out",
            ),
            (
                "floating positive labels",
                {
                    0: make_input(
                        torch.tensor([1]),
                        {_A_TO_B: (torch.tensor([[2.0]]), torch.tensor([[3]]))},
                    )
                },
                1,
                topology,
                "out",
            ),
            (
                "boolean negative labels",
                {
                    0: make_input(
                        torch.tensor([1]),
                        {_A_TO_B: (torch.tensor([[2]]), torch.tensor([[True]]))},
                    )
                },
                1,
                topology,
                "out",
            ),
            (
                "positive row mismatch",
                {
                    0: make_input(
                        torch.tensor([1, 2]),
                        {_A_TO_B: (torch.tensor([[3]]), torch.tensor([[4], [5]]))},
                    )
                },
                1,
                topology,
                "out",
            ),
            (
                "negative row mismatch",
                {
                    0: make_input(
                        torch.tensor([1, 2]),
                        {_A_TO_B: (torch.tensor([[3], [4]]), torch.tensor([[5]]))},
                    )
                },
                1,
                topology,
                "out",
            ),
            (
                "missing required negatives",
                {
                    0: make_input(
                        torch.tensor([1]), {_A_TO_B: (torch.tensor([[2]]), None)}
                    )
                },
                1,
                topology,
                "out",
            ),
            (
                "unexpected negatives",
                {0: valid_input},
                1,
                [positive_edge_type],
                "out",
            ),
        ]
        for name, input_nodes, num_storage_nodes, edge_types, edge_dir in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    _convert_graph_store_ablp_inputs(
                        input_nodes=input_nodes,
                        num_storage_nodes=num_storage_nodes,
                        edge_types=edge_types,
                        edge_dir=edge_dir,  # ty: ignore[invalid-argument-type]
                    )

    def test_convert_graph_store_ablp_inputs_rejects_invalid_direction(self):
        with self.assertRaises(ValueError):
            _convert_graph_store_ablp_inputs(
                input_nodes={
                    0: ABLPInputNodes(
                        anchor_nodes=torch.tensor([1]),
                        anchor_node_type=_A,
                        labels={_A_TO_B: (torch.tensor([[2]]), None)},
                    )
                },
                num_storage_nodes=1,
                edge_types=[message_passing_to_positive_label(_A_TO_B)],
                edge_dir="sideways",  # ty: ignore[invalid-argument-type]
            )

    def test_graph_store_setup_retains_single_supervision_gate(self):
        create_test_process_group()
        loader = DistABLPLoader.__new__(DistABLPLoader)
        loader._instance_count = 0
        loader._shutdowned = True
        dataset = MockRemoteDistDataset(
            num_storage_nodes=1,
            edge_types=[
                message_passing_to_positive_label(_A_TO_B),
                message_passing_to_positive_label(_A_TO_C),
            ],
        )
        with patch.object(
            BaseDistLoader, "create_graph_store_worker_options"
        ) as create_worker_options:
            with self.assertRaisesRegex(
                ValueError,
                "Graph Store mode currently only supports a single supervision edge type",
            ):
                loader._setup_for_graph_store(
                    input_nodes={
                        0: ABLPInputNodes(
                            anchor_nodes=torch.tensor([1]),
                            anchor_node_type=_A,
                            labels={
                                _A_TO_B: (torch.tensor([[2]]), None),
                                _A_TO_C: (torch.tensor([[3]]), None),
                            },
                        )
                    },
                    dataset=dataset,
                    num_workers=1,
                    worker_concurrency=1,
                    channel_size="1MB",
                    prefetch_size=1,
                )
        create_worker_options.assert_not_called()

    @parameterized.expand(
        [
            param("malformed peer payload", "malformed_payload", "rank 1:"),
            param("invalid peer direction", "invalid_direction", "rank 1:"),
            param("different peer schema", "schema_mismatch", "schemas differ"),
        ]
    )
    def test_graph_store_setup_synchronizes_validation(
        self, _: str, scenario: str, expected_error: str
    ):
        manager = mp.Manager()
        results = manager.dict()
        mp.spawn(
            fn=_run_graph_store_validation_protocol,
            args=(get_process_group_init_method(), scenario, results),
            nprocs=2,
            join=True,
        )

        self.assertLen(results, 2)
        for rank in range(2):
            error, worker_option_calls = results[rank]
            self.assertIn(expected_error, error)
            self.assertEqual(worker_option_calls, 0)

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
            # Anchor 11 has message-passing edges (11 -> {13, 17}) but is the
            # source of NO positive-label edge, so its positive-label row is
            # all-padding and y_positive[11] is a guaranteed-empty tensor. This
            # exercises the empty-anchor branch end-to-end for both outputs.
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
            ),
        ]
    )
    def test_edge_index_output_matches_dict_output(
        self, _, labeled_edges, input_nodes, batch_size, has_negatives
    ):
        """Both output formats contain the same global anchor-label pairs."""
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
        for use_edge_index_output in (False, True):
            mp.spawn(
                fn=_collect_homogeneous_labels,
                args=(
                    return_dict,
                    use_edge_index_output,
                    dataset,
                    input_nodes,
                    batch_size,
                    has_negatives,
                ),
            )
        self.assertEqual(return_dict[False][0], return_dict[True][0])
        self.assertEqual(return_dict[False][1], return_dict[True][1])

    @parameterized.expand(
        [
            param(
                "edge_dir=out",
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
                },
            ),
            # edge_dir="in" stores the supervision edge types reversed, so their
            # dst node type is the anchor type rather than the supervision type,
            # while the label edge types reaching collation stay outward-facing.
            # A loader that resolves supervision nodes off the former instead of
            # the latter fails here and passes the edge_dir="out" case above.
            param(
                "edge_dir=in",
                edge_dir="in",
                edge_index={
                    _B_TO_A: torch.tensor([[11, 12], [10, 10]]),
                    message_passing_to_positive_label(_B_TO_A): torch.tensor(
                        [[13, 14], [10, 10]]
                    ),
                    message_passing_to_negative_label(_B_TO_A): torch.tensor(
                        [[15, 16], [10, 10]]
                    ),
                    _C_TO_A: torch.tensor([[20, 21], [10, 10]]),
                    message_passing_to_positive_label(_C_TO_A): torch.tensor(
                        [[22, 23], [10, 10]]
                    ),
                },
            ),
        ]
    )
    def test_heterogeneous_edge_index_output_matches_dict_output(
        self,
        _,
        edge_dir: Literal["in", "out"],
        edge_index: dict[EdgeType, torch.Tensor],
    ):
        """Both output formats agree on a heterogeneous graph, for either edge_dir.

        Anchors are node type ``a``; supervision nodes are ``b`` and ``c``. Because
        the anchor and supervision node maps are distinct, a label remapped against
        the wrong node type yields a wrong global id here instead of going unnoticed.
        """
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
        for use_edge_index_output in (False, True):
            mp.spawn(
                fn=_collect_heterogeneous_labels,
                args=(
                    return_dict,
                    use_edge_index_output,
                    dataset,
                    (_A, torch.tensor([10])),
                    [_A_TO_B, _A_TO_C],
                    1,  # batch_size
                ),
            )
        # Both formats resolve to the same global ids...
        self.assertEqual(dict(return_dict[False][0]), dict(return_dict[True][0]))
        self.assertEqual(dict(return_dict[False][1]), dict(return_dict[True][1]))
        # ...and did so over labels that actually exist, so equality above cannot
        # hold vacuously.
        self.assertEqual(
            dict(return_dict[True][0]),
            {
                _edge_type_key(_A_TO_B): [(10, 13), (10, 14)],
                _edge_type_key(_A_TO_C): [(10, 22), (10, 23)],
            },
        )
        self.assertEqual(
            dict(return_dict[True][1]),
            {
                _edge_type_key(_A_TO_B): [(10, 15), (10, 16)],
            },
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
