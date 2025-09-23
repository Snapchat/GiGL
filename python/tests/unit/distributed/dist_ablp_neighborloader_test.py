import collections
import unittest
from typing import Optional, Union

import torch
import torch.multiprocessing as mp
from graphlearn_torch.distributed import shutdown_rpc
from graphlearn_torch.utils import reverse_edge_type
from parameterized import param, parameterized
from torch_geometric.data import Data, HeteroData

from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_ablp_neighborloader import DistABLPLoader
from gigl.distributed.dist_context import DistributedContext
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
from gigl.utils.data_splitters import HashedNodeAnchorLinkSplitter
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    get_process_group_init_method,
)

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

# TODO(svij) - swap the DistNeighborLoader tests to not user context/local_process_rank/local_process_world_size.

# GLT requires subclasses of DistNeighborLoader to be run in a separate process. Otherwise, we may run into segmentation fault
# or other memory issues. Calling these functions in separate proceses also allows us to use shutdown_rpc() to ensure cleanup of
# ports, providing stronger guarantees of isolation between tests.


# We require each of these functions to accept local_rank as the first argument since we use mp.spawn with `nprocs=1`


def _assert_labels(
    node: torch.Tensor, y: dict[int, torch.Tensor], expected: dict[int, torch.Tensor]
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
        node (torch.Tensor): The node tensor, N, where N is the number of nodes in the batch,
        where the local IDs are the index of a node in the tensor,
        and the global IDs are the values of the tensor.
        y (dict[int, torch.Tensor]): The labels in the local node space.
            The tensors are of shape [X], where X is the number of labels for the current anchor node.
        expected (dict[int, torch.Tensor]): The labels in the global node space.
            The tensors are of shape [X], where X is the number of labels for the current anchor node.
    Raises if:
    - The keys in `y` do not match the keys in `expected`
    - The values in `y` do not match the values in `expected`
    """
    supplied_global_nodes = node[list(y.keys())]
    assert set(supplied_global_nodes.tolist()) == set(
        expected.keys()
    ), f"Expected keys {expected.keys()} != {y.keys()}"
    for local_anchor in y:
        global_id = int(node[local_anchor].item())
        global_nodes = node[y[local_anchor]]
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

    torch.distributed.init_process_group(
        rank=0, world_size=1, init_method=get_process_group_init_method()
    )
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
    _assert_labels(datum.node, datum.y_positive, expected_positive_labels)
    if expected_negative_labels is not None:
        _assert_labels(datum.node, datum.y_negative, expected_negative_labels)
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
    torch.distributed.init_process_group(
        rank=0, world_size=1, init_method=get_process_group_init_method()
    )
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
    context: DistributedContext,
    supervision_edge_types: list[EdgeType],
):
    assert (
        len(supervision_edge_types) == 1
    ), "TODO (mkolodner-sc): Support multiple supervision edge types in dataloading"
    supervision_edge_type = supervision_edge_types[0]
    anchor_node_type = supervision_edge_type.src_node_type
    supervision_node_type = supervision_edge_type.dst_node_type
    assert isinstance(dataset.train_node_ids, dict)
    assert isinstance(dataset.graph, dict)
    fanout = [2, 2]
    num_neighbors = {edge_type: fanout for edge_type in dataset.graph.keys()}
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=num_neighbors,
        input_nodes=(anchor_node_type, dataset.train_node_ids[anchor_node_type]),
        context=context,
        local_process_rank=0,
        local_process_world_size=1,
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
    assert count == dataset.train_node_ids[anchor_node_type].size(0)

    shutdown_rpc()


def _run_toy_heterogeneous_ablp(
    _,
    dataset: DistDataset,
    context: DistributedContext,
    supervision_edge_types: list[EdgeType],
    fanout: Union[list[int], dict[EdgeType, list[int]]],
):
    anchor_node_type = NodeType("user")
    supervision_node_type = NodeType("story")
    assert (
        len(supervision_edge_types) == 1
    ), "TODO (mkolodner-sc): Support multiple supervision edge types in dataloading"
    supervision_edge_type = supervision_edge_types[0]
    assert isinstance(dataset.train_node_ids, dict)
    assert isinstance(dataset.graph, dict)
    labeled_edge_type = EdgeType(
        supervision_node_type, Relation("to_gigl_positive"), anchor_node_type
    )
    all_positive_supervision_nodes, all_anchor_nodes, _, _ = dataset.graph[
        labeled_edge_type
    ].topo.to_coo()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=fanout,
        input_nodes=(anchor_node_type, dataset.train_node_ids[anchor_node_type]),
        context=context,
        local_process_rank=0,
        local_process_world_size=1,
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
        dataset.train_node_ids[anchor_node_type], datum[anchor_node_type].batch
    )
    assert (
        dataset.train_node_ids[anchor_node_type].size(0)
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
    dataset: DistDataset,
    supervision_edge_types: list[EdgeType],
    expected_node: dict[NodeType, torch.Tensor],
    expected_batch: dict[NodeType, torch.Tensor],
    expected_edges: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]],
    expected_positive_labels: dict[EdgeType, dict[int, torch.Tensor]],
    expected_negative_labels: Optional[dict[EdgeType, dict[int, torch.Tensor]]],
):
    input_nodes = (NodeType("a"), torch.tensor([10]))
    batch_size = 1

    torch.distributed.init_process_group(
        rank=0, world_size=1, init_method=get_process_group_init_method()
    )
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
    assert set(datum.y_positive.keys()) == set(expected_positive_labels.keys())
    for edge_type in datum.y_positive.keys():
        for local_anchor in datum.y_positive[edge_type]:
            global_id = datum[edge_type[0]].node[local_anchor].item()
            global_positive_nodes = datum[edge_type[2]].node[
                datum.y_positive[edge_type][local_anchor]
            ]
            expected_positive_label = expected_positive_labels[edge_type][global_id]
            assert_tensor_equality(
                global_positive_nodes,
                expected_positive_label,
                dim=0,
            )
    if expected_negative_labels is not None:
        assert datum.y_negative.keys() == expected_negative_labels.keys()
        for edge_type in datum.y_negative.keys():
            for local_anchor in datum.y_negative[edge_type]:
                global_id = datum[edge_type[0]].node[local_anchor].item()
                global_negative_nodes = datum[edge_type[2]].node[
                    datum.y_negative[edge_type][local_anchor]
                ]
                expected_negative_label = expected_negative_labels[edge_type][global_id]
                assert_tensor_equality(
                    global_negative_nodes,
                    expected_negative_label,
                    dim=0,
                )
    else:
        assert not hasattr(datum, "y_negative")

    # Reverse as the dataset edge dir is "out" so GLT reverses under the hood.
    expected_edges = {
        reverse_edge_type(edge_type): edges
        for edge_type, edges in expected_edges.items()
    }
    dsts, srcs, *_ = datum.coo()
    assert set(expected_edges.keys()) == set(dsts.keys())
    assert set(expected_edges.keys()) == set(srcs.keys())
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


class DistABLPLoaderTest(unittest.TestCase):
    def setUp(self):
        self._master_ip_address = "localhost"
        self._world_size = 1
        self._num_rpc_threads = 4

        self._context = DistributedContext(
            main_worker_ip_address=self._master_ip_address,
            global_rank=0,
            global_world_size=self._world_size,
        )

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
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
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

        splitter = HashedNodeAnchorLinkSplitter(
            sampling_direction="in", should_convert_labels_to_edges=True
        )

        dataset = build_dataset(
            serialized_graph_metadata=serialized_graph_metadata,
            distributed_context=self._context,
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

        splitter = HashedNodeAnchorLinkSplitter(
            sampling_direction="in",
            supervision_edge_types=supervision_edge_types,
            should_convert_labels_to_edges=True,
        )

        dataset = build_dataset(
            serialized_graph_metadata=serialized_graph_metadata,
            distributed_context=self._context,
            sample_edge_direction="in",
            _ssl_positive_label_percentage=0.1,
            splitter=splitter,
        )

        mp.spawn(
            fn=_run_dblp_supervised,
            args=(dataset, self._context, supervision_edge_types),
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

        splitter = HashedNodeAnchorLinkSplitter(
            sampling_direction="in",
            supervision_edge_types=supervision_edge_types,
            should_convert_labels_to_edges=True,
        )

        dataset = build_dataset(
            serialized_graph_metadata=serialized_graph_metadata,
            distributed_context=self._context,
            sample_edge_direction="in",
            _ssl_positive_label_percentage=0.1,
            splitter=splitter,
            partitioner_class=partitioner_class,
        )

        mp.spawn(
            fn=_run_toy_heterogeneous_ablp,
            args=(dataset, self._context, supervision_edge_types, fanout),
        )

    @parameterized.expand(
        [
            param(
                "positive edges",
                edge_index={
                    _A_TO_B: torch.tensor([[10, 10], [11, 12]]),
                    message_passing_to_positive_label(_A_TO_B): torch.tensor(
                        [[10, 10], [13, 14]]
                    ),
                    _A_TO_C: torch.tensor([[10, 10], [20, 21]]),
                    message_passing_to_positive_label(_A_TO_C): torch.tensor(
                        [[10, 10], [22, 23]]
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
                    _A_TO_B: (torch.tensor([10, 10]), torch.tensor([11, 12])),
                    _A_TO_C: (torch.tensor([10, 10]), torch.tensor([20, 21])),
                },
                expected_positive_labels={
                    _A_TO_B: {10: torch.tensor([13, 14])},
                    _A_TO_C: {10: torch.tensor([22, 23])},
                },
                expected_negative_labels=None,
            ),
            param(
                "positive and negative edges",
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
            param(
                "same nodes, different relation",
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
            param(
                "edges that aren't supervision edge types",
                edge_index={
                    _A_TO_B: torch.tensor([[10, 10], [11, 12]]),
                    message_passing_to_positive_label(_A_TO_B): torch.tensor(
                        [[10, 10], [13, 14]]
                    ),
                    _A_TO_C: torch.tensor([[10, 10], [20, 21]]),
                },
                supervision_edge_types=[_A_TO_B],
                expected_node={
                    _A: torch.tensor([10]),
                    _B: torch.tensor([11, 12, 13, 14]),
                    _C: torch.tensor([20, 21]),
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
                },
                expected_negative_labels=None,
            ),
        ]
    )
    def test_ablp_dataloder_multiple_supervision_edge_types(
        self,
        _,
        edge_index: dict[EdgeType, torch.Tensor],
        supervision_edge_types: list[EdgeType],
        expected_node: dict[NodeType, torch.Tensor],
        expected_batch: dict[NodeType, Optional[torch.Tensor]],
        expected_edges: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]],
        expected_positive_labels: dict[EdgeType, dict[int, torch.Tensor]],
        expected_negative_labels: Optional[dict[EdgeType, dict[int, torch.Tensor]]],
    ):
        nodes: dict[NodeType, list[torch.Tensor]] = collections.defaultdict(list)
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
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)
        mp.spawn(
            fn=_run_distributed_ablp_neighbor_loader_multiple_supervision_edge_types,
            args=(
                dataset,  # dataset
                supervision_edge_types,
                expected_node,
                expected_batch,
                expected_edges,
                expected_positive_labels,
                expected_negative_labels,
            ),
        ),


if __name__ == "__main__":
    unittest.main()
