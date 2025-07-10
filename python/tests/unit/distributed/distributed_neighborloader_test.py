import unittest
from collections import abc
from typing import Optional, Union

import torch
import torch.multiprocessing as mp
from graphlearn_torch.distributed import shutdown_rpc
from parameterized import param, parameterized
from torch_geometric.data import Data, HeteroData

import gigl.distributed.utils
from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_ablp_neighborloader import DistABLPLoader
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.distributed.dist_partitioner import DistPartitioner
from gigl.distributed.dist_range_partitioner import DistRangePartitioner
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
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
from gigl.utils.iterator import InfiniteIterator
from tests.test_assets.distributed.run_distributed_dataset import (
    run_distributed_dataset,
)
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    get_process_group_init_method,
)

_POSITIVE_EDGE_TYPE = message_passing_to_positive_label(DEFAULT_HOMOGENEOUS_EDGE_TYPE)
_NEGATIVE_EDGE_TYPE = message_passing_to_negative_label(DEFAULT_HOMOGENEOUS_EDGE_TYPE)

# TODO(svij) - swap the DistNeighborLoader tests to not user context/local_process_rank/local_process_world_size.

# GLT requires subclasses of DistNeighborLoader to be run in a separate process. Otherwise, we may run into segmentation fault
# or other memory issues. Calling these functions in separate proceses also allows us to use shutdown_rpc() to ensure cleanup of
# ports, providing stronger guarantees of isolation between tests.


# We require each of these functions to accept local_rank as the first argument since we use mp.spawn with `nprocs=1`
def _run_distributed_neighbor_loader(
    _,
    dataset: DistLinkPredictionDataset,
    context: DistributedContext,
    expected_data_count: int,
):
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        context=context,
        local_process_rank=0,
        local_process_world_size=1,
        pin_memory_device=torch.device("cpu"),
    )

    count = 0
    for datum in loader:
        assert isinstance(datum, Data)
        count += 1

    # Cora has 2708 nodes, make sure we go over all of them.
    # https://paperswithcode.com/dataset/cora
    assert count == expected_data_count

    shutdown_rpc()


def _run_distributed_neighbor_loader_labeled_homogeneous(
    _,
    dataset: DistLinkPredictionDataset,
    context: DistributedContext,
    expected_data_count: int,
):
    assert isinstance(dataset.node_ids, abc.Mapping)
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=to_homogeneous(dataset.node_ids),
        num_neighbors=[2, 2],
        context=context,
        local_process_rank=0,
        local_process_world_size=1,
        pin_memory_device=torch.device("cpu"),
    )

    count = 0
    for datum in loader:
        assert isinstance(datum, Data)
        count += 1

    assert (
        count == expected_data_count
    ), f"Expected {expected_data_count} batches, but got {count}."

    shutdown_rpc()


def _run_infinite_distributed_neighbor_loader(
    _,
    dataset: DistLinkPredictionDataset,
    context: DistributedContext,
    max_num_batches: int,
):
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        context=context,
        local_process_rank=0,
        local_process_world_size=1,
        pin_memory_device=torch.device("cpu"),
    )

    infinite_loader: InfiniteIterator = InfiniteIterator(loader)

    count = 0
    for datum in infinite_loader:
        assert isinstance(datum, Data)
        count += 1
        if count == max_num_batches:
            break

    # Ensure we have looped through the dataloader for the max number of batches
    assert count == max_num_batches

    shutdown_rpc()


def _run_distributed_heterogeneous_neighbor_loader(
    _,
    dataset: DistLinkPredictionDataset,
    context: DistributedContext,
    expected_data_count: int,
):
    assert isinstance(dataset.node_ids, abc.Mapping)
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(NodeType("author"), dataset.node_ids[NodeType("author")]),
        num_neighbors=[2, 2],
        context=context,
        local_process_rank=0,
        local_process_world_size=1,
        pin_memory_device=torch.device("cpu"),
    )

    count = 0
    for datum in loader:
        assert isinstance(datum, HeteroData)
        count += 1

    assert count == expected_data_count

    shutdown_rpc()


def _run_distributed_ablp_neighbor_loader(
    _,
    dataset: DistLinkPredictionDataset,
    context: DistributedContext,
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
    for local_anchor in datum.y_positive:
        global_id = datum.node[local_anchor].item()
        global_positive_nodes = datum.node[datum.y_positive[local_anchor]]
        expected_positive_label = expected_positive_labels[global_id]
        assert_tensor_equality(
            global_positive_nodes,
            expected_positive_label,
            dim=0,
        )
    if expected_negative_labels is not None:
        for local_anchor in datum.y_negative:
            global_id = datum.node[local_anchor].item()
            global_negative_nodes = datum.node[datum.y_negative[local_anchor]]
            expected_negative_label = expected_negative_labels[global_id]
            assert_tensor_equality(
                global_negative_nodes,
                expected_negative_label,
                dim=0,
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
    dataset: DistLinkPredictionDataset,
    context: DistributedContext,
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


def _run_multiple_neighbor_loader(
    _,
    dataset: DistLinkPredictionDataset,
    context: DistributedContext,
    expected_data_count: int,
):
    torch.distributed.init_process_group(
        rank=0, world_size=1, init_method=get_process_group_init_method()
    )
    loader_one = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
    )

    loader_two = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
    )

    count = 0
    for datum_one, datum_two in zip(loader_one, loader_two):
        count += 1

    # Cora has 2708 nodes, make sure we go over all of them.
    # https://paperswithcode.com/dataset/cora
    assert count == expected_data_count

    loader_one.shutdown()
    loader_two.shutdown()

    loader_three = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
    )

    count = 0
    for datum_three in loader_three:
        count += 1

    assert count == expected_data_count

    shutdown_rpc()


def _run_dblp_supervised(
    _,
    dataset: DistLinkPredictionDataset,
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
    dataset: DistLinkPredictionDataset,
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


class DistributedNeighborLoaderTest(unittest.TestCase):
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

    def test_distributed_neighbor_loader(self):
        expected_data_count = 2708
        port = gigl.distributed.utils.get_free_port()

        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
            should_load_tensors_in_parallel=True,
            _port=port,
        )

        mp.spawn(
            fn=_run_distributed_neighbor_loader,
            args=(dataset, self._context, expected_data_count),
        )

    def test_infinite_distributed_neighbor_loader(self):
        port = gigl.distributed.utils.get_free_port()
        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
            should_load_tensors_in_parallel=True,
            _port=port,
        )

        assert isinstance(dataset.node_ids, torch.Tensor)

        num_nodes = dataset.node_ids.size(0)

        # Let's ensure we can iterate across the dataset twice with the infinite iterator
        max_num_batches = num_nodes * 2

        mp.spawn(
            fn=_run_infinite_distributed_neighbor_loader,
            args=(dataset, self._context, max_num_batches),
        )

    # TODO: (svij) - Figure out why this test is failing on Google Cloud Build
    @unittest.skip("Failing on Google Cloud Build - skiping for now")
    def test_distributed_neighbor_loader_heterogeneous(self):
        expected_data_count = 4057

        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            should_load_tensors_in_parallel=True,
        )

        mp.spawn(
            fn=_run_distributed_heterogeneous_neighbor_loader,
            args=(dataset, self._context, expected_data_count),
        )

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
        )
        dataset = DistLinkPredictionDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_distributed_ablp_neighbor_loader,
            args=(
                dataset,
                self._context,
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
                self._context,
                to_homogeneous(
                    dataset.train_node_ids
                ).numel(),  # Use to_homogeneous to make MyPy happy since dataset.train_node_ids is a dict.
            ),
        )

    def test_random_loading_labeled_homogeneous(self):
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
        assert isinstance(dataset.node_ids, abc.Mapping)
        mp.spawn(
            fn=_run_distributed_neighbor_loader_labeled_homogeneous,
            args=(dataset, self._context, to_homogeneous(dataset.node_ids).size(0)),
        )

    def test_multiple_neighbor_loader(self):
        port = gigl.distributed.utils.get_free_port()
        expected_data_count = 2708

        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
            should_load_tensors_in_parallel=True,
            _port=port,
        )

        mp.spawn(
            fn=_run_multiple_neighbor_loader,
            args=(dataset, self._context, expected_data_count),
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
                "Range-based partitioning, list fanout",
                partitioner_class=DistPartitioner,
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


if __name__ == "__main__":
    unittest.main()
