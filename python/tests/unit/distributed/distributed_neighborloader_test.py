import unittest
from collections.abc import Mapping

import torch
import torch.multiprocessing as mp
from graphlearn_torch.distributed import shutdown_rpc
from torch_geometric.data import Data, HeteroData

import gigl.distributed.utils
from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
    DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    to_homogeneous,
)
from gigl.utils.data_splitters import HashedNodeAnchorLinkSplitter, HashedNodeSplitter
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

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)
_STORY_TO_USER = EdgeType(_STORY, Relation("to"), _USER)

# TODO(svij) - swap the DistNeighborLoader tests to not user context/local_process_rank/local_process_world_size.

# GLT requires subclasses of DistNeighborLoader to be run in a separate process. Otherwise, we may run into segmentation fault
# or other memory issues. Calling these functions in separate proceses also allows us to use shutdown_rpc() to ensure cleanup of
# ports, providing stronger guarantees of isolation between tests.


# We require each of these functions to accept local_rank as the first argument since we use mp.spawn with `nprocs=1`
def _run_distributed_neighbor_loader(
    _,
    dataset: DistDataset,
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
    dataset: DistDataset,
    context: DistributedContext,
    expected_data_count: int,
):
    assert isinstance(dataset.node_ids, Mapping)
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
    dataset: DistDataset,
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
    dataset: DistDataset,
    context: DistributedContext,
    expected_data_count: int,
):
    assert isinstance(dataset.node_ids, Mapping)
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


def _run_multiple_neighbor_loader(
    _,
    dataset: DistDataset,
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


def _run_distributed_neighbor_loader_with_node_labels_homogeneous(
    _,
    dataset: DistDataset,
    batch_size: int,
):
    torch.distributed.init_process_group(
        rank=0, world_size=1, init_method=get_process_group_init_method()
    )

    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )

    for datum in loader:
        assert isinstance(
            datum, Data
        ), f"Subgraph should be a Data for homogeneous datasets, got {type(datum)}"
        assert hasattr(datum, "y"), "Subgraph is missing the `y` attribute for labels"
        # For this mocked data, the value of each label is equal to its Node ID
        assert_tensor_equality(datum.y, datum.node)

    shutdown_rpc()


def _run_distributed_neighbor_loader_with_node_labels_heterogeneous(
    _,
    dataset: DistDataset,
    batch_size: int,
):
    torch.distributed.init_process_group(
        rank=0, world_size=1, init_method=get_process_group_init_method()
    )

    assert isinstance(dataset.node_ids, Mapping)

    user_loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(_USER, dataset.node_ids[_USER]),
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )

    story_loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(_STORY, dataset.node_ids[_STORY]),
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )

    for user_datum, story_datum in zip(user_loader, story_loader):
        # For this mocked data, the value of each user/story label is equal to its corresponding Node ID
        assert isinstance(
            user_datum, HeteroData
        ), f"User subgraph should be a HeteroData for heterogeneous datasets, got {type(user_datum)}"
        assert hasattr(
            user_datum[_USER], "y"
        ), "User subgraph is missing the 'y' attribute for labels"
        assert_tensor_equality(user_datum[_USER].y, user_datum[_USER].node)

        assert isinstance(
            story_datum, HeteroData
        ), f"Story subgraph should be a HeteroData for heterogeneous datasets, got {type(story_datum)}"
        assert hasattr(
            story_datum[_STORY], "y"
        ), "Story subgraph is missing the 'y' attribute for labels"
        assert_tensor_equality(story_datum[_STORY].y, story_datum[_STORY].node)

    shutdown_rpc()


def _run_cora_supervised_node_classification(
    _,
    dataset: DistDataset,
    batch_size: int,
):
    """Run CORA supervised node classification test using DistNeighborLoader."""
    torch.distributed.init_process_group(
        rank=0, world_size=1, init_method=get_process_group_init_method()
    )

    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=to_homogeneous(dataset.train_node_ids),
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )

    for datum in loader:
        assert isinstance(
            datum, Data
        ), f"Subgraph should be a Data for homogeneous datasets, got {type(datum)}"
        assert hasattr(
            datum, "y"
        ), "Node labels should be present for supervised node classification"
        assert datum.y.size(0) == datum.node.size(
            0
        ), f"Number of labels should match number of nodes, got {datum.y.size(0)} labels and {datum.node.size(0)} nodes"

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
        assert isinstance(dataset.node_ids, Mapping)
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

    def test_distributed_neighbor_loader_with_node_labels_homogeneous(self):
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(5),
            edge_partition_book=torch.zeros(5),
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
                edge_ids=None,
            ),
            partitioned_node_features=FeaturePartitionData(
                feats=torch.zeros(10, 2), ids=torch.arange(10)
            ),
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=FeaturePartitionData(
                feats=torch.arange(10).unsqueeze(1), ids=torch.arange(10)
            ),
        )

        dataset = DistDataset(rank=0, world_size=1, edge_dir="in")

        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_distributed_neighbor_loader_with_node_labels_homogeneous,
            args=(dataset, 1),  # dataset  # batch_size
        )

    def test_distributed_neighbor_loader_with_node_labels_heterogeneous(self):
        partition_output = PartitionOutput(
            node_partition_book={
                _USER: torch.zeros(5),
                _STORY: torch.zeros(5),
            },
            edge_partition_book={
                _USER_TO_STORY: torch.zeros(5),
                _STORY_TO_USER: torch.zeros(5),
            },
            partitioned_edge_index={
                _USER_TO_STORY: GraphPartitionData(
                    edge_index=torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
                    edge_ids=None,
                ),
                _STORY_TO_USER: GraphPartitionData(
                    edge_index=torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
                    edge_ids=None,
                ),
            },
            partitioned_node_features={
                _USER: FeaturePartitionData(
                    feats=torch.zeros(5, 2), ids=torch.arange(5)
                ),
                _STORY: FeaturePartitionData(
                    feats=torch.zeros(5, 2), ids=torch.arange(5)
                ),
            },
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels={
                _USER: FeaturePartitionData(
                    feats=torch.arange(5).unsqueeze(1), ids=torch.arange(5)
                ),
                _STORY: FeaturePartitionData(
                    feats=torch.arange(5).unsqueeze(1), ids=torch.arange(5)
                ),
            },
        )

        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_distributed_neighbor_loader_with_node_labels_heterogeneous,
            args=(dataset, 1),  # dataset  # batch_size
        )

    def test_cora_supervised_node_classification(self):
        """Test CORA dataset for supervised node classification task."""

        torch.distributed.init_process_group(
            rank=0, world_size=1, init_method=get_process_group_init_method()
        )
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO.name
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

        splitter = HashedNodeSplitter()

        dataset = build_dataset(
            serialized_graph_metadata=serialized_graph_metadata,
            sample_edge_direction="in",
            splitter=splitter,
        )

        mp.spawn(
            fn=_run_cora_supervised_node_classification,
            args=(
                dataset,  # dataset
                32,  # batch_size
            ),
        )

    def test_isolated_heterogeneous_neighbor_loader(
        self,
    ):
        partition_output = PartitionOutput(
            node_partition_book={"author": torch.zeros(18)},
            edge_partition_book=None,
            partitioned_edge_index={
                EdgeType(
                    NodeType("author"), Relation("to"), NodeType("author")
                ): GraphPartitionData(
                    edge_index=torch.tensor([[10], [15]]), edge_ids=None
                )
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
            fn=_run_distributed_heterogeneous_neighbor_loader,
            args=(dataset, self._context, 18),
        )

    def test_isolated_homogeneous_neighbor_loader(
        self,
    ):
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(18),
            edge_partition_book=None,
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[10], [15]]), edge_ids=None
            ),
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_distributed_neighbor_loader,
            args=(dataset, self._context, 18),
        )


if __name__ == "__main__":
    unittest.main()
