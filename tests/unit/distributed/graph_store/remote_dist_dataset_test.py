from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Final, Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from absl.testing import absltest

import gigl.distributed.graph_store.dist_server as dist_server_module
from gigl.common import LocalUri
from gigl.distributed.graph_store.dist_server import DistServer, _call_func_on_server
from gigl.distributed.graph_store.messages import (
    FetchABLPInputRequest,
    FetchNodesRequest,
)
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.graph_store.sharding import ServerSlice, ShardStrategy
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    FeatureInfo,
)
from gigl.utils.sampling import ABLPInputNodes
from tests.test_assets.distributed.test_dataset import (
    DEFAULT_HETEROGENEOUS_EDGE_INDICES,
    DEFAULT_HOMOGENEOUS_EDGE_INDEX,
    STORY,
    STORY_TO_USER,
    USER,
    USER_TO_STORY,
    create_heterogeneous_dataset,
    create_heterogeneous_dataset_for_ablp,
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import (
    MockGraphStoreInfo,
    create_test_process_group,
    get_process_group_init_method,
)
from tests.test_assets.test_case import TestCase

# Module-level test server instance used by mock functions
_test_server: Optional[DistServer] = None

# Shared ABLP label data used by multiple test classes
# The keys represent the source (USER) node IDs and the values represent the destination (STORY) node IDs.
_DEFAULT_POSITIVE_ABLP_LABELS: Final[dict[int, list[int]]] = {
    0: [0, 1],
    1: [1, 2],
    2: [2, 3],
    3: [3, 4],
    4: [4, 0],
}
_DEFAULT_NEGATIVE_ABLP_LABELS: Final[dict[int, list[int]]] = {
    0: [2],
    1: [3],
    2: [4],
    3: [0],
    4: [1],
}
# The list of source (USER) node IDs in the different splits.
_DEFAULT_ABLP_TRAIN_IDS: Final[list[int]] = [0, 1, 2]
_DEFAULT_ABLP_VAL_IDS: Final[list[int]] = [3]
_DEFAULT_ABLP_TEST_IDS: Final[list[int]] = [4]


def _mock_request_server(
    server_rank: int, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Mock request_server that routes through _call_func_on_server."""
    return _call_func_on_server(func, *args, **kwargs)


def _mock_async_request_server(
    server_rank: int, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> torch.futures.Future:
    """Mock async_request_server that routes through _call_func_on_server and returns a future."""
    future: torch.futures.Future = torch.futures.Future()
    future.set_result(_call_func_on_server(func, *args, **kwargs))
    return future


def _create_mock_graph_store_info(
    num_storage_nodes: int = 1,
    num_compute_nodes: int = 1,
    compute_node_rank: int = 0,
    num_processes_per_compute: int = 1,
) -> GraphStoreInfo:
    """Create a mock GraphStoreInfo with placeholder values."""
    real_info = GraphStoreInfo(
        num_storage_nodes=num_storage_nodes,
        num_compute_nodes=num_compute_nodes,
        cluster_master_ip="127.0.0.1",
        storage_cluster_master_ip="127.0.0.1",
        compute_cluster_master_ip="127.0.0.1",
        cluster_master_port=12345,
        storage_cluster_master_port=12346,
        compute_cluster_master_port=12347,
        num_processes_per_compute=num_processes_per_compute,
        rpc_master_port=12348,
        rpc_wait_port=12349,
        readiness_uri=LocalUri("/tmp/mock_readiness.txt"),
    )
    return MockGraphStoreInfo(real_info, compute_node_rank)


@contextmanager
def _patch_remote_requests(
    async_side_effect: Callable[..., torch.futures.Future],
    sync_side_effect: Callable[..., Any],
) -> Iterator[None]:
    """Context manager that patches both async_request_server and request_server."""
    with (
        patch(
            "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
            side_effect=async_side_effect,
        ),
        patch(
            "gigl.distributed.graph_store.remote_dist_dataset.request_server",
            side_effect=sync_side_effect,
        ),
    ):
        yield


def _create_server_with_splits(
    edge_indices: Optional[dict] = None,
    src_node_type: NodeType = USER,
    dst_node_type: NodeType = STORY,
    supervision_edge_type: Optional[EdgeType] = None,
) -> None:
    """Create a DistServer with a dataset that has train/val/test splits.

    Args:
        edge_indices: Edge indices to use. Defaults to DEFAULT_HETEROGENEOUS_EDGE_INDICES.
        src_node_type: Source node type for labeled homogeneous datasets.
        dst_node_type: Destination node type for labeled homogeneous datasets.
        supervision_edge_type: Supervision edge type for labeled homogeneous datasets.
    """
    global _test_server
    create_test_process_group()

    dataset = create_heterogeneous_dataset_for_ablp(
        positive_labels=_DEFAULT_POSITIVE_ABLP_LABELS,
        negative_labels=_DEFAULT_NEGATIVE_ABLP_LABELS,
        train_node_ids=_DEFAULT_ABLP_TRAIN_IDS,
        val_node_ids=_DEFAULT_ABLP_VAL_IDS,
        test_node_ids=_DEFAULT_ABLP_TEST_IDS,
        edge_indices=edge_indices or DEFAULT_HETEROGENEOUS_EDGE_INDICES,
        src_node_type=src_node_type,
        dst_node_type=dst_node_type,
        supervision_edge_type=supervision_edge_type,
    )
    _test_server = DistServer(dataset)
    dist_server_module._dist_server = _test_server


class RemoteDistDatasetTestBase(TestCase):
    """Shared tearDown for all RemoteDistDataset test classes."""

    def tearDown(self) -> None:
        global _test_server
        _test_server = None
        dist_server_module._dist_server = None
        if dist.is_initialized():
            dist.destroy_process_group()


@patch(
    "gigl.distributed.graph_store.remote_dist_dataset.request_server",
    side_effect=_mock_request_server,
)
class TestRemoteDistDataset(RemoteDistDatasetTestBase):
    def setUp(self) -> None:
        global _test_server
        # 10 nodes in DEFAULT_HOMOGENEOUS_EDGE_INDEX ring graph
        node_features = torch.zeros(10, 3)
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
            node_features=node_features,
        )
        _test_server = DistServer(dataset)
        dist_server_module._dist_server = _test_server

    def test_graph_metadata_getters_homogeneous(self, mock_request):
        """Test fetch_node_feature_info, fetch_edge_feature_info, fetch_edge_dir, fetch_edge_types, fetch_node_types for homogeneous graphs."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        self.assertEqual(
            remote_dataset.fetch_node_feature_info(),
            FeatureInfo(dim=3, dtype=torch.float32),
        )
        self.assertIsNone(remote_dataset.fetch_edge_feature_info())
        self.assertEqual(remote_dataset.fetch_edge_dir(), "out")
        self.assertIsNone(remote_dataset.fetch_edge_types())
        self.assertIsNone(remote_dataset.fetch_node_types())

    def test_cluster_info_property(self, mock_request):
        cluster_info = _create_mock_graph_store_info(
            num_storage_nodes=3, num_compute_nodes=2
        )
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        self.assertEqual(remote_dataset.cluster_info.num_storage_nodes, 3)
        self.assertEqual(remote_dataset.cluster_info.num_compute_nodes, 2)

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_fetch_node_ids(self, mock_request, mock_async_request):
        """Test fetch_node_ids returns node ids, with optional sharding via rank/world_size."""
        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Basic: all nodes
        result = remote_dataset.fetch_node_ids()
        self.assertIn(0, result)
        self.assert_tensor_equality(result[0], torch.arange(10))

        # With sharding: first half (rank 0 of 2)
        result = remote_dataset.fetch_node_ids(rank=0, world_size=2)
        self.assert_tensor_equality(result[0], torch.arange(5))

        # With sharding: second half (rank 1 of 2)
        result = remote_dataset.fetch_node_ids(rank=1, world_size=2)
        self.assert_tensor_equality(result[0], torch.arange(5, 10))

    def test_fetch_node_partition_book_homogeneous(self, mock_request):
        """Test fetch_node_partition_book returns the tensor partition book for homogeneous graphs."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        result = remote_dataset.fetch_node_partition_book()
        self.assertIsInstance(result, torch.Tensor)
        assert isinstance(result, torch.Tensor)  # for type narrowing
        self.assertEqual(result.shape[0], 10)
        self.assert_tensor_equality(result, torch.zeros(10, dtype=torch.int64))

    def test_fetch_edge_partition_book_homogeneous(self, mock_request):
        """Test fetch_edge_partition_book returns the tensor partition book for homogeneous graphs."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        result = remote_dataset.fetch_edge_partition_book()
        self.assertIsInstance(result, torch.Tensor)
        assert isinstance(result, torch.Tensor)  # for type narrowing
        self.assertEqual(result.shape[0], 10)
        self.assert_tensor_equality(result, torch.zeros(10, dtype=torch.int64))

    def test_fetch_node_partition_book_homogeneous_rejects_node_type(
        self, mock_request
    ):
        """Test fetch_node_partition_book raises ValueError when node_type is given for homogeneous graphs."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        with self.assertRaises(ValueError):
            remote_dataset.fetch_node_partition_book(node_type=USER)


@patch(
    "gigl.distributed.graph_store.remote_dist_dataset.request_server",
    side_effect=_mock_request_server,
)
class TestRemoteDistDatasetHeterogeneous(RemoteDistDatasetTestBase):
    def setUp(self) -> None:
        global _test_server
        # 5 users, 5 stories in DEFAULT_HETEROGENEOUS_EDGE_INDICES
        node_features = {
            USER: torch.zeros(5, 2),
            STORY: torch.zeros(5, 2),
        }
        dataset = create_heterogeneous_dataset(
            edge_indices=DEFAULT_HETEROGENEOUS_EDGE_INDICES,
            node_features=node_features,
        )
        _test_server = DistServer(dataset)
        dist_server_module._dist_server = _test_server

    def test_graph_metadata_getters_heterogeneous(self, mock_request):
        """Test fetch_node_feature_info, fetch_edge_dir, fetch_edge_types, fetch_node_types for heterogeneous graphs."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        self.assertEqual(
            remote_dataset.fetch_node_feature_info(),
            {
                USER: FeatureInfo(dim=2, dtype=torch.float32),
                STORY: FeatureInfo(dim=2, dtype=torch.float32),
            },
        )
        self.assertEqual(remote_dataset.fetch_edge_dir(), "out")
        self.assertEqual(
            remote_dataset.fetch_edge_types(), [USER_TO_STORY, STORY_TO_USER]
        )
        node_types = remote_dataset.fetch_node_types()
        self.assertIsNotNone(node_types)
        assert node_types is not None  # for type narrowing
        self.assertEqual(set(node_types), {USER, STORY})

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_fetch_node_ids_with_node_type(self, mock_request, mock_async_request):
        """Test fetch_node_ids with node_type for heterogeneous graphs, with optional sharding."""
        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Get user nodes
        result = remote_dataset.fetch_node_ids(node_type=USER)
        self.assert_tensor_equality(result[0], torch.arange(5))

        # Get story nodes
        result = remote_dataset.fetch_node_ids(node_type=STORY)
        self.assert_tensor_equality(result[0], torch.arange(5))

        # With sharding: first half of user nodes (rank 0 of 2)
        result = remote_dataset.fetch_node_ids(rank=0, world_size=2, node_type=USER)
        self.assert_tensor_equality(result[0], torch.arange(2))

        # With sharding: second half of user nodes (rank 1 of 2)
        result = remote_dataset.fetch_node_ids(rank=1, world_size=2, node_type=USER)
        self.assert_tensor_equality(result[0], torch.arange(2, 5))

    def test_fetch_node_partition_book_heterogeneous(self, mock_request):
        """Test fetch_node_partition_book returns per-type partition books for heterogeneous graphs."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        user_pb = remote_dataset.fetch_node_partition_book(node_type=USER)
        self.assertIsInstance(user_pb, torch.Tensor)
        assert isinstance(user_pb, torch.Tensor)
        self.assertEqual(user_pb.shape[0], 5)
        self.assert_tensor_equality(user_pb, torch.zeros(5, dtype=torch.int64))

        story_pb = remote_dataset.fetch_node_partition_book(node_type=STORY)
        self.assertIsInstance(story_pb, torch.Tensor)
        assert isinstance(story_pb, torch.Tensor)
        self.assertEqual(story_pb.shape[0], 5)
        self.assert_tensor_equality(story_pb, torch.zeros(5, dtype=torch.int64))

    def test_fetch_edge_partition_book_heterogeneous(self, mock_request):
        """Test fetch_edge_partition_book returns per-type partition books for heterogeneous graphs."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        user_to_story_pb = remote_dataset.fetch_edge_partition_book(
            edge_type=USER_TO_STORY
        )
        self.assertIsInstance(user_to_story_pb, torch.Tensor)
        assert isinstance(user_to_story_pb, torch.Tensor)
        self.assert_tensor_equality(
            user_to_story_pb,
            torch.zeros(
                DEFAULT_HETEROGENEOUS_EDGE_INDICES[USER_TO_STORY].shape[1],
                dtype=torch.int64,
            ),
        )

    def test_fetch_node_partition_book_heterogeneous_requires_node_type(
        self, mock_request
    ):
        """Test fetch_node_partition_book raises ValueError when no node_type for heterogeneous graphs."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        with self.assertRaises(ValueError):
            remote_dataset.fetch_node_partition_book()

    def test_fetch_edge_partition_book_heterogeneous_requires_edge_type(
        self, mock_request
    ):
        """Test fetch_edge_partition_book raises ValueError when no edge_type for heterogeneous graphs."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        with self.assertRaises(ValueError):
            remote_dataset.fetch_edge_partition_book()


class TestRemoteDistDatasetWithSplits(RemoteDistDatasetTestBase):
    """Tests for fetch_node_ids with train/val/test splits."""

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_fetch_node_ids_with_splits(self, mock_async_request):
        """Test fetch_node_ids with train/val/test splits and optional sharding."""
        _create_server_with_splits()

        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Test each split returns correct nodes
        self.assert_tensor_equality(
            remote_dataset.fetch_node_ids(node_type=USER, split="train")[0],
            torch.tensor([0, 1, 2]),
        )
        self.assert_tensor_equality(
            remote_dataset.fetch_node_ids(node_type=USER, split="val")[0],
            torch.tensor([3]),
        )
        self.assert_tensor_equality(
            remote_dataset.fetch_node_ids(node_type=USER, split="test")[0],
            torch.tensor([4]),
        )

        # No split returns all nodes
        self.assert_tensor_equality(
            remote_dataset.fetch_node_ids(node_type=USER, split=None)[0],
            torch.arange(5),
        )

        # With sharding: train split [0, 1, 2] across 2 ranks
        self.assert_tensor_equality(
            remote_dataset.fetch_node_ids(
                rank=0, world_size=2, node_type=USER, split="train"
            )[0],
            torch.tensor([0]),
        )
        self.assert_tensor_equality(
            remote_dataset.fetch_node_ids(
                rank=1, world_size=2, node_type=USER, split="train"
            )[0],
            torch.tensor([1, 2]),
        )

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_fetch_ablp_input(self, mock_async_request):
        """Test fetch_ablp_input with train/val/test splits."""
        _create_server_with_splits()

        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Train split: nodes [0, 1, 2]
        result = remote_dataset.fetch_ablp_input(
            split="train", anchor_node_type=USER, supervision_edge_type=USER_TO_STORY
        )
        self.assertIn(0, result)
        ablp_input = result[0]
        self.assertIsInstance(ablp_input, ABLPInputNodes)
        self.assertEqual(ablp_input.anchor_node_type, USER)
        self.assert_tensor_equality(ablp_input.anchor_nodes, torch.tensor([0, 1, 2]))
        self.assertIn(USER_TO_STORY, ablp_input.labels)
        pos_labels, neg_labels = ablp_input.labels[USER_TO_STORY]
        self.assert_tensor_equality(
            pos_labels,
            torch.tensor([[0, 1], [1, 2], [2, 3]]),
        )
        assert neg_labels is not None
        self.assert_tensor_equality(
            neg_labels,
            torch.tensor([[2], [3], [4]]),
        )

        # Val split: node [3]
        result = remote_dataset.fetch_ablp_input(
            split="val", anchor_node_type=USER, supervision_edge_type=USER_TO_STORY
        )
        ablp_input = result[0]
        self.assert_tensor_equality(ablp_input.anchor_nodes, torch.tensor([3]))
        pos_labels, neg_labels = ablp_input.labels[USER_TO_STORY]
        self.assert_tensor_equality(
            pos_labels,
            torch.tensor([[3, 4]]),
        )
        assert neg_labels is not None
        self.assert_tensor_equality(
            neg_labels,
            torch.tensor([[0]]),
        )

        # Test split: node [4]
        # Note: Labels are stored in CSR format which sorts by destination indices,
        # so [4, 0] from the input becomes [0, 4] in the stored format.
        result = remote_dataset.fetch_ablp_input(
            split="test", anchor_node_type=USER, supervision_edge_type=USER_TO_STORY
        )
        ablp_input = result[0]
        self.assert_tensor_equality(ablp_input.anchor_nodes, torch.tensor([4]))
        pos_labels, neg_labels = ablp_input.labels[USER_TO_STORY]
        self.assert_tensor_equality(
            pos_labels,
            torch.tensor([[0, 4]]),
        )
        assert neg_labels is not None
        self.assert_tensor_equality(
            neg_labels,
            torch.tensor([[1]]),
        )

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_fetch_ablp_input_with_sharding(self, mock_async_request):
        """Test fetch_ablp_input with sharding across compute nodes."""
        _create_server_with_splits()

        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # With sharding: train split [0, 1, 2] across 2 ranks
        result_rank0 = remote_dataset.fetch_ablp_input(
            split="train",
            rank=0,
            world_size=2,
            anchor_node_type=USER,
            supervision_edge_type=USER_TO_STORY,
        )
        ablp_0 = result_rank0[0]
        self.assertIsInstance(ablp_0, ABLPInputNodes)
        self.assertEqual(ablp_0.anchor_node_type, USER)

        # Rank 0 should get node 0
        self.assert_tensor_equality(ablp_0.anchor_nodes, torch.tensor([0]))
        pos_labels_0, neg_labels_0 = ablp_0.labels[USER_TO_STORY]
        self.assert_tensor_equality(
            pos_labels_0,
            torch.tensor([[0, 1]]),
        )
        assert neg_labels_0 is not None
        self.assert_tensor_equality(
            neg_labels_0,
            torch.tensor([[2]]),
        )

        result_rank1 = remote_dataset.fetch_ablp_input(
            split="train",
            rank=1,
            world_size=2,
            anchor_node_type=USER,
            supervision_edge_type=USER_TO_STORY,
        )
        ablp_1 = result_rank1[0]
        self.assertIsInstance(ablp_1, ABLPInputNodes)

        # Rank 1 should get nodes 1, 2
        self.assert_tensor_equality(ablp_1.anchor_nodes, torch.tensor([1, 2]))
        pos_labels_1, neg_labels_1 = ablp_1.labels[USER_TO_STORY]
        self.assert_tensor_equality(
            pos_labels_1,
            torch.tensor([[1, 2], [2, 3]]),
        )
        assert neg_labels_1 is not None
        self.assert_tensor_equality(
            neg_labels_1,
            torch.tensor([[3], [4]]),
        )


class TestRemoteDistDatasetLabeledHomogeneous(RemoteDistDatasetTestBase):
    """Tests for datasets using DEFAULT_HOMOGENEOUS_NODE_TYPE / DEFAULT_HOMOGENEOUS_EDGE_TYPE.

    A 'labeled homogeneous' dataset is stored internally as heterogeneous
    (keyed by DEFAULT_HOMOGENEOUS_NODE_TYPE) but treated as homogeneous for
    sampling.  RemoteDistDataset should auto-infer the node/edge types so
    callers do not need to supply them explicitly.
    """

    _LABELED_HOMOGENEOUS_EDGE_INDICES: Final = {
        DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    }

    def _create_labeled_homogeneous_server(self) -> None:
        _create_server_with_splits(
            edge_indices=self._LABELED_HOMOGENEOUS_EDGE_INDICES,
            src_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
            dst_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
            supervision_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
        )

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_fetch_node_types_labeled_homogeneous(self, mock_request):
        """Test fetch_node_types returns DEFAULT_HOMOGENEOUS_NODE_TYPE for labeled homogeneous datasets."""
        self._create_labeled_homogeneous_server()
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        node_types = remote_dataset.fetch_node_types()
        self.assertIsNotNone(node_types)
        self.assertIn(DEFAULT_HOMOGENEOUS_NODE_TYPE, node_types)

    def test_fetch_node_ids_auto_detects_default_node_type(self):
        """Test fetch_node_ids without node_type auto-detects DEFAULT_HOMOGENEOUS_NODE_TYPE."""
        self._create_labeled_homogeneous_server()
        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)

        with _patch_remote_requests(_mock_async_request_server, _mock_request_server):
            remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

            # No node_type provided: _fetch_node_ids should auto-detect DEFAULT_HOMOGENEOUS_NODE_TYPE
            self.assert_tensor_equality(
                remote_dataset.fetch_node_ids(split="train")[0],
                torch.tensor([0, 1, 2]),
            )
            self.assert_tensor_equality(
                remote_dataset.fetch_node_ids(split="val")[0],
                torch.tensor([3]),
            )
            self.assert_tensor_equality(
                remote_dataset.fetch_node_ids(split="test")[0],
                torch.tensor([4]),
            )

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.async_request_server",
        side_effect=_mock_async_request_server,
    )
    def test_fetch_ablp_input_defaults_to_homogeneous_types(self, mock_async_request):
        """Test fetch_ablp_input without anchor_node_type/supervision_edge_type uses homogeneous defaults."""
        self._create_labeled_homogeneous_server()
        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # Train split: nodes [0, 1, 2] — no type params provided
        result = remote_dataset.fetch_ablp_input(split="train")
        self.assertIn(0, result)
        ablp_input = result[0]
        self.assertIsInstance(ablp_input, ABLPInputNodes)
        self.assertEqual(ablp_input.anchor_node_type, DEFAULT_HOMOGENEOUS_NODE_TYPE)
        self.assert_tensor_equality(ablp_input.anchor_nodes, torch.tensor([0, 1, 2]))
        self.assertIn(DEFAULT_HOMOGENEOUS_EDGE_TYPE, ablp_input.labels)
        pos_labels, neg_labels = ablp_input.labels[DEFAULT_HOMOGENEOUS_EDGE_TYPE]
        self.assert_tensor_equality(
            pos_labels,
            torch.tensor([[0, 1], [1, 2], [2, 3]]),
        )
        assert neg_labels is not None
        self.assert_tensor_equality(
            neg_labels,
            torch.tensor([[2], [3], [4]]),
        )

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_fetch_node_partition_book_auto_infers_default_node_type(
        self, mock_request
    ):
        """Test fetch_node_partition_book auto-infers DEFAULT_HOMOGENEOUS_NODE_TYPE when None."""
        self._create_labeled_homogeneous_server()
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # No node_type: should auto-infer DEFAULT_HOMOGENEOUS_NODE_TYPE
        result = remote_dataset.fetch_node_partition_book()
        self.assertIsInstance(result, torch.Tensor)
        assert isinstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 5)
        self.assert_tensor_equality(result, torch.zeros(5, dtype=torch.int64))

    @patch(
        "gigl.distributed.graph_store.remote_dist_dataset.request_server",
        side_effect=_mock_request_server,
    )
    def test_fetch_edge_partition_book_auto_infers_default_edge_type(
        self, mock_request
    ):
        """Test fetch_edge_partition_book auto-infers DEFAULT_HOMOGENEOUS_EDGE_TYPE when None."""
        self._create_labeled_homogeneous_server()
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        # No edge_type: should auto-infer DEFAULT_HOMOGENEOUS_EDGE_TYPE
        result = remote_dataset.fetch_edge_partition_book()
        self.assertIsInstance(result, torch.Tensor)
        assert isinstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 5)
        self.assert_tensor_equality(result, torch.zeros(5, dtype=torch.int64))

    def test_fetch_ablp_input_mismatched_params_raises(self):
        """Test fetch_ablp_input raises ValueError when exactly one type param is None."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        with self.assertRaises(ValueError):
            remote_dataset.fetch_ablp_input(
                split="train",
                anchor_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
                supervision_edge_type=None,
            )

        with self.assertRaises(ValueError):
            remote_dataset.fetch_ablp_input(
                split="train",
                anchor_node_type=None,
                supervision_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
            )


class TestRemoteDistDatasetContiguous(RemoteDistDatasetTestBase):
    """Tests for fetch_node_ids and fetch_ablp_input with ShardStrategy.CONTIGUOUS."""

    def _make_rank_aware_async_mock(
        self,
        server_data: dict[int, Any],
        captured_requests: Optional[list[tuple[int, Any]]] = None,
    ) -> Callable[..., torch.futures.Future]:
        """Create an async mock that returns pre-set data per server rank.

        Args:
            server_data: Maps server_rank to the value that server should return.
                Can be a tensor (for node ID tests) or a tuple (for ABLP tests).
            captured_requests: Optional list populated with ``(server_rank, request)``
                tuples for later assertions.
        """

        def _mock(
            server_rank: int, func: Callable[..., Any], *args: Any, **kwargs: Any
        ) -> torch.futures.Future:
            assert not kwargs
            assert len(args) == 1
            request = args[0]
            if captured_requests is not None:
                captured_requests.append((server_rank, request))

            future: torch.futures.Future = torch.futures.Future()
            response = server_data[server_rank]
            if isinstance(request, FetchNodesRequest):
                if request.server_slice is not None:
                    assert isinstance(response, torch.Tensor)
                    response = request.server_slice.slice_tensor(response)
            elif isinstance(request, FetchABLPInputRequest):
                if request.server_slice is not None:
                    anchors, positive_labels, negative_labels = response
                    response = (
                        request.server_slice.slice_tensor(anchors),
                        request.server_slice.slice_tensor(positive_labels),
                        (
                            request.server_slice.slice_tensor(negative_labels)
                            if negative_labels is not None
                            else None
                        ),
                    )
            future.set_result(response)
            return future

        return _mock

    @staticmethod
    def _mock_request_server_homogeneous(
        server_rank: int, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Mock request_server that returns None for node/edge types (homogeneous)."""
        if func == DistServer.get_node_types:
            return None
        if func == DistServer.get_edge_types:
            return None
        return _mock_request_server(server_rank, func, *args, **kwargs)

    def test_fetch_node_ids_contiguous_even_split(self) -> None:
        """CONTIGUOUS with 2 storage nodes and 2 compute nodes: each rank gets one server."""
        server_data: dict[int, torch.Tensor] = {
            0: torch.arange(10),
            1: torch.arange(10, 20),
        }
        captured_requests: list[tuple[int, Any]] = []
        mock_fn = self._make_rank_aware_async_mock(server_data, captured_requests)
        cluster_info = _create_mock_graph_store_info(
            num_storage_nodes=2, num_compute_nodes=2
        )

        with _patch_remote_requests(mock_fn, self._mock_request_server_homogeneous):
            ds = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

            # Rank 0: gets all of server 0, empty from server 1
            result = ds.fetch_node_ids(
                rank=0, world_size=2, shard_strategy=ShardStrategy.CONTIGUOUS
            )
            self.assert_tensor_equality(result[0], torch.arange(10))
            self.assertEqual(result[1].numel(), 0)
            self.assertEqual(
                captured_requests,
                [
                    (
                        0,
                        FetchNodesRequest(
                            split=None,
                            node_type=None,
                            server_slice=ServerSlice(
                                server_rank=0,
                                start_numerator=0,
                                end_numerator=2,
                                denominator=2,
                            ),
                        ),
                    )
                ],
            )

            # Rank 1: empty from server 0, gets all of server 1
            captured_requests.clear()
            result = ds.fetch_node_ids(
                rank=1, world_size=2, shard_strategy=ShardStrategy.CONTIGUOUS
            )
            self.assertEqual(result[0].numel(), 0)
            self.assert_tensor_equality(result[1], torch.arange(10, 20))
            self.assertEqual(
                captured_requests,
                [
                    (
                        1,
                        FetchNodesRequest(
                            split=None,
                            node_type=None,
                            server_slice=ServerSlice(
                                server_rank=1,
                                start_numerator=0,
                                end_numerator=2,
                                denominator=2,
                            ),
                        ),
                    )
                ],
            )

    def test_fetch_node_ids_contiguous_fractional_split(self) -> None:
        """CONTIGUOUS with 3 storage nodes and 2 compute nodes: server 1 is fractionally split."""
        server_data: dict[int, torch.Tensor] = {
            0: torch.arange(10),
            1: torch.arange(10, 20),
            2: torch.arange(20, 30),
        }
        captured_requests: list[tuple[int, Any]] = []
        mock_fn = self._make_rank_aware_async_mock(server_data, captured_requests)
        cluster_info = _create_mock_graph_store_info(
            num_storage_nodes=3, num_compute_nodes=2
        )

        with _patch_remote_requests(mock_fn, self._mock_request_server_homogeneous):
            ds = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

            # Rank 0: all of server 0, first half of server 1, nothing from server 2
            result = ds.fetch_node_ids(
                rank=0, world_size=2, shard_strategy=ShardStrategy.CONTIGUOUS
            )
            self.assert_tensor_equality(result[0], torch.arange(10))
            self.assert_tensor_equality(result[1], torch.arange(10, 15))
            self.assertEqual(result[2].numel(), 0)
            self.assertEqual(
                captured_requests,
                [
                    (
                        0,
                        FetchNodesRequest(
                            split=None,
                            node_type=None,
                            server_slice=ServerSlice(
                                server_rank=0,
                                start_numerator=0,
                                end_numerator=2,
                                denominator=2,
                            ),
                        ),
                    ),
                    (
                        1,
                        FetchNodesRequest(
                            split=None,
                            node_type=None,
                            server_slice=ServerSlice(
                                server_rank=1,
                                start_numerator=0,
                                end_numerator=1,
                                denominator=2,
                            ),
                        ),
                    ),
                ],
            )

            # Rank 1: nothing from server 0, second half of server 1, all of server 2
            captured_requests.clear()
            result = ds.fetch_node_ids(
                rank=1, world_size=2, shard_strategy=ShardStrategy.CONTIGUOUS
            )
            self.assertEqual(result[0].numel(), 0)
            self.assert_tensor_equality(result[1], torch.arange(15, 20))
            self.assert_tensor_equality(result[2], torch.arange(20, 30))
            self.assertEqual(
                captured_requests,
                [
                    (
                        1,
                        FetchNodesRequest(
                            split=None,
                            node_type=None,
                            server_slice=ServerSlice(
                                server_rank=1,
                                start_numerator=1,
                                end_numerator=2,
                                denominator=2,
                            ),
                        ),
                    ),
                    (
                        2,
                        FetchNodesRequest(
                            split=None,
                            node_type=None,
                            server_slice=ServerSlice(
                                server_rank=2,
                                start_numerator=0,
                                end_numerator=2,
                                denominator=2,
                            ),
                        ),
                    ),
                ],
            )

    def test_with_split_filtering(self) -> None:
        """CONTIGUOUS strategy with split='train' filtering."""
        server_data: dict[int, torch.Tensor] = {
            0: torch.tensor([0, 1, 2, 3]),
            1: torch.tensor([10, 11, 12, 13]),
        }
        mock_fn = self._make_rank_aware_async_mock(server_data)
        cluster_info = _create_mock_graph_store_info(
            num_storage_nodes=2, num_compute_nodes=2
        )

        with _patch_remote_requests(mock_fn, self._mock_request_server_homogeneous):
            ds = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

            result = ds.fetch_node_ids(
                rank=0,
                world_size=2,
                split="train",
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )
            self.assert_tensor_equality(result[0], torch.tensor([0, 1, 2, 3]))
            self.assertEqual(result[1].numel(), 0)

    def test_contiguous_requires_rank_and_world_size(self) -> None:
        """CONTIGUOUS without rank/world_size raises ValueError."""
        cluster_info = _create_mock_graph_store_info(
            num_storage_nodes=2, num_compute_nodes=2
        )
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        with self.assertRaises(ValueError):
            remote_dataset.fetch_node_ids(
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )
        with self.assertRaises(ValueError):
            remote_dataset.fetch_node_ids(
                rank=0,
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )
        with self.assertRaises(ValueError):
            remote_dataset.fetch_node_ids(
                world_size=2,
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )

    def test_contiguous_labeled_homogeneous_auto_inference(self) -> None:
        """CONTIGUOUS strategy auto-infers DEFAULT_HOMOGENEOUS_NODE_TYPE for labeled homogeneous datasets."""
        _create_server_with_splits(
            edge_indices={
                DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor(
                    [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
                )
            },
            src_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
            dst_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
            supervision_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
        )

        cluster_info = _create_mock_graph_store_info(num_storage_nodes=1)

        with _patch_remote_requests(_mock_async_request_server, _mock_request_server):
            remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

            result = remote_dataset.fetch_node_ids(
                rank=0,
                world_size=1,
                split="train",
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )
            self.assert_tensor_equality(
                result[0],
                torch.tensor([0, 1, 2]),
            )

    def test_fetch_ablp_input_contiguous_even_split(self) -> None:
        """ABLP CONTIGUOUS with 2 storage nodes and 2 compute nodes."""
        server_data: dict[
            int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        ] = {
            0: (
                torch.tensor([0, 1, 2]),
                torch.tensor([[0, 1], [1, 2], [2, 3]]),
                torch.tensor([[4], [5], [6]]),
            ),
            1: (
                torch.tensor([10, 11, 12]),
                torch.tensor([[10, 11], [11, 12], [12, 13]]),
                torch.tensor([[14], [15], [16]]),
            ),
        }
        captured_requests: list[tuple[int, Any]] = []
        mock_fn = self._make_rank_aware_async_mock(server_data, captured_requests)
        cluster_info = _create_mock_graph_store_info(
            num_storage_nodes=2, num_compute_nodes=2
        )

        with _patch_remote_requests(mock_fn, self._mock_request_server_homogeneous):
            ds = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

            # Rank 0: gets all of server 0, empty from server 1
            result = ds.fetch_ablp_input(
                split="train",
                rank=0,
                world_size=2,
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )
            ablp_0 = result[0]
            self.assert_tensor_equality(ablp_0.anchor_nodes, torch.tensor([0, 1, 2]))
            pos, neg = ablp_0.labels[DEFAULT_HOMOGENEOUS_EDGE_TYPE]
            self.assert_tensor_equality(pos, torch.tensor([[0, 1], [1, 2], [2, 3]]))
            assert neg is not None
            self.assert_tensor_equality(neg, torch.tensor([[4], [5], [6]]))

            ablp_1 = result[1]
            self.assertEqual(ablp_1.anchor_nodes.numel(), 0)
            pos_1, neg_1 = ablp_1.labels[DEFAULT_HOMOGENEOUS_EDGE_TYPE]
            self.assertEqual(pos_1.numel(), 0)
            self.assertIsNone(neg_1)
            self.assertEqual(
                captured_requests,
                [
                    (
                        0,
                        FetchABLPInputRequest(
                            split="train",
                            node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
                            supervision_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
                            server_slice=ServerSlice(
                                server_rank=0,
                                start_numerator=0,
                                end_numerator=2,
                                denominator=2,
                            ),
                        ),
                    )
                ],
            )

            # Rank 1: empty from server 0, gets all of server 1
            captured_requests.clear()
            result = ds.fetch_ablp_input(
                split="train",
                rank=1,
                world_size=2,
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )
            ablp_0 = result[0]
            self.assertEqual(ablp_0.anchor_nodes.numel(), 0)

            ablp_1 = result[1]
            self.assert_tensor_equality(ablp_1.anchor_nodes, torch.tensor([10, 11, 12]))
            pos, neg = ablp_1.labels[DEFAULT_HOMOGENEOUS_EDGE_TYPE]
            self.assert_tensor_equality(
                pos, torch.tensor([[10, 11], [11, 12], [12, 13]])
            )
            assert neg is not None
            self.assert_tensor_equality(neg, torch.tensor([[14], [15], [16]]))
            self.assertEqual(
                captured_requests,
                [
                    (
                        1,
                        FetchABLPInputRequest(
                            split="train",
                            node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
                            supervision_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
                            server_slice=ServerSlice(
                                server_rank=1,
                                start_numerator=0,
                                end_numerator=2,
                                denominator=2,
                            ),
                        ),
                    )
                ],
            )

    def test_fetch_ablp_input_contiguous_fractional_split(self) -> None:
        """ABLP CONTIGUOUS with 3 storage nodes and 2 compute nodes: server 1 fractionally split."""
        server_data: dict[
            int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        ] = {
            0: (
                torch.tensor([0, 1, 2, 3]),
                torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]]),
                torch.tensor([[10], [11], [12], [13]]),
            ),
            1: (
                torch.tensor([10, 11, 12, 13]),
                torch.tensor([[10, 11], [11, 12], [12, 13], [13, 14]]),
                torch.tensor([[20], [21], [22], [23]]),
            ),
            2: (
                torch.tensor([20, 21, 22, 23]),
                torch.tensor([[20, 21], [21, 22], [22, 23], [23, 24]]),
                torch.tensor([[30], [31], [32], [33]]),
            ),
        }
        captured_requests: list[tuple[int, Any]] = []
        mock_fn = self._make_rank_aware_async_mock(server_data, captured_requests)
        cluster_info = _create_mock_graph_store_info(
            num_storage_nodes=3, num_compute_nodes=2
        )

        with _patch_remote_requests(mock_fn, self._mock_request_server_homogeneous):
            ds = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

            # Rank 0: all of server 0, first half of server 1, nothing from server 2
            result = ds.fetch_ablp_input(
                split="train",
                rank=0,
                world_size=2,
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )
            ablp_0 = result[0]
            self.assert_tensor_equality(ablp_0.anchor_nodes, torch.tensor([0, 1, 2, 3]))
            pos, neg = ablp_0.labels[DEFAULT_HOMOGENEOUS_EDGE_TYPE]
            self.assert_tensor_equality(
                pos, torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
            )
            assert neg is not None
            self.assert_tensor_equality(neg, torch.tensor([[10], [11], [12], [13]]))

            ablp_1 = result[1]
            self.assert_tensor_equality(ablp_1.anchor_nodes, torch.tensor([10, 11]))
            pos_1, neg_1 = ablp_1.labels[DEFAULT_HOMOGENEOUS_EDGE_TYPE]
            self.assert_tensor_equality(pos_1, torch.tensor([[10, 11], [11, 12]]))
            assert neg_1 is not None
            self.assert_tensor_equality(neg_1, torch.tensor([[20], [21]]))

            ablp_2 = result[2]
            self.assertEqual(ablp_2.anchor_nodes.numel(), 0)
            self.assertEqual(
                captured_requests,
                [
                    (
                        0,
                        FetchABLPInputRequest(
                            split="train",
                            node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
                            supervision_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
                            server_slice=ServerSlice(
                                server_rank=0,
                                start_numerator=0,
                                end_numerator=2,
                                denominator=2,
                            ),
                        ),
                    ),
                    (
                        1,
                        FetchABLPInputRequest(
                            split="train",
                            node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
                            supervision_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
                            server_slice=ServerSlice(
                                server_rank=1,
                                start_numerator=0,
                                end_numerator=1,
                                denominator=2,
                            ),
                        ),
                    ),
                ],
            )

            # Rank 1: nothing from server 0, second half of server 1, all of server 2
            captured_requests.clear()
            result = ds.fetch_ablp_input(
                split="train",
                rank=1,
                world_size=2,
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )
            self.assertEqual(result[0].anchor_nodes.numel(), 0)

            ablp_1 = result[1]
            self.assert_tensor_equality(ablp_1.anchor_nodes, torch.tensor([12, 13]))
            pos_1, neg_1 = ablp_1.labels[DEFAULT_HOMOGENEOUS_EDGE_TYPE]
            self.assert_tensor_equality(pos_1, torch.tensor([[12, 13], [13, 14]]))
            assert neg_1 is not None
            self.assert_tensor_equality(neg_1, torch.tensor([[22], [23]]))

            ablp_2 = result[2]
            self.assert_tensor_equality(
                ablp_2.anchor_nodes, torch.tensor([20, 21, 22, 23])
            )
            pos_2, neg_2 = ablp_2.labels[DEFAULT_HOMOGENEOUS_EDGE_TYPE]
            self.assert_tensor_equality(
                pos_2,
                torch.tensor([[20, 21], [21, 22], [22, 23], [23, 24]]),
            )
            assert neg_2 is not None
            self.assert_tensor_equality(neg_2, torch.tensor([[30], [31], [32], [33]]))
            self.assertEqual(
                captured_requests,
                [
                    (
                        1,
                        FetchABLPInputRequest(
                            split="train",
                            node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
                            supervision_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
                            server_slice=ServerSlice(
                                server_rank=1,
                                start_numerator=1,
                                end_numerator=2,
                                denominator=2,
                            ),
                        ),
                    ),
                    (
                        2,
                        FetchABLPInputRequest(
                            split="train",
                            node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
                            supervision_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
                            server_slice=ServerSlice(
                                server_rank=2,
                                start_numerator=0,
                                end_numerator=2,
                                denominator=2,
                            ),
                        ),
                    ),
                ],
            )

    def test_ablp_contiguous_requires_rank_and_world_size(self) -> None:
        """ABLP CONTIGUOUS without rank/world_size raises ValueError."""
        cluster_info = _create_mock_graph_store_info(
            num_storage_nodes=2, num_compute_nodes=2
        )
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        with self.assertRaises(ValueError):
            remote_dataset.fetch_ablp_input(
                split="train",
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )
        with self.assertRaises(ValueError):
            remote_dataset.fetch_ablp_input(
                split="train",
                rank=0,
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )
        with self.assertRaises(ValueError):
            remote_dataset.fetch_ablp_input(
                split="train",
                world_size=2,
                shard_strategy=ShardStrategy.CONTIGUOUS,
            )


def _test_fetch_free_ports_on_storage_cluster(
    rank: int,
    world_size: int,
    init_process_group_init_method: str,
    num_ports: int,
    mock_ports: list[int],
):
    """Test function to run in spawned processes."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_process_group_init_method,
        world_size=world_size,
        rank=rank,
    )
    try:
        cluster_info = _create_mock_graph_store_info(
            num_compute_nodes=world_size,
            num_processes_per_compute=1,
            compute_node_rank=rank,
        )
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        with patch(
            "gigl.distributed.graph_store.remote_dist_dataset.request_server",
            return_value=mock_ports,
        ):
            ports = remote_dataset.fetch_free_ports_on_storage_cluster(num_ports)

        assert len(ports) == num_ports, f"Expected {num_ports} ports, got {len(ports)}"

        # Verify all ranks get the same ports via all_gather
        gathered_ports = [None] * world_size
        dist.all_gather_object(gathered_ports, ports)

        for i, rank_ports in enumerate(gathered_ports):
            assert (
                rank_ports == mock_ports
            ), f"Rank {i} got {rank_ports}, expected {mock_ports}"
    finally:
        dist.destroy_process_group()


class TestGetFreePortsOnStorageCluster(RemoteDistDatasetTestBase):
    def setUp(self) -> None:
        global _test_server
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)
        _test_server = DistServer(dataset)
        dist_server_module._dist_server = _test_server

    def test_fetch_free_ports_on_storage_cluster_distributed(self):
        """Test that free ports are correctly broadcast across all ranks."""
        init_method = get_process_group_init_method()
        world_size = 2
        num_ports = 3
        mock_ports = [10000, 10001, 10002]

        mp.spawn(
            fn=_test_fetch_free_ports_on_storage_cluster,
            args=(world_size, init_method, num_ports, mock_ports),
            nprocs=world_size,
        )

    def test_fetch_free_ports_fails_without_process_group(self):
        """Test that fetch_free_ports_on_storage_cluster raises when dist not initialized."""
        cluster_info = _create_mock_graph_store_info()
        remote_dataset = RemoteDistDataset(cluster_info=cluster_info, local_rank=0)

        with self.assertRaises(ValueError):
            remote_dataset.fetch_free_ports_on_storage_cluster(num_ports=1)


class TestCallFuncOnServer(RemoteDistDatasetTestBase):
    """Tests for the _call_func_on_server dispatch logic."""

    def setUp(self) -> None:
        global _test_server
        node_features = torch.zeros(10, 3)
        dataset = create_homogeneous_dataset(
            edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX,
            node_features=node_features,
        )
        _test_server = DistServer(dataset)
        dist_server_module._dist_server = _test_server

    def test_dispatches_server_method(self):
        """Test that _call_func_on_server correctly dispatches an unbound DistServer method."""
        result = _call_func_on_server(DistServer.get_edge_dir)
        self.assertEqual(result, "out")

    def test_non_callable_returns_none(self):
        """Test that _call_func_on_server returns None for non-callable input."""
        result: None = _call_func_on_server("not_a_function")  # type: ignore[arg-type]
        self.assertIsNone(result)

    def test_falls_back_for_non_server_function(self):
        """Test that _call_func_on_server falls back to calling func directly for functions not on the server."""

        def standalone_function(x: int, y: int) -> int:
            return x + y

        result = _call_func_on_server(standalone_function, 3, 7)
        self.assertEqual(result, 10)

    def test_raises_when_server_not_initialized(self):
        """Test that _call_func_on_server raises when _dist_server is None."""
        dist_server_module._dist_server = None
        with self.assertRaises(RuntimeError):
            _call_func_on_server(DistServer.get_edge_dir)


if __name__ == "__main__":
    absltest.main()
