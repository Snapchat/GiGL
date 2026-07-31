import os
import socket
import time
import unittest
from collections.abc import Mapping, MutableMapping
from types import SimpleNamespace
from typing import cast
from unittest import mock

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import init_rpc, init_worker_group, shutdown_rpc
from parameterized import param, parameterized
from torch_geometric.data import Data, HeteroData

from gigl.distributed.base_dist_loader import BaseDistLoader
from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_sampling_producer import (
    DistSamplingProducer,
    SamplingPortLease,
    SamplingWorkerRpcSpec,
    SamplingWorkerStatus,
)
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.utils import get_free_port
from gigl.distributed.utils.neighborloader import DatasetSchema
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.env.distributed import DistributedContext
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
    FeatureInfo,
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    to_homogeneous,
)
from gigl.utils.data_splitters import DistNodeAnchorLinkSplitter, DistNodeSplitter
from gigl.utils.iterator import InfiniteIterator
from tests.test_assets.distributed.run_distributed_dataset import (
    run_distributed_dataset,
)
from tests.test_assets.distributed.utils import (
    MockRemoteDistDataset,
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

# GLT requires subclasses of DistNeighborLoader to be run in a separate process. Otherwise, we may run into segmentation fault
# or other memory issues. Calling these functions in separate proceses also allows us to use shutdown_rpc() to ensure cleanup of
# ports, providing stronger guarantees of isolation between tests.


# We require each of these functions to accept local_rank as the first argument since we use mp.spawn with `nprocs=1`
def _run_distributed_neighbor_loader(
    _,
    dataset: DistDataset,
    expected_data_count: int,
):
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
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


def _run_isolated_distributed_neighbor_loader(
    _,
    dataset: DistDataset,
    expected_data_count: int,
):
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        num_workers=2,
        worker_concurrency=1,
        channel_size="64MB",
        process_start_gap_seconds=0,
        pin_memory_device=torch.device("cpu"),
        one_rpc_group_per_sampling_worker=True,
    )

    seed_ids: list[int] = []
    for datum in loader:
        assert isinstance(datum, Data)
        seed_ids.extend(datum.node[: datum.batch_size].tolist())
    assert sorted(seed_ids) == list(range(expected_data_count))
    workers = list(loader._mp_producer._workers)
    specs = loader._mp_producer._isolated_rpc_specs
    assert specs is not None
    assert len({worker.pid for worker in workers}) == len(specs)
    loader.shutdown()
    assert all(not worker.is_alive() for worker in workers)
    port_lease = loader._mp_producer._isolated_port_lease
    assert port_lease is not None and port_lease._closed
    for port in {spec.master_port for spec in specs}:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            assert probe.connect_ex(("127.0.0.1", port)) != 0
        finally:
            probe.close()
    shutdown_rpc()


def _run_two_parent_isolated_distributed_neighbor_loader(
    rank: int,
    dataset_port: int,
    loader_port: int,
    result_queue,
) -> None:
    dataset = run_distributed_dataset(
        rank=rank,
        world_size=2,
        mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
        _port=dataset_port,
    )
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{loader_port}",
        rank=rank,
        world_size=2,
    )
    context = DistributedContext(
        main_worker_ip_address="127.0.0.1",
        global_rank=rank,
        global_world_size=2,
    )
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        num_workers=2,
        batch_size=16,
        context=context,
        local_process_rank=0,
        local_process_world_size=1,
        worker_concurrency=1,
        channel_size="64MB",
        process_start_gap_seconds=0,
        pin_memory_device=torch.device("cpu"),
        one_rpc_group_per_sampling_worker=True,
    )
    seed_ids: list[int] = []
    for datum in loader:
        assert isinstance(datum, Data)
        seed_ids.extend(datum.node[: datum.batch_size].tolist())

    producer = loader._mp_producer
    specs = producer._isolated_rpc_specs
    assert specs is not None
    workers = list(producer._workers)
    result = {
        "rank": rank,
        "seed_ids": seed_ids,
        "ready_workers": sorted(producer._isolated_ready_workers),
        "mappings": [
            {
                "pid": worker.pid,
                "worker_index": spec.worker_index,
                "group_name": spec.group_name,
                "rank": spec.rank,
                "world_size": spec.world_size,
                "port": spec.master_port,
            }
            for worker, spec in zip(workers, specs)
        ],
    }
    loader.shutdown()
    assert all(not worker.is_alive() for worker in workers)
    port_lease = producer._isolated_port_lease
    assert port_lease is not None and port_lease._closed
    result["lease_closed"] = True
    result_queue.put(result)

    if rank == 0:
        for port in {spec.master_port for spec in specs}:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                assert probe.connect_ex(("127.0.0.1", port)) != 0
            finally:
                probe.close()
    torch.distributed.barrier()
    shutdown_rpc(graceful=False)
    torch.distributed.destroy_process_group()


def _cross_parent_missing_rpc_member(*args) -> None:
    worker_options = args[5]
    rpc_spec = args[-2]
    assert isinstance(rpc_spec, SamplingWorkerRpcSpec)
    init_worker_group(
        world_size=rpc_spec.world_size,
        rank=rpc_spec.rank,
        group_name=rpc_spec.group_name,
    )
    try:
        init_rpc(
            master_addr=worker_options.master_addr,
            master_port=rpc_spec.master_port,
            num_rpc_threads=1,
            rpc_timeout=rpc_spec.world_size,
        )
    finally:
        try:
            shutdown_rpc(graceful=False)
        except RuntimeError:
            pass


def _cross_parent_injected_rpc_error(*args) -> None:
    rpc_spec = args[-2]
    status_connection = args[-1]
    assert isinstance(rpc_spec, SamplingWorkerRpcSpec)
    status_connection.send(
        SamplingWorkerStatus(
            state="ERROR",
            worker_index=rpc_spec.worker_index,
            group_name=rpc_spec.group_name,
            rank=rpc_spec.rank,
            master_port=rpc_spec.master_port,
            pid=os.getpid(),
            phase="injected_remote_failure",
            elapsed_seconds=0.0,
            error="injected remote member failure",
        )
    )
    status_connection.close()


def _run_cross_parent_isolated_failure(
    rank: int,
    master_port: int,
    result_queue,
) -> None:
    producer = object.__new__(DistSamplingProducer)
    producer.sampling_config = SimpleNamespace(seed=None, shuffle=False)
    producer.worker_options = SimpleNamespace(
        worker_concurrency=1,
        rpc_timeout=2.0,
        master_addr="127.0.0.1",
    )
    producer.num_workers = 1
    producer.data = object()
    producer.sampler_input = object()
    producer.output_channel = object()
    producer.sampling_completed_worker_count = object()
    producer._sampler_options = object()
    producer._degree_tensors = None
    producer._sampling_run_seed = None
    producer._parent_global_rank = None
    producer._parent_world_size = None
    producer._isolated_rpc_specs = (
        SamplingWorkerRpcSpec(
            worker_index=0,
            group_name="cross_parent_failure_group",
            world_size=2,
            rank=rank,
            master_port=master_port,
        ),
    )
    producer._isolated_port_lease = None
    producer._task_queues = []
    producer._workers = []
    producer._isolated_status_connections = []
    producer._isolated_ready_workers = set()
    producer._isolated_barrier = None
    producer._isolated_resources_closed = False
    producer._isolated_cleanup_complete = False
    producer._shutdown = False
    producer._get_seeds_indexes = lambda: [torch.tensor([0])]

    target = (
        _cross_parent_missing_rpc_member
        if rank == 0
        else _cross_parent_injected_rpc_error
    )
    fork_context = mp.get_context("fork")
    start = time.monotonic()
    error = ""
    with (
        mock.patch(
            "gigl.distributed.dist_sampling_producer._sampling_worker_loop",
            target,
        ),
        mock.patch(
            "gigl.distributed.dist_sampling_producer.mp.get_context",
            return_value=fork_context,
        ),
    ):
        try:
            producer.init()
        except (RuntimeError, TimeoutError) as caught:
            error = str(caught)
    result_queue.put(
        {
            "rank": rank,
            "error": error,
            "elapsed": time.monotonic() - start,
            "worker_count": len(producer._workers),
            "all_dead": all(not worker.is_alive() for worker in producer._workers),
            "resources_closed": producer._isolated_resources_closed,
        }
    )


def _run_distributed_neighbor_loader_labeled_homogeneous(
    _,
    dataset: DistDataset,
    expected_data_count: int,
):
    create_test_process_group()
    assert isinstance(dataset.node_ids, Mapping)
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=to_homogeneous(dataset.node_ids),
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
    )

    count = 0
    for datum in loader:
        assert isinstance(datum, Data)
        count += 1

    assert count == expected_data_count, (
        f"Expected {expected_data_count} batches, but got {count}."
    )

    shutdown_rpc()


def _run_infinite_distributed_neighbor_loader(
    _,
    dataset: DistDataset,
    max_num_batches: int,
):
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
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
    expected_data_count: int,
):
    create_test_process_group()
    assert isinstance(dataset.node_ids, Mapping)
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(NodeType("author"), dataset.node_ids[NodeType("author")]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        num_neighbors=[2, 2],
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
    expected_data_count: int,
):
    create_test_process_group()
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
    create_test_process_group()

    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )

    for datum in loader:
        assert isinstance(datum, Data), (
            f"Subgraph should be a Data for homogeneous datasets, got {type(datum)}"
        )
        assert hasattr(datum, "y"), "Subgraph is missing the `y` attribute for labels"
        # For this mocked data, the value of each label is equal to its Node ID
        assert_tensor_equality(datum.y, datum.node)

    shutdown_rpc()


def _run_distributed_neighbor_loader_with_node_labels_heterogeneous(
    _,
    dataset: DistDataset,
    batch_size: int,
):
    create_test_process_group()

    assert isinstance(dataset.node_ids, Mapping)

    user_loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(_USER, dataset.node_ids[_USER]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )

    story_loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(_STORY, dataset.node_ids[_STORY]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )

    for user_datum, story_datum in zip(user_loader, story_loader):
        # For this mocked data, the value of each user/story label is equal to its corresponding Node ID
        assert isinstance(user_datum, HeteroData), (
            f"User subgraph should be a HeteroData for heterogeneous datasets, got {type(user_datum)}"
        )
        assert hasattr(user_datum[_USER], "y"), (
            "User subgraph is missing the 'y' attribute for labels"
        )
        assert_tensor_equality(user_datum[_USER].y, user_datum[_USER].node)

        assert isinstance(story_datum, HeteroData), (
            f"Story subgraph should be a HeteroData for heterogeneous datasets, got {type(story_datum)}"
        )
        assert hasattr(story_datum[_STORY], "y"), (
            "Story subgraph is missing the 'y' attribute for labels"
        )
        assert_tensor_equality(story_datum[_STORY].y, story_datum[_STORY].node)

    shutdown_rpc()


def _run_distributed_neighbor_loader_with_multi_label_nodes_homogeneous(
    _,
    dataset: DistDataset,
    batch_size: int,
):
    create_test_process_group()

    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )

    for datum in loader:
        assert isinstance(datum, Data), (
            f"Subgraph should be a Data for homogeneous datasets, got {type(datum)}"
        )
        assert hasattr(datum, "y"), "Subgraph is missing the `y` attribute for labels"
        assert datum.y.ndim == 2, (
            f"Expected 2-D label tensor for multi-label, got shape {datum.y.shape}"
        )
        assert datum.y.shape[1] == 2, (
            f"Expected 2 label columns, got {datum.y.shape[1]}"
        )
        # Label column 0 equals the node ID; column 1 equals node ID * 10.
        assert_tensor_equality(datum.y[:, 0], datum.node)
        assert_tensor_equality(datum.y[:, 1], datum.node * 10)

    shutdown_rpc()


def _run_distributed_neighbor_loader_with_multi_label_nodes_heterogeneous(
    _,
    dataset: DistDataset,
    batch_size: int,
):
    create_test_process_group()

    assert isinstance(dataset.node_ids, Mapping)

    user_loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(_USER, dataset.node_ids[_USER]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )

    story_loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(_STORY, dataset.node_ids[_STORY]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )

    for user_datum, story_datum in zip(user_loader, story_loader):
        assert isinstance(user_datum, HeteroData), (
            f"User subgraph should be a HeteroData for heterogeneous datasets, got {type(user_datum)}"
        )
        assert hasattr(user_datum[_USER], "y"), (
            "User subgraph is missing the 'y' attribute for labels"
        )
        assert user_datum[_USER].y.ndim == 2, (
            f"Expected 2-D label tensor for multi-label user, got shape {user_datum[_USER].y.shape}"
        )
        assert user_datum[_USER].y.shape[1] == 2, (
            f"Expected 2 label columns for user, got {user_datum[_USER].y.shape[1]}"
        )
        assert_tensor_equality(user_datum[_USER].y[:, 0], user_datum[_USER].node)
        assert_tensor_equality(user_datum[_USER].y[:, 1], user_datum[_USER].node * 10)

        assert isinstance(story_datum, HeteroData), (
            f"Story subgraph should be a HeteroData for heterogeneous datasets, got {type(story_datum)}"
        )
        assert hasattr(story_datum[_STORY], "y"), (
            "Story subgraph is missing the 'y' attribute for labels"
        )
        assert story_datum[_STORY].y.ndim == 2, (
            f"Expected 2-D label tensor for multi-label story, got shape {story_datum[_STORY].y.shape}"
        )
        assert story_datum[_STORY].y.shape[1] == 2, (
            f"Expected 2 label columns for story, got {story_datum[_STORY].y.shape[1]}"
        )
        assert_tensor_equality(story_datum[_STORY].y[:, 0], story_datum[_STORY].node)
        assert_tensor_equality(
            story_datum[_STORY].y[:, 1], story_datum[_STORY].node * 10
        )

    shutdown_rpc()


def _run_cora_supervised_node_classification(
    _,
    dataset: DistDataset,
    batch_size: int,
):
    """Run CORA supervised node classification test using DistNeighborLoader."""
    create_test_process_group()

    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=to_homogeneous(dataset.train_node_ids),
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )

    for datum in loader:
        assert isinstance(datum, Data), (
            f"Subgraph should be a Data for homogeneous datasets, got {type(datum)}"
        )
        assert hasattr(datum, "y"), (
            "Node labels should be present for supervised node classification"
        )
        assert datum.y.size(0) == datum.node.size(0), (
            f"Number of labels should match number of nodes, got {datum.y.size(0)} labels and {datum.node.size(0)} nodes"
        )

    shutdown_rpc()


def _run_featureless_edge_ids_absent(
    _,
    dataset: DistDataset,
    holder: MutableMapping,
):
    # A featureless dataset should sample with with_edge=False, so GLT never
    # attaches sampled edge ids (``data.edge``) to the produced batches.
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    edge_ids_absent = True
    for datum in loader:
        assert isinstance(datum, Data)
        if getattr(datum, "edge", None) is not None:
            edge_ids_absent = False
        count += 1
    holder["edge_ids_absent"] = edge_ids_absent
    holder["count"] = count
    shutdown_rpc()


class DistributedNeighborLoaderTest(TestCase):
    def setUp(self):
        super().setUp()
        self._world_size = 1

    def tearDown(self):
        if torch.distributed.is_initialized():
            print("Destroying process group")
            # Ensure the process group is destroyed after each test
            # to avoid interference with subsequent tests
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_distributed_neighbor_loader(self):
        expected_data_count = 2708

        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
            _port=get_free_port(),
        )
        mp.spawn(
            fn=_run_distributed_neighbor_loader,
            args=(dataset, expected_data_count),
        )

    def test_isolated_sampling_worker_groups_exact_once_and_shutdown(self):
        expected_data_count = 2708
        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
            _port=get_free_port(),
        )

        mp.spawn(
            fn=_run_isolated_distributed_neighbor_loader,
            args=(dataset, expected_data_count),
        )

    def test_two_parent_two_partition_isolated_groups(self):
        spawn_context = mp.get_context("spawn")
        result_queue = spawn_context.SimpleQueue()
        mp.spawn(
            fn=_run_two_parent_isolated_distributed_neighbor_loader,
            args=(get_free_port(), get_free_port(), result_queue),
            nprocs=2,
        )
        results = [result_queue.get() for _ in range(2)]
        results.sort(key=lambda result: result["rank"])

        self.assertEqual(
            sorted(seed_id for result in results for seed_id in result["seed_ids"]),
            list(range(2708)),
        )
        for rank, result in enumerate(results):
            self.assertTrue(result["lease_closed"])
            self.assertEqual(result["ready_workers"], [0, 1])
            self.assertEqual(
                {mapping["worker_index"] for mapping in result["mappings"]},
                {0, 1},
            )
            self.assertEqual(
                {mapping["rank"] for mapping in result["mappings"]}, {rank}
            )
            self.assertEqual(
                {mapping["world_size"] for mapping in result["mappings"]}, {2}
            )
            self.assertEqual(len({mapping["pid"] for mapping in result["mappings"]}), 2)
        self.assertEqual(
            [mapping["port"] for mapping in results[0]["mappings"]],
            [mapping["port"] for mapping in results[1]["mappings"]],
        )
        self.assertEqual(
            [mapping["group_name"] for mapping in results[0]["mappings"]],
            [mapping["group_name"] for mapping in results[1]["mappings"]],
        )

    def test_cross_parent_startup_failure_reaps_both_sides(self):
        spawn_context = mp.get_context("spawn")
        result_queue = spawn_context.SimpleQueue()
        master_port = get_free_port()
        mp.spawn(
            fn=_run_cross_parent_isolated_failure,
            args=(master_port, result_queue),
            nprocs=2,
        )
        results = [result_queue.get() for _ in range(2)]
        results.sort(key=lambda result: result["rank"])

        self.assertTrue(
            "before initialization completed" in results[0]["error"]
            or "timed out waiting" in results[0]["error"]
        )
        self.assertIn("injected remote member failure", results[1]["error"])
        for result in results:
            self.assertLess(result["elapsed"], 10)
            self.assertEqual(result["worker_count"], 1)
            self.assertTrue(result["all_dead"])
            self.assertTrue(result["resources_closed"])

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.assertNotEqual(probe.connect_ex(("127.0.0.1", master_port)), 0)
        finally:
            probe.close()

    def test_infinite_distributed_neighbor_loader(self):
        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
            _port=get_free_port(),
        )

        assert isinstance(dataset.node_ids, torch.Tensor)

        num_nodes = dataset.node_ids.size(0)

        # Let's ensure we can iterate across the dataset twice with the infinite iterator
        max_num_batches = num_nodes * 2

        mp.spawn(
            fn=_run_infinite_distributed_neighbor_loader,
            args=(dataset, max_num_batches),
        )

    # TODO: (svij) - Figure out why this test is failing on Google Cloud Build
    @unittest.skip("Failing on Google Cloud Build - skiping for now")
    def test_distributed_neighbor_loader_heterogeneous(self):
        expected_data_count = 4057

        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
        )

        mp.spawn(
            fn=_run_distributed_heterogeneous_neighbor_loader,
            args=(dataset, expected_data_count),
        )

    def test_random_loading_labeled_homogeneous(self):
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

        assert isinstance(dataset.node_ids, Mapping)
        mp.spawn(
            fn=_run_distributed_neighbor_loader_labeled_homogeneous,
            args=(dataset, to_homogeneous(dataset.node_ids).size(0)),
        )

    def test_multiple_neighbor_loader(self):
        expected_data_count = 2708

        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
            _port=get_free_port(),
        )
        mp.spawn(
            fn=_run_multiple_neighbor_loader,
            args=(dataset, expected_data_count),
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

    def test_distributed_neighbor_loader_with_multi_label_nodes_homogeneous(self):
        # Labels: shape (10, 2) where col 0 = node_id, col 1 = node_id * 10.
        # This verifies that the full multi-column tensor survives GLT's T[0] truncation.
        n = 10
        labels = torch.stack([torch.arange(n), torch.arange(n) * 10], dim=1)
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(n),
            edge_partition_book=torch.zeros(n),
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
                edge_ids=None,
            ),
            partitioned_node_features=FeaturePartitionData(
                feats=torch.zeros(n, 2), ids=torch.arange(n)
            ),
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=FeaturePartitionData(
                feats=labels, ids=torch.arange(n)
            ),
        )

        dataset = DistDataset(rank=0, world_size=1, edge_dir="in")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_distributed_neighbor_loader_with_multi_label_nodes_homogeneous,
            args=(dataset, 1),
        )

    def test_distributed_neighbor_loader_with_multi_label_nodes_heterogeneous(self):
        # Labels: shape (5, 2) where col 0 = node_id, col 1 = node_id * 10.
        n = 5
        labels = torch.stack([torch.arange(n), torch.arange(n) * 10], dim=1)
        partition_output = PartitionOutput(
            node_partition_book={
                _USER: torch.zeros(n),
                _STORY: torch.zeros(n),
            },
            edge_partition_book={
                _USER_TO_STORY: torch.zeros(n),
                _STORY_TO_USER: torch.zeros(n),
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
                    feats=torch.zeros(n, 2), ids=torch.arange(n)
                ),
                _STORY: FeaturePartitionData(
                    feats=torch.zeros(n, 2), ids=torch.arange(n)
                ),
            },
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels={
                _USER: FeaturePartitionData(feats=labels, ids=torch.arange(n)),
                _STORY: FeaturePartitionData(feats=labels, ids=torch.arange(n)),
            },
        )

        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        mp.spawn(
            fn=_run_distributed_neighbor_loader_with_multi_label_nodes_heterogeneous,
            args=(dataset, 1),
        )

    def test_cora_supervised_node_classification(self):
        """Test CORA dataset for supervised node classification task."""
        create_test_process_group()
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

        splitter = DistNodeSplitter()

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
            args=(dataset, 18),
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
            args=(dataset, 18),
        )

    @parameterized.expand(
        [
            param(
                "input_nodes is None and dataset.node_ids is None",
                expected_error=ValueError,
                dataset=DistDataset(rank=0, world_size=1, edge_dir="out"),
                num_neighbors=[2, 2],
                input_nodes=None,
            ),
            param(
                "input_nodes is None for heterogeneous dataset",
                expected_error=ValueError,
                dataset=DistDataset(
                    rank=0,
                    world_size=1,
                    edge_dir="out",
                    graph_partition={},
                    node_partition_book={},
                    node_ids={NodeType("a"): torch.tensor([1, 2, 3])},
                ),
                num_neighbors=[2, 2],
                input_nodes=None,
            ),
            param(
                "input_nodes is a dict (colocated mode expects Tensor)",
                expected_error=ValueError,
                dataset=DistDataset(rank=0, world_size=1, edge_dir="out"),
                num_neighbors=[2, 2],
                input_nodes={0: torch.tensor([10])},
            ),
            param(
                "input_nodes is tuple with dict as second element",
                expected_error=ValueError,
                dataset=DistDataset(rank=0, world_size=1, edge_dir="out"),
                num_neighbors=[2, 2],
                input_nodes=(NodeType("a"), {0: torch.tensor([10])}),
            ),
            param(
                "Heterogeneous dataset with tensor input_nodes (not labeled homogeneous)",
                expected_error=ValueError,
                dataset=DistDataset(
                    rank=0,
                    world_size=1,
                    edge_dir="out",
                    graph_partition={},
                    node_partition_book={},
                    node_ids={
                        NodeType("a"): torch.tensor([1, 2]),
                        NodeType("b"): torch.tensor([3, 4]),
                    },
                ),
                num_neighbors=[2, 2],
                input_nodes=torch.tensor([10]),
            ),
            param(
                "input_nodes is None (graph store mode)",
                expected_error=ValueError,
                dataset=MockRemoteDistDataset(num_storage_nodes=2),
                num_neighbors=[2, 2],
                input_nodes=None,
            ),
            param(
                "input_nodes is a Tensor (graph store mode expects Mapping)",
                expected_error=ValueError,
                dataset=MockRemoteDistDataset(num_storage_nodes=2),
                num_neighbors=[2, 2],
                input_nodes=torch.tensor([10, 20]),
            ),
            param(
                "input_nodes is tuple with Tensor (graph store mode expects Mapping)",
                expected_error=ValueError,
                dataset=MockRemoteDistDataset(num_storage_nodes=2),
                num_neighbors=[2, 2],
                input_nodes=(NodeType("a"), torch.tensor([10, 20])),
            ),
            param(
                "Heterogeneous input without edge_types",
                expected_error=ValueError,
                dataset=MockRemoteDistDataset(num_storage_nodes=2, edge_types=None),
                num_neighbors=[2, 2],
                input_nodes=(
                    NodeType("a"),
                    {0: torch.tensor([10]), 1: torch.tensor([20])},
                ),
            ),
            param(
                "Server rank exceeds num_storage_nodes",
                expected_error=ValueError,
                dataset=MockRemoteDistDataset(num_storage_nodes=2),
                num_neighbors=[2, 2],
                input_nodes={0: torch.tensor([10]), 5: torch.tensor([20])},
            ),
            param(
                "Server rank is negative",
                expected_error=ValueError,
                dataset=MockRemoteDistDataset(num_storage_nodes=2),
                num_neighbors=[2, 2],
                input_nodes={-1: torch.tensor([10]), 0: torch.tensor([20])},
            ),
        ]
    )
    def test_distributed_neighbor_loader_invalid_inputs_colocated(
        self,
        _: str,
        expected_error: type[BaseException],
        **kwargs,
    ):
        create_test_process_group()
        with self.assertRaises(expected_error):
            DistNeighborLoader(**kwargs)


class WithEdgeDerivationTest(TestCase):
    """Covers ``create_sampling_config`` deriving ``with_edge`` from edge-feature presence.

    Exercises both paths: ``with_edge=True`` when the dataset has edge features
    (homogeneous or per-edge-type), and ``with_edge=False`` when it has none.
    """

    def test_config_with_edge_false_when_no_edge_features(self) -> None:
        schema = DatasetSchema(
            is_homogeneous_with_labeled_edge_type=False,
            edge_types=None,
            node_feature_info=None,
            edge_feature_info=None,  # no edge features
            edge_dir="out",
        )
        config = BaseDistLoader.create_sampling_config(
            num_neighbors=[2, 2], dataset_schema=schema
        )
        self.assertFalse(config.with_edge)

    def test_config_with_edge_true_when_edge_features_present(self) -> None:
        schema = DatasetSchema(
            is_homogeneous_with_labeled_edge_type=False,
            edge_types=None,
            node_feature_info=None,
            edge_feature_info=FeatureInfo(dim=4, dtype=torch.float32),
            edge_dir="out",
        )
        config = BaseDistLoader.create_sampling_config(
            num_neighbors=[2, 2], dataset_schema=schema
        )
        self.assertTrue(config.with_edge)

    def test_config_with_edge_true_for_heterogeneous_edge_feature_dict(self) -> None:
        schema = DatasetSchema(
            is_homogeneous_with_labeled_edge_type=False,
            edge_types=[_USER_TO_STORY, _STORY_TO_USER],
            node_feature_info=None,
            edge_feature_info={
                _USER_TO_STORY: FeatureInfo(dim=4, dtype=torch.float32),
            },
            edge_dir="out",
        )
        config = BaseDistLoader.create_sampling_config(
            num_neighbors=[2, 2], dataset_schema=schema
        )
        self.assertTrue(config.with_edge)

    def test_featureless_dataset_produces_batches_without_edge_ids(self) -> None:
        # Homogeneous dataset with node features but no edge features: the loader
        # must still yield batches, and none carry sampled edge ids.
        partition_output = PartitionOutput(
            node_partition_book=torch.zeros(5),
            edge_partition_book=torch.zeros(5),
            partitioned_edge_index=GraphPartitionData(
                edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
                edge_ids=None,
            ),
            partitioned_node_features=FeaturePartitionData(
                feats=torch.zeros(5, 2), ids=torch.arange(5)
            ),
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        manager = mp.Manager()
        holder: MutableMapping = manager.dict()
        proc = mp.spawn(
            fn=_run_featureless_edge_ids_absent,
            args=(dataset, holder),
            join=False,
        )
        proc.join(timeout=120)
        self.assertGreater(holder["count"], 0)
        self.assertTrue(holder["edge_ids_absent"])


class ColocatedWorkerOptionsTest(TestCase):
    def test_post_lease_setup_failure_closes_lease(self) -> None:
        dataset = mock.Mock(spec=DistDataset)
        dataset.node_ids = torch.arange(4)
        dataset.num_partitions = 1
        lease = mock.Mock()
        loader = object.__new__(DistNeighborLoader)
        loader._shutdowned = True

        with (
            mock.patch.object(
                BaseDistLoader,
                "initialize_colocated_sampling_worker",
            ),
            mock.patch.object(
                BaseDistLoader,
                "create_colocated_sampling_rpc_specs",
                return_value=(20000, None, lease),
            ),
            mock.patch.object(
                BaseDistLoader,
                "create_colocated_worker_options",
                side_effect=ValueError("injected worker-options failure"),
            ),
            self.assertRaisesRegex(ValueError, "injected worker-options failure"),
        ):
            loader._setup_for_colocated(
                input_nodes=dataset.node_ids,
                dataset=dataset,
                local_rank=0,
                local_world_size=1,
                device=torch.device("cpu"),
                master_ip_address="127.0.0.1",
                node_rank=0,
                node_world_size=1,
                num_workers=2,
                worker_concurrency=1,
                channel_size="64MB",
                num_cpu_threads=1,
                num_rpc_threads=1,
                one_rpc_group_per_sampling_worker=True,
            )

        lease.close.assert_called_once_with()

    def test_post_lease_setup_and_transient_close_failure_retries_lease(self) -> None:
        dataset = mock.Mock(spec=DistDataset)
        dataset.node_ids = torch.arange(4)
        dataset.num_partitions = 1
        lease = SamplingPortLease(
            ports=(20000,),
            lock_file_descriptors=(10,),
        )
        loader = object.__new__(DistNeighborLoader)
        loader._shutdowned = True

        with (
            mock.patch.object(
                BaseDistLoader,
                "initialize_colocated_sampling_worker",
            ),
            mock.patch.object(
                BaseDistLoader,
                "create_colocated_sampling_rpc_specs",
                return_value=(20000, None, lease),
            ),
            mock.patch.object(
                BaseDistLoader,
                "create_colocated_worker_options",
                side_effect=ValueError("injected worker-options failure"),
            ),
            mock.patch(
                "gigl.distributed.dist_sampling_producer.os.close",
                side_effect=[OSError("transient fd close failure"), None],
            ) as close_fd,
            self.assertRaisesRegex(ValueError, "injected worker-options failure"),
        ):
            loader._setup_for_colocated(
                input_nodes=dataset.node_ids,
                dataset=dataset,
                local_rank=0,
                local_world_size=1,
                device=torch.device("cpu"),
                master_ip_address="127.0.0.1",
                node_rank=0,
                node_world_size=1,
                num_workers=2,
                worker_concurrency=1,
                channel_size="64MB",
                num_cpu_threads=1,
                num_rpc_threads=1,
                one_rpc_group_per_sampling_worker=True,
            )

        self.assertTrue(lease._closed)
        self.assertEqual(close_fd.call_count, 2)

    def test_loader_rejects_invalid_rpc_threads_before_setup(self) -> None:
        loader = DistNeighborLoader.__new__(DistNeighborLoader)

        with self.assertRaisesRegex(
            ValueError, "num_rpc_threads must be positive, received 0"
        ):
            loader.__init__(
                dataset=cast(DistDataset, object()),
                num_neighbors=[1],
                num_rpc_threads=0,
            )

    def test_loader_rejects_invalid_sampling_seed_before_setup(self) -> None:
        invalid_values = (-1, 1 << 32)
        for value in invalid_values:
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "must be uint32"),
            ):
                DistNeighborLoader.__init__(
                    DistNeighborLoader.__new__(DistNeighborLoader),
                    dataset=cast(DistDataset, object()),
                    num_neighbors=[1],
                    sampling_run_seed=value,
                )

        for value in (True, "1"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(TypeError, "integer uint32"),
            ):
                DistNeighborLoader.__init__(
                    DistNeighborLoader.__new__(DistNeighborLoader),
                    dataset=cast(DistDataset, object()),
                    num_neighbors=[1],
                    sampling_run_seed=cast(int, value),
                )

    def test_loader_forwards_seed_and_global_parent_identity(self) -> None:
        dataset = object.__new__(DistDataset)
        runtime = SimpleNamespace(
            rank=259,
            world_size=260,
            local_rank=3,
            local_world_size=4,
            node_rank=64,
            node_world_size=65,
            master_ip_address="127.0.0.1",
            should_cleanup_distributed_context=False,
        )
        worker_options = BaseDistLoader.create_colocated_worker_options(
            dataset_num_partitions=65,
            num_workers=2,
            worker_concurrency=1,
            num_rpc_threads=1,
            master_ip_address="127.0.0.1",
            master_port=20000,
            channel_size="64MB",
            pin_memory=False,
        )
        sampling_config = mock.Mock(seed=None)
        producer = mock.Mock(spec=DistSamplingProducer)

        with (
            mock.patch.object(BaseDistLoader, "resolve_runtime", return_value=runtime),
            mock.patch.object(BaseDistLoader, "validate_for_weighted_sampling"),
            mock.patch.object(
                DistNeighborLoader,
                "_setup_for_colocated",
                return_value=(
                    mock.Mock(),
                    worker_options,
                    mock.Mock(spec=DatasetSchema),
                    None,
                    None,
                ),
            ),
            mock.patch.object(
                BaseDistLoader,
                "create_sampling_config",
                return_value=sampling_config,
            ),
            mock.patch.object(
                BaseDistLoader, "create_mp_producer", return_value=producer
            ) as create_producer,
            mock.patch.object(BaseDistLoader, "__init__", return_value=None),
            mock.patch(
                "gigl.distributed.distributed_neighborloader.resolve_sampler_options",
                return_value=mock.Mock(),
            ),
            mock.patch(
                "gigl.distributed.distributed_neighborloader.gigl.distributed.utils.get_available_device",
                return_value=torch.device("cpu"),
            ),
        ):
            DistNeighborLoader(
                dataset=dataset,
                num_neighbors=[1],
                sampling_run_seed=0xA5A5A5A5,
                process_start_gap_seconds=0,
            )

        self.assertEqual(
            create_producer.call_args.kwargs["sampling_run_seed"], 0xA5A5A5A5
        )
        self.assertEqual(create_producer.call_args.kwargs["parent_global_rank"], 259)
        self.assertEqual(create_producer.call_args.kwargs["parent_world_size"], 260)
        self.assertIsNone(sampling_config.seed)

    def test_graph_store_seed_rejected_before_runtime_or_backend_setup(self) -> None:
        remote_dataset = MockRemoteDistDataset(num_storage_nodes=1)
        with (
            mock.patch.object(BaseDistLoader, "resolve_runtime") as resolve_runtime,
            mock.patch.object(
                DistNeighborLoader, "_setup_for_graph_store"
            ) as setup_graph_store,
            self.assertRaisesRegex(ValueError, "only supported in colocated"),
        ):
            DistNeighborLoader(
                dataset=remote_dataset,
                num_neighbors=[1],
                sampling_run_seed=1,
            )

        resolve_runtime.assert_not_called()
        setup_graph_store.assert_not_called()

    def test_rpc_thread_default_override_and_validation(self) -> None:
        common_options = {
            "dataset_num_partitions": 65,
            "num_workers": 4,
            "worker_concurrency": 2,
            "master_ip_address": "127.0.0.1",
            "master_port": 12345,
            "channel_size": "2GB",
            "pin_memory": False,
        }

        default_options = BaseDistLoader.create_colocated_worker_options(
            **common_options,
            num_rpc_threads=None,
        )
        overridden_options = BaseDistLoader.create_colocated_worker_options(
            **common_options,
            num_rpc_threads=8,
        )

        self.assertEqual(default_options.num_rpc_threads, 16)
        self.assertEqual(overridden_options.num_rpc_threads, 8)
        with self.assertRaises(ValueError):
            BaseDistLoader.create_colocated_worker_options(
                **common_options,
                num_rpc_threads=0,
            )

    def test_topology_agreement_rejects_mixed_worker_counts(self) -> None:
        def gather_mismatched(signatures, local_signature) -> None:
            signatures[:] = [
                local_signature,
                (*local_signature[:2], 3, *local_signature[3:]),
            ]

        with (
            mock.patch.object(torch.distributed, "get_world_size", return_value=2),
            mock.patch.object(
                torch.distributed,
                "all_gather_object",
                side_effect=gather_mismatched,
            ),
            self.assertRaisesRegex(ValueError, "topology differs"),
        ):
            BaseDistLoader._validate_colocated_sampling_topology_agreement(
                one_rpc_group_per_sampling_worker=True,
                use_all2all=False,
                num_workers=4,
                local_world_size=2,
                node_world_size=65,
            )

    def test_topology_agreement_rejects_mixed_all_to_all(self) -> None:
        def gather_mismatched(signatures, local_signature) -> None:
            signatures[:] = [
                local_signature,
                (
                    local_signature[0],
                    not local_signature[1],
                    *local_signature[2:],
                ),
            ]

        with (
            mock.patch.object(torch.distributed, "get_world_size", return_value=2),
            mock.patch.object(
                torch.distributed,
                "all_gather_object",
                side_effect=gather_mismatched,
            ),
            self.assertRaisesRegex(ValueError, "topology differs"),
        ):
            BaseDistLoader._validate_colocated_sampling_topology_agreement(
                one_rpc_group_per_sampling_worker=True,
                use_all2all=False,
                num_workers=4,
                local_world_size=2,
                node_world_size=65,
            )

    def test_context_partition_validation_fails_collectively(self) -> None:
        dataset = mock.Mock(spec=DistDataset)
        dataset.num_partitions = 2
        dataset.partition_idx = 0
        context = mock.Mock()
        context.is_worker.return_value = True
        context.world_size = 2
        context.rank = 0
        context.group_name = "inference_group"

        def gather(identities, local_identity) -> None:
            identities[:] = [
                local_identity,
                {
                    **local_identity,
                    "local_rank": 0,
                    "context_rank": 1,
                    "data_partition": 0,
                },
            ]

        with (
            mock.patch(
                "gigl.distributed.base_dist_loader.get_context",
                return_value=context,
            ),
            mock.patch.object(torch.distributed, "get_world_size", return_value=2),
            mock.patch.object(
                torch.distributed,
                "all_gather_object",
                side_effect=gather,
            ),
            self.assertRaisesRegex(RuntimeError, "context/partition identity"),
        ):
            BaseDistLoader._resolve_isolated_context_collectively(
                dataset=dataset,
                local_rank=0,
                local_world_size=1,
                node_world_size=2,
            )

    def test_default_mode_keeps_legacy_dynamic_port_path(self) -> None:
        dataset = mock.Mock(spec=DistDataset)
        with (
            mock.patch.object(
                BaseDistLoader,
                "_validate_colocated_sampling_topology_agreement",
            ),
            mock.patch(
                "gigl.distributed.base_dist_loader.gigl.distributed.utils.get_free_ports_from_master_node",
                return_value=[24000, 24001],
            ),
            mock.patch.object(
                BaseDistLoader,
                "_reserve_isolated_sampling_ports",
            ) as reserve,
        ):
            port, specs, lease = BaseDistLoader.create_colocated_sampling_rpc_specs(
                dataset=dataset,
                num_workers=4,
                local_rank=1,
                local_world_size=2,
                node_world_size=65,
                one_rpc_group_per_sampling_worker=False,
                use_all2all=False,
            )

        self.assertEqual(port, 24001)
        self.assertIsNone(specs)
        self.assertIsNone(lease)
        reserve.assert_not_called()

    def test_isolated_ports_are_nonoverlapping_by_local_rank(self) -> None:
        dataset = mock.Mock(spec=DistDataset)
        dataset.num_partitions = 65
        dataset.partition_idx = 13
        context = mock.Mock()
        context.is_worker.return_value = True
        context.world_size = 65
        context.rank = 13
        context.group_name = "inference_group"

        def reserve_ports(*, num_workers, local_rank, local_world_size, **_) -> tuple:
            start = 20000 + local_rank * num_workers
            ports = list(range(start, start + num_workers))
            return ports, SamplingPortLease(tuple(ports))

        with (
            mock.patch.object(
                BaseDistLoader,
                "_validate_colocated_sampling_topology_agreement",
            ),
            mock.patch.object(
                BaseDistLoader,
                "_resolve_isolated_context_collectively",
                return_value=context,
            ),
            mock.patch.object(
                BaseDistLoader,
                "_reserve_isolated_sampling_ports",
                side_effect=reserve_ports,
            ),
        ):
            port_0, specs_0, lease_0 = (
                BaseDistLoader.create_colocated_sampling_rpc_specs(
                    dataset=dataset,
                    num_workers=4,
                    local_rank=0,
                    local_world_size=2,
                    node_world_size=65,
                    one_rpc_group_per_sampling_worker=True,
                    use_all2all=False,
                )
            )
            port_1, specs_1, lease_1 = (
                BaseDistLoader.create_colocated_sampling_rpc_specs(
                    dataset=dataset,
                    num_workers=4,
                    local_rank=1,
                    local_world_size=2,
                    node_world_size=65,
                    one_rpc_group_per_sampling_worker=True,
                    use_all2all=False,
                )
            )

        assert specs_0 is not None and specs_1 is not None
        assert lease_0 is not None and lease_1 is not None
        self.assertEqual(port_0, 20000)
        self.assertEqual(port_1, 20004)
        self.assertEqual(
            [spec.master_port for spec in specs_0], list(range(20000, 20004))
        )
        self.assertEqual(
            [spec.master_port for spec in specs_1], list(range(20004, 20008))
        )
        self.assertTrue(
            {spec.master_port for spec in specs_0}.isdisjoint(
                spec.master_port for spec in specs_1
            )
        )

    def test_port_reservation_retries_after_any_parent_collision(self) -> None:
        first_lease = SamplingPortLease((20000, 20001, 20002, 20003))
        second_lease = SamplingPortLease((20008, 20009, 20010, 20011))
        gather_results = iter(
            [
                [0, None],
                [(True, ""), (False, "OSError: port occupied")],
                [(True, ""), (True, "")],
            ]
        )

        def gather(sequence, _) -> None:
            sequence[:] = next(gather_results)

        with (
            mock.patch("builtins.open", mock.mock_open(read_data="32768 60999")),
            mock.patch.object(torch.distributed, "get_rank", return_value=0),
            mock.patch.object(torch.distributed, "get_world_size", return_value=2),
            mock.patch.object(
                torch.distributed,
                "all_gather_object",
                side_effect=gather,
            ),
            mock.patch.object(
                BaseDistLoader,
                "_try_reserve_isolated_sampling_ports",
                side_effect=[first_lease, second_lease],
            ),
        ):
            ports, lease = BaseDistLoader._reserve_isolated_sampling_ports(
                num_workers=4,
                local_rank=0,
                local_world_size=2,
                is_group_master=True,
            )

        self.assertEqual(ports, list(range(20008, 20012)))
        self.assertIs(lease, second_lease)
        self.assertTrue(first_lease._closed)

    def test_port_candidate_collective_failure_retries_and_closes_lease(
        self,
    ) -> None:
        lease = SamplingPortLease(
            ports=(20000, 20001),
        )

        def gather(sequence, local_value) -> None:
            if local_value == 0:
                sequence[:] = [0]
                return
            raise RuntimeError("injected candidate result collective failure")

        with (
            mock.patch("builtins.open", mock.mock_open(read_data="32768 60999")),
            mock.patch(
                "gigl.distributed.base_dist_loader.secrets.randbelow",
                return_value=0,
            ),
            mock.patch.object(torch.distributed, "get_rank", return_value=0),
            mock.patch.object(torch.distributed, "get_world_size", return_value=1),
            mock.patch.object(
                torch.distributed,
                "all_gather_object",
                side_effect=gather,
            ),
            mock.patch.object(
                BaseDistLoader,
                "_try_reserve_isolated_sampling_ports",
                return_value=lease,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "injected candidate result collective failure",
            ),
        ):
            BaseDistLoader._reserve_isolated_sampling_ports(
                num_workers=2,
                local_rank=0,
                local_world_size=1,
                is_group_master=True,
            )

        self.assertTrue(lease._closed)

    def test_master_port_lease_holds_strict_listening_reservations(self) -> None:
        reservations = [mock.Mock(), mock.Mock()]
        with (
            mock.patch(
                "gigl.distributed.base_dist_loader.os.open",
                side_effect=[10, 11],
            ),
            mock.patch("gigl.distributed.base_dist_loader.os.close") as close_fd,
            mock.patch("gigl.distributed.base_dist_loader.fcntl.flock") as flock,
            mock.patch(
                "gigl.distributed.base_dist_loader.socket.socket",
                side_effect=reservations,
            ),
        ):
            lease = BaseDistLoader._try_reserve_isolated_sampling_ports(
                [21000, 21001], reserve_sockets=True
            )
            self.assertEqual(set(lease.reservations), {21000, 21001})
            for port, reservation in zip((21000, 21001), reservations):
                reservation.bind.assert_called_once_with(("", port))
                reservation.listen.assert_called_once_with(1)
                reservation.setsockopt.assert_not_called()
            lease.close()

        self.assertEqual(flock.call_count, 2)
        self.assertEqual(close_fd.call_count, 2)
        self.assertTrue(all(reservation.close.called for reservation in reservations))


# NOTE on the test strategy: GiGL loaders always sample via the multiprocess
# producer, which spawns worker subprocesses with a *fresh* interpreter
# (`mp.get_context("spawn")`, dist_sampling_producer.py). A `mock.patch` applied in the
# loader process therefore never reaches the sampler running in that subprocess, so we
# cannot inject a synthetic failure by mocking the sampler. Instead we reproduce a real
# sampler failure end-to-end: a heterogeneous dataset with edge features on only a
# subset of its message-passing edge types. When the featureless type is reached during
# sampling, its feature lookup raises `KeyError` inside the sampling coroutine — the exact
# swallowed-exception case this change surfaces. Without the change this hangs forever, so
# the test uses a bounded join.


def _run_partial_edge_feature_coverage_raises(
    _,
    dataset: DistDataset,
    error_holder,
):
    create_test_process_group()
    assert isinstance(dataset.node_ids, Mapping)
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(_USER, dataset.node_ids[_USER]),  # ty: ignore[invalid-argument-type]
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
    )
    try:
        for _datum in loader:
            pass
    except RuntimeError as e:
        error_holder["msg"] = str(e)
    finally:
        shutdown_rpc()


class TestSamplingErrorPropagation(TestCase):
    def _build_partial_edge_feature_dataset(self) -> DistDataset:
        """Build a hetero dataset with edge features on only one message-passing type.

        ``user-to-story`` has edge features; ``story-to-user`` does not. Both are
        reachable from ``user`` seeds within a 2-hop fanout, so the featureless type is
        actually sampled and its edge-feature lookup raises inside the coroutine.
        """
        n = 5
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        partition_output = PartitionOutput(
            node_partition_book={_USER: torch.zeros(n), _STORY: torch.zeros(n)},
            edge_partition_book={
                _USER_TO_STORY: torch.zeros(n),
                _STORY_TO_USER: torch.zeros(n),
            },
            partitioned_edge_index={
                _USER_TO_STORY: GraphPartitionData(
                    edge_index=edge_index, edge_ids=None
                ),
                _STORY_TO_USER: GraphPartitionData(
                    edge_index=edge_index, edge_ids=None
                ),
            },
            partitioned_node_features={
                _USER: FeaturePartitionData(
                    feats=torch.zeros(n, 2), ids=torch.arange(n)
                ),
                _STORY: FeaturePartitionData(
                    feats=torch.zeros(n, 2), ids=torch.arange(n)
                ),
            },
            partitioned_edge_features={
                _USER_TO_STORY: FeaturePartitionData(
                    feats=torch.ones(n, 3), ids=torch.arange(n)
                ),
            },
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
            partitioned_node_labels=None,
        )
        dataset = DistDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)
        return dataset

    def test_reachable_sampler_failure_raises_not_hangs(self) -> None:
        dataset = self._build_partial_edge_feature_dataset()
        manager = mp.Manager()
        error_holder = manager.dict()
        proc = mp.get_context("spawn").Process(
            target=_run_partial_edge_feature_coverage_raises,
            args=(0, dataset, error_holder),
        )
        proc.start()
        proc.join(timeout=180)  # bounded: the pre-fix behavior hangs indefinitely
        alive = proc.is_alive()
        if alive:
            proc.terminate()
            proc.join(timeout=10)
        self.assertFalse(
            alive, "loader hung instead of failing fast on a sampler error"
        )
        message = error_holder.get("msg", "")
        # The training process raised with the worker's real traceback embedded.
        self.assertIn("sampling worker failed", message.lower())
        self.assertIn("story-to-user", message)


if __name__ == "__main__":
    absltest.main()
