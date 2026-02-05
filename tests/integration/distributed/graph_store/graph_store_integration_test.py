import collections
import multiprocessing.context as py_mp_context
import os
import socket
import traceback
import unittest
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Literal, Optional, Union
from unittest import mock

import torch
import torch.multiprocessing as mp
from torch_geometric.data import Data, HeteroData

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.graph_store.compute import (
    init_compute_process,
    shutdown_compute_proccess,
)
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.graph_store.storage_main import storage_node_process
from gigl.distributed.utils.neighborloader import shard_nodes_by_process
from gigl.distributed.utils.networking import get_free_ports
from gigl.distributed.utils.partition_book import build_partition_book, get_ids_on_rank
from gigl.env.distributed import (
    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY,
    GraphStoreInfo,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
    DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
)
from gigl.utils.data_splitters import DistNodeAnchorLinkSplitter, DistNodeSplitter
from tests.test_assets.distributed.utils import assert_tensor_equality
from tests.test_assets.test_case import DEFAULT_TIMEOUT_SECONDS, TestCase

logger = Logger()


def _assert_sampler_input(
    cluster_info: GraphStoreInfo,
    sampler_input: dict[int, torch.Tensor],
    expected_sampler_input: dict[int, list[torch.Tensor]],
) -> None:
    rank_expected_sampler_input = expected_sampler_input[cluster_info.compute_node_rank]
    for i in range(cluster_info.compute_cluster_world_size):
        if i == torch.distributed.get_rank():
            logger.info(
                f"Verifying sampler input for rank {i} / {cluster_info.compute_cluster_world_size}"
            )
            logger.info(f"--------------------------------")
            assert len(sampler_input) == len(rank_expected_sampler_input)
            for j, expected in enumerate(rank_expected_sampler_input):
                assert_tensor_equality(sampler_input[j], expected)
            logger.info(
                f"{i} / {cluster_info.compute_cluster_world_size} compute node rank input nodes verified"
            )
        torch.distributed.barrier()

    torch.distributed.barrier()


def _assert_ablp_input(
    cluster_info: GraphStoreInfo,
    ablp_result: dict[int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
) -> None:
    """Assert ABLP input structure and verify consistency across ranks on same compute node."""
    for i in range(cluster_info.compute_cluster_world_size):
        if i == torch.distributed.get_rank():
            logger.info(
                f"Verifying ABLP input for rank {i} / {cluster_info.compute_cluster_world_size}"
            )
            logger.info(f"--------------------------------")

            # Verify structure: dict mapping server_rank to (anchors, positive_labels, negative_labels)
            assert isinstance(
                ablp_result, dict
            ), f"Expected dict, got {type(ablp_result)}"
            assert (
                len(ablp_result) == cluster_info.num_storage_nodes
            ), f"Expected {cluster_info.num_storage_nodes} storage nodes in result, got {len(ablp_result)}"

            for server_rank, (
                anchors,
                positive_labels,
                negative_labels,
            ) in ablp_result.items():
                # Verify anchors shape (1D tensor)
                assert isinstance(
                    anchors, torch.Tensor
                ), f"Anchors should be a tensor, got {type(anchors)}"
                assert anchors.dim() == 1, f"Anchors should be 1D, got {anchors.dim()}D"
                assert len(anchors) > 0, "Anchors should not be empty"

                # Verify positive_labels shape (2D tensor: [num_anchors, num_positive_labels])
                assert isinstance(
                    positive_labels, torch.Tensor
                ), f"Positive labels should be a tensor, got {type(positive_labels)}"
                assert (
                    positive_labels.dim() == 2
                ), f"Positive labels should be 2D, got {positive_labels.dim()}D"
                assert positive_labels.shape[0] == len(
                    anchors
                ), f"Positive labels first dim should match anchors length, got {positive_labels.shape[0]} vs {len(anchors)}"

                # Verify negative_labels is None or has correct shape
                if negative_labels is not None:
                    assert isinstance(
                        negative_labels, torch.Tensor
                    ), f"Negative labels should be a tensor, got {type(negative_labels)}"
                    assert (
                        negative_labels.dim() == 2
                    ), f"Negative labels should be 2D, got {negative_labels.dim()}D"
                    assert negative_labels.shape[0] == len(
                        anchors
                    ), f"Negative labels first dim should match anchors length"

                logger.info(
                    f"Server rank {server_rank}: anchors shape={anchors.shape}, "
                    f"positive_labels shape={positive_labels.shape}, "
                    f"negative_labels shape={negative_labels.shape if negative_labels is not None else None}"
                )

            logger.info(
                f"{i} / {cluster_info.compute_cluster_world_size} compute node rank ABLP input verified"
            )
        torch.distributed.barrier()

    torch.distributed.barrier()

    # Gather ABLP data from all ranks and verify processes on same compute_node_rank have identical data
    local_anchors, local_positive, local_negative = ablp_result[0]
    local_data = (
        cluster_info.compute_node_rank,
        local_anchors.clone(),
        local_positive.clone(),
        local_negative.clone() if local_negative is not None else None,
    )
    gathered_data: list[tuple[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = [None] * cluster_info.compute_cluster_world_size  # type: ignore[list-item]
    torch.distributed.all_gather_object(gathered_data, local_data)

    # Group by compute_node_rank and verify all processes in same group have identical ABLP input
    my_compute_node_rank = cluster_info.compute_node_rank
    for (
        other_compute_node_rank,
        other_anchors,
        other_positive,
        other_negative,
    ) in gathered_data:
        if other_compute_node_rank == my_compute_node_rank:
            assert_tensor_equality(local_anchors, other_anchors)
            assert_tensor_equality(local_positive, other_positive)
            if local_negative is not None and other_negative is not None:
                assert_tensor_equality(local_negative, other_negative)
            else:
                assert local_negative is None and other_negative is None, (
                    f"Negative labels mismatch: local={local_negative is not None}, "
                    f"other={other_negative is not None}"
                )

    torch.distributed.barrier()
    logger.info(
        f"Rank {torch.distributed.get_rank()} verified processes on same compute_node_rank "
        f"({my_compute_node_rank}) have identical ABLP input"
    )


def _run_compute_train_tests(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    mp_sharing_dict: Optional[MutableMapping[str, torch.Tensor]],
    node_type: Optional[NodeType],
) -> None:
    """
    Simplified compute test for training mode that only verifies ABLP input.
    """
    init_compute_process(client_rank, cluster_info, compute_world_backend="gloo")

    remote_dist_dataset = RemoteDistDataset(
        cluster_info=cluster_info,
        local_rank=client_rank,
        mp_sharing_dict=mp_sharing_dict,
    )

    # Use default types for homogeneous graph
    test_node_type = (
        node_type if node_type is not None else DEFAULT_HOMOGENEOUS_NODE_TYPE
    )
    supervision_edge_type = DEFAULT_HOMOGENEOUS_EDGE_TYPE

    # Test get_ablp_input for train split
    ablp_result = remote_dist_dataset.get_ablp_input(
        split="train",
        rank=cluster_info.compute_node_rank,
        world_size=cluster_info.num_compute_nodes,
        node_type=test_node_type,
        supervision_edge_type=supervision_edge_type,
    )

    _assert_ablp_input(cluster_info, ablp_result)

    shutdown_compute_proccess()


@dataclass(frozen=True)
class ClientTrainProcessArgs:
    """Arguments for the client training process.

    Attributes:
        client_rank: Rank of this client in the compute cluster.
        cluster_info: Information about the distributed cluster.
        node_type: Type of nodes to process, None for homogeneous graphs.
        exception_dict: Shared dictionary for storing exceptions from processes.
    """

    client_rank: int
    cluster_info: GraphStoreInfo
    node_type: Optional[NodeType]
    exception_dict: MutableMapping[str, str]


def _client_train_process(args: ClientTrainProcessArgs) -> None:
    """Client process for training mode that spawns compute train tests."""
    logger.info(
        f"Initializing train client node {args.client_rank} / {args.cluster_info.num_compute_nodes}. "
        f"OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']}"
    )
    process_name = f"client_train_{args.client_rank}"
    try:
        mp_context = torch.multiprocessing.get_context("spawn")
        mp_sharing_dict = torch.multiprocessing.Manager().dict()
        client_processes: list[py_mp_context.SpawnProcess] = []
        logger.info("Starting train client processes")
        for i in range(args.cluster_info.num_processes_per_compute):
            client_process = mp_context.Process(
                target=_run_compute_train_tests,
                args=[
                    i,  # client_rank
                    args.cluster_info,  # cluster_info
                    mp_sharing_dict,  # mp_sharing_dict
                    args.node_type,  # node_type
                ],
            )
            client_processes.append(client_process)
        for client_process in client_processes:
            client_process.start()
        for client_process in client_processes:
            client_process.join(DEFAULT_TIMEOUT_SECONDS)
    except Exception:
        args.exception_dict[process_name] = traceback.format_exc()
        raise


def _run_compute_tests(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    mp_sharing_dict: Optional[MutableMapping[str, torch.Tensor]],
    node_type: Optional[NodeType],
    expected_sampler_input: dict[int, list[torch.Tensor]],
    expected_edge_types: Optional[list[EdgeType]],
) -> None:
    """
    Process target for "compute" nodes.
    Each "Client Process" (e.g. cluster_info.num_compute_nodes) will spawn as a process for each "num_processes_per_compute"
    """
    init_compute_process(client_rank, cluster_info, compute_world_backend="gloo")

    remote_dist_dataset = RemoteDistDataset(
        cluster_info=cluster_info,
        local_rank=client_rank,
        mp_sharing_dict=mp_sharing_dict,
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert (
        remote_dist_dataset.get_edge_dir() == "in"
    ), f"Edge direction must be 'in' for the test dataset. Got {remote_dist_dataset.get_edge_dir()}"
    assert (
        remote_dist_dataset.get_edge_feature_info() is not None
    ), "Edge feature info must not be None for the test dataset"
    assert (
        remote_dist_dataset.get_node_feature_info() is not None
    ), "Node feature info must not be None for the test dataset"
    ports = remote_dist_dataset.get_free_ports_on_storage_cluster(num_ports=2)
    assert len(ports) == 2, "Expected 2 free ports"
    if rank == 0:
        all_ports = [None] * torch.distributed.get_world_size()
    else:
        all_ports = None
    torch.distributed.gather_object(ports, all_ports)
    logger.info(f"All ports: {all_ports}")

    if rank == 0:
        assert isinstance(all_ports, list)
        for i, received_ports in enumerate(all_ports):
            assert (
                received_ports == ports
            ), f"Expected {ports} free ports, got {received_ports}"

    torch.distributed.barrier()
    logger.info("Verified that all ranks received the same free ports")

    sampler_input = remote_dist_dataset.get_node_ids(
        node_type=node_type,
        rank=cluster_info.compute_node_rank,
        world_size=cluster_info.num_compute_nodes,
    )
    _assert_sampler_input(cluster_info, sampler_input, expected_sampler_input)

    # test "simple" case where we don't have mp sharing dict too
    simple_sampler_input = RemoteDistDataset(
        cluster_info=cluster_info,
        local_rank=client_rank,
        mp_sharing_dict=None,
    ).get_node_ids(
        node_type=node_type,
        rank=cluster_info.compute_node_rank,
        world_size=cluster_info.num_compute_nodes,
    )
    _assert_sampler_input(cluster_info, simple_sampler_input, expected_sampler_input)

    assert (
        remote_dist_dataset.get_edge_types() == expected_edge_types
    ), f"Expected edge types {expected_edge_types}, got {remote_dist_dataset.get_edge_types()}"

    torch.distributed.barrier()
    if node_type is not None:
        input_nodes: Union[
            dict[int, torch.Tensor], tuple[NodeType, dict[int, torch.Tensor]]
        ] = (
            node_type,
            sampler_input,
        )
    else:
        input_nodes = sampler_input
    # Test the DistNeighborLoader
    loader = DistNeighborLoader(
        dataset=remote_dist_dataset,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
        input_nodes=input_nodes,
        num_workers=2,
        worker_concurrency=2,
    )
    count = 0
    for datum in loader:
        if node_type is not None:
            assert isinstance(datum, HeteroData)
        else:
            assert isinstance(datum, Data)
        count += 1
    torch.distributed.barrier()
    logger.info(f"Rank {torch.distributed.get_rank()} loaded {count} batches")
    # Verify that we sampled all nodes.
    count_tensor = torch.tensor(count, dtype=torch.int64)
    all_node_count = 0
    for rank_expected_sampler_input in expected_sampler_input.values():
        all_node_count += sum(len(nodes) for nodes in rank_expected_sampler_input)
    torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
    assert (
        count_tensor.item() == all_node_count
    ), f"Expected {all_node_count} total nodes, got {count_tensor.item()}"
    shutdown_compute_proccess()


@dataclass(frozen=True)
class ClientProcessArgs:
    """Arguments for the client process.

    Attributes:
        client_rank: Rank of this client in the compute cluster.
        cluster_info: Information about the distributed cluster.
        node_type: Type of nodes to process, None for homogeneous graphs.
        expected_sampler_input: Expected sampler input for each compute rank.
        expected_edge_types: Expected edge types for heterogeneous graphs.
        exception_dict: Shared dictionary for storing exceptions from processes.
    """

    client_rank: int
    cluster_info: GraphStoreInfo
    node_type: Optional[NodeType]
    expected_sampler_input: dict[int, list[torch.Tensor]]
    expected_edge_types: Optional[list[EdgeType]]
    exception_dict: MutableMapping[str, str]


def _client_process(args: ClientProcessArgs) -> None:
    process_name = f"client_{args.client_rank}"
    try:
        logger.info(
            f"Initializing client node {args.client_rank} / {args.cluster_info.num_compute_nodes}. OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']}, local client rank: {args.client_rank}"
        )
        mp_context = torch.multiprocessing.get_context("spawn")
        mp_sharing_dict = torch.multiprocessing.Manager().dict()
        client_processes: list[py_mp_context.SpawnProcess] = []
        logger.info("Starting client processes")
        for i in range(args.cluster_info.num_processes_per_compute):
            client_process = mp_context.Process(
                target=_run_compute_tests,
                args=[
                    i,  # client_rank
                    args.cluster_info,  # cluster_info
                    mp_sharing_dict,  # mp_sharing_dict
                    args.node_type,  # node_type
                    args.expected_sampler_input,  # expected_sampler_input
                    args.expected_edge_types,  # expected_edge_types
                ],
            )
            client_processes.append(client_process)
        for client_process in client_processes:
            client_process.start()
        for client_process in client_processes:
            client_process.join(DEFAULT_TIMEOUT_SECONDS)
    except Exception:
        args.exception_dict[process_name] = traceback.format_exc()
        raise


@dataclass(frozen=True)
class ServerProcessArgs:
    """Arguments for the server process.

    Attributes:
        cluster_info: Information about the distributed cluster.
        task_config_uri: URI to the task configuration.
        sample_edge_direction: Direction for edge sampling ("in" or "out").
        exception_dict: Shared dictionary for storing exceptions from processes.
        splitter: Optional splitter for node anchor link or node splitting.
    """

    cluster_info: GraphStoreInfo
    task_config_uri: Uri
    sample_edge_direction: Literal["in", "out"]
    exception_dict: MutableMapping[str, str]
    splitter: Optional[Union[DistNodeAnchorLinkSplitter, DistNodeSplitter]] = None


def _run_server_processes(args: ServerProcessArgs) -> None:
    process_name = f"server_{args.cluster_info.storage_node_rank}"
    try:
        logger.info(
            f"Initializing server processes. OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']}"
        )
        storage_node_process(
            storage_rank=args.cluster_info.storage_node_rank,
            cluster_info=args.cluster_info,
            task_config_uri=args.task_config_uri,
            sample_edge_direction=args.sample_edge_direction,
            splitter=args.splitter,
            tf_record_uri_pattern=".*tfrecord",
            storage_world_backend="gloo",
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        )
    except Exception:
        args.exception_dict[process_name] = traceback.format_exc()
        raise


def _get_expected_input_nodes_by_rank(
    num_nodes: int, cluster_info: GraphStoreInfo
) -> dict[int, list[torch.Tensor]]:
    """Get the expected sampler input for each compute rank.

    We generate the expected sampler input for each compute rank by sharding the nodes across the compute ranks.
    We then append the generated nodes to the expected sampler input for each compute rank.
    Example for num_nodes = 16, num_processes_per_compute = 1, num_compute_nodes = 2, num_storage_nodes = 2:
    {
    0: # compute rank 0
    [
        [0, 1, 3, 4], # From storage rank 0
        [8, 9, 11, 12] # From storage rank 1
    ]
    1: # compute rank 1
    [
        [5, 6, 7, 8], # From storage rank 0
        [13, 14, 15, 16] # From storage rank 1
    ],
    }


    Args:
        num_nodes (int): The number of nodes in the graph.
        cluster_info (GraphStoreInfo): The cluster information.

    Returns:
        dict[int, list[torch.Tensor]]: The expected sampler input for each compute rank.
    """
    partition_book = build_partition_book(
        num_entities=num_nodes, rank=0, world_size=cluster_info.num_storage_nodes
    )
    expected_sampler_input = collections.defaultdict(list)
    for server_rank in range(cluster_info.num_storage_nodes):
        server_nodes = get_ids_on_rank(partition_book=partition_book, rank=server_rank)
        for compute_rank in range(cluster_info.num_compute_nodes):
            generated_nodes = shard_nodes_by_process(
                input_nodes=server_nodes,
                local_process_rank=compute_rank,
                local_process_world_size=cluster_info.num_compute_nodes,
            )
            expected_sampler_input[compute_rank].append(generated_nodes)
    return dict(expected_sampler_input)


class GraphStoreIntegrationTest(TestCase):
    def test_graph_store_homogeneous(self):
        # Simulating two server machine, two compute machines.
        # Each machine has one process.
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        task_config_uri = cora_supervised_info.frozen_gbml_config_uri
        (
            cluster_master_port,
            storage_cluster_master_port,
            compute_cluster_master_port,
            master_port,
            rpc_master_port,
            rpc_wait_port,
        ) = get_free_ports(num_ports=6)
        host_ip = socket.gethostbyname(socket.gethostname())
        cluster_info = GraphStoreInfo(
            num_storage_nodes=2,
            num_compute_nodes=4,
            num_processes_per_compute=2,
            cluster_master_ip=host_ip,
            storage_cluster_master_ip=host_ip,
            compute_cluster_master_ip=host_ip,
            cluster_master_port=cluster_master_port,
            storage_cluster_master_port=storage_cluster_master_port,
            compute_cluster_master_port=compute_cluster_master_port,
            rpc_master_port=rpc_master_port,
            rpc_wait_port=rpc_wait_port,
        )

        num_cora_nodes = 2708
        expected_sampler_input = _get_expected_input_nodes_by_rank(
            num_cora_nodes, cluster_info
        )

        ctx = mp.get_context("spawn")
        manager = mp.Manager()
        exception_dict = manager.dict()
        launched_processes: list[py_mp_context.SpawnProcess] = []
        for i in range(cluster_info.num_compute_nodes):
            with mock.patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": host_ip,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(i),
                    "WORLD_SIZE": str(cluster_info.num_cluster_nodes),
                    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY: str(
                        cluster_info.num_processes_per_compute
                    ),
                },
                clear=False,
            ):
                client_args = ClientProcessArgs(
                    client_rank=i,
                    cluster_info=cluster_info,
                    node_type=None,  # None for homogeneous dataset
                    expected_sampler_input=expected_sampler_input,
                    expected_edge_types=None,  # None for homogeneous dataset
                    exception_dict=exception_dict,
                )
                client_process = ctx.Process(
                    target=_client_process,
                    args=[client_args],
                    name=f"client_{i}",
                )
                client_process.start()
                launched_processes.append(client_process)
        # Start server process
        for i in range(cluster_info.num_storage_nodes):
            with mock.patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": host_ip,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(i + cluster_info.num_compute_nodes),
                    "WORLD_SIZE": str(cluster_info.num_cluster_nodes),
                    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY: str(
                        cluster_info.num_processes_per_compute
                    ),
                },
                clear=False,
            ):
                server_args = ServerProcessArgs(
                    cluster_info=cluster_info,
                    task_config_uri=task_config_uri,
                    sample_edge_direction="in",
                    exception_dict=exception_dict,
                )
                server_process = ctx.Process(
                    target=_run_server_processes,
                    args=[server_args],
                    name=f"server_{i}",
                )
                server_process.start()
                launched_processes.append(server_process)

        self.assert_all_processes_succeed(launched_processes, exception_dict)

    def test_homogeneous_training(self):
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        task_config_uri = cora_supervised_info.frozen_gbml_config_uri
        (
            cluster_master_port,
            storage_cluster_master_port,
            compute_cluster_master_port,
            master_port,
            rpc_master_port,
            rpc_wait_port,
        ) = get_free_ports(num_ports=6)
        host_ip = socket.gethostbyname(socket.gethostname())
        cluster_info = GraphStoreInfo(
            num_storage_nodes=2,
            num_compute_nodes=2,
            num_processes_per_compute=2,
            cluster_master_ip=host_ip,
            storage_cluster_master_ip=host_ip,
            compute_cluster_master_ip=host_ip,
            cluster_master_port=cluster_master_port,
            storage_cluster_master_port=storage_cluster_master_port,
            compute_cluster_master_port=compute_cluster_master_port,
            rpc_master_port=rpc_master_port,
            rpc_wait_port=rpc_wait_port,
        )

        ctx = mp.get_context("spawn")
        launched_processes: list[py_mp_context.SpawnProcess] = []
        exception_dict = mp.Manager().dict()
        for i in range(cluster_info.num_compute_nodes):
            with mock.patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": host_ip,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(i),
                    "WORLD_SIZE": str(cluster_info.num_cluster_nodes),
                    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY: str(
                        cluster_info.num_processes_per_compute
                    ),
                },
                clear=False,
            ):
                client_train_args = ClientTrainProcessArgs(
                    client_rank=i,
                    cluster_info=cluster_info,
                    node_type=None,  # None for homogeneous dataset
                    exception_dict=exception_dict,
                )
                client_process = ctx.Process(
                    target=_client_train_process,
                    args=[client_train_args],
                    name=f"client_train_{i}",
                )
                client_process.start()
                launched_processes.append(client_process)
        # Start server process
        splitter = DistNodeAnchorLinkSplitter(
            sampling_direction="in",
            should_convert_labels_to_edges=True,
        )
        for i in range(cluster_info.num_storage_nodes):
            with mock.patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": host_ip,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(i + cluster_info.num_compute_nodes),
                    "WORLD_SIZE": str(cluster_info.num_cluster_nodes),
                    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY: str(
                        cluster_info.num_processes_per_compute
                    ),
                },
                clear=False,
            ):
                server_args = ServerProcessArgs(
                    cluster_info=cluster_info,
                    task_config_uri=task_config_uri,
                    sample_edge_direction="in",
                    exception_dict=exception_dict,
                    splitter=splitter,
                )
                server_process = ctx.Process(
                    target=_run_server_processes,
                    args=[server_args],
                )
                server_process.start()
                launched_processes.append(server_process)

        self.assert_all_processes_succeed(launched_processes, exception_dict)

    # TODO: (mkolodner-sc) - Figure out why this test is failing on Google Cloud Build
    @unittest.skip("Failing on Google Cloud Build - skiping for now")
    def test_graph_store_heterogeneous(self):
        # Simulating two server machine, two compute machines.
        # Each machine has one process.
        dblp_supervised_info = get_mocked_dataset_artifact_metadata()[
            DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        task_config_uri = dblp_supervised_info.frozen_gbml_config_uri
        (
            cluster_master_port,
            storage_cluster_master_port,
            compute_cluster_master_port,
            master_port,
            rpc_master_port,
            rpc_wait_port,
        ) = get_free_ports(num_ports=6)
        host_ip = socket.gethostbyname(socket.gethostname())
        cluster_info = GraphStoreInfo(
            num_storage_nodes=2,
            num_compute_nodes=2,
            num_processes_per_compute=2,
            cluster_master_ip=host_ip,
            storage_cluster_master_ip=host_ip,
            compute_cluster_master_ip=host_ip,
            cluster_master_port=cluster_master_port,
            storage_cluster_master_port=storage_cluster_master_port,
            compute_cluster_master_port=compute_cluster_master_port,
            rpc_master_port=rpc_master_port,
            rpc_wait_port=rpc_wait_port,
        )

        num_dblp_nodes = 4057
        expected_sampler_input = _get_expected_input_nodes_by_rank(
            num_dblp_nodes, cluster_info
        )
        expected_edge_types = [
            EdgeType(NodeType("author"), Relation("to"), NodeType("paper")),
            EdgeType(NodeType("paper"), Relation("to"), NodeType("author")),
            EdgeType(NodeType("term"), Relation("to"), NodeType("paper")),
        ]
        ctx = mp.get_context("spawn")
        manager = mp.Manager()
        exception_dict = manager.dict()
        launched_processes: list[py_mp_context.SpawnProcess] = []
        for i in range(cluster_info.num_compute_nodes):
            with mock.patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": host_ip,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(i),
                    "WORLD_SIZE": str(cluster_info.num_cluster_nodes),
                    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY: str(
                        cluster_info.num_processes_per_compute
                    ),
                },
                clear=False,
            ):
                client_args = ClientProcessArgs(
                    client_rank=i,
                    cluster_info=cluster_info,
                    node_type=NodeType("author"),
                    expected_sampler_input=expected_sampler_input,
                    expected_edge_types=expected_edge_types,
                    exception_dict=exception_dict,
                )
                client_process = ctx.Process(
                    target=_client_process,
                    args=[client_args],
                    name=f"client_{i}",
                )
                client_process.start()
                launched_processes.append(client_process)
        # Start server process
        for i in range(cluster_info.num_storage_nodes):
            with mock.patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": host_ip,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(i + cluster_info.num_compute_nodes),
                    "WORLD_SIZE": str(cluster_info.num_cluster_nodes),
                    COMPUTE_CLUSTER_LOCAL_WORLD_SIZE_ENV_KEY: str(
                        cluster_info.num_processes_per_compute
                    ),
                },
                clear=False,
            ):
                server_args = ServerProcessArgs(
                    cluster_info=cluster_info,
                    task_config_uri=task_config_uri,
                    sample_edge_direction="in",
                    exception_dict=exception_dict,
                )
                server_process = ctx.Process(
                    target=_run_server_processes,
                    args=[server_args],
                    name=f"server_{i}",
                )
                server_process.start()
                launched_processes.append(server_process)

        self.assert_all_processes_succeed(launched_processes, exception_dict)
