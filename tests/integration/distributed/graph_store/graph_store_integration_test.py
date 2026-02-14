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
from gigl.distributed.dist_ablp_neighborloader import DistABLPLoader
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
from gigl.utils.sampling import ABLPInputNodes
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
    ablp_result: dict[int, ABLPInputNodes],
) -> None:
    """Assert ABLP input structure and verify consistency across ranks on same compute node."""
    for i in range(cluster_info.compute_cluster_world_size):
        if i == torch.distributed.get_rank():
            logger.info(
                f"Verifying ABLP input for rank {i} / {cluster_info.compute_cluster_world_size}"
            )
            logger.info(f"--------------------------------")

            # Verify structure: dict mapping server_rank to ABLPInputNodes
            assert isinstance(
                ablp_result, dict
            ), f"Expected dict, got {type(ablp_result)}"
            assert (
                len(ablp_result) == cluster_info.num_storage_nodes
            ), f"Expected {cluster_info.num_storage_nodes} storage nodes in result, got {len(ablp_result)}"

            for server_rank, ablp_input in ablp_result.items():
                assert isinstance(
                    ablp_input, ABLPInputNodes
                ), f"Expected ABLPInputNodes, got {type(ablp_input)}"

                anchors = ablp_input.anchor_nodes
                # Verify anchors shape (1D tensor)
                assert isinstance(
                    anchors, torch.Tensor
                ), f"Anchors should be a tensor, got {type(anchors)}"
                assert anchors.dim() == 1, f"Anchors should be 1D, got {anchors.dim()}D"
                assert len(anchors) > 0, "Anchors should not be empty"

                # Verify labels: dict[EdgeType, tuple[Tensor, Optional[Tensor]]]
                assert isinstance(
                    ablp_input.labels, dict
                ), f"Labels should be a dict, got {type(ablp_input.labels)}"
                for edge_type, (positive_labels, negative_labels) in ablp_input.labels.items():
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

                _has_negatives = any(neg is not None for _, neg in ablp_input.labels.values())
                logger.info(
                    f"Server rank {server_rank}: anchor_node_type={ablp_input.anchor_node_type}, "
                    f"anchors shape={anchors.shape}, "
                    f"labels edge types={list(ablp_input.labels.keys())}, "
                    f"has_negatives={_has_negatives}"
                )

            logger.info(
                f"{i} / {cluster_info.compute_cluster_world_size} compute node rank ABLP input verified"
            )
        torch.distributed.barrier()

    torch.distributed.barrier()

    # Gather ABLP data from all ranks and verify processes on same compute_node_rank have identical data
    first_input = ablp_result[0]
    local_anchors = first_input.anchor_nodes
    # Get the first (and currently only) positive/negative label tensor for comparison
    first_pos, first_neg = next(iter(first_input.labels.values()))
    local_positive = first_pos
    local_negative = first_neg
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
        anchor_node_type=test_node_type,
        supervision_edge_type=supervision_edge_type,
    )

    _assert_ablp_input(cluster_info, ablp_result)

    ablp_loader = DistABLPLoader(
        dataset=remote_dist_dataset,
        num_neighbors=[2, 2],
        input_nodes=ablp_result,
        pin_memory_device=torch.device("cpu"),
        num_workers=2,
        worker_concurrency=2,
    )

    random_negative_input = remote_dist_dataset.get_node_ids(
        split="train",
        node_type=test_node_type,
        rank=cluster_info.compute_node_rank,
        world_size=cluster_info.num_compute_nodes,
    )

    # Test that two loaders can both be initialized and sampled from simultaneously.
    random_negative_loader = DistNeighborLoader(
        dataset=remote_dist_dataset,
        num_neighbors=[2, 2],
        input_nodes=random_negative_input,
        pin_memory_device=torch.device("cpu"),
        num_workers=2,
        worker_concurrency=2,
    )
    count = 0
    for i, (ablp_batch, random_negative_batch) in enumerate(
        zip(ablp_loader, random_negative_loader)
    ):
        # Verify batch structure
        assert hasattr(ablp_batch, "y_positive"), "Batch should have y_positive labels"
        # y_positive should be dict mapping local anchor idx -> local label indices
        assert isinstance(
            ablp_batch.y_positive, dict
        ), f"y_positive should be dict, got {type(ablp_batch.y_positive)}"
        count += 1

    torch.distributed.barrier()
    logger.info(f"Rank {torch.distributed.get_rank()} loaded {count} ABLP batches")

    # Verify total count across all ranks
    count_tensor = torch.tensor(count, dtype=torch.int64)
    torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)

    # Calculate expected total anchors by summing across all compute nodes
    # Each process on the same compute node has the same anchor count, so we sum
    # across all processes and divide by num_processes_per_compute to get the true total
    local_total_anchors = sum(
        ablp_result[server_rank].anchor_nodes.shape[0] for server_rank in ablp_result
    )
    expected_anchors_tensor = torch.tensor(local_total_anchors, dtype=torch.int64)
    torch.distributed.all_reduce(
        expected_anchors_tensor, op=torch.distributed.ReduceOp.SUM
    )
    expected_batches = (
        expected_anchors_tensor.item() // cluster_info.num_processes_per_compute
    )
    assert (
        count_tensor.item() == expected_batches
    ), f"Expected {expected_batches} total batches, got {count_tensor.item()}"

    shutdown_compute_proccess()


def _run_compute_multiple_loaders_test(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    mp_sharing_dict: Optional[MutableMapping[str, torch.Tensor]],
    node_type: Optional[NodeType],
) -> None:
    """
    Compute test that validates multiple loader instances can coexist.

    Phase 1: Creates two ABLP loaders + two DistNeighborLoaders and iterates them in parallel.
    Phase 2: After shutting down phase 1 loaders (to free server-side producers and RPC
             resources), creates one more ABLP + one DistNeighborLoader pair sequentially.
    """
    init_compute_process(client_rank, cluster_info, compute_world_backend="gloo")

    remote_dist_dataset = RemoteDistDataset(
        cluster_info=cluster_info,
        local_rank=client_rank,
        mp_sharing_dict=mp_sharing_dict,
    )

    test_node_type = (
        node_type if node_type is not None else DEFAULT_HOMOGENEOUS_NODE_TYPE
    )
    supervision_edge_type = DEFAULT_HOMOGENEOUS_EDGE_TYPE

    ablp_result = remote_dist_dataset.get_ablp_input(
        split="train",
        rank=cluster_info.compute_node_rank,
        world_size=cluster_info.num_compute_nodes,
        anchor_node_type=test_node_type,
        supervision_edge_type=supervision_edge_type,
    )

    random_negative_input = remote_dist_dataset.get_node_ids(
        split="train",
        node_type=test_node_type,
        rank=cluster_info.compute_node_rank,
        world_size=cluster_info.num_compute_nodes,
    )

    # Calculate expected batch count (same logic as _run_compute_train_tests).
    local_total_anchors = sum(
        ablp_result[server_rank].anchor_nodes.shape[0] for server_rank in ablp_result
    )
    expected_anchors_tensor = torch.tensor(local_total_anchors, dtype=torch.int64)
    torch.distributed.all_reduce(
        expected_anchors_tensor, op=torch.distributed.ReduceOp.SUM
    )
    total_negative_seeds = sum(
        random_negative_input[server_rank].shape[0]
        for server_rank in random_negative_input
    )
    total_negative_seeds_tensor = torch.tensor(total_negative_seeds, dtype=torch.int64)
    torch.distributed.all_reduce(
        total_negative_seeds_tensor, op=torch.distributed.ReduceOp.SUM
    )
    total_negative_seeds = int(total_negative_seeds_tensor.item())
    expected_batches = int(
        expected_anchors_tensor.item() // cluster_info.num_processes_per_compute
    )
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} expected batches: {expected_batches}, total negative seeds: {total_negative_seeds}"
    )

    # ------------------------------------------------------------------
    # Phase 1: Two ABLP loaders + two DistNeighborLoaders in parallel
    # ------------------------------------------------------------------
    # Use prefetch_size=2 to limit concurrent fetch_one_sampled_message RPC calls
    # per server. With 4 loaders × 2 compute nodes × 2 prefetch = 16 calls,
    # matching the 16 RPC thread limit on the server.
    ablp_loader_1 = DistABLPLoader(
        dataset=remote_dist_dataset,
        num_neighbors=[2, 2],
        input_nodes=ablp_result,
        pin_memory_device=torch.device("cpu"),
        num_workers=2,
        worker_concurrency=2,
        prefetch_size=2,
    )
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} ablp_loader_1 producers: ({ablp_loader_1._producer_id_list})"
    )
    ablp_loader_2 = DistABLPLoader(
        dataset=remote_dist_dataset,
        num_neighbors=[2, 2],
        input_nodes=ablp_result,
        pin_memory_device=torch.device("cpu"),
        num_workers=2,
        worker_concurrency=2,
        prefetch_size=2,
    )
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} ablp_loader_2 producers: ({ablp_loader_2._producer_id_list})"
    )
    neighbor_loader_1 = DistNeighborLoader(
        dataset=remote_dist_dataset,
        num_neighbors=[2, 2],
        input_nodes=random_negative_input,
        pin_memory_device=torch.device("cpu"),
        num_workers=2,
        worker_concurrency=2,
    )
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} neighbor_loader_1 producers: ({neighbor_loader_1._producer_id_list})"
    )
    neighbor_loader_2 = DistNeighborLoader(
        dataset=remote_dist_dataset,
        num_neighbors=[2, 2],
        input_nodes=random_negative_input,
        pin_memory_device=torch.device("cpu"),
        num_workers=2,
        worker_concurrency=2,
    )
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} neighbor_loader_2 producers: ({neighbor_loader_2._producer_id_list})"
    )
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} phase 1: loading batches from 4 parallel loaders"
    )
    torch.distributed.barrier()
    phase1_count = 0
    for ablp_batch_1, ablp_batch_2, neg_batch_1, neg_batch_2 in zip(
        ablp_loader_1, ablp_loader_2, neighbor_loader_1, neighbor_loader_2
    ):
        assert hasattr(
            ablp_batch_1, "y_positive"
        ), "ABLP batch 1 should have y_positive"
        assert hasattr(
            ablp_batch_2, "y_positive"
        ), "ABLP batch 2 should have y_positive"
        phase1_count += 1
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} phase 1: loaded {phase1_count} batches from 4 parallel loaders"
    )
    torch.distributed.barrier()
    logger.info("All ranks have loaded phase 1 batches")

    phase1_count_tensor = torch.tensor(phase1_count, dtype=torch.int64)
    torch.distributed.all_reduce(phase1_count_tensor, op=torch.distributed.ReduceOp.SUM)
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} expected batches: {expected_batches}, total negative seeds: {total_negative_seeds}"
    )

    assert (
        phase1_count_tensor.item() == expected_batches
    ), f"Phase 1: Expected {expected_batches} total batches, got {phase1_count_tensor.item()}"

    # Shut down phase 1 loaders to free server-side producers and RPC resources
    # before creating new loaders. This mirrors GLT's DistLoader.shutdown() which
    # calls DistServer.destroy_sampling_producer for each remote producer.
    ablp_loader_1.shutdown()
    ablp_loader_2.shutdown()
    neighbor_loader_1.shutdown()
    neighbor_loader_2.shutdown()
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} shut down phase 1 loaders"
    )
    torch.distributed.barrier()

    # ------------------------------------------------------------------
    # Phase 2: One more ABLP + one more DistNeighborLoader (sequential)
    # ------------------------------------------------------------------
    ablp_loader_3 = DistABLPLoader(
        dataset=remote_dist_dataset,
        num_neighbors=[2, 2],
        input_nodes=ablp_result,
        pin_memory_device=torch.device("cpu"),
        num_workers=2,
        worker_concurrency=2,
    )
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} ablp_loader_3 producers: ({ablp_loader_3._producer_id_list})"
    )
    neighbor_loader_3 = DistNeighborLoader(
        dataset=remote_dist_dataset,
        num_neighbors=[2, 2],
        input_nodes=random_negative_input,
        pin_memory_device=torch.device("cpu"),
        num_workers=2,
        worker_concurrency=2,
    )
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} neighbor_loader_3 producers: ({neighbor_loader_3._producer_id_list})"
    )
    phase2_count = 0
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} phase 2: loading batches from 2 sequential loaders"
    )
    for ablp_batch_3, neg_batch_3 in zip(ablp_loader_3, neighbor_loader_3):
        assert hasattr(
            ablp_batch_3, "y_positive"
        ), "ABLP batch 3 should have y_positive"
        phase2_count += 1

    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} phase 2: loaded {phase2_count} batches from 2 sequential loaders"
    )
    torch.distributed.barrier()

    phase2_count_tensor = torch.tensor(phase2_count, dtype=torch.int64)
    torch.distributed.all_reduce(phase2_count_tensor, op=torch.distributed.ReduceOp.SUM)
    logger.info(
        f"Rank {torch.distributed.get_rank()} / {torch.distributed.get_world_size()} phase 2: loaded {phase2_count_tensor.item()} batches from 2 sequential loaders"
    )
    assert (
        phase2_count_tensor.item() == expected_batches
    ), f"Phase 2: Expected {expected_batches} total batches, got {phase2_count_tensor.item()}"

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


def _client_multiple_loaders_process(args: ClientTrainProcessArgs) -> None:
    """Client process for testing multiple loader instances in parallel and sequence."""
    logger.info(
        f"Initializing multiple loaders client node {args.client_rank} / {args.cluster_info.num_compute_nodes}. "
        f"OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']}"
    )
    process_name = f"client_multiple_loaders_{args.client_rank}"
    try:
        mp_context = torch.multiprocessing.get_context("spawn")
        mp_sharing_dict = torch.multiprocessing.Manager().dict()
        client_processes: list[py_mp_context.SpawnProcess] = []
        logger.info("Starting multiple loaders client processes")
        for i in range(args.cluster_info.num_processes_per_compute):
            client_process = mp_context.Process(
                target=_run_compute_multiple_loaders_test,
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
    """
    NOTE: Since these tests run on cloud build,
    and our python process memory footprint is quite large due to tf, torch, etc,
    We need to be careful to not spawn too many processes.
    Otherwise we will OOM and see "myterious" failures like the below:
    make: *** [Makefile:119: integration_test] Error 137
    ERROR: build step 0 "docker-img/path:tag" failed: step exited with non-zero status: 2
    ERROR: build step 0 "docker-img/path:tag" failed: step exited with non-zero status: 2
    """

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
            num_processes_per_compute=1,
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

    @unittest.skip("Not supported yet - skipping for now")
    def test_multiple_loaders_in_graph_store(self):
        """Test that multiple loader instances (2 ABLP + 2 DistNeighborLoader) can work
        in parallel, followed by another (ABLP, DistNeighborLoader) pair sequentially.
        """
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
        # Very small cluster to avoid OOMing on CICD.
        cluster_info = GraphStoreInfo(
            num_storage_nodes=1,
            num_compute_nodes=1,
            num_processes_per_compute=1,
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
                    node_type=None,
                    exception_dict=exception_dict,
                )
                client_process = ctx.Process(
                    target=_client_multiple_loaders_process,
                    args=[client_train_args],
                    name=f"client_multiple_loaders_{i}",
                )
                client_process.start()
                launched_processes.append(client_process)

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
                    name=f"server_{i}",
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
