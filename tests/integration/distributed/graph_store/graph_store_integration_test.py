import multiprocessing.context as py_mp_context
import os
import socket
import tempfile
import traceback
import unittest
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Literal, Optional, Union
from unittest import mock

import torch
import torch.multiprocessing as mp
from torch_geometric.data import Data, HeteroData

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.distributed.dist_ablp_neighborloader import DistABLPLoader
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.graph_store.compute import (
    init_compute_process,
    shutdown_compute_proccess,
)
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.graph_store.sharding import compute_server_assignments
from gigl.distributed.graph_store.storage_utils import (
    build_storage_dataset,
    run_storage_server,
)
from gigl.distributed.utils.networking import get_free_port, get_free_ports
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
from gigl.utils.data_splitters import DistNodeAnchorLinkSplitter, DistNodeSplitter
from gigl.utils.sampling import ABLPInputNodes
from tests.test_assets.distributed.utils import assert_tensor_equality
from tests.test_assets.test_case import DEFAULT_TIMEOUT_SECONDS, TestCase

logger = Logger()
TEST_BATCH_SIZE = 128
TEST_NUM_NEIGHBORS = [2, 2]
TEST_PIN_MEMORY_DEVICE = torch.device("cpu")
TEST_NUM_WORKERS = 2
TEST_WORKER_CONCURRENCY = 2


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def _to_long_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to long dtype on CPU."""
    return tensor.detach().cpu().to(dtype=torch.long)


def _concat_seed_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate tensors, normalizing to long CPU tensors and skipping empties."""
    non_empty = [_to_long_cpu(t) for t in tensors if t.numel() > 0]
    return (
        torch.cat(non_empty, dim=0) if non_empty else torch.empty(0, dtype=torch.long)
    )


def _sorted_seed_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Sort a tensor after normalizing to long CPU."""
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.sort(_to_long_cpu(tensor)).values


def _get_batch_seed_tensor(
    datum: Union[Data, HeteroData], node_type: Optional[NodeType]
) -> torch.Tensor:
    """Extract the batch seed tensor from a Data or HeteroData object."""
    if node_type is not None:
        assert isinstance(datum, HeteroData)
        batch = datum[node_type].batch
    else:
        assert isinstance(datum, Data)
        batch = datum.batch
    assert isinstance(batch, torch.Tensor), (
        f"Expected tensor batch field, got {type(batch)}"
    )
    return _to_long_cpu(batch)


# ---------------------------------------------------------------------------
# Distributed seed coverage assertion
# ---------------------------------------------------------------------------


def _assert_global_seed_coverage(
    *,
    name: str,
    cluster_info: GraphStoreInfo,
    local_seen: torch.Tensor,
    local_expected: torch.Tensor,
) -> None:
    """Assert that all expected seeds are covered globally across ranks.

    Gathers seen and expected seeds from all ranks and verifies full coverage.
    """
    # Gather seen seeds from all ranks
    all_seen: list[torch.Tensor] = [None] * cluster_info.compute_cluster_world_size  # type: ignore[list-item]  # ty: ignore[invalid-assignment] TODO(ty-torch-container-shapes): fix ty false positives for torch container and return shapes.
    torch.distributed.all_gather_object(all_seen, _to_long_cpu(local_seen))
    globally_seen = _sorted_seed_tensor(_concat_seed_tensors(all_seen))

    # Gather expected seeds from all ranks. In graph-store mode, input sharding is
    # per compute process, not per compute node.
    all_expected: list[torch.Tensor] = [None] * cluster_info.compute_cluster_world_size  # type: ignore[list-item]  # ty: ignore[invalid-assignment] TODO(ty-torch-container-shapes): fix ty false positives for torch container and return shapes.
    torch.distributed.all_gather_object(all_expected, _to_long_cpu(local_expected))
    globally_expected = _sorted_seed_tensor(_concat_seed_tensors(all_expected))

    assert_tensor_equality(globally_seen, globally_expected)
    logger.info(
        f"Rank {torch.distributed.get_rank()} verified {name} coverage for "
        f"{globally_seen.numel()} seeds"
    )


# ---------------------------------------------------------------------------
# Loader builders
# ---------------------------------------------------------------------------


def _build_ablp_loader(
    remote_dist_dataset: RemoteDistDataset,
    ablp_input: dict[int, ABLPInputNodes],
    *,
    prefetch_size: Optional[int] = None,
) -> DistABLPLoader:
    """Build a DistABLPLoader with standard test parameters."""
    return DistABLPLoader(
        dataset=remote_dist_dataset,
        num_neighbors=TEST_NUM_NEIGHBORS,
        input_nodes=ablp_input,
        pin_memory_device=TEST_PIN_MEMORY_DEVICE,
        num_workers=TEST_NUM_WORKERS,
        worker_concurrency=TEST_WORKER_CONCURRENCY,
        batch_size=TEST_BATCH_SIZE,
        prefetch_size=prefetch_size,
    )


def _build_neighbor_loader(
    remote_dist_dataset: RemoteDistDataset,
    sampler_input: dict[int, torch.Tensor],
    *,
    prefetch_size: Optional[int] = None,
) -> DistNeighborLoader:
    """Build a DistNeighborLoader with standard test parameters."""
    return DistNeighborLoader(
        dataset=remote_dist_dataset,
        num_neighbors=TEST_NUM_NEIGHBORS,
        input_nodes=sampler_input,
        pin_memory_device=TEST_PIN_MEMORY_DEVICE,
        num_workers=TEST_NUM_WORKERS,
        worker_concurrency=TEST_WORKER_CONCURRENCY,
        batch_size=TEST_BATCH_SIZE,
        prefetch_size=prefetch_size,
    )


# ---------------------------------------------------------------------------
# Input assertions
# ---------------------------------------------------------------------------


def _assert_sampler_input(
    cluster_info: GraphStoreInfo,
    sampler_input: dict[int, torch.Tensor],
    expected_sampler_input: dict[int, list[torch.Tensor]],
) -> None:
    rank_expected_sampler_input = expected_sampler_input[torch.distributed.get_rank()]
    assert len(sampler_input) == len(rank_expected_sampler_input)
    for server_rank, expected in enumerate(rank_expected_sampler_input):
        assert_tensor_equality(sampler_input[server_rank], expected)


def _assert_ablp_input(
    cluster_info: GraphStoreInfo,
    ablp_result: dict[int, ABLPInputNodes],
) -> None:
    """Assert the structure of the fetched ABLP input for the current rank."""
    assert isinstance(ablp_result, dict), f"Expected dict, got {type(ablp_result)}"
    assert len(ablp_result) == cluster_info.num_storage_nodes, (
        f"Expected {cluster_info.num_storage_nodes} storage nodes in result, got {len(ablp_result)}"
    )

    for server_rank, ablp_input in ablp_result.items():
        assert isinstance(ablp_input, ABLPInputNodes), (
            f"Expected ABLPInputNodes, got {type(ablp_input)}"
        )

        anchors = ablp_input.anchor_nodes
        assert isinstance(anchors, torch.Tensor), (
            f"Anchors should be a tensor, got {type(anchors)}"
        )
        assert anchors.dim() == 1, f"Anchors should be 1D, got {anchors.dim()}D"

        assert isinstance(ablp_input.labels, dict), (
            f"Labels should be a dict, got {type(ablp_input.labels)}"
        )
        for edge_type, (
            positive_labels,
            negative_labels,
        ) in ablp_input.labels.items():
            assert isinstance(positive_labels, torch.Tensor), (
                f"Positive labels should be a tensor, got {type(positive_labels)}"
            )
            assert positive_labels.dim() == 2, (
                f"Positive labels should be 2D, got {positive_labels.dim()}D"
            )
            assert positive_labels.shape[0] == len(anchors), (
                f"Positive labels first dim should match anchors length, got {positive_labels.shape[0]} vs {len(anchors)}"
            )

            if negative_labels is not None:
                assert isinstance(negative_labels, torch.Tensor), (
                    f"Negative labels should be a tensor, got {type(negative_labels)}"
                )
                assert negative_labels.dim() == 2, (
                    f"Negative labels should be 2D, got {negative_labels.dim()}D"
                )
                assert negative_labels.shape[0] == len(anchors), (
                    f"Negative labels first dim should match anchors length"
                )

        has_negatives = any(neg is not None for _, neg in ablp_input.labels.values())
        logger.info(
            f"Server rank {server_rank}: anchor_node_type={ablp_input.anchor_node_type}, "
            f"anchors shape={anchors.shape}, labels edge types={list(ablp_input.labels.keys())}, "
            f"has_negatives={has_negatives}"
        )


# ---------------------------------------------------------------------------
# Compute test targets (run inside spawned subprocesses)
# ---------------------------------------------------------------------------


def _run_compute_train_tests(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    node_type: Optional[NodeType],
) -> None:
    """Compute test for training mode that verifies ABLP input and seed coverage."""
    init_compute_process(client_rank, cluster_info, compute_world_backend="gloo")

    remote_dist_dataset = RemoteDistDataset(
        cluster_info=cluster_info,
        local_rank=client_rank,
    )

    ablp_result = remote_dist_dataset.fetch_ablp_input(
        split="train",
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )
    _assert_ablp_input(cluster_info, ablp_result)

    ablp_loader = _build_ablp_loader(remote_dist_dataset, ablp_result)
    random_negative_input = remote_dist_dataset.fetch_node_ids(
        split="train",
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )
    random_negative_loader = _build_neighbor_loader(
        remote_dist_dataset, random_negative_input
    )

    ablp_batches: list[torch.Tensor] = []
    random_negative_batches: list[torch.Tensor] = []
    for ablp_batch, random_negative_batch in zip_longest(
        ablp_loader, random_negative_loader
    ):
        assert ablp_batch is not None, (
            "ABLP loader exhausted before random negative loader"
        )
        assert random_negative_batch is not None, (
            "Random negative loader exhausted before ABLP loader"
        )
        assert hasattr(ablp_batch, "y_positive"), "Batch should have y_positive labels"
        assert isinstance(ablp_batch.y_positive, dict), (
            f"y_positive should be dict, got {type(ablp_batch.y_positive)}"
        )
        if node_type is not None:
            assert isinstance(ablp_batch, HeteroData)
            assert isinstance(random_negative_batch, HeteroData)
        else:
            assert isinstance(ablp_batch, Data)
            assert isinstance(random_negative_batch, Data)
        ablp_batches.append(_get_batch_seed_tensor(ablp_batch, node_type))
        random_negative_batches.append(
            _get_batch_seed_tensor(random_negative_batch, node_type)
        )

    local_expected_anchors = _concat_seed_tensors(
        [ablp_result[r].anchor_nodes for r in sorted(ablp_result)]
    )
    local_expected_negative_seeds = _concat_seed_tensors(
        [random_negative_input[r] for r in sorted(random_negative_input)]
    )
    _assert_global_seed_coverage(
        name="train ABLP loader",
        cluster_info=cluster_info,
        local_seen=_concat_seed_tensors(ablp_batches),
        local_expected=local_expected_anchors,
    )
    _assert_global_seed_coverage(
        name="train random negative loader",
        cluster_info=cluster_info,
        local_seen=_concat_seed_tensors(random_negative_batches),
        local_expected=local_expected_negative_seeds,
    )

    shutdown_compute_proccess()


def _run_compute_multiple_loaders_test(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    node_type: Optional[NodeType],
) -> None:
    """Compute test that validates multiple loader instances can coexist.

    Phase 1: Creates two ABLP loaders + two DistNeighborLoaders and iterates them
    in parallel.
    Phase 2: After shutting down phase 1 loaders (to free server-side producers and
    RPC resources), creates one more ABLP + one DistNeighborLoader pair sequentially.
    """
    init_compute_process(client_rank, cluster_info, compute_world_backend="gloo")

    remote_dist_dataset = RemoteDistDataset(
        cluster_info=cluster_info,
        local_rank=client_rank,
    )

    ablp_result = remote_dist_dataset.fetch_ablp_input(
        split="train",
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )
    random_negative_input = remote_dist_dataset.fetch_node_ids(
        split="train",
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )

    local_expected_anchors = _concat_seed_tensors(
        [ablp_result[r].anchor_nodes for r in sorted(ablp_result)]
    )
    local_expected_negative_seeds = _concat_seed_tensors(
        [random_negative_input[r] for r in sorted(random_negative_input)]
    )

    # ------------------------------------------------------------------
    # Phase 1: Two ABLP loaders + two DistNeighborLoaders in parallel
    # ------------------------------------------------------------------
    # Use prefetch_size=2 to limit concurrent fetch_one_sampled_message RPC calls
    # per server. With 4 loaders × 2 compute nodes × 2 prefetch = 16 calls,
    # matching the 16 RPC thread limit on the server.
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    ablp_loader_1 = _build_ablp_loader(
        remote_dist_dataset, ablp_result, prefetch_size=2
    )
    logger.info(
        f"Rank {rank} / {world_size} ablp_loader_1 backends/channels: "
        f"({ablp_loader_1._backend_id_list}, {ablp_loader_1._channel_id_list})"
    )
    ablp_loader_2 = _build_ablp_loader(
        remote_dist_dataset, ablp_result, prefetch_size=2
    )
    logger.info(
        f"Rank {rank} / {world_size} ablp_loader_2 backends/channels: "
        f"({ablp_loader_2._backend_id_list}, {ablp_loader_2._channel_id_list})"
    )
    neighbor_loader_1 = _build_neighbor_loader(
        remote_dist_dataset, random_negative_input
    )
    logger.info(
        f"Rank {rank} / {world_size} neighbor_loader_1 backends/channels: "
        f"({neighbor_loader_1._backend_id_list}, {neighbor_loader_1._channel_id_list})"
    )
    neighbor_loader_2 = _build_neighbor_loader(
        remote_dist_dataset, random_negative_input
    )
    logger.info(
        f"Rank {rank} / {world_size} neighbor_loader_2 backends/channels: "
        f"({neighbor_loader_2._backend_id_list}, {neighbor_loader_2._channel_id_list})"
    )
    gathered_ablp_loader_1_backends = [None] * world_size
    torch.distributed.all_gather_object(
        gathered_ablp_loader_1_backends, tuple(ablp_loader_1._backend_id_list)
    )
    assert all(
        backend_ids == gathered_ablp_loader_1_backends[0]
        for backend_ids in gathered_ablp_loader_1_backends
    ), "All ranks should share the same backend ids for one logical loader."
    gathered_neighbor_loader_1_backends = [None] * world_size
    torch.distributed.all_gather_object(
        gathered_neighbor_loader_1_backends, tuple(neighbor_loader_1._backend_id_list)
    )
    assert all(
        backend_ids == gathered_neighbor_loader_1_backends[0]
        for backend_ids in gathered_neighbor_loader_1_backends
    ), "All ranks should share the same backend ids for one logical loader."
    assert ablp_loader_1._backend_id_list != ablp_loader_2._backend_id_list, (
        "Concurrent ABLP loaders must use distinct backends."
    )
    assert neighbor_loader_1._backend_id_list != neighbor_loader_2._backend_id_list, (
        "Concurrent neighbor loaders must use distinct backends."
    )
    assert ablp_loader_1._backend_id_list != neighbor_loader_1._backend_id_list, (
        "ABLP and neighbor loaders must not share a backend."
    )
    logger.info(
        f"Rank {rank} / {world_size} phase 1: loading batches from 4 parallel loaders"
    )
    phase1_ablp_loader_1_batches: list[torch.Tensor] = []
    phase1_ablp_loader_2_batches: list[torch.Tensor] = []
    phase1_neighbor_loader_1_batches: list[torch.Tensor] = []
    phase1_neighbor_loader_2_batches: list[torch.Tensor] = []
    for ablp_batch_1, ablp_batch_2, neg_batch_1, neg_batch_2 in zip_longest(
        ablp_loader_1, ablp_loader_2, neighbor_loader_1, neighbor_loader_2
    ):
        assert ablp_batch_1 is not None, "ABLP loader 1 exhausted early in phase 1"
        assert ablp_batch_2 is not None, "ABLP loader 2 exhausted early in phase 1"
        assert neg_batch_1 is not None, "Neighbor loader 1 exhausted early in phase 1"
        assert neg_batch_2 is not None, "Neighbor loader 2 exhausted early in phase 1"
        assert hasattr(ablp_batch_1, "y_positive"), (
            "ABLP batch 1 should have y_positive"
        )
        assert hasattr(ablp_batch_2, "y_positive"), (
            "ABLP batch 2 should have y_positive"
        )
        phase1_ablp_loader_1_batches.append(
            _get_batch_seed_tensor(ablp_batch_1, node_type)
        )
        phase1_ablp_loader_2_batches.append(
            _get_batch_seed_tensor(ablp_batch_2, node_type)
        )
        phase1_neighbor_loader_1_batches.append(
            _get_batch_seed_tensor(neg_batch_1, node_type)
        )
        phase1_neighbor_loader_2_batches.append(
            _get_batch_seed_tensor(neg_batch_2, node_type)
        )
    logger.info("All ranks have loaded phase 1 batches")
    _assert_global_seed_coverage(
        name="phase 1 ABLP loader 1",
        cluster_info=cluster_info,
        local_seen=_concat_seed_tensors(phase1_ablp_loader_1_batches),
        local_expected=local_expected_anchors,
    )
    _assert_global_seed_coverage(
        name="phase 1 ABLP loader 2",
        cluster_info=cluster_info,
        local_seen=_concat_seed_tensors(phase1_ablp_loader_2_batches),
        local_expected=local_expected_anchors,
    )
    _assert_global_seed_coverage(
        name="phase 1 neighbor loader 1",
        cluster_info=cluster_info,
        local_seen=_concat_seed_tensors(phase1_neighbor_loader_1_batches),
        local_expected=local_expected_negative_seeds,
    )
    _assert_global_seed_coverage(
        name="phase 1 neighbor loader 2",
        cluster_info=cluster_info,
        local_seen=_concat_seed_tensors(phase1_neighbor_loader_2_batches),
        local_expected=local_expected_negative_seeds,
    )

    # Shut down phase 1 loaders to free server-side channels and backend resources
    # before creating new loaders.
    ablp_loader_1.shutdown()
    ablp_loader_2.shutdown()
    neighbor_loader_1.shutdown()
    neighbor_loader_2.shutdown()
    logger.info(f"Rank {rank} / {world_size} shut down phase 1 loaders")
    torch.distributed.barrier()

    # ------------------------------------------------------------------
    # Phase 2: One more ABLP + one more DistNeighborLoader (sequential)
    # ------------------------------------------------------------------
    ablp_loader_3 = _build_ablp_loader(remote_dist_dataset, ablp_result)
    logger.info(
        f"Rank {rank} / {world_size} ablp_loader_3 backends/channels: "
        f"({ablp_loader_3._backend_id_list}, {ablp_loader_3._channel_id_list})"
    )
    neighbor_loader_3 = _build_neighbor_loader(
        remote_dist_dataset, random_negative_input
    )
    logger.info(
        f"Rank {rank} / {world_size} neighbor_loader_3 backends/channels: "
        f"({neighbor_loader_3._backend_id_list}, {neighbor_loader_3._channel_id_list})"
    )
    logger.info(
        f"Rank {rank} / {world_size} phase 2: loading batches from 2 sequential loaders"
    )
    phase2_ablp_loader_3_batches: list[torch.Tensor] = []
    phase2_neighbor_loader_3_batches: list[torch.Tensor] = []
    for ablp_batch_3, neg_batch_3 in zip_longest(ablp_loader_3, neighbor_loader_3):
        assert ablp_batch_3 is not None, "ABLP loader 3 exhausted early in phase 2"
        assert neg_batch_3 is not None, "Neighbor loader 3 exhausted early in phase 2"
        assert hasattr(ablp_batch_3, "y_positive"), (
            "ABLP batch 3 should have y_positive"
        )
        phase2_ablp_loader_3_batches.append(
            _get_batch_seed_tensor(ablp_batch_3, node_type)
        )
        phase2_neighbor_loader_3_batches.append(
            _get_batch_seed_tensor(neg_batch_3, node_type)
        )

    logger.info(
        f"Rank {rank} / {world_size} phase 2: loaded batches from 2 sequential loaders"
    )
    _assert_global_seed_coverage(
        name="phase 2 ABLP loader 3",
        cluster_info=cluster_info,
        local_seen=_concat_seed_tensors(phase2_ablp_loader_3_batches),
        local_expected=local_expected_anchors,
    )
    _assert_global_seed_coverage(
        name="phase 2 neighbor loader 3",
        cluster_info=cluster_info,
        local_seen=_concat_seed_tensors(phase2_neighbor_loader_3_batches),
        local_expected=local_expected_negative_seeds,
    )
    ablp_loader_3.shutdown()
    neighbor_loader_3.shutdown()
    torch.distributed.barrier()

    shutdown_compute_proccess()


def _run_compute_tests(
    client_rank: int,
    cluster_info: GraphStoreInfo,
    node_type: Optional[NodeType],
    expected_sampler_input: dict[int, list[torch.Tensor]],
    expected_edge_types: Optional[list[EdgeType]],
) -> None:
    """Process target for "compute" nodes.

    Each "Client Process" (e.g. cluster_info.num_compute_nodes) will spawn as a
    process for each "num_processes_per_compute".
    """
    init_compute_process(client_rank, cluster_info, compute_world_backend="gloo")

    remote_dist_dataset = RemoteDistDataset(
        cluster_info=cluster_info,
        local_rank=client_rank,
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert remote_dist_dataset.fetch_edge_dir() == "in", (
        f"Edge direction must be 'in' for the test dataset. Got {remote_dist_dataset.fetch_edge_dir()}"
    )
    assert remote_dist_dataset.fetch_edge_feature_info() is not None, (
        "Edge feature info must not be None for the test dataset"
    )
    assert remote_dist_dataset.fetch_node_feature_info() is not None, (
        "Node feature info must not be None for the test dataset"
    )
    ports = remote_dist_dataset.fetch_free_ports_on_storage_cluster(num_ports=2)
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
            assert received_ports == ports, (
                f"Expected {ports} free ports, got {received_ports}"
            )

    torch.distributed.barrier()
    logger.info("Verified that all ranks received the same free ports")

    sampler_input = remote_dist_dataset.fetch_node_ids(
        node_type=node_type,
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )
    _assert_sampler_input(cluster_info, sampler_input, expected_sampler_input)

    assert remote_dist_dataset.fetch_edge_types() == expected_edge_types, (
        f"Expected edge types {expected_edge_types}, got {remote_dist_dataset.fetch_edge_types()}"
    )

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
        num_neighbors=TEST_NUM_NEIGHBORS,
        pin_memory_device=TEST_PIN_MEMORY_DEVICE,
        input_nodes=input_nodes,
        num_workers=TEST_NUM_WORKERS,
        worker_concurrency=TEST_WORKER_CONCURRENCY,
        batch_size=TEST_BATCH_SIZE,
    )
    loaded_batches: list[torch.Tensor] = []
    for datum in loader:
        if node_type is not None:
            assert isinstance(datum, HeteroData)
        else:
            assert isinstance(datum, Data)
        loaded_batches.append(_get_batch_seed_tensor(datum, node_type))

    _assert_global_seed_coverage(
        name="graph store neighbor loader",
        cluster_info=cluster_info,
        local_seen=_concat_seed_tensors(loaded_batches),
        local_expected=_concat_seed_tensors(
            expected_sampler_input[torch.distributed.get_rank()]
        ),
    )
    loader.shutdown()
    shutdown_compute_proccess()


# ---------------------------------------------------------------------------
# Client / server process wrappers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientProcessArgs:
    """Arguments for the client compute process.

    Attributes:
        client_rank: Rank of this client in the compute cluster.
        cluster_info: Information about the distributed cluster.
        node_type: Type of nodes to process, None for homogeneous graphs.
        exception_dict: Shared dictionary for storing exceptions from processes.
        compute_target: The function each subprocess runs (e.g. _run_compute_tests).
        compute_target_extra_args: Extra positional args appended after the common
            (client_rank, cluster_info, node_type) args.
    """

    client_rank: int
    cluster_info: GraphStoreInfo
    node_type: Optional[NodeType]
    exception_dict: MutableMapping[str, str]
    compute_target: Callable[..., None]
    compute_target_extra_args: tuple[Any, ...] = ()


def _client_compute_process(args: ClientProcessArgs) -> None:
    """Client process that spawns per-GPU compute subprocesses."""
    process_name = f"client_{args.client_rank}"
    try:
        logger.info(
            f"Initializing client node {args.client_rank} / "
            f"{args.cluster_info.num_compute_nodes}. "
            f"OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']}"
        )
        mp_context = torch.multiprocessing.get_context("spawn")
        client_processes: list[py_mp_context.SpawnProcess] = []
        for i in range(args.cluster_info.num_processes_per_compute):
            client_process = mp_context.Process(
                target=args.compute_target,
                args=[
                    i,
                    args.cluster_info,
                    args.node_type,
                    *args.compute_target_extra_args,
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
        num_server_sessions: Number of sequential server sessions to run
            (e.g. one per inference node type).
        splitter: Optional splitter for node anchor link or node splitting.
    """

    cluster_info: GraphStoreInfo
    task_config_uri: Uri
    sample_edge_direction: Literal["in", "out"]
    exception_dict: MutableMapping[str, str]
    num_server_sessions: int = 1
    splitter: Optional[Union[DistNodeAnchorLinkSplitter, DistNodeSplitter]] = None


def _run_storage_main_process(args: ServerProcessArgs) -> None:
    process_name = f"server_{args.cluster_info.storage_node_rank}"
    try:
        storage_rank = args.cluster_info.storage_node_rank
        cluster_info = args.cluster_info
        logger.info(
            f"Initializing server processes. OS rank: {os.environ['RANK']}, "
            f"OS world size: {os.environ['WORLD_SIZE']}"
        )
        # 1. Init process group for server comms
        init_method = f"tcp://{cluster_info.storage_cluster_master_ip}:{cluster_info.storage_cluster_master_port}"
        logger.info(
            f"Initializing storage node {storage_rank} / "
            f"{cluster_info.num_storage_nodes}. "
            f"OS rank: {os.environ['RANK']}, "
            f"OS world size: {os.environ['WORLD_SIZE']} "
            f"init method: {init_method}"
        )
        torch.distributed.init_process_group(
            backend="gloo",
            world_size=cluster_info.num_storage_nodes,
            rank=storage_rank,
            init_method=init_method,
            group_name="gigl_server_comms",
        )
        logger.info(
            f"Storage node {storage_rank} / "
            f"{cluster_info.num_storage_nodes} process group initialized"
        )

        # 2. Build the dataset
        dataset = build_storage_dataset(
            task_config_uri=args.task_config_uri,
            sample_edge_direction=args.sample_edge_direction,
            splitter=args.splitter,
            tf_record_uri_pattern=".*tfrecord",
        )

        # 3. Destroy the coordination process group before spawning server
        # subprocesses. The subprocess will create its own process group on the
        # same port, so we must release it here first.
        torch.distributed.destroy_process_group()

        # 4. Run the storage server sessions
        run_storage_server(
            storage_rank=storage_rank,
            cluster_info=cluster_info,
            dataset=dataset,
            num_server_sessions=args.num_server_sessions,
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        )
    except Exception:
        args.exception_dict[process_name] = traceback.format_exc()
        raise


# ---------------------------------------------------------------------------
# Expected input helpers
# ---------------------------------------------------------------------------


def _get_expected_input_nodes_by_rank(
    num_nodes: int, cluster_info: GraphStoreInfo
) -> dict[int, list[torch.Tensor]]:
    """Get the expected sampler input for each compute rank using contiguous server assignments.

    Each compute rank is assigned contiguous server(s) via
    :func:`compute_server_assignments`. For each rank, we compute which
    fraction of each server it owns and slice the server's node tensor
    accordingly.

    Args:
        num_nodes: The number of nodes in the graph.
        cluster_info: The cluster information.

    Returns:
        A dict mapping each global rank to a list of tensors, one per
        storage server (empty tensor for unassigned servers).
    """
    partition_book = build_partition_book(
        num_entities=num_nodes, rank=0, world_size=cluster_info.num_storage_nodes
    )
    expected_sampler_input: dict[int, list[torch.Tensor]] = {}
    for global_rank in range(cluster_info.compute_cluster_world_size):
        assignments = compute_server_assignments(
            num_servers=cluster_info.num_storage_nodes,
            num_compute_nodes=cluster_info.compute_cluster_world_size,
            compute_rank=global_rank,
        )
        rank_nodes: list[torch.Tensor] = []
        for server_rank in range(cluster_info.num_storage_nodes):
            server_nodes = get_ids_on_rank(
                partition_book=partition_book, rank=server_rank
            )
            if server_rank in assignments:
                rank_nodes.append(assignments[server_rank].slice_tensor(server_nodes))
            else:
                rank_nodes.append(torch.empty(0, dtype=torch.long))
        expected_sampler_input[global_rank] = rank_nodes
    return expected_sampler_input


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


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

    def _create_cluster_info(
        self,
        num_storage_nodes: int = 2,
        num_compute_nodes: int = 2,
        num_processes_per_compute: int = 2,
    ) -> GraphStoreInfo:
        (
            cluster_master_port,
            storage_cluster_master_port,
            compute_cluster_master_port,
            rpc_master_port,
            rpc_wait_port,
        ) = get_free_ports(num_ports=5)
        host_ip = socket.gethostbyname(socket.gethostname())
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_file_path = tmp_file.name
        self.addCleanup(tmp_file.close)
        readiness_uri = UriFactory.create_uri(tmp_file_path)
        return GraphStoreInfo(
            num_storage_nodes=num_storage_nodes,
            num_compute_nodes=num_compute_nodes,
            num_processes_per_compute=num_processes_per_compute,
            cluster_master_ip=host_ip,
            storage_cluster_master_ip=host_ip,
            compute_cluster_master_ip=host_ip,
            cluster_master_port=cluster_master_port,
            storage_cluster_master_port=storage_cluster_master_port,
            compute_cluster_master_port=compute_cluster_master_port,
            rpc_master_port=rpc_master_port,
            rpc_wait_port=rpc_wait_port,
            readiness_uri=readiness_uri,
        )

    def _launch_graph_store_test(
        self,
        *,
        cluster_info: GraphStoreInfo,
        task_config_uri: Uri,
        compute_target: Callable[..., None],
        node_type: Optional[NodeType] = None,
        compute_target_extra_args: tuple[Any, ...] = (),
        server_splitter: Optional[
            Union[DistNodeAnchorLinkSplitter, DistNodeSplitter]
        ] = None,
        num_server_sessions: int = 1,
    ) -> None:
        """Launch a graph store integration test with the given configuration.

        Spawns compute client and storage server processes, then asserts all
        processes complete successfully.
        """
        master_port = get_free_port()
        host_ip = socket.gethostbyname(socket.gethostname())
        ctx = mp.get_context("spawn")
        exception_dict = mp.Manager().dict()
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
                    node_type=node_type,
                    exception_dict=exception_dict,
                    compute_target=compute_target,
                    compute_target_extra_args=compute_target_extra_args,
                )
                client_process = ctx.Process(
                    target=_client_compute_process,
                    args=[client_args],
                    name=f"client_{i}",
                )
                client_process.start()
                launched_processes.append(client_process)

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
                    splitter=server_splitter,
                    num_server_sessions=num_server_sessions,
                )
                server_process = ctx.Process(
                    target=_run_storage_main_process,
                    args=[server_args],
                    name=f"server_{i}",
                )
                server_process.start()
                launched_processes.append(server_process)

        self.assert_all_processes_succeed(launched_processes, exception_dict)

    def test_graph_store_homogeneous(self):
        # Simulating two server machine, two compute machines.
        # Each machine has one process.
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        cluster_info = self._create_cluster_info()
        num_cora_nodes = 2708
        expected_sampler_input = _get_expected_input_nodes_by_rank(
            num_cora_nodes, cluster_info
        )
        self._launch_graph_store_test(
            cluster_info=cluster_info,
            task_config_uri=cora_supervised_info.frozen_gbml_config_uri,
            compute_target=_run_compute_tests,
            compute_target_extra_args=(expected_sampler_input, None),
        )

    def test_homogeneous_training(self):
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        cluster_info = self._create_cluster_info(
            num_storage_nodes=2, num_compute_nodes=2, num_processes_per_compute=1
        )
        self._launch_graph_store_test(
            cluster_info=cluster_info,
            task_config_uri=cora_supervised_info.frozen_gbml_config_uri,
            compute_target=_run_compute_train_tests,
            server_splitter=DistNodeAnchorLinkSplitter(
                sampling_direction="in",
                should_convert_labels_to_edges=True,
            ),
        )

    def test_multiple_loaders_in_graph_store(self):
        """Test that multiple loader instances (2 ABLP + 2 DistNeighborLoader) can work
        in parallel, followed by another (ABLP, DistNeighborLoader) pair sequentially.
        """
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        cluster_info = self._create_cluster_info(
            num_storage_nodes=1, num_compute_nodes=2, num_processes_per_compute=1
        )
        self._launch_graph_store_test(
            cluster_info=cluster_info,
            task_config_uri=cora_supervised_info.frozen_gbml_config_uri,
            compute_target=_run_compute_multiple_loaders_test,
            server_splitter=DistNodeAnchorLinkSplitter(
                sampling_direction="in",
                should_convert_labels_to_edges=True,
            ),
        )

    # TODO: (mkolodner-sc) - Figure out why this test is failing on Google Cloud Build
    @unittest.skip("Failing on Google Cloud Build - skiping for now")
    def test_graph_store_heterogeneous(self):
        # Simulating two server machine, two compute machines.
        # Each machine has one process.
        dblp_supervised_info = get_mocked_dataset_artifact_metadata()[
            DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        cluster_info = self._create_cluster_info(
            num_storage_nodes=2, num_compute_nodes=2, num_processes_per_compute=2
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
        self._launch_graph_store_test(
            cluster_info=cluster_info,
            task_config_uri=dblp_supervised_info.frozen_gbml_config_uri,
            compute_target=_run_compute_tests,
            node_type=NodeType("author"),
            compute_target_extra_args=(expected_sampler_input, expected_edge_types),
        )
