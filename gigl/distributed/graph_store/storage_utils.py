"""Composable utilities for Graph Store storage nodes.

Provides two building blocks that callers (examples, integration tests, CLI
entry points) can combine with their own orchestration logic:

* :func:`build_storage_dataset` — loads a task config, converts metadata,
  and builds a :class:`~gigl.distributed.dist_dataset.DistDataset` using
  :class:`~gigl.distributed.dist_range_partitioner.DistRangePartitioner`.

* :func:`run_storage_server` — initialises a GLT server, sets up a
  ``torch.distributed`` process group for the storage cluster, and blocks
  until compute nodes signal shutdown.
"""

import multiprocessing.context as py_mp_context
from typing import Literal, Optional, Union

import torch

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_range_partitioner import DistRangePartitioner
from gigl.distributed.graph_store.dist_server import (
    init_server,
    wait_and_shutdown_server,
)
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.utils.data_splitters import DistNodeAnchorLinkSplitter, DistNodeSplitter

logger = Logger()


# TODO(kmonte): Add support for TFDatasetOptions.
def build_storage_dataset(
    task_config_uri: Uri,
    sample_edge_direction: Literal["in", "out"],
    tf_record_uri_pattern: str = r".*-of-.*\.tfrecord(\.gz)?$",
    splitter: Optional[Union[DistNodeAnchorLinkSplitter, DistNodeSplitter]] = None,
    should_load_tensors_in_parallel: bool = True,
    ssl_positive_label_percentage: Optional[float] = None,
) -> DistDataset:
    """Build a :class:`DistDataset` for a storage node from a task config.

    Loads the GBML config from *task_config_uri*, translates the protobuf
    metadata into :class:`SerializedGraphMetadata`, and delegates to
    :func:`~gigl.distributed.dataset_factory.build_dataset` with
    :class:`~gigl.distributed.dist_range_partitioner.DistRangePartitioner`.

    This should be called **once per storage node** (machine).  A
    ``torch.distributed`` process group must already be initialised among
    all storage nodes before calling this function so that the dataset can
    be partitioned correctly.

    Args:
        task_config_uri: URI pointing to a frozen ``GbmlConfig`` protobuf.
        sample_edge_direction: Direction for edge sampling (``"in"`` or
            ``"out"``).
        tf_record_uri_pattern: Regex pattern to match TFRecord file URIs.
        splitter: Optional splitter for node-anchor-link or node splitting.
            If ``None``, the dataset will not be split.
        should_load_tensors_in_parallel: Whether to load TFRecord tensors
            in parallel.
        ssl_positive_label_percentage: Fraction of edges to select as
            self-supervised positive labels.  Must be ``None`` when
            supervised edge labels are already provided.  For example,
            ``0.1`` selects 10 % of edges.

    Returns:
        A partitioned :class:`DistDataset` ready to be served.
    """
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=task_config_uri
    )
    serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
        preprocessed_metadata_pb_wrapper=gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
        graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
        tfrecord_uri_pattern=tf_record_uri_pattern,
    )
    return build_dataset(
        serialized_graph_metadata=serialized_graph_metadata,
        sample_edge_direction=sample_edge_direction,
        should_load_tensors_in_parallel=should_load_tensors_in_parallel,
        partitioner_class=DistRangePartitioner,
        splitter=splitter,
        _ssl_positive_label_percentage=ssl_positive_label_percentage,
    )


def _run_storage_server_session(
    storage_rank: int,
    cluster_info: GraphStoreInfo,
    dataset: DistDataset,
) -> None:
    """Run a single storage-server session and block until shutdown.

    This is the subprocess target spawned by :func:`run_storage_server`.
    It performs the following steps:

    1. **Initialises the GiGL DistServer** with the dataset.  Under the
       hood this is synchronised with the clients initialising via
       :func:`gigl.distributed.graph_store.compute.init_compute_process`;
       after this call Torch RPC connections exist between storage and
       compute nodes.
    2. **Initialises a torch.distributed process group** among storage
       nodes for collective operations (e.g., degree tensor aggregation).
    3. **Waits for the server to exit.** The server blocks until clients
       call
       :func:`gigl.distributed.graph_store.compute.shutdown_compute_proccess`.

    .. note::
        The GLT server is initialised *before* the ``torch.distributed``
        process group.  Reversing this order caused intermittent hangs.

    Args:
        storage_rank: Rank of this storage node in the storage cluster.
        cluster_info: Cluster topology information.
        dataset: The :class:`DistDataset` to serve.
    """
    cluster_master_ip = cluster_info.storage_cluster_master_ip
    logger.info(
        f"Initializing GLT server for storage node process group "
        f"{storage_rank} / {cluster_info.num_storage_nodes} "
        f"on {cluster_master_ip}:{cluster_info.rpc_master_port}"
    )
    # Initialize the GLT server before starting the Torch Distributed
    # process group.  Otherwise, we saw intermittent hangs when
    # initializing the server.
    init_server(
        num_servers=cluster_info.num_storage_nodes,
        server_rank=storage_rank,
        dataset=dataset,
        master_addr=cluster_master_ip,
        master_port=cluster_info.rpc_master_port,
        num_clients=cluster_info.compute_cluster_world_size,
    )

    # Initialize process group among storage servers for collective operations
    # (e.g., all-reduce for degree tensor aggregation). Must be after init_server.
    logger.info(
        f"Initializing storage process group for storage node "
        f"{storage_rank} / {cluster_info.num_storage_nodes}"
    )
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=cluster_info.num_storage_nodes,
        rank=storage_rank,
        init_method=f"tcp://{cluster_master_ip}:{cluster_info.storage_cluster_master_port}",
    )

    logger.info(
        f"Waiting for storage node "
        f"{storage_rank} / {cluster_info.num_storage_nodes} to exit"
    )
    # Wait for the server to exit.  Will block until clients also shut
    # down (with `gigl.distributed.graph_store.compute.shutdown_compute_proccess`).
    wait_and_shutdown_server()

    torch.distributed.destroy_process_group()
    logger.info(f"Storage node {storage_rank} exited")


def run_storage_server(
    storage_rank: int,
    cluster_info: GraphStoreInfo,
    dataset: DistDataset,
    num_server_sessions: int,
    timeout_seconds: Optional[float] = None,
) -> None:
    """Spawn sequential storage-server sessions as subprocesses.

    Each server session requires its own spawned process because you
    cannot re-connect to the same GLT server process after it has been
    joined.  This function loops over *num_server_sessions*, spawning
    :func:`_run_storage_server_session` as a subprocess each time and
    joining it before starting the next.

    Args:
        storage_rank: Rank of this storage node in the storage cluster.
        cluster_info: Cluster topology information.
        dataset: The :class:`DistDataset` to serve.
        num_server_sessions: Number of sequential server sessions to run
            (typically one per inference node type).
        timeout_seconds: Timeout for joining each server subprocess.
            ``None`` waits indefinitely.
    """
    mp_context = torch.multiprocessing.get_context("spawn")
    for i in range(num_server_sessions):
        logger.info(
            f"Starting storage node rank {storage_rank} / "
            f"{cluster_info.num_storage_nodes} "
            f"(server session {i} / {num_server_sessions})"
        )
        server_processes: list[py_mp_context.SpawnProcess] = []
        # TODO(kmonte): Enable more than one server process per machine
        num_server_processes = 1
        for j in range(num_server_processes):
            server_process = mp_context.Process(
                target=_run_storage_server_session,
                args=(
                    storage_rank + j,  # storage_rank
                    cluster_info,  # cluster_info
                    dataset,  # dataset
                ),
            )
            server_processes.append(server_process)
        for server_process in server_processes:
            server_process.start()
        for server_process in server_processes:
            server_process.join(timeout_seconds)
        logger.info(
            f"All server processes on storage node rank {storage_rank} / "
            f"{cluster_info.num_storage_nodes} joined for "
            f"server session {i}"
        )
