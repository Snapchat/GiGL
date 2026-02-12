"""GiGL Example Graph Store Server.

Derived from https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/distributed/server_client_mode/sage_supervised_server.py

TODO(kmonte): Figure out how we should split out common utils from this file.

Cluster Setup
=============

In Graph Store mode, storage nodes hold the graph data and serve sampling requests from compute nodes.
Each storage node initializes a GLT (GraphLearn-Torch) server and waits for connections from compute nodes.

Storage nodes accept connections from compute nodes **sequentially, by compute node**. For example:
- First, all connections from Compute Node 0 are established to Storage Nodes 0, 1, 2, ...
- Then, all connections from Compute Node 1 are established to Storage Nodes 0, 1, 2, ...
- And so on.

It's important to distinguish between:
- **Compute Node**: A physical machine in the compute cluster (e.g., a VM with multiple GPUs).
- **Compute Process**: A process running on a compute node (typically one per GPU).

Each compute node may have multiple compute processes (e.g., one per GPU), and each compute process
establishes its own connection to every storage node. For example, if a compute node has 4 GPUs,
it will establish 4 connections to each storage node.

This sequential connection setup is required because the GLT server uses a per-server lock when
initializing samplers. If connections from multiple compute nodes were established concurrently,
it could cause a deadlock.

Connection Diagram
------------------

╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                         COMPUTE TO STORAGE NODE CONNECTIONS                            ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝

     COMPUTE NODES                                              STORAGE NODES
     ═════════════                                              ═════════════

  ┌──────────────────────┐          (1)                      ┌───────────────┐
  │    COMPUTE NODE 0    │                                   │               │
  │  ┌────┬────┬────┬────┤ ══════════════════════════════════│   STORAGE 0   │
  │  │GPU │GPU │GPU │GPU │                                 ╱ │               │
  │  │ 0  │ 1  │ 2  │ 3  │ ════════════════════╲         ╱   └───────────────┘
  │  └────┴────┴────┴────┤          (2)          ╲     ╱
  └──────────────────────┘                         ╲ ╱
                                                    ╳
                                          (3)     ╱   ╲     (4)
  ┌──────────────────────┐                      ╱       ╲    ┌───────────────┐
  │    COMPUTE NODE 1    │                    ╱           ╲  │               │
  │  ┌────┬────┬────┬────┤ ═════════════════╱               ═│   STORAGE 1   │
  │  │GPU │GPU │GPU │GPU │                                   │               │
  │  │ 0  │ 1  │ 2  │ 3  │ ══════════════════════════════════│               │
  │  └────┴────┴────┴────┤                                   └───────────────┘
  └──────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  (1) Compute Node 0  →  Storage 0   (4 connections, one per GPU)            │
  │  (2) Compute Node 0  →  Storage 1   (4 connections, one per GPU)            │
  │  (3) Compute Node 1  →  Storage 0   (4 connections, one per GPU)            │
  │  (4) Compute Node 1  →  Storage 1   (4 connections, one per GPU)            │
  └─────────────────────────────────────────────────────────────────────────────┘

Storage nodes wait for all compute processes to connect, then serve sampling requests until
the compute processes signal shutdown via `gigl.distributed.graph_store.compute.shutdown_compute_process`.

"""
import argparse
import ast
import multiprocessing.context as py_mp_context
import os
from distutils.util import strtobool
from typing import Literal, Optional, Union

import torch

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.os_utils import import_obj
from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_range_partitioner import DistRangePartitioner
from gigl.distributed.graph_store.dist_server import (
    init_server,
    wait_and_shutdown_server,
)
from gigl.distributed.graph_store.storage_utils import register_dataset
from gigl.distributed.utils import get_free_ports_from_master_node, get_graph_store_info
from gigl.distributed.utils.networking import get_free_ports_from_master_node
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.utils.data_splitters import DistNodeAnchorLinkSplitter, DistNodeSplitter

logger = Logger()


def _run_storage_process(
    storage_rank: int,
    cluster_info: GraphStoreInfo,
    dataset: DistDataset,
    torch_process_port: int,
    storage_world_backend: Optional[str],
) -> None:
    """
    Runs a storage process.

    This function does the following:

    1. "Registers" the dataset so that gigl.distributed.graph_store.remote_dist_dataset.RemoteDistDataset can access it.
    2. Initialized the GLT server.
        Under the hood this is synchronized with the clients initializing via gigl.distributed.graph_store.compute.init_compute_process,
        and after this call there will be Torch RPC connections between the storage nodes and compute nodes.
    3. Initializes the Torch Distributed process group for the storage node.
    4. Waits for the server to exit.
        Will wait until clients are also shutdown (with `gigl.distributed.graph_store.compute.shutdown_compute_proccess`)

    Args:
        storage_rank (int): The rank of the storage node.
        cluster_info (GraphStoreInfo): The cluster information.
        dataset (DistDataset): The dataset.
        torch_process_port (int): The port for the Torch process.
        storage_world_backend (Optional[str]): The backend for the storage Torch Distributed process group.
    """

    # "Register" the dataset so that gigl.distributed.graph_store.remote_dist_dataset.RemoteDistDataset can access it.
    register_dataset(dataset)
    cluster_master_ip = cluster_info.storage_cluster_master_ip
    logger.info(
        f"Initializing GLT server for storage node process group {storage_rank} / {cluster_info.num_storage_nodes} on {cluster_master_ip}:{cluster_info.rpc_master_port}"
    )
    # Initialize the GLT server before starting the Torch Distributed process group.
    # Otherwise, we saw intermittent hangs when initializing the server.
    init_server(
        num_servers=cluster_info.num_storage_nodes,
        server_rank=storage_rank,
        dataset=dataset,
        master_addr=cluster_master_ip,
        master_port=cluster_info.rpc_master_port,
        num_clients=cluster_info.compute_cluster_world_size,
    )

    init_method = f"tcp://{cluster_info.storage_cluster_master_ip}:{torch_process_port}"
    logger.info(
        f"Initializing storage node process group {storage_rank} / {cluster_info.num_storage_nodes} with backend {storage_world_backend} on {init_method}"
    )

    # Torch Distributed process group is needed so that the storage cluster can talk to each other.
    # This is needed for `RemoteDistDataset.get_free_ports_on_storage_cluster` to work.
    # Note this is called on the *compute* cluster, but requires the storage cluster to have a process group initialized.
    torch.distributed.init_process_group(
        backend=storage_world_backend,
        world_size=cluster_info.num_storage_nodes,
        rank=storage_rank,
        init_method=init_method,
    )

    logger.info(
        f"Waiting for storage node {storage_rank} / {cluster_info.num_storage_nodes} to exit"
    )
    # Wait for the server to exit.
    # Will wait until clients are also shutdown (with `gigl.distributed.graph_store.compute.shutdown_compute_proccess`)
    wait_and_shutdown_server()
    logger.info(f"Storage node {storage_rank} exited")


def storage_node_process(
    storage_rank: int,
    cluster_info: GraphStoreInfo,
    task_config_uri: Uri,
    sample_edge_direction: Literal["in", "out"],
    splitter: Optional[Union[DistNodeAnchorLinkSplitter, DistNodeSplitter]] = None,
    should_load_tf_records_in_parallel: bool = True,
    tf_record_uri_pattern: str = ".*-of-.*\.tfrecord(\.gz)?$",
    ssl_positive_label_percentage: Optional[float] = None,
    storage_world_backend: Optional[str] = None,
    num_server_sessions: Optional[int] = None,
) -> None:
    """Run a storage node process

    Should be called *once* per storage node (machine).

    Args:
        storage_rank (int): The rank of the storage node.
        cluster_info (GraphStoreInfo): The cluster information.
        task_config_uri (Uri): The task config URI.
        sample_edge_direction (Literal["in", "out"]): The sample edge direction.
        splitter (Optional[Union[DistNodeAnchorLinkSplitter, DistNodeSplitter]]): The splitter to use. If None, will not split the dataset.
        tf_record_uri_pattern (str): The TF Record URI pattern.
        ssl_positive_label_percentage (Optional[float]): The percentage of edges to select as self-supervised labels.
            Must be None if supervised edge labels are provided in advance.
            If 0.1 is provided, 10% of the edges will be selected as self-supervised labels.
        storage_world_backend (Optional[str]): The backend for the storage Torch Distributed process group.
        num_server_sessions (Optional[int]): Number of server sessions to run. For training, this should be 1
            (a single session for the entire training + testing lifecycle). If None, defaults to one session
            per inference node type (the existing inference behavior).
    """
    init_method = f"tcp://{cluster_info.storage_cluster_master_ip}:{cluster_info.storage_cluster_master_port}"
    logger.info(
        f"Initializing storage node {storage_rank} / {cluster_info.num_storage_nodes}. OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']} init method: {init_method}"
    )
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=cluster_info.num_storage_nodes,
        rank=storage_rank,
        init_method=init_method,
        group_name="gigl_server_comms",
    )
    logger.info(
        f"Storage node {storage_rank} / {cluster_info.num_storage_nodes} process group initialized"
    )
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=task_config_uri
    )
    serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
        preprocessed_metadata_pb_wrapper=gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
        graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
        tfrecord_uri_pattern=tf_record_uri_pattern,
    )
    # TODO(kmonte): Add support for TFDatasetOptions.
    dataset = build_dataset(
        serialized_graph_metadata=serialized_graph_metadata,
        sample_edge_direction=sample_edge_direction,
        should_load_tensors_in_parallel=should_load_tf_records_in_parallel,
        partitioner_class=DistRangePartitioner,
        splitter=splitter,
        _ssl_positive_label_percentage=ssl_positive_label_percentage,
    )
    # Determine the number of server sessions.
    # For inference, we default to one session per inference node type (each node type gets its own
    # complete RPC lifecycle). For training, the caller should set num_server_sessions=1 since the
    # compute side runs a single session for the entire training + testing lifecycle.
    if num_server_sessions is None:
        inference_node_types = sorted(
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_task_root_node_types()
        )
        num_sessions = len(inference_node_types)
        logger.info(
            f"num_server_sessions not set, defaulting to {num_sessions} sessions (one per inference node type: {inference_node_types})"
        )
    else:
        num_sessions = num_server_sessions
        logger.info(f"num_server_sessions explicitly set to {num_sessions}")

    torch_process_ports = get_free_ports_from_master_node(
        num_ports=num_sessions
    )
    torch.distributed.destroy_process_group()
    for session_idx in range(num_sessions):
        logger.info(
            f"Starting storage node rank {storage_rank} / {cluster_info.num_storage_nodes} for server session {session_idx} / {num_sessions}"
        )
        mp_context = torch.multiprocessing.get_context("spawn")
        server_processes: list[py_mp_context.SpawnProcess] = []
        # TODO(kmonte): Enable more than one server process per machine
        num_server_processes = 1
        for i in range(num_server_processes):
            server_process = mp_context.Process(
                target=_run_storage_process,
                args=(
                    storage_rank + i,  # storage_rank
                    cluster_info,  # cluster_info
                    dataset,  # dataset
                    torch_process_ports[session_idx],  # torch_process_port
                    storage_world_backend,  # storage_world_backend
                ),
            )
            server_processes.append(server_process)
            for server_process in server_processes:
                server_process.start()
            for server_process in server_processes:
                server_process.join()
        logger.info(
            f"All server processes on storage node rank {storage_rank} / {cluster_info.num_storage_nodes} joined for server session {session_idx} / {num_sessions}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config_uri", type=str, required=True)
    parser.add_argument("--resource_config_uri", type=str, required=True)
    parser.add_argument("--job_name", type=str, required=True)
    parser.add_argument("--sample_edge_direction", type=str, required=True)
    parser.add_argument(
        "--should_load_tf_records_in_parallel", type=str, default="True"
    )
    # Splitter configuration: use import_obj to dynamically load a splitter class.
    # This is needed for training (where the dataset needs train/val/test splits) but not for inference.
    parser.add_argument(
        "--splitter_cls_path",
        type=str,
        default=None,
        help="Fully qualified import path to splitter class, e.g. 'gigl.utils.data_splitters.DistNodeAnchorLinkSplitter'",
    )
    parser.add_argument(
        "--splitter_kwargs",
        type=str,
        default=None,
        help="Python dict literal of keyword arguments for the splitter constructor, "
        "parsed with ast.literal_eval. Tuples are supported directly, e.g. "
        "'supervision_edge_types': [('paper', 'to', 'author')].",
    )
    parser.add_argument(
        "--ssl_positive_label_percentage",
        type=str,
        default=None,
        help="Percentage of edges to select as self-supervised labels. "
        "Must be None if supervised edge labels are provided in advance.",
    )
    parser.add_argument(
        "--num_server_sessions",
        type=str,
        default=None,
        help="Number of server sessions. For training use '1'. "
        "If not set, defaults to one session per inference node type.",
    )
    args = parser.parse_args()
    logger.info(f"Running storage node with arguments: {args}")

    # Build splitter from args if provided.
    # We use ast.literal_eval instead of json.loads so that Python tuples (e.g. for EdgeType)
    # can be passed directly in the splitter_kwargs string without needing custom serialization.
    splitter: Optional[Union[DistNodeAnchorLinkSplitter, DistNodeSplitter]] = None
    ssl_positive_label_percentage: Optional[float] = None
    if args.splitter_cls_path:
        splitter_cls = import_obj(args.splitter_cls_path)
        splitter_kwargs = (
            ast.literal_eval(args.splitter_kwargs) if args.splitter_kwargs else {}
        )
        splitter = splitter_cls(**splitter_kwargs)
        logger.info(f"Built splitter: {splitter}")

    if args.ssl_positive_label_percentage:
        ssl_positive_label_percentage = float(args.ssl_positive_label_percentage)

    num_server_sessions = (
        int(args.num_server_sessions) if args.num_server_sessions else None
    )

    # Setup cluster-wide (e.g. storage and compute nodes) Torch Distributed process group.
    # This is needed so we can get the cluster information (e.g. number of storage and compute nodes) and rank/world_size.
    torch.distributed.init_process_group(backend="gloo")
    cluster_info = get_graph_store_info()
    logger.info(f"Cluster info: {cluster_info}")
    logger.info(
        f"World size: {torch.distributed.get_world_size()}, rank: {torch.distributed.get_rank()}, OS world size: {os.environ['WORLD_SIZE']}, OS rank: {os.environ['RANK']}"
    )
    # Tear down the "global" process group so we can have a server-specific process group.

    torch.distributed.destroy_process_group()
    storage_node_process(
        storage_rank=cluster_info.storage_node_rank,
        cluster_info=cluster_info,
        task_config_uri=UriFactory.create_uri(args.task_config_uri),
        sample_edge_direction=args.sample_edge_direction,
        splitter=splitter,
        ssl_positive_label_percentage=ssl_positive_label_percentage,
        should_load_tf_records_in_parallel=bool(
            strtobool(args.should_load_tf_records_in_parallel)
        ),
        num_server_sessions=num_server_sessions,
    )
