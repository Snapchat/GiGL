"""GiGL Example Graph Store Server.

Derived from https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/distributed/server_client_mode/sage_supervised_server.py

This module is a **CLI entry point** that composes the utility functions
from :mod:`gigl.distributed.graph_store.storage_utils` with
example-specific orchestration logic.

Note about "num_server_sessions":

For each (gigl.distributed.graph_store.dist_server.init_server / gigl.distributed.graph_store.compute.init_compute_process)
pair we must have one "server session". This is because each session is a process and we need to recreate
the RPC connections for every new process.
As in inference we often use one process per node type, we will have one server session per node type.

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
import os
from distutils.util import strtobool
from typing import Literal, Optional, Union

import torch

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.os_utils import import_obj
from gigl.distributed.graph_store.storage_utils import (
    build_storage_dataset,
    run_storage_server,
)
from gigl.distributed.utils import get_graph_store_info, write_readiness_signal
from gigl.env.distributed import GraphStoreInfo
from gigl.utils.data_splitters import DistNodeAnchorLinkSplitter, DistNodeSplitter

logger = Logger()


def storage_node_process(
    storage_rank: int,
    cluster_info: GraphStoreInfo,
    task_config_uri: Uri,
    sample_edge_direction: Literal["in", "out"],
    num_server_sessions: int,
    splitter: Optional[Union[DistNodeAnchorLinkSplitter, DistNodeSplitter]] = None,
    should_load_tf_records_in_parallel: bool = True,
    tf_record_uri_pattern: str = r".*-of-.*\.tfrecord(\.gz)?$",
    ssl_positive_label_percentage: Optional[float] = None,
    num_rpc_threads: int = 16,
    rpc_timeout: Optional[int] = None,
) -> None:
    """Run a storage node process.

    Should be called *once* per storage node (machine).

    This orchestration function:

    1. Initialises a ``torch.distributed`` process group among storage
       nodes for coordination (server comms).
    2. Builds the dataset via
       :func:`~gigl.distributed.graph_store.storage_utils.build_storage_dataset`.
    3. Destroys the coordination process group and spawns one
       :func:`~gigl.distributed.graph_store.storage_utils.run_storage_server`
       per session.

    Args:
        storage_rank (int): The rank of the storage node.
        cluster_info (GraphStoreInfo): The cluster information.
        task_config_uri (Uri): The task config URI.
        sample_edge_direction (Literal["in", "out"]): The sample edge direction.
        num_server_sessions (int): Number of server sessions to run. For training, this should be 1
            (a single session for the entire training + testing lifecycle). For inference, this should
            be one session per inference node type.
        splitter (Optional[Union[DistNodeAnchorLinkSplitter, DistNodeSplitter]]): The splitter to use. If None, will not split the dataset.
        tf_record_uri_pattern (str): The TF Record URI pattern.
        ssl_positive_label_percentage (Optional[float]): The percentage of edges to select as self-supervised labels.
            Must be None if supervised edge labels are provided in advance.
            If 0.1 is provided, 10% of the edges will be selected as self-supervised labels.
        num_rpc_threads (int): The number of RPC threads to use for the server.
            This is the maximum number of concurrent RPC requests that the server can handle.
            Should be set to the maximum number of concurrent RPCs a server *must* handle,
            in practice, the compute world size is an upper bound.
        rpc_timeout (Optional[int]): The max timeout in seconds for remote
            RPC requests. If ``None``, uses the ``init_server`` default of
            180 seconds.
            If there are long running RPCs (e.g.  producer creation), and they timeout,
            then this parameter should be increased to avoid timeout errors.
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

    dataset = build_storage_dataset(
        task_config_uri=task_config_uri,
        sample_edge_direction=sample_edge_direction,
        tf_record_uri_pattern=tf_record_uri_pattern,
        splitter=splitter,
        should_load_tensors_in_parallel=should_load_tf_records_in_parallel,
        ssl_positive_label_percentage=ssl_positive_label_percentage,
    )

    # Ensure all storage nodes have finished loading before signaling readiness.
    torch.distributed.barrier()
    if storage_rank == 0:
        write_readiness_signal(cluster_info.readiness_uri)

    logger.info(f"Number of server sessions: {num_server_sessions}")

    torch.distributed.destroy_process_group()

    run_storage_server(
        storage_rank=storage_rank,
        cluster_info=cluster_info,
        dataset=dataset,
        num_server_sessions=num_server_sessions,
        num_rpc_threads=num_rpc_threads,
        rpc_timeout=rpc_timeout,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config_uri", type=str, required=True)
    parser.add_argument("--resource_config_uri", type=str, required=True)
    parser.add_argument("--job_name", type=str, required=True)
    parser.add_argument("--sample_edge_direction", type=str, required=True)
    parser.add_argument("--num_server_sessions", type=int, required=True)
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
    parser.add_argument("--num_rpc_threads", type=int, default=16)
    parser.add_argument("--rpc_timeout", type=int, default=None)
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
        num_server_sessions=args.num_server_sessions,
        splitter=splitter,
        ssl_positive_label_percentage=ssl_positive_label_percentage,
        should_load_tf_records_in_parallel=bool(
            strtobool(args.should_load_tf_records_in_parallel)
        ),
        num_rpc_threads=args.num_rpc_threads,
        rpc_timeout=args.rpc_timeout,
    )
