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
import os
from distutils.util import strtobool
from typing import Literal, Optional, Union

import torch

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.distributed.graph_store.storage_utils import (
    build_storage_dataset,
    run_storage_server,
)
from gigl.distributed.utils import get_free_ports_from_master_node, get_graph_store_info
from gigl.distributed.utils.networking import get_free_ports_from_master_node
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
    storage_world_backend: Optional[str] = None,
) -> None:
    """Run a storage node process.

    Should be called *once* per storage node (machine).

    This orchestration function:

    1. Initialises a ``torch.distributed`` process group among storage
       nodes for coordination (server comms).
    2. Builds the dataset via
       :func:`~gigl.distributed.graph_store.storage_utils.build_storage_dataset`.
    3. Obtains free ports from the master node.
    4. Destroys the coordination process group and spawns one
       :func:`~gigl.distributed.graph_store.storage_utils.run_storage_server`
       per session.

    Args:
        storage_rank: Rank of this storage node.
        cluster_info: Cluster topology information.
        task_config_uri: URI pointing to a frozen ``GbmlConfig`` protobuf.
        sample_edge_direction: Direction for edge sampling.
        num_server_sessions: Number of sequential server sessions to run.
            Typically one per inference node type.
        splitter: Optional splitter for node-anchor-link or node
            splitting.  ``None`` means no splitting.
        should_load_tf_records_in_parallel: Whether to load TFRecord
            tensors in parallel.
        tf_record_uri_pattern: Regex pattern matching TFRecord file URIs.
        ssl_positive_label_percentage: Fraction of edges to select as
            self-supervised positive labels.  Must be ``None`` when
            supervised edge labels are already provided.
        storage_world_backend: Backend for the per-session storage
            ``torch.distributed`` process group (e.g. ``"gloo"``).
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

    logger.info(f"Number of server sessions: {num_server_sessions}")

    torch_process_ports = get_free_ports_from_master_node(num_ports=num_server_sessions)
    torch.distributed.destroy_process_group()

    run_storage_server(
        storage_rank=storage_rank,
        cluster_info=cluster_info,
        dataset=dataset,
        num_server_sessions=num_server_sessions,
        torch_process_ports=torch_process_ports,
        storage_world_backend=storage_world_backend,
    )


if __name__ == "__main__":
    # TODO(kmonte): We want to expose splitter class here probably.
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config_uri", type=str, required=True)
    parser.add_argument("--resource_config_uri", type=str, required=True)
    parser.add_argument("--job_name", type=str, required=True)
    parser.add_argument("--sample_edge_direction", type=str, required=True)
    parser.add_argument("--num_server_sessions", type=int, required=True)
    parser.add_argument(
        "--should_load_tf_records_in_parallel", type=str, default="True"
    )
    args = parser.parse_args()
    logger.info(f"Running storage node with arguments: {args}")

    # Setup cluster-wide (e.g. storage and compute nodes) Torch Distributed process group.
    # This is needed so we can get the cluster information (e.g. number of storage and compute nodes) and rank/world_size.
    torch.distributed.init_process_group(backend="gloo")
    cluster_info = get_graph_store_info()
    logger.info(f"Cluster info: {cluster_info}")
    logger.info(
        f"World size: {torch.distributed.get_world_size()}, "
        f"rank: {torch.distributed.get_rank()}, "
        f"OS world size: {os.environ['WORLD_SIZE']}, "
        f"OS rank: {os.environ['RANK']}"
    )
    # Tear down the """"global""" process group so we can have a server-specific process group.
    torch.distributed.destroy_process_group()
    storage_node_process(
        storage_rank=cluster_info.storage_node_rank,
        cluster_info=cluster_info,
        task_config_uri=UriFactory.create_uri(args.task_config_uri),
        sample_edge_direction=args.sample_edge_direction,
        num_server_sessions=args.num_server_sessions,
        should_load_tf_records_in_parallel=bool(
            strtobool(args.should_load_tf_records_in_parallel)
        ),
    )
