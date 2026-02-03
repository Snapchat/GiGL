"""Built-in GiGL Graph Store Server.

Derived from https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/distributed/server_client_mode/sage_supervised_server.py

TODO(kmonte): Remove this, and only expose utils.
We keep this around so we can use the utils in tests/integration/distributed/graph_store/graph_store_integration_test.py.
"""
import argparse
import os
from typing import Literal, Optional

import graphlearn_torch as glt
import torch

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_range_partitioner import DistRangePartitioner
from gigl.distributed.graph_store.storage_utils import register_dataset
from gigl.distributed.utils import get_free_ports_from_master_node, get_graph_store_info
from gigl.distributed.utils.networking import get_free_ports_from_master_node
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper

logger = Logger()


def _run_storage_process(
    storage_rank: int,
    cluster_info: GraphStoreInfo,
    dataset: DistDataset,
    torch_process_port: int,
    storage_world_backend: Optional[str],
) -> None:
    register_dataset(dataset)
    cluster_master_ip = cluster_info.storage_cluster_master_ip
    logger.info(
        f"Initializing GLT server for storage node process group {storage_rank} / {cluster_info.num_storage_nodes} on {cluster_master_ip}:{cluster_info.rpc_master_port}"
    )
    # Initialize the GLT server before starting the Torch Distributed process group.
    # Otherwise, we saw intermittent hangs when initializing the server.
    glt.distributed.init_server(
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
    torch.distributed.init_process_group(
        backend=storage_world_backend,
        world_size=cluster_info.num_storage_nodes,
        rank=storage_rank,
        init_method=init_method,
    )

    logger.info(
        f"Waiting for storage node {storage_rank} / {cluster_info.num_storage_nodes} to exit"
    )
    glt.distributed.wait_and_shutdown_server()
    logger.info(f"Storage node {storage_rank} exited")


def storage_node_process(
    storage_rank: int,
    cluster_info: GraphStoreInfo,
    task_config_uri: Uri,
    sample_edge_direction: Literal["in", "out"],
    tf_record_uri_pattern: str = ".*-of-.*\.tfrecord(\.gz)?$",
    storage_world_backend: Optional[str] = None,
) -> None:
    """Run a storage node process

    Should be called *once* per storage node (machine).

    Args:
        storage_rank (int): The rank of the storage node.
        cluster_info (GraphStoreInfo): The cluster information.
        task_config_uri (Uri): The task config URI.
        is_inference (bool): Whether the process is an inference process. Defaults to True.
        tf_record_uri_pattern (str): The TF Record URI pattern.
        storage_world_backend (Optional[str]): The backend for the storage Torch Distributed process group.
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
    dataset = build_dataset(
        serialized_graph_metadata=serialized_graph_metadata,
        sample_edge_direction=sample_edge_direction,
        partitioner_class=DistRangePartitioner,
    )
    task_config = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=task_config_uri
    )
    inference_node_types = sorted(
        task_config.task_metadata_pb_wrapper.get_task_root_node_types()
    )
    logger.info(f"Inference node types: {inference_node_types}")
    torch_process_port = get_free_ports_from_master_node(num_ports=1)[0]
    torch.distributed.destroy_process_group()
    mp_context = torch.multiprocessing.get_context("spawn")
    # Since we create a new inference process for each inference node type, we need to start a new server process for each inference node type.
    # We do this as you cannot re-connect to the same server process after it has been joined.
    for i, inference_node_type in enumerate(inference_node_types):
        logger.info(
            f"Starting storage node for inference node type {inference_node_type} (storage process group {i} / {len(inference_node_types)})"
        )
        server_processes = []
        # TODO(kmonte): Enable more than one server process per machine
        for i in range(1):
            server_process = mp_context.Process(
                target=_run_storage_process,
                args=(
                    storage_rank + i,  # storage_rank
                    cluster_info,  # cluster_info
                    dataset,  # dataset
                    torch_process_port,  # torch_process_port
                    storage_world_backend,  # storage_world_backend
                ),
            )
            server_processes.append(server_process)
            for server_process in server_processes:
                server_process.start()
            for server_process in server_processes:
                server_process.join()
            logger.info(
                f"All server processes for inference node type {inference_node_type} joined"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config_uri", type=str, required=True)
    parser.add_argument("--resource_config_uri", type=str, required=True)
    parser.add_argument("--job_name", type=str, required=True)
    parser.add_argument("--sample_edge_direction", type=str, required=True)
    args = parser.parse_args()
    logger.info(f"Running storage node with arguments: {args}")

    torch.distributed.init_process_group(backend="gloo")
    cluster_info = get_graph_store_info()
    logger.info(f"Cluster info: {cluster_info}")
    logger.info(
        f"World size: {torch.distributed.get_world_size()}, rank: {torch.distributed.get_rank()}, OS world size: {os.environ['WORLD_SIZE']}, OS rank: {os.environ['RANK']}"
    )
    # Tear down the """"global""" process group so we can have a server-specific process group.
    torch.distributed.destroy_process_group()
    storage_node_process(
        storage_rank=cluster_info.storage_node_rank,
        cluster_info=cluster_info,
        task_config_uri=UriFactory.create_uri(args.task_config_uri),
        sample_edge_direction=args.sample_edge_direction,
    )
