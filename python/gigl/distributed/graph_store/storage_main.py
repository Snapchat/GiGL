"""Built-in GiGL Graph Store Server.

Derivved from https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/distributed/server_client_mode/sage_supervised_server.py

"""
import argparse
import functools
import os
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
from datetime import timedelta
from typing import Optional

import graphlearn_torch as glt
import torch

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.distributed import build_dataset_from_task_config_uri
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.graph_store.storage_utils import register_dataset
from gigl.distributed.utils import get_graph_store_info
from gigl.distributed.utils.networking import get_free_ports_from_master_node
#from gigl.distributed.rpc import init_rpc, rpc_is_initialized, shutdown_rpc, barrier
#from gigl.distributed.dist_server import init_server, wait_and_shutdown_server
from graphlearn_torch.distributed.rpc import init_rpc, rpc_is_initialized, shutdown_rpc, barrier
from graphlearn_torch.distributed.dist_server import init_server, wait_and_shutdown_server, DistServer
from gigl.env.distributed import GraphStoreInfo

logger = Logger()


def _wrap_method_with_logging(method_name: str, original_method):
    """Wrap a method to log its calls with arguments."""
    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        # Skip 'self' in args for logging
        log_args = args[1:] if args else args
        logger.info(f"[DistServer] {method_name} called with args={log_args}, kwargs={kwargs}")
        try:
            result = original_method(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"[DistServer] {method_name} raised exception: {e}")
            raise
    return wrapper


def _instrument_dist_server_logging():
    """Wrap all DistServer methods with logging."""
    methods_to_wrap = [
        "shutdown",
        "wait_for_exit",
        "exit",
        "get_dataset_meta",
        "get_node_partition_id",
        "get_node_feature",
        "get_tensor_size",
        "get_node_label",
        "get_edge_index",
        "get_edge_size",
        "create_sampling_producer",
        "destroy_sampling_producer",
        "start_new_epoch_sampling",
        "fetch_one_sampled_message",
    ]
    for method_name in methods_to_wrap:
        if hasattr(DistServer, method_name):
            original_method = getattr(DistServer, method_name)
            wrapped_method = _wrap_method_with_logging(method_name, original_method)
            setattr(DistServer, method_name, wrapped_method)
    logger.info("[DistServer] All methods instrumented with logging")


# Instrument DistServer methods with logging at module load time
_instrument_dist_server_logging()


def _run_storage_process(
    storage_rank: int,
    cluster_info: GraphStoreInfo,
    dataset: DistDataset,
    torch_process_port: int,
    storage_world_backend: Optional[str],
) -> None:
    register_dataset(dataset)
    timeout = timedelta(minutes=15)
    world_size = (
        cluster_info.num_storage_nodes + cluster_info.compute_cluster_world_size
    )
    rank = storage_rank + cluster_info.compute_cluster_world_size
    cluster_master_ip = cluster_info.storage_cluster_master_ip
    # init_method = f"tcp://{cluster_master_ip}:{cluster_info.rpc_wait_port}"
    # logger.info(f"Storage node {storage_rank} / {cluster_info.num_storage_nodes} will wait for compute nodes to be ready. PG rank: {rank} / {world_size} and init method {init_method} and timeout {timeout}")
    # start_time = time.time()
    # torch.distributed.init_process_group(
    #     backend="gloo",
    #     world_size=world_size,
    #     rank=rank,
    #     init_method=init_method,
    #     timeout=timeout,
    # )
    # end_time = time.time()
    # logger.info(f"Storage node {storage_rank} / {cluster_info.num_storage_nodes} waited for {end_time - start_time} seconds to be ready")
    # torch.distributed.destroy_process_group()
    logger.info(
        f"Initializing GLT server for storage node process group {storage_rank} / {cluster_info.num_storage_nodes} on {cluster_master_ip}:{cluster_info.rpc_master_port}"
    )
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
    torch.distributed.init_process_group(
        backend=storage_world_backend,
        world_size=cluster_info.num_storage_nodes,
        rank=storage_rank,
        init_method=init_method,
    )

    logger.info(
        f"Waiting for storage node {storage_rank} / {cluster_info.num_storage_nodes} to exit"
    )
    wait_and_shutdown_server()
    logger.info(f"Storage node {storage_rank} exited")


def storage_node_process(
    storage_rank: int,
    cluster_info: GraphStoreInfo,
    task_config_uri: Uri,
    is_inference: bool,
    tf_record_uri_pattern: str = ".*-of-.*\.tfrecord(\.gz)?$",
    storage_world_backend: Optional[str] = None,
) -> None:
    """Run a storage node process

    Should be called *once* per storage node (machine).

    Args:
        storage_rank (int): The rank of the storage node.
        cluster_info (GraphStoreInfo): The cluster information.
        task_config_uri (Uri): The task config URI.
        is_inference (bool): Whether the process is an inference process.
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
    dataset = build_dataset_from_task_config_uri(
        task_config_uri=task_config_uri,
        is_inference=is_inference,
        _tfrecord_uri_pattern=tf_record_uri_pattern,
    )
    torch_process_port = get_free_ports_from_master_node(num_ports=1)[0]
    torch.distributed.destroy_process_group()
    server_processes = []
    mp_context = torch.multiprocessing.get_context("spawn")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config_uri", type=str, required=True)
    parser.add_argument("--resource_config_uri", type=str, required=True)
    parser.add_argument("--is_inference", action="store_true")
    parser.add_argument("--job_name", type=str, required=True)
    args = parser.parse_args()
    logger.info(f"Running storage node with arguments: {args}")

    is_inference = args.is_inference
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
        is_inference=is_inference,
    )
