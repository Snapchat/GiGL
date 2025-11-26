"""Built-in GiGL Graph Store Server.

Derivved from https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/distributed/server_client_mode/sage_supervised_server.py

"""
import argparse
import os

import graphlearn_torch as glt
import torch

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.distributed import build_dataset_from_task_config_uri
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.graph_store.remote_dataset import register_dataset
from gigl.distributed.utils import get_graph_store_info
from gigl.env.distributed import GraphStoreInfo

logger = Logger()


def _run_storage_process(
    server_rank: int,
    cluster_info: GraphStoreInfo,
    dataset: DistDataset,
) -> None:
    logger.info(
        f"Initializing server {server_rank} / {cluster_info.num_storage_nodes } on {cluster_info.cluster_master_ip}:{cluster_info.cluster_master_port}. Cluster rank: {os.environ.get('RANK')}"
    )
    register_dataset(dataset)
    glt.distributed.init_server(
        num_servers=cluster_info.num_storage_nodes,
        server_rank=server_rank,
        dataset=dataset,
        master_addr=cluster_info.cluster_master_ip,
        master_port=cluster_info.cluster_master_port,
        num_clients=cluster_info.compute_cluster_world_size,
    )

    logger.info(
        f"Waiting for server rank {server_rank} / {cluster_info.num_storage_nodes} to exit"
    )
    glt.distributed.wait_and_shutdown_server()
    logger.info(f"Server rank {server_rank} exited")


def storage_node_process(
    server_rank: int,
    cluster_info: GraphStoreInfo,
    task_config_uri: Uri,
    is_inference: bool,
    tf_record_uri_pattern: str = ".*-of-.*\.tfrecord(\.gz)?$",
) -> None:
    """Run a storage node process

    Should be called *once* per storage node (machine).

    Args:
        server_rank (int): The rank of the server.
        cluster_info (GraphStoreInfo): The cluster information.
        task_config_uri (Uri): The task config URI.
        is_inference (bool): Whether the process is an inference process.
        tf_record_uri_pattern (str): The TF Record URI pattern.
    """
    init_method = f"tcp://{cluster_info.storage_cluster_master_ip}:{cluster_info.storage_cluster_master_port}"
    logger.info(
        f"Initializing server {server_rank} / {cluster_info.num_storage_nodes}. OS rank: {os.environ['RANK']}, OS world size: {os.environ['WORLD_SIZE']} init method: {init_method}"
    )
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=cluster_info.num_storage_nodes,
        rank=server_rank,
        init_method=init_method,
        group_name="gigl_server_comms",
    )
    logger.info(
        f"Server {server_rank} / {cluster_info.num_storage_nodes} process group initialized"
    )
    dataset = build_dataset_from_task_config_uri(
        task_config_uri=task_config_uri,
        is_inference=is_inference,
        _tfrecord_uri_pattern=tf_record_uri_pattern,
    )
    server_processes = []
    mp_context = torch.multiprocessing.get_context("spawn")
    # TODO(kmonte): Enable more than one server process per machine
    for i in range(1):
        server_process = mp_context.Process(
            target=_run_storage_process,
            args=(
                server_rank + i,  # server_rank
                cluster_info,  # cluster_info
                dataset,  # dataset
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
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    is_inference = args.is_inference
    torch.distributed.init_process_group()
    cluster_info = get_graph_store_info()
    # Tear down the """"global""" process group so we can have a server-specific process group.
    torch.distributed.destroy_process_group()
    storage_node_process(
        server_rank=cluster_info.storage_node_rank,
        cluster_info=cluster_info,
        task_config_uri=UriFactory.create_uri(args.task_config_uri),
        is_inference=is_inference,
    )
