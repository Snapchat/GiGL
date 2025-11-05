import os

# Suppress TensorFlow logs
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # isort: skip

import argparse
import io
import uuid

import graphlearn_torch as glt
import torch

import gigl.distributed as gd
from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.distributed.utils import (
    get_free_port,
    get_free_ports_from_master_node,
    get_internal_ip_from_all_ranks,
    get_internal_ip_from_master_node,
    get_graph_store_info,
    get_free_ports_from_node,
)
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.utils.file_loader import FileLoader
from gigl.types.graph import to_homogeneous
from gigl.env.pipelines_config import get_resource_config
from snapchat.research.gbml.gigl_resource_config_pb2 import (
    DataflowResourceConfig,
    DataPreprocessorConfig,
    DistributedTrainerConfig,
    GiglResourceConfig,
    KFPResourceConfig,
    LocalResourceConfig,
    SharedResourceConfig,
    SparkResourceConfig,
    TrainerResourceConfig,
    VertexAiResourceConfig,
    VertexAiGraphStoreConfig,
)


logger = Logger()


def run_server(
    server_rank: int,
    num_servers: int,
    num_clients: int,
    host: str,
    port: int,
    task_config_uri: Uri,
    is_inference: bool,
) -> None:
    logger.info(f"Initializing server {server_rank} / {num_servers}. Cluster rank: {os.environ.get('RANK')}")
    dataset = gd.build_dataset_from_task_config_uri(
        task_config_uri=task_config_uri,
        is_inference=is_inference,
        _tfrecord_uri_pattern=".*tfrecord",
    )

    logger.info(f"Initializing server {server_rank} / {num_servers}")
    glt.distributed.init_server(
        num_servers=num_servers,
        server_rank=server_rank,
        dataset=dataset,
        master_addr=host,
        master_port=port,
        num_clients=num_clients,
    )

    logger.info(f"Waiting for server rank {server_rank} / {num_servers} to exit")
    glt.distributed.wait_and_shutdown_server()
    logger.info(f"Server rank {server_rank} exited")


def run_servers(
    server_rank: int,
    cluster_info: GraphStoreInfo,
    task_config_uri: Uri,
    is_inference: bool,
) -> list:
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=cluster_info.num_cluster_nodes,
        rank=server_rank,
        init_method=f"tcp://{cluster_info.storage_cluster_master_ip}:{cluster_info.storage_cluster_master_port}",
        group_name="gigl_server_comms",
    )
    glt_port = get_free_ports_from_node(
        num_ports=1,
        node_rank=cluster_info.num_compute_nodes,
    )[0]
    server_processes = []
    mp_context = torch.multiprocessing.get_context("spawn")
    for i in range(1):
        server_process = mp_context.Process(
            target=run_server,
            args=(
                server_rank, # server_rank
                cluster_info.num_cluster_nodes, # num_servers
                cluster_info.num_compute_nodes, # num_clients
                cluster_info.storage_cluster_master_ip, # host
                glt_port, # port
                task_config_uri, # task_config_uri
                is_inference, # is_inference
            ),
        )
        server_processes.append(server_process)
    for server_process in server_processes:
        server_process.start()
    return server_processes


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
    run_servers(
        server_rank = int(os.environ["RANK"]) - cluster_info.num_compute_nodes,
        cluster_info=cluster_info,
        task_config_uri=UriFactory.create_uri(args.task_config_uri),
        is_inference=is_inference,
    )
