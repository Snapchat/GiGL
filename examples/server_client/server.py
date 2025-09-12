import os

# Suppress TensorFlow logs
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # isort: skip

import argparse
import io
import uuid

import graphlearn_torch as glt
import torch

import gigl.distributed as gd
from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.distributed.sampler import RemoteNodeInfo
from gigl.distributed.utils import (
    get_free_port,
    get_free_ports_from_master_node,
    get_internal_ip_from_all_ranks,
    get_internal_ip_from_master_node,
)
from gigl.distributed.utils.networking import get_ports_for_server_client_clusters
from gigl.src.common.utils.file_loader import FileLoader
from gigl.types.graph import to_homogeneous

logger = Logger()


def run_server(
    server_rank: int,
    num_servers: int,
    num_clients: int,
    host: str,
    port: int,
    output_dir: str,
) -> None:
    logger.info(f"Initializing server {server_rank} / {num_servers}. Cluster rank: {os.environ.get('RANK')}")
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=num_servers,
        rank=server_rank,
        init_method=f"tcp://{host}:{43002}",
        group_name="gigl_server_comms",
    )
    dataset = gd.build_dataset_from_task_config_uri(
        task_config_uri="gs://public-gigl/mocked_assets/2024-07-15--21-30-07-UTC/cora_homogeneous_node_anchor_edge_features_user_defined_labels/frozen_gbml_config.yaml",
        is_inference=True,
        _tfrecord_uri_pattern=".*tfrecord",
    )
    node_id_uri = f"{output_dir}/node_ids.pt"
    if server_rank == 0:
        node_tensor_uris = []
        node_pb = to_homogeneous(dataset.node_pb)
        if isinstance(node_pb, torch.Tensor):
            total_node_ids = node_pb.numel()
        elif isinstance(node_pb, glt.partition.RangePartitionBook):
            total_node_ids = node_pb.partition_bounds[-1]
        else:
            raise ValueError(f"Unsupported node partition book type: {type(node_pb)}")

        num_shards = num_servers * num_clients
        for shard_rank in range(num_shards):
            node_tensor_uri = f"{output_dir}/node_ids_{shard_rank}.pt"
            node_tensor_uris.append(node_tensor_uri)
            logger.info(
                f"Dumping {total_node_ids // num_shards} node_ids to {node_tensor_uri}"
            )
            bytes_io = io.BytesIO()
            torch.save(
                torch.arange(
                    start=shard_rank * (total_node_ids // num_shards),
                    end=(shard_rank + 1) * (total_node_ids // num_shards),
                ),
                bytes_io,
            )
            bytes_io.seek(0)
            FileLoader().load_from_filelike(
                UriFactory.create_uri(node_tensor_uri), bytes_io
            )
            bytes_io.close()

        remote_node_info = RemoteNodeInfo(
            node_type=None,
            edge_types=dataset.get_edge_types(),
            node_tensor_uris=node_tensor_uris,
            node_feature_info=dataset.node_feature_info,
            edge_feature_info=dataset.edge_feature_info,
            num_partitions=dataset.num_partitions,
            edge_dir=dataset.edge_dir,
            master_addr=host,
            master_port=get_free_port(),
            num_servers=num_servers,
        )
        FileLoader().load_from_filelike(
            UriFactory.create_uri(f"{output_dir}/remote_node_info.pyast"),
            io.BytesIO(remote_node_info.dump().encode()),
        )
        print(f"Wrote remote node info to {output_dir}/remote_node_info.pyast")
    logger.info(f"Initializing serve {server_rank} / {num_servers}")
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
    num_servers: int,
    num_clients: int,
    host: str,
    port: int,
    output_dir: str,
) -> list:
    server_processes = []
    mp_context = torch.multiprocessing.get_context("spawn")
    for i in range(1):
        server_process = mp_context.Process(
            target=run_server,
            args=(server_rank, num_servers, num_clients, host, port, output_dir),
        )
        server_processes.append(server_process)
    for server_process in server_processes:
        server_process.start()
    return server_processes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=get_free_port())
    parser.add_argument("--num_servers", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"/tmp/gigl/server_client/output/{uuid.uuid4()}",
    )
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    if args.host == "FROM ENV" and args.port == -1:
        logger.info(f"Using host and port from process group")
        torch.distributed.init_process_group(backend="gloo")
        args.host = get_internal_ip_from_master_node()
        args.port = get_free_ports_from_master_node(num_ports=1)[0]
        server_port, client_port = get_ports_for_server_client_clusters(
            args.num_servers, args.num_clients
        )
        logger.info(f"Server port: {server_port}, client port: {client_port}")
        ips = get_internal_ip_from_all_ranks()
        logger.info(f"IPs: {ips}")
        torch.distributed.destroy_process_group()
    elif args.host == "FROM ENV" or args.port == -1:
        raise ValueError("Either host or port must be provided")
    logger.info(f"Using host: {args.host}")
    logger.info(f"Using port: {args.port}")
    run_servers(
        int(os.environ.get("RANK")),
        args.num_servers,
        args.num_clients,
        args.host,
        args.port,
        args.output_dir,
    )
