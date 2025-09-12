import os

# Suppress TensorFlow logs
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # isort: skip

import argparse
import json
import uuid

import graphlearn_torch as glt
import torch

import gigl.distributed as gd
from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.distributed.utils import (
    get_free_port,
    get_free_ports_from_master_node,
    get_internal_ip_from_all_ranks,
    get_internal_ip_from_master_node,
)
from gigl.distributed.utils.networking import get_ports_for_server_client_clusters

logger = Logger()


def run_client(
    client_rank: int,
    num_clients: int,
    num_servers: int,
    host: str,
    port: int,
    client_master_ip: str,
    client_port: int,
    output_dir: str,
) -> None:
    logger.info(
        f"Running client with args: {client_rank=} {num_clients=} {num_servers=} {host=} {port=} {client_master_ip=} {client_port=} {output_dir=}"
    )
    logger.info(
        f"Initializing client {client_rank} / {num_clients} for {num_servers} servers on host {host} and port {port}"
    )
    glt.distributed.init_client(
        num_servers=num_servers,
        num_clients=num_clients,
        client_rank=client_rank,
        master_addr=host,
        master_port=port,
    )
    logger.info(f"Client {client_rank} initialized")
    current_ctx = glt.distributed.get_context()
    print("Current context: ", current_ctx)
    if torch.cuda.is_available():
        current_device = torch.device(current_ctx.rank % torch.cuda.device_count())
    else:
        current_device = torch.device("cpu")
    logger.info(f"Client rank {client_rank} initialized on device {current_device}")
    logger.info(f"Client rank {client_rank} requesting dataset metadata from server...")
    metadata = glt.distributed.request_server(
        0, glt.distributed.DistServer.get_dataset_meta
    )
    logger.info(f"Dataset metadata: {metadata}")
    # logger.info(f"Loading node_ids from {output_dir}/node_ids.pt")
    # node_ids = torch.load(f"{output_dir}/node_ids.pt")
    # logger.info(f"Loaded {node_ids.numel()} node_ids")
    num_workers = 4

    # loader = glt.distributed.DistNeighborLoader(
    #     data=None,
    #     num_neighbors=[2, 2],
    #     input_nodes=f"{output_dir}/node_ids_{client_rank}.pt",
    #     worker_options=glt.distributed.RemoteDistSamplingWorkerOptions(
    #         server_rank=[server_rank for server_rank in range(num_servers)],
    #         num_workers=num_workers,
    #         worker_devices=[torch.device("cpu") for i in range(num_workers)],
    #         master_addr=host,
    #         master_port=32421,
    #     ),
    #     to_device=current_device,
    # )

    # for batch in loader:
    #     logger.info(f"Batch: {batch}")
    if os.environ.get("CLUSTER_SPEC"):
        server_spec = json.loads(os.environ.get("CLUSTER_SPEC"))
    else:
        server_spec = None
    logger.info(f"Server spec: {server_spec}")
    # if client_rank == 0:
    #     for k, v in os.environ.items():
    #         logger.info(f"Environment variable: {k} = {v}")

    init_method = f"tcp://{client_master_ip}:{client_port}"
    logger.info(f"Init method: {init_method}")
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=num_clients,
        rank=client_rank,
        group_name="gigl_loader_comms",
        init_method=init_method,
    )
    gigl_loader = gd.DistNeighborLoader(
        dataset=None,
        num_neighbors=[2, 2],
        input_nodes=UriFactory.create_uri(f"{output_dir}/remote_node_info.pyast"),
        num_workers=num_workers,
        batch_size=1,
        pin_memory_device=current_device,
        worker_concurrency=num_workers,
    )
    for i, batch in enumerate(gigl_loader):
        if i % 100 == 0:
            logger.info(f"Client rank {client_rank} gigl batch {i}: {batch}")

    logger.info(f"Client rank {client_rank} finished loading data for {i} batches")
    logger.info(f"Shutting down client")
    glt.distributed.shutdown_client()
    logger.info(f"Client rank {client_rank} exited")


def run_clients(
    num_clients: int,
    num_servers: int,
    host: str,
    port: int,
    client_master_ip: str,
    client_port: int,
    output_dir: str,
) -> list:
    client_processes = []
    mp_context = torch.multiprocessing.get_context("spawn")
    for client_rank in range(num_clients):
        client_process = mp_context.Process(
            target=run_client,
            args=(
                client_rank,
                num_clients,
                num_servers,
                host,
                port,
                client_master_ip,
                client_port,
                output_dir,
            ),
        )
        client_processes.append(client_process)
    for client_process in client_processes:
        client_process.start()
    return client_processes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=get_free_port())
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"/tmp/gigl/server_client/output/{uuid.uuid4()}",
    )
    parser.add_argument("--num_clients", type=int, default=1)
    parser.add_argument("--num_servers", type=int, default=1)
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    client_port = None
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
        client_master_ip = ips[args.num_servers]
        logger.info(f"Client master IP: {client_master_ip}")
        torch.distributed.destroy_process_group()
    elif args.host == "FROM ENV" or args.port == -1:
        raise ValueError("Either host or port must be provided")
    logger.info(f"Using host: {args.host}")
    logger.info(f"Using port: {args.port}")
    client_rank = int(os.environ.get("RANK")) - args.num_servers
    run_client(
        client_rank=client_rank,
        num_clients=args.num_clients,
        num_servers=args.num_servers,
        host=args.host,
        port=args.port,
        client_master_ip=client_master_ip,
        client_port=client_port,
        output_dir=args.output_dir,
    )
    # run_clients(
    #     args.num_clients, args.num_servers, args.host, args.port, client_port, args.output_dir
    # )
