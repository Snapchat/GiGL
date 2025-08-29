import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # isort: skip

import argparse
import time
import uuid
from pathlib import Path

import graphlearn_torch as glt
import torch

import gigl.distributed as gd
from gigl.common.logger import Logger
from gigl.distributed.utils import get_free_port
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
    dataset = gd.build_dataset_from_task_config_uri(
        task_config_uri="gs://public-gigl/mocked_assets/2024-07-15--21-30-07-UTC/cora_homogeneous_node_anchor_edge_features_user_defined_labels/frozen_gbml_config.yaml",
        is_inference=True,
        _tfrecord_uri_pattern=".*tfrecord",
    )
    logger.info(
        f"Dumping {to_homogeneous(dataset.node_ids).numel()} node_ids to {output_dir}/node_ids.pt"
    )
    torch.save(to_homogeneous(dataset.node_ids), f"{output_dir}/node_ids.pt")
    logger.info(f"Initializing server")
    glt.distributed.init_server(
        num_servers=num_servers,
        server_rank=server_rank,
        dataset=dataset,
        master_addr=host,
        master_port=port,
        num_clients=num_clients,
    )

    logger.info(f"Waiting for server rank {server_rank} to exit")
    glt.distributed.wait_and_shutdown_server()
    logger.info(f"Server rank {server_rank} exited")


def run_client(
    client_rank: int,
    num_clients: int,
    num_servers: int,
    host: str,
    port: int,
    output_dir: str,
) -> None:
    glt.distributed.init_client(
        num_servers=num_servers,
        num_clients=num_clients,
        client_rank=client_rank,
        master_addr=host,
        master_port=port,
    )
    current_ctx = glt.distributed.get_context()
    current_device = torch.device(current_ctx.rank % torch.cuda.device_count())
    logger.info(f"Client rank {client_rank} initialized on device {current_device}")

    logger.info(f"Loading node_ids from {output_dir}/node_ids.pt")
    node_ids = torch.load(f"{output_dir}/node_ids.pt")
    logger.info(f"Loaded {node_ids.numel()} node_ids")
    num_workers = 4

    loader = glt.distributed.DistNeighborLoader(
        data=None,
        num_neighbors=[2, 2],
        input_nodes=f"{output_dir}/node_ids.pt",
        worker_options=glt.distributed.RemoteDistSamplingWorkerOptions(
            server_rank=0,
            num_workers=num_workers,
            worker_devices=[torch.device("cpu") for i in range(num_workers)],
            master_addr=host,
            master_port=get_free_port(),
        ),
        to_device=current_device,
    )

    for batch in loader:
        logger.info(f"Batch: {batch}")

    logger.info(f"Shutting down client")
    glt.distributed.shutdown_client()
    logger.info(f"Client rank {client_rank} exited")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_servers", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default=f"/tmp/gigl/server_client/output/{uuid.uuid4()}")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=get_free_port())
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # Parse arguments
    num_servers = args.num_servers
    num_clients = args.num_clients
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    server_processes = []
    mp_context = torch.multiprocessing.get_context("spawn")

    for server_rank in range(num_servers):
        server_process = mp_context.Process(
            target=run_server,
            args=(
                server_rank,
                num_servers,
                num_clients,
                args.host,
                args.port,
                output_dir,
            ),
        )
        server_processes.append(server_process)

    for server_process in server_processes:
        server_process.start()

    output_file = Path(f"{output_dir}/node_ids.pt")

    while not output_file.exists():
        time.sleep(5)
        logger.info(
            f"Waiting for server rank {server_rank} to dump node_ids to {output_dir}/node_ids.pt"
        )

    client_processes = []

    for client_rank in range(num_clients):
        client_process = mp_context.Process(
            target=run_client,
            args=(
                client_rank,
                num_clients,
                num_servers,
                args.host,
                args.port,
                output_dir,
            ),
        )
        client_processes.append(client_process)

    for client_process in client_processes:
        client_process.start()

    logger.info(f"Waiting for client processes to exit")
    for client_process in client_processes:
        client_process.join()

    logger.info(f"Waiting for server processes to exit")
    for server_process in server_processes:
        server_process.join()

    logger.info(f"All processes exited")


if __name__ == "__main__":
    main()
