import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # isort: skip

import argparse
import uuid

import graphlearn_torch as glt
import torch

import gigl.distributed as gd
from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.distributed.utils import get_free_port

logger = Logger()


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
    print("Current context: ", current_ctx)
    current_device = torch.device(current_ctx.rank % torch.cuda.device_count())
    logger.info(f"Client rank {client_rank} initialized on device {current_device}")

    logger.info(f"Loading node_ids from {output_dir}/node_ids.pt")
    node_ids = torch.load(f"{output_dir}/node_ids.pt")
    logger.info(f"Loaded {node_ids.numel()} node_ids")
    num_workers = 4

    # loader = glt.distributed.DistNeighborLoader(
    #     data=None,
    #     num_neighbors=[2, 2],
    #     input_nodes=f"{output_dir}/node_ids.pt",
    #     worker_options=glt.distributed.RemoteDistSamplingWorkerOptions(
    #         server_rank=0,
    #         num_workers=num_workers,
    #         worker_devices=[torch.device("cpu") for i in range(num_workers)],
    #         master_addr=host,
    #         master_port=get_free_port(),
    #     ),
    #     to_device=current_device,
    # )
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=1,
        rank=0,
        init_method=f"tcp://{host}:{get_free_port()}",
        group_name="gigl_comms",
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

    # for batch in loader:
    #     logger.info(f"Batch: {batch}")

    for batch in gigl_loader:
        logger.info(f"Gigl Batch: {batch}")

    logger.info(f"Shutting down client")
    glt.distributed.shutdown_client()
    logger.info(f"Client rank {client_rank} exited")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=get_free_port())
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"/tmp/gigl/server_client/output/{uuid.uuid4()}",
    )
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    run_client(0, 1, 1, args.host, args.port, args.output_dir)
