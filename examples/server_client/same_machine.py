import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # isort: skip

import argparse
import uuid
from pathlib import Path

from examples.server_client.client import run_clients
from examples.server_client.server import run_servers

from gigl.common.logger import Logger
from gigl.distributed.utils import get_free_port

logger = Logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_servers", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"/tmp/gigl/server_client/output/{uuid.uuid4()}",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=get_free_port())
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # Parse arguments
    num_servers = args.num_servers
    num_clients = args.num_clients
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    server_processes = run_servers(
        num_servers, num_clients, args.host, args.port, output_dir
    )
    client_processes = run_clients(
        num_clients, num_servers, args.host, args.port, output_dir
    )

    logger.info(f"Waiting for client processes to exit")
    for client_process in client_processes:
        client_process.join()

    logger.info(f"Waiting for server processes to exit")
    for server_process in server_processes:
        server_process.join()

    logger.info(f"All processes exited")


if __name__ == "__main__":
    main()
