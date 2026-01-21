"""Built-in GiGL Graph Store Server.

Derivved from https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/distributed/server_client_mode/sage_supervised_server.py

"""
import argparse
import os

import torch

from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.distributed.graph_store.storage_process import storage_node_process
from gigl.distributed.utils import get_graph_store_info

logger = Logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config_uri", type=str, required=True)
    parser.add_argument("--resource_config_uri", type=str, required=True)
    parser.add_argument("--job_name", type=str, required=True)
    parser.add_argument("--is_inference", type=bool, required=True, action="store_true")
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
        is_inference=args.is_inference,
    )
