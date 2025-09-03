import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # isort: skip

import argparse
import io
import uuid

import graphlearn_torch as glt
import torch

import gigl.distributed as gd
from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.distributed.sampler import RemoteNodeInfo
from gigl.distributed.utils import get_free_port
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
    dataset = gd.build_dataset_from_task_config_uri(
        task_config_uri="gs://public-gigl/mocked_assets/2024-07-15--21-30-07-UTC/cora_homogeneous_node_anchor_edge_features_user_defined_labels/frozen_gbml_config.yaml",
        is_inference=True,
        _tfrecord_uri_pattern=".*tfrecord",
    )
    node_id_uri = f"{output_dir}/node_ids.pt"
    logger.info(
        f"Dumping {to_homogeneous(dataset.node_ids).numel()} node_ids to {node_id_uri}"
    )
    bytes_io = io.BytesIO()
    torch.save(to_homogeneous(dataset.node_ids), bytes_io)
    bytes_io.seek(0)
    FileLoader().load_from_filelike(UriFactory.create_uri(node_id_uri), bytes_io)
    bytes_io.close()

    remote_node_info = RemoteNodeInfo(
        node_type=None,
        edge_types=dataset.get_edge_types(),
        node_tensor_uri=node_id_uri,
        node_feature_info=dataset.node_feature_info,
        edge_feature_info=dataset.edge_feature_info,
        num_partitions=dataset.num_partitions,
        edge_dir=dataset.edge_dir,
        master_port=get_free_port(),
    )
    with open(f"{output_dir}/remote_node_info.pyast", "w") as f:
        f.write(remote_node_info.dump())
    print(f"Wrote remote node info to {output_dir}/remote_node_info.pyast")
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
    run_server(0, 1, 1, args.host, args.port, args.output_dir)
