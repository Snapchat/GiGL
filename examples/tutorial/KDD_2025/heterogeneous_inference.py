"""
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs isort: skip


import argparse
from collections.abc import Mapping

import torch
import torch.multiprocessing.spawn
from torch_geometric.nn import HGTConv

from gigl.common import GcsUri, Uri, UriFactory
from gigl.common.data.export import EmbeddingExporter, load_embeddings_to_bigquery
from gigl.common.logger import Logger
from gigl.distributed import (
    DistLinkPredictionDataset,
    DistNeighborLoader,
    build_dataset_from_task_config_uri,
)
from gigl.distributed.utils import get_free_port
from gigl.src.common.utils.model import load_state_dict_from_uri

logger = Logger()


@torch.no_grad()
def inference(
    process_number: int,
    process_count: int,
    port: int,
    dataset: DistLinkPredictionDataset,
    embedding_output_uri: GcsUri,
    saved_model_uri: Uri,
    project_id: str,
    dataset_id: str,
    table_id: str,
    batch_size: int = 4,
):
    """Run inference on the model."""
    logger.info(f"Starting inference on process {process_number} of {process_count}")

    # Initialize the distributed environment
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        world_size=process_count,
        rank=process_number,
    )

    # Create the model
    model = HGTConv(
        in_channels=-1,
        out_channels=16,
        metadata=[
            tuple(dataset.get_node_types()),
            tuple(dataset.get_edge_types()),
        ],
    )
    # Load the model state
    model.load_state_dict(load_state_dict_from_uri(saved_model_uri))
    logger.info(
        f"Used saved model at {saved_model_uri} to initialize the model {model}."
    )
    model.eval()

    for node_type in dataset.get_node_types():
        # Create a data loader
        logger.info(f"Process {process_number} is processing node type {node_type}.")
        assert isinstance(dataset.node_ids, Mapping)
        loader = DistNeighborLoader(
            dataset,
            num_neighbors=[2, 2],
            input_nodes=(node_type, dataset.node_ids[node_type]),
            batch_size=batch_size,
            shuffle=False,
            process_start_gap_seconds=0,
            pin_memory_device=torch.device("cpu"),
        )
        export_dir = GcsUri.join(embedding_output_uri, f"node_{node_type}")
        exporter = EmbeddingExporter(
            export_dir, file_prefix=f"embeddings_{process_number}_"
        )
        for batch in loader:
            embeddings = model(batch.x_dict, batch.edge_index_dict)
            # Save embeddings to the exporter
            exporter.add_embedding(
                id_batch=batch[node_type].batch,
                embedding_batch=embeddings[node_type],
                embedding_type=f"node_{node_type}",
            )
        exporter.flush_embeddings()
        torch.distributed.barrier()  # Wait for all ranks to finish exporting embeddings
        if process_number == 0:
            logger.info(
                f"Exported embeddings for node type {node_type} to BQ {project_id}.{dataset_id}.{table_id}."
            )
            load_embeddings_to_bigquery(
                gcs_folder=export_dir,
                project_id=project_id,
                dataset_id=dataset_id,
                table_id=table_id,
            )
    logger.info(f"Finished inference on process {process_number}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a GiGL model.")
    parser.add_argument(
        "--task_config_uri",
        type=str,
        default="examples/tutorial/KDD_2025/toy_graph_task_config.yaml",
        help="Path to the task config URI.",
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host for distributed inference."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=get_free_port(),
        help="Port for distributed communication",
    )
    parser.add_argument(
        "--process_count", type=int, default=1, help="Number of processes to spawn"
    )
    parser.add_argument(
        "--embedding_output_uri",
        type=str,
        required=True,
        help="GCS URI to save embeddings",
    )
    parser.add_argument(
        "--saved_model_uri",
        type=str,
        default="/tmp/gigl/toy_hgt_model.pt",
        help="URI of the saved model",
    )
    parser.add_argument(
        "--project_id", type=str, required=True, help="GCP project ID for BigQuery"
    )
    parser.add_argument(
        "--dataset_id", type=str, required=True, help="BigQuery dataset ID"
    )
    parser.add_argument("--table_id", type=str, required=True, help="BigQuery table ID")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for inference"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of the process (for distributed training).",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of nodes in the training cluster.",
    )

    args = parser.parse_args()
    torch.distributed.init_process_group(
        backend="gloo",  # Use the Gloo backend for CPU training.
        init_method=f"tcp://{args.host}:{args.port}",
        rank=args.rank,
        world_size=args.world_size,
    )
    # Build the dataset from the task config URI
    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    dataset = build_dataset_from_task_config_uri(
        task_config_uri,
        _tfrecord_uri_pattern=".*tfrecord",
    )

    # Spawn processes for distributed inference
    inference_port = get_free_port()
    torch.multiprocessing.spawn(
        inference,
        args=(
            args.process_count,  # process_count
            inference_port,  # port
            dataset,  # dataset
            UriFactory.create_uri(args.embedding_output_uri),  # embedding_output_uri
            UriFactory.create_uri(args.saved_model_uri),  # saved_model_uri
            args.project_id,  # project_id
            args.dataset_id,  # dataset_id
            args.table_id,  # table_id
            args.batch_size,  # batch_size
        ),
        nprocs=args.process_count,
        join=True,
    )
