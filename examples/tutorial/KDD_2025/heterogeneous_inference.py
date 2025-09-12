"""
Example heterogeneous inference script for GiGL.
This script demonstrates how to run inference on a heterogeneous graph model using GiGL.
It initializes a model, loads the state from a saved URI, and performs inference on the dataset.
It also exports the embeddings to a specified output URI, which can be a GCS bucket or a local directory.

Example usage:
    python -m examples.tutorial.KDD_2025.heterogeneous_inference --task_config_uri <path_to_frozen_task_config>

To generate a frozen config from a template task config, see instructions at top of `examples/tutorial/KDD_2025/task_config.yaml`.

Args:
    --task_config_uri: Path to the task config URI.
    --torch_process_group_init_method: Method to initialize the torch process group.
    --process_count: Number of processes to spawn.
    --embedding_output_uri: URI to save embeddings.
    --batch_size: Batch size for inference.
    --use_local_saved_model: Use a local saved model instead of a remote URI.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs isort: skip


import argparse
import datetime
from collections.abc import Mapping
from distutils.util import strtobool
from pathlib import Path

import fastavro
import pandas as pd
import torch
import torch.multiprocessing.spawn
from examples.tutorial.KDD_2025.utils import LOCAL_SAVED_MODEL_URI, init_model

from gigl.common import Uri, UriFactory
from gigl.common.data.export import EmbeddingExporter
from gigl.common.logger import Logger
from gigl.distributed import (
    DistDataset,
    DistNeighborLoader,
    build_dataset_from_task_config_uri,
)
from gigl.distributed.utils import get_free_port
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.model import load_state_dict_from_uri

logger = Logger()


@torch.no_grad()
def inference(
    process_number: int,
    process_count: int,
    port: int,
    dataset: DistDataset,
    embedding_output_uri: Uri,
    saved_model_uri: Uri,
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
    model = init_model()
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
        export_dir = embedding_output_uri.join(
            embedding_output_uri, f"node_{node_type}"
        )
        Path(export_dir.uri).mkdir(parents=True, exist_ok=True)
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
    logger.info(f"Finished inference on process {process_number}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a GiGL model.")
    parser.add_argument(
        "--task_config_uri",
        type=str,
        help="Path to the frozen task config URI.",
    )
    parser.add_argument(
        "--torch_process_group_init_method",
        type=str,
        default=f"tcp://localhost:{get_free_port()}?rank=0&world_size=1",
    )
    parser.add_argument(
        "--process_count", type=str, default="1", help="Number of processes to spawn"
    )
    parser.add_argument(
        "--embedding_output_uri",
        type=str,
        default=f"/tmp/gigl/embeddings/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="URI to save embeddings",
    )
    parser.add_argument(
        "--use_local_saved_model",
        type=str,
        default="False",
        help="Use a local saved model instead of a remote URI",
    )

    args, unknown = parser.parse_known_args()
    logger.info(f"Using args: {args}, unknown args: {unknown}")
    torch.distributed.init_process_group(
        backend="gloo",  # Use the Gloo backend for CPU training.
        init_method=args.torch_process_group_init_method,
    )
    # Build the dataset from the task config URI
    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        task_config_uri
    )
    dataset = build_dataset_from_task_config_uri(
        task_config_uri,
        _tfrecord_uri_pattern=".*tfrecord",
    )
    if strtobool(args.use_local_saved_model):
        saved_model_uri = LOCAL_SAVED_MODEL_URI
    else:
        saved_model_uri = (
            gbml_config_pb_wrapper.shared_config.trained_model_metadata.trained_model_uri
        )
    logger.info(f"Using saved model URI: {saved_model_uri}")
    # Spawn processes for distributed inference
    inference_port = get_free_port()
    torch.multiprocessing.spawn(
        inference,
        args=(
            int(args.process_count),  # process_count
            inference_port,  # port
            dataset,  # dataset
            UriFactory.create_uri(args.embedding_output_uri),  # embedding_output_uri
            UriFactory.create_uri(saved_model_uri),  # saved_model_uri
            gbml_config_pb_wrapper.inferencer_config.inference_batch_size,  # batch_size
        ),
        nprocs=int(args.process_count),
        join=True,
    )

    # Now let's load the embeddings to a dataframe
    # Note in a "production" setting we have `gigl.common.data.export.load_embeddings_to_bigquery`
    # to upload the embeddings to BigQuery.
    # You make use it like:
    # load_embeddings_to_bigquery(
    #     gcs_folder=args.embedding_output_uri,
    #     project_id=<your project>,
    #     dataset_id=<your dataset>,
    #     table_id=<your table>,
    # )
    avro_files = list(
        f for f in Path(args.embedding_output_uri).rglob("*.avro") if f.is_file()
    )
    # Schema here is:
    # node_id: int
    # embedding: list[float]
    # node_type: str
    avro_data: list = []
    for avro_file in avro_files:
        avro_data.extend(fastavro.reader(avro_file.open("rb")))
    print(f"First data: {avro_data[0] if avro_data else 'No data found'}")
    df = pd.DataFrame.from_records(avro_data)
    logger.info(f"Dataframe {df}.")
