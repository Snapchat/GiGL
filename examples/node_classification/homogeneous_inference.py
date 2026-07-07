"""
This file contains an example for how to run homogeneous supervised node classification (SNC)
inference using GiGL's GraphLearn-for-PyTorch (GLT) bindings. The example exports the per-anchor
argmax class label to a temporary GCS folder, then loads it into a BigQuery table at the end of
the run. While `_run_example_inference` is coupled with GiGL orchestration, the
`_inference_process` function is generic and can be used as a reference for writing inference for
pipelines not dependent on GiGL orchestration.

To run this file with GiGL orchestration, set the fields similar to below:

inferencerConfig:
  inferencerArgs:
    # Example argument to inferencer
    log_every_n_batch: "25"
  inferenceBatchSize: 512
  command: python -m examples.node_classification.homogeneous_inference
featureFlags:
  should_run_glt_backend: 'True'
  # Disable embeddings-path population so the post-processor's unenumerator only expects the
  # predictions table this example actually writes. NODE_BASED_TASK tasks unconditionally
  # populate `predictions_path` already.
  should_populate_embeddings_path: 'False'

You can run this example in a full pipeline with `make run_hom_cora_snc_e2e_test`
from GiGL root.

Each anchor node is exported as a single `FLOAT64` `pred` field (the predicted class label, i.e.
`argmax(logits)` cast to float) into BigQuery via GiGL's `PredictionExporter`.
"""

import argparse
import gc
import time
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp

import gigl.distributed
import gigl.distributed.utils
from examples.node_classification.models import (
    init_example_gigl_homogeneous_node_classification_model,
)
from gigl.common import GcsUri, Uri, UriFactory
from gigl.common.data.export import PredictionExporter, load_predictions_to_bigquery
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.distributed import DistDataset, build_dataset_from_task_config_uri
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.bq import BqUtils
from gigl.src.common.utils.model import load_state_dict_from_uri
from gigl.src.inference.lib.assets import InferenceAssets
from gigl.utils.sampling import parse_fanout

logger = Logger()

# Default number of inference processes per machine when one isn't provided via
# `local_world_size` in inferencer args and there are no GPUs available.
DEFAULT_CPU_BASED_LOCAL_WORLD_SIZE = 4


@dataclass(frozen=True)
class InferenceProcessArgs:
    """
    Arguments for the homogeneous SNC inference process.

    Attributes:
        local_world_size (int): Number of inference processes spawned by each machine.
        machine_rank (int): Rank of the current machine in the cluster.
        machine_world_size (int): Total number of machines in the cluster.
        master_ip_address (str): IP address of the master node for process group initialization.
        master_default_process_group_port (int): Port for the default process group.
        dataset (DistDataset): Loaded Distributed Dataset for inference.
        inference_node_type (NodeType): Node type that predicted class labels are generated for.
        model_uri (Uri): URI to load the trained model state dict from.
        hid_dim (int): Encoder hidden dimension.
        num_classes (int): Number of output classes.
        node_feature_dim (int): Input node feature dimension.
        prediction_gcs_path (GcsUri): GCS path to write predicted class labels to.
        inference_batch_size (int): Batch size to use for inference.
        num_neighbors (list[int]): Fanout for subgraph sampling.
        sampling_workers_per_process (int): Sampling workers per inference process.
        sampling_worker_shared_channel_size (str): Shared-memory buffer size (e.g. ``"4GB"``).
        log_every_n_batch (int): Frequency to log batch information during inference.
    """

    local_world_size: int
    machine_rank: int
    machine_world_size: int
    master_ip_address: str
    master_default_process_group_port: int

    dataset: DistDataset
    inference_node_type: NodeType

    model_uri: Uri
    hid_dim: int
    num_classes: int
    node_feature_dim: int

    prediction_gcs_path: GcsUri
    inference_batch_size: int
    num_neighbors: list[int] | dict[EdgeType, list[int]]
    sampling_workers_per_process: int
    sampling_worker_shared_channel_size: str
    log_every_n_batch: int


@torch.no_grad()
def _inference_process(
    local_rank: int,
    args: InferenceProcessArgs,
) -> None:
    """
    Spawned per local rank: initializes the dataloader, runs the inference loop, and writes
    the per-anchor predicted class label to GCS.

    Args:
        local_rank (int): Process number on the current machine.
        args (InferenceProcessArgs): Dataclass containing all inference arguments.
    """
    # The device is automatically inferred based off the local process rank and the available devices.
    device = gigl.distributed.utils.get_available_device(
        local_process_rank=local_rank,
    )
    if torch.cuda.is_available():
        # Set the device for the current process. Without this, NCCL will fail when multiple GPUs are available.
        torch.cuda.set_device(device)

    torch.distributed.init_process_group(
        backend="gloo" if device.type == "cpu" else "nccl",
        init_method=f"tcp://{args.master_ip_address}:{args.master_default_process_group_port}",
        rank=args.machine_rank * args.local_world_size + local_rank,
        world_size=args.machine_world_size * args.local_world_size,
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    logger.info(
        f"Local rank {local_rank} in machine {args.machine_rank} has rank {rank}/{world_size} and is using device {device}"
    )

    data_loader = gigl.distributed.DistNeighborLoader(
        dataset=args.dataset,
        num_neighbors=args.num_neighbors,
        input_nodes=None,  # Homogeneous case: `None` defaults to using all nodes for inference.
        num_workers=args.sampling_workers_per_process,
        batch_size=args.inference_batch_size,
        pin_memory_device=device,
        worker_concurrency=args.sampling_workers_per_process,
        channel_size=args.sampling_worker_shared_channel_size,
        process_start_gap_seconds=0,
    )

    model_state_dict = load_state_dict_from_uri(
        load_from_uri=args.model_uri, device=device
    )
    model = init_example_gigl_homogeneous_node_classification_model(
        node_feature_dim=args.node_feature_dim,
        num_classes=args.num_classes,
        hid_dim=args.hid_dim,
        device=device,
        state_dict=model_state_dict,
    )
    model.eval()

    logger.info(f"Model initialized on device {device}")

    output_filename = f"machine_{args.machine_rank}_local_process_{local_rank}"

    # Clean any stale files at the destination — GiGL orchestration cleans this automatically,
    # but a local retry would otherwise leave stale files.
    gcs_utils = GcsUtils()
    gcs_base_uri = GcsUri.join(args.prediction_gcs_path, output_filename)
    num_files_at_gcs_path = gcs_utils.count_blobs_in_gcs_path(gcs_base_uri)
    if num_files_at_gcs_path > 0:
        logger.warning(
            f"{num_files_at_gcs_path} files already detected at base gcs path. "
            f"Cleaning up files at path ... "
        )
        gcs_utils.delete_files_in_bucket_dir(gcs_base_uri)

    # The BigQuery predictions schema stores a single FLOAT64 `pred` per node
    # (see gigl/common/data/export.py:67); for multi-class classification we write
    # `argmax(logits)` cast to float.
    exporter = PredictionExporter(export_dir=gcs_base_uri)

    # Barrier so all processes have initialized their dataloader before the inference loop starts;
    # otherwise on-the-fly subgraph sampling can fail.
    torch.distributed.barrier()

    t = time.time()
    data_loading_start_time = time.time()
    cumulative_data_loading_time = 0.0
    cumulative_inference_time = 0.0

    for batch_idx, data in enumerate(data_loader):
        cumulative_data_loading_time += time.time() - data_loading_start_time

        inference_start_time = time.time()

        logits = model(data=data, device=device)
        anchor_logits = logits[: data.batch_size]
        anchor_predictions = anchor_logits.argmax(dim=-1).float().cpu()
        node_ids = data.batch.cpu()

        exporter.add_prediction(
            id_batch=node_ids,
            prediction_batch=anchor_predictions,
            prediction_type=str(args.inference_node_type),
        )

        cumulative_inference_time += time.time() - inference_start_time

        if batch_idx > 0 and batch_idx % args.log_every_n_batch == 0:
            logger.info(
                f"rank {rank} processed {batch_idx} batches. "
                f"{args.log_every_n_batch} batches took {time.time() - t:.2f} seconds. "
                f"Among them, data loading took {cumulative_data_loading_time:.2f} seconds "
                f"and model inference took {cumulative_inference_time:.2f} seconds."
            )
            t = time.time()
            cumulative_data_loading_time = 0
            cumulative_inference_time = 0

        data_loading_start_time = time.time()

    logger.info(f"--- Rank {rank} finished inference.")

    write_start_time = time.time()
    exporter.flush_records()
    logger.info(
        f"--- Rank {rank} finished writing predictions to GCS, "
        f"which took {time.time() - write_start_time:.2f} seconds"
    )

    # Barrier before shutting down so all processes finish sampling first; otherwise still-active
    # samplers will fail when their peer's loader exits.
    torch.distributed.barrier()

    data_loader.shutdown()
    gc.collect()
    torch.distributed.destroy_process_group()

    logger.info(
        f"--- All machines local rank {local_rank} finished inference. Deleted data loader"
    )


def _run_example_inference(
    job_name: str,
    task_config_uri: str,
) -> None:
    """
    Runs an example SNC inference pipeline using GiGL Orchestration.

    Args:
        job_name (str): Name of current job.
        task_config_uri (str): Path to frozen GbmlConfig URI.
    """
    program_start_time = time.time()
    mp.set_start_method("spawn")
    logger.info(f"Starting sub process method: {mp.get_start_method()}")

    # One main process per machine needs to coordinate partitioning + synchronization; assuming
    # spawn-via-Vertex sets up env:// init for us.
    torch.distributed.init_process_group(backend="gloo")

    logger.info(
        f"Took {time.time() - program_start_time:.2f} seconds to connect worker pool"
    )

    dataset = build_dataset_from_task_config_uri(task_config_uri=task_config_uri)

    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=UriFactory.create_uri(task_config_uri)
    )
    # `model_uri` is read from the same `trained_model_metadata.trained_model_uri` slot
    # the trainer wrote to (see homogeneous_training.py `model_uri` derivation).
    model_uri = UriFactory.create_uri(
        gbml_config_pb_wrapper.gbml_config_pb.shared_config.trained_model_metadata.trained_model_uri
    )
    graph_metadata = gbml_config_pb_wrapper.graph_metadata_pb_wrapper
    output_bq_table_path = InferenceAssets.get_enumerated_predictions_table_path(
        gbml_config_pb_wrapper, graph_metadata.homogeneous_node_type
    )
    bq_project_id, bq_dataset_id, bq_table_name = BqUtils.parse_bq_table_path(
        bq_table_path=output_bq_table_path
    )
    # Write to a temporary GCS folder during the inference loop, then load to BigQuery at the end.
    prediction_output_gcs_folder = InferenceAssets.get_gcs_asset_write_path_prefix(
        applied_task_identifier=AppliedTaskIdentifier(job_name),
        bq_table_path=output_bq_table_path,
    )
    node_feature_dim = gbml_config_pb_wrapper.node_type_to_feature_dim_map[
        graph_metadata.homogeneous_node_type
    ]

    inferencer_args = dict(gbml_config_pb_wrapper.inferencer_config.inferencer_args)
    # `inference_batch_size` lives on the dedicated proto field, NOT inside the `inferencer_args`
    # string map.
    inference_batch_size = gbml_config_pb_wrapper.inferencer_config.inference_batch_size

    hid_dim = int(inferencer_args.get("hid_dim", "16"))
    num_classes = int(inferencer_args.get("num_classes", "7"))

    arg_local_world_size = inferencer_args.get("local_world_size")
    if arg_local_world_size is not None:
        local_world_size = int(arg_local_world_size)
        logger.info(f"Using local_world_size from inferencer_args: {local_world_size}")
    elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
        local_world_size = torch.cuda.device_count()
        logger.info(
            f"Detected {local_world_size} GPUs. Setting local_world_size to {local_world_size}"
        )
    else:
        logger.info(
            f"No GPUs detected. Setting local_world_size to "
            f"`{DEFAULT_CPU_BASED_LOCAL_WORLD_SIZE}`"
        )
        local_world_size = DEFAULT_CPU_BASED_LOCAL_WORLD_SIZE

    if torch.cuda.is_available() and local_world_size > torch.cuda.device_count():
        raise ValueError(
            f"Specified a local world size of {local_world_size} which exceeds the "
            f"number of devices {torch.cuda.device_count()}"
        )

    master_ip_address = gigl.distributed.utils.get_internal_ip_from_master_node()
    machine_rank = torch.distributed.get_rank()
    machine_world_size = torch.distributed.get_world_size()
    master_default_process_group_port = (
        gigl.distributed.utils.get_free_ports_from_master_node(num_ports=1)[0]
    )
    torch.distributed.destroy_process_group()

    inference_start_time = time.time()

    num_neighbors = parse_fanout(inferencer_args.get("num_neighbors", "[10, 10]"))

    sampling_workers_per_process = int(
        inferencer_args.get("sampling_workers_per_process", "4")
    )

    sampling_worker_shared_channel_size = inferencer_args.get(
        "sampling_worker_shared_channel_size", "4GB"
    )

    log_every_n_batch = int(inferencer_args.get("log_every_n_batch", "25"))

    inference_args = InferenceProcessArgs(
        local_world_size=local_world_size,
        machine_rank=machine_rank,
        machine_world_size=machine_world_size,
        master_ip_address=master_ip_address,
        master_default_process_group_port=master_default_process_group_port,
        dataset=dataset,
        inference_node_type=graph_metadata.homogeneous_node_type,
        model_uri=model_uri,
        hid_dim=hid_dim,
        num_classes=num_classes,
        node_feature_dim=node_feature_dim,
        prediction_gcs_path=prediction_output_gcs_folder,
        inference_batch_size=inference_batch_size,
        num_neighbors=num_neighbors,
        sampling_workers_per_process=sampling_workers_per_process,
        sampling_worker_shared_channel_size=sampling_worker_shared_channel_size,
        log_every_n_batch=log_every_n_batch,
    )

    mp.spawn(
        fn=_inference_process,
        args=(inference_args,),
        nprocs=local_world_size,
        join=True,
    )

    logger.info(
        f"--- Inference finished on rank {machine_rank}, which took "
        f"{time.time() - inference_start_time:.2f} seconds"
    )

    # Machine 0 loads the per-rank GCS shards into BigQuery.
    if machine_rank == 0:
        logger.info("--- Machine 0 triggers loading predictions from GCS to BigQuery")
        _ = load_predictions_to_bigquery(
            gcs_folder=prediction_output_gcs_folder,
            project_id=bq_project_id,
            dataset_id=bq_dataset_id,
            table_id=bq_table_name,
        )

    logger.info(
        f"--- Program finished, which took {time.time() - program_start_time:.2f} seconds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for distributed SNC model inference on VertexAI"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="Inference job name",
    )
    parser.add_argument("--task_config_uri", type=str, help="Gbml config uri")

    # parse_known_args is required because Vertex AI's gigl/src/common/vertex_ai_launcher.py
    # appends extra runtime flags (e.g. --resource_config_uri, --use_cuda) that this module
    # does not declare.
    args, unused_args = parser.parse_known_args()
    logger.info(f"Unused arguments: {unused_args}")

    _run_example_inference(
        job_name=args.job_name,
        task_config_uri=args.task_config_uri,
    )
