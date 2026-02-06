"""
This file contains an example for how to run heterogeneous inference in **graph store mode** using GiGL.

Graph Store Mode vs Standard Mode:
----------------------------------
Graph store mode uses a heterogeneous cluster architecture with two distinct sub-clusters:
  1. **Storage Cluster (graph_store_pool)**: Dedicated machines for storing and serving the graph
     data. These are typically high-memory machines without GPUs (e.g., n2-highmem-32).
  2. **Compute Cluster (compute_pool)**: Dedicated machines for running model inference/training.
     These typically have GPUs attached (e.g., n1-standard-16 with NVIDIA_TESLA_T4).

This separation allows for:
  - Independent scaling of storage and compute resources
  - Better memory utilization (graph data stays on storage nodes)
  - Cost optimization by using appropriate hardware for each role

In contrast, the standard inference mode (see `examples/link_prediction/heterogeneous_inference.py`)
uses a homogeneous cluster where each machine handles both graph storage and computation.

Key Implementation Differences:
-------------------------------
This file (graph store mode):
  - Uses `RemoteDistDataset` to connect to a remote graph store cluster
  - Uses `init_compute_process` to initialize the compute node connection to storage
  - Obtains cluster topology via `get_graph_store_info()` which returns `GraphStoreInfo`
  - Uses `mp_sharing_dict` for efficient tensor sharing between local processes

Standard mode (`heterogeneous_inference.py`):
  - Uses `DistDataset` with `build_dataset_from_task_config_uri` where each node loads its partition
  - Manually manages distributed process groups with master IP and port
  - Each machine stores its own partition of the graph data

Resource Configuration:
-----------------------
Graph store mode requires a different resource config structure. Compare:

**Graph Store Mode** (e2e_glt_gs_resource_config.yaml):
```yaml
inferencer_resource_config:
  vertex_ai_graph_store_inferencer_config:
    graph_store_pool:
      machine_type: n2-highmem-32      # High memory for graph storage
      gpu_type: ACCELERATOR_TYPE_UNSPECIFIED
      gpu_limit: 0
      num_replicas: 2
    compute_pool:
      machine_type: n1-standard-16     # Standard machines with GPUs
      gpu_type: NVIDIA_TESLA_T4
      gpu_limit: 2
      num_replicas: 2
```

**Standard Mode** (e2e_glt_resource_config.yaml):
```yaml
inferencer_resource_config:
  vertex_ai_inferencer_config:
    machine_type: n1-highmem-32
    gpu_type: NVIDIA_TESLA_T4
    gpu_limit: 2
    num_replicas: 2
```

To run this file with GiGL orchestration, set the fields similar to below:

inferencerConfig:
  inferencerArgs:
    # Example argument to inferencer
    log_every_n_batch: "50"
  inferenceBatchSize: 512
  command: python -m examples.link_prediction.graph_store.heterogeneous_inference
  graphStoreStorageConfig:
    command: python -m examples.link_prediction.graph_store.storage_main
    storageArgs:
      sample_edge_direction: "in"
featureFlags:
  should_run_glt_backend: 'True'

Note: Ensure you use a resource config with `vertex_ai_graph_store_inferencer_config` when
running in graph store mode.

You can run this example in a full pipeline with `make run_het_dblp_sup_gs_e2e_test` from GiGL root."""

import argparse
import gc
import os
import sys
import time
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Union

import torch
import torch.distributed
import torch.multiprocessing as mp
from examples.link_prediction.models import init_example_gigl_heterogeneous_model

import gigl.distributed
import gigl.distributed.utils
from gigl.common import GcsUri, Uri, UriFactory
from gigl.common.data.export import EmbeddingExporter, load_embeddings_to_bigquery
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.distributed.graph_store.compute import (
    init_compute_process,
    shutdown_compute_proccess,
)
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.utils import get_graph_store_info
from gigl.env.distributed import GraphStoreInfo
from gigl.nn import LinkPredictionGNN
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.bq import BqUtils
from gigl.src.common.utils.model import load_state_dict_from_uri
from gigl.src.inference.lib.assets import InferenceAssets
from gigl.utils.sampling import parse_fanout

logger = Logger()


# We don't see logs for graph store mode for whatever reason.
# TOOD(#442): Revert this once the GCP issues are resolved.
def flush():
    sys.stdout.write("\n")
    sys.stdout.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()


@dataclass(frozen=True)
class InferenceProcessArgs:
    """
    Arguments for the heterogeneous inference process.

    Contains all configuration needed to run distributed inference for heterogeneous graph neural
    networks, including distributed context, data configuration, model parameters, and inference
    configuration.

    Attributes:
        local_world_size (int): Number of inference processes spawned by each machine.
        machine_rank (int): Rank of the current machine in the cluster.
        machine_world_size (int): Total number of machines in the cluster.
        cluster_info (GraphStoreInfo): Cluster topology info for graph store mode, containing
            information about storage and compute node ranks and addresses.
        inference_node_type (NodeType): Node type that embeddings should be generated for.
        mp_sharing_dict (MutableMapping[str, torch.Tensor]): Shared dictionary for efficient tensor
            sharing between local processes.
        model_state_dict_uri (Uri): URI to load the trained model state dict from.
        hid_dim (int): Hidden dimension of the model.
        out_dim (int): Output dimension of the model.
        node_type_to_feature_dim (dict[NodeType, int]): Mapping of node types to their feature
            dimensions.
        edge_type_to_feature_dim (dict[EdgeType, int]): Mapping of edge types to their feature
            dimensions.
        embedding_gcs_path (GcsUri): GCS path to write embeddings to.
        inference_batch_size (int): Batch size to use for inference.
        num_neighbors (Union[list[int], dict[EdgeType, list[int]]]): Fanout for subgraph sampling,
            where the ith item corresponds to the number of items to sample for the ith hop.
        sampling_workers_per_inference_process (int): Number of sampling workers per inference
            process.
        sampling_worker_shared_channel_size (str): Shared-memory buffer size (bytes) allocated for
            the channel during sampling (e.g., "4GB").
        log_every_n_batch (int): Frequency to log batch information during inference.
    """

    # Distributed context
    local_world_size: int
    machine_rank: int
    machine_world_size: int
    cluster_info: GraphStoreInfo

    # Data
    inference_node_type: NodeType
    mp_sharing_dict: MutableMapping[str, torch.Tensor]

    # Model
    model_state_dict_uri: Uri
    hid_dim: int
    out_dim: int
    node_type_to_feature_dim: dict[NodeType, int]
    edge_type_to_feature_dim: dict[EdgeType, int]

    # Inference config
    embedding_gcs_path: GcsUri
    inference_batch_size: int
    num_neighbors: Union[list[int], dict[EdgeType, list[int]]]
    sampling_workers_per_inference_process: int
    sampling_worker_shared_channel_size: str
    log_every_n_batch: int


@torch.no_grad()
def _inference_process(
    # When spawning processes, each process will be assigned a rank ranging
    # from [0, num_processes).
    local_rank: int,
    args: InferenceProcessArgs,
):
    """
    This function is spawned by multiple processes per machine and is responsible for:
        1. Intializing the dataLoader
        2. Running the inference loop to get the embeddings for each anchor node
        3. Writing embeddings to GCS

    Args:
        local_rank (int): Process number on the current machine
        args (InferenceProcessArgs): Dataclass containing all inference process arguments
    """

    device = gigl.distributed.utils.get_available_device(
        local_process_rank=local_rank,
    )  # The device is automatically inferred based off the local process rank and the available devices
    if torch.cuda.is_available():
        torch.cuda.set_device(
            device
        )  # Set the device for the current process. Without this, NCCL will fail when multiple GPUs are available.

    rank = args.machine_rank * args.local_world_size + local_rank
    world_size = args.machine_world_size * args.local_world_size
    # Note: This is a *critical* step in Graph Store mode. It initializes the connection to the storage cluster.
    # If this is not done, the dataloader will not be able to sample from the graph store and will crash.
    logger.info(
        f"Initializing compute process for rank {local_rank} in machine {args.machine_rank} with cluster info {args.cluster_info} for inference node type {args.inference_node_type}"
    )
    flush()
    init_compute_process(local_rank, args.cluster_info)
    dataset = RemoteDistDataset(
        args.cluster_info, local_rank, mp_sharing_dict=args.mp_sharing_dict
    )
    logger.info(
        f"Local rank {local_rank} in machine {args.machine_rank} has rank {rank}/{world_size} and using device {device} for inference"
    )

    # Get the node ids on the current machine for the current node type
    input_nodes = dataset.get_node_ids(node_type=args.inference_node_type)
    logger.info(
        f"Rank {rank} got input nodes of shapes: {[f'{rank}: {node.shape}' for rank, node in input_nodes.items()]}"
    )
    flush()
    data_loader = gigl.distributed.DistNeighborLoader(
        dataset=dataset,
        num_neighbors=args.num_neighbors,
        # We must pass in a tuple of (node_type, node_ids_on_current_process) for heterogeneous input
        input_nodes=(args.inference_node_type, input_nodes),
        num_workers=args.sampling_workers_per_inference_process,
        batch_size=args.inference_batch_size,
        pin_memory_device=device,
        worker_concurrency=args.sampling_workers_per_inference_process,
        channel_size=args.sampling_worker_shared_channel_size,
        # For large-scale settings, consider setting this field to 30-60 seconds to ensure dataloaders
        # don't compete for memory during initialization, causing OOM
        process_start_gap_seconds=0,
    )
    flush()
    # Initialize a LinkPredictionGNN model and load parameters from
    # the saved model.
    model_state_dict = load_state_dict_from_uri(
        load_from_uri=args.model_state_dict_uri, device=device
    )
    model: LinkPredictionGNN = init_example_gigl_heterogeneous_model(
        node_type_to_feature_dim=args.node_type_to_feature_dim,
        edge_type_to_feature_dim=args.edge_type_to_feature_dim,
        hid_dim=args.hid_dim,
        out_dim=args.out_dim,
        device=device,
        state_dict=model_state_dict,
    )

    # Set the model to evaluation mode for inference.
    model.eval()

    logger.info(f"Model initialized on device {device}")

    embedding_filename = (
        f"machine_{args.machine_rank}_local_process_number_{local_rank}"
    )

    # Get temporary GCS folder to write outputs of inference to. GiGL orchestration automatic cleans this, but
    # if running manually, you will need to clean this directory so that retries don't end up with stale files.
    gcs_utils = GcsUtils()
    gcs_base_uri = GcsUri.join(args.embedding_gcs_path, embedding_filename)
    num_files_at_gcs_path = gcs_utils.count_blobs_in_gcs_path(gcs_base_uri)
    if num_files_at_gcs_path > 0:
        logger.warning(
            f"{num_files_at_gcs_path} files already detected at base gcs path. Cleaning up files at path ... "
        )
        gcs_utils.delete_files_in_bucket_dir(gcs_base_uri)

    # GiGL class for exporting embeddings to GCS. This is achieved by writing ids and embeddings to an in-memory buffer which gets
    # flushed to GCS. Setting the min_shard_size_threshold_bytes field of this class sets the frequency of flushing to GCS, and defaults
    # to only flushing when flush_records() is called explicitly or after exiting via a context manager.
    exporter = EmbeddingExporter(export_dir=gcs_base_uri)

    # We add a barrier here so that all machines and processes have initialized their dataloader at the start of the inference loop. Otherwise, on-the-fly subgraph
    # sampling may fail.
    flush()
    torch.distributed.barrier()

    t = time.time()
    data_loading_start_time = time.time()
    inference_start_time = time.time()
    cumulative_data_loading_time = 0.0
    cumulative_inference_time = 0.0
    flush()

    # Begin inference loop

    # Iterating through the dataloader yields a `torch_geometric.data.Data` type
    for batch_idx, data in enumerate(data_loader):
        cumulative_data_loading_time += time.time() - data_loading_start_time

        inference_start_time = time.time()

        # These arguments to forward are specific to the GiGL heterogeneous LinkPredictionGNN model.
        # If just using a nn.Module, you can just use output = model(data)
        output = model(
            data=data, output_node_types=[args.inference_node_type], device=device
        )[args.inference_node_type]

        # The anchor node IDs are contained inside of the .batch field of the data
        node_ids = data[args.inference_node_type].batch.cpu()

        # Only the first `batch_size` rows of the node embeddings contain the embeddings of the anchor nodes
        node_embeddings = output[: data[args.inference_node_type].batch_size].cpu()

        # We add ids and embeddings to the in-memory buffer
        exporter.add_embedding(
            id_batch=node_ids,
            embedding_batch=node_embeddings,
            embedding_type=str(args.inference_node_type),
        )

        cumulative_inference_time += time.time() - inference_start_time

        if batch_idx == 0 or (
            batch_idx > 0 and batch_idx % args.log_every_n_batch == 0
        ):
            logger.info(
                f"Rank {rank} processed {batch_idx} batches for node type {args.inference_node_type}. "
                f"{args.log_every_n_batch} batches took {time.time() - t:.2f} seconds for node type {args.inference_node_type}. "
                f"Among them, data loading took {cumulative_data_loading_time:.2f} seconds."
                f"and model inference took {cumulative_inference_time:.2f} seconds."
            )
            t = time.time()
            cumulative_data_loading_time = 0
            cumulative_inference_time = 0
            flush()

        data_loading_start_time = time.time()

    logger.info(
        f"--- Rank {rank} finished inference for node type {args.inference_node_type}."
    )

    write_embedding_start_time = time.time()
    # Flushes all remaining embeddings to GCS
    exporter.flush_records()

    logger.info(
        f"--- Rank {rank} finished writing embeddings to GCS for node type {args.inference_node_type}, which took {time.time()-write_embedding_start_time:.2f} seconds"
    )

    # We first call barrier to ensure that all machines and processes have finished inference.
    # Only once all machines have finished inference is it safe to shutdown the data loader.
    # Otherwise, processes which are still sampling *will* fail as the loaders they are trying to communicatate with will be shutdown.
    # We then call `gc.collect()` to cleanup the memory used by the data_loader on the current machine.

    torch.distributed.barrier()

    data_loader.shutdown()
    shutdown_compute_proccess()
    gc.collect()

    logger.info(
        f"--- All machines local rank {local_rank} finished inference for node type {args.inference_node_type}. Deleted data loader and shutdown compute process"
    )

    flush()


def _run_example_inference(
    job_name: str,
    task_config_uri: str,
) -> None:
    """
    Runs an example inference pipeline using GiGL Orchestration.
    Args:
        job_name (str): Name of current job
        task_config_uri (str): Path to frozen GBMLConfigPbWrapper
    """
    # All machines run this logic to connect together, and return a distributed context with:
    # - the (GCP) internal IP address of the rank 0 machine, which will be used for building RPC connections.
    # - the current machine rank
    # - the total number of machines (world size)

    program_start_time = time.time()
    # The main process per machine needs to be able to talk with each other to partition and synchronize the graph data.
    # Thus, the user is responsible here for 1. spinning up a single process per machine,
    # and 2. init_process_group amongst these processes.
    # Assuming this is spinning up inside VAI; it already sets up the env:// init method for us; thus we don't need anything
    # special here.
    torch.distributed.init_process_group(backend="gloo")

    logger.info(
        f"Took {time.time() - program_start_time:.2f} seconds to connect worker pool"
    )
    logger.info(
        f"World size: {torch.distributed.get_world_size()}, rank: {torch.distributed.get_rank()}, OS world size: {os.environ['WORLD_SIZE']}, OS rank: {os.environ['RANK']}"
    )

    cluster_info = get_graph_store_info()
    logger.info(f"Cluster info: {cluster_info}")
    torch.distributed.destroy_process_group()
    logger.info(
        f"Took {time.time() - program_start_time:.2f} seconds to connect worker pool"
    )
    flush()

    # Read from GbmlConfig for preprocessed data metadata, GNN model uri, and bigquery embedding table path, and additional inference args
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=UriFactory.create_uri(task_config_uri)
    )

    model_uri = UriFactory.create_uri(
        gbml_config_pb_wrapper.gbml_config_pb.shared_config.trained_model_metadata.trained_model_uri
    )

    graph_metadata = gbml_config_pb_wrapper.graph_metadata_pb_wrapper

    node_type_to_feature_dim: dict[NodeType, int] = {
        graph_metadata.condensed_node_type_to_node_type_map[
            condensed_node_type
        ]: node_feature_dim
        for condensed_node_type, node_feature_dim in gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_dim_map.items()
    }

    edge_type_to_feature_dim: dict[EdgeType, int] = {
        graph_metadata.condensed_edge_type_to_edge_type_map[
            condensed_edge_type
        ]: edge_feature_dim
        for condensed_edge_type, edge_feature_dim in gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper.condensed_edge_type_to_feature_dim_map.items()
    }

    inference_node_types = sorted(
        gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_task_root_node_types()
    )

    inferencer_args = dict(gbml_config_pb_wrapper.inferencer_config.inferencer_args)

    inference_batch_size = gbml_config_pb_wrapper.inferencer_config.inference_batch_size

    hid_dim = int(inferencer_args.get("hid_dim", "16"))
    out_dim = int(inferencer_args.get("out_dim", "16"))

    if torch.cuda.is_available():
        default_num_inference_processes_per_machine = torch.cuda.device_count()
    else:
        default_num_inference_processes_per_machine = 2
    num_inference_processes_per_machine = int(
        inferencer_args.get(
            "num_inference_processes_per_machine",
            default_num_inference_processes_per_machine,
        )
    )  # Current large-scale setting sets this value to 4

    if (
        torch.cuda.is_available()
        and num_inference_processes_per_machine > torch.cuda.device_count()
    ):
        raise ValueError(
            f"Number of inference processes per machine ({num_inference_processes_per_machine}) must not be more than the number of GPUs: ({torch.cuda.device_count()})"
        )
    flush()

    ## Inference Start
    flush()
    inference_start_time = time.time()

    for process_num, inference_node_type in enumerate(inference_node_types):
        logger.info(
            f"Starting inference process for node type {inference_node_type} ..."
        )
        output_bq_table_path = InferenceAssets.get_enumerated_embedding_table_path(
            gbml_config_pb_wrapper, inference_node_type
        )

        bq_project_id, bq_dataset_id, bq_table_name = BqUtils.parse_bq_table_path(
            bq_table_path=output_bq_table_path
        )

        # We write embeddings to a temporary GCS path during the inference loop, since writing directly to bigquery for each embedding is slow.
        # After inference has finished, we then load all embeddings to bigquery from GCS.
        embedding_output_gcs_folder = InferenceAssets.get_gcs_asset_write_path_prefix(
            applied_task_identifier=AppliedTaskIdentifier(job_name),
            bq_table_path=output_bq_table_path,
        )

        # Parses the fanout as a string. For the heterogeneous case, the fanouts can be specified
        # as a string of a list of integers, such as "[10, 10]", which will apply this fanout to
        # each edge type in the graph, or as string of format dict[(tuple[str, str, str])),
        # list[int]] which will specify fanouts per edge type. In the case of the latter, the keys
        # should be specified with format (SRC_NODE_TYPE, RELATION, DST_NODE_TYPE). For the default
        # example, we make a decision to keep the fanouts for all edge types the same, specifying
        # the `fanout` with a `list[int]`. To see an example of a 'fanout' with different behaviors
        # per edge type, refer to `examples/link_prediction/graph_store/configs/e2e_het_dblp_sup_gs_task_config.yaml`.
        num_neighbors = parse_fanout(inferencer_args.get("num_neighbors", "[10, 10]"))

        # While the ideal value for `sampling_workers_per_inference_process` has been identified to
        # be between `2` and `4`, this may need some tuning depending on the pipeline. We default
        # this value to `4` here for simplicity. A `sampling_workers_per_process` which is too
        # small may not have enough parallelization for sampling, which would slow down inference,
        # while a value which is too large may slow down each sampling process due to competing
        # resources, which would also then slow down inference.
        sampling_workers_per_inference_process = int(
            inferencer_args.get("sampling_workers_per_inference_process", "4")
        )

        # This value represents the shared-memory buffer size (bytes) allocated for the channel
        # during sampling, and is the place to store pre-fetched data, so if it is too small then
        # prefetching is limited, causing sampling slowdown. This parameter is a string with
        # `{numeric_value}{storage_size}`, where storage size could be `MB`, `GB`, etc. We default
        # this value to 4GB, but in production may need some tuning.
        sampling_worker_shared_channel_size = inferencer_args.get(
            "sampling_worker_shared_channel_size", "4GB"
        )

        log_every_n_batch = int(inferencer_args.get("log_every_n_batch", "50"))

        # When using mp.spawn with `nprocs`, the first argument is implicitly set to be the process number on the current machine.
        inference_args = InferenceProcessArgs(
            local_world_size=num_inference_processes_per_machine,
            machine_rank=cluster_info.compute_node_rank,
            machine_world_size=cluster_info.num_compute_nodes,
            cluster_info=cluster_info,
            inference_node_type=inference_node_type,
            mp_sharing_dict=torch.multiprocessing.Manager().dict(),
            model_state_dict_uri=model_uri,
            hid_dim=hid_dim,
            out_dim=out_dim,
            node_type_to_feature_dim=node_type_to_feature_dim,
            edge_type_to_feature_dim=edge_type_to_feature_dim,
            embedding_gcs_path=embedding_output_gcs_folder,
            inference_batch_size=inference_batch_size,
            num_neighbors=num_neighbors,
            sampling_workers_per_inference_process=sampling_workers_per_inference_process,
            sampling_worker_shared_channel_size=sampling_worker_shared_channel_size,
            log_every_n_batch=log_every_n_batch,
        )
        logger.info(
            f"Rank {cluster_info.compute_node_rank} started inference process for node type {inference_node_type} with {num_inference_processes_per_machine} processes\nargs: {inference_args}"
        )
        flush()

        mp.spawn(
            fn=_inference_process,
            args=(inference_args,),
            nprocs=num_inference_processes_per_machine,
            join=True,
        )

        logger.info(
            f"--- Inference finished on rank {cluster_info.compute_node_rank} for node type {inference_node_type}, which took {time.time()-inference_start_time:.2f} seconds"
        )
        flush()

        # After inference is finished, we use the process on the Machine 0 to load embeddings from GCS to BQ.
        if cluster_info.compute_node_rank == 0:
            logger.info(
                f"--- Machine 0 triggers loading embeddings from GCS to BigQuery for node type {inference_node_type}"
            )
            # If we are on the last inference process, we should wait for this last write process to complete. Otherwise, we should
            # load embeddings to bigquery in the background so that we are not blocking the start of the next inference process
            should_run_async = process_num != len(inference_node_types) - 1

            # The `load_embeddings_to_bigquery` API returns a BigQuery LoadJob object
            # representing the load operation, which allows user to monitor and retrieve
            # details about the job status and result.
            _ = load_embeddings_to_bigquery(
                gcs_folder=embedding_output_gcs_folder,
                project_id=bq_project_id,
                dataset_id=bq_dataset_id,
                table_id=bq_table_name,
                should_run_async=should_run_async,
            )
            flush()
    logger.info(
        f"--- Program finished, which took {time.time()-program_start_time:.2f} seconds"
    )


if __name__ == "__main__":
    # TODO(#442): Revert this once the GCP issues are resolved.
    # Per the GCP folks this try/except may help - though in practice it seems to not.
    try:
        parser = argparse.ArgumentParser(
            description="Arguments for distributed model inference on VertexAI"
        )
        parser.add_argument(
            "--job_name",
            type=str,
            help="Inference job name",
        )
        parser.add_argument("--task_config_uri", type=str, help="Gbml config uri")

        # We use parse_known_args instead of parse_args since we only need job_name and task_config_uri for distributed inference
        args, unused_args = parser.parse_known_args()
        logger.info(f"Args: {args}, Unused arguments: {unused_args}")
        flush()

        # We only need `job_name` and `task_config_uri` for running inference
        _run_example_inference(
            job_name=args.job_name,
            task_config_uri=args.task_config_uri,
        )
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.flush()
        raise e
    finally:
        # Note that `print` logs more reliably due to a Vertex AI bug.
        # TODO(#442): Revert this once the GCP issues are resolved.
        print("Finally block")
        print("flush stdout")
        sys.stdout.flush()
        print("flush stderr")
        sys.stderr.flush()
