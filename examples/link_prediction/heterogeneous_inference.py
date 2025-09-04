"""
This file contains an example for how to run heterogeneous inference on pretrained torch.nn.Module in GiGL (or elsewhere) using new
GLT (GraphLearn-for-PyTorch) bindings that GiGL has. Note that example should be applied to use cases which already have
some pretrained `nn.Module` and are looking to utilize cost-savings with distributed inference. While `run_example_inference` is coupled with
GiGL orchestration, the `_inference_process` function is generic and can be used as references
for writing inference for pipelines not dependent on GiGL orchestration.

To run this file with GiGL orchestration, set the fields similar to below:

inferencerConfig:
  inferencerArgs:
    # Example argument to inferencer
    log_every_n_batch: "50"
  inferenceBatchSize: 512
  command: python -m examples.link_prediction.heterogeneous_inference
featureFlags:
  should_run_glt_backend: 'True'

You can run this example in a full pipeline with `make run_het_dblp_sup_test` from GiGL root.
"""

import argparse
import gc
import time
from typing import Optional, Union

import torch
import torch.distributed
import torch.multiprocessing as mp
from examples.link_prediction.models import init_example_gigl_heterogeneous_model
from graphlearn_torch.distributed import barrier, shutdown_rpc

import gigl.distributed
import gigl.distributed.utils
from gigl.common import GcsUri, UriFactory
from gigl.common.data.export import EmbeddingExporter, load_embeddings_to_bigquery
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.distributed import DistDataset, build_dataset_from_task_config_uri
from gigl.module.models import LinkPredictionGNN
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.bq import BqUtils
from gigl.src.common.utils.model import load_state_dict_from_uri
from gigl.src.inference.lib.assets import InferenceAssets
from gigl.utils.sampling import parse_fanout

logger = Logger()


@torch.no_grad()
def _inference_process(
    # When spawning processes, each process will be assigned a rank ranging
    # from [0, num_processes).
    local_rank: int,
    local_world_size: int,
    machine_rank: int,
    machine_world_size: int,
    master_ip_address: str,
    master_default_process_group_port: int,
    embedding_gcs_path: GcsUri,
    model_state_dict_uri: GcsUri,
    inference_batch_size: int,
    hid_dim: int,
    out_dim: int,
    dataset: DistDataset,
    inferencer_args: dict[str, str],
    inference_node_type: NodeType,
    node_type_to_feature_dim: dict[NodeType, int],
    edge_type_to_feature_dim: dict[EdgeType, int],
):
    """
    This function is spawned by multiple processes per machine and is responsible for:
        1. Intializing the dataLoader
        2. Running the inference loop to get the embeddings for each anchor node
        3. Writing embeddings to GCS

    Args:
        local_rank (int): Process number on the current machine
        local_world_size (int): Number of inference processes spawned by each machine
        machine_rank (int): Machine number in the distributed setup
        machine_world_size (int): Total number of machines in the distributed setup
        master_ip_address (str): IP address of the master node in the distributed setup
        master_default_process_group_port (int): Port on the master node in the distributed setup to setup Torch process group on
        embedding_gcs_path (GcsUri): GCS path to load embeddings from
        model_state_dict_uri (GcsUri): GCS path to load model from
        inference_batch_size (int): Batch size to use for inference
        hid_dim (int): Hidden dimension of the model
        out_dim (int): Output dimension of the model
        dataset (DistDataset): Loaded Distributed Dataset for inference
        inferencer_args (dict[str, str]): Additional arguments for inferencer
        inference_node_type (NodeType): Node Type that embeddings should be generated for in current inference process. This is used to
            tag the embeddings written to GCS.
        node_type_to_feature_dim (dict[NodeType, int]): Input node feature dimension per node type for the model
        edge_type_to_feature_dim (dict[EdgeType, int]): Input edge feature dimension per edge type for the model
    """

    # Parses the fanout as a string.
    # For the heterogeneous case, the fanouts can be specified as a string of a list of integers, such as "[10, 10]", which will apply this fanout
    # to each edge type in the graph, or as string of format dict[(tuple[str, str, str])), list[int]] which will specify fanouts per edge type.
    # In the case of the latter, the keys should be specified with format (SRC_NODE_TYPE, RELATION, DST_NODE_TYPE).
    # For the default example, we make a decision to keep the fanouts for all edge types the same, specifying the `fanout` with a `list[int]`.
    # To see an example of a 'fanout' with different behaviors per edge type, refer to `examples/link_prediction.configs/e2e_het_dblp_sup_task_config.yaml`.

    fanout = inferencer_args.get("num_neighbors", "[10, 10]")
    num_neighbors = parse_fanout(fanout)

    # While the ideal value for `sampling_workers_per_inference_process` has been identified to be between `2` and `4`, this may need some tuning depending on the
    # pipeline. We default this value to `4` here for simplicity. A `sampling_workers_per_process` which is too small may not have enough parallelization for
    # sampling, which would slow down inference, while a value which is too large may slow down each sampling process due to competing resources, which would also
    # then slow down inference.
    sampling_workers_per_inference_process: int = int(
        inferencer_args.get("sampling_workers_per_inference_process", "4")
    )

    # This value represents the the shared-memory buffer size (bytes) allocated for the channel during sampling, and
    # is the place to store pre-fetched data, so if it is too small then prefetching is limited, causing sampling slowdown. This parameter is a string
    # with `{numeric_value}{storage_size}`, where storage size could be `MB`, `GB`, etc. We default this value to 4GB,
    # but in production may need some tuning.
    sampling_worker_shared_channel_size: str = inferencer_args.get(
        "sampling_worker_shared_channel_size", "4GB"
    )

    log_every_n_batch = int(inferencer_args.get("log_every_n_batch", "50"))
    device = gigl.distributed.utils.get_available_device(
        local_process_rank=local_rank,
    )  # The device is automatically inferred based off the local process rank and the available devices
    rank = machine_rank * local_world_size + local_rank
    world_size = machine_world_size * local_world_size
    torch.cuda.set_device(
        device
    )  # Set the device for the current process. Without this, NCCL will fail when multiple GPUs are available.
    torch.distributed.init_process_group(
        backend="gloo" if device.type == "cpu" else "nccl",
        init_method=f"tcp://{master_ip_address}:{master_default_process_group_port}",
        rank=rank,
        world_size=world_size,
    )
    logger.info(
        f"Local rank {local_rank} in machine {machine_rank} has rank {rank}/{world_size} and using device {device} for inference"
    )

    # Get the node ids on the current machine for the current node type
    node_type_to_input_node_ids: Optional[
        Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    ] = dataset.node_ids
    assert isinstance(
        node_type_to_input_node_ids, dict
    ), f"Node IDs must be a dictionary for heterogeneous inference, got {type(node_type_to_input_node_ids)}"
    input_node_ids: torch.Tensor = node_type_to_input_node_ids[inference_node_type]

    data_loader = gigl.distributed.DistNeighborLoader(
        dataset=dataset,
        num_neighbors=num_neighbors,
        # We must pass in a tuple of (node_type, node_ids_on_current_process) for heterogeneous input
        input_nodes=(inference_node_type, input_node_ids),
        num_workers=sampling_workers_per_inference_process,
        batch_size=inference_batch_size,
        pin_memory_device=device,
        worker_concurrency=sampling_workers_per_inference_process,
        channel_size=sampling_worker_shared_channel_size,
        # For large-scale settings, consider setting this field to 30-60 seconds to ensure dataloaders
        # don't compete for memory during initialization, causing OOM
        process_start_gap_seconds=0,
    )
    # Initialize a LinkPredictionGNN model and load parameters from
    # the saved model.
    model_state_dict = load_state_dict_from_uri(
        load_from_uri=model_state_dict_uri, device=device
    )
    model: LinkPredictionGNN = init_example_gigl_heterogeneous_model(
        node_type_to_feature_dim=node_type_to_feature_dim,
        edge_type_to_feature_dim=edge_type_to_feature_dim,
        hid_dim=hid_dim,
        out_dim=out_dim,
        device=device,
        state_dict=model_state_dict,
    )

    # Set the model to evaluation mode for inference.
    model.eval()

    logger.info(f"Model initialized on device {device}")

    embedding_filename = f"machine_{machine_rank}_local_process_number_{local_rank}"

    # Get temporary GCS folder to write outputs of inference to. GiGL orchestration automatic cleans this, but
    # if running manually, you will need to clean this directory so that retries don't end up with stale files.
    gcs_utils = GcsUtils()
    gcs_base_uri = GcsUri.join(embedding_gcs_path, embedding_filename)
    num_files_at_gcs_path = gcs_utils.count_blobs_in_gcs_path(gcs_base_uri)
    if num_files_at_gcs_path > 0:
        logger.warning(
            f"{num_files_at_gcs_path} files already detected at base gcs path. Cleaning up files at path ... "
        )
        gcs_utils.delete_files_in_bucket_dir(gcs_base_uri)

    # GiGL class for exporting embeddings to GCS. This is achieved by writing ids and embeddings to an in-memory buffer which gets
    # flushed to GCS. Setting the min_shard_size_threshold_bytes field of this class sets the frequency of flushing to GCS, and defaults
    # to only flushing when flush_embeddings() is called explicitly or after exiting via a context manager.
    exporter = EmbeddingExporter(export_dir=gcs_base_uri)

    # We add a barrier here so that all machines and processes have initialized their dataloader at the start of the inference loop. Otherwise, on-the-fly subgraph
    # sampling may fail.

    barrier()

    t = time.time()
    data_loading_start_time = time.time()
    inference_start_time = time.time()
    cumulative_data_loading_time = 0.0
    cumulative_inference_time = 0.0

    # Begin inference loop

    # Iterating through the dataloader yields a `torch_geometric.data.Data` type
    for batch_idx, data in enumerate(data_loader):
        cumulative_data_loading_time += time.time() - data_loading_start_time

        inference_start_time = time.time()

        # These arguments to forward are specific to the GiGL heterogeneous LinkPredictionGNN model.
        # If just using a nn.Module, you can just use output = model(data)
        output = model(
            data=data, output_node_types=[inference_node_type], device=device
        )[inference_node_type]

        # The anchor node IDs are contained inside of the .batch field of the data
        node_ids = data[inference_node_type].batch.cpu()

        # Only the first `batch_size` rows of the node embeddings contain the embeddings of the anchor nodes
        node_embeddings = output[: data[inference_node_type].batch_size].cpu()

        # We add ids and embeddings to the in-memory buffer
        exporter.add_embedding(
            id_batch=node_ids,
            embedding_batch=node_embeddings,
            embedding_type=str(inference_node_type),
        )

        cumulative_inference_time += time.time() - inference_start_time

        if batch_idx > 0 and batch_idx % log_every_n_batch == 0:
            logger.info(
                f"Rank {rank} processed {batch_idx} batches for node type {inference_node_type}. "
                f"{log_every_n_batch} batches took {time.time() - t:.2f} seconds for node type {inference_node_type}. "
                f"Among them, data loading took {cumulative_data_loading_time:.2f} seconds."
                f"and model inference took {cumulative_inference_time:.2f} seconds."
            )
            t = time.time()
            cumulative_data_loading_time = 0
            cumulative_inference_time = 0

        data_loading_start_time = time.time()

    logger.info(
        f"--- Rank {rank} finished inference for node type {inference_node_type}."
    )

    write_embedding_start_time = time.time()
    # Flushes all remaining embeddings to GCS
    exporter.flush_embeddings()

    logger.info(
        f"--- Rank {rank} finished writing embeddings to GCS for node type {inference_node_type}, which took {time.time()-write_embedding_start_time:.2f} seconds"
    )

    # We first call barrier to ensure that all machines and processes have finished inference. Only once this is ensured is it safe to delete the data loader on the current
    # machine + process -- otherwise we may fail on processes which are still doing on-the-fly subgraph sampling. We then call `gc.collect()` to cleanup the memory
    # used by the data_loader on the current machine.

    barrier()

    del data_loader
    gc.collect()

    logger.info(
        f"--- All machines local rank {local_rank} finished inference for node type {inference_node_type}. Deleted data loader"
    )

    # Clean up for a graceful exit
    shutdown_rpc()


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

    # We call a GiGL function to launch a process for loading TFRecords into memory, partitioning the graph across multiple machines,
    # and registering that information to a DistDataset class.
    dataset = build_dataset_from_task_config_uri(
        task_config_uri=task_config_uri,
    )

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

    master_ip_address = gigl.distributed.utils.get_internal_ip_from_master_node()
    machine_rank = torch.distributed.get_rank()
    machine_world_size = torch.distributed.get_world_size()
    master_default_process_group_port = (
        gigl.distributed.utils.get_free_ports_from_master_node(num_ports=1)[0]
    )

    ## Inference Start

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

        # When using mp.spawn with `nprocs`, the first argument is implicitly set to be the process number on the current machine.
        mp.spawn(
            fn=_inference_process,
            args=(
                num_inference_processes_per_machine,  # local_world_size
                machine_rank,  # machine_rank
                machine_world_size,  # machine_world_size
                master_ip_address,  # master_ip_address
                master_default_process_group_port,  # master_default_process_group_port
                embedding_output_gcs_folder,  # embedding_gcs_path
                model_uri,  # model_state_dict_uri
                inference_batch_size,  # inference_batch_size
                hid_dim,  # hid_dim
                out_dim,  # out_dim
                dataset,  # dataset
                inferencer_args,  # inferencer_args
                inference_node_type,  # inference_node_type
                node_type_to_feature_dim,  # node_type_to_feature_dim
                edge_type_to_feature_dim,  # edge_type_to_feature_dim
            ),
            nprocs=num_inference_processes_per_machine,
            join=True,
        )

        logger.info(
            f"--- Inference finished on rank {machine_rank} for node type {inference_node_type}, which took {time.time()-inference_start_time:.2f} seconds"
        )

        # After inference is finished, we use the process on the Machine 0 to load embeddings from GCS to BQ.
        if machine_rank == 0:
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

    logger.info(
        f"--- Program finished, which took {time.time()-program_start_time:.2f} seconds"
    )


if __name__ == "__main__":
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
    logger.info(f"Unused arguments: {unused_args}")

    # We only need `job_name` and `task_config_uri` for running inference
    _run_example_inference(
        job_name=args.job_name,
        task_config_uri=args.task_config_uri,
    )
