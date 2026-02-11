"""
This file contains an example for how to run heterogeneous training in **graph store mode** using GiGL.

Graph Store Mode vs Standard Mode:
----------------------------------
Graph store mode uses a heterogeneous cluster architecture with two distinct sub-clusters:
  1. **Storage Cluster (graph_store_pool)**: Dedicated machines for storing and serving the graph
     data. These are typically high-memory machines without GPUs (e.g., n2-highmem-32).
  2. **Compute Cluster (compute_pool)**: Dedicated machines for running model training.
     These typically have GPUs attached (e.g., n1-standard-16 with NVIDIA_TESLA_T4).

This separation allows for:
  - Independent scaling of storage and compute resources
  - Better memory utilization (graph data stays on storage nodes)
  - Cost optimization by using appropriate hardware for each role

In contrast, the standard training mode (see `examples/link_prediction/heterogeneous_training.py`)
uses a homogeneous cluster where each machine handles both graph storage and computation.

Key Implementation Differences:
-------------------------------
This file (graph store mode):
  - Uses `RemoteDistDataset` to connect to a remote graph store cluster
  - Uses `init_compute_process` to initialize the compute node connection to storage
  - Obtains cluster topology via `get_graph_store_info()` which returns `GraphStoreInfo`
  - Uses `mp_sharing_dict` for efficient tensor sharing between local processes
  - Fetches ABLP input via `RemoteDistDataset.get_ablp_input()` for the train/val/test splits
  - Fetches random negative node IDs via `RemoteDistDataset.get_node_ids()`

Standard mode (`heterogeneous_training.py`):
  - Uses `DistDataset` with `build_dataset_from_task_config_uri` where each node loads its partition
  - Manually manages distributed process groups with master IP and port
  - Each machine stores its own partition of the graph data

To run this file with GiGL orchestration, set the fields similar to below:

trainerConfig:
  trainerArgs:
    log_every_n_batch: "50"
    ssl_positive_label_percentage: "0.05"
  command: python -m examples.link_prediction.graph_store.heterogeneous_training
  graphStoreStorageConfig:
    command: python -m examples.link_prediction.graph_store.storage_main
    storageArgs:
      sample_edge_direction: "in"
      splitter_cls_path: "gigl.utils.data_splitters.DistNodeAnchorLinkSplitter"
      splitter_kwargs: '{"sampling_direction": "in", "should_convert_labels_to_edges": true, "num_val": 0.1, "num_test": 0.1}'
      ssl_positive_label_percentage: "0.05"
      num_server_sessions: "1"
featureFlags:
  should_run_glt_backend: 'True'

Note: Ensure you use a resource config with `vertex_ai_graph_store_trainer_config` when
running in graph store mode.

You can run this example in a full pipeline with `make run_het_dblp_sup_gs_e2e_test` from GiGL root.

Note that the DBLP Dataset does not have specified labeled edges so we use the `ssl_positive_label_percentage`
field in the config to indicate what percentage of edges we should select as self-supervised labeled edges.
"""

import argparse
import gc
import os
import statistics
import sys
import time
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
import torch.distributed
import torch.multiprocessing as mp
from examples.link_prediction.models import init_example_gigl_heterogeneous_model
from torch_geometric.data import HeteroData

import gigl.distributed
import gigl.distributed.utils
from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.torch_training import is_distributed_available_and_initialized
from gigl.distributed import DistABLPLoader
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.graph_store.compute import (
    init_compute_process,
    shutdown_compute_proccess,
)
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.utils import get_available_device, get_graph_store_info
from gigl.env.distributed import GraphStoreInfo
from gigl.nn import LinkPredictionGNN, RetrievalLoss
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.model import load_state_dict_from_uri, save_state_dict
from gigl.utils.iterator import InfiniteIterator
from gigl.utils.sampling import parse_fanout

logger = Logger()


# We don't see logs for graph store mode for whatever reason.
# TODO(#442): Revert this once the GCP issues are resolved.
def flush():
    sys.stdout.write("\n")
    sys.stdout.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()


def _sync_metric_across_processes(metric: torch.Tensor) -> float:
    """
    Takes the average of a training metric across multiple processes. Note that this function requires DDP to be initialized.
    Args:
        metric (torch.Tensor): The metric, expressed as a torch Tensor, which should be synced across multiple processes
    Returns:
        float: The average of the provided metric across all training processes
    """
    assert is_distributed_available_and_initialized(), "DDP is not initialized"
    # Make a copy of the local loss tensor
    loss_tensor = metric.detach().clone()
    torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
    return loss_tensor.item() / torch.distributed.get_world_size()


def _setup_dataloaders(
    dataset: RemoteDistDataset,
    split: Literal["train", "val", "test"],
    cluster_info: GraphStoreInfo,
    supervision_edge_type: EdgeType,
    num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
    sampling_workers_per_process: int,
    main_batch_size: int,
    random_batch_size: int,
    device: torch.device,
    sampling_worker_shared_channel_size: str,
    process_start_gap_seconds: int,
) -> tuple[DistABLPLoader, DistNeighborLoader]:
    """
    Sets up main and random dataloaders for training and testing purposes using a remote graph store dataset.
    Args:
        dataset (RemoteDistDataset): Remote dataset connected to the graph store cluster.
        split (Literal["train", "val", "test"]): The current split which we are loading data for.
        cluster_info (GraphStoreInfo): Cluster topology info for graph store mode.
        supervision_edge_type (EdgeType): The supervision edge type to use for training.
        num_neighbors: Fanout for subgraph sampling, where the ith item corresponds to the number of items to sample for the ith hop.
        sampling_workers_per_process (int): Number of sampling workers per training/testing process.
        main_batch_size (int): Batch size for main dataloader with query and labeled nodes.
        random_batch_size (int): Batch size for random negative dataloader.
        device (torch.device): Device to put loaded subgraphs on.
        sampling_worker_shared_channel_size (str): Shared-memory buffer size (bytes) allocated for the channel during sampling.
        process_start_gap_seconds (int): The amount of time to sleep for initializing each dataloader.
    Returns:
        DistABLPLoader: Dataloader for loading main batch data with query and labeled nodes.
        DistNeighborLoader: Dataloader for loading random negative data.
    """
    rank = torch.distributed.get_rank()

    query_node_type = supervision_edge_type.src_node_type
    labeled_node_type = supervision_edge_type.dst_node_type

    shuffle = split == "train"

    # In graph store mode, we fetch ABLP input (anchors + positive/negative labels) from the storage cluster.
    # This returns dict[server_rank, (anchors, pos_labels, neg_labels)] which the DistABLPLoader knows how to handle.
    logger.info(f"---Rank {rank} fetching ABLP input for split={split}")
    flush()
    ablp_input = dataset.get_ablp_input(
        split=split,
        rank=cluster_info.compute_node_rank,
        world_size=cluster_info.num_compute_nodes,
        node_type=query_node_type,
        supervision_edge_type=supervision_edge_type,
    )

    main_loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=num_neighbors,
        input_nodes=(query_node_type, ablp_input),
        num_workers=sampling_workers_per_process,
        batch_size=main_batch_size,
        pin_memory_device=device,
        worker_concurrency=sampling_workers_per_process,
        channel_size=sampling_worker_shared_channel_size,
        process_start_gap_seconds=process_start_gap_seconds,
        shuffle=shuffle,
    )

    logger.info(f"---Rank {rank} finished setting up main loader for split={split}")
    flush()

    # We need to wait for all processes to finish initializing the main_loader before creating the
    # random_negative_loader so that its initialization doesn't compete for memory with the main_loader.
    torch.distributed.barrier()

    # For the random negative loader, we get all node IDs of the labeled node type from the storage cluster.
    all_node_ids = dataset.get_node_ids(
        rank=cluster_info.compute_node_rank,
        world_size=cluster_info.num_compute_nodes,
        node_type=labeled_node_type,
    )

    random_negative_loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=num_neighbors,
        input_nodes=(labeled_node_type, all_node_ids),
        num_workers=sampling_workers_per_process,
        batch_size=random_batch_size,
        pin_memory_device=device,
        worker_concurrency=sampling_workers_per_process,
        channel_size=sampling_worker_shared_channel_size,
        process_start_gap_seconds=process_start_gap_seconds,
        shuffle=shuffle,
    )

    logger.info(f"---Rank {rank} finished setting up random negative loader for split={split}")
    flush()

    # Wait for all processes to finish initializing the random_loader
    torch.distributed.barrier()

    return main_loader, random_negative_loader


def _compute_loss(
    model: LinkPredictionGNN,
    main_data: HeteroData,
    random_negative_data: HeteroData,
    loss_fn: RetrievalLoss,
    supervision_edge_type: EdgeType,
    device: torch.device,
) -> torch.Tensor:
    """
    With the provided model and loss function, computes the forward pass on the main batch data and random negative data.
    Args:
        model (LinkPredictionGNN): DDP-wrapped LinkPredictionGNN model for training and testing
        main_data (HeteroData): The batch of data containing query nodes, positive nodes, and hard negative nodes
        random_negative_data (HeteroData): The batch of data containing random negative nodes
        loss_fn (RetrievalLoss): Initialized class to use for loss calculation
        supervision_edge_type (EdgeType): The supervision edge type to use for training in format query_node -> relation -> labeled_node
        device (torch.device): Device for training or validation
    Returns:
        torch.Tensor: Final loss for the current batch on the current process
    """
    # Extract relevant node types from the supervision edge
    query_node_type = supervision_edge_type.src_node_type
    labeled_node_type = supervision_edge_type.dst_node_type

    if query_node_type == labeled_node_type:
        inference_node_types = [query_node_type]
    else:
        inference_node_types = [query_node_type, labeled_node_type]

    # Forward pass through encoder

    main_embeddings = model(
        data=main_data, output_node_types=inference_node_types, device=device
    )
    random_negative_embeddings = model(
        data=random_negative_data,
        output_node_types=inference_node_types,
        device=device,
    )

    # Extracting local query, random negative, positive, hard_negative, and random_negative indices.
    query_node_idx: torch.Tensor = torch.arange(
        main_data[query_node_type].batch_size
    ).to(device)
    random_negative_batch_size = random_negative_data[labeled_node_type].batch_size

    positive_idx: torch.Tensor = torch.cat(list(main_data.y_positive.values())).to(
        device
    )
    repeated_query_node_idx = query_node_idx.repeat_interleave(
        torch.tensor([len(v) for v in main_data.y_positive.values()]).to(device)
    )
    if hasattr(main_data, "y_negative"):
        hard_negative_idx: torch.Tensor = torch.cat(
            list(main_data.y_negative.values())
        ).to(device)
    else:
        hard_negative_idx = torch.empty(0, dtype=torch.long).to(device)

    # Use local IDs to get the corresponding embeddings in the tensors

    repeated_query_embeddings = main_embeddings[query_node_type][
        repeated_query_node_idx
    ]
    positive_node_embeddings = main_embeddings[labeled_node_type][positive_idx]
    hard_negative_embeddings = main_embeddings[labeled_node_type][hard_negative_idx]
    random_negative_embeddings = random_negative_embeddings[labeled_node_type][
        :random_negative_batch_size
    ]

    # Decode the query embeddings and the candidate embeddings

    repeated_candidate_scores = model.decode(
        query_embeddings=repeated_query_embeddings,
        candidate_embeddings=torch.cat(
            [
                positive_node_embeddings,
                hard_negative_embeddings,
                random_negative_embeddings,
            ],
            dim=0,
        ),
    )

    # Compute the global candidate ids and concatenate into a single tensor

    global_candidate_ids = torch.cat(
        (
            main_data[labeled_node_type].node[positive_idx],
            main_data[labeled_node_type].node[hard_negative_idx],
            random_negative_data[labeled_node_type].node[:random_negative_batch_size],
        )
    )

    global_repeated_query_ids = main_data[query_node_type].node[repeated_query_node_idx]

    # Feed scores and ids into the RetrievalLoss forward pass to get the final loss

    loss = loss_fn(
        repeated_candidate_scores=repeated_candidate_scores,
        candidate_ids=global_candidate_ids,
        repeated_query_ids=global_repeated_query_ids,
        device=device,
    )

    return loss


@dataclass(frozen=True)
class TrainingProcessArgs:
    """
    Arguments for the heterogeneous training process in graph store mode.

    Attributes:
        local_world_size (int): Number of training processes spawned by each machine.
        cluster_info (GraphStoreInfo): Cluster topology info for graph store mode.
        mp_sharing_dict (MutableMapping[str, torch.Tensor]): Shared dictionary for efficient tensor
            sharing between local processes.
        supervision_edge_type (EdgeType): The supervision edge type for training.
        model_uri (Uri): URI to save/load the trained model state dict.
        hid_dim (int): Hidden dimension of the model.
        out_dim (int): Output dimension of the model.
        node_type_to_feature_dim (dict[NodeType, int]): Mapping of node types to their feature dimensions.
        edge_type_to_feature_dim (dict[EdgeType, int]): Mapping of edge types to their feature dimensions.
        num_neighbors (Union[list[int], dict[EdgeType, list[int]]]): Fanout for subgraph sampling.
        sampling_workers_per_process (int): Number of sampling workers per training/testing process.
        sampling_worker_shared_channel_size (str): Shared-memory buffer size for the channel during sampling.
        process_start_gap_seconds (int): Time to sleep between dataloader initializations.
        main_batch_size (int): Batch size for main dataloader.
        random_batch_size (int): Batch size for random negative dataloader.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        num_max_train_batches (int): Maximum number of training batches across all processes.
        num_val_batches (int): Number of validation batches across all processes.
        val_every_n_batch (int): Frequency to run validation during training.
        log_every_n_batch (int): Frequency to log batch information during training.
        should_skip_training (bool): If True, skip training and only run testing.
    """

    # Distributed context
    local_world_size: int
    cluster_info: GraphStoreInfo
    mp_sharing_dict: MutableMapping[str, torch.Tensor]

    # Data
    supervision_edge_type: EdgeType

    # Model
    model_uri: Uri
    hid_dim: int
    out_dim: int
    node_type_to_feature_dim: dict[NodeType, int]
    edge_type_to_feature_dim: dict[EdgeType, int]

    # Sampling config
    num_neighbors: Union[list[int], dict[EdgeType, list[int]]]
    sampling_workers_per_process: int
    sampling_worker_shared_channel_size: str
    process_start_gap_seconds: int

    # Training hyperparameters
    main_batch_size: int
    random_batch_size: int
    learning_rate: float
    weight_decay: float
    num_max_train_batches: int
    num_val_batches: int
    val_every_n_batch: int
    log_every_n_batch: int
    should_skip_training: bool


def _training_process(
    local_rank: int,
    args: TrainingProcessArgs,
) -> None:
    """
    This function is spawned by each machine for training a GNN model using graph store mode.
    Args:
        local_rank (int): Process number on the current machine
        args (TrainingProcessArgs): Dataclass containing all training process arguments
    """

    # Note: This is a *critical* step in Graph Store mode. It initializes the connection to the storage cluster
    # and sets up torch.distributed with the appropriate backend (NCCL if CUDA available, gloo otherwise).
    logger.info(
        f"Initializing compute process for local_rank {local_rank} in machine {args.cluster_info.compute_node_rank}"
    )
    flush()
    init_compute_process(local_rank, args.cluster_info)
    dataset = RemoteDistDataset(
        args.cluster_info, local_rank, mp_sharing_dict=args.mp_sharing_dict
    )

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    logger.info(
        f"---Current training process rank: {rank}, training process world size: {world_size}"
    )
    flush()

    # We use one training device for each local process
    device = get_available_device(local_process_rank=local_rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    logger.info(f"---Rank {rank} training process set device {device}")

    loss_fn = RetrievalLoss(
        loss=torch.nn.CrossEntropyLoss(reduction="mean"),
        temperature=0.07,
        remove_accidental_hits=True,
    )

    if not args.should_skip_training:
        train_main_loader, train_random_negative_loader = _setup_dataloaders(
            dataset=dataset,
            split="train",
            cluster_info=args.cluster_info,
            supervision_edge_type=args.supervision_edge_type,
            num_neighbors=args.num_neighbors,
            sampling_workers_per_process=args.sampling_workers_per_process,
            main_batch_size=args.main_batch_size,
            random_batch_size=args.random_batch_size,
            device=device,
            sampling_worker_shared_channel_size=args.sampling_worker_shared_channel_size,
            process_start_gap_seconds=args.process_start_gap_seconds,
        )

        train_main_loader_iter = InfiniteIterator(train_main_loader)
        train_random_negative_loader_iter = InfiniteIterator(
            train_random_negative_loader
        )

        val_main_loader, val_random_negative_loader = _setup_dataloaders(
            dataset=dataset,
            split="val",
            cluster_info=args.cluster_info,
            supervision_edge_type=args.supervision_edge_type,
            num_neighbors=args.num_neighbors,
            sampling_workers_per_process=args.sampling_workers_per_process,
            main_batch_size=args.main_batch_size,
            random_batch_size=args.random_batch_size,
            device=device,
            sampling_worker_shared_channel_size=args.sampling_worker_shared_channel_size,
            process_start_gap_seconds=args.process_start_gap_seconds,
        )

        val_main_loader_iter = InfiniteIterator(val_main_loader)
        val_random_negative_loader_iter = InfiniteIterator(val_random_negative_loader)

        model = init_example_gigl_heterogeneous_model(
            node_type_to_feature_dim=args.node_type_to_feature_dim,
            edge_type_to_feature_dim=args.edge_type_to_feature_dim,
            hid_dim=args.hid_dim,
            out_dim=args.out_dim,
            device=device,
            wrap_with_ddp=True,
            find_unused_encoder_parameters=True,
        )
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        logger.info(
            f"Model initialized on rank {rank} training device {device}\n{model}"
        )
        flush()

        # We add a barrier to wait for all processes to finish preparing the dataloader and initializing the model
        torch.distributed.barrier()

        # Entering the training loop
        training_start_time = time.time()
        batch_idx = 0
        avg_train_loss = 0.0
        last_n_batch_avg_loss: list[float] = []
        last_n_batch_time: list[float] = []
        num_max_train_batches_per_process = args.num_max_train_batches // world_size
        num_val_batches_per_process = args.num_val_batches // world_size
        logger.info(
            f"num_max_train_batches_per_process is set to {num_max_train_batches_per_process}"
        )

        model.train()

        batch_start = time.time()
        for main_data, random_data in zip(
            train_main_loader_iter, train_random_negative_loader_iter
        ):
            if batch_idx >= num_max_train_batches_per_process:
                logger.info(
                    f"num_max_train_batches_per_process={num_max_train_batches_per_process} reached, "
                    f"stopping training on machine {args.cluster_info.compute_node_rank} local rank {local_rank}"
                )
                break
            loss = _compute_loss(
                model=model,
                main_data=main_data,
                random_negative_data=random_data,
                loss_fn=loss_fn,
                supervision_edge_type=args.supervision_edge_type,
                device=device,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_train_loss = _sync_metric_across_processes(metric=loss)
            last_n_batch_avg_loss.append(avg_train_loss)
            last_n_batch_time.append(time.time() - batch_start)
            batch_start = time.time()
            batch_idx += 1
            if batch_idx % args.log_every_n_batch == 0:
                logger.info(
                    f"rank={rank}, batch={batch_idx}, latest local train_loss={loss:.6f}"
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                logger.info(
                    f"rank={rank}, batch={batch_idx}, mean(batch_time)={statistics.mean(last_n_batch_time):.3f} sec, max(batch_time)={max(last_n_batch_time):.3f} sec, min(batch_time)={min(last_n_batch_time):.3f} sec"
                )
                last_n_batch_time.clear()
                logger.info(
                    f"rank={rank}, latest avg_train_loss={avg_train_loss:.6f}, last {args.log_every_n_batch} mean(avg_train_loss)={statistics.mean(last_n_batch_avg_loss):.6f}"
                )
                last_n_batch_avg_loss.clear()
                flush()

            if batch_idx % args.val_every_n_batch == 0:
                logger.info(f"rank={rank}, batch={batch_idx}, validating...")
                model.eval()
                _run_validation_loops(
                    model=model,
                    main_loader=val_main_loader_iter,
                    random_negative_loader=val_random_negative_loader_iter,
                    loss_fn=loss_fn,
                    supervision_edge_type=args.supervision_edge_type,
                    device=device,
                    log_every_n_batch=args.log_every_n_batch,
                    num_batches=num_val_batches_per_process,
                )
                model.train()

        logger.info(f"---Rank {rank} finished training")
        flush()

        # Memory cleanup and waiting for all processes to finish
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        torch.distributed.barrier()

        # We explicitly shutdown all the dataloaders to reduce their memory footprint.
        train_main_loader.shutdown()
        train_random_negative_loader.shutdown()
        val_main_loader.shutdown()
        val_random_negative_loader.shutdown()

        # We save the model on the process with rank 0.
        if torch.distributed.get_rank() == 0:
            logger.info(
                f"Training loop finished, took {time.time() - training_start_time:.3f} seconds, saving model to {args.model_uri}"
            )
            save_state_dict(
                model=model.unwrap_from_ddp(), save_to_path_uri=args.model_uri
            )
            flush()

    else:  # should_skip_training is True, meaning we should only run testing
        state_dict = load_state_dict_from_uri(
            load_from_uri=args.model_uri, device=device
        )
        model = init_example_gigl_heterogeneous_model(
            node_type_to_feature_dim=args.node_type_to_feature_dim,
            edge_type_to_feature_dim=args.edge_type_to_feature_dim,
            hid_dim=args.hid_dim,
            out_dim=args.out_dim,
            device=device,
            wrap_with_ddp=True,
            find_unused_encoder_parameters=True,
            state_dict=state_dict,
        )
        logger.info(
            f"Model initialized on rank {rank} training device {device}\n{model}"
        )

    logger.info(f"---Rank {rank} started testing")
    flush()
    testing_start_time = time.time()

    model.eval()

    test_main_loader, test_random_negative_loader = _setup_dataloaders(
        dataset=dataset,
        split="test",
        cluster_info=args.cluster_info,
        supervision_edge_type=args.supervision_edge_type,
        num_neighbors=args.num_neighbors,
        sampling_workers_per_process=args.sampling_workers_per_process,
        main_batch_size=args.main_batch_size,
        random_batch_size=args.random_batch_size,
        device=device,
        sampling_worker_shared_channel_size=args.sampling_worker_shared_channel_size,
        process_start_gap_seconds=args.process_start_gap_seconds,
    )

    # Since we are doing testing, we only want to go through the data once.
    test_main_loader_iter = iter(test_main_loader)
    test_random_negative_loader_iter = iter(test_random_negative_loader)

    _run_validation_loops(
        model=model,
        main_loader=test_main_loader_iter,
        random_negative_loader=test_random_negative_loader_iter,
        loss_fn=loss_fn,
        supervision_edge_type=args.supervision_edge_type,
        device=device,
        log_every_n_batch=args.log_every_n_batch,
    )

    # Memory cleanup and waiting for all processes to finish
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    torch.distributed.barrier()

    test_main_loader.shutdown()
    test_random_negative_loader.shutdown()

    logger.info(
        f"---Rank {rank} finished testing in {time.time() - testing_start_time:.3f} seconds"
    )
    flush()

    # Graph store mode cleanup: shutdown the compute process connection to the storage cluster.
    shutdown_compute_proccess()
    gc.collect()

    logger.info(
        f"---Rank {rank} finished all training and testing, shut down compute process"
    )
    flush()


@torch.inference_mode()
def _run_validation_loops(
    model: LinkPredictionGNN,
    main_loader: Iterator[HeteroData],
    random_negative_loader: Iterator[HeteroData],
    loss_fn: RetrievalLoss,
    supervision_edge_type: EdgeType,
    device: torch.device,
    log_every_n_batch: int,
    num_batches: Optional[int] = None,
) -> None:
    """
    Runs validation using the provided models and dataloaders.
    This function is shared for both validation while training and testing after training has completed.
    Args:
        model (LinkPredictionGNN): DDP-wrapped LinkPredictionGNN model for training and testing
        main_loader (Iterator[HeteroData]): Dataloader for loading main batch data with query and labeled nodes
        random_negative_loader (Iterator[HeteroData]): Dataloader for loading random negative data
        loss_fn (RetrievalLoss): Initialized class to use for loss calculation
        supervision_edge_type (EdgeType): The supervision edge type to use for training
        device (torch.device): Device to use for training or testing
        log_every_n_batch (int): The frequency we should log batch information
        num_batches (Optional[int]): The number of batches to run the validation loop for.
    """

    rank = torch.distributed.get_rank()

    logger.info(
        f"Running validation loop on rank={rank}, log_every_n_batch={log_every_n_batch}, num_batches={num_batches}"
    )
    if num_batches is None:
        if isinstance(main_loader, InfiniteIterator) or isinstance(
            random_negative_loader, InfiniteIterator
        ):
            raise ValueError(
                "Must set `num_batches` field when the provided data loaders are wrapped with InfiniteIterator"
            )

    batch_idx = 0
    batch_losses: list[float] = []
    last_n_batch_time: list[float] = []
    batch_start = time.time()

    while True:
        if num_batches and batch_idx >= num_batches:
            break
        try:
            main_data = next(main_loader)
            random_data = next(random_negative_loader)
        except StopIteration:
            break

        loss = _compute_loss(
            model=model,
            main_data=main_data,
            random_negative_data=random_data,
            loss_fn=loss_fn,
            supervision_edge_type=supervision_edge_type,
            device=device,
        )

        batch_losses.append(loss.item())
        last_n_batch_time.append(time.time() - batch_start)
        batch_start = time.time()
        batch_idx += 1
        if batch_idx % log_every_n_batch == 0:
            logger.info(f"rank={rank}, batch={batch_idx}, latest test_loss={loss:.6f}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info(
                f"rank={rank}, batch={batch_idx}, mean(batch_time)={statistics.mean(last_n_batch_time):.3f} sec, max(batch_time)={max(last_n_batch_time):.3f} sec, min(batch_time)={min(last_n_batch_time):.3f} sec"
            )
            last_n_batch_time.clear()
            flush()
    local_avg_loss = statistics.mean(batch_losses)
    logger.info(
        f"rank={rank} finished validation loop, local loss: {local_avg_loss=:.6f}"
    )
    global_avg_val_loss = _sync_metric_across_processes(
        metric=torch.tensor(local_avg_loss, device=device)
    )
    logger.info(f"rank={rank} got global validation loss {global_avg_val_loss=:.6f}")
    flush()

    return


def _run_example_training(
    task_config_uri: str,
):
    """
    Runs an example training + testing loop using GiGL Orchestration in graph store mode.
    Args:
        task_config_uri (str): Path to YAML-serialized GbmlConfig proto.
    """
    program_start_time = time.time()
    mp.set_start_method("spawn")
    logger.info(f"Starting sub process method: {mp.get_start_method()}")

    # Step 1: Initialize global process group to get cluster info
    torch.distributed.init_process_group(backend="gloo")
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

    # Step 2: Read config
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=UriFactory.create_uri(task_config_uri)
    )

    # Training Hyperparameters
    trainer_args = dict(gbml_config_pb_wrapper.trainer_config.trainer_args)

    if torch.cuda.is_available():
        default_local_world_size = torch.cuda.device_count()
    else:
        default_local_world_size = 2
    local_world_size = int(
        trainer_args.get("local_world_size", str(default_local_world_size))
    )

    if torch.cuda.is_available():
        if local_world_size > torch.cuda.device_count():
            raise ValueError(
                f"Specified a local world size of {local_world_size} which exceeds the number of devices {torch.cuda.device_count()}"
            )

    fanout = trainer_args.get("num_neighbors", "[10, 10]")
    num_neighbors = parse_fanout(fanout)

    sampling_workers_per_process: int = int(
        trainer_args.get("sampling_workers_per_process", "4")
    )

    main_batch_size = int(trainer_args.get("main_batch_size", "16"))
    random_batch_size = int(trainer_args.get("random_batch_size", "16"))

    hid_dim = int(trainer_args.get("hid_dim", "16"))
    out_dim = int(trainer_args.get("out_dim", "16"))

    sampling_worker_shared_channel_size: str = trainer_args.get(
        "sampling_worker_shared_channel_size", "4GB"
    )

    process_start_gap_seconds = int(trainer_args.get("process_start_gap_seconds", "0"))
    log_every_n_batch = int(trainer_args.get("log_every_n_batch", "25"))

    learning_rate = float(trainer_args.get("learning_rate", "0.0005"))
    weight_decay = float(trainer_args.get("weight_decay", "0.0005"))
    num_max_train_batches = int(trainer_args.get("num_max_train_batches", "1000"))
    num_val_batches = int(trainer_args.get("num_val_batches", "100"))
    val_every_n_batch = int(trainer_args.get("val_every_n_batch", "50"))

    logger.info(
        f"Got training args local_world_size={local_world_size}, \
        num_neighbors={num_neighbors}, \
        sampling_workers_per_process={sampling_workers_per_process}, \
        main_batch_size={main_batch_size}, \
        random_batch_size={random_batch_size}, \
        hid_dim={hid_dim}, \
        out_dim={out_dim}, \
        sampling_worker_shared_channel_size={sampling_worker_shared_channel_size}, \
        process_start_gap_seconds={process_start_gap_seconds}, \
        log_every_n_batch={log_every_n_batch}, \
        learning_rate={learning_rate}, \
        weight_decay={weight_decay}, \
        num_max_train_batches={num_max_train_batches}, \
        num_val_batches={num_val_batches}, \
        val_every_n_batch={val_every_n_batch}"
    )

    # Step 3: Extract model/data config
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

    model_uri = UriFactory.create_uri(
        gbml_config_pb_wrapper.gbml_config_pb.shared_config.trained_model_metadata.trained_model_uri
    )

    should_skip_training = gbml_config_pb_wrapper.shared_config.should_skip_training

    supervision_edge_types = (
        gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_types()
    )
    if len(supervision_edge_types) != 1:
        raise NotImplementedError(
            "GiGL Training currently only supports 1 supervision edge type."
        )
    supervision_edge_type = supervision_edge_types[0]

    # Step 4: Create shared dict for inter-process tensor sharing
    mp_sharing_dict = mp.Manager().dict()

    # Step 5: Spawn training processes
    logger.info("--- Launching training processes ...\n")
    flush()
    start_time = time.time()

    training_args = TrainingProcessArgs(
        local_world_size=local_world_size,
        cluster_info=cluster_info,
        mp_sharing_dict=mp_sharing_dict,
        supervision_edge_type=supervision_edge_type,
        model_uri=model_uri,
        hid_dim=hid_dim,
        out_dim=out_dim,
        node_type_to_feature_dim=node_type_to_feature_dim,
        edge_type_to_feature_dim=edge_type_to_feature_dim,
        num_neighbors=num_neighbors,
        sampling_workers_per_process=sampling_workers_per_process,
        sampling_worker_shared_channel_size=sampling_worker_shared_channel_size,
        process_start_gap_seconds=process_start_gap_seconds,
        main_batch_size=main_batch_size,
        random_batch_size=random_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_max_train_batches=num_max_train_batches,
        num_val_batches=num_val_batches,
        val_every_n_batch=val_every_n_batch,
        log_every_n_batch=log_every_n_batch,
        should_skip_training=should_skip_training,
    )

    torch.multiprocessing.spawn(
        _training_process,
        args=(training_args,),
        nprocs=local_world_size,
        join=True,
    )
    logger.info(
        f"--- Training finished, took {time.time() - start_time} seconds"
    )
    logger.info(
        f"--- Program finished, which took {time.time() - program_start_time:.2f} seconds"
    )
    flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for distributed model training on VertexAI (graph store mode)"
    )
    parser.add_argument("--task_config_uri", type=str, help="Gbml config uri")

    args, unused_args = parser.parse_known_args()
    logger.info(f"Unused arguments: {unused_args}")

    _run_example_training(
        task_config_uri=args.task_config_uri,
    )
