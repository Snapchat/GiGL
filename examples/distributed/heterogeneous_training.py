"""
This file contains an example for how to run heterogeneous training using live subgraph sampling powered by GraphLearn-for-PyTorch (GLT).
While `run_example_training` is coupled with GiGL orchestration, the `_training_process` and `testing_process` functions are generic
and can be used as references for writing training for pipelines not dependent on GiGL orchestration.

To run this file with GiGL orchestration, set the fields similar to below:

trainerConfig:
  trainerArgs:
    log_every_n_batch: "50"
    ssl_positive_label_percentage: "0.05"
  command: python -m examples.distributed.heterogeneous_training
featureFlags:
  should_run_glt_backend: 'True'

You can run this example in a full pipeline with `make run_dblp_glt_kfp_test` from GiGL root.
TODO (mkolodner-sc): Add example of how to run locally once CPU support is enabled

Note that the DBLP Dataset does not  have specified labeled edges so we use the `ssl_positive_label_percentage` field in the config to indicate what
percentage of edges we should select as self-supervised labeled edges. Doing this randomly sets 5% as "labels". Note that the current GiGL implementation
does not remove these selected edges from the global set of edges, which may have a slight negative impact on training specifically with self-supervised learning.
This will improved on in the future.
"""

import argparse
import statistics
import time
from collections.abc import Iterator, Mapping
from typing import Optional

import torch
import torch.distributed
import torch.multiprocessing as mp
from examples.models import init_example_gigl_heterogeneous_dblp_model
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import HeteroData

import gigl.distributed.utils
from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.torch_training import sync_metric_across_processes
from gigl.distributed import (
    DistABLPLoader,
    DistLinkPredictionDataset,
    build_dataset_from_task_config_uri,
)
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.module.loss import RetrievalLoss
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.model import load_state_dict_from_uri, save_state_dict
from gigl.utils.iterator import InfiniteIterator

logger = Logger()


def _compute_loss(
    model: DistributedDataParallel,
    main_data: HeteroData,
    random_negative_data: HeteroData,
    loss_fn: RetrievalLoss,
    supervision_edge_type: EdgeType,
    device: torch.device,
) -> torch.Tensor:
    """
    With the provided model and loss function, computes the forward pass on the main batch data and random negative data.
    Args:
        model (DistributedDataParallel): DDP-wrapped torch model for training and testing
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

    # Extracting local query, random negative, positive, hard_negative, and random_negative ids.
    # Local in this case refers to the local index in the batch, while global subsequently refers to the node's unique global ID across all nodes in the dataset.
    # Global ids are stored in data.node, ex. `data.node = [50, 20, 10]` where the `0` is the local id for global id `50`
    query_node_ids: torch.Tensor = torch.arange(
        main_data[query_node_type].batch_size
    ).to(device)
    random_negative_batch_size = random_negative_data[labeled_node_type].batch_size

    # main_data.y_positive is a dict[query_node_local_id: int, labeled_node_local_ids: torch.Tensor], even in the heterogeneous setting.
    positive_ids: torch.Tensor = torch.cat(list(main_data.y_positive.values())).to(
        device
    )
    # We also extract a repeated query node id tensor which upsamples each query node based on the number of positives it has
    repeated_query_node_ids = query_node_ids.repeat_interleave(
        torch.tensor([len(v) for v in main_data.y_positive.values()]).to(device)
    )
    if hasattr(main_data, "y_negative"):
        hard_negative_ids: torch.Tensor = torch.cat(
            list(main_data.y_negative.values())
        ).to(device)
    else:
        hard_negative_ids = torch.empty(0, dtype=torch.long).to(device)

    # Use local IDs to get the corresponding embeddings in the tensors

    repeated_query_embeddings = main_embeddings[query_node_type][
        repeated_query_node_ids
    ]
    hard_negative_embeddings = main_embeddings[labeled_node_type][hard_negative_ids]
    positive_node_embeddings = main_embeddings[labeled_node_type][positive_ids]
    hard_negative_embeddings = main_embeddings[labeled_node_type][hard_negative_ids]
    random_negative_embeddings = random_negative_embeddings[labeled_node_type][
        :random_negative_batch_size
    ]

    # Decode the query embeddings and the candidate embeddings to get a tensor of scores of shape [num_positives, num_positives + num_hard_negatives + num_random_negatives]

    repeated_candidate_scores = model.module.decode(
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

    # Compute the global candidate ids and concatentate into a single tensor

    global_candidate_ids = torch.cat(
        (
            main_data[labeled_node_type].node[positive_ids],
            main_data[labeled_node_type].node[hard_negative_ids],
            random_negative_data[labeled_node_type].node[:random_negative_batch_size],
        )
    )

    global_repeated_query_ids = main_data[query_node_type].node[repeated_query_node_ids]

    # Feed scores and ids into the RetrievalLoss forward pass to get the final loss

    loss = loss_fn(
        repeated_candidate_scores=repeated_candidate_scores,
        candidate_ids=global_candidate_ids,
        repeated_query_ids=global_repeated_query_ids,
        device=device,
    )

    return loss


def _training_process(
    local_rank: int,
    local_world_size: int,
    node_rank: int,
    node_world_size: int,
    dataset: DistLinkPredictionDataset,
    supervision_edge_type: EdgeType,
    node_type_to_feature_dim: dict[NodeType, int],
    edge_type_to_feature_dim: dict[EdgeType, int],
    master_ip_address: str,
    master_default_process_group_port: int,
    model_uri: Uri,
    subgraph_fanout: list[int],
    sampling_workers_per_process: int,
    main_batch_size: int,
    random_batch_size: int,
    sampling_worker_shared_channel_size: str,
    process_start_gap_seconds: int,
    log_every_n_batch: int,
    learning_rate: float,
    weight_decay: float,
    num_max_train_batches: int,
    num_val_batches: int,
    val_every_n_batch: int,
) -> None:
    """
    This function is spawned by each machine for training a GNN model given some loaded distributed dataset.
    Args:
        local_rank (int): Process number on the current machine
        local_world_size (int): Number of training processes spawned by each machine
        node_rank (int): Rank of the current machine
        node_world_size (int): Total number of machines
        dataset (DistLinkPredictionDataset): Loaded Distributed Dataset for training
        supervision_edge_type (EdgeType): The supervision edge type to use for training in format query_node -> relation -> labeled_node
        node_type_to_feature_dim (dict[NodeType, int]): Input node feature dimension for the model per node type
        edge_type_to_feature_dim (dict[EdgeType, int]): Input edge feature dimension for the model per edge type
        master_ip_address (str): IP Address of the master worker for distributed communication
        master_default_process_group_port (int): Port on the master worker for setting up distributed process group communication
        model_uri (Uri): URI Path to save the model to
        subgraph_fanout: list[int]: Fanout for subgraph sampling, where the ith item corresponds to the number of items to sample for the ith hop
        sampling_workers_per_process (int): Number of sampling workers per training process
        main_batch_size (int): Batch size for main dataloader with query and labeled nodes
        random_batch_size (int): Batch size for random negative dataloader
        sampling_worker_shared_channel_size (str): Shared-memory buffer size (bytes) allocated for the channel during sampling
        process_start_gap_seconds (int): The amount of time to sleep for initializing each dataloader. For large-scale settings, consider setting this
            field to 30-60 seconds to ensure dataloaders don't compete for memory during initialization, causing OOM.
        log_every_n_batch (int): The frequency we should log batch information when training
        learning_rate (float): Learning rate for training
        weight_decay (float): Weight decay for training
        num_max_train_batches (int): The maximum number of batches to train for across all training processes
        num_val_batches (int): The number of batches to do validation for across all training processes
        val_every_n_batch: (int): The frequency we should log batch information when validating
    """
    # TODO (mkolodner-sc): Investigate work needed + add support for CPU training
    if not torch.cuda.is_available():
        raise NotImplementedError(
            "Currently, only GPU training is supported with this example training loop"
        )

    logger.info(
        f"---Machine {node_rank} local rank {local_rank} training process started"
    )

    # We use one training device for each local process
    training_device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(training_device)
    logger.info(
        f"---Machine {node_rank} local rank {local_rank} training process set device {training_device}"
    )

    training_process_world_size = node_world_size * local_world_size
    training_process_global_rank = node_rank * local_world_size + local_rank
    logger.info(
        f"---Current training process global rank: {training_process_global_rank}, training process world size: {training_process_world_size}"
    )

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_ip_address}:{master_default_process_group_port}",
        rank=training_process_global_rank,
        world_size=training_process_world_size,
    )

    logger.info(
        f"---Machine {node_rank} local rank {local_rank} training process group initialized"
    )

    # We initialize the train dataloaders in order: main_loader_0, random_loader_0, main_loader_1, random_loader_1, ...
    # where the number indicates the local process rank. There is a `process_start_gap_seconds / 2` time.sleep() call in between each of
    # these initializations. We do this since initializing these NeighborLoaders creates a spike in memory usage, and initializing multiple
    # loaders at the same time can lead to OOM.

    query_node_type = supervision_edge_type.src_node_type

    labeled_node_type = supervision_edge_type.dst_node_type

    assert isinstance(dataset.train_node_ids, Mapping)

    # When initializing a DistABLPLoader for heterogeneous use cases, it is important that `input_nodes` be a Tuple(NodeType, torch.Tensor) to indicate that we should
    # be loading heterogeneous data. The NodeType in this case should be the query node type. Additionally, it is important that `supervision_edge_type` be provided so
    # that the dataloader knows how to fanout around the labeled edge types. A heterogeneous DistABLPLoader will return a HeteroData object.
    train_main_loader: Iterator[HeteroData] = DistABLPLoader(
        dataset=dataset,
        num_neighbors=subgraph_fanout,
        input_nodes=(query_node_type, dataset.train_node_ids[query_node_type]),
        supervision_edge_type=supervision_edge_type,
        num_workers=sampling_workers_per_process,
        batch_size=main_batch_size,
        pin_memory_device=training_device,
        worker_concurrency=sampling_workers_per_process,
        channel_size=sampling_worker_shared_channel_size,
        # Each train_main_loader will wait for `process_start_gap_seconds` * `local_process_rank` seconds before initializing to reduce peak memory usage.
        process_start_gap_seconds=process_start_gap_seconds,
        shuffle=True,
    )

    train_main_loader = InfiniteIterator(train_main_loader)

    logger.info("Finished setting up train main loader.")

    # The random_negative_loader will wait for `process_start_gap_seconds / 2` seconds after initializing the main_loader so that it doesn't interfere with other processes.
    time.sleep(process_start_gap_seconds / 2)

    assert isinstance(dataset.node_ids, Mapping)

    # Similarly, the heterogeneous DistNeighborLoader should be provided as a Tuple[NodeType, torch.Tensor], where the NodeType is the node type of
    # the labeled node in the link prediction task.
    train_random_negative_loader: Iterator[HeteroData] = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=subgraph_fanout,
        input_nodes=(labeled_node_type, dataset.node_ids[labeled_node_type]),
        num_workers=sampling_workers_per_process,
        batch_size=random_batch_size,
        pin_memory_device=training_device,
        worker_concurrency=sampling_workers_per_process,
        channel_size=sampling_worker_shared_channel_size,
        process_start_gap_seconds=0,
        shuffle=True,
    )

    train_random_negative_loader = InfiniteIterator(train_random_negative_loader)

    logger.info("Finished setting up train random negative loader")

    logger.info(
        f"---Machine {node_rank} local rank {local_rank} training data loaders initialized"
    )

    # We add a barrier to wait for all the training loaders to finish initializing before we initialize the val loaders to reduce the peak memory usage.
    torch.distributed.barrier()

    # We initialize the val dataloaders in order: main_loader_0, random_loader_0, main_loader_1, random_loader_1, ...
    # where the number indicates the local process rank. There is a `process_start_gap_seconds / 2` time.sleep() call in between each of
    # these initializations. We do this since initializing these NeighborLoaders creates a spike in memory usage, and initializing multiple
    # loaders at the same time can lead to OOM.

    assert isinstance(dataset.val_node_ids, Mapping)

    # When initializing a DistABLPLoader for heterogeneous use cases, it is important that `input_nodes` be a Tuple(NodeType, torch.Tensor) to indicate that we should
    # be loading heterogeneous data. The NodeType in this case should be the query node type. Additionally, it is important that `supervision_edge_type` be provided so
    # that the dataloader knows how to fanout around the labeled edge types. A heterogeneous DistABLPLoader will return a HeteroData object.
    val_main_loader: Iterator[HeteroData] = DistABLPLoader(
        dataset=dataset,
        num_neighbors=subgraph_fanout,
        input_nodes=(query_node_type, dataset.val_node_ids[query_node_type]),
        supervision_edge_type=supervision_edge_type,
        num_workers=sampling_workers_per_process,
        batch_size=main_batch_size,
        pin_memory_device=training_device,
        worker_concurrency=sampling_workers_per_process,
        channel_size=sampling_worker_shared_channel_size,
        # Each val_main_loader will wait for `process_start_gap_seconds` * `local_process_rank` seconds before initializing to reduce peak memory usage.
        process_start_gap_seconds=process_start_gap_seconds,
    )

    val_main_loader = InfiniteIterator(val_main_loader)

    logger.info("Finished setting up val main loader.")

    # The random_negative_loader will wait for `process_start_gap_seconds / 2` seconds after initializing the main_loader so that it doesn't interfere with other processes.
    time.sleep(process_start_gap_seconds / 2)

    assert isinstance(dataset.node_ids, Mapping)

    # Similarly, the heterogeneous DistNeighborLoader should be provided as a Tuple[NodeType, torch.Tensor], where the NodeType is the node type of
    # the labeled node in the link prediction task.
    val_random_negative_loader: Iterator[HeteroData] = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=subgraph_fanout,
        input_nodes=(labeled_node_type, dataset.node_ids[labeled_node_type]),
        num_workers=sampling_workers_per_process,
        batch_size=random_batch_size,
        pin_memory_device=training_device,
        worker_concurrency=sampling_workers_per_process,
        channel_size=sampling_worker_shared_channel_size,
        process_start_gap_seconds=0,
    )

    val_random_negative_loader = InfiniteIterator(val_random_negative_loader)

    logger.info("Finished setting up val random negative loader.")

    logger.info(
        f"---Machine {node_rank} local rank {local_rank} validation data loaders initialized"
    )

    model = DistributedDataParallel(
        init_example_gigl_heterogeneous_dblp_model(
            node_type_to_feature_dim=node_type_to_feature_dim,
            edge_type_to_feature_dim=edge_type_to_feature_dim,
            device=training_device,
        ),
        device_ids=[training_device],
        # We should set `find_unused_parameters` to True since not all of the model parameters may be used in backward pass in the heterogeneous setting
        find_unused_parameters=True,
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = RetrievalLoss(
        loss=torch.nn.CrossEntropyLoss(reduction="mean"),
        temperature=0.07,
        remove_accidental_hits=True,
    )
    logger.info(
        f"Model initialized on machine {node_rank} training device {training_device}\n{model.module}"
    )

    # We add a barrier to wait for all processes to finish preparing the dataloader prior to the start of training
    torch.distributed.barrier()

    # Entering the training loop
    training_start_time = time.time()
    batch_idx = 0
    avg_train_loss = 0.0
    last_n_batch_avg_loss: list[float] = []
    last_n_batch_time: list[float] = []
    num_max_train_batches_per_process = (
        num_max_train_batches // training_process_world_size
    )
    num_val_batches_per_process = num_val_batches // training_process_world_size
    logger.info(
        f"num_max_train_batches_per_process is set to {num_max_train_batches_per_process}"
    )

    model.train()

    # start_time gets updated every log_every_n_batch batches, batch_start gets updated every batch
    batch_start = time.time()
    for main_data, random_data in zip(train_main_loader, train_random_negative_loader):
        if (
            num_max_train_batches_per_process
            and batch_idx >= num_max_train_batches_per_process
        ):
            logger.info(
                f"num_max_train_batches_per_process={num_max_train_batches_per_process} reached, "
                f"stopping training on machine {node_rank} local rank {local_rank}"
            )
            break
        if batch_idx % val_every_n_batch == 0:
            logger.info(f"process_rank={local_rank}, batch={batch_idx}, validating...")
            _run_validation_loops(
                local_rank=local_rank,
                model=model,
                main_loader=val_main_loader,
                random_negative_loader=val_random_negative_loader,
                loss_fn=loss_fn,
                supervision_edge_type=supervision_edge_type,
                device=training_device,
                log_every_n_batch=log_every_n_batch,
                num_batches=num_val_batches_per_process,
            )
        loss = _compute_loss(
            model=model,
            main_data=main_data,
            random_negative_data=random_data,
            loss_fn=loss_fn,
            supervision_edge_type=supervision_edge_type,
            device=training_device,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_train_loss = sync_metric_across_processes(metric=loss)
        last_n_batch_avg_loss.append(avg_train_loss)
        last_n_batch_time.append(time.time() - batch_start)
        batch_start = time.time()
        batch_idx += 1
        if batch_idx % log_every_n_batch == 0:
            logger.info(
                f"process_rank={local_rank}, batch={batch_idx}, latest local train_loss={loss:.6f}"
            )
            # Wait for GPU operations to finish
            torch.cuda.synchronize()
            logger.info(
                f"process_rank={local_rank}, batch={batch_idx}, mean(batch_time)={statistics.mean(last_n_batch_time):.3f} sec, max(batch_time)={max(last_n_batch_time):.3f} sec, min(batch_time)={min(last_n_batch_time):.3f} sec"
            )
            last_n_batch_time.clear()
            # log the global average training loss
            logger.info(
                f"process_rank={local_rank}, batch={batch_idx}, latest avg_train_loss={avg_train_loss:.6f}, last {log_every_n_batch} mean(avg_train_loss)={statistics.mean(last_n_batch_avg_loss):.6f}"
            )
            last_n_batch_avg_loss.clear()

    torch.distributed.barrier()

    logger.info(f"---Machine {node_rank} local rank {local_rank} finished training")

    # We save the model on the process with the 0th node rank and 0th local rank.
    if node_rank == 0 and local_rank == 0:
        logger.info(
            f"Training loop finished, took {time.time() - training_start_time:.3f} seconds, saving model to {model_uri}"
        )
        save_state_dict(model=model, save_to_path_uri=model_uri)

    torch.distributed.destroy_process_group()

    # We explicitly delete all the dataloaders to reduce their memory footprint. Otherwise, experimentally we have
    # observed that not all memory may be cleaned up, leading to OOM.
    del train_main_loader, train_random_negative_loader
    del val_main_loader, val_random_negative_loader


@torch.inference_mode()
def _run_validation_loops(
    local_rank: int,
    model: DistributedDataParallel,
    main_loader: Iterator,
    random_negative_loader: Iterator,
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
        local_rank (int): Process number on the current machine
        model (DistributedDataParallel): DDP-wrapped torch model for training and testing
        main_loader (Iterator[Data]): Dataloader for loading main batch data with query and labeled nodes
        random_negative_loader (Iterator[Data]): Dataloader for loading random negative data
        loss_fn (RetrievalLoss): Initialized class to use for loss calculation
        supervision_edge_type (EdgeType): The supervision edge type to use for training in format query_node -> relation -> labeled_node
        device (torch.device): Device to use for training or testing
        log_every_n_batch (int): The frequency we should log batch information when training and validating
        num_batches (Optional[int]): The number of batches to run the validation loop for. If this is not set, this function will loop until the data loaders are exhausted.
            For validation, this field is required to be set, as the data loaders are wrapped with InfiniteIterator.
    """
    logger.info(
        f"Running validation loop on process_rank={local_rank}, log_every_n_batch={log_every_n_batch}, num_batches={num_batches}"
    )
    if num_batches is None:
        if isinstance(main_loader, InfiniteIterator) or isinstance(
            random_negative_loader, InfiniteIterator
        ):
            raise ValueError(
                "Must set `num_batches` field when the provided data loaders are wrapped with InfiniteIterator"
            )

    model.eval()
    batch_idx = 0
    batch_losses: list[float] = []
    last_n_batch_time: list[float] = []
    batch_start = time.time()

    while True:
        if num_batches and batch_idx >= num_batches:
            # If num_batches is set, we stop the validation loop after processing num_batches batches. This is not expected to be used for _testing_process.
            break
        try:
            main_data = next(main_loader)
            random_data = next(random_negative_loader)
        except StopIteration:
            # If the test data loader is exhausted, we stop the test loop.
            # Note that validation dataloaders are infinite, so this is not expected to happen during training.
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
            logger.info(
                f"process_rank={local_rank}, batch={batch_idx}, latest test_loss={loss:.6f}"
            )
            # Wait for GPU operations to finish
            torch.cuda.synchronize()
            logger.info(
                f"process_rank={local_rank}, batch={batch_idx}, mean(batch_time)={statistics.mean(last_n_batch_time):.3f} sec, max(batch_time)={max(last_n_batch_time):.3f} sec, min(batch_time)={min(last_n_batch_time):.3f} sec"
            )
            last_n_batch_time.clear()
    local_avg_loss = statistics.mean(batch_losses)
    logger.info(
        f"process_rank={local_rank} finished validation loop, local loss: {local_avg_loss=:.6f}"
    )
    global_avg_val_loss = sync_metric_across_processes(
        metric=torch.tensor(local_avg_loss, device=device)
    )
    logger.info(
        f"process_rank={local_rank} got global validation loss {global_avg_val_loss=:.6f}"
    )

    # Put the model back in train mode
    model.train()
    return


def _testing_process(
    local_rank: int,
    local_world_size: int,
    node_rank: int,
    node_world_size: int,
    dataset: DistLinkPredictionDataset,
    supervision_edge_type: EdgeType,
    node_type_to_feature_dim: dict[NodeType, int],
    edge_type_to_feature_dim: dict[EdgeType, int],
    master_ip_address: str,
    master_default_process_group_port: int,
    model_uri: Uri,
    subgraph_fanout: list[int],
    sampling_workers_per_process: int,
    main_batch_size: int,
    random_batch_size: int,
    sampling_worker_shared_channel_size: str,
    process_start_gap_seconds: int,
    log_every_n_batch: int,
):
    """
    Each machine spawns process(es) running this function for training a GNN model given some loaded distributed dataset.
    Args:
        local_rank (int): Process number on the current machine
        local_world_size (int): Number of training processes spawned by each machine
        node_rank (int): Rank of the current machine
        node_world_size (int): Total number of machines
        dataset (DistLinkPredictionDataset): Loaded Distributed Dataset for testing
        supervision_edge_type (EdgeType): The supervision edge type to use for training in format query_node -> relation -> labeled_node
        node_type_to_feature_dim (dict[NodeType, int]): Input node feature dimension for the model per node type
        edge_type_to_feature_dim (dict[EdgeType, int]): Input edge feature dimension for the model per edge type
        master_ip_address (str): IP Address of the master worker for distributed communication
        master_default_process_group_port (int): Port on the master worker for setting up distributed process group communication
        model_uri (Uri): URI Path to save the model to
        subgraph_fanout: list[int]: Fanout for subgraph sampling, where the ith item corresponds to the number of items to sample for the ith hop
        sampling_workers_per_process (int): Number of sampling workers per training process
        main_batch_size (int): Batch size for main dataloader with query and labeled nodes
        random_batch_size (int): Batch size for random negative dataloader
        sampling_worker_shared_channel_size (str): Shared-memory buffer size (bytes) allocated for the channel during sampling
        process_start_gap_seconds (int): The amount of time to sleep for initializing each dataloader. For large-scale settings, consider setting this
            field to 30-60 seconds to ensure dataloaders don't compete for memory during initialization, causing OOM.
        log_every_n_batch (int): The frequency we should log batch information when training and validating
    """
    # TODO (mkolodner-sc): Investigate work needed + add support for CPU training
    if not torch.cuda.is_available():
        raise NotImplementedError(
            "Currently, only GPU training is supported with this example training loop"
        )

    test_process_world_size = node_world_size * local_world_size
    test_process_global_rank = node_rank * local_world_size + local_rank
    logger.info(
        f"---Current test process global rank: {test_process_global_rank}, test process world size: {test_process_world_size}"
    )

    # We initialize distributed process group to connect all GPUs across all machines so that the final metrics can be synchronized
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_ip_address}:{master_default_process_group_port}",
        rank=test_process_global_rank,
        world_size=test_process_world_size,
    )
    logger.info(
        f"---Machine {node_rank} local rank {local_rank} test process group initialized"
    )
    logger.info(f"---Machine {node_rank} local rank {local_rank} test process started")

    # We use one testing device for each local process
    test_device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    torch.cuda.set_device(test_device)
    logger.info(
        f"---Machine {node_rank} local rank {local_rank} test process set device {test_device}"
    )

    # We initialize the test dataloaders in order: main_loader_0, random_loader_0, main_loader_1, random_loader_1, ...
    # where the number indicates the local process rank. There is a `process_start_gap_seconds / 2` time.sleep() call in between each of
    # these initializations. We do this since initializing these NeighborLoaders creates a spike in memory usage, and initializing multiple
    # loaders at the same time can lead to OOM.

    query_node_type = supervision_edge_type.src_node_type

    labeled_node_type = supervision_edge_type.dst_node_type

    assert isinstance(dataset.test_node_ids, Mapping)

    # When initializing a DistABLPLoader for heterogeneous use cases, it is important that `input_nodes` be a Tuple(NodeType, torch.Tensor) to indicate that we should
    # be loading heterogeneous data. The NodeType in this case should be the query node type. Additionally, it is important that `supervision_edge_type` be provided so
    # that the dataloader knows how to fanout around the labeled edge types. A heterogeneous DistABLPLoader will return a HeteroData object.
    test_main_loader: Iterator[HeteroData] = DistABLPLoader(
        dataset=dataset,
        num_neighbors=subgraph_fanout,
        input_nodes=(query_node_type, dataset.test_node_ids[query_node_type]),
        supervision_edge_type=supervision_edge_type,
        num_workers=sampling_workers_per_process,
        batch_size=main_batch_size,
        pin_memory_device=test_device,
        worker_concurrency=sampling_workers_per_process,
        channel_size=sampling_worker_shared_channel_size,
        # Each test_main_loader will wait for `process_start_gap_seconds` * `local_process_rank` seconds before initializing to reduce peak memory usage.
        process_start_gap_seconds=process_start_gap_seconds,
    )

    test_main_loader = iter(test_main_loader)

    logger.info("Finished setting up test main loader.")

    # The random_negative_loader will wait for `process_start_gap_seconds / 2` seconds after initializing the main_loader so that it doesn't interfere with other processes.
    time.sleep(process_start_gap_seconds / 2)

    assert isinstance(dataset.node_ids, Mapping)

    # Similarly, the heterogeneous DistNeighborLoader should be provided as a Tuple[NodeType, torch.Tensor], where the NodeType is the node type of
    # the labeled node in the link prediction task.
    test_random_negative_loader: Iterator[HeteroData] = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=subgraph_fanout,
        input_nodes=(labeled_node_type, dataset.node_ids[labeled_node_type]),
        num_workers=sampling_workers_per_process,
        batch_size=random_batch_size,
        pin_memory_device=test_device,
        worker_concurrency=sampling_workers_per_process,
        channel_size=sampling_worker_shared_channel_size,
        process_start_gap_seconds=0,
    )

    test_random_negative_loader = iter(test_random_negative_loader)

    logger.info("Finished setting up test random negative loader.")

    model_state_dict = load_state_dict_from_uri(
        load_from_uri=model_uri, device=test_device
    )

    model = DistributedDataParallel(
        init_example_gigl_heterogeneous_dblp_model(
            node_type_to_feature_dim=node_type_to_feature_dim,
            edge_type_to_feature_dim=edge_type_to_feature_dim,
            device=test_device,
            state_dict=model_state_dict,
        ),
        device_ids=[test_device],
    )

    logger.info(
        f"Model initialized on machine {node_rank} test device {test_device} with weights loaded from {model_uri.uri}\n{model}"
    )
    loss_fn = RetrievalLoss(
        loss=torch.nn.CrossEntropyLoss(reduction="mean"),
        temperature=0.07,
        remove_accidental_hits=True,
    )

    # We add a barrier to wait for all processes to finish preparing the dataloader prior to the start of training
    torch.distributed.barrier()

    _run_validation_loops(
        local_rank=local_rank,
        model=model,
        main_loader=test_main_loader,
        random_negative_loader=test_random_negative_loader,
        loss_fn=loss_fn,
        supervision_edge_type=supervision_edge_type,
        device=test_device,
        log_every_n_batch=log_every_n_batch,
    )

    torch.distributed.destroy_process_group()

    # We explicitly delete all the dataloaders to reduce their memory footprint. Otherwise, experimentally we have
    # observed that not all memory may be cleaned up, leading to OOM.
    del test_main_loader, test_random_negative_loader


def _run_example_training(
    task_config_uri: str,
):
    """
    Runs an example training + testing loop using GiGL Orchestration.
    Args:
        task_config_uri (str): Path to YAML-serialized GbmlConfig proto.
    """
    start_time = time.time()
    mp.set_start_method("spawn")
    logger.info(f"Starting sub process method: {mp.get_start_method()}")

    torch.distributed.init_process_group(backend="gloo")

    master_ip_address = gigl.distributed.utils.get_internal_ip_from_master_node()
    node_rank = torch.distributed.get_rank()
    node_world_size = torch.distributed.get_world_size()
    master_default_process_group_ports = (
        gigl.distributed.utils.get_free_ports_from_master_node(num_ports=2)
    )
    master_process_group_port_for_training = master_default_process_group_ports[0]
    master_process_group_port_for_testing = master_default_process_group_ports[1]
    # Destroying the process group as one will be re-initialized in the training process using above information
    torch.distributed.destroy_process_group()

    logger.info(f"--- Launching data loading process ---")
    dataset = build_dataset_from_task_config_uri(
        task_config_uri=task_config_uri,
        is_inference=False,
    )
    logger.info(
        f"--- Data loading process finished, took {time.time() - start_time:.3f} seconds"
    )
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=UriFactory.create_uri(task_config_uri)
    )

    trainer_args = dict(gbml_config_pb_wrapper.trainer_config.trainer_args)

    local_world_size = int(trainer_args.get("local_world_size", "1"))
    if torch.cuda.is_available():
        if local_world_size > torch.cuda.device_count():
            raise ValueError(
                f"Specified a local world size of {local_world_size} which exceeds the number of devices {torch.cuda.device_count()}"
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

    model_uri = UriFactory.create_uri(
        gbml_config_pb_wrapper.gbml_config_pb.shared_config.trained_model_metadata.trained_model_uri
    )

    supervision_edge_types = (
        gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_types()
    )
    if len(supervision_edge_types) != 1:
        raise NotImplementedError(
            "GiGL Training currently only supports 1 supervision edge type."
        )
    supervision_edge_type = supervision_edge_types[0]

    # Training Hyperparameters shared among the training and test processes
    fanout_per_hop = int(trainer_args.get("fanout_per_hop", "10"))

    # If `subgraph_fanout` is provided as a list, it will use that fanout for each edge type. Otherwise, subgraph fanout should be provided as a
    # dict[EdgeType, list[int]], indicating the fanout behavior for each edge type. Different edge types can have different fanout strategies. For simplicity,
    # we make an opinionated decision to keep the fanouts for all edge types the same, specifying the `subraph_fanout` with a `list[int]`.

    # TODO (mkolodner-sc): Make the heterogeneous example use a dict[EdgeType, list[int]] instead of list[int] for specifying fanout
    # For example:
    # subgraph_fanout: dict[EdgeType, list[int]] = {
    #    EdgeType(NodeType("author"), Relation("to"), NodeType("paper")): [10, 10].
    #    EdgeType(NodeType("term"), Relation("to"), NodeType("paper")): [10, 10].
    # }
    subgraph_fanout: list[int] = [fanout_per_hop, fanout_per_hop]

    # While the ideal value for `sampling_workers_per_process` has been identified to be between `2` and `4`, this may need some tuning depending on the
    # pipeline. We default this value to `4` here for simplicity. A `sampling_workers_per_process` which is too small may not have enough parallelization for
    # sampling, which would slow down training, while a value which is too large may slow down each sampling process due to competing resources, which would also
    # then slow down training.

    sampling_workers_per_process: int = int(
        trainer_args.get("sampling_workers_per_process", "4")
    )

    main_batch_size = int(trainer_args.get("main_batch_size", 16))
    random_batch_size = int(trainer_args.get("random_batch_size", 16))

    # This value represents the the shared-memory buffer size (bytes) allocated for the channel during sampling, and
    # is the place to store pre-fetched data, so if it is too small then prefetching is limited, causing sampling slowdown.
    # This parameter is a string with `{numeric_value}{storage_size}`, where storage size could be `MB`, `GB`, etc. We default this value to 4GB,
    # but in production may need some tuning.
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
        fanout_per_hop={fanout_per_hop}, \
        sampling_workers_per_process={sampling_workers_per_process}, \
        main_batch_size={main_batch_size}, \
        random_batch_size={random_batch_size}, \
        sampling_worker_shared_channel_size={sampling_worker_shared_channel_size}, \
        process_start_gap_seconds={process_start_gap_seconds}, \
        log_every_n_batch={log_every_n_batch}, \
        learning_rate={learning_rate}, \
        weight_decay={weight_decay}, \
        num_max_train_batches={num_max_train_batches}, \
        num_val_batches={num_val_batches}, \
        val_every_n_batch={val_every_n_batch}"
    )

    logger.info("--- Launching training processes ...\n")
    start_time = time.time()
    torch.multiprocessing.spawn(
        _training_process,
        args=(  # Corresponding arguments in `_training_process` function
            local_world_size,  # local_world_size
            node_rank,  # node_rank
            node_world_size,  # node_world_size
            dataset,  # dataset
            supervision_edge_type,  # supervision_edge_type
            node_type_to_feature_dim,  # node_type_to_feature_dim
            edge_type_to_feature_dim,  # edge_type_to_feature_dim
            master_ip_address,  # master_ip_address
            master_process_group_port_for_training,  # master_default_process_group_port
            model_uri,  # model_uri
            subgraph_fanout,  # subgraph_fanout
            sampling_workers_per_process,  # sampling_workers_per_process
            main_batch_size,  # main_batch_size
            random_batch_size,  # random_batch_size
            sampling_worker_shared_channel_size,  # sampling_worker_shared_channel_size
            process_start_gap_seconds,  # process_start_gap_seconds
            log_every_n_batch,  # log_every_n_batch
            learning_rate,  # learning_rate
            weight_decay,  # weight_decay
            num_max_train_batches,  # num_max_train_batches
            num_val_batches,  # num_val_batches
            val_every_n_batch,  # val_every_n_batch
        ),
        nprocs=local_world_size,
        join=True,
    )
    logger.info(f"--- Training finished, took {time.time() - start_time} seconds")
    logger.info("--- Launching test processes ...\n")
    start_time = time.time()
    torch.multiprocessing.spawn(
        _testing_process,
        args=(  # Corresponding arguments in `_testing_process` function
            local_world_size,  # local_world_size
            node_rank,  # node_rank
            node_world_size,  # node_world_size
            dataset,  # dataset
            supervision_edge_type,  # supervision_edge_type
            node_type_to_feature_dim,  # node_type_to_feature_dim
            edge_type_to_feature_dim,  # edge_type_to_feature_dim
            master_ip_address,  # master_ip_address
            master_process_group_port_for_testing,  # master_default_process_group_port
            model_uri,  # model_uri
            subgraph_fanout,  # subgraph_fanout
            sampling_workers_per_process,  # sampling_workers_per_process
            main_batch_size,  # main_batch_size
            random_batch_size,  # random_batch_size
            sampling_worker_shared_channel_size,  # sampling_worker_shared_channel_size
            process_start_gap_seconds,  # process_start_gap_seconds
            log_every_n_batch,  # log_every_n_batch
        ),
        nprocs=local_world_size,
        join=True,
    )
    logger.info(f"--- Testing finished, took {time.time() - start_time} seconds")
    logger.info("--- All steps finished, exiting main ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for distributed model training on VertexAI"
    )
    parser.add_argument("--task_config_uri", type=str, help="Gbml config uri")

    # We use parse_known_args instead of parse_args since we only need job_name and task_config_uri for distributed trainer
    args, unused_args = parser.parse_known_args()
    logger.info(f"Unused arguments: {unused_args}")

    # We only need `task_config_uri` for running trainer
    _run_example_training(
        task_config_uri=args.task_config_uri,
    )
