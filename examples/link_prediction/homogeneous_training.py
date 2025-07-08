"""
This file contains an example for how to run homogeneous training using live subgraph sampling powered by GraphLearn-for-PyTorch (GLT).
While `run_example_training` is coupled with GiGL orchestration, the `_training_process` and `testing_process` functions are generic
and can be used as references for writing training for pipelines not dependent on GiGL orchestration.

To run this file with GiGL orchestration, set the fields similar to below:

trainerConfig:
  trainerArgs:
    # Example argument to trainer
    log_every_n_batch: "50"
  command: python -m examples.link_prediction.homogeneous_training
featureFlags:
  should_run_glt_backend: 'True'

You can run this example in a full pipeline with `make run_hom_cora_sup_test` from GiGL root.
TODO (mkolodner-sc): Add example of how to run locally once CPU support is enabled
"""

import argparse
import statistics
import time
from collections.abc import Iterator
from typing import Literal, Optional

import torch
import torch.distributed
import torch.multiprocessing as mp
from examples.link_prediction.models import init_example_gigl_homogeneous_model
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Data

import gigl.distributed.utils
from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.torch_training import is_distributed_available_and_initialized
from gigl.distributed import (
    DistABLPLoader,
    DistLinkPredictionDataset,
    build_dataset_from_task_config_uri,
)
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.module.loss import RetrievalLoss
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.model import load_state_dict_from_uri, save_state_dict
from gigl.types.graph import to_homogeneous
from gigl.utils.iterator import InfiniteIterator

logger = Logger()


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
    dataset: DistLinkPredictionDataset,
    split: Literal["train", "val", "test"],
    subgraph_fanout: list[int],
    sampling_workers_per_process: int,
    main_batch_size: int,
    random_batch_size: int,
    device: torch.device,
    sampling_worker_shared_channel_size: str,
    process_start_gap_seconds: int,
) -> tuple[Iterator[Data], Iterator[Data]]:
    """
    Sets up main and random dataloaders for training and testing purposes
    Args:
        dataset (DistLinkPredictionDataset): Loaded Distributed Dataset for training and testing
        split (Literal["train", "val", "test"]): The current split which we are loading data for
        subgraph_fanout: list[int]: Fanout for subgraph sampling, where the ith item corresponds to the number of items to sample for the ith hop
        sampling_workers_per_process (int): sampling_workers_per_process (int): Number of sampling workers per training/testing process
        main_batch_size (int): Batch size for main dataloader with query and labeled nodes
        random_batch_size (int): Batch size for random negative dataloader
        device (torch.device): Device to put loaded subgraphs on
        sampling_worker_shared_channel_size (str): Shared-memory buffer size (bytes) allocated for the channel during sampling
        process_start_gap_seconds (int): The amount of time to sleep for initializing each dataloader. For large-scale settings, consider setting this
            field to 30-60 seconds to ensure dataloaders don't compete for memory during initialization, causing OOM.
    Returns:
        Iterator[Data]: Dataloader for loading main batch data with query and labeled nodes
        Iterator[Data]: Dataloader for loading random negative data
    """

    rank = torch.distributed.get_rank()

    if split == "train":
        main_input_nodes = to_homogeneous(dataset.train_node_ids)
        shuffle = True
    elif split == "val":
        main_input_nodes = to_homogeneous(dataset.val_node_ids)
        shuffle = False
    else:
        main_input_nodes = to_homogeneous(dataset.test_node_ids)
        shuffle = False

    main_loader: Iterator[Data] = DistABLPLoader(
        dataset=dataset,
        num_neighbors=subgraph_fanout,
        input_nodes=main_input_nodes,
        num_workers=sampling_workers_per_process,
        batch_size=main_batch_size,
        pin_memory_device=device,
        worker_concurrency=sampling_workers_per_process,
        channel_size=sampling_worker_shared_channel_size,
        # Each main_loader will wait for `process_start_gap_seconds` * `local_process_rank` seconds before initializing to reduce peak memory usage.
        # This is done so that each process on the current machine which initializes a `main_loader` doesn't compete for memory, causing potential OOM
        process_start_gap_seconds=process_start_gap_seconds,
        shuffle=shuffle,
    )

    if split == "test":
        main_loader = iter(main_loader)
    else:
        main_loader = InfiniteIterator(main_loader)

    logger.info(f"---Rank {rank} finished setting up main loader")

    # We need to wait for all processes to finish initializing the main_loader before creating the random_negative_loader so that its initialization doesn't compete for memory with the main_loader, causing potential OOM.
    torch.distributed.barrier()

    random_negative_loader: Iterator[Data] = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=subgraph_fanout,
        input_nodes=to_homogeneous(dataset.node_ids),
        num_workers=sampling_workers_per_process,
        batch_size=random_batch_size,
        pin_memory_device=device,
        worker_concurrency=sampling_workers_per_process,
        channel_size=sampling_worker_shared_channel_size,
        process_start_gap_seconds=process_start_gap_seconds,
        shuffle=shuffle,
    )

    # If we are doing testing, we only want to go through the data once.
    if split == "test":
        random_negative_loader = iter(random_negative_loader)
    # Otherwise, we will want to continue looping and should use an InfiniteIterator
    else:
        random_negative_loader = InfiniteIterator(random_negative_loader)

    logger.info(f"--Rank {rank} finished setting up random negative loader")

    # Wait for all processes to finish initializing the random_loader
    torch.distributed.barrier()

    return main_loader, random_negative_loader


def _compute_loss(
    model: DistributedDataParallel,
    main_data: Data,
    random_negative_data: Data,
    loss_fn: RetrievalLoss,
    device: torch.device,
) -> torch.Tensor:
    """
    With the provided model and loss function, computes the forward pass on the main batch data and random negative data.
    Args:
        model (DistributedDataParallel): DDP-wrapped torch model for training and testing
        main_data (Data): The batch of data containing query nodes, positive nodes, and hard negative nodes
        random_negative_data (Data): The batch of data containing random negative nodes
        loss_fn (RetrievalLoss): Initialized class to use for loss calculation
        device (torch.device): Device for training or validation
    Returns:
        torch.Tensor: Final loss for the current batch on the current process
    """
    # Forward pass through encoder
    main_embeddings = model(data=main_data, device=device)
    random_negative_embeddings = model(data=random_negative_data, device=device)

    # Extracting local query, random negative, positive, hard_negative, and random_negative indices.
    # Local in this case refers to the local index in the batch, while global subsequently refers to the node's unique global ID across all nodes in the dataset.
    # Global ids are stored in data.node, ex. `data.node = [50, 20, 10]` where the `0` is the local index for global id `50`. Note that each global id in data.node is unique.
    query_node_idx: torch.Tensor = torch.arange(main_data.batch_size).to(device)
    random_negative_batch_size = random_negative_data.batch_size

    # main_data.y_positive is a dict[query_node_local_index: int, labeled_node_local_indices: torch.Tensor]
    positive_idx: torch.Tensor = torch.cat(list(main_data.y_positive.values())).to(
        device
    )
    # We also extract a repeated query node index tensor which upsamples each query node based on the number of positives it has
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

    repeated_query_embeddings = main_embeddings[repeated_query_node_idx]
    positive_node_embeddings = main_embeddings[positive_idx]
    hard_negative_embeddings = main_embeddings[hard_negative_idx]
    random_negative_embeddings = random_negative_embeddings[:random_negative_batch_size]

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
            main_data.node[positive_idx],
            main_data.node[hard_negative_idx],
            random_negative_data.node[:random_negative_batch_size],
        )
    )

    global_repeated_query_ids = main_data.node[repeated_query_node_idx]

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
    machine_rank: int,
    machine_world_size: int,
    dataset: DistLinkPredictionDataset,
    node_feature_dim: int,
    edge_feature_dim: int,
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
    should_skip_training: bool,
) -> None:
    """
    This function is spawned by each machine for training a GNN model given some loaded distributed dataset.
    Args:
        local_rank (int): Process number on the current machine
        local_world_size (int): Number of training processes spawned by each machine
        machine_rank (int): Rank of the current machine
        machine_world_size (int): Total number of machines
        dataset (DistLinkPredictionDataset): Loaded Distributed Dataset for training
        node_feature_dim (int): Input node feature dimension for the model
        edge_feature_dim (int): Input edge feature dimension for the model
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
        should_skip_training (bool): Whether training should be skipped and we should only run testing. Assumes model has been uploaded to the model_uri.
    """
    # TODO (mkolodner-sc): Investigate work needed + add support for CPU training
    if not torch.cuda.is_available():
        raise NotImplementedError(
            "Currently, only GPU training is supported with this example training loop"
        )

    world_size = machine_world_size * local_world_size
    rank = machine_rank * local_world_size + local_rank
    logger.info(
        f"---Current training process rank: {rank}, training process world size: {world_size}"
    )

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_ip_address}:{master_default_process_group_port}",
        rank=rank,
        world_size=world_size,
    )

    logger.info(f"---Rank {rank} training process started")

    # We use one training device for each local process
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    logger.info(f"---Rank {rank}  training process set device {device}")

    logger.info(f"---Rank {rank} training process group initialized")

    loss_fn = RetrievalLoss(
        loss=torch.nn.CrossEntropyLoss(reduction="mean"),
        temperature=0.07,
        remove_accidental_hits=True,
    )

    if not should_skip_training:
        train_main_loader, train_random_negative_loader = _setup_dataloaders(
            dataset=dataset,
            split="train",
            subgraph_fanout=subgraph_fanout,
            sampling_workers_per_process=sampling_workers_per_process,
            main_batch_size=main_batch_size,
            random_batch_size=random_batch_size,
            device=device,
            sampling_worker_shared_channel_size=sampling_worker_shared_channel_size,
            process_start_gap_seconds=process_start_gap_seconds,
        )

        val_main_loader, val_random_negative_loader = _setup_dataloaders(
            dataset=dataset,
            split="val",
            subgraph_fanout=subgraph_fanout,
            sampling_workers_per_process=sampling_workers_per_process,
            main_batch_size=main_batch_size,
            random_batch_size=random_batch_size,
            device=device,
            sampling_worker_shared_channel_size=sampling_worker_shared_channel_size,
            process_start_gap_seconds=process_start_gap_seconds,
        )
        assert isinstance(train_main_loader, InfiniteIterator)
        assert isinstance(train_random_negative_loader, InfiniteIterator)
        assert isinstance(val_main_loader, InfiniteIterator)
        assert isinstance(val_random_negative_loader, InfiniteIterator)

        model = DistributedDataParallel(
            init_example_gigl_homogeneous_model(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                device=device,
            ),
            device_ids=[device],
        )

        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        logger.info(
            f"Model initialized on rank {rank} training device {device}\n{model.module}"
        )

        # We add a barrier to wait for all processes to finish preparing the dataloader and initializing the model prior to the start of training
        torch.distributed.barrier()

        # Entering the training loop
        training_start_time = time.time()
        batch_idx = 0
        avg_train_loss = 0.0
        last_n_batch_avg_loss: list[float] = []
        last_n_batch_time: list[float] = []
        num_max_train_batches_per_process = num_max_train_batches // world_size
        num_val_batches_per_process = num_val_batches // world_size
        logger.info(
            f"num_max_train_batches_per_process is set to {num_max_train_batches_per_process}"
        )

        model.train()

        # start_time gets updated every log_every_n_batch batches, batch_start gets updated every batch
        batch_start = time.time()
        for main_data, random_data in zip(
            train_main_loader, train_random_negative_loader
        ):
            if batch_idx >= num_max_train_batches_per_process:
                logger.info(
                    f"num_max_train_batches_per_process={num_max_train_batches_per_process} reached, "
                    f"stopping training on machine {machine_rank} local rank {local_rank}"
                )
                break
            loss = _compute_loss(
                model=model,
                main_data=main_data,
                random_negative_data=random_data,
                loss_fn=loss_fn,
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
            if batch_idx % log_every_n_batch == 0:
                logger.info(
                    f"rank={rank}, batch={batch_idx}, latest local train_loss={loss:.6f}"
                )
                # Wait for GPU operations to finish
                torch.cuda.synchronize()
                logger.info(
                    f"rank={rank}, mean(batch_time)={statistics.mean(last_n_batch_time):.3f} sec, max(batch_time)={max(last_n_batch_time):.3f} sec, min(batch_time)={min(last_n_batch_time):.3f} sec"
                )
                last_n_batch_time.clear()
                # log the global average training loss
                logger.info(
                    f"rank={rank}, latest avg_train_loss={avg_train_loss:.6f}, last {log_every_n_batch} mean(avg_train_loss)={statistics.mean(last_n_batch_avg_loss):.6f}"
                )
                last_n_batch_avg_loss.clear()

            if batch_idx % val_every_n_batch == 0:
                logger.info(f"rank={rank}, batch={batch_idx}, validating...")
                model.eval()
                _run_validation_loops(
                    model=model,
                    main_loader=val_main_loader,
                    random_negative_loader=val_random_negative_loader,
                    loss_fn=loss_fn,
                    device=device,
                    log_every_n_batch=log_every_n_batch,
                    num_batches=num_val_batches_per_process,
                )
                model.train()

        logger.info(f"---Rank {rank} finished training")

        # Memory cleanup and waiting for all processes to finish
        torch.cuda.empty_cache()  # Releases all unoccupied cached memory currently held by the caching allocator on the CUDA-enabled GPU
        torch.cuda.synchronize()  # Ensures all CUDA operations have finished
        torch.distributed.barrier()  # Waits for all processes to reach the current point

        # We explicitly shutdown all the dataloaders to reduce their memory footprint. Otherwise, experimentally we have
        # observed that not all memory may be cleaned up, leading to OOM.
        train_main_loader.shutdown()
        train_random_negative_loader.shutdown()
        val_main_loader.shutdown()
        val_random_negative_loader.shutdown()
    else:
        state_dict = load_state_dict_from_uri(load_from_uri=model_uri, device=device)
        model = DistributedDataParallel(
            init_example_gigl_homogeneous_model(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                device=device,
                state_dict=state_dict,
            ),
            device_ids=[device],
        )
        logger.info(
            f"Model initialized on rank {rank} training device {device}\n{model.module}"
        )

    logger.info(f"---Rank {rank} started testing")

    model.eval()

    test_main_loader, test_random_negative_loader = _setup_dataloaders(
        dataset=dataset,
        split="test",
        subgraph_fanout=subgraph_fanout,
        sampling_workers_per_process=sampling_workers_per_process,
        main_batch_size=main_batch_size,
        random_batch_size=random_batch_size,
        device=device,
        sampling_worker_shared_channel_size=sampling_worker_shared_channel_size,
        process_start_gap_seconds=process_start_gap_seconds,
    )

    _run_validation_loops(
        model=model,
        main_loader=test_main_loader,
        random_negative_loader=test_random_negative_loader,
        loss_fn=loss_fn,
        device=device,
        log_every_n_batch=log_every_n_batch,
    )

    logger.info(f"---Rank {rank} finished testing")

    # Memory cleanup and waiting for all processes to finish
    torch.cuda.empty_cache()  # Releases all unoccupied cached memory currently held by the caching allocator on the CUDA-enabled GPU
    torch.cuda.synchronize()  # Ensures all CUDA operations have finished
    torch.distributed.barrier()  # Waits for all processes to reach the current point

    # We save the model on the process with the 0th node rank and 0th local rank.
    if machine_rank == 0 and local_rank == 0:
        logger.info(
            f"Training loop finished, took {time.time() - training_start_time:.3f} seconds, saving model to {model_uri}"
        )
        save_state_dict(model=model, save_to_path_uri=model_uri)

    torch.distributed.destroy_process_group()


@torch.inference_mode()
def _run_validation_loops(
    model: DistributedDataParallel,
    main_loader: Iterator,
    random_negative_loader: Iterator,
    loss_fn: RetrievalLoss,
    device: torch.device,
    log_every_n_batch: int,
    num_batches: Optional[int] = None,
) -> None:
    """
    Runs validation using the provided models and dataloaders.
    This function is shared for both validation while training and testing after training has completed.
    Args:
        model (DistributedDataParallel): DDP-wrapped torch model for training and testing
        main_loader (Iterator[Data]): Dataloader for loading main batch data with query and labeled nodes
        random_negative_loader (Iterator[Data]): Dataloader for loading random negative data
        loss_fn (RetrievalLoss): Initialized class to use for loss calculation
        device (torch.device): Device to use for training or testing
        log_every_n_batch (int): The frequency we should log batch information when training and validating
        num_batches (Optional[int]): The number of batches to run the validation loop for. If this is not set, this function will loop until the data loaders are exhausted.
            For validation, this field is required to be set, as the data loaders are wrapped with InfiniteIterator.
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
            device=device,
        )

        batch_losses.append(loss.item())
        last_n_batch_time.append(time.time() - batch_start)
        batch_start = time.time()
        batch_idx += 1
        if batch_idx % log_every_n_batch == 0:
            logger.info(f"rank={rank}, batch={batch_idx}, latest test_loss={loss:.6f}")
            # Wait for GPU operations to finish
            torch.cuda.synchronize()
            logger.info(
                f"rank={rank}, batch={batch_idx}, mean(batch_time)={statistics.mean(last_n_batch_time):.3f} sec, max(batch_time)={max(last_n_batch_time):.3f} sec, min(batch_time)={min(last_n_batch_time):.3f} sec"
            )
            last_n_batch_time.clear()
    local_avg_loss = statistics.mean(batch_losses)
    logger.info(
        f"rank={rank} finished validation loop, local loss: {local_avg_loss=:.6f}"
    )
    global_avg_val_loss = _sync_metric_across_processes(
        metric=torch.tensor(local_avg_loss, device=device)
    )
    logger.info(f"rank={rank} got global validation loss {global_avg_val_loss=:.6f}")

    return


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

    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=UriFactory.create_uri(task_config_uri)
    )

    # Training Hyperparameters for the training and test processes

    trainer_args = dict(gbml_config_pb_wrapper.trainer_config.trainer_args)

    local_world_size = int(trainer_args.get("local_world_size", "1"))
    if torch.cuda.is_available():
        if local_world_size > torch.cuda.device_count():
            raise ValueError(
                f"Specified a local world size of {local_world_size} which exceeds the number of devices {torch.cuda.device_count()}"
            )

    fanout_per_hop = int(trainer_args.get("fanout_per_hop", "10"))
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
    # is the place to store pre-fetched data, so if it is too small then prefetching is limited, causing sampling slowdown. This parameter is a string
    # with `{numeric_value}{storage_size}`, where storage size could be `MB`, `GB`, etc. We default this value to 4GB,
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

    # This `init_process_group` is only called to get the master_ip_address, master port, and rank/world_size fields which help with partitioning, sampling,
    # and distributed training/testing. We can use `gloo` here since these fields we are extracting don't require GPU capabilities provided by `nccl`.
    # Note that this init_process_group uses env:// to setup the connection.
    # In VAI we create one process per node thus these variables are exposed through env i.e. MASTER_PORT , MASTER_ADDR , WORLD_SIZE , RANK that VAI sets up for us.
    # If running locally, these env variables will need to be setup by the user manually.
    torch.distributed.init_process_group(backend="gloo")

    master_ip_address = gigl.distributed.utils.get_internal_ip_from_master_node()
    machine_rank = torch.distributed.get_rank()
    machine_world_size = torch.distributed.get_world_size()
    master_default_process_group_port = (
        gigl.distributed.utils.get_free_ports_from_master_node(num_ports=1)
    )[0]
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

    graph_metadata = gbml_config_pb_wrapper.graph_metadata_pb_wrapper

    node_feature_dim = gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_dim_map[
        graph_metadata.homogeneous_condensed_node_type
    ]
    edge_feature_dim = gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper.condensed_edge_type_to_feature_dim_map[
        graph_metadata.homogeneous_condensed_edge_type
    ]

    model_uri = UriFactory.create_uri(
        gbml_config_pb_wrapper.gbml_config_pb.shared_config.trained_model_metadata.trained_model_uri
    )

    should_skip_training = gbml_config_pb_wrapper.shared_config.should_skip_training

    logger.info("--- Launching training processes ...\n")
    start_time = time.time()
    torch.multiprocessing.spawn(
        _training_process,
        args=(  # Corresponding arguments in `_training_process` function
            local_world_size,  # local_world_size
            machine_rank,  # machine_rank
            machine_world_size,  # machine_world_size
            dataset,  # dataset
            node_feature_dim,  # node_feature_dim
            edge_feature_dim,  # edge_feature_dim
            master_ip_address,  # master_ip_address
            master_default_process_group_port,  # master_default_process_group_port
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
            should_skip_training,  # should_skip_training
        ),
        nprocs=local_world_size,
        join=True,
    )
    logger.info(f"--- Training finished, took {time.time() - start_time} seconds")


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
