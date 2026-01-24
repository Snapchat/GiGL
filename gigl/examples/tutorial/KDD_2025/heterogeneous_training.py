"""
Dead simple heterogenous training loop for CPU training with DistributedDataParallel.

Supports multi process/multi node training.
Does not support GPU training.

Run with:
    python -m gigl.examples.tutorial.KDD_2025.heterogeneous_training --task_config_uri <path_to_frozen_task_config>

To generate a frozen config from a template task config, see instructions at top of `gigl/examples/tutorial/KDD_2025/task_config.yaml`.


This example is meant to be run on the "toy graph" dataset,
which is a small heterogeneous graph with two node types (user) and (story)
and two edge types (user to story) and (story to user).

The dataset it reads from may be configured in the `task_config_uri` argument,
if using a different dataset, also update the following fields:
 - QUERY_NODE_TYPE
 - TARGET_NODE_TYPE
 - SUPERVISION_EDGE_TYPE
 and the metadata in the `init_model` function.

Args:
    --task_config_uri: Path to the task config URI.
    --torch_process_group_init_method: Method to initialize the torch process group.
    --process_count: Number of processes to spawn.
    --batch_size: Batch size for training and validation.
    --val_every: Run validation every N batches.
    --use_local_saved_model: Use a local saved model instead of a remote URI.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs isort: skip


import argparse
from collections.abc import Iterable, Mapping
from distutils.util import strtobool
from typing import Literal

import torch
from gigl.examples.tutorial.KDD_2025.utils import LOCAL_SAVED_MODEL_URI, init_model
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import HeteroData

from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.distributed import (
    DistABLPLoader,
    DistDataset,
    build_dataset_from_task_config_uri,
)
from gigl.distributed.utils import get_free_port
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.model import save_state_dict
from gigl.utils.iterator import InfiniteIterator

logger = Logger()

# Make sure to update these if using a different dataset.
QUERY_NODE_TYPE = NodeType("user")
TARGET_NODE_TYPE = NodeType("story")

SUPERVISION_EDGE_TYPE = EdgeType(QUERY_NODE_TYPE, Relation("to"), TARGET_NODE_TYPE)

# Arbitrary fanout for the example, can be adjusted based on dataset and model.
FANOUT = [10, 10]


def compute_loss(model: torch.nn.Module, data: HeteroData) -> torch.Tensor:
    main_out: dict[str, torch.Tensor] = model(data.x_dict, data.edge_index_dict)
    anchor_nodes = torch.arange(data[QUERY_NODE_TYPE].batch_size).repeat_interleave(
        torch.tensor([len(v) for v in data.y_positive.values()])
    )
    target_nodes = torch.cat([v for v in data.y_positive.values()])
    loss_fn = torch.nn.MarginRankingLoss()
    query_embeddings = main_out[QUERY_NODE_TYPE][anchor_nodes]
    target_embeddings = main_out[TARGET_NODE_TYPE][target_nodes]
    loss = loss_fn(
        input1=query_embeddings,
        input2=target_embeddings,
        target=torch.ones_like(query_embeddings, dtype=torch.float32),
    )
    return loss


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    loader: Iterable[HeteroData],
    num_val_batches: int = 1,
) -> float:
    model.eval()
    total_loss = 0.0
    for batch_idx, data in enumerate(loader):
        loss = compute_loss(model, data)
        total_loss += loss.item()
        if batch_idx >= num_val_batches - 1:
            break
    torch.distributed.all_reduce(
        torch.tensor(total_loss / num_val_batches), op=torch.distributed.ReduceOp.SUM
    )
    total_loss = total_loss / torch.distributed.get_world_size()
    logger.info(
        f"Validation loss: {total_loss:.3f} (averaged over {num_val_batches} batches)"
    )
    return total_loss


def get_data_loader(
    split: Literal["train", "val", "test"],
    dataset: DistDataset,
    batch_size: int,
) -> Iterable[HeteroData]:
    node_type = QUERY_NODE_TYPE
    if split == "train":
        assert isinstance(dataset.train_node_ids, Mapping)
        input_nodes = (node_type, dataset.train_node_ids[node_type])
    elif split == "val":
        assert isinstance(dataset.val_node_ids, Mapping)
        input_nodes = (node_type, dataset.val_node_ids[node_type])
    elif split == "test":
        assert isinstance(dataset.test_node_ids, Mapping)
        input_nodes = (node_type, dataset.test_node_ids[node_type])
    else:
        raise ValueError(f"Unknown split: {split}")

    # Wrap with InfiniteIterator to fully support distributed training.
    return InfiniteIterator(
        DistABLPLoader(
            dataset=dataset,
            num_neighbors=FANOUT,
            input_nodes=input_nodes,
            batch_size=batch_size,
            supervision_edge_type=SUPERVISION_EDGE_TYPE,
            pin_memory_device=torch.device(
                "cpu"
            ),  # Only CPU training for this example.
            shuffle=True,
        )
    )


def train(
    process_number: int,
    process_count: int,
    port: int,
    dataset: DistDataset,
    max_training_batches: int,
    batch_size: int = 4,
    val_every: int = 20,
    saved_model_path: str = "/tmp/gigl/dblp_model.pt",
):
    torch.distributed.init_process_group(
        backend="gloo",  # Use the Gloo backend for CPU training.
        init_method=f"tcp://localhost:{port}",  # Use the provided port for communication.
        rank=process_number,  # Each process has a unique rank.
        world_size=process_count,  # Total number of processes.
    )
    train_loader = get_data_loader(
        split="train", dataset=dataset, batch_size=batch_size
    )
    val_loader = get_data_loader(split="val", dataset=dataset, batch_size=batch_size)
    hgt = init_model()
    compute_loss(hgt, next(iter(train_loader)))  # initialize model weights for DDP
    model = DistributedDataParallel(
        hgt,
        find_unused_parameters=True,
    )
    logger.info(f"Process {process_number} initialized model: {model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    for batch_idx, main_data in enumerate(train_loader):
        if batch_idx >= max_training_batches:
            break
        loss = compute_loss(model, main_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % val_every == 0:
            logger.info(
                f"Process {process_number} running validation for batch {batch_idx} ..."
            )
            val_loss = run_validation(model, val_loader, num_val_batches=2)
            model.train()
            logger.info(f"Process {process_number} validation loss: {val_loss:.3f}")

    logger.info(f"Process {process_number} final training loss: {loss.item():.3f}")

    logger.info(f"Process {process_number} running test loops...")
    test_loader = get_data_loader(split="test", dataset=dataset, batch_size=batch_size)
    test_loss = run_validation(
        model, test_loader, num_val_batches=50  # Run validation on the test set
    )
    logger.info(f"Process {process_number} test loss: {test_loss:.3f}")
    if process_number == 0:
        logger.info(f"Saving model to {saved_model_path}")
        save_state_dict(model.module, UriFactory.create_uri(saved_model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a simple training loop on a heterogenous dataset."
    )
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
        "--process_count", type=str, default="1", help="Number of processes to spawn."
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="4",
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--val_every", type=str, default="400", help="Run validation every N batches."
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
    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        task_config_uri
    )
    dataset = build_dataset_from_task_config_uri(
        task_config_uri=task_config_uri,
        is_inference=False,
        _tfrecord_uri_pattern=".*tfrecord",
    )
    assert isinstance(dataset.train_node_ids, Mapping)
    process_count = int(args.process_count)
    for node_type, node_ids in dataset.train_node_ids.items():
        logger.info(f"Training node type {node_type} has {node_ids.size(0)} nodes.")
        max_training_batches = node_ids.size(0) // (
            int(args.batch_size) * torch.distributed.get_world_size() * process_count
        )
    assert isinstance(dataset.val_node_ids, Mapping)
    for node_type, node_ids in dataset.val_node_ids.items():
        logger.info(f"Validation node type {node_type} has {node_ids.size(0)} nodes.")
    assert isinstance(dataset.test_node_ids, Mapping)
    for node_type, node_ids in dataset.test_node_ids.items():
        logger.info(f"Test node type {node_type} has {node_ids.size(0)} nodes.")
    training_process_port = get_free_port()
    logger.info(f"Will train for {max_training_batches} batches.")
    if strtobool(args.use_local_saved_model):
        saved_model_uri = LOCAL_SAVED_MODEL_URI
    else:
        saved_model_uri = (
            gbml_config_pb_wrapper.shared_config.trained_model_metadata.trained_model_uri
        )

    logger.info(f"Using saved model URI: {saved_model_uri}")
    torch.multiprocessing.spawn(
        train,
        args=(
            process_count,  # process_count
            training_process_port,  # port
            dataset,  # dataset
            max_training_batches,  # max_training_batches
            int(args.batch_size),  # batch_size
            int(args.val_every),  # val_every
            saved_model_uri,  # saved_model_path
        ),
        nprocs=process_count,
    )
