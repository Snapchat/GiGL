"""
Dead simple heterogenous training loop for CPU training with DistributedDataParallel.

Supports multi process/multi node training.
Does not support GPU training.

Run with:
    python examples/tutorial/KDD_2025/heterogeneous_training.py

This example is meant to be run on the "toy graph" dataset,
which is a small heterogeneous graph with two node types (user) and (story)
and two edge types (user to story) and (story to user).

The dataset it reads from may be configured in the `task_config_uri` argument,
if using a different dataset, also update the following fields:
 - QUERY_NODE_TYPE
 - TARGET_NODE_TYPE
 - SUPERVISION_EDGE_TYPE
 and the metadata in the `init_model` function.

 Multi node training is supported by via the --rank and --world_size arguments.
 if doing multi node training, make sure to set the `--host` and `--port` arguments
 to the same values across all nodes.

 You may use the `--process_count` argument to control how many processes.

"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs isort: skip


import argparse
from collections.abc import Iterable, Mapping
from typing import Literal

import torch
import torch.multiprocessing.spawn
from torch.nn.parallel import DistributedDataParallel
from torch_geometric import typing as pyg_typing
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

from gigl.common.logger import Logger
from gigl.common.types.uri.uri_factory import UriFactory
from gigl.distributed import (
    DistABLPLoader,
    DistLinkPredictionDataset,
    build_dataset_from_task_config_uri,
)
from gigl.distributed.utils import get_free_port
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.common.utils.model import save_state_dict
from gigl.utils.iterator import InfiniteIterator

logger = Logger()

# Make sure to update these if using a different dataset.
QUERY_NODE_TYPE = NodeType("user")
TARGET_NODE_TYPE = NodeType("story")

SUPERVISION_EDGE_TYPE = EdgeType(QUERY_NODE_TYPE, Relation("to"), TARGET_NODE_TYPE)

# Arbitrary fanout for the example, can be adjusted based on dataset and model.
FANOUT = [10, 10]


def init_model(
    out_channels: int = 16,
    metadata: pyg_typing.Metadata = (
        ["user", "story"],
        [
            ("user", "to", "story"),
            ("story", "to", "user"),
        ],
    ),
) -> HGTConv:
    return HGTConv(
        in_channels=-1,  # Input channels will be set dynamically based on the dataset.
        out_channels=out_channels,
        metadata=metadata,
    )


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
    dataset: DistLinkPredictionDataset,
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
    dataset: DistLinkPredictionDataset,
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
        default="examples/tutorial/KDD_2025/toy_graph_task_config.yaml",
        help="Path to the task config URI.",
    )
    parser.add_argument(
        "--process_count", type=int, default=1, help="Number of processes to spawn."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--val_every", type=int, default=400, help="Run validation every N batches."
    )
    parser.add_argument(
        "--saved_model_path",
        type=str,
        default="/tmp/gigl/dblp_model.pt",
        help="Path to save the trained model.",
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
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host for distributed training."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=get_free_port(),
        help="Port for distributed training communication.",
    )

    args = parser.parse_args()
    logger.info(f"Using args: {args}")
    torch.distributed.init_process_group(
        backend="gloo",  # Use the Gloo backend for CPU training.
        init_method=f"tcp://{args.host}:{args.port}",
        rank=args.rank,
        world_size=args.world_size,
    )

    dataset = build_dataset_from_task_config_uri(
        task_config_uri=args.task_config_uri,
        is_inference=False,
        _tfrecord_uri_pattern=".*tfrecord",
    )
    logger.info(
        f"Splits: {dataset.train_node_ids=}, {dataset.val_node_ids=}, {dataset.test_node_ids=}"
    )
    assert isinstance(dataset.train_node_ids, Mapping)
    for node_type, node_ids in dataset.train_node_ids.items():
        logger.info(f"Training node type {node_type} has {node_ids.size(0)} nodes.")
        max_training_batches = node_ids.size(0) // (
            args.batch_size * args.world_size * args.process_count
        )
    assert isinstance(dataset.val_node_ids, Mapping)
    for node_type, node_ids in dataset.val_node_ids.items():
        logger.info(f"Validation node type {node_type} has {node_ids.size(0)} nodes.")
    assert isinstance(dataset.test_node_ids, Mapping)
    for node_type, node_ids in dataset.test_node_ids.items():
        logger.info(f"Test node type {node_type} has {node_ids.size(0)} nodes.")
    training_process_port = get_free_port()
    process_count = args.process_count
    logger.info(f"Will train for {max_training_batches} batches.")
    torch.multiprocessing.spawn(
        train,
        args=(
            process_count,  # process_count
            training_process_port,  # port
            dataset,  # dataset
            max_training_batches,  # max_training_batches
            args.batch_size,  # batch_size
            args.val_every,  # val_every
            args.saved_model_path,  # saved_model_path
        ),
        nprocs=process_count,
    )
