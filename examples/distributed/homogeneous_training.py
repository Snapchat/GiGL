"""Distributed homogeneous training example.

Run locally with:
MASTER_ADDR=localhost MASTER_PORT=9762 python examples/distributed/homogeneous_training.py --task_config_uri gs://gigl-cicd-perm/cora_glt_udl_test_on__54606/config_populator/frozen_gbml_config.yaml

# To simulate distributed training with 2 machines:
# Make sure your VM has two GPUs, or you are running on CPU.
# One terminal:
MASTER_ADDR=localhost MASTER_PORT=9762 python examples/distributed/homogeneous_training.py --task_config_uri gs://gigl-cicd-perm/cora_glt_udl_test_on__54606/config_populator/frozen_gbml_config.yaml --rank=0 --world_size=2

# Another terminal:
MASTER_ADDR=localhost MASTER_PORT=9762 python examples/distributed/homogeneous_training.py --task_config_uri gs://gigl-cicd-perm/cora_glt_udl_test_on__54606/config_populator/frozen_gbml_config.yaml --rank=1 --world_size=2
"""
import argparse
import datetime
import pickle
import time
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp
from torch import nn
from torch_geometric.data import Data

from gigl.common.logger import Logger
from gigl.common.types.uri.uri_factory import UriFactory
from gigl.common.utils.torch_training import is_distributed_available_and_initialized
from gigl.distributed import (
    DistABLPLoader,
    DistLinkPredictionDataset,
    DistNeighborLoader,
    DistributedContext,
    build_dataset_from_task_config_uri,
)
from gigl.distributed.utils import get_available_device
from gigl.src.common.models.layers.loss import RetrievalLoss
from gigl.src.common.models.pyg.homogeneous import GraphSAGE
from gigl.src.common.models.pyg.link_prediction import (
    LinkPredictionDecoder,
    LinkPredictionGNN,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.task_inputs import BatchCombinedScores
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    to_homogeneous,
)

logger = Logger()


def sync_loss_across_processes(local_loss: torch.Tensor) -> float:
    assert is_distributed_available_and_initialized(), "DDP is not initialized"
    # Make a copy of the local loss tensor
    loss_tensor = local_loss.detach().clone()
    torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
    return loss_tensor.item() / torch.distributed.get_world_size()


def split_parameters_by_layer_names(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    layer_names_to_separate: list[str],
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    """
    Split layer parameters of our interest from the rest of the model parameters based on a given list of layer names.
    Inputs:
    - named_parameters: The full list of parameters we'd like to split
    - layer_names_to_separate: Defines layers that we'd like to separate from the rest of the model parameters.
    Returns:
    - params_to_separate: A list of tuples containing the names and parameters of the layer to separate.
    - params_left: A list of tuples containing the names and parameters of the layers left.
    """
    params_to_separate = []
    params_left = []
    for name, param in named_parameters:
        # We match by substring to allow for matching a submodule of the model
        if any([layer_name in name for layer_name in layer_names_to_separate]):
            params_to_separate.append((name, param))
        else:
            params_left.append((name, param))
    return params_to_separate, params_left


def _init_example_gigl_model(
    state_dict: dict[str, torch.Tensor],
    node_feature_dim: int,
    edge_feature_dim: int,
    inferencer_args: dict[str, str],
    device: Optional[torch.device] = None,
) -> LinkPredictionGNN:
    """
    Initializes a hard-coded GiGL LinkPredictionGNN model, which inherits from `nn.Module`. Note that this is just an example --
    any `nn.Module` subclass can work with GLT.
    This model is trained based on the following CORA UDL E2E config:
    `python/gigl/src/mocking/configs/e2e_udl_node_anchor_based_link_prediction_template_gbml_config.yaml`

    Args:
        state_dict (Dict[str, torch.Tensor]): State dictionary for pretrained model
        node_feature_dim (int): Input node feature dimension for the model
        edge_feature_dim (int): Input edge feature dimension for the model
        inferencer_args (Dict[str, str]): Arguments for inferencer
        device (Optional[torch.device]): Torch device of the model, if None defaults to CPU
    Returns:
        LinkPredictionGNN: Link Prediction model for inference
    """
    # TODO (mkolodner-sc): Add asserts to ensure that model shape aligns with shape of state dict

    # We use the GiGL GraphSAGE implementation since the model shape needs to conform to the
    # state_dict that the trained model used, which was done with the GiGL GraphSAGE
    encoder_model = GraphSAGE(
        in_dim=node_feature_dim,
        hid_dim=int(inferencer_args.get("hid_dim", 16)),
        out_dim=int(inferencer_args.get("out_dim", 16)),
        edge_dim=edge_feature_dim if edge_feature_dim > 0 else None,
        num_layers=int(inferencer_args.get("num_layers", 2)),
        conv_kwargs={},  # Use default conv args for this model type
        should_l2_normalize_embedding_layer_output=True,
    )

    decoder_model = LinkPredictionDecoder()  # Defaults to inner product decoder

    model: LinkPredictionGNN = LinkPredictionGNN(
        encoder=encoder_model,
        decoder=decoder_model,
    )

    # Push the model to the specified device.
    if device is None:
        device = torch.device("cpu")
    model.to(device)

    # Override the initiated model's parameters with the saved model's parameters.
    # model.load_state_dict(state_dict)

    return model


def index_select(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.tensor([torch.nonzero(a == e).squeeze(1) for e in b], dtype=torch.long)


def _infer_inputs(
    model: nn.Module, main_data: Data, random_data: Data, device: torch.device
) -> tuple[BatchCombinedScores, torch.Tensor]:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        decoder = model.module.decode
    else:
        decoder = model.decode
    anchor_nodes: torch.Tensor = torch.tensor(list(main_data.y_positive.keys())).to(
        device
    )
    pos_nodes: torch.Tensor = torch.cat(list(main_data.y_positive.values())).to(device)
    neg_nodes: torch.Tensor = torch.cat(list(main_data.y_negative.values())).to(device)
    repeated_query_node_ids = anchor_nodes.repeat_interleave(
        torch.tensor([len(v) for v in main_data.y_positive.values()]).to(device)
    )
    main_embeddings = to_homogeneous(
        model(
            main_data,
            output_node_types=[DEFAULT_HOMOGENEOUS_NODE_TYPE],
            device=device,
        )
    )
    random_embeddings = to_homogeneous(
        model(
            random_data,
            output_node_types=[DEFAULT_HOMOGENEOUS_NODE_TYPE],
            device=device,
        )
    )
    repeated_query_embeddigns = main_embeddings[repeated_query_node_ids]
    repeated_candidate_scores = decoder(
        query_embeddings=repeated_query_embeddigns,
        candidate_embeddings=torch.cat(
            [
                main_embeddings[pos_nodes],
                main_embeddings[neg_nodes],
                random_embeddings[: len(random_data.batch)],
            ]
        ),
    )

    batched_combined_scores = BatchCombinedScores(
        repeated_candidate_scores=repeated_candidate_scores,
        positive_ids=main_data.node[pos_nodes],
        hard_neg_ids=main_data.node[neg_nodes],
        random_neg_ids=random_data.batch,
        repeated_query_ids=repeated_query_node_ids,
        num_unique_query_ids=len(anchor_nodes),
    )
    return batched_combined_scores, repeated_query_embeddigns


def _train_process(
    process_number: int,
    total_processes: int,
    dist_context: DistributedContext,
    dataset: DistLinkPredictionDataset,
    num_neighbors: list[int],
    negative_samples: torch.Tensor,
    model_state_dict_uri: str,
    node_feature_dim: int,
    edge_feature_dim: int,
    num_random_samples: int,
    validate_every: int = 100,
    learning_rate: int = 1e-6,
    num_epoch: int = 1,
    use_amp: bool = True,
) -> None:
    device = get_available_device(dist_context.global_rank + process_number)
    train_nodes = to_homogeneous(dataset.train_node_ids)
    train_loader = DistABLPLoader(
        dataset=dataset,
        context=dist_context,
        num_neighbors=num_neighbors,
        local_process_rank=process_number,
        local_process_world_size=total_processes,
        input_nodes=train_nodes,
        pin_memory_device=device,
        shuffle=True,
        drop_last=True,
    )
    val_nodes = to_homogeneous(dataset.val_node_ids)
    val_loader = iter(
        DistABLPLoader(
            dataset=dataset,
            context=dist_context,
            num_neighbors=num_neighbors,
            local_process_rank=process_number,
            local_process_world_size=total_processes,
            input_nodes=val_nodes,
            pin_memory_device=device,
            shuffle=True,
            drop_last=True,
            _main_sampling_port=50_000 + process_number * 100 + 4,
        )
    )
    zero_fanout = [0] * len(num_neighbors)
    random_negative_loader = iter(
        DistNeighborLoader(
            dataset=dataset,
            context=dist_context,
            num_neighbors={
                DEFAULT_HOMOGENEOUS_EDGE_TYPE: num_neighbors,
                message_passing_to_negative_label(
                    DEFAULT_HOMOGENEOUS_EDGE_TYPE
                ): zero_fanout,
                message_passing_to_positive_label(
                    DEFAULT_HOMOGENEOUS_EDGE_TYPE
                ): zero_fanout,
            },
            local_process_rank=process_number,
            local_process_world_size=total_processes,
            input_nodes=(DEFAULT_HOMOGENEOUS_NODE_TYPE, negative_samples),
            pin_memory_device=device,
            batch_size=num_random_samples,
            shuffle=True,
            drop_last=True,
            _main_inference_port=40_000 + process_number * 100 + 4,
            _main_sampling_port=30_000 + process_number * 100 + 4,
        )
    )
    logger.info(f"device_type: {device.type}, device: {device}")
    torch.distributed.init_process_group(
        # TODO: "nccl" when GPU
        backend="gloo" if device.type == "cpu" else "nccl",
        rank=dist_context.global_rank * total_processes + process_number,
        world_size=dist_context.global_world_size * total_processes,
        timeout=datetime.timedelta(minutes=30),
    )

    # Initialize a LinkPredictionGNN model and load parameters from
    # the saved model.
    # model_state_dict = load_state_dict_from_uri(
    #     load_from_uri=model_state_dict_uri, device=device
    # )
    model: nn.Module = _init_example_gigl_model(
        state_dict={},
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        inferencer_args={},
        device=device,
    )
    ddp_model = nn.parallel.DistributedDataParallel(
        model, device_ids=[device] if device.type == "cuda" else None
    )
    ddp_model.train()
    all_named_parameters = list(ddp_model.named_parameters())
    embedding_params, other_params = split_parameters_by_layer_names(
        named_parameters=all_named_parameters,
        layer_names_to_separate=[".feature_embedding_layer."],
    )
    param_groups = [
        {
            "params": [param for _, param in embedding_params],
            "lr": learning_rate,
            "weight_decay": 0.00005,
        },
        {
            "params": [param for _, param in other_params],
            "lr": learning_rate,
            "weight_decay": 0.00005,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups)
    loss = RetrievalLoss(
        loss=torch.nn.CrossEntropyLoss(reduction="mean"),
        temperature=0.07,
        remove_accidental_hits=True,
    )
    scaler = torch.GradScaler(
        "cuda", enabled=use_amp
    )  # Used in mixed precision training to avoid gradients underflow due to float16 precision
    logger.info(
        f"Model initialized on machine {process_number} training device {device}\n{model}"
    )
    assert is_distributed_available_and_initialized()

    # (Optional) Add a explicit barrier to make sure all processes finished setting up all dataloaders
    torch.distributed.barrier()
    print(
        f"process {process_number} started training. {train_nodes.shape=}, {val_nodes.shape=}, {negative_samples.shape=}"
    )
    for epoch in range(num_epoch):
        logger.info(f"Epoch {epoch} started")
        for batch_idx, main_data in enumerate(train_loader):
            # TODO enable early stop
            # TODO enable AMP
            random_data = next(random_negative_loader)
            random_data = random_data.edge_type_subgraph(
                [DEFAULT_HOMOGENEOUS_EDGE_TYPE]
            ).to_homogeneous(add_edge_type=False, add_node_type=False)
            scores, repeated_query_embeddings = _infer_inputs(
                model=ddp_model,
                main_data=main_data,
                random_data=random_data,
                device=device,
            )
            loss_output = loss(
                batch_combined_scores=scores,
                repeated_query_embeddings=repeated_query_embeddings,
                candidate_sampling_probability=None,
                device=device,
            )
            loss_tensor: torch.Tensor = loss_output[0]

            if loss_tensor.grad_fn is None:
                # print("Skipping loss")
                continue
            optimizer.zero_grad()
            scaler.scale(loss_tensor).backward()
            scaler.step(optimizer)
            scaler.update()

            # Validation
            if batch_idx % validate_every == 0:
                print(f"validating for batch idx {batch_idx} at epoch {epoch}")
                with torch.no_grad():
                    random_data = next(random_negative_loader)
                    random_data = random_data.edge_type_subgraph(
                        [DEFAULT_HOMOGENEOUS_EDGE_TYPE]
                    ).to_homogeneous(add_edge_type=False, add_node_type=False)
                    val_scores, repeated_query_embeddings = _infer_inputs(
                        model=ddp_model,
                        main_data=next(val_loader),
                        random_data=random_data,
                        device=device,
                    )
                    loss_output = loss(
                        batch_combined_scores=val_scores,
                        repeated_query_embeddings=repeated_query_embeddings,
                        candidate_sampling_probability=None,
                        device=device,
                    )
                    val_loss_tensor: torch.Tensor = loss_output[0]

    # (Optional) Add a explicit barrier to make sure all processes finished setting up all dataloaders
    torch.distributed.barrier()
    # barrier()
    torch.distributed.destroy_process_group()
    del train_loader, val_loader, random_negative_loader


def train(
    task_config_uri: str,
    rank: int,
    world_size: int,
) -> None:
    train_start_time = time.time()

    # dist_context = connect_worker_pool()
    dist_context = DistributedContext("localhost", rank, world_size)
    local_dataset = Path(__file__).parent / "local_dataset.pkl"
    if not local_dataset.exists():
        dataset = build_dataset_from_task_config_uri(
            task_config_uri, dist_context, is_inference=False
        )
        pickle.dump(dataset.share_ipc(), open(local_dataset, "wb"))
    else:
        dataset = DistLinkPredictionDataset.from_ipc_handle(
            pickle.load(open(local_dataset, "rb"))
        )
    gbml_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        UriFactory.create_uri(task_config_uri)
    )
    num_training_processes = int(
        gbml_pb_wrapper.trainer_config.trainer_args.get(
            "num_training_processes_per_machine", 1
        )
    )
    num_random_samples = int(
        gbml_pb_wrapper.trainer_config.trainer_args.get("num_random_samples", 100)
    )
    num_neighbors = [10, 10]
    negative_samples = torch.arange(len(to_homogeneous(dataset.node_pb))).repeat(
        num_random_samples
    )
    negative_samples.share_memory_()

    node_feature_dim = gbml_pb_wrapper.preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_dim_map[
        gbml_pb_wrapper.graph_metadata_pb_wrapper.homogeneous_condensed_node_type
    ]

    edge_feature_dim = gbml_pb_wrapper.preprocessed_metadata_pb_wrapper.condensed_edge_type_to_feature_dim_map[
        gbml_pb_wrapper.graph_metadata_pb_wrapper.homogeneous_condensed_edge_type
    ]

    mp.spawn(
        _train_process,
        args=(
            num_training_processes,
            dist_context,
            dataset,
            num_neighbors,
            negative_samples,
            gbml_pb_wrapper.trainer_config.trainer_cls_path,
            node_feature_dim,
            edge_feature_dim,
            num_random_samples,
        ),
        nprocs=num_training_processes,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed homogeneous training example"
    )
    parser.add_argument(
        "--task_config_uri",
        type=str,
        required=True,
        help="URI to the task config file",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of the process in the distributed training",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of processes in the distributed training",
    )

    args = parser.parse_args()
    train(
        task_config_uri=args.task_config_uri,
        rank=args.rank,
        world_size=args.world_size,
    )
