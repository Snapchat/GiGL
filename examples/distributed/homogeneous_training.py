import argparse
import datetime
import time
from typing import Optional

import torch
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from gigl.common.logger import Logger
from gigl.common.types.uri.uri_factory import UriFactory
from gigl.common.utils.vertex_ai_context import connect_worker_pool
from gigl.distributed import (
    DistABLPLoader,
    DistLinkPredictionDataset,
    DistNeighborLoader,
    DistributedContext,
    build_dataset_from_task_config_uri,
)
from gigl.distributed.utils import get_available_device
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.common.utils.os_utils import import_obj
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import PreprocessedMetadataPbWrapper
from gigl.types.graph import to_homogeneous
from gigl.src.common.models.pyg.homogeneous import GraphSAGE
from gigl.src.common.models.pyg.link_prediction import (
    LinkPredictionDecoder,
    LinkPredictionGNN,
)
from gigl.src.common.utils.model import load_state_dict_from_uri
from snapchat.research.gbml.preprocessed_metadata_pb2 import PreprocessedMetadata
from gigl.common.utils.torch_training import is_distributed_available_and_initialized
from gigl.src.common.models.layers.loss import RetrievalLoss

logger = Logger()

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
    model.load_state_dict(state_dict)

    return model

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
) -> None:
    validate_every = 50
    learning_rate = 0.005
    num_epoch = 1
    device = get_available_device(process_number)
    train_loader = DistABLPLoader(
        dataset=dataset,
        context=dist_context,
        num_neighbors=num_neighbors,
        local_process_rank=process_number,
        local_process_world_size=total_processes,
        input_nodes=to_homogeneous(dataset.train_node_ids),
        pin_memory_device=device,
        shuffle=True,
        drop_last=True,
    )
    # val_loader = DistABLPLoader(
    #     dataset=dataset,
    #     context=dist_context,
    #     num_neighbors=num_neighbors,
    #     local_process_rank=process_number,
    #     local_process_world_size=total_processes,
    #     input_nodes=to_homogeneous(dataset.val_node_ids),
    #     pin_memory_device=device,
    #     shuffle=True,
    #     drop_last=True,
    # )
    random_negative_loader = DistNeighborLoader(
        dataset=dataset,
        context=dist_context,
        num_neighbors=num_neighbors,
        local_process_rank=process_number,
        local_process_world_size=total_processes,
        input_nodes=negative_samples,
        pin_memory_device=device,
        shuffle=True,
        drop_last=True,
        _main_inference_port=50_300,
        _main_sampling_port=50_400,
    )
    torch.distributed.init_process_group(
        backend="nccl",
        rank=dist_context.global_rank * total_processes + process_number,
        world_size=dist_context.global_world_size * total_processes,
        timeout=datetime.timedelta(minutes=30),
    )

    # Initialize a LinkPredictionGNN model and load parameters from
    # the saved model.
    model_state_dict = load_state_dict_from_uri(
        load_from_uri=model_state_dict_uri, device=device
    )
    model: nn.Module = _init_example_gigl_model(
        state_dict=model_state_dict,
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        inferencer_args={},
        device=device,
    )
    ddp_model = DistributedDataParallel(model, device_ids=[device])
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
    logger.info(
        f"Model initialized on machine {process_number} training device {device}\n{model}"
    )
    assert is_distributed_available_and_initialized()
    # (Optional) Add a explicit barrier to make sure all processes finished setting up all dataloaders
    torch.distributed.barrier()
    for epoch in range(num_epoch):
        for batch_idx, (main_data, random_data) in enumerate(
            zip(train_loader, random_negative_loader)
        ):
            # TODO enable early stop
            # TODO enable AMP
            main_embeddings = ddp_model(main_data)



def train(
    task_config_uri: str,
) -> None:
    train_start_time = time.time()

    #dist_context = connect_worker_pool()
    dist_context = DistributedContext("localhost", 0, 1)
    dataset = build_dataset_from_task_config_uri(
        task_config_uri, dist_context, is_inference=False
    )
    gbml_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        UriFactory.create_uri(task_config_uri)
    )
    num_training_processes = int(
        gbml_pb_wrapper.trainer_config.trainer_args.get(
            "num_training_processes_per_machine", 2
        )
    )
    num_neighbors = [10, 10]
    negative_samples = torch.arange(len(dataset.node_pb))
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

    args = parser.parse_args()
    train(
        task_config_uri=args.task_config_uri,
    )
