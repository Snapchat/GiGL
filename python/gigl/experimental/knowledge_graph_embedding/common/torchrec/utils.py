from typing import Any, Dict, Iterable, Optional, Type

import torch
import torch.nn as nn
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.optim import Optimizer
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import ShardingPlan
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad

from gigl.common.logger import Logger

logger = Logger()


def maybe_shard_model(
    model,
    device: torch.device,
    sharding_plan: ShardingPlan = None,
):
    """
    If in a distributed environment, apply DistributedModelParallel to the model,
    using an optionally specified ShardingPlan.
    If not in a distributed environment, return the model directly.
    Args:
        model: The model to be wrapped.
        device: The device to use for the model.
        sharding_plan: An optional ShardingPlan to use for the DistributedModelParallel.
    Returns:
        The model wrapped in DistributedModelParallel if in a distributed environment,
        otherwise the model itself.
    """

    if torch.distributed.is_initialized():
        # Build a sharding plan
        logger.info("***** Wrapping in DistributedModelParallel *****")
        logger.info(f"Model before wrapping: {model}")
        model = DistributedModelParallel(
            module=model,
            device=device,
            plan=sharding_plan,
        )
        logger.info(f"Model after wrapping: {model}")

    return model


def get_sharding_plan(
    model: nn.Module,
    batch_size: int,
    local_world_size: int,
    world_size: int,
    use_cuda: bool = False,
    storage_reservation_percentage: float = 0.15,
    qcomm_forward_precision: CommType = CommType.FP32,
    qcomm_backward_precision: CommType = CommType.FP32,
) -> ShardingPlan:
    """
    Create a sharding plan for the model using the EmbeddingShardingPlanner.
    Args:
        model: The model to be sharded.
        batch_size: The batch size for the sharding plan.
        use_cuda: Whether to use CUDA for the sharding plan.
        storage_reservation_percentage: The percentage of storage reservation.
        qcomm_forward_precision: The precision for forward communication (can be FP32, FP16, etc.).
        qcomm_backward_precision: The precision for backward communication (can be FP32, FP16, etc.).
    Returns:
        A ShardingPlan object representing the sharding plan for the model.
    """

    topology = Topology(
        world_size=world_size,
        local_world_size=local_world_size,  # TODO(nshah): We should expose this in torch_training.py
        compute_device="cuda" if use_cuda else "cpu",
        hbm_cap=torch.cuda.get_device_properties(0).total_memory if use_cuda else 0,
    )

    planner = EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        storage_reservation=HeuristicalStorageReservation(
            percentage=storage_reservation_percentage
        ),  # bumping this % can alleviate OOM issues by being more conservative
    )

    # Enable custom fwd/bkwd precisions for QComms when using GPU
    qcomm_codecs_registry = (
        get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                forward_precision=qcomm_forward_precision,
                backward_precision=qcomm_backward_precision,
            )
        )
        if use_cuda
        else None
    )
    ebc_sharder = EmbeddingBagCollectionSharder(
        qcomm_codecs_registry=qcomm_codecs_registry
    )
    plan = planner.collective_plan(
        model, [ebc_sharder], torch.distributed.GroupMember.WORLD
    )
    return plan


def apply_sparse_optimizer(
    parameters: Iterable[nn.Parameter],
    optimizer_cls: Type[Optimizer] = None,
    optimizer_kwargs: Dict[str, Any] = dict(),
) -> None:
    """
    Apply a sparse optimizer to the sparse/EBC parts of a model.
    This optimizer is fused, so it will be applied directly in the backward pass.

    This should only be used for sparse parameters.

    Args:
        parameters (Iterable[nn.Parameter]): The sparse parameters to apply the optimizer to.
        optimizer_cls (Type[Optimizer], optional): The optimizer class to use. Defaults to RowWiseAdagrad.
        optimizer_kwargs (Dict[str, Any], optional): Additional keyword arguments for the optimizer.
    """

    if not optimizer_cls and optimizer_kwargs:
        optimizer_cls = RowWiseAdagrad
        optimizer_kwargs = {"lr": 0.01}
    apply_optimizer_in_backward(optimizer_cls, parameters, optimizer_kwargs)


def apply_dense_optimizer(
    model: nn.Module,
    optimizer_cls: Type[Optimizer],
    optimizer_kwargs: Dict[str, Any] = dict(),
) -> Optional[KeyedOptimizerWrapper]:
    """
    This creates an optimizer for the dense parts of the model.
    It uses the `KeyedOptimizerWrapper` to wrap the optimizer.

    Args:
        model (nn.Module): The model containing dense parameters.
        optimizer_cls (Type[Optimizer]): The optimizer class to use for dense parameters.
        optimizer_kwargs (Dict[str, Any], optional): Additional keyword arguments for the optimizer.

    Returns:
        Optional[KeyedOptimizerWrapper]: A wrapped optimizer for dense parameters, or
            None if no dense parameters are found.
    """
    dense_params = dict(in_backward_optimizer_filter(model.named_parameters()))
    if not dense_params:
        # We cannot apply a dense optimizer if there are no dense parameters.
        logger.warning("No dense parameters found in the model.")
        return None
    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        lambda params: optimizer_cls(params, **optimizer_kwargs),
    )
    return dense_optimizer
