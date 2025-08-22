from dataclasses import dataclass, field
from typing import Optional

import torch

from gigl.experimental.knowledge_graph_embedding.lib.config.dataloader import (
    DataloaderConfig,
)
from gigl.experimental.knowledge_graph_embedding.lib.config.sampling import (
    SamplingConfig,
)


@dataclass
class OptimizerParamsConfig:
    """
    Configuration for optimizer hyperparameters.

    Attributes:
        lr (float): Learning rate for the optimizer. Controls the step size during gradient descent.
            Higher values lead to faster convergence but may overshoot the minimum.
            Defaults to 0.001.
        weight_decay (float): L2 regularization coefficient applied to model parameters.
            Helps prevent overfitting by penalizing large weights. Defaults to 0.001.
    """

    lr: float = 0.001
    weight_decay: float = (
        0.001  # TODO(nshah): consider supporting weight decay for sparse embeddings.
    )


@dataclass
class OptimizerConfig:
    """
    Configuration for separate optimizers for sparse and dense parameters.

    Knowledge graph embedding models typically have both sparse embeddings (updated only
    for nodes/edges in each batch) and dense parameters (updated every batch). Different
    learning rates are often beneficial for these parameter types.

    Attributes:
        sparse (OptimizerParamsConfig): Optimizer parameters for sparse embeddings (for nodes).
            Defaults to OptimizerParamsConfig(lr=0.01, weight_decay=0.001).
        dense (OptimizerParamsConfig): Optimizer parameters for dense model parameters (linear layers, etc.).
            Defaults to OptimizerParamsConfig(lr=0.01, weight_decay=0.001).
    """

    sparse: OptimizerParamsConfig = field(
        default_factory=lambda: OptimizerParamsConfig(lr=0.01, weight_decay=0.001)
    )
    dense: OptimizerParamsConfig = field(
        default_factory=lambda: OptimizerParamsConfig(lr=0.01, weight_decay=0.001)
    )


@dataclass
class DistributedConfig:
    """
    Configuration for distributed training across multiple GPUs or processes.

    Attributes:
        num_processes_per_machine (int): Number of training processes to spawn per machine.
            Each process typically uses one GPU. Defaults to torch.cuda.device_count()
            if CUDA is available, otherwise 1.
        storage_reservation_percentage (float): Storage percentage buffer used by TorchRec.
            to account for overhead on dense tensor and KJT storage. Defaults to 0.1 (10%).
    """

    num_processes_per_machine: int = (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )
    storage_reservation_percentage: float = 0.1


@dataclass
class CheckpointingConfig:
    """
    Configuration for model checkpointing during training.

    Attributes:
        save_every (int): Save a checkpoint every N training steps. Allows recovery from
            failures and monitoring of training progress. Defaults to 10,000 steps.
        should_save_async (bool): Whether to save checkpoints asynchronously to avoid blocking
            training. Improves training efficiency but may use additional memory.
            Defaults to True.
        load_from_path (Optional[str]): Path to a checkpoint file to resume training from. If None,
            training starts from scratch. Defaults to None.
        save_to_path (Optional[str]): Directory path where checkpoints will be saved. If None,
            checkpoints are not saved. Defaults to None.
    """

    save_every: int = 10_000
    should_save_async: bool = True
    load_from_path: Optional[str] = None
    save_to_path: Optional[str] = None


@dataclass
class LoggingConfig:
    """
    Configuration for training progress logging.

    Attributes:
        log_every (int): Log training metrics every N steps. More frequent logging provides
            better monitoring but may slow down training. Defaults to 1 (log every step).
    """

    log_every: int = 1


@dataclass
class EarlyStoppingConfig:
    """
    Configuration for early stopping based on validation performance.

    Attributes:
        patience (Optional[int]): Number of evaluation steps to wait for improvement before stopping
            training. Helps prevent overfitting by stopping when validation performance
            plateaus. If None, early stopping is disabled. Defaults to None.
    """

    patience: Optional[int] = None


@dataclass
class TrainConfig:
    """
    Main training configuration that orchestrates all training-related settings.

    This configuration combines optimization, data loading, distributed training,
    checkpointing, and monitoring settings for knowledge graph embedding training.

    Attributes:
        max_steps (Optional[int]): Maximum number of training steps to perform. If None, training
            continues until early stopping or manual interruption. Defaults to None.
        early_stopping (EarlyStoppingConfig): Configuration for early stopping based on validation metrics.
            Defaults to EarlyStoppingConfig() with no patience limit.
        dataloader (DataloaderConfig): Configuration for data loading (number of workers, memory pinning).
            Defaults to DataloaderConfig() with standard settings.
        sampling (SamplingConfig): Configuration for negative sampling strategy during training.
            Defaults to SamplingConfig() with standard settings.
        optimizer (OptimizerConfig): Configuration for separate sparse and dense optimizers.
            Defaults to OptimizerConfig() with standard settings.
        distributed (DistributedConfig): Configuration for multi-GPU/multi-process training.
            Defaults to DistributedConfig() with auto-detected GPU count.
        checkpointing (CheckpointingConfig): Configuration for saving and loading model checkpoints.
            Defaults to CheckpointingConfig() with standard settings.
        logging (LoggingConfig): Configuration for training progress logging frequency.
            Defaults to LoggingConfig() with log-every-step setting.
    """

    max_steps: Optional[int] = None
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
