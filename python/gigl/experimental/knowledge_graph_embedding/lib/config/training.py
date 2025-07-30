from dataclasses import dataclass, field
from typing import Optional

import torch
from applied_tasks.knowledge_graph_embedding.lib.config.dataloader import (
    DataloaderConfig,
)
from applied_tasks.knowledge_graph_embedding.lib.config.sampling import SamplingConfig


@dataclass
class OptimizerParamsConfig:
    # TODO(nshah): consider supporting LR scheduling in the Optimizer config.
    lr: float = 0.001
    weight_decay: float = 0.001


@dataclass
class OptimizerConfig:
    sparse: OptimizerParamsConfig = field(
        default_factory=lambda: OptimizerParamsConfig(lr=0.01, weight_decay=0.001)
    )
    dense: OptimizerParamsConfig = field(
        default_factory=lambda: OptimizerParamsConfig(lr=0.01, weight_decay=0.001)
    )


@dataclass
class DistributedConfig:
    num_processes_per_machine: int = (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )
    storage_reservation_percentage: float = 0.1


@dataclass
class CheckpointingConfig:
    save_every: int = 10_000
    should_save_async: bool = True
    load_from_path: Optional[str] = None
    save_to_path: Optional[str] = None


@dataclass
class LoggingConfig:
    log_every: int = 1


@dataclass
class EarlyStoppingConfig:
    patience: Optional[int] = None


@dataclass
class TrainConfig:
    max_steps: Optional[int] = None
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
