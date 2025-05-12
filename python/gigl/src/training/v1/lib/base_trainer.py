from __future__ import annotations

from typing import Optional

import torch

from gigl.common.logger import Logger
from gigl.src.common.modeling_task_specs.utils.profiler_wrapper import TorchProfiler
from gigl.src.common.types.model import BaseModelOperationsProtocol
from gigl.src.common.types.model_eval_metrics import EvalMetricsCollection
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper

logger = Logger()


class BaseTrainer(BaseModelOperationsProtocol):
    """
    The Protocol that you need to implement for your trainer to function with Training
    in tabularized mode.

    Note: the BaseTrainer class also implements the BaseModelOperationsProtocol
    which requires the init_model method, and the getter and setter for the model property.

    """

    def train(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        device: torch.device,
        profiler: Optional[TorchProfiler] = None,
    ) -> None:
        raise NotImplementedError

    def eval(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        device: torch.device,
    ) -> EvalMetricsCollection:
        raise NotImplementedError

    def setup_for_training(self) -> None:
        raise NotImplementedError

    @property
    def supports_distributed_training(self) -> bool:
        raise NotImplementedError
