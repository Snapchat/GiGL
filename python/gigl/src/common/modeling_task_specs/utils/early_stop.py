from copy import deepcopy
from typing import Any, Dict

import torch.nn as nn

from gigl.common.logger import Logger
from gigl.src.common.types.model_eval_metrics import EvalMetricType

logger = Logger()


class EarlyStopper:
    def __init__(self, early_stop_criterion: EvalMetricType, early_stop_patience: int):
        supported_early_stop_criteria = [
            m for m in EvalMetricType.get_all_criteria() if m != "hits"
        ]
        if early_stop_criterion.name not in supported_early_stop_criteria:
            raise NotImplementedError(
                f"Found invalid early stop criterion {early_stop_criterion.name}. Please make sure to supply one of {supported_early_stop_criteria}."
            )
        self.criterion = early_stop_criterion

        self._should_maximize: bool = self.criterion != EvalMetricType.loss
        self.prev_best = float("-inf") if self._should_maximize else float("inf")
        self.early_stop_counter = 0
        self.early_stop_patience = early_stop_patience
        self.best_val_model: Dict[str, Any] = {}

    def has_improved(self, value: float):
        return (self._should_maximize and value > self.prev_best) or (
            not self._should_maximize and value < self.prev_best
        )

    def should_early_stop(
        self, metrics: Dict[EvalMetricType, Any], model: nn.Module
    ) -> bool:
        value = metrics[self.criterion]
        if self.has_improved(value=value):
            self.early_stop_counter = 0
            logger.info(
                f"Validation {self.criterion.name} improved to {value:.4f} over previous best {self.prev_best}. Resetting early stop counter."
            )
            self.prev_best = value
            self.best_val_model = deepcopy(model.state_dict())
        else:
            self.early_stop_counter += 1
            logger.info(
                f"Got validation {self.criterion.name} {value}, which is worse than previous best {self.prev_best}. No improvement in validation {self.criterion.name} for {self.early_stop_counter} consecutive checks. Early Stop Counter: {self.early_stop_counter}"
            )

        if self.early_stop_counter >= self.early_stop_patience:
            logger.info(
                f"Early stopping triggered after {self.early_stop_counter} checks without improvement"
            )
            return True
        else:
            return False
