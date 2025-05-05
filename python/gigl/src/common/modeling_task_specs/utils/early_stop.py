import gc
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn

from gigl.common.logger import Logger

logger = Logger()


class EarlyStopper:
    """
    Handles early stopping logic, keeping track of the best performing model provided some criterion
    """

    def __init__(
        self,
        early_stop_patience: int,
        should_maximize: bool,
        model: Optional[nn.Module] = None,
    ):
        """
        Args:
            early_stop_patience (int): Maximum allowed number of steps for consecutive decreases in performance
            should_maximize (bool): Whether we minimize or maximize the provided criterion
        """
        self._should_maximize = should_maximize
        self._early_stop_counter = 0
        self._early_stop_patience = early_stop_patience
        self._prev_best = float("-inf") if self._should_maximize else float("inf")
        self._model = model
        self._best_model: Optional[dict[str, torch.Tensor]] = None

    def _has_metric_improved(self, value: float) -> bool:
        if self._should_maximize:
            return value > self._prev_best
        else:
            return value < self._prev_best

    def step(self, value: float) -> bool:
        """
        Steps through the early stopper provided some criterion. Returns whether the provided criterion improved over the previous best criterion.
        Args:
            value (float): Criterion used for stepping through early stopper
        Returns:
            bool: Whether there was improvement over previous best criteiron
        """
        if self._has_metric_improved(value=value):
            self._early_stop_counter = 0
            logger.info(
                f"Validation criteria improved to {value:.4f} over previous best {self._prev_best}. Resetting early stop counter."
            )
            self._prev_best = value
            if self._model is not None:
                # Making a deep copy of the best model and moving to CPU to save GPU memory
                best_model = deepcopy(self._model.state_dict())
                self._best_model = {}
                for identifier, layer in best_model.items():
                    self._best_model[identifier] = layer.cpu()
                gc.collect()
            return True
        else:
            self._early_stop_counter += 1
            logger.info(
                f"Got validation {value}, which is worse than previous best {self._prev_best}. No improvement in validation criteria for {self._early_stop_counter} consecutive checks. Early Stop Counter: {self._early_stop_counter}"
            )
            return False

    def should_early_stop(self) -> bool:
        """
        Identifies whether early stopping should occur based on provided early stop patience
        Returns:
            bool: Whether the `early_stop_patience` number of consecutive steps have occured without improvement
        """
        if self._early_stop_counter >= self._early_stop_patience:
            logger.info(
                f"Early stopping triggered after {self._early_stop_counter} checks without improvement"
            )
            return True
        else:
            return False

    @property
    def best_model_state_dict(self):
        return self._best_model

    @property
    def best_criterion(self):
        return self._prev_best
