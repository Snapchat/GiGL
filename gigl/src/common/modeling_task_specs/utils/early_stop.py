import io
from typing import Optional, Tuple

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
            model (Optional[nn.Module]): Optional model to provide to early stopper class. If provided, will
                keep track of the state dict of the best model.
        """
        self._should_maximize = should_maximize
        self._early_stop_counter = 0
        self._early_stop_patience = early_stop_patience
        self._prev_best = float("-inf") if self._should_maximize else float("inf")
        self._model = model
        self._best_model_buffer: Optional[io.BytesIO] = None

    def _has_metric_improved(self, value: float) -> bool:
        if self._should_maximize:
            return value > self._prev_best
        else:
            return value < self._prev_best

    def step(self, value: float) -> Tuple[bool, bool]:
        """
        Steps through the early stopper provided some criterion. Returns whether the provided criterion improved over the previous best criterion and
        whether we should early stop.
        Args:
            value (float): Criterion used for stepping through early stopper
        Returns:
            bool: Whether there was improvement over previous best criterion
            bool: Whether early stop patience has been reached, indicating early stopping
        """
        has_metric_improved: bool
        should_early_stop: bool
        if self._has_metric_improved(value=value):
            self._early_stop_counter = 0
            logger.info(
                f"Validation criteria improved to {value:.4f} over previous best {self._prev_best}. Resetting early stop counter."
            )
            self._prev_best = value
            if self._model is not None:
                self._best_model_buffer = io.BytesIO()
                self._best_model_buffer.seek(0)
                torch.save(self._model.state_dict(), self._best_model_buffer)
            has_metric_improved = True
        else:
            self._early_stop_counter += 1
            logger.info(
                f"Got validation {value}, which is worse than previous best {self._prev_best}. No improvement in validation criteria for {self._early_stop_counter} consecutive checks. Early Stop Counter: {self._early_stop_counter}"
            )
            has_metric_improved = False

        if self._early_stop_counter >= self._early_stop_patience:
            logger.info(
                f"Early stopping triggered after {self._early_stop_counter} checks without improvement"
            )
            should_early_stop = True
        else:
            should_early_stop = False

        return has_metric_improved, should_early_stop

    @property
    def best_model_state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        if self._best_model_buffer is None:
            return None
        else:
            self._best_model_buffer.seek(0)
            return torch.load(self._best_model_buffer)

    @property
    def best_criterion(self) -> float:
        return self._prev_best
