import unittest
from typing import List, Optional

import torch
import torch.nn as nn
from parameterized import param, parameterized

from gigl.src.common.modeling_task_specs.utils.early_stop import EarlyStopper
from tests.test_assets.distributed.utils import assert_tensor_equality

_EARLY_STOP_PATIENCE = 3


class _DummyModel(nn.Module):
    def __init__(self):
        super(_DummyModel, self).__init__()
        self.register_buffer("foo", torch.tensor(0.0))

    def forward(self, x):
        return x


class EarlyStopTests(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Test loss early stopping without model checkpointing",
                mocked_criteria_values=[
                    150.0,
                    100.0,
                    50.0,
                    60.0,
                    70.0,
                    30.0,
                    40.0,
                    50.0,
                    80.0,
                ],
                improvement_steps=[0, 1, 2, 5],
                should_maximize=False,
                model=None,
                expected_best_criterion=30.0,
            ),
            param(
                "Test MRR early stopping without model checkpointing",
                mocked_criteria_values=[0.1, 0.3, 0.5, 0.45, 0.4, 0.6, 0.5, 0.4, 0.3],
                improvement_steps=[0, 1, 2, 5],
                should_maximize=True,
                model=None,
                expected_best_criterion=0.6,
            ),
            param(
                "Test loss early stopping with model checkpointing",
                mocked_criteria_values=[
                    150.0,
                    100.0,
                    50.0,
                    60.0,
                    70.0,
                    30.0,
                    40.0,
                    50.0,
                    80.0,
                ],
                improvement_steps=[0, 1, 2, 5],
                should_maximize=False,
                model=_DummyModel(),
                expected_best_criterion=30.0,
            ),
            param(
                "Test MRR early stopping with model checkpointing",
                mocked_criteria_values=[0.1, 0.3, 0.5, 0.45, 0.4, 0.6, 0.5, 0.4, 0.3],
                improvement_steps=[0, 1, 2, 5],
                should_maximize=True,
                model=_DummyModel(),
                expected_best_criterion=0.6,
            ),
        ]
    )
    def test_early_stopping(
        self,
        _,
        mocked_criteria_values: List[float],
        improvement_steps: List[int],
        should_maximize: bool,
        model: Optional[nn.Module],
        expected_best_criterion: float,
    ):
        early_stopper = EarlyStopper(
            early_stop_patience=_EARLY_STOP_PATIENCE,
            should_maximize=should_maximize,
            model=model,
        )
        for step_num, value in enumerate(mocked_criteria_values):
            has_metric_improved, should_early_stop = early_stopper.step(value=value)
            if model is not None:
                model.foo += 1
            if step_num in improvement_steps:
                self.assertTrue(has_metric_improved)
            else:
                self.assertFalse(has_metric_improved)
            if step_num < len(mocked_criteria_values) - 1:
                self.assertFalse(should_early_stop)
            else:
                self.assertTrue(should_early_stop)
        if model is not None:
            assert early_stopper.best_model_state_dict is not None
            assert_tensor_equality(
                early_stopper.best_model_state_dict["foo"], torch.tensor(5.0)
            )
            self.assertTrue(early_stopper.best_model_state_dict["foo"].is_cpu)
        else:
            self.assertIsNone(early_stopper.best_model_state_dict)
        self.assertEqual(early_stopper.best_criterion, expected_best_criterion)
