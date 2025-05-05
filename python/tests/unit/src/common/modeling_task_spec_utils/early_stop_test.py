import unittest
from typing import List

from parameterized import param, parameterized

from gigl.src.common.modeling_task_specs.utils.early_stop import EarlyStopper

_EARLY_STOP_PATIENCE = 3


class EarlyStopTests(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Test loss early stopping",
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
                improvement_steps=[0, 1, 4],
                should_maximize=False,
            ),
            param(
                "Test MRR early stopping",
                mocked_criteria_values=[0.1, 0.3, 0.5, 0.45, 0.4, 0.6, 0.5, 0.4, 0.3],
                improvement_steps=[0, 1, 4],
                should_maximize=True,
            ),
        ]
    )
    def test_early_stopping(
        self,
        _,
        mocked_criteria_values: List[float],
        improvement_steps: List[int],
        should_maximize: bool,
    ):
        early_stoppper = EarlyStopper(
            early_stop_patience=_EARLY_STOP_PATIENCE, should_maximize=should_maximize
        )
        for step_num, value in enumerate(mocked_criteria_values):
            has_metric_improved = early_stoppper.step(value=value)
            if step_num in improvement_steps:
                self.assertTrue(has_metric_improved)
            else:
                self.assertFalse(has_metric_improved)
            if step_num < len(mocked_criteria_values) - 1:
                self.assertFalse(early_stoppper.should_early_stop())
            else:
                self.assertTrue(early_stoppper.should_early_stop())
