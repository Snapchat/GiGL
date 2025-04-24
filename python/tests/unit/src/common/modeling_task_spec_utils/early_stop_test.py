import unittest

import torch
import torch.nn as nn

from gigl.src.common.modeling_task_specs.utils.early_stop import EarlyStopper
from gigl.src.common.types.model_eval_metrics import EvalMetricType


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.register_buffer("dummy_value", torch.tensor(0.0))

    def forward(self, x):
        return x


class EarlyStopTests(unittest.TestCase):
    def setUp(self) -> None:
        test_loss_values = [150.0, 100.0, 50.0, 60.0, 70.0, 30.0, 40.0, 50.0, 80.0]
        test_mrr_values = [0.1, 0.3, 0.5, 0.45, 0.4, 0.6, 0.5, 0.4, 0.3]
        self.training_metrics_list = [
            {
                EvalMetricType.loss: test_loss_values[i],
                EvalMetricType.mrr: test_mrr_values[i],
            }
            for i in range(len(test_loss_values))
        ]
        self.early_stop_patience = 3
        self.model = DummyModel()

    def test_mrr_early_stopping(self):
        criterion = EvalMetricType.mrr
        early_stopper = EarlyStopper(
            early_stop_criterion=criterion,
            early_stop_patience=self.early_stop_patience,
        )
        for metric in self.training_metrics_list[:-1]:
            self.assertFalse(
                early_stopper.should_early_stop(metrics=metric, model=self.model)
            )
            self.model.dummy_value += 1
        # We expect this to fail on the last element since we have failed to increase MRR for 3 consecutive checks, which is our patience
        self.assertTrue(
            early_stopper.should_early_stop(
                metrics=self.training_metrics_list[-1], model=self.model
            )
        )
        self.model.load_state_dict(early_stopper.best_val_model)
        self.assertEqual(self.model.dummy_value, 5)

    def test_loss_early_stopping(self):
        criterion = EvalMetricType.loss
        early_stopper = EarlyStopper(
            early_stop_criterion=criterion,
            early_stop_patience=self.early_stop_patience,
        )
        for metric in self.training_metrics_list[:-1]:
            self.assertFalse(
                early_stopper.should_early_stop(metrics=metric, model=self.model)
            )
            self.model.dummy_value += 1
        # We expect this to fail on the last element since we have failed to decrease loss for 3 consecutive checks, which is our patience
        self.assertTrue(
            early_stopper.should_early_stop(
                metrics=self.training_metrics_list[-1], model=self.model
            )
        )
        self.model.load_state_dict(early_stopper.best_val_model)
        self.assertEqual(self.model.dummy_value, 5)

    def tearDown(self) -> None:
        pass
