from typing import Optional, OrderedDict

import torch.utils.data

from gigl.src.common.modeling_task_specs.utils.profiler_wrapper import TorchProfiler
from gigl.src.common.types.model_eval_metrics import (
    EvalMetric,
    EvalMetricsCollection,
    EvalMetricType,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.inference.v1.lib.base_inferencer import (
    InferBatchResults,
    SupervisedNodeClassificationBaseInferencer,
    no_grad_eval,
)
from gigl.src.training.v1.lib.base_trainer import BaseTrainer
from gigl.src.training.v1.lib.data_loaders.supervised_node_classification_data_loader import (
    SupervisedNodeClassificationBatch,
)


class TestModelingTaskSpec(BaseTrainer, SupervisedNodeClassificationBaseInferencer):
    def __init__(self, is_training: bool = True, **kwargs) -> None:
        self._optim_lr = float(kwargs.get("optim_lr", 0.01))
        self._optim_weight_decay = float(kwargs.get("optim_weight_decay", 5e-4))
        self._num_epochs = int(kwargs.get("num_epochs", 5))
        super().__init__(**kwargs)

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        self.__model = model

    @property
    def supports_distributed_training(self) -> bool:
        return False

    def init_model(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        state_dict: Optional[OrderedDict[str, torch.Tensor]] = None,
    ) -> torch.nn.Module:
        return self.model

    def setup_for_training(self):
        pass

    @no_grad_eval
    def infer_batch(
        self,
        batch: SupervisedNodeClassificationBatch,
        device: torch.device = torch.device("cpu"),
    ) -> InferBatchResults:
        return InferBatchResults(embeddings=None, predictions=None)

    @no_grad_eval
    def score(
        self, data_loader: torch.utils.data.DataLoader, device: torch.device
    ) -> float:
        return 1.0

    def train(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        device: torch.device,
        profiler: Optional[TorchProfiler] = None,
    ) -> None:
        pass

    def eval(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        device: torch.device,
    ) -> EvalMetricsCollection:
        model_metric = EvalMetric.from_eval_metric_type(
            eval_metric_type=EvalMetricType.acc, value=1.0
        )
        model_eval_metrics = EvalMetricsCollection(metrics=[model_metric])
        return model_eval_metrics
