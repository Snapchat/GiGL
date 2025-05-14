from contextlib import ExitStack
from typing import Callable, Dict, Optional, OrderedDict

import tensorflow as tf
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data
from torch.distributed.algorithms.join import Join, Joinable

from gigl.common.logger import Logger
from gigl.common.utils.torch_training import (
    get_rank,
    is_distributed_available_and_initialized,
)
from gigl.src.common.constants.graph_metadata import DEFAULT_CONDENSED_NODE_TYPE
from gigl.src.common.modeling_task_specs.utils.profiler_wrapper import TorchProfiler
from gigl.src.common.models.pyg.homogeneous import TwoLayerGCN
from gigl.src.common.types.model_eval_metrics import (
    EvalMetric,
    EvalMetricsCollection,
    EvalMetricType,
)
from gigl.src.common.types.pb_wrappers.dataset_metadata import DatasetMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.dataset_metadata_utils import (
    SupervisedNodeClassificationDatasetDataloaders,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.inference.v1.lib.base_inferencer import (
    InferBatchResults,
    SupervisedNodeClassificationBaseInferencer,
    no_grad_eval,
)
from gigl.src.training.v1.lib.base_trainer import BaseTrainer
from gigl.src.training.v1.lib.data_loaders.common import DataloaderTypes
from gigl.src.training.v1.lib.data_loaders.supervised_node_classification_data_loader import (
    SupervisedNodeClassificationBatch,
)
from snapchat.research.gbml import dataset_metadata_pb2

logger = Logger()


class NodeClassificationModelingTaskSpec(
    BaseTrainer, SupervisedNodeClassificationBaseInferencer
):
    def __init__(self, is_training: bool = True, **kwargs) -> None:
        self.__optim_lr = float(kwargs.get("optim_lr", 0.01))
        self.__optim_weight_decay = float(kwargs.get("optim_weight_decay", 5e-4))
        self.__num_epochs = int(kwargs.get("num_epochs", 5))
        self.__out_dim = int(kwargs.get("out_dim", 7))
        self.__is_training = is_training

        main_sample_batch_size = int(kwargs.get("main_sample_batch_size", 16))

        dataloader_batch_size_map: Dict[DataloaderTypes, int] = {
            DataloaderTypes.train_main: main_sample_batch_size,
            DataloaderTypes.val_main: main_sample_batch_size,
            DataloaderTypes.test_main: main_sample_batch_size,
        }
        # TODO (mkolodner-sc): Investigate how we can automatically infer num_worker values
        dataloader_num_workers_map: Dict[DataloaderTypes, int] = {
            DataloaderTypes.train_main: int(kwargs.get("train_main_num_workers", 0)),
            DataloaderTypes.val_main: int(kwargs.get("val_main_num_workers", 0)),
            DataloaderTypes.test_main: int(kwargs.get("test_main_num_workers", 0)),
        }

        self._dataloaders: SupervisedNodeClassificationDatasetDataloaders = (
            SupervisedNodeClassificationDatasetDataloaders(
                batch_size_map=dataloader_batch_size_map,
                num_workers_map=dataloader_num_workers_map,
            )
        )
        super().__init__(**kwargs)

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        self.__model = model

    @property
    def gbml_config_pb_wrapper(self) -> GbmlConfigPbWrapper:
        if not self.__gbml_config_pb_wrapper:
            raise ValueError(
                "gbml_config_pb_wrapper is not initialized before use, "
                "run init_model to set."
            )
        return self.__gbml_config_pb_wrapper

    @property
    def supports_distributed_training(self) -> bool:
        return True

    def init_model(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        state_dict: Optional[OrderedDict[str, torch.Tensor]] = None,
    ) -> torch.nn.Module:
        self.__gbml_config_pb_wrapper = gbml_config_pb_wrapper
        preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper = (
            gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper
        )
        model = TwoLayerGCN(
            in_dim=preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_dim_map[
                DEFAULT_CONDENSED_NODE_TYPE
            ],
            out_dim=self.__out_dim,
            is_training=self.__is_training,
        )
        if state_dict is not None:
            model.load_state_dict(state_dict)
        self.model = model
        self._graph_backend = model.graph_backend

        return model

    def setup_for_training(self):
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.__optim_lr,
            weight_decay=self.__optim_weight_decay,
        )
        self._train_loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = lambda input, target: F.cross_entropy(input=input, target=target)
        self.model.train()

    def _train(
        self, data_loader: torch.utils.data.DataLoader, device: torch.device
    ) -> Optional[torch.Tensor]:
        self.model.train()
        loss: Optional[torch.Tensor] = None
        with ExitStack() as stack:
            if is_distributed_available_and_initialized():
                assert isinstance(
                    self.model, Joinable
                ), "The model should be Joinable, i.e. wrapped with DistributedDataParallel"
                # See https://pytorch.org/tutorials/advanced/generic_join.html for context,
                # also: https://github.com/pytorch/pytorch/issues/38174
                # and: https://github.com/pytorch/pytorch/issues/33148
                # This is needed to train model with unequal batch sizes across different Ranks
                stack.enter_context(Join([self.model]))
                logger.info(f"Model on rank {get_rank()} joined.")
            batch: SupervisedNodeClassificationBatch
            for batch in data_loader:
                self._optimizer.zero_grad()
                inputs = batch.graph.to(device=device)
                root_node_indices = batch.root_node_indices.to(device=device)
                assert (
                    batch.root_node_labels is not None
                ), "Labels required for training."
                root_node_labels = batch.root_node_labels.to(device=device)
                out = self.model(inputs)
                # Figure out why below is a typing issue
                loss = self._train_loss_fn(
                    input=out[root_node_indices], target=root_node_labels
                )  # type: ignore
                loss.backward()
                self._optimizer.step()
        logger.info(
            f"Rank {get_rank()} has exhausted all of its inputs for current epoch of training!"
        )

        if is_distributed_available_and_initialized():
            torch.distributed.barrier()

        return loss

    @no_grad_eval
    def infer_batch(
        self,
        batch: SupervisedNodeClassificationBatch,
        device: torch.device = torch.device("cpu"),
    ) -> InferBatchResults:
        inputs = batch.graph.to(device)
        root_node_indices = batch.root_node_indices.to(device)

        out = self.model(inputs)
        embed = out[root_node_indices]
        pred = embed.argmax(dim=1)
        return InferBatchResults(embeddings=embed, predictions=pred)

    @no_grad_eval
    def score(
        self, data_loader: torch.utils.data.DataLoader, device: torch.device
    ) -> float:
        num_correct = 0
        num_evaluated = 0
        batch: SupervisedNodeClassificationBatch
        for batch in data_loader:
            assert batch.root_node_labels is not None, "Labels required for scoring."
            root_node_labels = batch.root_node_labels.to(device)
            assert root_node_labels is not None

            results: InferBatchResults = self.infer_batch(batch=batch, device=device)
            num_correct_in_batch = int((results.predictions == root_node_labels).sum())
            num_correct += num_correct_in_batch
            num_evaluated += len(batch.root_node_labels)

        logger.info(f"Rank {get_rank()} has exhausted all of its inputs!")
        if is_distributed_available_and_initialized():
            torch.distributed.barrier()
            num_correct_tensor = torch.Tensor([num_correct]).to(device)
            num_evaluated_tensor = torch.Tensor([num_evaluated]).to(device)
            logger.info(
                f"Will reduce num_correct: {num_correct}, and num_evaluated: {num_evaluated}"
            )

            torch.distributed.all_reduce(
                num_correct_tensor, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                num_evaluated_tensor, op=torch.distributed.ReduceOp.SUM
            )
            num_correct = int(num_correct_tensor.item())
            num_evaluated = int(num_evaluated_tensor.item())

        acc = num_correct / num_evaluated
        logger.info(f"Computed acc: {acc}, in rank: {get_rank()}")

        return acc

    def train(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        device: torch.device,
        profiler: Optional[TorchProfiler] = None,
    ) -> None:
        dataset_metadata_pb_wrapper: DatasetMetadataPbWrapper = (
            gbml_config_pb_wrapper.dataset_metadata_pb_wrapper
        )
        assert (
            dataset_metadata_pb_wrapper.output_metadata_type
            == dataset_metadata_pb2.SupervisedNodeClassificationDataset
        ), "Expected a node classification dataset"
        data_loaders = self._dataloaders.get_training_dataloaders(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            graph_backend=self._graph_backend,
            device=device,
        )
        best_val_acc = 0.0
        for epoch in range(self.__num_epochs):
            logger.info(f"Batch training... for epoch {epoch}/{self.__num_epochs }")
            train_loss = self._train(
                data_loader=data_loaders.train_main, device=device  # type: ignore
            )
            train_loss_str = (
                f"{train_loss.item():.3f}" if train_loss is not None else None
            )
            val_acc = self.score(data_loader=data_loaders.val_main, device=device)

            if best_val_acc < val_acc:
                best_val_acc = val_acc

            if train_loss is not None:
                tf.summary.scalar("Train Loss", train_loss.item(), step=epoch)
                tf.summary.scalar("Acc", round(val_acc, 3), step=epoch)

            logger.info(
                f"Train Epoch {epoch}/{self.__num_epochs } | Loss: {train_loss_str} | Val Acc: {val_acc:.3f} | Best Val Acc: {best_val_acc:.3f}"
            )

        logger.info(f"Finished training... ")
        self._dataloaders.cleanup_dataloaders()

    def eval(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        device: torch.device,
    ) -> EvalMetricsCollection:
        logger.info("Start testing... ")
        dataset_metadata_pb_wrapper: DatasetMetadataPbWrapper = (
            gbml_config_pb_wrapper.dataset_metadata_pb_wrapper
        )
        assert (
            dataset_metadata_pb_wrapper.output_metadata_type
            == dataset_metadata_pb2.SupervisedNodeClassificationDataset
        ), "Expected a node classification dataset"

        data_loaders = self._dataloaders.get_test_dataloaders(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            graph_backend=self._graph_backend,
            device=device,
        )
        test_acc = self.score(data_loader=data_loaders.test_main, device=device)

        logger.info(f"global test acc: {test_acc:.3f})")
        test_acc_metric = EvalMetric.from_eval_metric_type(
            eval_metric_type=EvalMetricType.acc, value=test_acc
        )
        model_eval_metrics = EvalMetricsCollection(metrics=[test_acc_metric])
        self._dataloaders.cleanup_dataloaders()
        return model_eval_metrics
