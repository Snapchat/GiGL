from contextlib import ExitStack
from distutils.util import strtobool
from time import time
from typing import Any, Dict, Optional, OrderedDict, Type

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed.algorithms.join import Join, Joinable
from torch.optim.lr_scheduler import LRScheduler

from gigl.common.logger import Logger
from gigl.common.utils import os_utils
from gigl.common.utils.torch_training import (
    get_rank,
    get_world_size,
    is_distributed_available_and_initialized,
)
from gigl.src.common.modeling_task_specs.utils.early_stop import EarlyStopper
from gigl.src.common.modeling_task_specs.utils.infer import infer_task_inputs
from gigl.src.common.modeling_task_specs.utils.profiler_wrapper import TorchProfiler
from gigl.src.common.models.layers.decoder import LinkPredictionDecoder
from gigl.src.common.models.layers.task import (
    NodeAnchorBasedLinkPredictionBaseTask,
    NodeAnchorBasedLinkPredictionTasks,
)
from gigl.src.common.models.pyg.link_prediction import LinkPredictionGNN
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    EdgeType,
    NodeType,
)
from gigl.src.common.types.model_eval_metrics import (
    EvalMetric,
    EvalMetricsCollection,
    EvalMetricType,
)
from gigl.src.common.types.pb_wrappers.dataset_metadata_utils import (
    Dataloaders,
    DataloaderTypes,
    NodeAnchorBasedLinkPredictionDatasetDataloaders,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.common.utils.eval_metrics import hit_rate_at_k, mean_reciprocal_rank
from gigl.src.inference.v1.lib.base_inferencer import (
    InferBatchResults,
    NodeAnchorBasedLinkPredictionBaseInferencer,
    no_grad_eval,
)
from gigl.src.training.v1.lib.base_trainer import BaseTrainer
from gigl.src.training.v1.lib.data_loaders.node_anchor_based_link_prediction_data_loader import (
    NodeAnchorBasedLinkPredictionBatch,
)
from gigl.src.training.v1.lib.data_loaders.rooted_node_neighborhood_data_loader import (
    RootedNodeNeighborhoodBatch,
)
from gigl.src.training.v1.lib.eval_metrics import KS_FOR_EVAL

logger = Logger()


class NodeAnchorBasedLinkPredictionModelingTaskSpec(
    BaseTrainer, NodeAnchorBasedLinkPredictionBaseInferencer
):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        # Model Arguments
        # Supported homogeneous models can be found in gigl.src.common.models.pyg.homogeneous
        # Supported heterogeneous models can be found in gigl.src.common.models.pyg.heterogeneous
        gnn_model_class_path = str(
            kwargs.get(
                "gnn_model_class_path",
                "gigl.src.common.models.pyg.homogeneous.GraphSAGE",
            )
        )

        self.gnn_model: nn.Module
        try:
            self.gnn_model = os_utils.import_obj(gnn_model_class_path)
        except ImportError as e:
            logger.error(f"Could not import {gnn_model_class_path}: {e}")
            raise e

        # Model Arguments
        self.hidden_dim = int(kwargs.get("hidden_dim", 16))
        self.num_layers = int(kwargs.get("num_layers", 2))
        self.out_channels = int(kwargs.get("out_channels", 16))
        self.num_heads = int(kwargs.get("num_heads", 2))

        self.should_l2_normalize_embedding_layer_output = bool(
            kwargs.get("should_l2_normalize_embedding_layer_output", True)
        )

        # Validation Arguments
        self.validate_every_n_batches = int(kwargs.get("val_every_num_batches", 20))
        self.num_val_batches = int(kwargs.get("num_val_batches", 10))
        self.num_test_batches = int(kwargs.get("num_test_batches", 100))

        # Optimizer Arguments
        optim_cls_path = str(kwargs.get("optim_class_path", "torch.optim.Adam"))
        self._optim_cls: Type[torch.optim.Optimizer]
        try:
            self._optim_cls = os_utils.import_obj(optim_cls_path)
        except ImportError as e:
            logger.error(f"Could not import optimizer from {optim_cls_path}: {e}")
            raise e
        self._optim_kwargs: Dict[str, Any] = {}
        self._optim_kwargs["lr"] = float(kwargs.get("optim_lr", 5e-3))
        self._optim_kwargs["weight_decay"] = float(
            kwargs.get("optim_weight_decay", 1e-6)
        )

        self.clip_grad_norm = float(kwargs.get("clip_grad_norm", 0.0))

        # LR Scheduler Arguments
        lr_scheduler_path = kwargs.get(
            "lr_scheduler_name", "torch.optim.lr_scheduler.ConstantLR"
        )
        self._lr_scheduler_cls: Type[LRScheduler]
        try:
            self._lr_scheduler_cls = os_utils.import_obj(lr_scheduler_path)
        except ImportError as e:
            logger.error(f"Could not import LRScheduler from {lr_scheduler_path}: {e}")
            raise e
        self._lr_scheduler_kwargs: Dict[str, Any] = {}
        self._lr_scheduler_kwargs["factor"] = float(kwargs.get("factor", 1.0))
        self._lr_scheduler_kwargs["total_iters"] = int(kwargs.get("total_iters", 10))

        # Dataloader Arguments
        main_sample_batch_size = int(kwargs.get("main_sample_batch_size", 2048))
        random_negative_sample_batch_size = int(
            kwargs.get("random_negative_sample_batch_size", 512)
        )
        random_negative_sample_batch_size_for_evaluation = int(
            kwargs.get("random_negative_sample_batch_size_for_evaluation", 512)
        )
        dataloader_batch_size_map: Dict[DataloaderTypes, int] = {
            DataloaderTypes.train_main: main_sample_batch_size,
            DataloaderTypes.val_main: main_sample_batch_size,
            DataloaderTypes.test_main: main_sample_batch_size,
            DataloaderTypes.train_random_negative: random_negative_sample_batch_size,
            DataloaderTypes.val_random_negative: random_negative_sample_batch_size_for_evaluation,
            DataloaderTypes.test_random_negative: random_negative_sample_batch_size_for_evaluation,
        }

        # TODO (mkolodner-sc): Investigate how we can automatically infer num_worker values
        dataloader_num_workers_map: Dict[DataloaderTypes, int] = {
            DataloaderTypes.train_main: int(kwargs.get("train_main_num_workers", 4)),
            DataloaderTypes.val_main: int(kwargs.get("val_main_num_workers", 2)),
            DataloaderTypes.test_main: int(kwargs.get("test_main_num_workers", 2)),
            DataloaderTypes.train_random_negative: int(
                kwargs.get("train_random_negative_num_workers", 2)
            ),
            DataloaderTypes.val_random_negative: int(
                kwargs.get("val_random_negative_num_workers", 1)
            ),
            DataloaderTypes.test_random_negative: int(
                kwargs.get("test_random_negative_num_workers", 1)
            ),
        }

        self._dataloaders: NodeAnchorBasedLinkPredictionDatasetDataloaders = (
            NodeAnchorBasedLinkPredictionDatasetDataloaders(
                batch_size_map=dataloader_batch_size_map,
                num_workers_map=dataloader_num_workers_map,
            )
        )

        # Early Stop Arguments
        self._early_stop_criterion = EvalMetricType[
            kwargs.get("early_stop_criterion", "loss")
        ]
        self._early_stop_patience = int(kwargs.get("early_stop_patience", 3))
        self._should_maximize = (
            False if self._early_stop_criterion == EvalMetricType.loss else True
        )

        # Loading Task
        self.tasks = NodeAnchorBasedLinkPredictionTasks()

        # Other supported internal models can be found in gigl.src.common.models.layers.task.py
        task_path = kwargs.get(
            "task_path", "gigl.src.common.models.layers.task.Retrieval"
        )
        try:
            base_task: NodeAnchorBasedLinkPredictionBaseTask = os_utils.import_obj(
                task_path
            )
        except ImportError as e:
            logger.error(f"Could not import task from {task_path}: {e}")
            raise e

        logger.info(f"Identified task {base_task}")

        # Retrieval-specific Task Parameters
        softmax_temp = float(kwargs.get("softmax_temp", 0.07))
        should_remove_accidental_hits = bool(
            strtobool(kwargs.get("should_remove_accidental_hits", "True"))
        )
        task = base_task(
            temperature=softmax_temp,
            remove_accidental_hits=should_remove_accidental_hits,
        )

        # Assuming weight of 1.0 for one loss
        self.tasks.add_task(task=task, weight=1.0)

    @property
    def gbml_config_pb_wrapper(self) -> GbmlConfigPbWrapper:
        if not self.__gbml_config_pb_wrapper:
            raise ValueError(
                "gbml_config_pb_wrapper is not initialized before use, "
                "run init_model to set."
            )
        return self.__gbml_config_pb_wrapper

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        self.__model = model

    def init_model(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        state_dict: Optional[OrderedDict[str, torch.Tensor]] = None,
    ) -> torch.nn.Module:
        self.__gbml_config_pb_wrapper = gbml_config_pb_wrapper
        preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper = (
            gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper
        )
        condensed_node_type_to_feat_dim_map: Dict[
            CondensedNodeType, int
        ] = preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_dim_map
        condensed_edge_type_to_feat_dim_map: Dict[
            CondensedEdgeType, int
        ] = preprocessed_metadata_pb_wrapper.condensed_edge_type_to_feature_dim_map
        encoder_model: nn.Module
        if gbml_config_pb_wrapper.graph_metadata_pb_wrapper.is_heterogeneous:
            node_type_to_feat_dim_map: Dict[NodeType, int] = {
                gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                    condensed_node_type
                ]: condensed_node_type_to_feat_dim_map[
                    condensed_node_type
                ]
                for condensed_node_type in condensed_node_type_to_feat_dim_map
            }
            edge_type_to_feat_dim_map: Dict[EdgeType, int] = {
                gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map[
                    condensed_edge_type
                ]: condensed_edge_type_to_feat_dim_map[
                    condensed_edge_type
                ]
                for condensed_edge_type in condensed_edge_type_to_feat_dim_map
            }

            encoder_model = self.gnn_model(
                node_type_to_feat_dim_map=node_type_to_feat_dim_map,
                edge_type_to_feat_dim_map=edge_type_to_feat_dim_map,
                hid_dim=self.hidden_dim,
                out_dim=self.out_channels,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                should_l2_normalize_embedding_layer_output=self.should_l2_normalize_embedding_layer_output,
            )
            logger.info(
                f"Heterogeneous Encoder model will be instantiated with {self.gnn_model}"
            )
        else:
            condensed_node_type = (
                gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_types[0]
            )
            condensed_edge_type = (
                gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_types[0]
            )
            node_feat_dim = condensed_node_type_to_feat_dim_map[condensed_node_type]
            edge_feat_dim = condensed_edge_type_to_feat_dim_map[condensed_edge_type]

            # See gigl.src.common.models.pyg.homogeneous.py for more parameter options
            encoder_model = self.gnn_model(
                in_dim=node_feat_dim,
                hid_dim=self.hidden_dim,
                out_dim=self.out_channels,
                edge_dim=edge_feat_dim if edge_feat_dim > 0 else None,
                num_layers=self.num_layers,
                conv_kwargs={},  # Use default conv args for this model type
                should_l2_normalize_embedding_layer_output=self.should_l2_normalize_embedding_layer_output,
            )

            logger.info(
                f"Homogeneous Encoder model will be instantiated with {self.gnn_model}"
            )

        decoder_model = LinkPredictionDecoder()  # Defaults to inner product decoder

        model: LinkPredictionGNN = LinkPredictionGNN(
            encoder=encoder_model,
            decoder=decoder_model,
            tasks=self.tasks,
        )

        if state_dict is not None:
            logger.info(f"model state dict is {state_dict}")
            model.load_state_dict(state_dict)

        self.model = model
        self.tasks = model.tasks
        self._graph_backend = model.graph_backend

        return self.model

    # function for setting up things like optimizer, scheduler, criterion etc.
    def setup_for_training(self):
        self._optimizer = self._optim_cls(
            params=self.model.parameters(), **self._optim_kwargs
        )
        logger.info(
            f"Using Optimizer={self._optim_cls} with params={self._optim_kwargs}"
        )
        self._lr_scheduler = self._lr_scheduler_cls(
            self._optimizer, **self._lr_scheduler_kwargs
        )
        logger.info(
            f"Using LRScheduler={self._lr_scheduler_cls} with params={self._lr_scheduler_kwargs}"
        )
        self.model.train()

    def train(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        device: torch.device,
        profiler: Optional[TorchProfiler] = None,
    ):
        """
        Main Training loop for the model.

        Args:
            gbml_config_pb_wrapper: GbmlConfigPbWrapper for gbmlConfig proto
            device: torch.device to run the training on
            profiler: Profiler object to profile the training
        """

        # Retrieving training and validation dataloaders
        data_loaders: Dataloaders = self._dataloaders.get_training_dataloaders(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            graph_backend=self._graph_backend,
            device=device,
        )

        self._early_stopper = EarlyStopper(
            early_stop_patience=self._early_stop_patience,
            should_maximize=self._should_maximize,
            model=self.model,
        )

        main_data_loader = data_loaders.train_main
        random_negative_data_loader = data_loaders.train_random_negative
        val_main_data_loader_iter = iter(data_loaders.val_main)  # type: ignore
        val_random_data_loader_iter = iter(data_loaders.val_random_negative)  # type: ignore

        main_batch: NodeAnchorBasedLinkPredictionBatch
        random_negative_batch: RootedNodeNeighborhoodBatch

        logger.info("Starting Training...")
        with ExitStack() as stack:
            if is_distributed_available_and_initialized():
                assert isinstance(self.model, Joinable)
                stack.enter_context(Join([self.model]))
                logger.info(f"Model on rank {get_rank()} joined.")

            self.model.train()

            for batch_index, (main_batch, random_negative_batch) in enumerate(
                zip(main_data_loader, random_negative_data_loader), start=1  # type: ignore
            ):
                batch_st = time()
                self._optimizer.zero_grad()

                # Retrieving all possible task inputs
                # See gigl.src.common.types.task_inputs.py for more info on possible task inputs
                task_inputs = infer_task_inputs(
                    model=self.model,
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                    main_batch=main_batch,
                    random_neg_batch=random_negative_batch,
                    should_eval=False,
                    device=device,
                )

                # Handling loss calculation, can optionally return breakdown of loss per loss_type in multi-loss setting
                (
                    loss,
                    _,
                ) = self.tasks.calculate_losses(
                    batch_results=task_inputs,
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                    should_eval=False,
                    device=device,
                )
                loss.backward()
                if self.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self._optimizer.step()
                self._lr_scheduler.step()

                elapsed_batch_time = time() - batch_st
                logger.info(
                    f"Batch {batch_index} for rank {get_rank()} (took {elapsed_batch_time:.3f}s), "
                    f"Loss: {loss.item()}, "
                    f"last batch learning rate: {self._lr_scheduler.get_last_lr()[0]:.4f}"
                )

                # Validate every so often
                if (
                    batch_index % (self.validate_every_n_batches // get_world_size())
                    == 0
                ):
                    if is_distributed_available_and_initialized():
                        torch.distributed.barrier()
                    logger.info(f"Validating at batch {batch_index}")

                    eval_metrics = self.validate(
                        main_data_loader=val_main_data_loader_iter,
                        random_negative_data_loader=val_random_data_loader_iter,
                        gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                        device=device,
                        num_batches=self.num_val_batches,
                    )

                    has_metric_improved, should_early_stop = self._early_stopper.step(
                        value=eval_metrics[self._early_stop_criterion]
                    )
                    # Check if we need to early stop based on patience
                    if should_early_stop:
                        break

                profiler.step() if profiler else None  # type: ignore

        logger.info(
            f"Reverting model to parameters which achieved best val {self._early_stop_criterion.name}: {self._early_stopper.best_criterion:.3f}."
        )

        # Asserting our model has improved for at least one validation check
        assert self._early_stopper.best_model_state_dict is not None

        # Loading model with highest validation performance
        self.model.load_state_dict(self._early_stopper.best_model_state_dict)

        if is_distributed_available_and_initialized():
            torch.distributed.barrier()

        self._dataloaders.cleanup_dataloaders()

    @no_grad_eval
    def validate(
        self,
        main_data_loader: torch.utils.data.dataloader._BaseDataLoaderIter,
        random_negative_data_loader: torch.utils.data.dataloader._BaseDataLoaderIter,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        device: torch.device,
        num_batches: int,
    ) -> Dict[EvalMetricType, Any]:
        """
        Get the validation metrics for the model using the similarity scores for the positive and negative samples.

        Args:
            main_data_loader: DataLoader for the positive samples
            random_negative_data_loader: DataLoader for the random negative samples
            device: torch.device to run the validation on

        Returns:
            Dict[str, Any]: Metrics for validation
        """

        self.model.eval()

        # hit@k values for k are defaulted to [1, 5, 10, 50, 100, 500]
        ks_for_evaluation = torch.LongTensor(KS_FOR_EVAL).to(device)
        num_nodes_for_rank_eval_computation = 0
        # Currently support mrr, loss, and hits@k metrics for validation and testing
        metrics: Dict[EvalMetricType, torch.Tensor] = {
            EvalMetricType.mrr: torch.FloatTensor([0.0]).to(device),
            EvalMetricType.loss: torch.FloatTensor([0.0]).to(device),
            EvalMetricType.hits: torch.zeros_like(
                ks_for_evaluation, dtype=torch.float32
            ).to(device),
        }
        final_metrics: Dict[EvalMetricType, Any] = {}

        if is_distributed_available_and_initialized():
            # In cases of uneven batch sizes per rank, we force to be even, reducing overall batch size to evaluate
            num_batches_per_rank = num_batches // get_world_size()
        else:
            num_batches_per_rank = num_batches

        logger.info(
            f"Started evaluation on rank {get_rank()} reading up to {num_batches_per_rank} batches."
        )

        for batch_idx, (main_batch, random_negative_batch) in enumerate(
            zip(main_data_loader, random_negative_data_loader)
        ):
            if batch_idx >= num_batches_per_rank:
                break

            # Retrieving all possible task inputs
            # See gigl.src.common.types.task_inputs.py for more info on possible task inputs
            # Always returns batch-scores when should_eval=True
            task_inputs = infer_task_inputs(
                model=self.model,
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                main_batch=main_batch,
                random_neg_batch=random_negative_batch,
                should_eval=True,
                device=device,
            )

            # Handling loss calculation, can optionally return breakdown of loss per loss_type in multi-loss setting
            (
                loss,
                _,
            ) = self.tasks.calculate_losses(
                batch_results=task_inputs,
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                should_eval=True,
                device=device,
            )

            metrics[EvalMetricType.loss] += loss

            for result in task_inputs.batch_scores:
                for condensed_edge_type, batch_scores in result.items():
                    # Can only compute HR / MRR if we have at least 1 positive.
                    if batch_scores.pos_scores.numel():
                        num_nodes_for_rank_eval_computation += 1
                        hr_result = hit_rate_at_k(
                            pos_scores=batch_scores.pos_scores,
                            neg_scores=batch_scores.random_neg_scores,
                            ks=ks_for_evaluation,  # type: ignore
                        )
                        mrr_result = mean_reciprocal_rank(
                            pos_scores=batch_scores.pos_scores,
                            neg_scores=batch_scores.random_neg_scores,
                        )
                        metrics[EvalMetricType.hits] += hr_result
                        metrics[EvalMetricType.mrr] += mrr_result

        metrics[EvalMetricType.hits] /= num_nodes_for_rank_eval_computation
        metrics[EvalMetricType.mrr] /= num_nodes_for_rank_eval_computation
        metrics[EvalMetricType.loss] /= num_batches_per_rank
        logger.info(f"Rank {get_rank()} finished reading inputs for scoring.")
        # Reduce the total_mrr, total_loss and total_hit_rates across all ranks (DDP)
        if is_distributed_available_and_initialized():
            torch.distributed.barrier()
            for metric in metrics:
                metric_value = metrics[metric]
                torch.distributed.all_reduce(
                    metric_value, op=torch.distributed.ReduceOp.SUM
                )
                metrics[metric] = metric_value / get_world_size()

        for metric in metrics:
            metric_result = metrics[metric]
            if metric_result.shape[0] > 1:
                final_metrics[metric] = metric_result.tolist()
            else:
                final_metrics[metric] = metric_result.item()

        logger.info(
            f"Computed Hits@{ks_for_evaluation.tolist()}: {final_metrics[EvalMetricType.hits]}, MRR: {final_metrics[EvalMetricType.mrr]:.3f}, Loss: {final_metrics[EvalMetricType.loss]:.3f}, across all ranks"
        )
        return final_metrics

    def eval(
        self, gbml_config_pb_wrapper: GbmlConfigPbWrapper, device: torch.device
    ) -> EvalMetricsCollection:
        """
        Evaluate the model using the test data loaders.

        Args:
            gbml_config_pb_wrapper: GbmlConfigPbWrapper for gbmlConfig proto
            device: torch.device to run the evaluation on
        """

        logger.info("Start testing...")
        # Retrieving testing dataloaders
        data_loaders: Dataloaders = self._dataloaders.get_test_dataloaders(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            graph_backend=self._graph_backend,
            device=device,
        )

        eval_metrics = self.validate(
            main_data_loader=iter(data_loaders.test_main),  # type: ignore
            random_negative_data_loader=iter(data_loaders.test_random_negative),  # type: ignore
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
            num_batches=self.num_test_batches,
        )

        hit_rates_model_metrics = [
            EvalMetric(
                name=f"HitRate_at_{k}",
                value=rate,
            )
            for k, rate in zip(KS_FOR_EVAL, eval_metrics[EvalMetricType.hits])
        ]

        metric_list = [
            EvalMetric.from_eval_metric_type(
                eval_metric_type=EvalMetricType.mrr,
                value=eval_metrics[EvalMetricType.mrr],
            ),
            EvalMetric.from_eval_metric_type(
                eval_metric_type=EvalMetricType.loss,
                value=eval_metrics[EvalMetricType.loss],
            ),
            *hit_rates_model_metrics,
        ]

        metrics = EvalMetricsCollection(metrics=metric_list)

        self._dataloaders.cleanup_dataloaders()
        return metrics

    @no_grad_eval
    def infer_batch(
        self,
        batch: RootedNodeNeighborhoodBatch,
        device: torch.device = torch.device("cpu"),
    ) -> InferBatchResults:
        batch_graph = batch.graph.to(device=device)
        batch_root_condensed_node_types = list(
            batch.condensed_node_type_to_root_node_indices_map.keys()
        )
        batch_root_node_types = [
            self.gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                condensed_node_type
            ]
            for condensed_node_type in batch_root_condensed_node_types
        ]
        assert (
            len(batch_root_node_types) == 1
        ), f"{RootedNodeNeighborhoodBatch.__name__} for inference must have only one root node type. Found root node types: {batch_root_node_types}"
        output_node_type, output_condensed_node_type = (
            batch_root_node_types[0],
            batch_root_condensed_node_types[0],
        )
        batch_root_node_indices = batch.condensed_node_type_to_root_node_indices_map[
            output_condensed_node_type
        ].to(device=device)
        out = self.model(
            data=batch_graph, output_node_types=[output_node_type], device=device
        )[output_node_type]
        embed = out[batch_root_node_indices]
        return InferBatchResults(embeddings=embed, predictions=None)

    @property
    def supports_distributed_training(self) -> bool:
        return True
