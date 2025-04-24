from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms.join import Join, Joinable
from torch_geometric.nn import GraphSAGE

from gigl.common.logger import Logger
from gigl.common.utils.torch_training import (
    get_rank,
    get_world_size,
    is_distributed_available_and_initialized,
)
from gigl.src.common.graph_builder.graph_builder_factory import GraphBuilderFactory
from gigl.src.common.types.graph_data import CondensedEdgeType, CondensedNodeType
from gigl.src.common.types.model import GraphBackend
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
from gigl.src.common.utils.eval_metrics import hit_rate_at_k
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
from gigl.src.training.v1.lib.eval_metrics import KS_FOR_EVAL as ks

logger = Logger()


class GraphSageTemplateTrainerSpec(
    BaseTrainer, NodeAnchorBasedLinkPredictionBaseInferencer
):
    """
    Template Simple Training Spec that uses GraphSAGE for Node Anchor Based Link Prediction with DDP support.
    Arguments are to be passed in via trainerArgs in GBML Config.

    Args:
        hidden_dim (int): Hidden dimension to use for the model (default: 64)
        num_layers (int): Number of layers to use for the model (default: 2)
        out_channels (int): Output channels to use for the model (default: 64)
        validate_every_n_batches (int): Number of batches to validate after (default: 20)
        num_val_batches (int): Number of batches to validate on (default: 10)
        num_test_batches (int): Number of batches to test on (default: 100)
        early_stop_patience (int): Number of consecutive checks without improvement to trigger early stopping (default: 3)
        num_epochs (int): Number of epochs to train the model for (default: 5)
        optim_lr (float): Learning rate to use for the optimizer (default: 0.001)
        main_sample_batch_size (int): Batch size to use for the main samples (default: 256)
        random_negative_batch_size (int): Batch size to use for the random negative samples (default: 64)
        train_main_num_workers (int): Number of workers to use for the train main dataloader (default: 2)
        val_main_num_workers (int): Number of workers to use for the val main dataloader (default: 1)

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.hidden_dim = int(kwargs.get("hidden_dim", 64))
        self.num_layers = int(kwargs.get("num_layers", 2))
        self.out_channels = int(kwargs.get("out_channels", 64))
        self.validate_every_n_batches = int(kwargs.get("validate_every_n_batches", 20))
        self.num_val_batches = int(kwargs.get("num_val_batches", 10))
        self.num_test_batches = int(kwargs.get("num_test_batches", 100))
        self.early_stop_patience = int(kwargs.get("early_stop_patience", 3))
        self.num_epochs = int(kwargs.get("num_epochs", 5))
        self.optim_lr = float(kwargs.get("optim_lr", 0.001))
        self.main_sample_batch_size = int(kwargs.get("main_sample_batch_size", 256))
        self.random_negative_batch_size = int(
            kwargs.get("random_negative_batch_size", 64)
        )
        self._graph_builder = GraphBuilderFactory.get_graph_builder(
            backend_name=GraphBackend("PyG")
        )

        # Prepare dataloader configurations
        dataloader_batch_size_map: Dict[DataloaderTypes, int] = {
            DataloaderTypes.train_main: self.main_sample_batch_size,
            DataloaderTypes.val_main: self.main_sample_batch_size,
            DataloaderTypes.test_main: self.main_sample_batch_size,
            DataloaderTypes.train_random_negative: self.random_negative_batch_size,
            DataloaderTypes.val_random_negative: self.random_negative_batch_size,
            DataloaderTypes.test_random_negative: self.random_negative_batch_size,
        }

        dataloader_num_workers_map: Dict[DataloaderTypes, int] = {
            DataloaderTypes.train_main: int(kwargs.get("train_main_num_workers", 2)),
            DataloaderTypes.val_main: int(kwargs.get("val_main_num_workers", 1)),
            DataloaderTypes.test_main: int(kwargs.get("test_main_num_workers", 1)),
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

        # Utility for data loader initialization
        self._dataloaders: NodeAnchorBasedLinkPredictionDatasetDataloaders = (
            NodeAnchorBasedLinkPredictionDatasetDataloaders(
                batch_size_map=dataloader_batch_size_map,
                num_workers_map=dataloader_num_workers_map,
            )
        )

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        self.__model = model
        self.__model.graph_backend = GraphBackend.PYG  # type: ignore

    def init_model(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        state_dict: Optional[dict] = None,
        device: torch.device = torch.device("cuda"),
    ) -> nn.Module:
        node_feat_dim = gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_dim_map[
            CondensedNodeType(0)
        ]
        model = GraphSAGE(
            in_channels=node_feat_dim,
            hidden_channels=self.hidden_dim,
            num_layers=self.num_layers,
            out_channels=self.out_channels,
        )
        self.model = model
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        return self.model

    # function for setting up things like optimizer, scheduler, criterion etc.
    def setup_for_training(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optim_lr)

    def train(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        device: torch.device,
        profiler=None,
    ):
        """
        Main Training loop for the GraphSAGE model.

        Args:
            gbml_config_pb_wrapper: GbmlConfigPbWrapper for gbmlConfig proto
            device: torch.device to run the training on
            num_epochs: Number of epochs to train the model for
            profiler: Profiler object to profile the training
        """
        early_stop_counter = 0
        best_val_loss = float("inf")

        data_loaders: Dataloaders = self._dataloaders.get_training_dataloaders(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            graph_backend=self.model.graph_backend,
            device=device,
        )

        assert (
            data_loaders.train_main is not None
            and data_loaders.val_main is not None
            and data_loaders.train_random_negative is not None
            and data_loaders.val_random_negative is not None
        )

        logger.info("Data loaders initialized")

        main_data_loader = data_loaders.train_main
        random_negative_data_loader = data_loaders.train_random_negative
        val_main_data_loader_iter = iter(data_loaders.val_main)
        val_random_data_loader_iter = iter(data_loaders.val_random_negative)

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
                zip(main_data_loader, random_negative_data_loader), start=1
            ):
                pos_scores, hard_neg_scores, random_neg_scores = self._process_batch(
                    main_batch=main_batch,
                    random_negative_batch=random_negative_batch,
                    device=device,
                )
                loss = self._compute_loss(
                    pos_scores, hard_neg_scores, random_neg_scores, device
                )

                logger.info(f"Processed batch {batch_index}, Loss: {loss.item()}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (
                    batch_index % (self.validate_every_n_batches // get_world_size())
                    == 0
                ):
                    if is_distributed_available_and_initialized():
                        torch.distributed.barrier()
                    logger.info(f"Validating at batch {batch_index}")
                    avg_val_loss = self.validate(
                        val_main_data_loader_iter,
                        val_random_data_loader_iter,
                        device,
                    )

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        early_stop_counter = 0
                        logger.info(
                            f"Validation Loss Improved to {best_val_loss:.4f}. Resetting early stop counter."
                        )
                    else:
                        early_stop_counter += 1
                        logger.info(
                            f"No improvement in Validation Loss for {early_stop_counter} consecutive checks. Early Stop Counter: {early_stop_counter}"
                        )

                    if early_stop_counter >= self.early_stop_patience:
                        logger.info(
                            f"Early stopping triggered after {early_stop_counter} checks without improvement. Best Validation Loss: {best_val_loss:.4f}"
                        )
                        break

                    logger.info(
                        f"Validation Loss: {avg_val_loss:.4f} at batch {batch_index}"
                    )

    def _compute_loss(
        self,
        pos_scores_list: List[torch.Tensor],
        hard_neg_scores_list: List[torch.Tensor],
        random_neg_scores_list: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        total_loss: torch.Tensor = torch.tensor(0.0, device=device)
        total_sample_size = 0
        for pos_scores, hard_neg_scores, random_neg_scores in zip(
            pos_scores_list, hard_neg_scores_list, random_neg_scores_list
        ):
            all_neg_scores = torch.cat((hard_neg_scores, random_neg_scores), dim=1)
            # shape=[1, num_hard_neg_nodes + num_random_neg_nodes]
            if all_neg_scores.numel() > 0 and pos_scores.numel() > 0:
                all_neg_scores_repeated = all_neg_scores.repeat(1, pos_scores.shape[1])
                pos_scores_repeated = pos_scores.repeat_interleave(
                    all_neg_scores.shape[1], dim=1
                )
                targets = torch.ones_like(pos_scores_repeated).to(device=device)
                loss = F.margin_ranking_loss(
                    input1=pos_scores_repeated,
                    input2=all_neg_scores_repeated,
                    target=targets,
                    margin=0.5,
                    reduction="sum",
                )
                total_loss += loss
                total_sample_size += pos_scores_repeated.numel()

        if total_sample_size > 0:
            average_loss = total_loss / total_sample_size
        else:
            average_loss = torch.tensor(0.0, device=device)

        return average_loss

    def _process_batch(
        self,
        main_batch: NodeAnchorBasedLinkPredictionBatch,
        random_negative_batch: RootedNodeNeighborhoodBatch,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        main_embeddings = self.model(
            main_batch.graph.x.to(device), main_batch.graph.edge_index.to(device)
        )
        random_negative_embeddings = self.model(
            random_negative_batch.graph.x.to(device),
            random_negative_batch.graph.edge_index.to(device),
        )

        pos_score_list: List[torch.Tensor] = []
        hard_neg_score_list: List[torch.Tensor] = []
        random_neg_score_list: List[torch.Tensor] = []

        main_batch_root_node_indices = main_batch.root_node_indices.to(device=device)

        # For homogenous graph, we only have one condensed node type
        random_neg_root_node_indices = (
            random_negative_batch.condensed_node_type_to_root_node_indices_map[
                CondensedNodeType(0)
            ].to(device=device)
        )

        # inner product (decoder)
        batch_random_neg_scores = torch.mm(
            main_embeddings[main_batch_root_node_indices],
            random_negative_embeddings[random_neg_root_node_indices].T,
        )

        for root_node_index, root_node in enumerate(main_batch_root_node_indices):
            root_node = torch.unsqueeze(root_node, 0)
            pos_scores = torch.FloatTensor([]).to(device=device)
            hard_neg_scores = torch.FloatTensor([]).to(device=device)

            pos_nodes: torch.LongTensor = main_batch.pos_supervision_edge_data[
                CondensedEdgeType(0)
            ].root_node_to_target_node_id[root_node.item()]

            if pos_nodes.numel():
                pos_scores = torch.mm(
                    main_embeddings[root_node], main_embeddings[pos_nodes].T
                )

            hard_neg_nodes: (
                torch.LongTensor
            ) = main_batch.hard_neg_supervision_edge_data[
                CondensedEdgeType(0)
            ].root_node_to_target_node_id[
                root_node.item()
            ]  # shape=[num_hard_neg_nodes]

            if hard_neg_nodes.numel():
                hard_neg_scores = torch.mm(
                    main_embeddings[root_node], main_embeddings[hard_neg_nodes].T
                )

            random_neg_scores = batch_random_neg_scores[[root_node_index], :].to(
                device=device
            )

            pos_score_list.append(pos_scores)
            hard_neg_score_list.append(hard_neg_scores)
            random_neg_score_list.append(random_neg_scores)

        return pos_score_list, hard_neg_score_list, random_neg_score_list

    @no_grad_eval
    def validate(
        self,
        main_data_loader: torch.utils.data.dataloader._BaseDataLoaderIter,
        random_negative_data_loader: torch.utils.data.dataloader._BaseDataLoaderIter,
        device: torch.device,
    ) -> float:
        """
        Get the validation loss for the model using the similarity scores for the positive and negative samples.

        Args:
            main_data_loader: DataLoader for the positive samples
            random_negative_data_loader: DataLoader for the random negative samples
            device: torch.device to run the validation on

        Returns:
            float: Average validation loss
        """
        validation_metrics = self._compute_metrics(
            main_data_loader=main_data_loader,
            random_negative_data_loader=random_negative_data_loader,
            device=device,
            num_batches=self.num_val_batches,
        )

        avg_val_mrr = validation_metrics["avg_mrr"]
        avg_val_loss = validation_metrics["avg_loss"]

        logger.info(f"Validation got MRR: {avg_val_mrr}")

        return avg_val_loss

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

        data_loaders: Dataloaders = self._dataloaders.get_test_dataloaders(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            graph_backend=self.model.graph_backend,
            device=device,
        )

        assert (
            data_loaders.test_main is not None
            and data_loaders.test_random_negative is not None
        )

        eval_metrics = self._compute_metrics(
            main_data_loader=iter(data_loaders.test_main),
            random_negative_data_loader=iter(data_loaders.test_random_negative),
            device=device,
            num_batches=self.num_test_batches,
        )

        avg_mrr = eval_metrics["avg_mrr"]
        avg_hit_rates = eval_metrics["avg_hit_rates"]

        logger.info(f"Average MRR: {avg_mrr}")
        for k, hit_rate in zip(ks, avg_hit_rates):
            logger.info(f"Hit Rate@{k}: {hit_rate.item()}")
        hit_rates_model_metrics = [
            EvalMetric(
                name=f"HitRate_at_{k}",
                value=rate,
            )
            for k, rate in zip(ks, avg_hit_rates)
        ]
        metric_list = [
            EvalMetric.from_eval_metric_type(
                eval_metric_type=EvalMetricType.mrr,
                value=avg_mrr,
            ),
            *hit_rates_model_metrics,
        ]

        metrics = EvalMetricsCollection(metrics=metric_list)

        return metrics

    def _compute_metrics(
        self,
        main_data_loader: torch.utils.data.dataloader._BaseDataLoaderIter,
        random_negative_data_loader: torch.utils.data.dataloader._BaseDataLoaderIter,
        device: torch.device,
        num_batches: int,
    ) -> Dict[str, Any]:
        self.model.eval()
        total_mrr: float = 0.0
        total_loss: float = 0.0
        ks = [1, 5, 10, 50, 100, 500]
        total_hit_rates = torch.zeros(len(ks), device=device)
        num_batches_processed = 0

        if is_distributed_available_and_initialized():
            num_batches_per_rank = num_batches // get_world_size()
        else:
            num_batches_per_rank = num_batches

        with torch.no_grad():
            for batch_idx, (main_batch, random_negative_batch) in enumerate(
                zip(main_data_loader, random_negative_data_loader), start=1
            ):
                if batch_idx > num_batches_per_rank:
                    break

                (
                    pos_scores_list,
                    hard_neg_scores_list,
                    random_neg_scores_list,
                ) = self._process_batch(
                    main_batch=main_batch,
                    random_negative_batch=random_negative_batch,
                    device=device,
                )

                loss = self._compute_loss(
                    pos_scores_list=pos_scores_list,
                    hard_neg_scores_list=hard_neg_scores_list,
                    random_neg_scores_list=random_neg_scores_list,
                    device=device,
                )

                total_loss += loss.item()

                for pos_scores, hard_neg_scores, random_neg_scores in zip(
                    pos_scores_list, hard_neg_scores_list, random_neg_scores_list
                ):
                    neg_scores = torch.cat((hard_neg_scores, random_neg_scores), dim=1)

                    if pos_scores.numel() == 0:
                        continue

                    combined_scores = torch.cat((pos_scores, neg_scores), dim=1)
                    ranks = torch.argsort(combined_scores, dim=1, descending=True)
                    pos_rank = (ranks == 0).nonzero(as_tuple=True)[1] + 1
                    mrr = 1.0 / pos_rank.float()

                    hit_rates = hit_rate_at_k(
                        pos_scores=pos_scores,  # type: ignore
                        neg_scores=neg_scores,  # type: ignore
                        ks=torch.tensor(ks, device=device, dtype=torch.long),  # type: ignore
                    )

                    total_mrr += mrr.mean().item()
                    total_hit_rates += hit_rates.to(device)
                    num_batches_processed += 1

        # Reduce the total_mrr, total_loss and total_hit_rates across all ranks (DDP)
        if is_distributed_available_and_initialized():
            total_mrr_tensor = torch.tensor(total_mrr, device=device)
            torch.distributed.all_reduce(
                total_mrr_tensor, op=torch.distributed.ReduceOp.SUM
            )
            total_mrr = total_mrr_tensor.item() / get_world_size()
            total_loss_tensor = torch.tensor(total_loss, device=device)
            torch.distributed.all_reduce(
                total_loss_tensor, op=torch.distributed.ReduceOp.SUM
            )
            total_loss = total_loss_tensor.item()
            torch.distributed.all_reduce(
                total_hit_rates, op=torch.distributed.ReduceOp.SUM
            )
            total_hit_rates /= get_world_size()
            num_batches_tensor = torch.tensor(num_batches_processed, device=device)
            torch.distributed.all_reduce(
                num_batches_tensor, op=torch.distributed.ReduceOp.SUM
            )
            num_batches_processed = int(num_batches_tensor.item())

        avg_mrr = total_mrr / num_batches_processed if num_batches_processed > 0 else 0
        avg_loss = (
            total_loss / num_batches_processed if num_batches_processed > 0 else 0
        )
        avg_hit_rates = (
            total_hit_rates / num_batches_processed
            if num_batches_processed > 0
            else torch.zeros(len(ks), device=device)
        )

        metrics = {
            "avg_mrr": avg_mrr,
            "avg_loss": avg_loss,
            "avg_hit_rates": avg_hit_rates,
        }

        return metrics

    @no_grad_eval
    def infer_batch(
        self,
        batch: RootedNodeNeighborhoodBatch,
        device: torch.device = torch.device("cpu"),
    ) -> InferBatchResults:
        out = self.model(batch.graph.x.to(device), batch.graph.edge_index.to(device))
        batch_root_node_indices = batch.condensed_node_type_to_root_node_indices_map[
            CondensedNodeType(0)  # For homogenous graph only one condensed node type
        ].to(device=device)

        embeddings = out[batch_root_node_indices]
        return InferBatchResults(embeddings=embeddings, predictions=None)

    @property
    def supports_distributed_training(self) -> bool:
        return True
