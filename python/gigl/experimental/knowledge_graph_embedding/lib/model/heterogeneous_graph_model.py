from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrec

from gigl.common.logger import Logger
from gigl.experimental.knowledge_graph_embedding.common.torchrec.large_embedding_lookup import (
    LargeEmbeddingLookup,
)
from gigl.experimental.knowledge_graph_embedding.lib.config import ModelConfig
from gigl.experimental.knowledge_graph_embedding.lib.config.sampling import (
    SamplingConfig,
)
from gigl.experimental.knowledge_graph_embedding.lib.data.edge_batch import EdgeBatch
from gigl.experimental.knowledge_graph_embedding.lib.data.node_batch import NodeBatch
from gigl.experimental.knowledge_graph_embedding.lib.model.negative_sampling import (
    against_batch_relationwise_contrastive_similarity,
    in_batch_relationwise_contrastive_similarity,
)
from gigl.experimental.knowledge_graph_embedding.lib.model.types import (
    ModelPhase,
    SimilarityType,
)

logger = Logger()


# TODO(nshah): This could be refactored to be more modular and have individualized APIs for individual KGE model variants.
class HeterogeneousGraphSparseEmbeddingModel(nn.Module):
    """
    A backbone model to support sparse embedding of (possibly multi-relational) graphs.
    Can also be used to implement matrix factorization and variants.

    Useful overviews on Knowledge Graph Embedding:
        - Knowledge Graph Embedding: An Overview (Ge et al, 2023): https://arxiv.org/pdf/2309.12501
        - Stanford CS224W: ML with Graphs: Knowledge Graph Embeddings (2023): https://www.youtube.com/watch?v=isI_TUMoP60

    Args:
        model_config (ModelConfig): Configuration object containing model parameters.
    """

    def __init__(self, model_config: ModelConfig):
        """Initialize the model with the given embedding configurations."""
        super().__init__()

        self.num_edge_types = model_config.num_edge_types
        self.node_emb_dim = model_config.node_embedding_dim
        self.training_sampling_config = model_config.training_sampling
        self.validation_sampling_config = model_config.validation_sampling
        self.testing_sampling_config = model_config.testing_sampling
        self.similarity_type = model_config.embedding_similarity_type
        self._phase: ModelPhase = ModelPhase.TRAIN

        self._assert_sampling_config_is_valid()

        # Define the embedding layers.
        self.large_embeddings = LargeEmbeddingLookup(
            embeddings_config=model_config.embeddings_config
        )

        # Define the operators applied to src and dst node types respectively
        self.src_operator = model_config.src_operator.get_corresponding_module()(
            num_edge_types=self.num_edge_types,
            node_emb_dim=self.node_emb_dim,
        )
        self.dst_operator = model_config.dst_operator.get_corresponding_module()(
            num_edge_types=self.num_edge_types,
            node_emb_dim=self.node_emb_dim,
        )

        logger.info(f"Initialized model with: {self.__dict__}")

    def _assert_sampling_config_is_valid(self):
        for sampling_config in (
            self.training_sampling_config,
            self.validation_sampling_config,
            self.testing_sampling_config,
        ):
            assert sampling_config is not None, "Sampling config must be provided."
            assert (
                sampling_config.positive_edge_batch_size > 0
            ), "Positive edge batch size must be greater than 0."
            assert (
                sampling_config.num_inbatch_negatives_per_edge
                + sampling_config.num_random_negatives_per_edge
                > 0
            ), "At least one type of negative sampling must be specified."

    @property
    def active_sampling_config(self) -> SamplingConfig:
        if self.phase == ModelPhase.TRAIN:
            return self.training_sampling_config
        elif self.phase == ModelPhase.VAL:
            return self.validation_sampling_config
        elif self.phase == ModelPhase.TEST:
            return self.testing_sampling_config
        elif (
            self.phase == ModelPhase.INFERENCE_SRC
            or self.phase == ModelPhase.INFERENCE_DST
        ):
            raise ValueError(
                "Active sampling config is not defined for inference phase. "
            )
        else:
            raise ValueError(
                f"Unknown model phase: {self.phase}. Cannot determine active sampling config."
            )

    def set_phase(self, phase: ModelPhase):
        """
        Set the phase of the model. This is used to determine which sampling
        configuration to use during training, validation, or testing.

        Note that this affects
        (i) how data that is passed into the model is interpreted (e.g. #s of positives, negatives)
        (ii) whether inbatch negatives are used to compute logits and labels

        Args:
            phase (ModelPhase): The current phase of the model (TRAIN, VALIDATION, TEST).
        """
        old_phase = self._phase
        self._phase = phase
        logger.info(f"Changed model phase from {old_phase} to {phase}")

    @property
    def phase(self) -> ModelPhase:
        return self._phase

    def fetch_src_and_dst_embeddings(
        self, edge_batch: EdgeBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_edges = edge_batch.batch_size
        node_embeddings_kt: torchrec.KeyedTensor = self.large_embeddings(
            edge_batch.src_dst_pairs
        )
        logger.debug(f"node embeddings kt: {node_embeddings_kt}")
        node_embeddings = (
            node_embeddings_kt.values()
        )  # [2 * num_edges, num_node_types * node_dim]
        logger.debug(
            f"node embeddings kt values: {node_embeddings, node_embeddings.shape}"
        )
        node_embeddings = node_embeddings.reshape(
            2 * num_edges, -1, self.node_emb_dim
        )  # [2 * num_edges, num_node_types, node_dim]
        logger.debug(
            f"node embeddings values reshaped: {node_embeddings, node_embeddings.shape}"
        )
        node_embeddings = node_embeddings.sum(dim=1)  # [2 * num_edges, node_dim]
        logger.debug(
            f"node embeddings collapse middle axis: {node_embeddings, node_embeddings.shape}"
        )
        node_embeddings = node_embeddings.reshape(
            num_edges, 2, self.node_emb_dim
        )  # [num_edges, 2, node_dim]
        logger.debug(
            f"node embeddings reshape into correct tensor: {node_embeddings, node_embeddings.shape}"
        )
        src_embeddings = node_embeddings[:, 0, :]
        dst_embeddings = node_embeddings[:, 1, :]
        return src_embeddings, dst_embeddings

    def apply_relation_operator(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
        condensed_edge_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the src and dst relation operators to the source and destination embeddings.

        Some reasonable configurations to reimplement common KG embedding algorithms:
        - TransE: Translation on src embeddings, dst embeddings remain unchanged
        - CompleX: Complex diagonal on src embeddings, dst embeddings remain unchanged
        - DistMult: Diagonal on src embeddings, dst embeddings remain unchanged
        - RESCAL: Linear on src embeddings, dst embeddings remain unchanged

        This can also be used to implement things like raw Matrix Factorization
        by using identity operators, or other custom operators.

        Args:
            src_embeddings: Source node embeddings.
            dst_embeddings: Destination node embeddings.
            condensed_edge_types: Edge types for the current batch.
        Returns:
            Tuple of transformed source and destination embeddings.

        """
        # Apply the src operator to the source embeddings
        src_embeddings = self.src_operator(
            embeddings=src_embeddings,
            condensed_edge_types=condensed_edge_types,
        )

        # Apply the dst operator to the destination embeddings
        dst_embeddings = self.dst_operator(
            embeddings=dst_embeddings,
            condensed_edge_types=condensed_edge_types,
        )

        return src_embeddings, dst_embeddings

    def score_edges(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
    ):
        # Compute the scores using the specified scoring function
        if self.similarity_type == SimilarityType.DOT:
            scores = torch.sum(src_embeddings * dst_embeddings, dim=1)
        elif self.similarity_type == SimilarityType.COSINE:
            scores = F.cosine_similarity(src_embeddings, dst_embeddings, dim=1)
        else:
            raise ValueError(f"Unknown scoring function: {self.similarity_type}")

        return scores

    def infer_node_batch(
        self,
        node_batch: NodeBatch,
    ) -> torch.Tensor:
        """
        Infer node embeddings for a given NodeBatch.

        Args:
            node_batch (NodeBatch): The batch of nodes to infer embeddings for.

        Returns:
            torch.Tensor: The inferred node embeddings.
        """
        # Fetch node embeddings from the embedding layer
        num_nodes = node_batch.nodes.values().numel()
        node_embeddings_kt: torchrec.KeyedTensor = self.large_embeddings(
            node_batch.nodes
        )
        node_embeddings = node_embeddings_kt.values()

        node_embeddings = node_embeddings.reshape(
            num_nodes, -1, self.node_emb_dim
        )  # [num_nodes, num_node_types, node_dim]
        node_embeddings = node_embeddings.sum(dim=1)  # [num_nodes, node_dim]

        operator = (
            self.src_operator
            if self.phase == ModelPhase.INFERENCE_SRC
            else self.dst_operator
        )

        # Apply the operator to the node embeddings
        node_embeddings = operator(
            embeddings=node_embeddings,
            condensed_edge_types=node_batch.condensed_edge_type.repeat(num_nodes),
        )
        return node_embeddings

    def forward(
        self, edge_batch: EdgeBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Fetch node embeddings from the embedding layer
        src_embeddings, dst_embeddings = self.fetch_src_and_dst_embeddings(edge_batch)
        condensed_edge_types = edge_batch.condensed_edge_types
        labels = edge_batch.labels

        # Apply relation operator to transform embeddings based on edge types
        src_embeddings, dst_embeddings = self.apply_relation_operator(
            src_embeddings=src_embeddings,
            dst_embeddings=dst_embeddings,
            condensed_edge_types=condensed_edge_types,
        )

        pos_src_embeddings, batch_neg_src_embeddings = (
            src_embeddings[: self.active_sampling_config.positive_edge_batch_size],
            src_embeddings[self.active_sampling_config.positive_edge_batch_size :],
        )
        pos_dst_embeddings, batch_neg_dst_embeddings = (
            dst_embeddings[: self.active_sampling_config.positive_edge_batch_size],
            dst_embeddings[self.active_sampling_config.positive_edge_batch_size :],
        )
        pos_condensed_edge_types, batch_neg_condensed_edge_types = (
            condensed_edge_types[
                : self.active_sampling_config.positive_edge_batch_size
            ],
            condensed_edge_types[
                self.active_sampling_config.positive_edge_batch_size :
            ],
        )
        pos_labels, batch_neg_labels = (
            labels[: self.active_sampling_config.positive_edge_batch_size],
            labels[self.active_sampling_config.positive_edge_batch_size :],
        )

        pos_logits: torch.Tensor = torch.tensor([])
        pos_labels: torch.Tensor = torch.tensor([])
        neg_logits: list[torch.Tensor] = list()
        neg_labels: list[torch.Tensor] = list()

        if self.active_sampling_config.num_inbatch_negatives_per_edge:
            # Do inbatch negative sampling and compute logits and labels
            (
                in_batch_logits,
                in_batch_labels,
            ) = in_batch_relationwise_contrastive_similarity(
                src_embeddings=pos_src_embeddings,
                dst_embeddings=pos_dst_embeddings,
                condensed_edge_types=pos_condensed_edge_types,
                scoring_function=self.similarity_type,
                corrupt_side=self.active_sampling_config.negative_corruption_side,
                num_negatives=self.active_sampling_config.num_inbatch_negatives_per_edge,
            )
            pos_logits = in_batch_logits[:, 0].unsqueeze(1)
            pos_labels = in_batch_labels[:, 0].unsqueeze(1)
            neg_logits.append(in_batch_logits[:, 1:])
            neg_labels.append(in_batch_labels[:, 1:])

        if self.active_sampling_config.num_random_negatives_per_edge:
            (
                against_batch_logits,
                against_batch_labels,
            ) = against_batch_relationwise_contrastive_similarity(
                positive_src_embeddings=pos_src_embeddings,
                positive_dst_embeddings=pos_dst_embeddings,
                positive_condensed_edge_types=pos_condensed_edge_types,
                negative_batch_src_embeddings=batch_neg_src_embeddings,
                negative_batch_dst_embeddings=batch_neg_dst_embeddings,
                batch_condensed_edge_types=batch_neg_condensed_edge_types,
                scoring_function=self.similarity_type,
                corrupt_side=self.active_sampling_config.negative_corruption_side,
                num_negatives=self.active_sampling_config.num_random_negatives_per_edge,
            )

            # These pos_logits and pos_labels are the same as those from in-batch similarity calculations.
            # We keep them here for consistency.
            pos_logits = against_batch_logits[:, 0].unsqueeze(1)
            pos_labels = against_batch_labels[:, 0].unsqueeze(1)
            neg_logits.append(against_batch_logits[:, 1:])
            neg_labels.append(against_batch_labels[:, 1:])

        # Concatenate positive and negative samples
        neg_logits = torch.cat(neg_logits, dim=1)
        neg_labels = torch.cat(neg_labels, dim=1)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.cat([pos_labels, neg_labels], dim=1)

        return logits, labels, pos_condensed_edge_types


class HeterogeneousGraphSparseEmbeddingModelAndLoss(nn.Module):
    """
    A simple heterogeneous information network model with loss. This module
    wraps the `HeterogeneousGraphSparseEmbeddingModel` model for use with
    torchrec TrainPipeline abstraction, which requires specific input/output
    expectations regarding loss and outputs.  This is required by TorchRec's
    convention.  For more details, see:

    - https://github.com/pytorch/torchrec/blob/3ec6f537bf230556b58f5a527ed32e23cc50849d/examples/golden_training/train_dlrm.py#L111
    - https://github.com/pytorch/torchrec/blob/3ec6f537bf230556b58f5a527ed32e23cc50849d/torchrec/models/dlrm.py#L850
    """

    def __init__(
        self,
        encoder_model: HeterogeneousGraphSparseEmbeddingModel,
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = F.binary_cross_entropy_with_logits,
    ):
        """
        Initialize the model with the given encoder model and loss function.

        Args:
            encoder_model (HeterogeneousGraphSparseEmbeddingModel): The underlying model for encoding.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to compute the loss.
                Defaults to binary cross-entropy with logits.
        """

        super().__init__()
        self.encoder_model = encoder_model
        self.loss_fn = loss_fn

    def set_phase(self, phase: ModelPhase):
        """
        Set the phase of the encoder model. This is used to determine which sampling
        configuration to use during training, validation, or testing.

        Note that this affects
        (i) how data that is passed into the model is interpreted (e.g. #s of positives, negatives)
        (ii) whether inbatch negatives are used to compute logits and labels

        Args:
            phase (ModelPhase): The current phase of the model (TRAIN, VAL, TEST, INFERENCE).
        """

        self.encoder_model.set_phase(phase=phase)

    @property
    def phase(self) -> ModelPhase:
        return self.encoder_model.phase

    def forward(self, batch: Union[EdgeBatch, NodeBatch]) -> tuple[torch.Tensor, tuple]:
        """
        If the batch is an EdgeBatch, compute the loss and return it along with
        the logits and labels.

        If the batch is a NodeBatch, infer node embeddings instead.

        Args:
            batch (Union[EdgeBatch, NodeBatch]): The input batch, which can be either an EdgeBatch or a NodeBatch.

        Returns:
            Tuple[torch.Tensor, Tuple]: A tuple containing the loss and a tuple of (loss, logits, labels) for EdgeBatch,
                                        or (dummy_loss, node_embeddings) for NodeBatch.
        """
        if (
            self.phase == ModelPhase.INFERENCE_SRC
            or self.phase == ModelPhase.INFERENCE_DST
        ):
            # In inference phase, we only handle NodeBatch.
            batch: NodeBatch
            node_embeddings = self.encoder_model.infer_node_batch(node_batch=batch)
            dummy_loss = torch.tensor(0.0)
            node_ids = batch.nodes.values()
            return dummy_loss, (
                node_ids.detach(),
                node_embeddings.detach(),
            )
        else:
            # We expect an EdgeBatch in training, validation, or testing phases.
            batch: EdgeBatch
            logits: torch.Tensor
            labels: torch.Tensor
            logits, labels, condensed_edge_types = self.encoder_model(edge_batch=batch)
            loss = self.loss_fn(logits, labels)
            return loss, (
                loss.detach(),
                logits.detach(),
                labels.detach(),
                condensed_edge_types.detach(),
            )
