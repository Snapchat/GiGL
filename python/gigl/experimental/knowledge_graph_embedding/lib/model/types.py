from enum import Enum
from typing import Callable, Type

import torch
import torch.nn.functional as F
from gigl.experimental.knowledge_graph_embedding.lib.model.operators import (
    ComplexDiagonalOperator,
    DiagonalOperator,
    IdentityOperator,
    LinearOperator,
    RelationwiseOperatorBase,
    TranslationOperator,
)

# Many knowledge graph-style embeddings involve pairing relation operators and scoring functions.


class SimilarityType(str, Enum):
    """Enum for different scoring functions."""

    DOT = "DOT"
    COSINE = "COSINE"
    EUCLIDEAN = "EUCLIDEAN"

    def __str__(self):
        return self.value

    def get_similarity_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns the corresponding similarity function for the similarity type.

        The returned function takes two tensors x and y and returns their similarity
        matrix. For x with shape [B, D] and y with shape [N, D], returns [B, N].

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: The similarity function.
        """
        if self == SimilarityType.DOT:
            return lambda x, y: x @ y.T
        elif self == SimilarityType.COSINE:
            return lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).T
        elif self == SimilarityType.EUCLIDEAN:

            def euclidean_sim(x, y):
                x_sq = x.pow(2).sum(dim=1, keepdim=True)  # [B, 1]
                y_sq = y.pow(2).sum(dim=1, keepdim=True)  # [N, 1]
                return -(x_sq - 2 * x @ y.T + y_sq.T)  # negative squared L2

            return euclidean_sim
        else:
            raise ValueError(f"Unknown similarity type: {self}")


class OperatorType(str, Enum):
    """
    Enum for different types of relation operators.
    """

    TRANSLATION = "TRANSLATION"
    DIAGONAL = "DIAGONAL"
    COMPLEX_DIAGONAL = "COMPLEX_DIAGONAL"
    LINEAR = "LINEAR"
    IDENTITY = "IDENTITY"

    def __str__(self):
        return self.value

    def get_corresponding_module(self) -> Type[RelationwiseOperatorBase]:
        """
        Returns the corresponding (uninstantiated) module for the
        operator type by using a lookup table of available types.

        This is useful for dynamically creating instances of the
        appropriate operator class based on the operator type.

        Returns:
            Type[RelationwiseOperatorBase]: The class corresponding
            to the operator type.
        """

        # Lookup table for operator types to their corresponding classes
        available_operator_classes = {
            OperatorType.TRANSLATION: TranslationOperator,
            OperatorType.DIAGONAL: DiagonalOperator,
            OperatorType.COMPLEX_DIAGONAL: ComplexDiagonalOperator,
            OperatorType.LINEAR: LinearOperator,
            OperatorType.IDENTITY: IdentityOperator,
        }
        # Check if the operator type is in the lookup table
        if self in available_operator_classes:
            return available_operator_classes[self]
        # If not, raise an error
        else:
            raise ValueError(f"Unknown operator type: {self}")


class NegativeSamplingCorruptionType(str, Enum):
    """
    Enum for different types of corruption for negative sampling.
    """

    SRC = "SRC"
    DST = "DST"
    BOTH = "BOTH"

    def __str__(self):
        return self.value


class ModelPhase(str, Enum):
    """
    Enum for different phases of model use.

    This is used to differentiate operations the model should perform
    during training, validation, testing, and inference.  TorchRec
    models are fx-traced, so dynamic control flow with traced
    variables within the model is not supported.  These phases are
    used to establish static control flow.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    INFERENCE_SRC = "inference_src"
    INFERENCE_DST = "inference_dst"

    def __str__(self):
        return self.value
