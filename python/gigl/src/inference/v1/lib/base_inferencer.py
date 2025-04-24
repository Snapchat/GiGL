from dataclasses import dataclass
from functools import wraps
from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

import torch
import torch.utils.data

from gigl.common.logger import Logger
from gigl.common.utils.torch_training import is_distributed_available_and_initialized
from gigl.src.common.types.model import BaseModelOperationsProtocol
from gigl.src.training.v1.lib.data_loaders.rooted_node_neighborhood_data_loader import (
    RootedNodeNeighborhoodBatch,
)
from gigl.src.training.v1.lib.data_loaders.supervised_node_classification_data_loader import (
    SupervisedNodeClassificationBatch,
)

T = TypeVar("T", contravariant=True)

logger = Logger()


@dataclass
class InferBatchResults:
    embeddings: Optional[torch.Tensor]
    predictions: Optional[torch.Tensor]


def no_grad_eval(f):
    @wraps(f)
    def wrapper(self: BaseInferencer, *args, **kwargs):
        curr_model = self.model
        if is_distributed_available_and_initialized() and isinstance(
            self.model, torch.nn.parallel.DistributedDataParallel
        ):
            # We don't need to make use of DDPs unecessary synchronization here
            self.model = self.model.module

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            ret_val = f(self, *args, **kwargs)  # Call infer_batch
        self.model.train(
            mode=was_training
        )  # reset the model to whether it was training or not
        self.model = curr_model
        return ret_val

    return wrapper


@runtime_checkable
class BaseInferencer(BaseModelOperationsProtocol, Protocol, Generic[T]):
    def infer_batch(
        self, batch: T, device: torch.device = torch.device("cpu")
    ) -> InferBatchResults:
        raise NotImplementedError


class SupervisedNodeClassificationBaseInferencer(
    BaseInferencer[SupervisedNodeClassificationBatch]
):
    pass


class NodeAnchorBasedLinkPredictionBaseInferencer(
    BaseInferencer[RootedNodeNeighborhoodBatch]
):
    pass
