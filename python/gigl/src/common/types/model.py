from enum import Enum
from typing import Optional, OrderedDict, Protocol, runtime_checkable

import torch

from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper


@runtime_checkable
class BaseModelOperationsProtocol(Protocol):
    @property
    def model(self) -> torch.nn.Module:
        ...

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        ...

    def init_model(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        state_dict: Optional[OrderedDict[str, torch.Tensor]] = None,
    ) -> torch.nn.Module:
        ...


class GraphBackend(str, Enum):
    PYG = "PyG"


class GnnModel(Protocol):
    """
    read-only property to infer graph-backend from a GNN model
    """

    @property
    def graph_backend(self) -> GraphBackend:
        ...
