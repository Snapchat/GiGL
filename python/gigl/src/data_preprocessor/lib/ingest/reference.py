from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType
from gigl.src.data_preprocessor.lib.types import InstanceDictPTransform

# Type hints for abstract dataclasses are currently not supported. https://github.com/python/mypy/issues/5374


@dataclass(frozen=True)  # type: ignore
class DataReference(ABC):
    """
    Contains a URI string to the data reference, and provides a means of yielding
    instance dicts via a beam PTransform.

    A single DataReference is currently assumed to have data relevant to a *single* node or edge type.
    A single DataReference *cannot* currently house mixed-type data.
    """

    reference_uri: str

    @abstractmethod
    def yield_instance_dict_ptransform(self, *args, **kwargs) -> InstanceDictPTransform:
        """
        Returns a PTransform whose expand method returns a PCollection of InstanceDicts, which can be subsequently
        ingested and transformed via Tensorflow Transform.

        TODO: extend to support multiple edge types being in the same table.
        :param args:
        :param kwargs:
        :return:
        """

        raise NotImplementedError


@dataclass(frozen=True)  # type: ignore
class NodeDataReference(DataReference, ABC):
    """
    DataReference which stores node data.
    """

    node_type: NodeType
    identifier: Optional[str] = None

    def __repr__(self) -> str:
        return f"NodeDataReference(node_type={self.node_type}, identifier={self.identifier}, reference_uri={self.reference_uri})"


@dataclass(frozen=True)  # type: ignore
class EdgeDataReference(DataReference, ABC):
    """
    DataReference which stores edge data
    """

    edge_type: EdgeType
    edge_usage_type: EdgeUsageType = EdgeUsageType.MAIN
    src_identifier: Optional[str] = None
    dst_identifier: Optional[str] = None

    def __repr__(self) -> str:
        return f"EdgeDataReference(edge_type={self.edge_type}, src_identifier={self.src_identifier}, dst_identifier={self.dst_identifier}, reference_uri={self.reference_uri})"
