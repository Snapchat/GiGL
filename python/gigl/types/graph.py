from collections import abc
from dataclasses import dataclass
from typing import Optional, TypeVar, Union, overload

import torch
from graphlearn_torch.partition import PartitionBook

from gigl.common.logger import Logger

# TODO(kmonte) - we should move gigl.src.common.types.graph_data to this file.
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation

logger = Logger()

DEFAULT_HOMOGENEOUS_NODE_TYPE = NodeType("default_homogeneous_node_type")
DEFAULT_HOMOGENEOUS_EDGE_TYPE = EdgeType(
    src_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
    relation=Relation("to"),
    dst_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
)

_POSITIVE_LABEL_TAG = "gigl_positive"
_NEGATIVE_LABEL_TAG = "gigl_negative"

# We really should support PyG EdgeType natively but since we type ignore it that's not ideal atm...
# We can use this TypeVar to try and stem the bleeding (hopefully).
_EdgeType = TypeVar("_EdgeType", EdgeType, tuple[str, str, str])


# TODO(kmonte, mkolodner): Move SerializedGraphMetadata and maybe convert_pb_to_serialized_graph_metadata here.


@dataclass(frozen=True)
class FeaturePartitionData:
    """Data and indexing info of a node/edge feature partition."""

    # node/edge feature tensor
    feats: torch.Tensor
    # node/edge ids tensor corresponding to `feats`. This is Optional since we do not need this field for range-based partitioning
    ids: Optional[torch.Tensor]


@dataclass(frozen=True)
class GraphPartitionData:
    """Data and indexing info of a graph partition."""

    # edge index (rows, cols)
    edge_index: torch.Tensor
    # edge ids tensor corresponding to `edge_index`
    edge_ids: torch.Tensor
    # weights tensor corresponding to `edge_index`
    weights: Optional[torch.Tensor] = None


# This dataclass should not be frozen, as we are expected to delete partition outputs once they have been registered inside of GLT DistDataset
# in order to save memory.
@dataclass
class PartitionOutput:
    # Node partition book
    node_partition_book: Union[PartitionBook, dict[NodeType, PartitionBook]]

    # Edge partition book
    edge_partition_book: Union[PartitionBook, dict[EdgeType, PartitionBook]]

    # Partitioned edge index on current rank. This field will always be populated after partitioning. However, we may set this
    # field to None during dataset.build() in order to minimize the peak memory usage, and as a result type this as Optional.
    partitioned_edge_index: Optional[
        Union[GraphPartitionData, dict[EdgeType, GraphPartitionData]]
    ]

    # Node features on current rank, May be None if node features are not partitioned
    partitioned_node_features: Optional[
        Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]
    ]

    # Edge features on current rank, May be None if edge features are not partitioned
    partitioned_edge_features: Optional[
        Union[FeaturePartitionData, dict[EdgeType, FeaturePartitionData]]
    ]

    # Positive edge indices on current rank, May be None if positive edge labels are not partitioned
    partitioned_positive_labels: Optional[
        Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    ]

    # Negative edge indices on current rank, May be None if negative edge labels are not partitioned
    partitioned_negative_labels: Optional[
        Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    ]


# This dataclass should not be frozen, as we are expected to delete its members once they have been registered inside of the partitioner
# in order to save memory.
@dataclass
class LoadedGraphTensors:
    # Unpartitioned Node Ids
    node_ids: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    # Unpartitioned Node Features
    node_features: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]
    # Unpartitioned Edge Index
    edge_index: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    # Unpartitioned Edge Features
    edge_features: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]
    # Unpartitioned Positive Edge Label
    positive_label: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]
    # Unpartitioned Negative Edge Label
    negative_label: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]

    def treat_labels_as_edges(self) -> None:
        """Convert positive and negative labels to edges. Converts this object in-place to a "heterogeneous" representation.

        This requires the following conditions and will throw if they are not met:
            1. The node_ids, node_features, edge_index, and edge_features are not dictionaries (we loaded a homogeneous graph).
            2. The positive_label and negative_label are not None and are Tensors, not dictionaries.
        """
        # TODO(kmonte): We should support heterogeneous graphs in the future.
        if (
            isinstance(self.node_ids, abc.Mapping)
            or isinstance(self.node_features, abc.Mapping)
            or isinstance(self.edge_index, abc.Mapping)
            or isinstance(self.edge_features, abc.Mapping)
            or isinstance(self.positive_label, abc.Mapping)
            or isinstance(self.negative_label, abc.Mapping)
        ):
            raise ValueError(
                "Cannot treat labels as edges when using heterogeneous graph tensors."
            )
        if self.positive_label is None or self.negative_label is None:
            raise ValueError(
                "Cannot treat labels as edges when positive or negative labels are None."
            )

        edge_index_with_labels = to_heterogeneous_edge(self.edge_index)
        main_edge_type = next(iter(edge_index_with_labels.keys()))
        logger.info(
            f"Basing positive and negative labels on edge types on main label edge type: {main_edge_type}."
        )
        positive_label_edge_type = message_passing_to_positive_label(main_edge_type)
        edge_index_with_labels[positive_label_edge_type] = self.positive_label
        negative_label_edge_type = message_passing_to_negative_label(main_edge_type)
        edge_index_with_labels[negative_label_edge_type] = self.negative_label
        logger.info(
            f"Treating positive labels as edge type {positive_label_edge_type} and negative labels as edge type {negative_label_edge_type}."
        )

        self.node_ids = to_heterogeneous_node(self.node_ids)
        self.node_features = to_heterogeneous_node(self.node_features)
        self.edge_index = edge_index_with_labels
        self.edge_features = to_heterogeneous_edge(self.edge_features)
        self.positive_label = None
        self.negative_label = None


def message_passing_to_positive_label(
    message_passing_edge_type: _EdgeType,
) -> _EdgeType:
    """Convert a message passing edge type to a positive label edge type.

    Args:
        message_passing_edge_type (EdgeType): The message passing edge type.

    Returns:
        EdgeType: The positive label edge type.
    """
    edge_type = (
        str(message_passing_edge_type[0]),
        f"{message_passing_edge_type[1]}_{_POSITIVE_LABEL_TAG}",
        str(message_passing_edge_type[2]),
    )
    if isinstance(message_passing_edge_type, EdgeType):
        return EdgeType(
            NodeType(edge_type[0]), Relation(edge_type[1]), NodeType(edge_type[2])
        )
    else:
        return edge_type


def message_passing_to_negative_label(
    message_passing_edge_type: _EdgeType,
) -> _EdgeType:
    """Convert a message passing edge type to a negative label edge type.

    Args:
        message_passing_edge_type (EdgeType): The message passing edge type.

    Returns:
        EdgeType: The negative label edge type.
    """
    edge_type = (
        str(message_passing_edge_type[0]),
        f"{message_passing_edge_type[1]}_{_NEGATIVE_LABEL_TAG}",
        str(message_passing_edge_type[2]),
    )
    if isinstance(message_passing_edge_type, EdgeType):
        return EdgeType(
            NodeType(edge_type[0]), Relation(edge_type[1]), NodeType(edge_type[2])
        )
    else:
        return edge_type


def select_label_edge_types(
    message_passing_edge_type: _EdgeType, edge_entities: abc.Iterable[_EdgeType]
) -> tuple[_EdgeType, Optional[_EdgeType]]:
    """Select label edge types for a given message passing edge type.

    Args:
        message_passing_edge_type (EdgeType): The message passing edge type.
        edge_entities (abc.Iterable[EdgeType]): The edge entities to select from.

    Returns:
        tuple[EdgeType, Optional[EdgeType]]: A tuple containing the positive label edge type and optionally the negative label edge type.
    """
    positive_label_type = None
    negative_label_type = None
    for edge_type in edge_entities:
        if message_passing_to_positive_label(message_passing_edge_type) == edge_type:
            positive_label_type = edge_type
        if message_passing_to_negative_label(message_passing_edge_type) == edge_type:
            negative_label_type = edge_type
    if positive_label_type is None:
        raise ValueError(
            f"Could not find positive label edge type for message passing edge type {message_passing_edge_type} from edge entities {edge_entities}."
        )
    return positive_label_type, negative_label_type


_T = TypeVar("_T")


@overload
def to_heterogeneous_node(x: None) -> None:
    ...


@overload
def to_heterogeneous_node(x: Union[_T, dict[NodeType, _T]]) -> dict[NodeType, _T]:
    ...


def to_heterogeneous_node(
    x: Optional[Union[_T, dict[NodeType, _T]]]
) -> Optional[dict[NodeType, _T]]:
    """Convert a value to a heterogeneous node representation.

    If the input is None, return None.
    If the input is a dictionary, return it as is.
    If the input is a single value, return it as a dictionary with the default homogeneous node type as the key.

    Args:
        x (Optional[Union[_T, dict[NodeType, _T]]]): The input value to convert.

    Returns:
        Optional[dict[NodeType, _T]]: The converted heterogeneous node representation.
    """
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    return {DEFAULT_HOMOGENEOUS_NODE_TYPE: x}


@overload
def to_heterogeneous_edge(x: None) -> None:
    ...


@overload
def to_heterogeneous_edge(x: Union[_T, dict[EdgeType, _T]]) -> dict[EdgeType, _T]:
    ...


def to_heterogeneous_edge(
    x: Optional[Union[_T, dict[EdgeType, _T]]]
) -> Optional[dict[EdgeType, _T]]:
    """Convert a value to a heterogeneous edge representation.

    If the input is None, return None.
    If the input is a dictionary, return it as is.
    If the input is a single value, return it as a dictionary with the default homogeneous edge type as the key.

    Args:
        x (Optional[Union[_T, dict[EdgeType, _T]]]): The input value to convert.

    Returns:
        Optional[dict[EdgeType, _T]]: The converted heterogeneous edge representation.
    """
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    return {DEFAULT_HOMOGENEOUS_EDGE_TYPE: x}


@overload
def to_homogeneous(x: None) -> None:
    ...


@overload
def to_homogeneous(x: Union[_T, dict[Union[NodeType, EdgeType], _T]]) -> _T:
    ...


def to_homogeneous(
    x: Optional[Union[_T, dict[Union[NodeType, EdgeType], _T]]]
) -> Optional[_T]:
    """Convert a value to a homogeneous representation.

    If the input is None, return None.
    If the input is a dictionary, return the single value in the dictionary.
    If the input is a single value, return it as is.

    Args:
        x (Optional[Union[_T, dict[Union[NodeType, EdgeType], _T]]]): The input value to convert.

    Returns:
        Optional[_T]: The converted homogeneous representation.
    """
    if x is None:
        return None
    if isinstance(x, dict):
        if len(x) != 1:
            raise ValueError(
                f"Expected a single value in the dictionary, but got multiple keys: {x.keys()}"
            )
        n = next(iter(x.values()))
        return n
    return x
