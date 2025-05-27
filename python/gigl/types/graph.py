from collections import abc
from dataclasses import dataclass
from typing import Optional, TypeVar, Union, overload

import torch
from graphlearn_torch.partition import PartitionBook

from gigl.common.data.dataloaders import SerializedTFRecordInfo
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
        """
        Convert positive and negative labels to edges. Converts this object in-place to a "heterogeneous" representation.

        This function requires the following conditions and will throw if they are not met:
            1. The positive_label is not None

        """
        if self.positive_label is None:
            raise ValueError(
                "Cannot treat labels as edges when positive label is None."
            )

        edge_index_with_labels = to_heterogeneous_edge(self.edge_index)

        if len(edge_index_with_labels) == 1:
            main_edge_type = next(iter(edge_index_with_labels.keys()))
            logger.info(
                f"Basing positive and negative labels on edge types on edge type: {main_edge_type}."
            )
        else:
            main_edge_type = None

        if isinstance(self.positive_label, torch.Tensor):
            if main_edge_type is None:
                raise ValueError(
                    "Detected multiple edge types in provided edge_index, but no edge types specified for provided positive label."
                )
            positive_label_edge_type = message_passing_to_positive_label(main_edge_type)
            logger.info(
                f"Treating homogeneous positive labels as edge type {positive_label_edge_type}."
            )
            edge_index_with_labels[positive_label_edge_type] = self.positive_label
        elif isinstance(self.positive_label, dict):
            for (
                positive_label_type,
                positive_label_tensor,
            ) in self.positive_label.items():
                positive_label_edge_type = message_passing_to_positive_label(
                    positive_label_type
                )
                logger.info(
                    f"Treating heterogeneous positive labels {positive_label_type} as edge type {positive_label_edge_type}."
                )
                edge_index_with_labels[positive_label_edge_type] = positive_label_tensor

        if isinstance(self.negative_label, torch.Tensor):
            if main_edge_type is None:
                raise ValueError(
                    "Detected multiple edge types in provided edge_index, but no edge types specified for provided negative label."
                )
            negative_label_edge_type = message_passing_to_negative_label(main_edge_type)
            logger.info(
                f"Treating homogeneous negative labels as edge type {negative_label_edge_type}."
            )
            edge_index_with_labels[negative_label_edge_type] = self.negative_label
        elif isinstance(self.negative_label, dict):
            for (
                negative_label_type,
                negative_label_tensor,
            ) in self.negative_label.items():
                negative_label_edge_type = message_passing_to_negative_label(
                    negative_label_type
                )
                logger.info(
                    f"Treating heterogeneous negative labels {negative_label_type} as edge type {negative_label_edge_type}."
                )
                edge_index_with_labels[negative_label_edge_type] = negative_label_tensor

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


# Entities that represent a graph, somehow.
# Ideally, this would be anything, e.g. `_T = TypeVar("_T")`, but we need to be more specific.
# As if we type `to_homogeneous(x: _T | dict[NodeType, _T] | dict[EdgeType, _T]) -> _T`,
# then `_T` captures the "dict" types, and the output type is not correctly narrowed.
# e.g. `reveal_type(to_homogeneous(d: Tensor | dict[..., Tensor] | None]))` is `object`
# Instead, we enumerate these types, as MyPy does not allow "not" in a TypeVar.
# We should extend this as necessary, just make sure *never* add any Mapping types.
# NOTE: We have `Optional[SerializedTFRecordInfo]` in the type,
# As adding `None` and `SerializedTFRecordInfo` separately do not accomplish the equivalent thing.
# I believe this is due to the fact that the contraints on a `TypeVar` are not
# are not treated as a union of the types, but rather each as their own case.
_GraphEntity = TypeVar(
    "_GraphEntity",
    torch.Tensor,
    GraphPartitionData,
    FeaturePartitionData,
    SerializedTFRecordInfo,
    Optional[SerializedTFRecordInfo],
    list,
    # TODO(kmonte): Add GLT Partition book here
    # We cannot at the moment as we mypy ignore GLT
    # And adding it as a type here will break mypy.
    # PartitionBook
)


@overload
def to_heterogeneous_node(x: None) -> None:
    ...


@overload
def to_heterogeneous_node(
    x: Union[_GraphEntity, dict[NodeType, _GraphEntity]]
) -> dict[NodeType, _GraphEntity]:
    ...


def to_heterogeneous_node(
    x: Optional[Union[_GraphEntity, dict[NodeType, _GraphEntity]]]
) -> Optional[dict[NodeType, _GraphEntity]]:
    """Convert a value to a heterogeneous node representation.

    If the input is None, return None.
    If the input is a dictionary, return it as is.
    If the input is a single value, return it as a dictionary with the default homogeneous node type as the key.

    Args:
        x (Optional[Union[_GraphEntity, dict[NodeType, _GraphEntity]]]): The input value to convert.

    Returns:
        Optional[dict[NodeType, _GraphEntity]]: The converted heterogeneous node representation.
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
def to_heterogeneous_edge(
    x: Union[_GraphEntity, dict[EdgeType, _GraphEntity]]
) -> dict[EdgeType, _GraphEntity]:
    ...


def to_heterogeneous_edge(
    x: Optional[Union[_GraphEntity, dict[EdgeType, _GraphEntity]]]
) -> Optional[dict[EdgeType, _GraphEntity]]:
    """Convert a value to a heterogeneous edge representation.

    If the input is None, return None.
    If the input is a dictionary, return it as is.
    If the input is a single value, return it as a dictionary with the default homogeneous edge type as the key.

    Args:
        x (Optional[Union[_GraphEntity, dict[EdgeType, _GraphEntity]]]): The input value to convert.

    Returns:
        Optional[dict[EdgeType, _GraphEntity]]: The converted heterogeneous edge representation.
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
def to_homogeneous(x: abc.Mapping[NodeType, _GraphEntity]) -> _GraphEntity:
    ...


@overload
def to_homogeneous(x: abc.Mapping[EdgeType, _GraphEntity]) -> _GraphEntity:
    ...


@overload
def to_homogeneous(x: _GraphEntity) -> _GraphEntity:
    ...


def to_homogeneous(
    x: Optional[
        Union[
            _GraphEntity,
            abc.Mapping[NodeType, _GraphEntity],
            abc.Mapping[EdgeType, _GraphEntity],
        ]
    ]
) -> Optional[_GraphEntity]:
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
    if isinstance(x, abc.Mapping):
        if len(x) != 1:
            raise ValueError(
                f"Expected a single value in the dictionary, but got multiple keys: {x.keys()}"
            )
        n = next(iter(x.values()))
        return n
    return x
