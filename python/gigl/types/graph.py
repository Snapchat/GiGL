import gc
from collections import abc
from dataclasses import dataclass
from typing import Literal, Optional, TypeVar, Union, overload

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
    # edge ids tensor corresponding to `edge_index`. This should only be `None` if there are no edge features to partition, removing the need for storing edge ids.
    edge_ids: Optional[torch.Tensor]
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
    partitioned_positive_edge_labels: Optional[
        Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    ]

    # Negative edge indices on current rank, May be None if negative edge labels are not partitioned
    partitioned_negative_edge_labels: Optional[
        Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    ]

    # Partitioned node labels, May be None if node labels are not partitioned.
    # In practice, we require the IDS of the partitioned node labels field to be equal to the ids of the partitioned node features field, if it exists.
    # This is because the partitioned node labels should be partitioned along with the node features so that we don't need to track two separate node ID stores,
    # which saves a lot of memory.
    partitioned_node_labels: Optional[
        Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]
    ]


@dataclass(frozen=True)
class FeatureInfo:
    """Data class containing information about the feature dimension and feature data type for a particular feature"""

    dim: int
    dtype: torch.dtype


def _get_label_edges(
    labeled_edge_index: torch.Tensor,
    edge_dir: Literal["in", "out"],
    labeled_edge_type: EdgeType,
) -> tuple[EdgeType, torch.Tensor]:
    """
    If edge direction is `out`, return the provided edge type and edge index. Otherwise, reverse the edge type and flip the edge index rows
    so that the labeled edge index may be the same direction as the rest of the edges.
    Args:
        labeled_edge_index (torch.Tensor): Edge index containing positive or negative labels for supervision
        edge_dir (Literal["in", "out"]): Direction of edges in the graph
        labeled_edge_type (EdgeType): Edge type used for the positive or negative labeled edges
    Returns:
        EdgeType: Labeled edge type, which has been reversed if edge_dir = "in"
        torch.Tensor: Labeled edge index, which has its rows flipped if edge_dir = "in"
    """
    if edge_dir == "in":
        rev_edge_type = reverse_edge_type(labeled_edge_type)
        rev_labeled_edge_index = labeled_edge_index.flip(0)
        return rev_edge_type, rev_labeled_edge_index
    else:
        return labeled_edge_type, labeled_edge_index


# This dataclass should not be frozen, as we are expected to delete its members once they have been registered inside of the partitioner
# in order to save memory.
@dataclass
class LoadedGraphTensors:
    # Unpartitioned Node Ids
    node_ids: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    # Unpartitioned Node Features
    node_features: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]
    # Unpartitioned Node Labels
    node_labels: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]
    # Unpartitioned Edge Index
    edge_index: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    # Unpartitioned Edge Features
    edge_features: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]
    # Unpartitioned Positive Edge Label
    positive_label: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]
    # Unpartitioned Negative Edge Label
    negative_label: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]

    def treat_labels_as_edges(self, edge_dir: Literal["in", "out"]) -> None:
        """
        Convert positive and negative labels to edges. Converts this object in-place to a "heterogeneous" representation.
        If the edge direction is "in", we must reverse the supervision edge type. This is because we assume that provided labels are directed
        outwards in form (`anchor_node_type`, `relation`, `supervision_node_type`), and all edges in the edge index must be in the same direction.

        This function requires the following conditions and will throw if they are not met:
            1. The positive_label is not None

        Args:
            edge_dir: The edge direction of the graph.

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
            labeled_edge_type, edge_index = _get_label_edges(
                labeled_edge_index=self.positive_label,
                edge_dir=edge_dir,
                labeled_edge_type=positive_label_edge_type,
            )
            edge_index_with_labels[labeled_edge_type] = edge_index
            logger.info(
                f"Treating homogeneous positive labels as edge type {positive_label_edge_type}."
            )

        elif isinstance(self.positive_label, dict):
            for (
                positive_label_type,
                positive_label_tensor,
            ) in self.positive_label.items():
                positive_label_edge_type = message_passing_to_positive_label(
                    positive_label_type
                )
                labeled_edge_type, edge_index = _get_label_edges(
                    labeled_edge_index=positive_label_tensor,
                    edge_dir=edge_dir,
                    labeled_edge_type=positive_label_edge_type,
                )
                edge_index_with_labels[labeled_edge_type] = edge_index
                logger.info(
                    f"Treating heterogeneous positive labels {positive_label_type} as edge type {positive_label_edge_type}."
                )

        if isinstance(self.negative_label, torch.Tensor):
            if main_edge_type is None:
                raise ValueError(
                    "Detected multiple edge types in provided edge_index, but no edge types specified for provided negative label."
                )
            negative_label_edge_type = message_passing_to_negative_label(main_edge_type)
            labeled_edge_type, edge_index = _get_label_edges(
                labeled_edge_index=self.negative_label,
                edge_dir=edge_dir,
                labeled_edge_type=negative_label_edge_type,
            )
            edge_index_with_labels[labeled_edge_type] = edge_index
            logger.info(
                f"Treating homogeneous negative labels as edge type {negative_label_edge_type}."
            )
        elif isinstance(self.negative_label, dict):
            for (
                negative_label_type,
                negative_label_tensor,
            ) in self.negative_label.items():
                negative_label_edge_type = message_passing_to_negative_label(
                    negative_label_type
                )
                labeled_edge_type, edge_index = _get_label_edges(
                    labeled_edge_index=negative_label_tensor,
                    edge_dir=edge_dir,
                    labeled_edge_type=negative_label_edge_type,
                )
                edge_index_with_labels[labeled_edge_type] = edge_index
                logger.info(
                    f"Treating heterogeneous negative labels {negative_label_type} as edge type {negative_label_edge_type}."
                )

        self.node_ids = to_heterogeneous_node(self.node_ids)
        self.node_features = to_heterogeneous_node(self.node_features)
        self.edge_index = edge_index_with_labels
        self.edge_features = to_heterogeneous_edge(self.edge_features)
        self.positive_label = None
        self.negative_label = None
        gc.collect()


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


def is_label_edge_type(
    edge_type: _EdgeType,
) -> bool:
    """Check if an edge type is a label edge type.

    Args:
        edge_type (EdgeType): The edge type to check.

    Returns:
        bool: True if the edge type is a label edge type, False otherwise.
    """
    return _POSITIVE_LABEL_TAG in str(edge_type[1]) or _NEGATIVE_LABEL_TAG in str(
        edge_type[1]
    )


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
            f"Could not find positive label edge type for message passing edge type {message_passing_to_positive_label(message_passing_edge_type)} from edge entities {edge_entities}."
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
    str,
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


def reverse_edge_type(edge_type: _EdgeType) -> _EdgeType:
    """
    Reverses the source and destination node types of the provided edge type
    Args:
        edge_type (EdgeType): The target edge to have its source and destinated node types reversed
    Returns:
        EdgeType: The reversed edge type
    """
    if isinstance(edge_type, EdgeType):
        return EdgeType(edge_type[2], edge_type[1], edge_type[0])
    else:
        return (edge_type[2], edge_type[1], edge_type[0])
