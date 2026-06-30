"""Utils for Neighbor loaders."""

import ast
from collections import abc
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, TypeVar, Union

import torch
from graphlearn_torch.channel import SampleMessage
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType, NodeType

from gigl.common.logger import Logger
from gigl.common.utils.feature_quantization import dequantize_feature_tensor
from gigl.distributed.sampler import NODE_QUANTIZED_FEATURES_METADATA_KEY
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    FeatureInfo,
    FeatureQuantizationMetadata,
    is_label_edge_type,
)

logger = Logger()

_GraphType = TypeVar("_GraphType", Data, HeteroData)


class SamplingClusterSetup(Enum):
    """
    The setup of the sampling cluster.
    """

    COLOCATED = "colocated"
    GRAPH_STORE = "graph_store"


@dataclass(frozen=True)
class DatasetSchema:
    """
    Shared metadata between the local and remote datasets.
    """

    # If the dataset is homogeneous with labeled edge type. E.g. one node type, one edge type, and "label" edges.
    # This happens in an otherwise homogeneous dataset when doing ABLP and when we split the dataset.
    is_homogeneous_with_labeled_edge_type: bool
    # List of all edge types in the graph.
    edge_types: Optional[list[EdgeType]]
    # Node feature info.
    node_feature_info: Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]]
    # Packed uint8 node feature info.
    node_quantized_feature_info: Optional[
        Union[FeatureInfo, dict[NodeType, FeatureInfo]]
    ]
    # Quantization metadata for append-only packed node features.
    node_quantization_metadata: Optional[
        Union[
            FeatureQuantizationMetadata, dict[NodeType, FeatureQuantizationMetadata]
        ]
    ]
    # Edge feature info.
    edge_feature_info: Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]]
    # Edge direction.
    edge_dir: Union[str, Literal["in", "out"]]


def patch_fanout_for_sampling(
    edge_types: Optional[list[EdgeType]],
    num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
) -> Union[list[int], dict[EdgeType, list[int]]]:
    """
    Normalizes the user-provided fanout into the per-edge-type form the samplers expect.

    For heterogeneous datasets, broadcasts a single fanout list to every message-passing
    edge type, or validates a per-edge-type dict.

    Label edge types (positive/negative supervision edges injected by ABLP) are never
    sampled -- the samplers skip them during traversal (see
    ``DistNeighborSampler._sample_from_nodes`` and ``DistPPRNeighborSampler``), so they
    take no fanout. Callers must not specify them: any label edge type in ``num_neighbors``
    raises ``ValueError``.

    For homogeneous datasets (``edge_types`` is None) the fanout list is returned unchanged.

    Args:
        edge_types (Optional[list[EdgeType]]): List of all edge types in the graph, is None for homogeneous datasets
        num_neighbors (Union[list[int], dict[EdgeType, list[int]]]): Specified fanout by the user
    Returns:
        Union[list[int], dict[EdgeType, list[int]]]: Normalized fanout. A list[int] for
            homogeneous datasets, otherwise a dict[EdgeType, list[int]] over message-passing edges.
    Raises:
        ValueError: If a label edge type is supplied in ``num_neighbors``, if the dataset has
            no message-passing edge types to fan out around, or on malformed fanout (extra or
            missing edge types, inconsistent hop counts, or negative fanout).
    """
    if edge_types is None:
        if isinstance(num_neighbors, abc.Mapping):
            raise ValueError(
                "When dataset is homogeneous, the num_neighbors field cannot be a dictionary."
            )
        if not all(hop >= 0 for hop in num_neighbors):
            raise ValueError(f"Hops provided must be non-negative, got {num_neighbors}")
        return num_neighbors

    message_passing_edge_types = [
        edge_type for edge_type in edge_types if not is_label_edge_type(edge_type)
    ]
    # A graph with only label edges and no message-passing edges cannot be sampled;
    # fail explicitly rather than letting the validation tail raise StopIteration on an
    # empty dict.
    if not message_passing_edge_types:
        raise ValueError(
            f"No message-passing edge types found in dataset edge types {edge_types}; "
            "cannot construct a fanout."
        )

    if isinstance(num_neighbors, list):
        original_fanout = num_neighbors
        num_neighbors = {
            edge_type: original_fanout for edge_type in message_passing_edge_types
        }
    else:
        # Label (positive/negative supervision) edges are injected internally and are
        # never sampled, so callers must not specify them in the fanout. Reject them
        # explicitly: they are part of the dataset edge types, so the extra-edge-types
        # check below would not catch them.
        provided_label_edge_types = {
            edge_type for edge_type in num_neighbors if is_label_edge_type(edge_type)
        }
        if provided_label_edge_types:
            raise ValueError(
                f"Label edge types {provided_label_edge_types} were provided in num_neighbors. "
                "Label (positive/negative supervision) edges are injected internally and are never "
                "sampled, so they must not be specified in the fanout."
            )
        extra_edge_types = set(num_neighbors.keys()) - set(edge_types)
        if extra_edge_types:
            raise ValueError(
                f"Found extra edge types {extra_edge_types} in fanout which is not in dataset edge types {edge_types}."
            )
        num_neighbors = deepcopy(num_neighbors)
        missing_edge_types = set(message_passing_edge_types) - set(num_neighbors.keys())
        if missing_edge_types:
            raise ValueError(
                f"Found non-labeled edge type(s) {missing_edge_types} in the dataset which are not in "
                f"the provided fanout {set(num_neighbors.keys())}. If fanout is provided as a dict, "
                "all message-passing edges must be present."
            )

    hops = len(next(iter(num_neighbors.values())))
    if not all(len(fanout) == hops for fanout in num_neighbors.values()):
        raise ValueError(
            f"num_neighbors must be a dict of edge types with the same number of hops. Received: {num_neighbors}"
        )
    if not all(
        hop >= 0 for edge_type in num_neighbors for hop in num_neighbors[edge_type]
    ):
        raise ValueError(f"Hops provided must be non-negative, got {num_neighbors}")

    logger.info(f"Overwrote num_neighbors to: {num_neighbors}.")
    return num_neighbors


def shard_nodes_by_process(
    input_nodes: torch.Tensor,
    local_process_rank: int,
    local_process_world_size: int,
) -> torch.Tensor:
    """
    Shards input nodes based on the local process rank
    Args:
        input_nodes (torch.Tensor): Nodes which are split across each training or inference process
        local_process_rank (int): Rank of the current local process
        local_process_world_size (int): Total number of local processes on the current machine
    Returns:
        torch.Tensor: The sharded nodes for the current local process
    """
    num_node_ids_per_process = input_nodes.size(0) // local_process_world_size
    start_index = local_process_rank * num_node_ids_per_process
    end_index = (
        input_nodes.size(0)
        if local_process_rank == local_process_world_size - 1
        else start_index + num_node_ids_per_process
    )
    nodes_for_current_process = input_nodes[start_index:end_index]
    return nodes_for_current_process


def labeled_to_homogeneous(supervision_edge_type: EdgeType, data: HeteroData) -> Data:
    """
    Returns a Data object with the label edges removed.

    Args:
        supervision_edge_type (EdgeType): The edge type that contains the supervision edges.
        data (HeteroData): Heterogeneous graph with the supervision edge type
    Returns:
        data (Data): Homogeneous graph with the labeled edge type removed
    """
    homogeneous_data = data.edge_type_subgraph([supervision_edge_type]).to_homogeneous(
        add_edge_type=False, add_node_type=False
    )
    # Since this is "homogeneous", supervision_edge_type[0] and supervision_edge_type[2] are the same.
    sample_node_type = supervision_edge_type[0]
    homogeneous_data.num_sampled_nodes = data.num_sampled_nodes[sample_node_type]
    homogeneous_data.num_sampled_edges = data.num_sampled_edges[supervision_edge_type]
    homogeneous_data.batch_size = homogeneous_data.batch.numel()
    return homogeneous_data


def strip_non_ppr_edge_types(
    data: HeteroData, ppr_edge_types: set[EdgeType]
) -> HeteroData:
    """Remove all edge types not in ``ppr_edge_types`` from a HeteroData object.

    GLT's collate function creates edge stores for all edge types registered in
    the sampler (including original graph and reverse edge types) even when the
    PPR sampler provides empty row/col tensors.  This removes those ghost stores
    so the output contains only PPR edge types.

    Modifies the input in place.

    Args:
        data: The HeteroData object to clean up.
        ppr_edge_types: The exact set of PPR edge types to keep, as returned
            by ``attach_ppr_outputs``.

    Returns:
        The same object with non-PPR edge types removed.
    """
    for edge_type in list(data.edge_types):
        if edge_type not in ppr_edge_types:
            del data[edge_type]
            # num_sampled_edges is set by GLT's standard k-hop sampler but not
            # by PPR sampling, which constructs HeteroData manually.  Guard with
            # hasattr rather than assuming it's always present.
            if hasattr(data, "num_sampled_edges"):
                data.num_sampled_edges.pop(edge_type, None)
    return data


def strip_label_edges(data: HeteroData) -> HeteroData:
    """
    Removes all edges of a specific type from a heterogeneous graph.

    Modifies the input in place.

    Args:
        data (HeteroData): The input heterogeneous graph.

    Returns:
        HeteroData: The graph with the label edge types removed.
    """

    label_edge_types = [
        e_type for e_type in data.edge_types if is_label_edge_type(e_type)
    ]
    for edge_type in label_edge_types:
        del data[edge_type]
        del data.num_sampled_edges[edge_type]

    return data


def set_missing_features(
    data: _GraphType,
    node_feature_info: Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]],
    edge_feature_info: Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]],
    device: torch.device,
) -> _GraphType:
    """
    If a feature is missing from a produced Data or HeteroData object due to not fanning out to it, populates it in-place with an empty tensor
    with the appropriate feature dim.
    Note that PyG natively does this with their DistNeighborLoader for missing edge features + edge indices and missing node features:
    https://pytorch-geometric.readthedocs.io/en/2.4.0/_modules/torch_geometric/sampler/neighbor_sampler.html#NeighborSampler

    However, native Graphlearn-for-PyTorch only does this for edge indices:
    https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/sampler/base.py#L294-L301

    so we should do this our sampled node/edge features as well

    # TODO (mkolodner-sc): Migrate this utility to GLT once we fork their repo

    Args:
        data (_GraphType): Data or HeteroData object which we are setting the missing features for
        node_feature_info (Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]]): Node feature dimension and data type.
            Note that if heterogeneous, only node types with features should be provided. Can be None in the homogeneous case if there are no node features
        edge_feature_info (Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]]): Edge feature dimension and data type.
            Note that if heterogeneous, only edge types with features should be provided. Can be None in the homogeneous case if there are no edge features
        device (torch.device): Device to move the empty features to
    Returns:
        _GraphType: Data or HeteroData type with the updated feature fields
    """
    if isinstance(data, Data):
        if isinstance(node_feature_info, dict):
            raise ValueError(
                f"Expected node feature dimension to be a FeatureInfo or None for homogeneous data, got {node_feature_info} of type {type(node_feature_info)}"
            )
        if isinstance(edge_feature_info, dict):
            raise ValueError(
                f"Expected edge feature dimension to be an int or None for homogeneous data, got {edge_feature_info} of type {type(edge_feature_info)}"
            )
        # For homogeneous case, the Data object will always have the x or edge_attr fields -- we should check if it is None to see if it set
        if node_feature_info and data.x is None:
            data.x = torch.empty(
                (0, node_feature_info.dim), dtype=node_feature_info.dtype, device=device
            )
        if edge_feature_info and data.edge_attr is None:
            data.edge_attr = torch.empty(
                (0, edge_feature_info.dim), dtype=edge_feature_info.dtype, device=device
            )

    elif isinstance(data, HeteroData):
        if isinstance(node_feature_info, FeatureInfo):
            raise ValueError(
                f"Expected node feature dimension to be an dict or None for heterogeneous data, got {node_feature_info} of type {type(node_feature_info)}"
            )
        if isinstance(edge_feature_info, FeatureInfo):
            raise ValueError(
                f"Expected edge feature dimension to be an dict or None for heterogeneous data, got {edge_feature_info} of type {type(edge_feature_info)}"
            )
        # For heterogeneous case, the HeteroData object will never have the x or edge attr for a given entity type if doesn't exist, even if we set it to None,
        # thus we should check if it hasattr to see they are present
        if node_feature_info:
            for node_type, feature_info in node_feature_info.items():
                if not hasattr(data[node_type], "x"):
                    data[node_type].x = torch.empty(
                        (0, feature_info.dim), dtype=feature_info.dtype, device=device
                    )
        if edge_feature_info:
            for edge_type, feature_info in edge_feature_info.items():
                if not hasattr(data[edge_type], "edge_attr"):
                    data[edge_type].edge_attr = torch.empty(
                        (0, feature_info.dim), dtype=feature_info.dtype, device=device
                    )
    else:
        raise ValueError(
            f"Expected provided data object to be of type `Data` or `HeteroData`, got {type(data)}"
        )

    return data


def materialize_quantized_node_features(
    data: _GraphType,
    metadata: dict[str, torch.Tensor],
    node_quantization_metadata: Optional[
        Union[
            FeatureQuantizationMetadata, dict[NodeType, FeatureQuantizationMetadata]
        ]
    ],
) -> tuple[_GraphType, dict[str, torch.Tensor]]:
    """Materialize packed quantized node features into PyG node feature tensors."""
    if node_quantization_metadata is None:
        return data, metadata

    def materialize_node_store(
        node_store,
        packed_features: torch.Tensor,
        quantization_metadata: FeatureQuantizationMetadata,
    ) -> None:
        dequantized_features = dequantize_feature_tensor(
            packed_features=packed_features,
            quantization_metadata=quantization_metadata,
        )
        node_x = getattr(node_store, "x", None)
        if node_x is None:
            node_store.x = dequantized_features
            return
        if node_x.size(0) != dequantized_features.size(0):
            raise ValueError(
                "Cannot materialize quantized features with "
                f"{dequantized_features.size(0)} rows into existing x with "
                f"{node_x.size(0)} rows."
            )
        node_store.x = torch.cat([node_x, dequantized_features], dim=1)

    if isinstance(data, Data):
        if isinstance(node_quantization_metadata, dict):
            quantization_metadata = node_quantization_metadata.get(
                DEFAULT_HOMOGENEOUS_NODE_TYPE
            )
            metadata_key = (
                f"{NODE_QUANTIZED_FEATURES_METADATA_KEY}."
                f"{DEFAULT_HOMOGENEOUS_NODE_TYPE}"
            )
        else:
            quantization_metadata = node_quantization_metadata
            metadata_key = NODE_QUANTIZED_FEATURES_METADATA_KEY
        if quantization_metadata is None:
            return data, metadata
        packed_features = metadata.pop(metadata_key, None)
        if packed_features is None:
            packed_features = metadata.pop(NODE_QUANTIZED_FEATURES_METADATA_KEY, None)
        if packed_features is None:
            return data, metadata
        materialize_node_store(data, packed_features, quantization_metadata)
        return data, metadata

    if isinstance(node_quantization_metadata, FeatureQuantizationMetadata):
        raise ValueError(
            "Expected per-node-type quantization metadata for heterogeneous data."
        )

    for node_type, quantization_metadata in node_quantization_metadata.items():
        metadata_key = f"{NODE_QUANTIZED_FEATURES_METADATA_KEY}.{node_type}"
        packed_features = metadata.pop(metadata_key, None)
        if packed_features is None:
            continue
        materialize_node_store(data[node_type], packed_features, quantization_metadata)
    return data, metadata


def extract_metadata(
    msg: SampleMessage, device: torch.device
) -> tuple[dict[str, torch.Tensor], SampleMessage]:
    """Separate user-defined metadata from a SampleMessage.

    GLT's ``to_hetero_data`` misinterprets ``#META.``-prefixed keys as
    edge types, causing failures with ``edge_dir="out"`` (it tries to call
    ``reverse_edge_type`` on metadata key strings).  This function separates
    metadata from the sampling data so the stripped message can be passed to
    GLT's ``_collate_fn`` without triggering the bug.

    The original ``msg`` is not modified.

    Args:
        msg: The SampleMessage to extract metadata from.
        device: The device to move metadata tensors to.

    Returns:
        A 2-tuple of:
        - metadata: Dict mapping metadata key (without ``#META.`` prefix) to tensor.
        - stripped_msg: A new SampleMessage with ``#META.``-prefixed keys removed.
    """
    meta_prefix = "#META."
    metadata: dict[str, torch.Tensor] = {}
    stripped_msg: SampleMessage = {}
    for k, v in msg.items():
        if k.startswith(meta_prefix):
            metadata[k[len(meta_prefix) :]] = v.to(device)
        else:
            stripped_msg[k] = v
    return metadata, stripped_msg


def attach_ppr_outputs(
    data: Union[Data, HeteroData],
    ppr_edge_indices: dict[EdgeType, torch.Tensor],
    ppr_weights: dict[EdgeType, torch.Tensor],
) -> None:
    """Attach PPR edge indices and weights onto a Data/HeteroData object.

    For each PPR edge type, sets ``data[edge_type].edge_index`` and
    ``data[edge_type].edge_attr`` in-place.  Called from the loader's
    ``_collate_fn`` only when a PPR sampler is active.

    Args:
        data: The Data or HeteroData object to attach outputs to.
        ppr_edge_indices: Dict mapping PPR edge type to ``[2, N]`` edge-index tensor.
        ppr_weights: Dict mapping PPR edge type to ``[N]`` weight tensor.

    Raises:
        AssertionError: If ``ppr_edge_indices`` and ``ppr_weights`` have different edge-type keys.
        ValueError: If homogeneous ``Data`` does not have exactly one PPR edge type.
    """
    assert ppr_edge_indices.keys() == ppr_weights.keys(), (
        f"PPR edge index and weight edge types must match, "
        f"got {set(ppr_edge_indices.keys())} vs {set(ppr_weights.keys())}"
    )
    if isinstance(data, Data):
        if len(ppr_edge_indices) != 1:
            raise ValueError(
                "Expected exactly one PPR edge type for homogeneous Data output, "
                f"got {set(ppr_edge_indices.keys())}"
            )
        edge_type = next(iter(ppr_edge_indices))
        data.edge_index = ppr_edge_indices[edge_type]
        data.edge_attr = ppr_weights[edge_type]
        # Homogeneous Data has no per-edge-type stores; the PPR edges are attached.
        return

    for edge_type, edge_index in ppr_edge_indices.items():
        data[edge_type].edge_index = edge_index
        data[edge_type].edge_attr = ppr_weights[edge_type]


def extract_edge_type_metadata(
    metadata: dict[str, torch.Tensor],
    prefixes: list[str],
) -> tuple[dict[str, dict[EdgeType, torch.Tensor]], dict[str, torch.Tensor]]:
    """Extract entries matching any of the given prefixes from metadata, grouped by prefix.

    Scans ``metadata`` for keys that start with any of the provided ``prefixes``.
    For each match, the suffix (everything after the matched prefix) is parsed via
    ``ast.literal_eval`` as an ``EdgeType`` tuple and added to that prefix's sub-dict.
    All unmatched keys are placed in the remaining dict.

    Each prefix gets its own sub-dict in the result, so distinct categories (e.g.
    positive labels, negative labels) can never collide even when extracted in one call.

    The original ``metadata`` is not modified.

    Example:

        matched, remaining = extract_edge_type_metadata(
            metadata=metadata,
            prefixes=[POSITIVE_LABEL_METADATA_KEY, NEGATIVE_LABEL_METADATA_KEY],
        )
        positive_labels = matched[POSITIVE_LABEL_METADATA_KEY]
        negative_labels = matched[NEGATIVE_LABEL_METADATA_KEY]

    Args:
        metadata: Dict of string keys to tensors.
        prefixes: List of prefixes to match against. Prefixes should be unique (no repeats).

    Returns:
        A 2-tuple of:
        - matched: Dict mapping each prefix to a sub-dict of
          ``{EdgeType: tensor}`` for all keys that started with that prefix.
          Every prefix in ``prefixes`` is guaranteed to be present as a key
          (with an empty dict if nothing matched).
        - remaining: Dict of all key/value pairs that matched no prefix.
    """
    matched: dict[str, dict[EdgeType, torch.Tensor]] = {p: {} for p in prefixes}
    remaining: dict[str, torch.Tensor] = {}
    for key, value in metadata.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                edge_type: EdgeType = ast.literal_eval(key[len(prefix) :])
                matched[prefix][edge_type] = value
                break
        else:
            remaining[key] = value
    return matched, remaining
