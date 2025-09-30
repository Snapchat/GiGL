from collections.abc import Callable, Mapping
from typing import Optional

import torch
from graphlearn_torch.data import Feature

from gigl.common.logger import Logger
from gigl.distributed.dist_dataset import DistDataset
from gigl.src.common.types.graph_data import NodeType

logger = Logger()


def down_sample_node_ids_from_labels(
    node_ids: torch.Tensor,
    id2idx: torch.Tensor,
    node_label_feats: torch.Tensor,
    label_filter_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x >= 0,
) -> torch.Tensor:
    """
    Downsample the provided node ids based on the provided node labels. Downsampling means that
    we will sample a subset of the node ids to use for any downstream purpose. The node label values will be used to
    determine what gets downsampled, and can be customized with the `label_filter_fn` argument.
    By default, this will filter out negative label values from the node ids.

    Args:
        node_ids (torch.Tensor): The node ids to down_sample Should be a 1D tensor of shape (N,).
        id2idx (torch.Tensor): Mapping from node IDs to indices in the feature tensor. Should be a 1D tensor of shape (max_node_id,).
        node_label_feats (torch.Tensor): The node label features. Should be a 2D tensor of shape (N, 1), where N is the number of nodes.
        label_filter_fn (Callable[[torch.Tensor], torch.Tensor]): A callable that takes a tensor of
            labels and returns a boolean mask, indicating which labels should be included. By default,
            filters out negative values.
    Returns:
        torch.Tensor: The down_sampled node ids. Only nodes with valid labels are included in the output.
    """

    # Use id2idx to get the feature indices directly
    feature_indices = id2idx[node_ids]

    # Apply the label filter function
    labels_to_include = label_filter_fn(node_label_feats[feature_indices]).squeeze(-1)

    # Return the down_sampled valid node ids
    down_sampled_valid_node_ids = node_ids[labels_to_include]

    return down_sampled_valid_node_ids


def down_sample_node_ids_from_dataset_labels(
    dataset: DistDataset,
    node_ids: torch.Tensor,
    node_type: Optional[NodeType] = None,
    label_filter_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x >= 0,
) -> torch.Tensor:
    """
    Down sample node ids using the provided dataset's labels. Downsampling means that
    we will sample a subset of the node ids to use for any downstream purpose. The node label values will be used to
    determine what gets downsampled, and can be customized with the `label_filter_fn` argument.
    By default, this will filter out negative label values from the node ids.

    Args:
        dataset (DistDataset): The dataset to down_sample.
        node_ids (torch.Tensor): The node ids to down_sample. Should be a 1D tensor of shape (N,).
        node_type (Optional[NodeType]): The node type to down_sample. If the dataset is heterogeneous, this must be provided.
        label_filter_fn (Callable[[torch.Tensor], torch.Tensor]): A callable that takes a tensor of
            labels and returns a boolean mask, indicating which labels should be included. By default,
            filters out negative values.
    Returns:
        The down_sampled node ids. Only nodes with valid labels are included in the output.
    """

    if isinstance(dataset.node_labels, Mapping):
        if node_type is None:
            raise ValueError(
                "node_type must be provided if the dataset is heterogeneous"
            )
        labels: Feature = dataset.node_labels[node_type]
    else:
        labels = dataset.node_labels

    # We need to lazy init the labels so that its feature and id2index fields are populated. Otherwise, these values
    # will be None.
    labels.lazy_init_with_ipc_handle()

    down_sampled_node_ids = down_sample_node_ids_from_labels(
        node_ids=node_ids,
        id2idx=labels.id2index,
        node_label_feats=labels.feature_tensor,
        label_filter_fn=label_filter_fn,
    )

    return down_sampled_node_ids
