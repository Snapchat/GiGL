from collections.abc import Callable, Mapping
from typing import Optional

import torch
from graphlearn_torch.data import Feature

from gigl.common.logger import Logger
from gigl.distributed.dist_dataset import DistDataset
from gigl.src.common.types.graph_data import NodeType

logger = Logger()


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
            labels of shape [N, 1] and returns a boolean mask of the same [N, 1] shape, indicating which labels should be included.
            By default, filters out negative values.
    Returns:
        torch.Tensor: The down_sampled node ids of shape [M,], where M is the number of down_sampled node ids.
            Only nodes with valid labels are included in the output.
    """

    if isinstance(dataset.node_labels, Mapping):
        if node_type is None:
            raise ValueError(
                "node_type must be provided if the dataset is heterogeneous"
            )
        labels: Feature = dataset.node_labels[node_type]
    else:
        labels = dataset.node_labels

    labels_to_include = label_filter_fn(labels[node_ids]).squeeze(-1)

    down_sampled_node_ids = node_ids[labels_to_include]

    return down_sampled_node_ids
