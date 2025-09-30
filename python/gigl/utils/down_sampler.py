from collections.abc import Callable, Mapping
from typing import Protocol, Tuple, Union, overload, runtime_checkable

import torch

from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import NodeType
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE

logger = Logger()


@runtime_checkable
class NodeDownsampler(Protocol):
    """Protocol that should be satisfied for anything that is used to downsample node splits based on labels.

    This protocol handles downsampling of pre-computed train/val/test splits by filtering out nodes
    that don't have valid labels.

    Args:
        splits: The train/val/test splits to downsample. Either a tuple of tensors for homogeneous graphs or a
            mapping from node types to tuples of tensors for heterogeneous graphs.
        node_label_ids: The node label IDs tensor(s). For homogeneous graphs, this is a single tensor.
            For heterogeneous graphs, this is a mapping from node types to tensors.
        node_label_feats: The node label features tensor(s). For homogeneous graphs, this is a single tensor.
            For heterogeneous graphs, this is a mapping from node types to tensors.

    Returns:
        The downsampled train/val/test splits in the same format as the input splits.
    """

    @overload
    def __call__(
        self,
        splits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        node_label_ids: torch.Tensor,
        node_label_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    @overload
    def __call__(
        self,
        splits: Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        node_label_ids: Mapping[NodeType, torch.Tensor],
        node_label_feats: Mapping[NodeType, torch.Tensor],
    ) -> Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ...

    def __call__(
        self, *args, **kwargs
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        ...


class NodeLabelDownsampler:
    """
    Standard implementation of node downsampling based on label values.

    This downsampler filters train/val/test splits using a configurable label filter function.
    By default, nodes with negative label values are excluded from the splits.
    """

    def __init__(
        self,
        should_downsample_train_split: bool = True,
        should_downsample_val_split: bool = True,
        should_downsample_test_split: bool = True,
        label_filter_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x >= 0,
    ):
        """
        Initialize the LabelDownsampler.

        Args:
            should_downsample_train_split (bool): Whether to apply downsampling to the training split.
            should_downsample_val_split (bool): Whether to apply downsampling to the validation split.
            should_downsample_test_split (bool): Whether to apply downsampling to the test split.
            label_filter_fn (Callable[[torch.Tensor], torch.Tensor]): A callable that takes a tensor of
                labels and returns a boolean mask, indicating which labels should be included. By default,
                filters out negative values.
        """
        self._should_downsample_train_split = should_downsample_train_split
        self._should_downsample_val_split = should_downsample_val_split
        self._should_downsample_test_split = should_downsample_test_split
        self._label_filter_fn = label_filter_fn

    @overload
    def __call__(
        self,
        splits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        node_label_ids: torch.Tensor,
        node_label_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    @overload
    def __call__(
        self,
        splits: Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        node_label_ids: Mapping[NodeType, torch.Tensor],
        node_label_feats: Mapping[NodeType, torch.Tensor],
    ) -> Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ...

    def __call__(
        self,
        splits: Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ],
        node_label_ids: Union[
            torch.Tensor,
            Mapping[NodeType, torch.Tensor],
        ],
        node_label_feats: Union[
            torch.Tensor,
            Mapping[NodeType, torch.Tensor],
        ],
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        """Downsample the provided splits based on the provided node labels.

        Args:
            splits: The train/val/test splits to downsample. For homogeneous graphs, this is
                a tuple of (train, val, test) tensors. For heterogeneous graphs, this is
                a mapping from node types to (train, val, test) tuples.
            node_label_ids: The node label IDs. For homogeneous graphs, this is a single tensor.
                For heterogeneous graphs, this is a mapping from node types to tensors.
            node_label_feats: The node label features. For homogeneous graphs, this is a single tensor.
                For heterogeneous graphs, this is a mapping from node types to tensors.

        Returns:
            The downsampled splits in the same format as the input. Only nodes with valid
            labels are included in the output splits.
        """
        # Convert everything to heterogeneous format internally
        if isinstance(splits, Mapping):
            if not isinstance(node_label_ids, Mapping) or not isinstance(
                node_label_feats, Mapping
            ):
                raise ValueError(
                    "When splits is a mapping (heterogeneous), node_label_ids and node_label_feats must also be mappings"
                )
            is_heterogeneous = True
            splits_dict = splits
            node_label_ids_dict = node_label_ids
            node_label_feats_dict = node_label_feats
        else:
            if isinstance(node_label_ids, Mapping) or isinstance(
                node_label_feats, Mapping
            ):
                raise ValueError(
                    "When splits is a tuple (homogeneous), node_label_ids and node_label_feats must be single tensors"
                )
            is_heterogeneous = False
            splits_dict = {DEFAULT_HOMOGENEOUS_NODE_TYPE: splits}
            node_label_ids_dict = {DEFAULT_HOMOGENEOUS_NODE_TYPE: node_label_ids}
            node_label_feats_dict = {DEFAULT_HOMOGENEOUS_NODE_TYPE: node_label_feats}

        # Process all as heterogeneous
        downsampled_splits = self._downsample_heterogeneous_internal(
            splits_dict, node_label_ids_dict, node_label_feats_dict
        )

        if is_heterogeneous:
            return downsampled_splits
        else:
            return downsampled_splits[DEFAULT_HOMOGENEOUS_NODE_TYPE]

    def _downsample_heterogeneous_internal(
        self,
        splits: Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        node_label_ids: Mapping[NodeType, torch.Tensor],
        node_label_feats: Mapping[NodeType, torch.Tensor],
    ) -> Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Internal method to downsample heterogeneous splits based on node labels.

        Args:
            splits: Mapping from node types to (train, val, test) tuples.
            node_label_ids: Mapping from node types to label ID tensors.
            node_label_feats: Mapping from node types to label feature tensors.

        Returns:
            Mapping from node types to downsampled (train, val, test) tuples.
        """
        downsampled_splits = {}

        for node_type, (train_node_ids, val_node_ids, test_node_ids) in splits.items():
            if node_type not in node_label_ids:
                raise ValueError(
                    f"Node type {node_type} found in splits but not in node labels ids."
                )
            if node_type not in node_label_feats:
                raise ValueError(
                    f"Node type {node_type} found in splits but not in node label features."
                )

            label_ids = node_label_ids[node_type]
            label_feats = node_label_feats[node_type]
            (
                downsampled_train_node_ids,
                downsampled_val_node_ids,
                downsampled_test_node_ids,
            ) = self._downsample_single_splits(
                train_node_ids, val_node_ids, test_node_ids, label_ids, label_feats
            )
            downsampled_splits[node_type] = (
                downsampled_train_node_ids,
                downsampled_val_node_ids,
                downsampled_test_node_ids,
            )

        return downsampled_splits

    def _downsample_single_splits(
        self,
        train_node_ids: torch.Tensor,
        val_node_ids: torch.Tensor,
        test_node_ids: torch.Tensor,
        node_label_ids: torch.Tensor,
        node_label_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Downsample a single set of train/val/test splits based on node labels.

        Args:
            train_node_ids: Training node indices.
            val_node_ids: Validation node indices.
            test_node_ids: Test node indices.
            node_label_ids: Node label IDs tensor.
            node_label_feats: Node label features tensor.

        Returns:
            Tuple of downsampled (train_node_ids, val_node_ids, test_node_ids) tensors.
        """

        # Sort node_label_ids to enable binary search via searchsorted.
        # sort_indices tracks where each original element ended up after sorting,
        # allowing us to map back from sorted positions to original positions.
        sorted_node_label_ids, sort_indices = torch.sort(node_label_ids)

        # Use binary search to find where each train/val/test node ID would be inserted
        # in the sorted array. This gives us the position of each node ID in the sorted tensor.
        train_sorted_indices = torch.searchsorted(sorted_node_label_ids, train_node_ids)
        val_sorted_indices = torch.searchsorted(sorted_node_label_ids, val_node_ids)
        test_sorted_indices = torch.searchsorted(sorted_node_label_ids, test_node_ids)

        # Map from sorted positions back to original positions in the unsorted node_label_ids.
        # sort_indices[i] gives the original index of the element that is now at sorted position i.
        # This gives us the final indices we need to index into node_label_feats.
        train_label_indices = sort_indices[train_sorted_indices]
        val_label_indices = sort_indices[val_sorted_indices]
        test_label_indices = sort_indices[test_sorted_indices]

        train_labels_to_include = self._label_filter_fn(
            node_label_feats[train_label_indices]
        ).squeeze(-1)

        val_labels_to_include = self._label_filter_fn(
            node_label_feats[val_label_indices]
        ).squeeze(-1)

        test_labels_to_include = self._label_filter_fn(
            node_label_feats[test_label_indices]
        ).squeeze(-1)

        if self._should_downsample_train_split:
            downsampled_train_node_ids = train_node_ids[train_labels_to_include]
        else:
            downsampled_train_node_ids = train_node_ids

        if self._should_downsample_val_split:
            downsampled_val_node_ids = val_node_ids[val_labels_to_include]
        else:
            downsampled_val_node_ids = val_node_ids

        if self._should_downsample_test_split:
            downsampled_test_node_ids = test_node_ids[test_labels_to_include]
        else:
            downsampled_test_node_ids = test_node_ids

        return (
            downsampled_train_node_ids,
            downsampled_val_node_ids,
            downsampled_test_node_ids,
        )
