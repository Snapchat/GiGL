import gc
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Callable, Literal, Optional, Protocol, Tuple, Union, overload

import torch

from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
)

logger = Logger()


class NodeAnchorLinkSplitter(Protocol):
    """Protocol that should be satisfied for anything that is used to split on edges.

    The edges must be provided in COO format, as dense tensors.
    https://tbetcke.github.io/hpc_lecture_notes/sparse_data_structures.html

    Args:
        edge_index: The edges to split on in COO format. 2 x N
    Returns:
        The train (1 x X), val (1 X Y), test (1 x Z) nodes. X + Y + Z = N
    """

    @overload
    def __call__(
        self,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    @overload
    def __call__(
        self,
        edge_index: Mapping[EdgeType, torch.Tensor],
    ) -> Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ...

    def __call__(
        self, *args, **kwargs
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        ...


def _fast_hash(x: torch.Tensor) -> torch.Tensor:
    """Fast hash function.

    Hashes each element of the input tensor `x` using the fast hash function.
    Based on https://stackoverflow.com/a/12996028

    We use the `Tensor.bitwise_xor_` and `Tensor.multiply_` to avoid creating new tensors.
    Sadly, we cannot avoid the out-place shifts (I think, there may be some bit-wise voodoo here),
    but in testing we do not increase memory but more than a few MB for a 1G input so it should be fine.

    Note that _fast_hash(0) = 0.

    Arguments:
        x (torch.Tensor): The input tensor to hash. N x M

    Returns:
        The hash values of the input tensor `x`. N x M
    """
    x = x.clone().detach()
    if x.dtype == torch.int32:
        x.bitwise_xor_(x >> 16)
        x.multiply_(0x7FEB352D)
        x.bitwise_xor_(x >> 15)
        x.multiply_(0x846CA68B)
        x.bitwise_xor_(x >> 16)
    elif x.dtype == torch.int64:
        x.bitwise_xor_(x >> 30)
        x.multiply_(0xBF58476D1CE4E5B9)
        x.bitwise_xor_(x >> 27)
        x.multiply_(0x94D049BB133111EB)
        x.bitwise_xor_(x >> 31)
    else:
        raise ValueError(f"Unsupported dtype {x.dtype}")

    return x


class HashedNodeAnchorLinkSplitter:
    """Selects train, val, and test nodes based on some provided edge index.

    In node-based splitting, a node may only ever live in one split. E.g. if one
    node has two label edges, *both* of those edges will be placed into the same split.

    The edges must be provided in COO format, as dense tensors.
    https://tbetcke.github.io/hpc_lecture_notes/sparse_data_structures.html
    Where the first row of out input are the node ids we that are the "source" of the edge,
    and the second row are the node ids that are the "destination" of the edge.


    Note that there is some tricky interplay with this and the `sampling_direction` parameter.
    Take the graph [A -> B] as an example.
    If `sampling_direction` is "in", then B is the source and A is the destination.
    If `sampling_direction` is "out", then A is the source and B is the destination.
    """

    def __init__(
        self,
        sampling_direction: Union[Literal["in", "out"], str],
        num_val: Union[float, int] = 0.1,
        num_test: Union[float, int] = 0.1,
        hash_function: Callable[[torch.Tensor], torch.Tensor] = _fast_hash,
        edge_types: Optional[Union[EdgeType, Sequence[EdgeType]]] = None,
    ):
        """Initializes the HashedNodeAnchorLinkSplitter.

        Args:
            sampling_direction (Union[Literal["in", "out"], str]): The direction to sample the nodes. Either "in" or "out".
            num_val (Union[float, int]): The percentage of nodes to use for training. Defaults to 0.1 (10%).
                                         If an integer is provided, than exactly that number of nodes will be in the validation split.
            num_test (Union[float, int]): The percentage of nodes to use for validation. Defaults to 0.1 (10%).
                                          If an integer is provided, than exactly that number of nodes will be in the test split.
            hash_function (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The hash function to use. Defaults to `_fast_hash`.
            edge_types: The supervision edge types we should use for splitting.
                        Must be provided if we are splitting a heterogeneous graph.
        """
        _check_sampling_direction(sampling_direction)
        _check_val_test_percentage(num_val, num_test)

        self._sampling_direction = sampling_direction
        self._num_val = num_val
        self._num_test = num_test
        self._hash_function = hash_function

        if edge_types is None:
            edge_types = [DEFAULT_HOMOGENEOUS_EDGE_TYPE]
        elif isinstance(edge_types, EdgeType):
            edge_types = [edge_types]
        self._supervision_edge_types: Sequence[EdgeType] = edge_types

    def __call__(
        self,
        edge_index: Union[
            torch.Tensor, Mapping[EdgeType, torch.Tensor]
        ],  # 2 x N (num_edges)
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        if isinstance(edge_index, torch.Tensor):
            if self._supervision_edge_types != [DEFAULT_HOMOGENEOUS_EDGE_TYPE]:
                logger.warning(
                    f"You provided edge-types {self._supervision_edge_types} but the edge index is homogeneous. Ignoring edge types."
                )
            is_heterogeneous = False
            edge_index = {DEFAULT_HOMOGENEOUS_EDGE_TYPE: edge_index}

        else:
            if (
                self._supervision_edge_types == [DEFAULT_HOMOGENEOUS_EDGE_TYPE]
                or not self._supervision_edge_types
            ):
                raise ValueError(
                    "If edge_index is a mapping, edges_to_split must be provided."
                )
            missing = set(self._supervision_edge_types) - edge_index.keys()
            if missing:
                raise ValueError(
                    f"Missing edge types from provided edge index: {missing}. Expected edges types {self._supervision_edge_types} to be in the mapping, but got {edge_index.keys()}."
                )
            is_heterogeneous = True

        # First, find max node id per node type.
        # This way, we can de-dup via torch.bincount, which is much faster than torch.unique.
        # NOTE: For cases where we have large ranges of nodes ids that are all much > 0 (e. [0, 100_000, ...,1_000_000])])
        # It may be faster to use `torch.unique` instead of `torch.bincount`, since `torch.bincount` will create a tensor of size 1_000_000.
        # TODO(kmonte): investigate this.
        # We also store references to all tensors of a given node type, for convenient access later.
        max_node_id_by_type: dict[NodeType, int] = defaultdict(int)
        node_ids_by_node_type: dict[NodeType, list[torch.Tensor]] = defaultdict(list)
        for edge_type_to_split in self._supervision_edge_types:
            coo_edges = edge_index[edge_type_to_split]
            _check_edge_index(coo_edges)
            anchor_nodes = (
                coo_edges[1] if self._sampling_direction == "in" else coo_edges[0]
            )
            anchor_node_type = (
                edge_type_to_split.dst_node_type
                if self._sampling_direction == "in"
                else edge_type_to_split.src_node_type
            )
            max_node_id_by_type[anchor_node_type] = int(
                max(
                    max_node_id_by_type[anchor_node_type],
                    torch.max(anchor_nodes).item() + 1,
                )
            )
            node_ids_by_node_type[anchor_node_type].append(anchor_nodes)
        # Second, we go through all node types and split them.
        # Note the approach here (with `torch.argsort`) isn't the quickest
        # we could avoid calling `torch.argsort` and do something like:
        # hash_values = ...
        # train_mask = hash_values < train_percentage
        # train = nodes_to_select[train_mask]
        # That approach is about 2x faster (30s -> 15s on 1B nodes),
        # but with this `argsort` approach we can be more exact with the number of nodes per split.
        # The memory usage seems the same across both approaches.

        # De-dupe this way instead of using `unique` to avoid the overhead of sorting.
        # This approach, goes from ~60s to ~30s on 1B edges.
        # collected_anchor_nodes (the values of node_ids_by_node_type) is a list of tensors for a given node type.
        # For example if we have `{(A to B): [0, 1], (A to C): [0, 2]}` then we will have
        # `collected_anchor_nodes` = [[0, 1], [0, 2]].
        splits: dict[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for anchor_node_type, collected_anchor_nodes in node_ids_by_node_type.items():
            max_node_id = max_node_id_by_type[anchor_node_type]
            node_id_count = torch.zeros(max_node_id, dtype=torch.int64)
            for anchor_nodes in collected_anchor_nodes:
                node_id_count.add_(torch.bincount(anchor_nodes, minlength=max_node_id))
            # This line takes us from a count of all node ids, e.g. `[0, 2, 0, 1]`
            # To a tensor of the non-zero counts, e.g. `[[1], [3]]`
            # and the `squeeze` converts that to a 1d tensor (`[1, 3]`).
            nodes_to_select = torch.nonzero(node_id_count).squeeze()
            # node_id_count no longer needed, so we can clean up it's memory.
            del node_id_count
            gc.collect()

            hash_values = torch.argsort(self._hash_function(nodes_to_select))  # 1 x M
            nodes_to_select = nodes_to_select[hash_values]  # 1 x M

            # hash_values no longer needed, so we can clean up it's memory.
            del hash_values
            gc.collect()

            if isinstance(self._num_val, int):
                num_val = self._num_val
            else:
                num_val = int(nodes_to_select.numel() * self._num_val)
            if isinstance(self._num_test, int):
                num_test = self._num_test
            else:
                num_test = int(nodes_to_select.numel() * self._num_test)

            num_train = nodes_to_select.numel() - num_val - num_test
            if num_train <= 0:
                raise ValueError(
                    f"Invalid number of nodes to split. Expected more than 0. Originally had {nodes_to_select.numel()} nodes but due to having `num_test` = {self._num_test} and `num_val` = {self._num_val} got no training node.."
                )

            train = nodes_to_select[:num_train]  # 1 x num_train_nodes
            val = nodes_to_select[num_train : num_val + num_train]  # 1 x num_val_nodes
            test = nodes_to_select[num_train + num_val :]  # 1 x num_test_nodes
            splits[anchor_node_type] = (train, val, test)
        if is_heterogeneous:
            return splits
        else:
            return splits[DEFAULT_HOMOGENEOUS_NODE_TYPE]


def _check_sampling_direction(sampling_direction: str):
    if sampling_direction not in ["in", "out"]:
        raise ValueError(
            f"Invalid sampling direction {sampling_direction}. Expected 'in' or 'out'."
        )


def _check_val_test_percentage(
    val_percentage: Union[float, int], test_percentage: Union[float, int]
):
    """Checks that the val and test percentages make sense, e.g. we can still have train nodes, and they are non-negative."""
    if val_percentage < 0:
        raise ValueError(
            f"Invalid val percentage {val_percentage}. Expected a value greater than 0."
        )
    if test_percentage < 0:
        raise ValueError(
            f"Invalid test percentage {test_percentage}. Expected a value greater than 0."
        )
    if isinstance(val_percentage, float) and isinstance(test_percentage, float):
        if not 0 <= test_percentage < 1:
            raise ValueError(
                f"Invalid test percentage {test_percentage}. Expected a value between 0 and 1."
            )
        if val_percentage <= 0:
            raise ValueError(
                f"Invalid val percentage {val_percentage}. Expected a value greater than 0."
            )
        if val_percentage + test_percentage >= 1:
            raise ValueError(
                f"Invalid val percentage {val_percentage} and test percentage ({test_percentage}). Expected values such that test percentages + val percentage < 1."
            )


def _check_edge_index(edge_index: torch.Tensor):
    """Asserts edge index is the appropriate shape and is not sparse."""
    size = edge_index.size()
    if size[0] != 2 or len(size) != 2:
        raise ValueError(
            f"Expected edges to be provided in COO format in the form of a 2xN tensor. Recieved a tensor of shape: {size}."
        )
    if edge_index.is_sparse:
        raise ValueError("Expected a dense tensor. Received a sparse tensor.")


def select_ssl_positive_label_edges(
    edge_index: torch.Tensor, positive_label_percentage: float
) -> torch.Tensor:
    """
    Selects a percentage of edges from an edge index to use for self-supervised positive labels.
    Note that this function does not mask these labeled edges from the edge index tensor.

    Args:
        edge_index (torch.Tensor): Edge Index tensor of shape [2, num_edges]
        positive_label_percentage (float): Percentage of edges to select as positive labels
    Returns:
        torch.Tensor: Tensor of positive edges of shape [2, num_labels]
    """
    if not (0 <= positive_label_percentage <= 1):
        raise ValueError(
            f"Label percentage must be between 0 and 1, got {positive_label_percentage}"
        )
    if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Provided edge index tensor must have shape [2, num_edges], got {edge_index.shape}"
        )
    num_labels = int(edge_index.shape[1] * positive_label_percentage)
    label_inds = torch.randperm(edge_index.size(1))[:num_labels]
    return edge_index[:, label_inds]
