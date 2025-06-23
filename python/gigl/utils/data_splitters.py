import gc
from collections import defaultdict
from collections.abc import Mapping
from typing import (
    Callable,
    Final,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    overload,
)

import torch
from graphlearn_torch.data import Dataset, Topology
from torch_geometric.typing import EdgeType as PyGEdgeType

from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    reverse_edge_type,
)

logger = Logger()

PADDING_NODE: Final[torch.Tensor] = torch.tensor(-1, dtype=torch.int64)


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

    @property
    def should_convert_labels_to_edges(self):
        ...


def _fast_hash(x: torch.Tensor) -> torch.Tensor:
    """Fast hash function.

    Hashes each element of the input tensor `x` using the fast hash function.
    Based on https://stackoverflow.com/a/12996028

    We use the `Tensor.bitwise_xor_` and `Tensor.multiply_` to avoid creating new tensors.
    Sadly, we cannot avoid the out-place shifts (I think, there may be some bit-wise voodoo here),
    but in testing we do not increase memory but more than a few MB for a 1G input so it should be fine.


    Arguments:
        x (torch.Tensor): The input tensor to hash. N x M

    Returns:
        The hash values of the input tensor `x`. N x M
    """
    x = x.clone().detach()

    # Add one so that _fast_hash(0) != 0
    x.add_(1)
    if x.dtype == torch.int32:
        x.bitwise_xor_(x >> 16)
        x.multiply_(0x7FEB352D)
        x.bitwise_xor_(x >> 15)
        x.multiply_(0x846CA68B)
        x.bitwise_xor_(x >> 16)
        # And again for better mixing ;)
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
        # And again for better mixing ;)
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

    NOTE: This splitter must be called when a Torch distributed process group is initialized.
    e.g. `torch.distributed.init_process_group` must be called before using this splitter.


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
        num_val: float = 0.1,
        num_test: float = 0.1,
        hash_function: Callable[[torch.Tensor], torch.Tensor] = _fast_hash,
        supervision_edge_types: Optional[list[EdgeType]] = None,
        should_convert_labels_to_edges: bool = True,
    ):
        """Initializes the HashedNodeAnchorLinkSplitter.

        Args:
            sampling_direction (Union[Literal["in", "out"], str]): The direction to sample the nodes. Either "in" or "out".
            num_val (float): The percentage of nodes to use for training. Defaults to 0.1 (10%).
            num_test (float): The percentage of nodes to use for validation. Defaults to 0.1 (10%).
            hash_function (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The hash function to use. Defaults to `_fast_hash`.
            supervision_edge_types (Optional[list[EdgeType]]): The supervision edge types we should use for splitting.
                Must be provided if we are splitting a heterogeneous graph. If None, uses the default message passing edge type in the graph.
            should_convert_labels_to_edges (bool): Whether label should be converted into an edge type in the graph. If provided, will make
                `gigl.distributed.build_dataset` convert all labels into edges, and will infer positive and negative edge types based on
                `supervision_edge_types`.
        """
        _check_sampling_direction(sampling_direction)
        _check_val_test_percentage(num_val, num_test)

        self._sampling_direction = sampling_direction
        self._num_val = num_val
        self._num_test = num_test
        self._hash_function = hash_function
        self._should_convert_labels_to_edges = should_convert_labels_to_edges

        if supervision_edge_types is None:
            supervision_edge_types = [DEFAULT_HOMOGENEOUS_EDGE_TYPE]

        # Supervision edge types are the edge type which will be used for positive and negative labels.
        # Labeled edge types are the actual edge type that be injected into the edge index tensor.
        # If should_convert_labels_to_edges=False, supervision edge types and labeled edge types will be the same,
        # since the supervision edge type already exists in the edge index graph. Otherwise, we assume that the
        # edge index tensor initially only contains the message passing edges, and will inject the labeled edge into the
        # edge index based on some provided labels. As a result, the labeled edge types will be an augmented version of the
        # supervision edge types.

        # For example, if `should_convert_labels_to_edges=True` and we provide supervision_edge_types=[("user", "to", "story")], the
        # labeled edge types will be ("user", "to_gigl_positive", "story") and ("user", "to_gigl_negative", "story"), if there are negative labels.

        # If `should_convert_labels_to_edges=False` and we provide supervision_edge_types=[("user", "positive", "story")], the labeled edge type will
        # also be ("user", "positive", "story"), meaning that all edges in the loaded edge index tensor with this edge type will be treated as a labeled
        # edge and will be used for splitting.

        self._supervision_edge_types: Sequence[EdgeType] = supervision_edge_types
        self._labeled_edge_types: Sequence[EdgeType]
        if should_convert_labels_to_edges:
            labeled_edge_types = [
                message_passing_to_positive_label(supervision_edge_type)
                for supervision_edge_type in supervision_edge_types
            ] + [
                message_passing_to_negative_label(supervision_edge_type)
                for supervision_edge_type in supervision_edge_types
            ]
            # If the edge direction is "in", we must reverse the labeled edge type, since separately provided labels are expected to be initially outgoing, and all edges
            # in the graph must have the same edge direction.
            if sampling_direction == "in":
                self._labeled_edge_types = [
                    reverse_edge_type(labeled_edge_type)
                    for labeled_edge_type in labeled_edge_types
                ]
            else:
                self._labeled_edge_types = labeled_edge_types
        else:
            self._labeled_edge_types = supervision_edge_types

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
        self,
        edge_index: Union[
            torch.Tensor, Mapping[EdgeType, torch.Tensor]
        ],  # 2 x N (num_edges)
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        if isinstance(edge_index, torch.Tensor):
            if self._labeled_edge_types != [DEFAULT_HOMOGENEOUS_EDGE_TYPE]:
                logger.warning(
                    f"You provided edge-types {self._labeled_edge_types} but the edge index is homogeneous. Ignoring supervision edge types."
                )
            is_heterogeneous = False
            edge_index = {DEFAULT_HOMOGENEOUS_EDGE_TYPE: edge_index}

        else:
            is_heterogeneous = True

        # First, find max node id per node type.
        # This way, we can de-dup via torch.bincount, which is much faster than torch.unique.
        # NOTE: For cases where we have large ranges of nodes ids that are all much > 0 (e. [0, 100_000, ...,1_000_000])])
        # It may be faster to use `torch.unique` instead of `torch.bincount`, since `torch.bincount` will create a tensor of size 1_000_000.
        # TODO(kmonte): investigate this.
        # We also store references to all tensors of a given node type, for convenient access later.
        max_node_id_by_type: dict[NodeType, int] = defaultdict(int)
        node_ids_by_node_type: dict[NodeType, list[torch.Tensor]] = defaultdict(list)
        for edge_type_to_split, coo_edges in edge_index.items():
            # In this case, the labels should be converted to an edge type in graph with relation containing `is_gigl_positive` or `is_gigl_negative`.
            if edge_type_to_split not in self._labeled_edge_types:
                # We skip if the current edge type is not a labeled edge type, since we don't want to generate splits for edges which aren't used for supervision
                continue

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
            node_id_count = torch.zeros(max_node_id, dtype=torch.uint8)
            for anchor_nodes in collected_anchor_nodes:
                node_id_count.add_(torch.bincount(anchor_nodes, minlength=max_node_id))
            # This line takes us from a count of all node ids, e.g. `[0, 2, 0, 1]`
            # To a tensor of the non-zero counts, e.g. `[[1], [3]]`
            # and the `squeeze` converts that to a 1d tensor (`[1, 3]`).
            nodes_to_select = torch.nonzero(node_id_count).squeeze()
            # node_id_count no longer needed, so we can clean up it's memory.
            del node_id_count
            gc.collect()

            hash_values = self._hash_function(nodes_to_select)  # 1 x M
            # Now, we want to normalize the hash values to [0, 1) range so we can select them easily into splits.
            # We want to do this *globally* e.g. across all processes,
            # so that we can ensure that the same nodes are selected for the same split across all processes.
            # If we don't do this, then if we have `[0, 1, 2, 3, 4]` on one process and `[4, 5, 6, 7]` on another,
            # with the identity hash `4` may end up in Test in one rank and Train in another.
            min_hash_value, max_hash_value = map(
                torch.Tensor.item, hash_values.aminmax()
            )
            if torch.distributed.is_initialized():
                all_max_and_mins = [
                    torch.zeros(2, dtype=torch.int64)
                    for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather(
                    all_max_and_mins,
                    torch.tensor([max_hash_value, min_hash_value], dtype=torch.int64),
                )
                global_max_hash_value = max_hash_value
                global_min_hash_value = min_hash_value
                for max_and_min in all_max_and_mins:
                    global_max_hash_value = max(
                        global_max_hash_value, max_and_min[0].item()
                    )
                    global_min_hash_value = min(
                        global_min_hash_value, max_and_min[1].item()
                    )
            else:
                raise RuntimeError(
                    f"{type(self).__name__} requires a Torch distributed process group, but none was found. Please initialize a process group (`torch.distributed.init_process_group`) before using this splitter."
                )
            hash_values = (
                hash_values - global_min_hash_value
            ) / global_max_hash_value  # Normalize the hash values to [0, 1)

            # Now that we've normalized the hash values, we can select the train, val, and test nodes.
            test_inds = hash_values >= 1 - self._num_test  # 1 x M
            val_inds = (
                hash_values >= 1 - self._num_test - self._num_val
            ) & ~test_inds  # 1 x M
            del hash_values
            gc.collect()
            train_inds = ~test_inds & ~val_inds  # 1 x M
            train = nodes_to_select[train_inds]  # 1 x num_train_nodes
            val = nodes_to_select[val_inds]  # 1 x num_val_nodes
            test = nodes_to_select[test_inds]  # 1 x num_test_nodes
            splits[anchor_node_type] = (train, val, test)
            # We no longer need the nodes to select, so we can clean up their memory.
            del nodes_to_select, train_inds, val_inds, test_inds
            gc.collect()
        if len(splits) == 0:
            raise ValueError(
                f"Found no edge types to split from the provided edge index: {edge_index.keys()} using labeled edge types {self._labeled_edge_types}"
            )

        if is_heterogeneous:
            return splits
        else:
            return splits[DEFAULT_HOMOGENEOUS_NODE_TYPE]

    @property
    def should_convert_labels_to_edges(self):
        return self._should_convert_labels_to_edges


def get_labels_for_anchor_nodes(
    dataset: Dataset,
    node_ids: torch.Tensor,
    positive_label_edge_type: PyGEdgeType,
    negative_label_edge_type: Optional[PyGEdgeType] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Selects labels for the given node ids based on the provided edge types.

    The labels returned are padded with `PADDING_NODE` to the maximum number of labels, so that we don't need to work with jagged tensors.
    The labels are N x M, where N is the number of nodes and M is the max number of labels.
    For a given ith node id, the ith row of the labels tensor will contain the labels for the given node id.
    e.g. if we have node_ids = [0, 1, 2] and the following topology:
        0 -> 1 -> 2
        0 -> 2
    and we provide node_ids = [0, 1]
    then the returned tensor will be:
        [
            [
                1, # Positive node (0 -> 1)
                2, # Positive node (0 -> 2)
            ],
            [
                2, # Positive node (1 -> 2)
                -1, # Positive node (padded)
            ],
        ]
    If positive and negative label edge types are provided:
        * All negative label node ids must be present in the positive label node ids.
        * For any positive label node id that does not have a negative label, the negative label will be padded with `PADDING_NODE`.
    Args:
        dataset (Dataset): The dataset storing the graph info, must be heterogeneous.
        node_ids (torch.Tensor): The node ids to use for the labels. [N]
        positive_label_edge_type (PyGEdgeType): The edge type to use for the positive labels.
        negative_label_edge_type (Optional[PyGEdgeType]): The edge type to use for the negative labels.
            Defaults to None. If not provided no negative labels will be returned.
    Returns:
        Tuple of (positive labels, negative_labels?)
        negative labels may be None depending on if negative_label_edge_type is provided.
        The returned tensors are of shape N x M where N is the number of nodes and M is the max number of labels, per type.
    """
    if not isinstance(dataset.graph, Mapping):
        raise ValueError(
            "The dataset must be heterogeneous to select labels for anchor nodes."
        )
    positive_node_topo = dataset.graph[positive_label_edge_type].topo
    if negative_label_edge_type is not None:
        negative_node_topo = dataset.graph[negative_label_edge_type].topo
    else:
        negative_node_topo = None

    # Labels is NxM, where N is the number of nodes, and M is the max number of labels.
    positive_labels = _get_padded_labels(
        node_ids, positive_node_topo, allow_non_existant_node_ids=False
    )

    if negative_node_topo is not None:
        # Labels is NxM, where N is the number of nodes, and M is the max number of labels.
        negative_labels = _get_padded_labels(
            node_ids, negative_node_topo, allow_non_existant_node_ids=True
        )
    else:
        negative_labels = None

    return positive_labels, negative_labels


def _get_padded_labels(
    anchor_node_ids: torch.Tensor,
    topo: Topology,
    allow_non_existant_node_ids: bool = False,
) -> torch.Tensor:
    """Returns the padded labels and the max range of labels.

    Given anchor node ids and a topology, this function returns a tensor
    which contains all of the node ids that are connected to the anchor node ids.
    The tensor is padded with `PADDING_NODE` to the maximum number of labels.

    Args:
        anchor_node_ids (torch.Tensor): The anchor node ids to use for the labels. [N]
        topo (Topology): The topology to use for the labels.
        allow_non_existant_node_ids (bool): If True, will allow anchor node ids that do not exist in the topology.
            This means that the returned tensor will be padded with `PADDING_NODE` for those anchor node ids.
    Returns:
        The shape of the returned tensor is [N, max_number_of_labels].
    """
    # indptr is the ROW_INDEX of a CSR matrix.
    # and indices is the COL_INDEX of a CSR matrix.
    # See https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
    # Note that GLT defaults to CSR under the hood, if this changes, we will need to update this.
    indptr = topo.indptr  # [N]
    indices = topo.indices  # [M]
    extra_nodes_to_pad = 0
    if allow_non_existant_node_ids:
        valid_ids = anchor_node_ids < (indptr.size(0) - 1)
        extra_nodes_to_pad = int(torch.count_nonzero(~valid_ids).item())
        anchor_node_ids = anchor_node_ids[valid_ids]
    starts = indptr[anchor_node_ids]  # [N]
    ends = indptr[anchor_node_ids + 1]  # [N]

    max_range = int(torch.max(ends - starts).item())

    # Sample all labels based on the CSR start/stop indices.
    # Creates "indices" for us to us, e.g [[0, 1], [2, 3]]
    ranges = starts.unsqueeze(1) + torch.arange(max_range)  # [N, max_range]
    # Clamp the ranges to be valid indices into `indices`.
    ranges.clamp_(min=0, max=ends.max().item() - 1)
    # Mask out the parts of "ranges" that are not applicable to the current label
    # filling out the rest with `PADDING_NODE`.
    mask = torch.arange(max_range) >= (ends - starts).unsqueeze(1)
    labels = torch.where(
        mask, torch.full_like(ranges, PADDING_NODE.item()), indices[ranges]
    )
    labels = torch.cat(
        [
            labels,
            torch.ones(extra_nodes_to_pad, max_range, dtype=torch.int64) * PADDING_NODE,
        ],
        dim=0,
    )
    return labels


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
