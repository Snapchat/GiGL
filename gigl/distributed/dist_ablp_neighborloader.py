from collections import abc
from dataclasses import dataclass
from itertools import count
from typing import Optional, Union

import torch
from graphlearn_torch.channel import SampleMessage
from graphlearn_torch.distributed import (
    MpDistSamplingWorkerOptions,
    RemoteDistSamplingWorkerOptions,
)
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

import gigl.distributed.utils
from gigl.common.logger import Logger
from gigl.distributed.base_dist_loader import BaseDistLoader
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_sampling_producer import DistSamplingProducer
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.sampler import (
    NEGATIVE_LABEL_METADATA_KEY,
    POSITIVE_LABEL_METADATA_KEY,
    ABLPNodeSamplerInput,
)
from gigl.distributed.sampler_options import (
    SamplerOptions,
    resolve_sampler_options,
)
from gigl.distributed.utils.neighborloader import (
    DatasetSchema,
    SamplingClusterSetup,
    extract_edge_type_metadata,
    extract_metadata,
    labeled_to_homogeneous,
    set_missing_features,
    shard_nodes_by_process,
    strip_label_edges,
)
from gigl.src.common.types.graph_data import (
    NodeType,  # TODO (mkolodner-sc): Change to use torch_geometric.typing
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    label_edge_type_to_message_passing_edge_type,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    reverse_edge_type,
    select_label_edge_types,
)
from gigl.utils.data_splitters import PADDING_NODE, get_labels_for_anchor_nodes
from gigl.utils.sampling import ABLPInputNodes

logger = Logger()


@dataclass(frozen=True)
class AnchorLabels:
    """ABLP labels for one edge type, stored as a flat (anchor, label) edge list.

    Each anchor can carry a different number of labels, so the natural shape is
    ragged. Rather than a ``dict[int, torch.Tensor]`` -- which needs padding to
    batch and a Python loop to read -- we keep two co-indexed ``long`` tensors so
    the loss can index straight into them with no per-anchor iteration. The data
    example below makes the layout concrete.

    Order within an anchor is deliberately left unspecified. The ABLP contrastive
    loss (:class:`gigl.nn.loss.RetrievalLoss`) scores every (anchor, label) pair
    independently and sums, so permuting the labels of an anchor leaves the loss
    unchanged. We therefore emit pairs in the order they fall out of the source
    ``[N_anchors, M]`` tensor (row-major, padding removed) and never pay to sort.
    The two index tensors are always co-indexed, which is the only invariant that
    matters.

    Empty anchors contribute no pairs at all. ``num_anchors`` is carried
    separately so :meth:`to_dict` can still emit a key for every anchor, even the
    ones that matched nothing.

    Dimension vocabulary (used throughout the label-remap code):

    - ``N_anchors`` -- anchor rows (rows of the source padded label tensor).
    - ``M`` -- padded label columns per anchor.
    - ``N_nodes`` -- nodes in the supervision local->global map.
    - ``K`` -- non-padding candidate labels (after dropping the ``-1`` pad, before
      membership filtering; still includes globals absent from the subgraph).
    - ``E`` -- surviving ``(anchor, label)`` pairs after membership filtering
      (``E <= K``); the length of ``anchor_index`` / ``label_index``.

    The ragged dict and this edge list hold the same labels in different
    containers (three anchors, two of which carry labels)::

        dict form        {0: [3], 1: [5, 7], 2: []}
        edge-list form   anchor_index = [0, 1, 1]   # [E] = 3
                         label_index  = [3, 5, 7]   # [E] = 3
                         num_anchors  = 3           # anchor 2 contributes no pair

    Example::

        >>> import torch
        >>> labels = AnchorLabels(
        ...     anchor_index=torch.tensor([0, 1, 1], dtype=torch.long),
        ...     label_index=torch.tensor([3, 5, 7], dtype=torch.long),
        ...     num_anchors=3,
        ... )
        >>> labels.to_dict()
        {0: tensor([3]), 1: tensor([5, 7]), 2: tensor([], dtype=torch.int64)}

    Args:
        anchor_index (torch.Tensor): ``[E]`` long tensor of local anchor rows.
        label_index (torch.Tensor): ``[E]`` long tensor of local label node ids.
        num_anchors (int): Total number of anchors ``N_anchors`` (rows of the
            source padded label tensor), including anchors with no labels.
    """

    anchor_index: torch.Tensor
    label_index: torch.Tensor
    num_anchors: int

    def to_dict(self) -> dict[int, torch.Tensor]:
        """Expand to the legacy ragged ``dict[int, torch.Tensor]`` form.

        Every anchor ``0..num_anchors-1`` receives a key; anchors with no labels
        map to an empty ``long`` tensor on the same device as ``label_index``.

        Returns:
            Mapping from anchor index to its 1-D ``long`` tensor of local label
            node ids.
        """
        counts = torch.bincount(self.anchor_index, minlength=self.num_anchors)
        per_anchor = torch.split(self.label_index, counts.tolist())
        return {anchor: per_anchor[anchor] for anchor in range(self.num_anchors)}


def _membership_remap(
    label_tensor: torch.Tensor,
    sorted_node: torch.Tensor,
    sort_perm: torch.Tensor,
    to_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Resolve one padded label tensor to a flat ``(anchor, label)`` pair stream.

    This is the actual global-id-to-local-index lookup; :class:`AnchorLabels` and
    :func:`edge_list_set_labels` just package what it returns. (See
    :class:`AnchorLabels` for the ``N_anchors`` / ``M`` / ``N_nodes`` / ``K`` / ``E``
    dimension vocabulary used below.)

    The lookup is a sorted-membership join rather than a per-anchor scan, which is
    what lets the loader remap labels without a Python loop over anchors: it trades
    an ``O(N_anchors * M * N_nodes)`` broadcast-compare for a single
    ``O(K log N_nodes)`` search.

    Worked example (generic ids)::

        node map (local -> global):  [40, 10, 30]      # N_nodes = 3
          sorted_node = [10, 30, 40], sort_perm = [1, 2, 0]
        label_tensor ([N_anchors=2, M=2], -1 = pad):
          [[30, -1],
           [40, 10]]

        step 0  flatten row-major, tag each entry with its anchor row, drop pad:
                flat        = [30, 40, 10]    # [K] = 3 candidates
                anchor_of_* = [ 0,  1,  1]    # [K]
        step 1  searchsorted(sorted_node, flat) -> [1, 2, 0]   # [K] positions
        step 2  keep exact members (sorted_node[pos] == flat): all 3 -> [E]
        step 3  sort_perm[pos] -> local index: [2, 0, 1]       # [E]
        result  anchor_index = [0, 1, 1], label_index = [2, 0, 1]   # [E] = 3

    Check by hand: g30 is local 2, g40 is local 0, g10 is local 1 -- matches. Here
    every candidate is a member so ``K == E``; the two differ once a global id is
    absent from the node map (step 2 drops it). The code names the step-3 output
    ``local_index``; :class:`AnchorLabels` stores that same tensor as ``label_index``.

    Because ``anchor_of_*`` is built row-major (step 0) and every mask preserves
    order, the result is already grouped by anchor (non-decreasing ``anchor_index``)
    with no argsort; order *within* an anchor is unspecified by contract, since the
    loss does not care (see :class:`AnchorLabels`).

    The lookup is only correct if ``sorted_node`` has unique values:
    :func:`torch.searchsorted` returns the left-most equal position, so a repeated
    global id would map every match to the same local index and silently drop the
    rest. GiGL ``node`` maps are unique by construction (one entry per subgraph
    node), so this holds in production; the ``__debug__`` assertion guards against
    misuse with a cheap adjacent-difference check on the already-sorted map.

    Args:
        label_tensor (torch.Tensor): ``[N_anchors, M]`` ``-1``-padded global
            label ids.
        sorted_node (torch.Tensor): ``[N_nodes]`` sorted values of the supervision
            node map (the ``values`` half of ``torch.sort``).
        sort_perm (torch.Tensor): ``[N_nodes]`` permutation from ``torch.sort``
            mapping sorted positions back to original local indices.
        to_device (torch.device): Device for the returned index tensors.

    Returns:
        Tuple ``(anchor_index, local_index, num_anchors)``. ``anchor_index`` and
        ``local_index`` are co-indexed ``[E]`` tensors grouped by anchor (empty when
        nothing matched); ``local_index`` becomes :attr:`AnchorLabels.label_index`.
        ``num_anchors == N_anchors == label_tensor.size(0)``.
    """
    num_anchors = int(label_tensor.size(0))
    num_nodes = int(sorted_node.size(0))
    empty = torch.empty(0, dtype=torch.long, device=to_device)
    if num_anchors == 0:
        return empty, empty, num_anchors

    num_labels = int(label_tensor.size(1))
    flat = label_tensor.reshape(-1)  # [N_anchors * M] before the pad mask
    # step 0 (see docstring example): tag each flattened entry with its anchor row.
    # Build on the label tensor's device: `anchor_of_entry` is indexed below by
    # `is_present` (derived from `label_tensor`).  On GPU, a CPU arange would
    # raise "indices should be either on cpu or on the same device as the indexed
    # tensor".  CPU-only unit tests cannot catch this; see the CUDA-gated test.
    anchor_of_entry = torch.arange(
        num_anchors, device=label_tensor.device
    ).repeat_interleave(num_labels)  # [N_anchors * M] before the pad mask

    # step 0 cont.: drop the -1 pad before any search so we never gather with a
    # sentinel.  This is the [N_anchors * M] -> [K] (candidate) reduction.
    is_present = flat != PADDING_NODE
    flat = flat[is_present]  # [K]
    anchor_of_entry = anchor_of_entry[is_present]  # [K]

    if num_nodes == 0 or flat.numel() == 0:
        return empty, empty, num_anchors

    if __debug__:
        # Precondition for step 1 (see docstring): `sorted_node` is already sorted,
        # so uniqueness is equivalent to being strictly increasing -- a cheap
        # adjacent-difference check, no re-sort.
        assert bool((sorted_node[1:] > sorted_node[:-1]).all()), (
            "vectorized label remap requires a unique node local->global map; "
            "duplicate global ids break the searchsorted membership lookup."
        )

    # step 1 (see docstring example): position of each candidate id in sorted_node.
    sorted_positions = torch.searchsorted(sorted_node, flat)  # [K]
    # searchsorted returns N_nodes for an id larger than every entry, which would
    # gather out of bounds at step 2; clamp it back into range.
    sorted_positions = sorted_positions.clamp_(max=num_nodes - 1)

    # step 2 (see docstring example): keep only true members (a neighboring entry
    # in the sorted array is not a match).  This is the [K] -> [E] filter.
    is_exact_match = sorted_node[sorted_positions] == flat  # [K] bool

    # step 3 (see docstring example): sorted position -> original local node index
    # via sort_perm (becomes AnchorLabels.label_index).
    local_index = sort_perm[sorted_positions][is_exact_match]  # [E]
    anchor_of_matched = anchor_of_entry[is_exact_match]  # [E]

    # Result rows stay grouped by anchor (step 0 tagging was row-major and the masks
    # preserve order), so no argsort is needed; within-anchor order is unspecified --
    # the ABLP loss is order-invariant (see AnchorLabels).
    return (
        anchor_of_matched.to(to_device).to(torch.long),
        local_index.to(to_device).to(torch.long),
        num_anchors,
    )


def edge_list_set_labels(
    node_local_to_global_by_type: dict[NodeType, torch.Tensor],
    positive_labels_by_edge_type: dict[EdgeType, torch.Tensor],
    negative_labels_by_edge_type: dict[EdgeType, torch.Tensor],
    to_device: torch.device,
) -> tuple[dict[EdgeType, AnchorLabels], dict[EdgeType, AnchorLabels]]:
    """Remap ABLP labels from global ids to local indices, as dense edge lists.

    This is the loader's single label-remap entry point. Sampling hands back
    labels as global node ids in ``[N_anchors, M]`` padded blocks; training needs
    them as local indices into the sampled subgraph. For each edge type this
    resolves that mapping and returns an :class:`AnchorLabels` edge list. Callers
    wanting the ragged ``dict[int, torch.Tensor]`` instead can expand each result
    with :meth:`AnchorLabels.to_dict`; the two are the same labels in a different
    container, and the loss treats them identically.

    Positive and negative labels are remapped the same way, against the same
    sorted node maps, so the work is shared via a memoized sort per supervision
    node type. A zero-anchor tensor is skipped rather than emitted as an empty
    entry: there are simply no anchors to label for that edge type in this batch,
    so no key is emitted for it.

    Correctness rests on each ``node`` local->global map having unique global ids;
    the membership lookup relies on it (see :func:`_membership_remap`). GiGL node
    maps satisfy this by construction.

    Args:
        node_local_to_global_by_type (dict[NodeType, torch.Tensor]): Per node
            type, a ``[N_nodes]`` tensor whose ``i``-th entry is the global id of
            local node ``i``. Global ids MUST be unique within each map.
        positive_labels_by_edge_type (dict[EdgeType, torch.Tensor]): Per
            positive-label edge type, a ``[N_anchors, M]`` ``-1``-padded tensor
            of global label ids.
        negative_labels_by_edge_type (dict[EdgeType, torch.Tensor]): As above,
            for negative-label edge types. May be empty.
        to_device (torch.device): Device for every output tensor.

    Returns:
        Tuple ``(y_positive, y_negative)``, each a
        ``dict[message_passing_edge_type, AnchorLabels]`` with no entry for an
        edge type that had no anchors this batch.
    """
    sorted_cache: dict[NodeType, tuple[torch.Tensor, torch.Tensor]] = {}

    def _sorted_for(node_type: NodeType) -> tuple[torch.Tensor, torch.Tensor]:
        if node_type not in sorted_cache:
            sorted_cache[node_type] = torch.sort(
                node_local_to_global_by_type[node_type]
            )
        return sorted_cache[node_type]

    def _remap(
        labels_by_edge_type: dict[EdgeType, torch.Tensor],
    ) -> dict[EdgeType, AnchorLabels]:
        # Supervision edge types are (anchor_type, relation, supervision_type), so
        # the supervision node type is index 2 (used below).
        supervision_node_type_index = 2
        output: dict[EdgeType, AnchorLabels] = {}
        for edge_type, label_tensor in labels_by_edge_type.items():
            # No anchors for this edge type this batch -> no key, not an empty one.
            if label_tensor.size(0) == 0:
                continue
            sorted_node, sort_perm = _sorted_for(edge_type[supervision_node_type_index])
            # Remap globals -> locals via the sorted-membership join (see the
            # labeled steps in _membership_remap).
            output[label_edge_type_to_message_passing_edge_type(edge_type)] = (
                AnchorLabels(
                    *_membership_remap(label_tensor, sorted_node, sort_perm, to_device)
                )
            )
        return output

    return _remap(positive_labels_by_edge_type), _remap(negative_labels_by_edge_type)


class DistABLPLoader(BaseDistLoader):
    # Counts instantiations of this class, per process.
    # This is needed so we can generate unique worker key for each instance, for graph store mode.
    # NOTE: This is per-class, not per-instance.
    _counter = count(0)

    def __init__(
        self,
        dataset: Union[DistDataset, RemoteDistDataset],
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
        input_nodes: Optional[
            Union[
                torch.Tensor,
                tuple[NodeType, torch.Tensor],
                # Graph Store mode inputs
                dict[int, ABLPInputNodes],
            ]
        ] = None,
        supervision_edge_type: Optional[Union[EdgeType, list[EdgeType]]] = None,
        num_workers: int = 1,
        batch_size: int = 1,
        pin_memory_device: Optional[torch.device] = None,
        worker_concurrency: int = 4,
        prefetch_size: Optional[int] = None,
        channel_size: str = "4GB",
        process_start_gap_seconds: float = 60.0,
        num_cpu_threads: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        with_weight: bool = False,
        sampler_options: Optional[SamplerOptions] = None,
        context: Optional[DistributedContext] = None,  # TODO: (svij) Deprecate this
        local_process_rank: Optional[int] = None,  # TODO: (svij) Deprecate this
        local_process_world_size: Optional[int] = None,  # TODO: (svij) Deprecate this
        non_blocking_transfers: bool = True,
        use_list_output: bool = False,
    ):
        """
        Neighbor loader for Anchor Based Link Prediction (ABLP) tasks.

        The dataset must *always* be heterogeneous here, since positive and
        negative labels are carried as separate edge types.

        By default, the loader will return {py:class} `torch_geometric.data.HeteroData` (heterogeneous) objects,
        but will return a {py:class}`torch_geometric.data.Data` (homogeneous) object if the dataset is "labeled homogeneous".

        The following fields may also be present (this describes the default
        `use_list_output=False` shape; see `use_list_output` below for the
        `AnchorLabels` edge-list alternative):
        - `y_positive`: `dict[int, torch.Tensor]` mapping from local anchor node id to a tensor of positive
                label node ids.
        - `y_negative`: (Optional) `dict[int, torch.Tensor]` mapping from local anchor node id to a tensor of negative
                label node ids. This will only be present if the supervision edge type has negative labels.


        NOTE: for both y_positive, and y_negative, the values represented in both the key and value of the dicts are
        the *local* node ids of the sampled nodes, not the global node ids.
        In order to get the global node ids, you can use the `node` field of the Data/HeteroData object.
        e.g. global_positive_node_id_labels = data.node[data.y_positive[local_anchor_node_id]].

        The underlying graph engine may also add the following fields to the output Data object:
            - num_sampled_nodes: If heterogeneous. a dictionary mapping from node type to the number of sampled nodes for that type, by hop.
            if homogeneous, a tensor the number of sampled nodes, by hop.
            - num_sampled_edges: If heterogeneous, a dictionary mapping from edge type to the number of sampled edges for that type, by hop.
            If homogeneous, a tensor denoting the number of sampled edges, by hop.

        Let's use the following homogeneous graph (https://is.gd/a8DK15) as an example:
            0 -> 1 [label="Positive example" color="green"]
            0 -> 2 [label="Negative example" color="red"]

            0 -> {3, 4}
            3 -> {5, 6}
            4 -> {7, 8}

            1 -> 9 # shouldn't be sampled
            2 -> 10 # shouldn't be sampled

        For sampling around node `0`, the fields on the output Data object will be:
            - `y_positive`: {0: torch.tensor([1])} # 1 is the only positive label for node 0
            - `y_negative`: {0: torch.tensor([2])} # 2 is the only negative label for node 0

        NOTE: both label fields will instead be `dict[EdgeType, dict[int, torch.Tensor]]` if multiple supervision edge types are provided.
        e.g. if there are supervision edge types: (a, to, b) and (a, to, c), then the label fields could be:
            - `y_positive`: {(a, to, b): {0: torch.tensor([1])}, (a, to, c): {0: torch.tensor([2])}}
            - `y_negative`: {(a, to, b): {0: torch.tensor([3])}, (a, to, c): {0: torch.tensor([4])}}

        With `use_list_output=True`, the labels arrive instead as an `AnchorLabels`
        edge-list (or `dict[EdgeType, AnchorLabels]` for several supervision edge
        types); see :class:`AnchorLabels` for its shape. The edge-list keeps the
        ragged per-anchor labels as flat tensors the loss can index directly, with
        no padding or per-anchor Python loop; within-anchor order is unspecified
        but the ABLP loss is order-invariant, and `AnchorLabels.to_dict()` recovers
        the dict form.

        Args:
            dataset (Union[DistDataset, RemoteDistDataset]): The dataset to sample from.
                If this is a `RemoteDistDataset`, then we are in "Graph Store" mode.
            num_neighbors (list[int] or dict[tuple[str, str, str], list[int]]):
                The number of neighbors to sample for each node in each iteration.
                If an entry is set to `-1`, all neighbors will be included.
                In heterogeneous graphs, may also take in a dictionary denoting
                the amount of neighbors to sample for each individual edge type.
                If ``KHopNeighborSamplerOptions`` is also provided, they must match.
            input_nodes: Indices of seed nodes to start sampling from.
                For Colocated mode: `torch.Tensor` or `tuple[NodeType, torch.Tensor]`.
                    If set to `None` for homogeneous settings, all nodes will be considered.
                    In heterogeneous graphs, this flag must be passed in as a tuple that holds
                    the node type and node indices.
                    NOTE: We intend to migrate colocated mode to have a similar input format to Graph Store mode in the future.
                    We want to do this so that users can easily control labels per anchor.
                For Graph Store mode: `dict[int, ABLPInputNodes]`
                    Maps server_rank to an ABLPInputNodes dataclass containing anchor nodes,
                    positive labels, and negative labels with explicit node type and edge type info.
                    This is the return type of `RemoteDistDataset.fetch_ablp_input()`.
            supervision_edge_type (Optional[Union[EdgeType, list[EdgeType]]]):
                The edge type(s) to use for supervision.
                For Colocated mode: Must be None iff the dataset is labeled homogeneous.
                    If set to a single EdgeType, the positive and negative labels will be stored in the `y_positive` and `y_negative` fields of the Data object.
                    If set to a list of EdgeTypes, the positive and negative labels will be stored in the `y_positive` and `y_negative` fields of the Data object,
                    with the key being the EdgeType. (default: `None`)
                For Graph Store mode: Must not be provided (must be None). The supervision edge types are
                    inferred from the label edge type keys in ABLPInputNodes.
            num_workers (int): How many workers to use (subprocesses to spwan) for
                    distributed neighbor sampling of the current process. (default: ``1``).
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            pin_memory_device (str, optional): The target device that the sampled
                results should be copied to. If set to ``None``, the device is inferred based off of
                (got by ``gigl.distributed.utils.device.get_available_device``). Which uses the
                local_process_rank and torch.cuda.device_count() to assign the device. If cuda is not available,
                the cpu device will be used. (default: ``None``).
            worker_concurrency (int): The max sampling concurrency for each sampling
                worker. Load testing has showed that setting worker_concurrency to 4 yields the best performance
                for sampling. Although, you may whish to explore higher/lower settings when performance tuning.
                (default: `4`).
            prefetch_size (Optional[int]): Max number of sampled messages to prefetch on the
                client side, per server. Only applies to Graph Store mode (remote workers).
                Lower values reduce server-side RPC thread contention when multiple loaders
                are active concurrently. (default: ``None``).
                If supplied and not it Graph Store mode, an error will be raised.
            channel_size (int or str): The shared-memory buffer size (bytes) allocated
                for the channel. Can be modified for performance tuning; a good starting point is: ``num_workers * 64MB``
                (default: "4GB").
            process_start_gap_seconds (float): Delay between each process for initializing neighbor loader
                in colocated mode. Each process sleeps ``local_rank * process_start_gap_seconds``
                before initializing. Only applies to colocated mode.
            num_cpu_threads (Optional[int]): Number of cpu threads PyTorch should use for CPU training/inference
                neighbor loading; on top of the per process parallelism.
                Defaults to `2` if set to `None` when using cpu training/inference.
            shuffle (bool): Whether to shuffle the input nodes. (default: ``False``).
            drop_last (bool): Whether to drop the last incomplete batch. (default: ``False``).
            with_weight (bool): Whether to use edge weights for neighbor sampling.
                Requires edge weights to have been provided via
                ``build_dataset(weight_edge_feat_name=...)`` during dataset construction.
                Defaults to ``False``.
            sampler_options (Optional[SamplerOptions]): Controls which sampler class is
                instantiated. Defaults to `KHopNeighborSamplerOptions`, which will use the num_neighbors argument
                to instantiate the sampler.
            context (deprecated - will be removed soon) (Optional[DistributedContext]): Distributed context information of the current process.
            local_process_rank (deprecated - will be removed soon) (int): The local rank of the current process within a node.
            local_process_world_size (deprecated - will be removed soon) (int): The total number of processes within a node.
            non_blocking_transfers (bool): If True (default), batch-transfers all
                sampled tensors to the target CUDA device using non-blocking copies
                before collation, which can overlap data transfer with computation
                when source tensors reside in pinned memory.  If False, the bulk
                transfer is skipped and GLT's default (blocking) device placement
                is used instead.
                See https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
                for background on pinned memory and non-blocking transfers.
            use_list_output (bool): Return labels as an ``AnchorLabels`` edge-list
                (or ``dict[EdgeType, AnchorLabels]`` for multiple supervision edge
                types) instead of the ragged ``dict[anchor_local_index,
                torch.Tensor]``. The edge-list lets the loss read the co-indexed
                ``y.label_index`` and ``query_idx[y.anchor_index]`` (both ``[E]``)
                directly; see :class:`AnchorLabels` for the shape and the ``[E]``
                vocabulary. Defaults to ``False`` (the backward-compatible ragged
                dict).
        """

        # Set self._shutdowned right away, that way if we throw here, and __del__ is called,
        # then we can properly clean up and don't get extraneous error messages.
        self._shutdowned = True
        self._use_list_output = use_list_output

        sampler_options = resolve_sampler_options(num_neighbors, sampler_options)

        # Determine sampling cluster setup based on dataset type
        if isinstance(dataset, RemoteDistDataset):
            self._sampling_cluster_setup = SamplingClusterSetup.GRAPH_STORE
            if supervision_edge_type is not None:
                raise ValueError(
                    "supervision_edge_type must not be provided when using Graph Store mode. "
                    "The supervision edge types are inferred from the ABLPInputNodes label keys in input_nodes."
                )
            # self._supervision_edge_types will be set in _setup_for_graph_store
        else:
            self._sampling_cluster_setup = SamplingClusterSetup.COLOCATED
            if supervision_edge_type is None:
                self._supervision_edge_types: list[EdgeType] = [
                    DEFAULT_HOMOGENEOUS_EDGE_TYPE
                ]
            elif isinstance(supervision_edge_type, list):
                if not supervision_edge_type:
                    raise ValueError(
                        "supervision_edge_type must be a non-empty list when providing multiple supervision edge types."
                    )
                self._supervision_edge_types = supervision_edge_type
            else:
                self._supervision_edge_types = [supervision_edge_type]
            if prefetch_size is not None:
                raise ValueError(
                    f"prefetch_size must be None when using Colocated mode, received {prefetch_size}"
                )
        logger.info(f"Sampling cluster setup: {self._sampling_cluster_setup.value}")

        del supervision_edge_type
        self._instance_count = next(self._counter)

        # Resolve distributed context
        runtime = BaseDistLoader.resolve_runtime(
            context, local_process_rank, local_process_world_size
        )
        del context, local_process_rank, local_process_world_size

        BaseDistLoader.validate_for_weighted_sampling(
            with_weight, dataset, sampler_options
        )

        device = (
            pin_memory_device
            if pin_memory_device
            else gigl.distributed.utils.get_available_device(
                local_process_rank=runtime.local_rank
            )
        )
        self.to_device = device

        # Mode-specific setup
        if self._sampling_cluster_setup == SamplingClusterSetup.COLOCATED:
            assert isinstance(dataset, DistDataset), (
                "When using colocated mode, dataset must be a DistDataset."
            )
            # Validate input_nodes type for colocated mode
            if isinstance(input_nodes, dict):
                raise ValueError(
                    f"When using Colocated mode, input_nodes must be of type "
                    f"(torch.Tensor | tuple[NodeType, torch.Tensor] | None), "
                    f"received {type(input_nodes)}"
                )
            setup_info = self._setup_for_colocated(
                input_nodes=input_nodes,
                dataset=dataset,
                local_rank=runtime.local_rank,
                local_world_size=runtime.local_world_size,
                device=device,
                master_ip_address=runtime.master_ip_address,
                node_rank=runtime.node_rank,
                node_world_size=runtime.node_world_size,
                num_workers=num_workers,
                worker_concurrency=worker_concurrency,
                channel_size=channel_size,
                num_cpu_threads=num_cpu_threads,
            )
            sampler_input: Union[ABLPNodeSamplerInput, list[ABLPNodeSamplerInput]] = (
                setup_info[0]
            )
            worker_options: Union[
                MpDistSamplingWorkerOptions, RemoteDistSamplingWorkerOptions
            ] = setup_info[1]
            dataset_schema: DatasetSchema = setup_info[2]
            backend_key: Optional[str] = None
        else:  # Graph Store mode
            assert isinstance(dataset, RemoteDistDataset), (
                "When using Graph Store mode, dataset must be a RemoteDistDataset."
            )
            # Validate input_nodes type for Graph Store mode
            if not isinstance(input_nodes, dict):
                raise ValueError(
                    f"When using Graph Store mode, input_nodes must be of type "
                    f"dict[int, ABLPInputNodes], "
                    f"received {type(input_nodes)}"
                )
            if prefetch_size is None:
                logger.info(f"prefetch_size is not provided, using default of 4")
                prefetch_size = 4
            (
                sampler_input,
                worker_options,
                dataset_schema,
                backend_key,
            ) = self._setup_for_graph_store(
                input_nodes=input_nodes,  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
                dataset=dataset,
                num_workers=num_workers,
                worker_concurrency=worker_concurrency,
                channel_size=channel_size,
                prefetch_size=prefetch_size,
            )

        # Cleanup temporary process group if needed
        if (
            runtime.should_cleanup_distributed_context
            and torch.distributed.is_initialized()
        ):
            logger.info(
                f"Cleaning up process group as it was initialized inside {self.__class__.__name__}.__init__."
            )
            torch.distributed.destroy_process_group()

        # Create SamplingConfig (with patched fanout)
        sampling_config = BaseDistLoader.create_sampling_config(
            num_neighbors=num_neighbors,
            dataset_schema=dataset_schema,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            with_weight=with_weight,
        )

        producer: Optional[DistSamplingProducer] = None
        if self._sampling_cluster_setup == SamplingClusterSetup.COLOCATED:
            assert isinstance(dataset, DistDataset)
            assert isinstance(worker_options, MpDistSamplingWorkerOptions)
            producer = BaseDistLoader.create_mp_producer(
                dataset=dataset,
                sampler_input=sampler_input,
                sampling_config=sampling_config,
                worker_options=worker_options,
                sampler_options=sampler_options,
            )

        # Call base class — handles metadata storage and connection initialization
        # (including staggered init for colocated mode).
        super().__init__(
            dataset=dataset,
            sampler_input=sampler_input,
            dataset_schema=dataset_schema,
            worker_options=worker_options,
            sampling_config=sampling_config,
            device=device,
            runtime=runtime,
            producer=producer,
            sampler_options=sampler_options,
            backend_key=backend_key,
            process_start_gap_seconds=process_start_gap_seconds,
            non_blocking_transfers=non_blocking_transfers,
        )

    def _setup_for_colocated(
        self,
        input_nodes: Optional[
            Union[
                torch.Tensor,
                tuple[NodeType, torch.Tensor],
            ]
        ],
        dataset: DistDataset,
        local_rank: int,
        local_world_size: int,
        device: torch.device,
        master_ip_address: str,
        node_rank: int,
        node_world_size: int,
        num_workers: int,
        worker_concurrency: int,
        channel_size: str,
        num_cpu_threads: Optional[int],
    ) -> tuple[ABLPNodeSamplerInput, MpDistSamplingWorkerOptions, DatasetSchema]:
        """
        Setup method for colocated (non-Graph Store) mode.

        Args:
            input_nodes: Input nodes for sampling (tensor or tuple of node type and tensor).
            dataset: The DistDataset to sample from.
            local_rank: Local rank of the current process.
            local_world_size: Total number of processes on this machine.
            device: Target device for sampled data.
            master_ip_address: IP address of the master node.
            node_rank: Rank of the current machine.
            node_world_size: Total number of machines.
            num_workers: Number of sampling workers.
            worker_concurrency: Max sampling concurrency per worker.
            channel_size: Size of shared memory channel.
            num_cpu_threads: Number of CPU threads for PyTorch.

        Returns:
            Tuple of (ABLPNodeSamplerInput, MpDistSamplingWorkerOptions, DatasetSchema).
        """
        # Validate input format - should not be Graph Store format
        if isinstance(input_nodes, abc.Mapping):
            raise ValueError(
                f"When using Colocated mode, input_nodes must be of type (torch.Tensor | tuple[NodeType, torch.Tensor]), "
                f"received {type(input_nodes)}"
            )
        elif isinstance(input_nodes, tuple) and isinstance(input_nodes[1], abc.Mapping):
            raise ValueError(
                f"When using Colocated mode, input_nodes must be of type (torch.Tensor | tuple[NodeType, torch.Tensor]), "
                f"received tuple with second element of type {type(input_nodes[1])}"
            )

        if not isinstance(dataset.graph, abc.Mapping):
            raise ValueError(
                f"The dataset must be heterogeneous for ABLP. Received dataset with graph of type: {type(dataset.graph)}"
            )

        is_homogeneous_with_labeled_edge_type: bool = True
        if isinstance(input_nodes, tuple):
            if self._supervision_edge_types == [DEFAULT_HOMOGENEOUS_EDGE_TYPE]:
                raise ValueError(
                    "When using heterogeneous ABLP, you must provide supervision_edge_types."
                )
            is_homogeneous_with_labeled_edge_type = False
            anchor_node_type, anchor_node_ids = input_nodes
            # TODO (mkolodner-sc): We currently assume supervision edges are directed outward, revisit in future if
            # this assumption is no longer valid and/or is too opinionated
            for supervision_edge_type in self._supervision_edge_types:
                assert supervision_edge_type[0] == anchor_node_type, (
                    f"Label EdgeType are currently expected to be provided in outward edge direction as tuple (`anchor_node_type`,`relation`,`supervision_node_type`), \
                    got supervision edge type {supervision_edge_type} with anchor node type {anchor_node_type}"
                )
            if dataset.edge_dir == "in":
                self._supervision_edge_types = [
                    reverse_edge_type(supervision_edge_type)
                    for supervision_edge_type in self._supervision_edge_types
                ]
        elif isinstance(input_nodes, torch.Tensor):
            if self._supervision_edge_types != [DEFAULT_HOMOGENEOUS_EDGE_TYPE]:
                raise ValueError(
                    f"Expected supervision edge type to be None for homogeneous input nodes, got {self._supervision_edge_types}"
                )
            anchor_node_ids = input_nodes
            anchor_node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
        elif input_nodes is None:
            if dataset.node_ids is None:
                raise ValueError(
                    "Dataset must have node ids if input_nodes are not provided."
                )
            if isinstance(dataset.node_ids, abc.Mapping):
                raise ValueError(
                    f"input_nodes must be provided for heterogeneous datasets, received node_ids of type: {dataset.node_ids.keys()}"
                )
            if self._supervision_edge_types != [DEFAULT_HOMOGENEOUS_EDGE_TYPE]:
                raise ValueError(
                    f"Expected supervision edge type to be None for homogeneous input nodes, got {self._supervision_edge_types}"
                )
            anchor_node_ids = dataset.node_ids
            anchor_node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
        else:
            raise ValueError(f"Unexpected input_nodes type: {type(input_nodes)}")

        missing_edge_types = set(self._supervision_edge_types) - set(
            dataset.graph.keys()
        )
        if missing_edge_types:
            raise ValueError(
                f"Missing edge types in dataset: {missing_edge_types}. Edge types in dataset: {dataset.graph.keys()}"
            )

        # Type narrowing - anchor_node_ids is always a Tensor in colocated mode
        assert isinstance(anchor_node_ids, torch.Tensor)

        if len(anchor_node_ids.shape) != 1:
            raise ValueError(
                f"input_nodes must be a 1D tensor, got {anchor_node_ids.shape}."
            )

        curr_process_nodes = shard_nodes_by_process(
            input_nodes=anchor_node_ids,
            local_process_rank=local_rank,
            local_process_world_size=local_world_size,
        )

        self._positive_label_edge_types: list[EdgeType] = []
        self._negative_label_edge_types: list[EdgeType] = []
        positive_labels_by_label_edge_type: dict[EdgeType, torch.Tensor] = {}
        negative_labels_by_label_edge_type: dict[EdgeType, torch.Tensor] = {}
        for supervision_edge_type in self._supervision_edge_types:
            (
                positive_label_edge_type,
                negative_label_edge_type,
            ) = select_label_edge_types(supervision_edge_type, dataset.graph.keys())
            self._positive_label_edge_types.append(positive_label_edge_type)
            if negative_label_edge_type is not None:
                self._negative_label_edge_types.append(negative_label_edge_type)

            positive_labels, negative_labels = get_labels_for_anchor_nodes(
                dataset=dataset,
                node_ids=curr_process_nodes,
                positive_label_edge_type=positive_label_edge_type,
                negative_label_edge_type=negative_label_edge_type,
                max_labels_per_anchor_node=dataset.max_labels_per_anchor_node,
            )
            positive_labels_by_label_edge_type[positive_label_edge_type] = (
                positive_labels
            )
            if negative_label_edge_type is not None and negative_labels is not None:
                negative_labels_by_label_edge_type[negative_label_edge_type] = (
                    negative_labels
                )

        sampler_input = ABLPNodeSamplerInput(
            node=curr_process_nodes,
            input_type=anchor_node_type,
            positive_label_by_edge_types=positive_labels_by_label_edge_type,
            negative_label_by_edge_types=negative_labels_by_label_edge_type,
        )

        BaseDistLoader.initialize_colocated_sampling_worker(
            local_rank=local_rank,
            local_world_size=local_world_size,
            node_rank=node_rank,
            node_world_size=node_world_size,
            master_ip_address=master_ip_address,
            device=device,
            num_cpu_threads=num_cpu_threads,
        )

        # Sets up worker options for the dataloader
        dist_sampling_ports = gigl.distributed.utils.get_free_ports_from_master_node(
            num_ports=local_world_size
        )
        dist_sampling_port_for_current_rank = dist_sampling_ports[local_rank]
        worker_options = BaseDistLoader.create_colocated_worker_options(
            dataset_num_partitions=dataset.num_partitions,
            num_workers=num_workers,
            worker_concurrency=worker_concurrency,
            master_ip_address=master_ip_address,
            master_port=dist_sampling_port_for_current_rank,
            channel_size=channel_size,
            pin_memory=device.type == "cuda",
        )

        edge_types = list(dataset.graph.keys())

        return (
            sampler_input,
            worker_options,
            DatasetSchema(
                is_homogeneous_with_labeled_edge_type=is_homogeneous_with_labeled_edge_type,
                edge_types=edge_types,
                node_feature_info=dataset.node_feature_info,
                edge_feature_info=dataset.edge_feature_info,
                edge_dir=dataset.edge_dir,
            ),
        )

    def _setup_for_graph_store(
        self,
        input_nodes: dict[int, ABLPInputNodes],
        dataset: RemoteDistDataset,
        num_workers: int,
        worker_concurrency: int,
        channel_size: str,
        prefetch_size: int,
    ) -> tuple[
        list[ABLPNodeSamplerInput],
        RemoteDistSamplingWorkerOptions,
        DatasetSchema,
        str,
    ]:
        """
        Setup method for Graph Store mode.

        Args:
            input_nodes: ABLP input from RemoteDistDataset.fetch_ablp_input().
                Maps server_rank to ABLPInputNodes containing anchor nodes, positive/negative
                labels with explicit node type and edge type information.
            dataset: The RemoteDistDataset to sample from.
            num_workers: Number of sampling workers.
            worker_concurrency: Max sampling concurrency per worker.
            channel_size: Size of the remote shared-memory buffer.
            prefetch_size: Max prefetched sampled messages per server on client side.

        Returns:
            Tuple of (list[ABLPNodeSamplerInput], RemoteDistSamplingWorkerOptions,
            DatasetSchema, backend_key).
        """
        node_feature_info = dataset.fetch_node_feature_info()
        edge_feature_info = dataset.fetch_edge_feature_info()
        edge_types = dataset.fetch_edge_types()
        compute_rank = torch.distributed.get_rank()
        backend_key = f"dist_ablp_loader_{self._instance_count}"
        worker_key = f"{backend_key}_compute_rank_{compute_rank}"
        logger.info(f"rank: {compute_rank}, worker_key: {worker_key}")
        worker_options = BaseDistLoader.create_graph_store_worker_options(
            dataset=dataset,
            worker_key=worker_key,
            num_workers=num_workers,
            worker_concurrency=worker_concurrency,
            channel_size=channel_size,
            prefetch_size=prefetch_size,
        )
        logger.info(
            f"Rank {torch.distributed.get_rank()}! init for sampling rpc: "
            f"tcp://{worker_options.master_addr}:{worker_options.master_port}"
        )

        # Validate server ranks
        servers = input_nodes.keys()
        if len(servers) > 0:
            if (
                max(servers) >= dataset.cluster_info.num_storage_nodes
                or min(servers) < 0
            ):
                raise ValueError(
                    f"When using Graph Store mode, the server ranks must be in range "
                    f"[0, {dataset.cluster_info.num_storage_nodes}), "
                    f"received inputs for servers: {list(servers)}"
                )

        # Extract node type and label edge types from the ABLPInputNodes dataclass.
        # All entries should have the same anchor_node_type and edge type keys.
        first_input = next(iter(input_nodes.values()))
        input_type = first_input.anchor_node_type
        is_homogeneous_with_labeled_edge_type = (
            input_type == DEFAULT_HOMOGENEOUS_NODE_TYPE
        )

        # Extract supervision edge types and derive label edge types from the
        # ABLPInputNodes.labels dict (keyed by supervision edge type).
        self._supervision_edge_types = list(first_input.labels.keys())
        has_negatives = False
        for ablp_input in input_nodes.values():
            for maybe_negative_labels in ablp_input.labels.values():
                if maybe_negative_labels is not None:
                    has_negatives = True
                    break

        self._positive_label_edge_types = [
            message_passing_to_positive_label(et) for et in self._supervision_edge_types
        ]
        self._negative_label_edge_types = (
            [
                message_passing_to_negative_label(et)
                for et in self._supervision_edge_types
            ]
            if has_negatives
            else []
        )

        # Graph Store mode currently only supports a single supervision edge type,
        # so the labels dict must have exactly 1 entry.
        if len(self._supervision_edge_types) != 1:
            raise ValueError(
                f"Graph Store mode currently only supports a single supervision edge type, "
                f"but ABLPInputNodes.labels has {len(self._supervision_edge_types)} "
                f"entries: {self._supervision_edge_types}"
            )

        logger.info(f"Positive label edge types: {self._positive_label_edge_types}")
        logger.info(f"Negative label edge types: {self._negative_label_edge_types}")

        # Convert from ABLPInputNodes to list of ABLPNodeSamplerInput (one per server)
        input_data: list[ABLPNodeSamplerInput] = []
        for server_rank in range(dataset.cluster_info.num_storage_nodes):
            positive_label_by_edge_type: dict[EdgeType, torch.Tensor] = {}
            negative_label_by_edge_type: dict[EdgeType, torch.Tensor] = {}
            if server_rank in input_nodes:
                ablp_input_nodes = input_nodes[server_rank]
                anchors = ablp_input_nodes.anchor_nodes
                for supervision_edge_type, (
                    positive_labels,
                    negative_labels,
                ) in ablp_input_nodes.labels.items():
                    positive_label_by_edge_type[
                        message_passing_to_positive_label(supervision_edge_type)
                    ] = positive_labels
                    if negative_labels is not None:
                        negative_label_by_edge_type[
                            message_passing_to_negative_label(supervision_edge_type)
                        ] = negative_labels
            else:
                # Empty input for servers with no data for this rank
                anchors = torch.empty(0, dtype=torch.long)
                positive_label_by_edge_type = {
                    et: torch.empty(0, 0, dtype=torch.long)
                    for et in self._positive_label_edge_types
                }
                if has_negatives:
                    negative_label_by_edge_type = {
                        et: torch.empty(0, 0, dtype=torch.long)
                        for et in self._negative_label_edge_types
                    }

            logger.info(
                f"Rank: {torch.distributed.get_rank()}! Building ABLPNodeSamplerInput for server rank: {server_rank} "
                f"with input type: {input_type}. anchors: {anchors.shape}, "
                f"positive_labels edge types: {list(positive_label_by_edge_type.keys())}, "
                f"negative_labels edge types: {list(negative_label_by_edge_type.keys())}"
            )
            ablp_input = ABLPNodeSamplerInput(
                node=anchors,
                input_type=input_type,
                positive_label_by_edge_types=positive_label_by_edge_type,
                negative_label_by_edge_types=negative_label_by_edge_type,
            )
            input_data.append(ablp_input)

        return (
            input_data,
            worker_options,
            DatasetSchema(
                is_homogeneous_with_labeled_edge_type=is_homogeneous_with_labeled_edge_type,
                edge_types=edge_types,
                node_feature_info=node_feature_info,
                edge_feature_info=edge_feature_info,
                edge_dir=dataset.fetch_edge_dir(),
            ),
            backend_key,
        )

    def _set_labels(
        self,
        data: Union[Data, HeteroData],
        positive_labels_by_label_edge_type: dict[EdgeType, torch.Tensor],
        negative_labels_by_label_edge_type: dict[EdgeType, torch.Tensor],
    ) -> Union[Data, HeteroData]:
        """Attach ABLP labels to the collated graph, remapped to subgraph-local indices.

        This is the collation hook that turns the sampler's global-id labels into the
        ``y_positive`` / ``y_negative`` fields downstream training reads.

        The actual remap is delegated to :func:`edge_list_set_labels` (the single
        kernel): with ``use_list_output`` the labels are attached as an
        :class:`AnchorLabels` edge list, otherwise expanded to the ragged
        ``dict[anchor_local_index, torch.Tensor]`` via :meth:`AnchorLabels.to_dict`.
        Both are the same labels in a different container.

        The supervision edge type is an internal sampling artifact, so it is stripped
        before return and never appears on the output object.

        ``y_positive`` / ``y_negative`` collapse to a single value when there is one
        supervision edge type, or a ``dict[EdgeType, ...]`` for several; see
        :meth:`DistABLPLoader.__init__` for the full shape contract.

        Args:
            data (Union[Data, HeteroData]): Graph to attach labels to.
            positive_labels_by_label_edge_type (dict[EdgeType, torch.Tensor]): Per
                positive-label edge type, a ``[N_anchors, M]`` tensor whose ``i``-th
                row holds the global label ids of the ``i``-th anchor.
            negative_labels_by_label_edge_type (dict[EdgeType, torch.Tensor]): As
                above, for negative-label edge types.

        Returns:
            Union[Data, HeteroData]: The same object with the supervision edge fields
            stripped and ``y_positive`` (and ``y_negative`` when present) attached.

        Raises:
            ValueError: If no positive labels are found in ``data``.
        """
        # node_type_to_local_node_to_global_node[t][i]: global id of local node i;
        # each value tensor is [N_nodes] for its node type.
        node_type_to_local_node_to_global_node: dict[NodeType, torch.Tensor] = {}
        if isinstance(data, HeteroData):
            for e_type in self._supervision_edge_types:
                node_type_to_local_node_to_global_node[e_type[0]] = data[e_type[0]].node
                node_type_to_local_node_to_global_node[e_type[2]] = data[e_type[2]].node
        else:
            node_type_to_local_node_to_global_node[DEFAULT_HOMOGENEOUS_NODE_TYPE] = (
                data.node
            )
        # The edge-list kernel is the single remap path; the ragged dict is just
        # one view of it (AnchorLabels.to_dict), so when the caller wants the dict
        # we expand here rather than maintaining a second kernel. Both forms feed
        # an order-invariant contrastive loss, so the choice is purely about the
        # consumer's preferred shape.
        output_positive_labels, output_negative_labels = edge_list_set_labels(
            node_local_to_global_by_type=node_type_to_local_node_to_global_node,
            positive_labels_by_edge_type=positive_labels_by_label_edge_type,
            negative_labels_by_edge_type=negative_labels_by_label_edge_type,
            to_device=self.to_device,
        )
        if not self._use_list_output:
            output_positive_labels = {
                et: anchor_labels.to_dict()
                for et, anchor_labels in output_positive_labels.items()
            }
            output_negative_labels = {
                et: anchor_labels.to_dict()
                for et, anchor_labels in output_negative_labels.items()
            }
        if not output_positive_labels:
            raise ValueError("No positive labels were found in the data!")
        elif len(output_positive_labels) == 1:
            data.y_positive = next(iter(output_positive_labels.values()))
        else:
            data.y_positive = output_positive_labels

        if len(output_negative_labels) == 1:
            data.y_negative = next(iter(output_negative_labels.values()))
        elif len(output_negative_labels) > 0:
            data.y_negative = output_negative_labels
        return data

    def _collate_fn(self, msg: SampleMessage) -> Union[Data, HeteroData]:
        # extract_metadata separates #META. keys from the message to work
        # around a GLT bug in to_hetero_data.  extract_edge_type_metadata then
        # pulls out labels by prefix.
        # TODO (mkolodner-sc): Remove the need to extract metadata once GLT's `to_hetero_data` function is fixed
        metadata, stripped_msg = extract_metadata(msg, self.to_device)

        data = super()._collate_fn(stripped_msg)

        data = set_missing_features(
            data=data,
            node_feature_info=self._node_feature_info,
            edge_feature_info=self._edge_feature_info,
            device=self.to_device,
        )

        matched, metadata = extract_edge_type_metadata(
            metadata=metadata,
            prefixes=[POSITIVE_LABEL_METADATA_KEY, NEGATIVE_LABEL_METADATA_KEY],
        )
        positive_labels = matched[POSITIVE_LABEL_METADATA_KEY]
        negative_labels = matched[NEGATIVE_LABEL_METADATA_KEY]
        # When edge_dir="in", GLT internally swaps src/dst on all edge types during sampling,
        # so the sampler encodes label edge types in their reversed (incoming) form.
        # We reverse them back here to restore the original outward edge type that
        # _set_labels and downstream code expect.
        if self.edge_dir == "in":
            positive_labels = {
                reverse_edge_type(et): v for et, v in positive_labels.items()
            }
            negative_labels = {
                reverse_edge_type(et): v for et, v in negative_labels.items()
            }

        if isinstance(data, HeteroData):
            data = strip_label_edges(data)
        if self._is_homogeneous_with_labeled_edge_type:
            if len(self._supervision_edge_types) != 1:
                raise ValueError(
                    f"Expected 1 supervision edge type, got {len(self._supervision_edge_types)}"
                )
            data = labeled_to_homogeneous(self._supervision_edge_types[0], data)

        data = self._set_labels(data, positive_labels, negative_labels)

        data, metadata = self._apply_ppr_outputs(data, metadata)

        # Attach any remaining metadata (e.g. custom user-defined keys) directly onto the
        # data object so downstream code can access them via attribute lookup.
        for key, value in metadata.items():
            data[key] = value
        return data
