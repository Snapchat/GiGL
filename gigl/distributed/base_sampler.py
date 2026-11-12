import asyncio
import queue
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Union

import torch
from graphlearn_torch.channel import ChannelBase, SampleMessage
from graphlearn_torch.data import Feature, Graph
from graphlearn_torch.distributed import DistDataset
from graphlearn_torch.distributed import DistNeighborSampler as GLTDistNeighborSampler
from graphlearn_torch.distributed.dist_feature import DistFeature
from graphlearn_torch.distributed.dist_graph import DistGraph
from graphlearn_torch.distributed.dist_neighbor_sampler import (
    RpcSamplingCallee,
    RpcSubGraphCallee,
)
from graphlearn_torch.distributed.event_loop import (
    ConcurrentEventLoop,
    wrap_torch_future,
)
from graphlearn_torch.distributed.rpc import (
    RpcDataPartitionRouter,
    rpc_register,
    rpc_sync_data_partitions,
)
from graphlearn_torch.partition import RangePartitionBook
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    NeighborOutput,
    NeighborSampler,
    NodeSamplerInput,
    SamplerOutput,
)
from graphlearn_torch.typing import EdgeType, NodeType, NumNeighbors, as_str
from graphlearn_torch.utils import (
    ensure_device,
    get_available_device,
    reverse_edge_type,
)
from jaxtyping import Int64

from gigl.common.logger import Logger
from gigl.distributed.sampler import (
    EDGE_PACKED_FEATURES_METADATA_KEY,
    NEGATIVE_LABEL_METADATA_KEY,
    NODE_PACKED_FEATURES_METADATA_KEY,
    POSITIVE_LABEL_METADATA_KEY,
    ABLPNodeSamplerInput,
)
from gigl.distributed.utils.fused_feature import (
    FusedDistFeature,
    build_fused_dist_feature,
)
from gigl.distributed.utils.sampling_errors import (
    SAMPLING_ERROR_KEY,
    encode_sampling_error,
)
from gigl.utils.csr import CompactTopology, get_row_offset, to_local_rows
from gigl.utils.data_splitters import PADDING_NODE

logger = Logger()


def _stable_unique_preserve_order(nodes: torch.Tensor) -> torch.Tensor:
    """Return unique 1-D values while preserving first-occurrence order.

    Args:
        nodes: A 1-D tensor of node IDs (may contain duplicates).

    Returns:
        A 1-D tensor of unique node IDs in first-occurrence order.

    Raises:
        ValueError: If ``nodes`` is not 1-D.
    """
    if nodes.dim() != 1:
        raise ValueError(
            f"Expected a 1-D tensor of node ids, got shape {tuple(nodes.shape)}."
        )
    if nodes.numel() <= 1:
        return nodes

    unique_nodes, inverse = torch.unique(nodes, sorted=False, return_inverse=True)
    first_positions = torch.full(
        (unique_nodes.numel(),),
        fill_value=nodes.numel(),
        dtype=torch.long,
        device=nodes.device,
    )
    positions = torch.arange(nodes.numel(), device=nodes.device)
    first_positions.scatter_reduce_(
        0,
        inverse,
        positions,
        reduce="amin",
        include_self=True,
    )
    stable_order = torch.argsort(first_positions)
    return unique_nodes[stable_order]


@dataclass
class SampleLoopInputs:
    """Inputs prepared for the neighbor sampling loop in _sample_from_nodes.

    This dataclass holds the processed inputs that are passed to the core
    sampling loop. It allows _prepare_sample_loop_inputs to customize what nodes
    are sampled from and what metadata is attached to the output, without
    duplicating the sampling loop logic.

    Attributes:
        nodes_to_sample: For homogeneous graphs, a tensor of node IDs. For
            heterogeneous graphs, a dict mapping node types to tensors.
            For ABLP, this also includes supervision nodes (positive/negative labels).
        metadata: Metadata dict to attach to the sampler output (e.g., label tensors).
    """

    nodes_to_sample: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    metadata: dict[str, torch.Tensor]


def _stores_only_owned_rows(data: DistDataset) -> bool:
    """Whether ``data`` was built with each rank's topology holding only the rows it owns.

    GiGL's compact build (:class:`gigl.utils.csr.CompactTopology`) under a range partition book
    stores rows from the rank's range start, as recorded in ``row_offset``.
    This holds on every rank, including rank 0, whose offset is 0, so all ranks agree.
    Graphs that fall back to GLT's build, and tensor partition books, keep a row per id.

    Args:
        data: The dataset the sampler reads.

    Returns:
        bool: Whether the topologies store only the owned rows.
    """
    if isinstance(data.node_pb, dict):
        books = list(data.node_pb.values())
    else:
        books = [data.node_pb]
    graphs: list[Graph]
    if isinstance(data.graph, dict):
        graphs = list(data.graph.values())
    else:
        graphs = [data.graph]
    range_partitioned = any(isinstance(book, RangePartitionBook) for book in books)
    compact = any(isinstance(graph.topo, CompactTopology) for graph in graphs)
    return range_partitioned and compact


class _GlobalIdNeighborSampler(NeighborSampler):
    """GLT's ``NeighborSampler``, taking global seed ids.

    Under range partitioning a rank's topology stores rows only for the ids the rank owns (see
    ``row_offset`` of :class:`gigl.utils.csr.CompactTopology`), but GLT's kernels index ``indptr``
    by the seed id.
    ``sample_one_hop`` translates seeds to local rows first, on every topology: with a row offset
    of 0 the translation only maps ids past the last row to the row count.
    Neighbor ids and edge ids are stored as global ids, so the output needs no translation.

    Negative sampling and subgraph extraction read ``indptr`` without ``sample_one_hop``, so they
    raise when the topologies store only the owned rows.

    Args:
        *args: Positional arguments of GLT's ``NeighborSampler``.
        stores_only_owned_rows: Whether the topologies store only the rows the rank owns, as
            :func:`_stores_only_owned_rows` returns.
        **kwargs: Keyword arguments of GLT's ``NeighborSampler``.
    """

    def __init__(self, *args, stores_only_owned_rows: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stores_only_owned_rows = stores_only_owned_rows

    def sample_one_hop(
        self,
        input_seeds: torch.Tensor,
        req_num: int,
        etype: Optional[EdgeType] = None,
    ) -> NeighborOutput:
        """Sample one hop from global seed ids.

        Args:
            input_seeds: Global ids of the nodes to sample from.
            req_num: The number of neighbors to sample per seed.
            etype: The edge type to sample, or ``None`` on a homogeneous graph.

        Returns:
            NeighborOutput: The sampled neighbors, as GLT's ``sample_one_hop`` returns them.
        """
        graph = self.graph if etype is None else self.graph[etype]
        local_seeds = to_local_rows(
            input_seeds, get_row_offset(graph.topo), graph.topo.row_count
        )
        return super().sample_one_hop(local_seeds, req_num, etype)

    def lazy_init_neg_sampler(self) -> None:
        """Build GLT's negative sampler when negative sampling is on.

        Raises:
            ValueError: If negative sampling is on and the topologies store only the owned rows.
        """
        if self.with_neg and self._stores_only_owned_rows:
            raise ValueError(
                "Negative sampling is not supported on a range-partitioned compact topology"
            )
        super().lazy_init_neg_sampler()

    def lazy_init_subgraph_op(self) -> None:
        """Build GLT's subgraph operator.

        Raises:
            ValueError: If the topologies store only the owned rows.
        """
        if self._stores_only_owned_rows:
            raise ValueError(
                "Subgraph extraction is not supported on a range-partitioned compact topology"
            )
        super().lazy_init_subgraph_op()


class BaseDistNeighborSampler(GLTDistNeighborSampler):
    """Base class for GiGL distributed samplers.

    Extends GLT's DistNeighborSampler with shared utilities for preparing
    sampling inputs, including ABLP (anchor-based link prediction) support.

    Subclasses must override ``_sample_from_nodes`` with their specific
    sampling strategy (e.g., k-hop neighbor sampling, PPR-based sampling).
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the sampler, its fused feature lookups, and the one-time sampling-error guard.

        Every node or edge type that has features, whether a raw float store, a packed ``uint8`` sidecar, or both, is
        fetched through one ``FusedDistFeature`` per entity kind, with one request per partition per type. They are
        stored in GLT's ``dist_node_feature`` and ``dist_edge_feature`` attributes.

        GLT's constructor runs as :meth:`_init_glt_sampler`, a copy that builds ``self.sampler`` as a
        ``_GlobalIdNeighborSampler``.

        ``GLTDistNeighborSampler`` has no GiGL-owned state; we only add
        ``_sampling_error_sent`` so ``_send_adapter`` can forward at most one
        poison pill per sampler instance. Initializing it here (rather than
        lazily) guarantees the failure handler never raises ``AttributeError``,
        which GLT's event loop would swallow the same way it swallows the
        original sampling exception.
        """
        collect_features = kwargs.get("collect_features", False)
        data = kwargs.get("data")
        # GLT would otherwise build its own DistFeature objects, and register an RPC callee for each, that this class
        # never uses. Re-check this against GLT's constructor on upgrades:
        # https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/python/distributed/dist_neighbor_sampler.py#L159-L172
        self._init_glt_sampler(*args, **{**kwargs, "collect_features": False})
        self.collect_features = collect_features
        self._sampling_error_sent: bool = False

        self.dist_node_feature: Optional[FusedDistFeature] = None
        self.dist_edge_feature: Optional[FusedDistFeature] = None
        if collect_features and data is not None:
            self.dist_node_feature = build_fused_dist_feature(
                data.num_partitions,
                data.partition_idx,
                data.node_features,
                data.node_packed_features,
                data.node_feat_pb,
                self.rpc_router,
                self.device,
            )
            if self.with_edge:
                self.dist_edge_feature = build_fused_dist_feature(
                    data.num_partitions,
                    data.partition_idx,
                    data.edge_features,
                    data.edge_packed_features,
                    data.edge_feat_pb,
                    self.rpc_router,
                    self.device,
                )

    def _init_glt_sampler(
        self,
        data: DistDataset,
        num_neighbors: Optional[NumNeighbors] = None,
        with_edge: bool = False,
        with_neg: bool = False,
        with_weight: bool = False,
        edge_dir: Literal["in", "out"] = "out",
        collect_features: bool = False,
        channel: Optional[ChannelBase] = None,
        use_all2all: bool = False,
        concurrency: int = 1,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Set up GLT's sampler state, building ``self.sampler`` as a ``_GlobalIdNeighborSampler``.

        Copied from ``graphlearn_torch/distributed/dist_neighbor_sampler.py``,
        ``DistNeighborSampler.__init__`` (GLT 0.2.4):
        https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/python/distributed/dist_neighbor_sampler.py#L120-L216

        GLT's constructor builds a ``NeighborSampler`` and hands that object to the RPC callees
        that serve other ranks' requests in the same call, so a subclass cannot be passed in.
        Two lines differ from GLT:

        - ``self.sampler`` is a ``_GlobalIdNeighborSampler``, so this rank's own sampling and the
          requests other ranks send it both translate seeds to local rows.
        - The final ``super().__init__`` call names GLT's base, ``ConcurrentEventLoop``, because
          ``super()`` here resolves to GLT's ``DistNeighborSampler``.

        The callees are registered in GLT's order; callee ids must match on every rank.

        Args:
            data: The graph and feature data with partition info.
            num_neighbors: The number of neighbors to sample on each hop.
            with_edge: Whether to sample with edge ids.
            with_neg: Whether to do negative sampling.
            with_weight: Whether to sample by edge weight.
            edge_dir: The edge direction to sample along, ``"in"`` or ``"out"``.
            collect_features: Whether to build GLT's feature lookups for sampled results.
            channel: The channel to send sampled results to, or ``None`` to return them.
            use_all2all: Whether to use all2all to collect cross-node features.
            concurrency: The max number of seed batches sampled at once.
            device: The device to sample on, or ``None`` for GLT's default.
            seed: The random seed for the sampler, or ``None`` to leave it unset.

        Raises:
            ValueError: If ``data`` is not a GLT ``DistDataset``.
        """
        self.data = data
        self.use_all2all = use_all2all
        self.num_neighbors = num_neighbors
        self.max_input_size = 0
        self.with_edge = with_edge
        self.with_neg = with_neg
        self.with_weight = with_weight
        self.edge_dir = edge_dir
        self.collect_features = collect_features
        self.channel = channel
        self.concurrency = concurrency
        self.device = get_available_device(device)
        self.seed = seed

        if isinstance(data, DistDataset):
            partition2workers = rpc_sync_data_partitions(
                num_data_partitions=self.data.num_partitions,
                current_partition_idx=self.data.partition_idx,
            )
            self.rpc_router = RpcDataPartitionRouter(partition2workers)

            self.dist_graph = DistGraph(
                data.num_partitions,
                data.partition_idx,
                data.graph,
                data.node_pb,
                data.edge_pb,
            )

            self.dist_node_feature = None
            self.dist_edge_feature = None
            if self.collect_features:
                if data.node_features is not None:
                    self.dist_node_feature = DistFeature(
                        data.num_partitions,
                        data.partition_idx,
                        data.node_features,
                        data.node_feat_pb,
                        local_only=False,
                        rpc_router=self.rpc_router,
                        device=self.device,
                    )
                if self.with_edge and data.edge_features is not None:
                    self.dist_edge_feature = DistFeature(
                        data.num_partitions,
                        data.partition_idx,
                        data.edge_features,
                        data.edge_feat_pb,
                        local_only=False,
                        rpc_router=self.rpc_router,
                        device=self.device,
                    )
            # dist_node_labels should is initialized as a DistFeature object in the v6d case
            self.dist_node_labels = self.data.node_labels
            if self.dist_graph.data_cls == "homo":
                if self.dist_node_labels is not None and not isinstance(
                    self.dist_node_labels, torch.Tensor
                ):
                    self.dist_node_labels = DistFeature(
                        self.data.num_partitions,
                        self.data.partition_idx,
                        self.dist_node_labels,
                        self.data.node_feat_pb,
                        local_only=False,
                        rpc_router=self.rpc_router,
                        device=self.device,
                    )
            else:
                assert self.dist_node_labels is None or isinstance(
                    self.dist_node_labels, Dict
                )
                if self.dist_node_labels is not None and all(
                    isinstance(value, Feature)
                    for value in self.dist_node_labels.values()
                ):
                    self.dist_node_labels = DistFeature(
                        self.data.num_partitions,
                        self.data.partition_idx,
                        self.data.node_labels,
                        self.data.node_feat_pb,
                        local_only=False,
                        rpc_router=self.rpc_router,
                        device=self.device,
                    )
        else:
            raise ValueError(
                f"'{self.__class__.__name__}': found invalid input "
                f"data type '{type(data)}'"
            )

        self.sampler = _GlobalIdNeighborSampler(
            self.dist_graph.local_graph,
            self.num_neighbors,
            self.device,
            self.with_edge,
            self.with_neg,
            self.with_weight,
            self.edge_dir,
            seed=self.seed,
            stores_only_owned_rows=_stores_only_owned_rows(data),
        )
        self.inducer_pool = queue.Queue(maxsize=self.concurrency)

        # rpc register
        rpc_sample_callee = RpcSamplingCallee(self.sampler, self.device)
        self.rpc_sample_callee_id = rpc_register(rpc_sample_callee)
        rpc_subgraph_callee = RpcSubGraphCallee(self.sampler, self.device)
        self.rpc_subgraph_callee_id = rpc_register(rpc_subgraph_callee)

        if self.dist_graph.data_cls == "hetero":
            self.num_neighbors = self.sampler.num_neighbors
            self.num_hops = self.sampler.num_hops
            self.edge_types = self.sampler.edge_types

        ConcurrentEventLoop.__init__(self, self.concurrency)
        self._loop.call_soon_threadsafe(ensure_device, self.device)

    def _prepare_sample_loop_inputs(
        self,
        inputs: NodeSamplerInput,
    ) -> SampleLoopInputs:
        """Prepare inputs for the sampling loop.

        Handles both standard NodeSamplerInput and ABLPNodeSamplerInput.
        For ABLP inputs, adds supervision nodes to the sampling seeds and
        builds label metadata.

        Args:
            inputs: Either a NodeSamplerInput or ABLPNodeSamplerInput.

        Returns:
            SampleLoopInputs containing the nodes to sample from and any
            metadata related to the task (e.g., label tensors for ABLP).
        """
        input_seeds = inputs.node.to(self.device)
        input_type = inputs.input_type

        if isinstance(inputs, ABLPNodeSamplerInput):
            return self._prepare_ablp_inputs(inputs, input_seeds, input_type)

        # For homogeneous graphs (input_type is None), return tensor directly.
        # For heterogeneous graphs, return dict mapping node type to tensor.
        if input_type is None:
            return SampleLoopInputs(
                nodes_to_sample=input_seeds,
                metadata={},
            )
        return SampleLoopInputs(
            nodes_to_sample={input_type: input_seeds},
            metadata={},
        )

    def _prepare_ablp_inputs(
        self,
        inputs: ABLPNodeSamplerInput,
        input_seeds: Int64[torch.Tensor, "{inputs.node.shape[0]}"],
        input_type: NodeType,
    ) -> SampleLoopInputs:
        """Prepare ABLP inputs with supervision nodes and label metadata.

        Args:
            inputs: The ABLPNodeSamplerInput containing label information.
            input_seeds: The anchor node seeds (already moved to device).
            input_type: The node type of the anchor seeds.

        Returns:
            SampleLoopInputs with supervision nodes included in nodes_to_sample
            and label tensors in metadata.
        """
        # Since GLT swaps src/dst for edge_dir = "out",
        # and GiGL assumes that supervision edge types are always
        # (anchor_node_type, to, supervision_node_type),
        # we need to index into supervision edge types accordingly.
        label_edge_index = 0 if self.edge_dir == "in" else 2

        # Build metadata and input nodes from positive/negative labels.
        # We need to sample from the supervision nodes as well, and ensure
        # that we are sampling from the correct node type.
        metadata: dict[str, torch.Tensor] = {}
        input_seeds_builder: dict[Union[str, NodeType], list[torch.Tensor]] = (
            defaultdict(list)
        )
        input_seeds_builder[input_type].append(input_seeds)

        for edge_type, label_tensor in inputs.positive_label_by_edge_types.items():
            filtered_label_tensor = label_tensor[label_tensor != PADDING_NODE].to(
                self.device
            )
            input_seeds_builder[edge_type[label_edge_index]].append(
                filtered_label_tensor
            )
            # Update the metadata per positive label edge type.
            # We do this because GLT only supports dict[str, torch.Tensor] for metadata.
            metadata[f"{POSITIVE_LABEL_METADATA_KEY}{str(tuple(edge_type))}"] = (
                label_tensor
            )

        for edge_type, label_tensor in inputs.negative_label_by_edge_types.items():
            filtered_label_tensor = label_tensor[label_tensor != PADDING_NODE].to(
                self.device
            )
            input_seeds_builder[edge_type[label_edge_index]].append(
                filtered_label_tensor
            )
            # Update the metadata per negative label edge type.
            # We do this because GLT only supports dict[str, torch.Tensor] for metadata.
            metadata[f"{NEGATIVE_LABEL_METADATA_KEY}{str(tuple(edge_type))}"] = (
                label_tensor
            )

        nodes_to_sample: dict[Union[str, NodeType], torch.Tensor] = {
            # Keep first-occurrence order so anchor seeds remain at the front of
            # their node type; graph-transformer paths rely on that convention.
            node_type: _stable_unique_preserve_order(
                torch.cat(seeds, dim=0).to(self.device)
            )
            for node_type, seeds in input_seeds_builder.items()
        }

        return SampleLoopInputs(
            nodes_to_sample=nodes_to_sample,
            metadata=metadata,
        )

    async def _send_adapter(
        self,
        async_func,
        *args,
        **kwargs,
    ) -> Optional[SampleMessage]:
        """Override GLT's ``_send_adapter`` to call ``_collate_fn`` (corrected spelling).

        GLT's original calls ``self._colloate_fn`` (typo). This override is the
        only place in GiGL that references the typo — everything else uses
        ``_collate_fn``.

        Copied from ``graphlearn_torch.distributed.DistNeighborSampler._send_adapter``
        (GLT 0.2.4) with the single change of ``_colloate_fn`` → ``_collate_fn``.

        Additionally, any exception raised while sampling or collating is caught
        and surfaced instead of swallowed. GLT's ``ConcurrentEventLoop`` logs a
        failed coroutine as a one-line ``str(e)`` and drops the batch, so the
        channel never receives a message and the loader hangs forever. Here we
        log the full traceback and forward a one-time poison-pill ``SampleMessage``
        (in channel mode) so the consumer raises promptly, or re-raise (in
        channel-less mode, where GLT's ``run_task`` / torch RPC propagates it).
        """
        try:
            sampler_output = await async_func(*args, **kwargs)
            res = await self._collate_fn(sampler_output)
        except Exception:
            logger.exception(
                "Sampling coroutine failed; forwarding error to the loader."
            )
            if self.channel is not None:
                # Send at most one poison pill per sampler instance. The consumer
                # raises on the first pill anyway, and ``ShmChannel.send`` blocks;
                # a failure storm into a bounded channel that nobody drains would
                # wedge the sampler-side event loop.
                if not self._sampling_error_sent:
                    self._sampling_error_sent = True
                    self.channel.send(
                        {
                            SAMPLING_ERROR_KEY: encode_sampling_error(
                                traceback.format_exc()
                            )
                        }
                    )
                return None
            raise  # channel-less mode: propagates via run_task / torch RPC
        if self.channel is None:
            return res
        self.channel.send(res)
        return None

    async def _collate_fn(
        self,
        output: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> SampleMessage:
        """Collect labels and features for the sampled subgraph into a SampleMessage.

        Copied from ``graphlearn_torch.distributed.DistNeighborSampler._colloate_fn``
        (GLT 0.2.4).  The method name preserves GLT's original typo so that this
        override is matched correctly at runtime.

        The only behavioural change from the GLT original is in the ``DistFeature``
        label-fetch paths (both homogeneous and heterogeneous): GLT writes
        ``nlabels.T[0]``, which silently discards all label columns beyond the first
        and breaks multi-label node classification.  This override writes the full
        ``nlabels`` tensor instead, avoiding the extra RPC call that a super()-then-
        re-fetch approach would require.  The non-``DistFeature`` path (plain
        ``torch.Tensor`` labels) is unchanged — it never applied ``.T[0]``.

        This method also issues all independent ``async_get`` requests before
        awaiting them, so label, node-feature, and edge-feature fetches can overlap.

        Features come from the fused lookups, one request per partition per type.
        A type that has features always gets ``nfeats`` / ``efeats``, which have zero
        columns if the type has only a packed sidecar. Its packed sidecar rows go to
        the ``#META.node_packed_features`` / ``#META.edge_packed_features`` keys, and only
        for types that have a sidecar.

        # TODO (mkolodner-sc): Now that GiGL owns this method, investigate whether
        # post-processing steps in DistNeighborLoader._collate_fn can be folded in
        # here and simplified — e.g. set_missing_features (populating empty tensors
        # for node/edge features not fanned out to) and extract_metadata (stripping
        # #META. keys before to_hetero_data to work around a GLT bug where those
        # keys are misinterpreted as edge types).

        Args:
            output: The ``SamplerOutput`` or ``HeteroSamplerOutput`` returned by
                ``_sample_from_nodes``.

        Returns:
            A ``SampleMessage`` (``dict[str, torch.Tensor]``) ready to be sent
            over the sampling channel or returned directly to the loader.
        """
        result_map: SampleMessage = {}
        is_hetero = self.dist_graph.data_cls == "hetero"
        result_map["#IS_HETERO"] = torch.LongTensor([int(is_hetero)])
        if isinstance(output.metadata, dict):
            for k, v in output.metadata.items():
                result_map[f"#META.{k}"] = v

        futs: dict[str, asyncio.Future[torch.Tensor]] = {}
        label_keys: set[str] = set()
        # Result key of each fused lookup's raw matrix -> result key of its packed sidecar rows.
        packed_keys: dict[str, str] = {}

        def request_features(
            feature: FusedDistFeature,
            ids: torch.Tensor,
            entity_type: Optional[Union[NodeType, EdgeType]],
            raw_key: str,
            packed_key: str,
        ) -> None:
            futs[raw_key] = wrap_torch_future(
                feature.async_get(ids.to(torch.long), entity_type)
            )
            packed_keys[raw_key] = packed_key

        if is_hetero:
            for ntype, nodes in output.node.items():
                result_map[f"{as_str(ntype)}.ids"] = nodes
                if output.num_sampled_nodes is not None:
                    if ntype in output.num_sampled_nodes:
                        result_map[f"{as_str(ntype)}.num_sampled_nodes"] = torch.tensor(
                            output.num_sampled_nodes[ntype], device=self.device
                        )
            for etype, rows in output.row.items():
                etype_str = as_str(etype)
                result_map[f"{etype_str}.rows"] = rows
                result_map[f"{etype_str}.cols"] = output.col[etype]
                if self.with_edge:
                    result_map[f"{etype_str}.eids"] = output.edge[etype]
                if output.num_sampled_edges is not None:
                    if etype in output.num_sampled_edges:
                        result_map[f"{etype_str}.num_sampled_edges"] = torch.tensor(
                            output.num_sampled_edges[etype], device=self.device
                        )
            input_type = output.input_type
            assert input_type is not None
            # TODO: fetch node labels in the same fused request as the features (separate PR).
            if not isinstance(input_type, tuple):
                if self.dist_node_labels is not None:
                    if isinstance(self.dist_node_labels, DistFeature):
                        result_key = f"{as_str(input_type)}.nlabels"
                        futs[result_key] = wrap_torch_future(
                            self.dist_node_labels.async_get(
                                output.node[input_type], input_type
                            )
                        )
                        label_keys.add(result_key)
                    else:
                        node_labels = self.dist_node_labels.get(input_type, None)
                        if node_labels is not None:
                            result_map[f"{as_str(input_type)}.nlabels"] = node_labels[
                                output.node[input_type].to(node_labels.device)
                            ]
            if self.dist_node_feature is not None:
                for ntype, nodes in output.node.items():
                    if ntype in self.dist_node_feature.local_feature:
                        request_features(
                            self.dist_node_feature,
                            nodes,
                            ntype,
                            f"{as_str(ntype)}.nfeats",
                            f"#META.{NODE_PACKED_FEATURES_METADATA_KEY}.{as_str(ntype)}",
                        )
            if self.dist_edge_feature is not None:
                for etype in self.edge_types:
                    if etype not in self.dist_edge_feature.local_feature:
                        continue
                    result_edge_type = (
                        reverse_edge_type(etype) if self.edge_dir == "in" else etype
                    )
                    eids = result_map.get(f"{as_str(result_edge_type)}.eids")
                    if eids is None:
                        continue
                    # GLT maps wire edge types to output stores during collation.
                    # Metadata bypasses that mapping, so key it by the output store.
                    output_edge_type = (
                        reverse_edge_type(etype) if self.edge_dir == "out" else etype
                    )
                    request_features(
                        self.dist_edge_feature,
                        eids,
                        etype,
                        f"{as_str(result_edge_type)}.efeats",
                        f"#META.{EDGE_PACKED_FEATURES_METADATA_KEY}.{output_edge_type}",
                    )
            if output.batch is not None:
                for ntype, batch in output.batch.items():
                    result_map[f"{as_str(ntype)}.batch"] = batch
        else:
            result_map["ids"] = output.node
            result_map["rows"] = output.row
            result_map["cols"] = output.col
            if output.num_sampled_nodes is not None:
                result_map["num_sampled_nodes"] = torch.tensor(
                    output.num_sampled_nodes, device=self.device
                )
                result_map["num_sampled_edges"] = torch.tensor(
                    output.num_sampled_edges, device=self.device
                )
            if self.with_edge:
                result_map["eids"] = output.edge
            if self.dist_node_labels is not None:
                if isinstance(self.dist_node_labels, DistFeature):
                    futs["nlabels"] = wrap_torch_future(
                        self.dist_node_labels.async_get(output.node)
                    )
                    label_keys.add("nlabels")
                else:
                    result_map["nlabels"] = self.dist_node_labels[
                        output.node.to(self.dist_node_labels.device)
                    ]
            if self.dist_node_feature is not None:
                request_features(
                    self.dist_node_feature,
                    output.node,
                    None,
                    "nfeats",
                    f"#META.{NODE_PACKED_FEATURES_METADATA_KEY}",
                )
            if self.dist_edge_feature is not None:
                request_features(
                    self.dist_edge_feature,
                    result_map["eids"],
                    None,
                    "efeats",
                    f"#META.{EDGE_PACKED_FEATURES_METADATA_KEY}",
                )
            if output.batch is not None:
                result_map["batch"] = output.batch

        values = await asyncio.gather(*futs.values())
        for result_key, value in zip(futs, values):
            if result_key in label_keys:
                # DistFeature always returns [N, K]. We collapse K=1 to 1-D
                # [N] to match GLT's convention and what downstream code
                # (e.g. CrossEntropyLoss) expects for data.y. Multi-label
                # (K>1) keeps the full 2-D matrix.
                # TODO (mkolodner-sc): Consider investigating always returning
                # 2-D — this may be a breaking change for single-label
                # training pipelines (e.g. CrossEntropyLoss expects 1-D data.y).
                value = value if value.shape[1] > 1 else value.T[0]
            if result_key in packed_keys:
                value, packed = value
                if packed is not None:
                    result_map[packed_keys[result_key]] = packed
            result_map[result_key] = value

        return result_map

    async def _sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        """Sample subgraph from seed nodes.

        Subclasses must override this method with their specific sampling
        strategy.

        Args:
            inputs: The seed nodes to sample from.

        Raises:
            NotImplementedError: Always — subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _sample_from_nodes."
        )
