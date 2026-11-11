import asyncio
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import torch
from graphlearn_torch.channel import SampleMessage
from graphlearn_torch.distributed import DistNeighborSampler as GLTDistNeighborSampler
from graphlearn_torch.distributed.dist_feature import DistFeature
from graphlearn_torch.distributed.dist_neighbor_sampler import PartialNeighborOutput
from graphlearn_torch.distributed.event_loop import wrap_torch_future
from graphlearn_torch.distributed.rpc import rpc_request_async
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    NeighborOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from graphlearn_torch.typing import EdgeType, NodeType, as_str
from graphlearn_torch.utils import default_id_select, reverse_edge_type
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
from gigl.distributed.utils.partition_routing import (
    group_by_partition,
    grouped_routing_enabled,
)
from gigl.distributed.utils.sampling_errors import (
    SAMPLING_ERROR_KEY,
    encode_sampling_error,
)
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
        super().__init__(*args, **{**kwargs, "collect_features": False})
        self.collect_features = collect_features
        self._sampling_error_sent: bool = False

        self.dist_node_feature: Optional[FusedDistFeature] = None
        self.dist_edge_feature: Optional[FusedDistFeature] = None
        if collect_features and data is not None:
            self.dist_node_feature = build_fused_dist_feature(
                data.num_partitions,
                data.partition_idx,
                data.node_features,
                data.node_quantized_features,
                data.node_feat_pb,
                self.rpc_router,
                self.device,
            )
            if self.with_edge:
                self.dist_edge_feature = build_fused_dist_feature(
                    data.num_partitions,
                    data.partition_idx,
                    data.edge_features,
                    data.edge_quantized_features,
                    data.edge_feat_pb,
                    self.rpc_router,
                    self.device,
                )

        # Opt-in sort-based routing of ids to partitions (``GIGL_SAMPLING_GROUPED_ROUTING``), read once here.
        self._grouped_routing: bool = grouped_routing_enabled()

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

    def _route_hop(
        self,
        srcs: torch.Tensor,
        orders: torch.Tensor,
        partition_ids: torch.Tensor,
        node_pb: object,
    ) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        """Yields ``(partition, ids, positions)`` of each partition that owns some of ``srcs``.

        Partitions come in order starting at this process's own partition. ``positions`` are the indices of
        ``ids`` in ``srcs``. With grouped routing on and GLT's default ``id_select``, one stable sort routes all
        partitions. Otherwise one mask per partition does, which also serves a custom ``id_select``.
        """
        num_partitions = self.data.num_partitions
        grouped = None
        if self._grouped_routing and self.data.id_select is default_id_select:
            grouped = group_by_partition(srcs, partition_ids, num_partitions)
        for i in range(num_partitions):
            pidx = (self.data.partition_idx + i) % num_partitions
            if grouped is None:
                p_mask = partition_ids == pidx
                p_ids = self.data.id_select(srcs, p_mask, node_pb)
                if p_ids.shape[0] == 0:
                    continue
                yield pidx, p_ids, torch.masked_select(orders, p_mask)
            else:
                if grouped[0][pidx].shape[0] == 0:
                    continue
                yield pidx, grouped[0][pidx], grouped[1][pidx]

    async def _sample_one_hop(
        self,
        srcs: torch.Tensor,
        num_nbr: int,
        etype: Optional[EdgeType],
    ) -> NeighborOutput:
        """Sample one hop of neighbors for ``srcs`` across all partitions.

        Delegates to GLT when grouped routing is off. Otherwise this is a copy of
        ``graphlearn_torch.distributed.DistNeighborSampler._sample_one_hop`` (GLT 0.2.4) in which the ids
        are routed to partitions by ``group_by_partition``. A unit test hashes GLT's source so that a
        change to it is noticed.

        Args:
            srcs: Source node ids for this hop.
            num_nbr: Requested neighbors per source node.
            etype: Edge type to sample along, or ``None`` for homogeneous graphs.

        Returns:
            The sampled neighbors of ``srcs`` in input order.
        """
        if not self._grouped_routing:
            return await super()._sample_one_hop(srcs, num_nbr, etype)
        device = self.device
        srcs = srcs.to(device)
        if self.data.graph_caching:
            nbr_out = None
            if srcs is not None and srcs.numel() > 0:
                nbr_out = self.sampler.sample_one_hop(srcs, num_nbr, etype)
            return nbr_out

        orders = torch.arange(srcs.size(0), dtype=torch.long, device=device)
        src_ntype = None
        if etype is not None:
            if self.edge_dir == "out":
                src_ntype = etype[0]
            elif self.edge_dir == "in":
                src_ntype = etype[-1]
        partition_ids = self.dist_graph.get_node_partitions(srcs, src_ntype).to(device)
        node_pb = self.dist_graph.node_pb
        if isinstance(node_pb, dict):
            node_pb = node_pb[src_ntype]
        partition_results: list[PartialNeighborOutput] = []
        remote_orders_list: list[torch.Tensor] = []
        futs: list[torch.futures.Future] = []
        for pidx, p_ids, p_orders in self._route_hop(
            srcs, orders, partition_ids, node_pb
        ):
            if pidx == self.data.partition_idx:
                p_nbr_out = self.sampler.sample_one_hop(p_ids, num_nbr, etype)
                partition_results.append(PartialNeighborOutput(p_orders, p_nbr_out))
            else:
                remote_orders_list.append(p_orders)
                to_worker = self.rpc_router.get_to_worker(pidx)
                futs.append(
                    rpc_request_async(
                        to_worker,
                        self.rpc_sample_callee_id,
                        args=(p_ids.cpu(), num_nbr, etype),
                    )
                )
        if len(remote_orders_list) == 0:
            if len(partition_results) > 0:
                return partition_results[0].output
            # Mirrors GLT, which returns an empty tensor rather than a NeighborOutput here.
            return torch.tensor([], device=device, dtype=torch.int64)  # type: ignore[return-value]
        res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
        for i, res_fut in enumerate(res_fut_list):
            partition_results.append(
                PartialNeighborOutput(
                    index=remote_orders_list[i], output=res_fut.wait().to(device)
                )
            )
        return self._stitch_sample_results(srcs, partition_results)

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
