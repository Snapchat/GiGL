import asyncio
import json
import os
import socket
import time
import traceback
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Coroutine, Optional, Union

import torch
from graphlearn_torch.channel import SampleMessage
from graphlearn_torch.distributed import DistNeighborSampler as GLTDistNeighborSampler
from graphlearn_torch.distributed.dist_feature import DistFeature
from graphlearn_torch.distributed.event_loop import wrap_torch_future
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from graphlearn_torch.typing import NodeType, as_str
from graphlearn_torch.utils import reverse_edge_type

from gigl.common.logger import Logger
from gigl.distributed.sampler import (
    NEGATIVE_LABEL_METADATA_KEY,
    POSITIVE_LABEL_METADATA_KEY,
    ABLPNodeSamplerInput,
)
from gigl.distributed.utils.sampling_errors import (
    SAMPLING_ERROR_KEY,
    encode_sampling_error,
)
from gigl.utils.data_splitters import PADDING_NODE

logger = Logger()

_SAMPLER_TIMING_ENV_VAR = "GIGL_SAMPLER_TIMING_LOG_EVERY_N"


def _get_sampler_timing_log_every_n() -> int:
    """Parses and validates the opt-in sampler timing interval."""
    raw_timing_interval = os.environ.get(_SAMPLER_TIMING_ENV_VAR, "0")
    try:
        timing_interval = int(raw_timing_interval)
    except ValueError as error:
        raise ValueError(
            f"{_SAMPLER_TIMING_ENV_VAR} must be a non-negative integer, "
            f"got {raw_timing_interval!r}"
        ) from error
    if timing_interval < 0:
        raise ValueError(
            f"{_SAMPLER_TIMING_ENV_VAR} must be non-negative, got {timing_interval}"
        )
    return timing_interval


class _SamplingTimingRecorder:
    """Aggregates sampler stage time without logging in the hot path."""

    def __init__(self, log_every_n: int) -> None:
        if log_every_n <= 0:
            raise ValueError(f"log_every_n must be positive, got {log_every_n}")
        self._log_every_n = log_every_n
        self._lock = Lock()
        self._total_completed_batches = 0
        self._admission_events = 0
        self._admission_blocked_seconds = 0.0
        self._sample_await_seconds = 0.0
        self._collate_seconds = 0.0
        self._channel_send_seconds = 0.0
        self._loop_wall_start: Optional[float] = None
        self._loop_thread_cpu_start: Optional[float] = None

    def begin_loop_observation(self) -> None:
        """Starts the event-loop thread utilization window once."""
        with self._lock:
            if self._loop_wall_start is not None:
                return
            self._loop_wall_start = time.perf_counter()
            self._loop_thread_cpu_start = time.thread_time()

    def record_admission(self, blocked_seconds: float) -> None:
        """Records one wait for an event-loop concurrency slot."""
        with self._lock:
            self._admission_events += 1
            self._admission_blocked_seconds += blocked_seconds

    def record_completed(
        self,
        *,
        sample_await_seconds: float,
        collate_seconds: float,
        channel_send_seconds: float,
    ) -> Optional[dict[str, Union[int, float]]]:
        """Records one completed batch and returns a full-window payload."""
        with self._lock:
            self._total_completed_batches += 1
            self._sample_await_seconds += sample_await_seconds
            self._collate_seconds += collate_seconds
            self._channel_send_seconds += channel_send_seconds
            if self._total_completed_batches % self._log_every_n:
                return None

            # Read CPU before wall at the end, complementing wall-before-CPU
            # at the start. This nests the CPU interval inside the wall interval.
            loop_thread_cpu_now = time.thread_time()
            loop_wall_now = time.perf_counter()
            if self._loop_wall_start is None:
                self._loop_wall_start = loop_wall_now
                self._loop_thread_cpu_start = loop_thread_cpu_now
            window_batches = self._log_every_n
            admission_events = self._admission_events
            loop_wall_seconds = max(loop_wall_now - self._loop_wall_start, 0.0)
            assert self._loop_thread_cpu_start is not None
            loop_thread_cpu_seconds = max(
                loop_thread_cpu_now - self._loop_thread_cpu_start, 0.0
            )
            loop_thread_busy_fraction = min(
                loop_thread_cpu_seconds / loop_wall_seconds
                if loop_wall_seconds
                else 0.0,
                1.0,
            )
            payload: dict[str, Union[int, float]] = {
                "completed_batches": self._total_completed_batches,
                "window_batches": window_batches,
                "admission_events": admission_events,
                "admission_blocked_s": round(self._admission_blocked_seconds, 6),
                "sample_await_s": round(self._sample_await_seconds, 6),
                "collate_s": round(self._collate_seconds, 6),
                "channel_send_blocked_s": round(self._channel_send_seconds, 6),
                "loop_wall_s": round(loop_wall_seconds, 6),
                "loop_thread_cpu_s": round(loop_thread_cpu_seconds, 6),
                "loop_thread_busy_fraction": round(loop_thread_busy_fraction, 6),
                "admission_blocked_ms_per_event": round(
                    self._admission_blocked_seconds / max(admission_events, 1) * 1000,
                    3,
                ),
                "sample_await_ms_per_batch": round(
                    self._sample_await_seconds / window_batches * 1000,
                    3,
                ),
                "collate_ms_per_batch": round(
                    self._collate_seconds / window_batches * 1000,
                    3,
                ),
                "channel_send_blocked_ms_per_batch": round(
                    self._channel_send_seconds / window_batches * 1000,
                    3,
                ),
            }
            self._admission_events = 0
            self._admission_blocked_seconds = 0.0
            self._sample_await_seconds = 0.0
            self._collate_seconds = 0.0
            self._channel_send_seconds = 0.0
            # Establish a fresh wall-before-CPU pair for the next window.
            self._loop_wall_start = time.perf_counter()
            self._loop_thread_cpu_start = time.thread_time()
            return payload


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
        """Initialize the sampler and the one-time sampling-error guard.

        ``GLTDistNeighborSampler`` has no GiGL-owned state; we only add
        ``_sampling_error_sent`` so ``_send_adapter`` can forward at most one
        poison pill per sampler instance. Initializing it here (rather than
        lazily) guarantees the failure handler never raises ``AttributeError``,
        which GLT's event loop would swallow the same way it swallows the
        original sampling exception.
        """
        super().__init__(*args, **kwargs)
        self._sampling_error_sent: bool = False
        timing_interval = _get_sampler_timing_log_every_n()
        self._sampling_timing_recorder = (
            _SamplingTimingRecorder(timing_interval) if timing_interval else None
        )

    def add_task(
        self,
        coro: Coroutine[Any, Any, Optional[SampleMessage]],
        callback: Optional[Callable[[Optional[SampleMessage]], Any]] = None,
    ) -> None:
        """Schedules sampling and measures time blocked on the concurrency limit."""
        recorder = getattr(self, "_sampling_timing_recorder", None)
        if recorder is None:
            super().add_task(coro, callback)
            return

        admission_start = time.perf_counter()
        self._sem.acquire()
        recorder.record_admission(time.perf_counter() - admission_start)

        def on_done(future: Future[Optional[SampleMessage]]) -> None:
            try:
                result = future.result()
                if callback is not None:
                    callback(result)
            except Exception as error:
                logger.error(f"coroutine task failed: {error}")
            finally:
                self._sem.release()

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.add_done_callback(on_done)

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
        input_seeds: torch.Tensor,
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
        recorder = getattr(self, "_sampling_timing_recorder", None)
        if recorder is not None:
            # This coroutine runs on GLT's single sampling event-loop thread.
            # thread_time() therefore measures the serialized producer resource
            # directly, including work performed by all concurrent coroutines.
            recorder.begin_loop_observation()
        try:
            if recorder is None:
                sampler_output = await async_func(*args, **kwargs)
                res = await self._collate_fn(sampler_output)
                sample_await_seconds = 0.0
                collate_seconds = 0.0
            else:
                sample_start = time.perf_counter()
                sampler_output = await async_func(*args, **kwargs)
                sample_await_seconds = time.perf_counter() - sample_start
                collate_start = time.perf_counter()
                res = await self._collate_fn(sampler_output)
                collate_seconds = time.perf_counter() - collate_start
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
        channel_send_seconds = 0.0
        if self.channel is None:
            result = res
        elif recorder is None:
            self.channel.send(res)
            result = None
        else:
            send_start = time.perf_counter()
            self.channel.send(res)
            channel_send_seconds = time.perf_counter() - send_start
            result = None

        if recorder is not None:
            timing_payload = recorder.record_completed(
                sample_await_seconds=sample_await_seconds,
                collate_seconds=collate_seconds,
                channel_send_seconds=channel_send_seconds,
            )
            if timing_payload is not None:
                timing_payload.update(
                    {
                        "hostname": socket.gethostname(),
                        "process_id": os.getpid(),
                        "concurrency": self.concurrency,
                        "torch_num_threads": torch.get_num_threads(),
                    }
                )
                logger.info(
                    f"GIGL_SAMPLER_TIMING {json.dumps(timing_payload, sort_keys=True)}"
                )
        return result

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

        In non-all2all mode, this method also issues all independent
        ``async_get`` requests before awaiting them, so label, node-feature, and
        edge-feature fetches can overlap.

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
                if self.use_all2all:
                    sorted_ntype = sorted(self.dist_node_feature.feature_pb.keys())
                    nfeat_dict = self.dist_node_feature.get_all2all(
                        output, sorted_ntype
                    )
                    for ntype, nfeats in nfeat_dict.items():
                        result_map[f"{as_str(ntype)}.nfeats"] = nfeats
                else:
                    for ntype, nodes in output.node.items():
                        nodes = nodes.to(torch.long)
                        futs[f"{as_str(ntype)}.nfeats"] = wrap_torch_future(
                            self.dist_node_feature.async_get(nodes, ntype)
                        )
            if self.dist_edge_feature is not None and self.with_edge:
                for etype in self.edge_types:
                    if self.edge_dir == "in":
                        eids = result_map.get(
                            f"{as_str(reverse_edge_type(etype))}.eids", None
                        )
                    elif self.edge_dir == "out":
                        eids = result_map.get(f"{as_str(etype)}.eids", None)
                    if eids is not None:
                        eids = eids.to(torch.long)
                        if self.edge_dir == "in":
                            result_key = f"{as_str(reverse_edge_type(etype))}.efeats"
                        else:
                            result_key = f"{as_str(etype)}.efeats"
                        futs[result_key] = wrap_torch_future(
                            self.dist_edge_feature.async_get(eids, etype)
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
                futs["nfeats"] = wrap_torch_future(
                    self.dist_node_feature.async_get(output.node)
                )
            if self.dist_edge_feature is not None:
                eids = result_map["eids"]
                futs["efeats"] = wrap_torch_future(
                    self.dist_edge_feature.async_get(eids)
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
