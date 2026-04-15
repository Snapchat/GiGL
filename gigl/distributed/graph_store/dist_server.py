"""
GiGL implementation of GLT DistServer.

Uses GiGL's DistSamplingProducer which supports neighbor sampling
and ABLP (anchor-based link prediction) via BaseGiGLSampler subclasses
(DistNeighborSampler for k-hop, DistPPRNeighborSampler for PPR).

Based on https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_server.py
"""

import logging
import threading
import time
from collections import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, TypeVar, Union

import graphlearn_torch.distributed.dist_server as glt_dist_server
import torch
from graphlearn_torch.channel import QueueTimeoutError, SampleMessage, ShmChannel
from graphlearn_torch.distributed import (
    RemoteDistSamplingWorkerOptions,
    barrier,
    init_rpc,
    shutdown_rpc,
)
from graphlearn_torch.partition import PartitionBook
from graphlearn_torch.sampler import (
    EdgeSamplerInput,
    NodeSamplerInput,
    RemoteSamplerInput,
    SamplingConfig,
)

from gigl.common.logger import Logger
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.graph_store.messages import (
    FetchABLPInputRequest,
    FetchNodesRequest,
    InitSamplingBackendRequest,
    RegisterBackendRequest,
)
from gigl.distributed.graph_store.sharding import ServerSlice
from gigl.distributed.graph_store.shared_dist_sampling_producer import (
    SharedDistSamplingBackend,
)
from gigl.distributed.sampler import ABLPNodeSamplerInput
from gigl.distributed.sampler_options import PPRSamplerOptions, SamplerOptions
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import FeatureInfo, select_label_edge_types
from gigl.utils.data_splitters import get_labels_for_anchor_nodes

SERVER_EXIT_STATUS_CHECK_INTERVAL = 5.0
r""" Interval (in seconds) to check exit status of server.
"""
FETCH_SLOW_LOG_SECS = 1.0

logger = Logger()

R = TypeVar("R")


@dataclass
class ChannelState:
    """Per-channel state for a registered sampling input.

    Args:
        backend_id: The ID of the backend this channel belongs to.
        worker_key: The unique key identifying this compute-rank channel.
        channel: The shared-memory channel for passing sampled messages.
        epoch: The last epoch started on this channel.
        lock: A reentrant lock guarding channel-level operations.
    """

    backend_id: int
    worker_key: str
    channel: ShmChannel
    epoch: int = -1
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass
class SamplingBackendState:
    """Per-backend state for a shared sampling backend.

    Args:
        backend_id: The unique ID of this backend.
        backend_key: The key identifying this backend (e.g. ``"dist_neighbor_loader_0"``).
        runtime: The shared sampling backend runtime.
        active_channels: Set of channel IDs currently registered on this backend.
        lock: A reentrant lock guarding backend-level operations.
    """

    backend_id: int
    backend_key: str
    runtime: SharedDistSamplingBackend
    active_channels: set[int] = field(default_factory=set)
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass
class _ChannelFetchStats:
    """Per-channel fetch timing stats for ``fetch_one_sampled_message``."""

    fetch_count: int = 0
    fetch_total_elapsed: float = 0.0
    fetch_slow_count: int = 0


class DistServer:
    r"""A server that supports launching remote sampling workers for
    training clients.

    Note that this server is enabled only when the distribution mode is a
    server-client framework, and the graph and feature store will be partitioned
    and managed by all server nodes.

    Args:
      dataset (DistDataset): The ``DistDataset`` object of a partition of graph
        data and feature data, along with distributed patition books.
      log_every_n (int): Log aggregated ``fetch_one_sampled_message`` timing
        stats after every ``N`` fetch attempts per channel.
    """

    def __init__(self, dataset: DistDataset, log_every_n: int = 50) -> None:
        self.dataset = dataset
        self._lock = threading.RLock()
        self._exit = False
        self._next_backend_id = 0
        self._next_channel_id = 0
        self._backend_id_by_backend_key: dict[str, int] = {}
        self._backend_state_by_backend_id: dict[int, SamplingBackendState] = {}
        self._channel_state_by_channel_id: dict[int, ChannelState] = {}
        self._fetch_stats_by_channel_id: dict[int, _ChannelFetchStats] = {}
        self._log_every_n = log_every_n

    def shutdown(self) -> None:
        with self._lock:
            backends = list(self._backend_state_by_backend_id.values())
            self._backend_id_by_backend_key.clear()
            self._backend_state_by_backend_id.clear()
            self._channel_state_by_channel_id.clear()
        for backend_state in backends:
            try:
                backend_state.runtime.shutdown()
            except Exception:
                logger.warning(
                    f"Failed to shut down backend backend_id={backend_state.backend_id} "
                    f"backend_key={backend_state.backend_key}",
                    exc_info=True,
                )

    def wait_for_exit(self) -> None:
        r"""Block until the exit flag been set to ``True``."""
        while not self._exit:
            time.sleep(SERVER_EXIT_STATUS_CHECK_INTERVAL)

    def exit(self) -> bool:
        r"""Set the exit flag to ``True``."""
        self._exit = True
        return self._exit

    def get_dataset_meta(
        self,
    ) -> tuple[int, int, Optional[list[NodeType]], Optional[list[EdgeType]]]:
        r"""Get the meta info of the distributed dataset managed by the current
        server, including partition info and graph types.
        """
        return (
            self.dataset.num_partitions,
            self.dataset.partition_idx,
            self.dataset.get_node_types(),
            self.dataset.get_edge_types(),
        )

    def get_node_partition_id(
        self, node_type: Optional[NodeType], index: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if isinstance(self.dataset.node_pb, PartitionBook):
            if node_type is not None:
                raise ValueError(
                    f"node_type must be None for homogeneous dataset. Received: {node_type}"
                )
            partition_id = self.dataset.node_pb[index]
            return partition_id
        elif isinstance(self.dataset.node_pb, dict):
            if node_type is None:
                raise ValueError(
                    f"node_type must be provided for heterogeneous dataset. Received: {node_type}"
                )
            partition_id = self.dataset.node_pb[node_type][index]
            return partition_id
        return None

    def get_node_partition_book(
        self, node_type: Optional[NodeType]
    ) -> Optional[PartitionBook]:
        """
        Gets the partition book for the specified node type.

        Args:
            node_type: The node type to look up.  Must be ``None`` for
                homogeneous datasets and non-``None`` for heterogeneous ones.

        Returns:
            The partition book for the requested node type, or ``None`` if
            no partition book is available.

        Raises:
            ValueError: If ``node_type`` is mismatched with the dataset type.
        """
        node_pb = self.dataset.node_pb
        if isinstance(node_pb, dict):
            if node_type is None:
                raise ValueError(
                    "node_type must be provided for heterogeneous dataset. "
                    f"Available node types: {list(node_pb.keys())}"
                )
            return node_pb[node_type]
        else:
            if node_type is not None:
                raise ValueError(
                    f"node_type must be None for homogeneous dataset. Received: {node_type}"
                )
            return node_pb

    def get_edge_partition_book(
        self, edge_type: Optional[EdgeType]
    ) -> Optional[PartitionBook]:
        """
        Gets the partition book for the specified edge type.
        Args:
            edge_type: The edge type to look up.  Must be ``None`` for
                homogeneous datasets and non-``None`` for heterogeneous ones.

        Returns:
            The partition book for the requested edge type, or ``None`` if
            no partition book is available.

        Raises:
            ValueError: If ``edge_type`` is mismatched with the dataset type.
        """
        edge_pb = self.dataset.edge_pb
        if isinstance(edge_pb, dict):
            if edge_type is None:
                raise ValueError(
                    "edge_type must be provided for heterogeneous dataset. "
                    f"Available edge types: {list(edge_pb.keys())}"
                )
            return edge_pb[edge_type]
        else:
            if edge_type is not None:
                raise ValueError(
                    f"edge_type must be None for homogeneous dataset. Received: {edge_type}"
                )
            return edge_pb

    def get_node_feature(
        self, node_type: Optional[NodeType], index: torch.Tensor
    ) -> torch.Tensor:
        feature = self.dataset.get_node_feature(node_type)
        return feature[index].cpu()

    def get_tensor_size(self, node_type: Optional[NodeType]) -> torch.Size:
        feature = self.dataset.get_node_feature(node_type)
        return feature.shape

    def get_node_label(
        self, node_type: Optional[NodeType], index: torch.Tensor
    ) -> torch.Tensor:
        label = self.dataset.get_node_label(node_type)
        return label[index]

    def get_edge_index(
        self, edge_type: Optional[EdgeType], layout: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph = self.dataset.get_graph(edge_type)
        row = None
        col = None
        result = None
        if layout == "coo":
            row, col, _, _ = graph.topo.to_coo()
            result = (row, col)
        else:
            raise ValueError(f"Invalid layout {layout}")
        return result

    def get_edge_size(
        self, edge_type: Optional[EdgeType], layout: str
    ) -> tuple[int, int]:
        graph = self.dataset.get_graph(edge_type)
        if layout == "coo":
            row_count = graph.row_count
            col_count = graph.col_count
        else:
            raise ValueError(f"Invalid layout {layout}")
        return (row_count, col_count)

    def get_node_feature_info(
        self,
    ) -> Union[FeatureInfo, dict[NodeType, FeatureInfo], None]:
        """Get node feature information from the dataset.

        Returns:
            Node feature information, which can be:
            - A single FeatureInfo object for homogeneous graphs
            - A dict mapping NodeType to FeatureInfo for heterogeneous graphs
            - None if no node features are available
        """
        return self.dataset.node_feature_info

    def get_edge_feature_info(
        self,
    ) -> Union[FeatureInfo, dict[EdgeType, FeatureInfo], None]:
        """Get edge feature information from the dataset.

        Returns:
            Edge feature information, which can be:
            - A single FeatureInfo object for homogeneous graphs
            - A dict mapping EdgeType to FeatureInfo for heterogeneous graphs
            - None if no edge features are available
        """
        return self.dataset.edge_feature_info

    def get_edge_dir(self) -> Literal["in", "out"]:
        """Get the edge direction from the dataset.

        Returns:
            The edge direction.
        """
        return self.dataset.edge_dir

    def get_node_ids(
        self,
        request: FetchNodesRequest,
    ) -> torch.Tensor:
        """Get the node ids from the dataset.

        Args:
            request: The node-fetch request, including split, node type,
                and an optional contiguous server slice.

        Returns:
            The node ids.

        Raises:
            ValueError:
                * If the split is invalid
                * If the node ids are not a torch.Tensor or a dict[NodeType, torch.Tensor]
                * If the node type is provided for a homogeneous dataset
                * If the node ids are not a dict[NodeType, torch.Tensor] when no node type is provided

            Note: When `split=None`, all nodes are queryable. This means nodes from any
            split (train, val, or test) may be returned. This is useful when you need
            to sample neighbors during inference, as neighbor nodes may belong to any split.
        """
        return self._get_node_ids(
            split=request.split,
            node_type=request.node_type,
            server_slice=request.server_slice,
        )

    def _get_node_ids(
        self,
        split: Optional[Union[Literal["train", "val", "test"], str]],
        node_type: Optional[NodeType],
        server_slice: Optional[ServerSlice] = None,
    ) -> torch.Tensor:
        """Core implementation for fetching node IDs by split, type, and optional slicing.

        Args:
            split: The dataset split to fetch from (``"train"``, ``"val"``,
                ``"test"``, or ``None`` for all nodes).
            node_type: The node type to select. Must be ``None`` for
                homogeneous datasets.
            server_slice: An optional :class:`ServerSlice` to return only a
                fraction of the nodes. When ``None``, all nodes are returned.

        Returns:
            The node IDs tensor, optionally sliced by ``server_slice``.

        Raises:
            ValueError: If the split is invalid, or the node type is
                inconsistent with the dataset type (homogeneous vs.
                heterogeneous).
        """
        if split == "train":
            nodes = self.dataset.train_node_ids
        elif split == "val":
            nodes = self.dataset.val_node_ids
        elif split == "test":
            nodes = self.dataset.test_node_ids
        elif split is None:
            nodes = self.dataset.node_ids
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be one of 'train', 'val', 'test', or None."
            )

        if node_type is not None:
            if not isinstance(nodes, abc.Mapping):
                raise ValueError(
                    f"node_type was provided as {node_type}, so node ids must be a dict[NodeType, torch.Tensor] "
                    f"(e.g. a heterogeneous dataset), got {type(nodes)}"
                )
            nodes = nodes[node_type]
        elif not isinstance(nodes, torch.Tensor):
            raise ValueError(
                f"node_type was not provided, so node ids must be a torch.Tensor (e.g. a homogeneous dataset), got {type(nodes)}."
            )

        if server_slice is not None:
            return server_slice.slice_tensor(nodes)
        return nodes

    def get_edge_types(self) -> Optional[list[EdgeType]]:
        """Get the edge types from the dataset.

        Returns:
            The edge types in the dataset, None if the dataset is homogeneous.
        """
        if isinstance(self.dataset.graph, dict):
            return list(self.dataset.graph.keys())
        else:
            return None

    def get_node_types(self) -> Optional[list[NodeType]]:
        """Get the node types from the dataset.

        Returns:
            The node types in the dataset, None if the dataset is homogeneous.
        """
        if isinstance(self.dataset.graph, dict):
            return list(self.dataset.get_node_types())
        else:
            return None

    def get_ablp_input(
        self,
        request: FetchABLPInputRequest,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get the ABLP (Anchor Based Link Prediction) input for distributed processing.

        Args:
            request: The ABLP fetch request, including split, node type,
                supervision edge type, and an optional contiguous server slice.

        Returns:
            A tuple containing the anchor nodes for the rank, the positive labels, and the negative labels.
            The positive labels are of shape [N, M], where N is the number of anchor nodes and M is the number of positive labels.
            The negative labels are of shape [N, M], where N is the number of anchor nodes and M is the number of negative labels.
            The negative labels may be None if no negative labels are available.

        Raises:
            ValueError: If the split is invalid.
        """
        anchors = self._get_node_ids(
            split=request.split,
            node_type=request.node_type,
            server_slice=request.server_slice,
        )
        positive_label_edge_type, negative_label_edge_type = select_label_edge_types(
            request.supervision_edge_type, self.dataset.get_edge_types()
        )
        positive_labels, negative_labels = get_labels_for_anchor_nodes(
            self.dataset, anchors, positive_label_edge_type, negative_label_edge_type
        )
        return anchors, positive_labels, negative_labels

    def init_sampling_backend(self, opts: InitSamplingBackendRequest) -> int:
        """Create or reuse a shared sampling backend for one loader instance.

        If a backend with the same ``backend_key`` already exists, returns
        its ID without creating a new one.

        Args:
            opts: The initialization request containing the backend key,
                worker options, sampler options, and sampling config.

        Returns:
            The unique backend ID.
        """
        request_start_time = time.monotonic()
        with self._lock:
            backend_id = self._backend_id_by_backend_key.get(opts.backend_key)
            if backend_id is not None:
                return backend_id
            backend_id = self._next_backend_id
            self._next_backend_id += 1
            backend_state = SamplingBackendState(
                backend_id=backend_id,
                backend_key=opts.backend_key,
                runtime=SharedDistSamplingBackend(
                    data=self.dataset,
                    worker_options=opts.worker_options,
                    sampling_config=opts.sampling_config,
                    sampler_options=opts.sampler_options,
                    # We only need degree tensor for PPR sampling
                    degree_tensors=self.dataset.degree_tensor
                    if isinstance(opts.sampler_options, PPRSamplerOptions)
                    else None,
                ),
            )
            self._backend_id_by_backend_key[opts.backend_key] = backend_id
            self._backend_state_by_backend_id[backend_id] = backend_state
        init_start_time = time.monotonic()
        try:
            backend_state.runtime.init_backend()
        except Exception:
            with self._lock:
                self._backend_id_by_backend_key.pop(opts.backend_key, None)
                self._backend_state_by_backend_id.pop(backend_id, None)
            raise
        init_elapsed = time.monotonic() - init_start_time
        total_elapsed = time.monotonic() - request_start_time
        logger.info(
            f"Initialized sampling backend backend_key={opts.backend_key} "
            f"backend_id={backend_id} "
            f"init_backend={init_elapsed:.2f}s total={total_elapsed:.2f}s"
        )
        return backend_id

    def register_sampling_input(self, opts: RegisterBackendRequest) -> int:
        """Register one compute-rank input channel on an existing backend.

        Args:
            opts: The registration request containing the backend ID,
                worker key, sampler input, sampling config, and buffer settings.

        Returns:
            The unique channel ID for this input.
        """
        request_start_time = time.monotonic()
        with self._lock:
            backend_state = self._backend_state_by_backend_id[opts.backend_id]
            channel_id = self._next_channel_id
            self._next_channel_id += 1
            channel = ShmChannel(opts.buffer_capacity, opts.buffer_size)
            channel_state = ChannelState(
                backend_id=opts.backend_id,
                worker_key=opts.worker_key,
                channel=channel,
            )
            self._channel_state_by_channel_id[channel_id] = channel_state
            backend_state.active_channels.add(channel_id)

        sampler_input = opts.sampler_input
        if isinstance(sampler_input, RemoteSamplerInput):
            sampler_input = sampler_input.to_local_sampler_input(dataset=self.dataset)

        try:
            with backend_state.lock:
                backend_state.runtime.register_input(
                    channel_id=channel_id,
                    worker_key=opts.worker_key,
                    sampler_input=sampler_input,
                    sampling_config=opts.sampling_config,
                    channel=channel,
                )
        except Exception:
            with self._lock:
                self._channel_state_by_channel_id.pop(channel_id, None)
                backend_state.active_channels.discard(channel_id)
            raise

        logger.info(
            f"Registered sampling input backend_id={opts.backend_id} "
            f"channel_id={channel_id} worker_key={opts.worker_key} "
            f"active_channels={len(backend_state.active_channels)} "
            f"in {time.monotonic() - request_start_time:.2f}s"
        )
        return channel_id

    def destroy_sampling_input(self, channel_id: int) -> None:
        """Destroy one registered sampling channel and maybe its backend.

        If this is the last channel on the backend, the backend is shut down
        and removed.

        Args:
            channel_id: The ID of the channel to destroy.
        """
        self._fetch_stats_by_channel_id.pop(channel_id, None)
        with self._lock:
            channel_state = self._channel_state_by_channel_id.pop(channel_id, None)
            if channel_state is None:
                return
            backend_state = self._backend_state_by_backend_id.get(
                channel_state.backend_id
            )
        if backend_state is None:
            return

        with backend_state.lock:
            backend_state.runtime.unregister_input(channel_id)

        should_shutdown_backend = False
        with self._lock:
            backend_state.active_channels.discard(channel_id)
            if not backend_state.active_channels:
                self._backend_state_by_backend_id.pop(backend_state.backend_id, None)
                self._backend_id_by_backend_key.pop(backend_state.backend_key, None)
                should_shutdown_backend = True
        if should_shutdown_backend:
            backend_state.runtime.shutdown()

    def create_sampling_producer(
        self,
        sampler_input: Union[
            NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput, ABLPNodeSamplerInput
        ],
        sampling_config: SamplingConfig,
        worker_options: RemoteDistSamplingWorkerOptions,
        sampler_options: SamplerOptions,
    ) -> int:
        """Create a sampling producer by delegating to the two-phase API.

        Bridge method that keeps existing loaders working. Internally calls
        :meth:`init_sampling_backend` and :meth:`register_sampling_input`,
        returning the ``channel_id`` as the ``producer_id``.

        Args:
            sampler_input: The input data for sampling.
            sampling_config: Configuration of sampling meta info.
            worker_options: Options for launching remote sampling workers.
            sampler_options: Controls which sampler class is instantiated.

        Returns:
            A unique ID (channel_id) usable as a producer_id.
        """
        backend_id = self.init_sampling_backend(
            InitSamplingBackendRequest(
                backend_key=worker_options.worker_key,
                worker_options=worker_options,
                sampler_options=sampler_options,
                sampling_config=sampling_config,
            )
        )
        channel_id = self.register_sampling_input(
            RegisterBackendRequest(
                backend_id=backend_id,
                worker_key=worker_options.worker_key,
                sampler_input=sampler_input,
                sampling_config=sampling_config,
                buffer_capacity=worker_options.buffer_capacity,
                buffer_size=worker_options.buffer_size,
            )
        )
        return channel_id

    def destroy_sampling_producer(self, producer_id: int) -> None:
        """Destroy a sampling producer by delegating to :meth:`destroy_sampling_input`.

        Bridge method that keeps existing loaders working.

        Args:
            producer_id: The producer ID (channel_id) to destroy.
        """
        self.destroy_sampling_input(producer_id)

    def start_new_epoch_sampling(self, channel_id: int, epoch: int) -> None:
        """Start one new epoch on one registered channel.

        Args:
            channel_id: The ID of the channel to start the epoch on.
            epoch: The epoch number to start.

        Raises:
            RuntimeError: If the channel or its backend is not found.
        """
        with self._lock:
            channel_state = self._channel_state_by_channel_id.get(channel_id)
            if channel_state is None:
                raise RuntimeError(
                    f"start_new_epoch_sampling: channel_id={channel_id} not found"
                )
            backend_state = self._backend_state_by_backend_id.get(
                channel_state.backend_id
            )
        if backend_state is None:
            raise RuntimeError(
                f"start_new_epoch_sampling: backend for channel_id={channel_id} "
                f"backend_id={channel_state.backend_id} not found"
            )

        if channel_state.epoch >= epoch:
            return
        channel_state.epoch = epoch
        logger.info(
            f"Starting epoch channel_id={channel_id} backend_id={channel_state.backend_id} "
            f"epoch={epoch}"
        )
        backend_state.runtime.start_new_epoch_sampling(channel_id, epoch)

    def _log_fetch_stats_if_due(
        self, channel_id: int, worker_key: str, elapsed: float
    ) -> None:
        """Accumulate per-channel fetch timing and log every ``self._log_every_n`` fetches.

        ``log_every_n`` is tracked independently per ``channel_id``, so each
        channel's RPC thread updates its own counters without contention.

        Args:
            channel_id: The channel whose stats to update.
            worker_key: The worker key for logging.
            elapsed: The elapsed time for this fetch call.
        """
        stats = self._fetch_stats_by_channel_id.get(channel_id)
        if stats is None:
            stats = _ChannelFetchStats()
            self._fetch_stats_by_channel_id[channel_id] = stats
        stats.fetch_count += 1
        stats.fetch_total_elapsed += elapsed
        if elapsed >= FETCH_SLOW_LOG_SECS:
            stats.fetch_slow_count += 1
        if stats.fetch_count >= self._log_every_n:
            avg_elapsed = stats.fetch_total_elapsed / stats.fetch_count
            logger.info(
                f"fetch_one_sampled_message stats: worker_key={worker_key} "
                f"avg_elapsed={avg_elapsed:.3f}s "
                f"slow_count={stats.fetch_slow_count}/{stats.fetch_count}"
            )
            stats.fetch_count = 0
            stats.fetch_total_elapsed = 0.0
            stats.fetch_slow_count = 0

    def fetch_one_sampled_message(
        self, channel_id: int
    ) -> tuple[Optional[SampleMessage], bool]:
        """Fetch one sampled message from a registered channel.

        Args:
            channel_id: The ID of the channel to fetch from.

        Returns:
            A tuple of (message, is_done). If ``is_done`` is ``True``, no more
            messages will be produced for this epoch.
        """
        request_start_time = time.monotonic()
        with self._lock:
            channel_state = self._channel_state_by_channel_id.get(channel_id)
            if channel_state is None:
                return None, True
            backend_state = self._backend_state_by_backend_id.get(
                channel_state.backend_id
            )
        if backend_state is None:
            return None, True

        with channel_state.lock:
            while True:
                try:
                    msg = channel_state.channel.recv(timeout_ms=100)
                    self._log_fetch_stats_if_due(
                        channel_id,
                        channel_state.worker_key,
                        time.monotonic() - request_start_time,
                    )
                    return msg, False
                except QueueTimeoutError:
                    if (
                        backend_state.runtime.is_channel_epoch_done(
                            channel_id, channel_state.epoch
                        )
                        and channel_state.channel.empty()
                    ):
                        self._log_fetch_stats_if_due(
                            channel_id,
                            channel_state.worker_key,
                            time.monotonic() - request_start_time,
                        )
                        return None, True


_dist_server: Optional[DistServer] = None
r"""``DistServer`` instance of the current process."""


def get_server() -> DistServer:
    r"""Get the ``DistServer`` instance on the current process."""
    if _dist_server is None:
        raise RuntimeError("DistServer not initialized! Call init_server() first.")
    return _dist_server


def init_server(
    num_servers: int,
    server_rank: int,
    dataset: DistDataset,
    master_addr: str,
    master_port: int,
    num_clients: int = 0,
    num_rpc_threads: int = 16,
    request_timeout: int = 180,
    server_group_name: Optional[str] = None,
    is_dynamic: bool = False,
) -> None:
    r"""Initialize the current process as a server and establish connections
    with all other servers and clients. Note that this method should be called
    only in the server-client distribution mode.

    Args:
      num_servers (int): Number of processes participating in the server group.
      server_rank (int): Rank of the current process withing the server group (it
        should be a number between 0 and ``num_servers``-1).
      dataset (DistDataset): The ``DistDataset`` object of a partition of graph
        data and feature data, along with distributed patition book info.
      master_addr (str): The master TCP address for RPC connection between all
        servers and clients, the value of this parameter should be same for all
        servers and clients.
      master_port (int): The master TCP port for RPC connection between all
        servers and clients, the value of this parameter should be same for all
        servers and clients.
      num_clients (int): Number of processes participating in the client group.
        if ``is_dynamic`` is ``True``, this parameter will be ignored.
      num_rpc_threads (int): The number of RPC worker threads used for the
        current server to respond remote requests. (Default: ``16``).
      request_timeout (int): The max timeout seconds for remote requests,
        otherwise an exception will be raised. (Default: ``16``).
      server_group_name (str): A unique name of the server group that current
        process belongs to. If set to ``None``, a default name will be used.
        (Default: ``None``).
      is_dynamic (bool): Whether the world size is dynamic. (Default: ``False``).
    """
    if server_group_name:
        server_group_name = server_group_name.replace("-", "_")
    glt_dist_server._set_server_context(
        num_servers, server_rank, server_group_name, num_clients
    )
    global _dist_server
    _dist_server = DistServer(dataset=dataset)
    # Also set GLT's _dist_server so that GLT's RPC mechanism routes to GiGL's server
    glt_dist_server._dist_server = _dist_server
    init_rpc(
        master_addr,
        master_port,
        num_rpc_threads,
        request_timeout,
        is_dynamic=is_dynamic,
    )


def wait_and_shutdown_server() -> None:
    r"""Block until all client have been shutdowned, and further shutdown the
    server on the current process and destroy all RPC connections.
    """
    current_context = glt_dist_server.get_context()
    if current_context is None:
        logging.warning(
            "'wait_and_shutdown_server': try to shutdown server when "
            "the current process has not been initialized as a server."
        )
        return
    if not current_context.is_server():
        raise RuntimeError(
            f"'wait_and_shutdown_server': role type of "
            f"the current process context is not a server, "
            f"got {current_context.role}."
        )
    global _dist_server
    if _dist_server is not None:
        _dist_server.wait_for_exit()
        _dist_server.shutdown()
        _dist_server = None
        # Also clear GLT's _dist_server
        glt_dist_server._dist_server = None
    barrier()
    shutdown_rpc()


def _call_func_on_server(func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
    r"""A callee entry for remote requests on the server side."""
    if not callable(func):
        logging.warning(
            f"'_call_func_on_server': receive a non-callable " f"function target {func}"
        )
        return None

    server = get_server()
    if hasattr(server, func.__name__):
        # NOTE: method does not respect inheritance.
        # `func` is the full name of the function, e.g. gigl.distributed.graph_store.dist_server.DistServer.get_edge_dir
        # And so if something subclasses DistServer, the *base* class method will be called, not the subclass method.
        return func(server, *args, **kwargs)

    return func(*args, **kwargs)
