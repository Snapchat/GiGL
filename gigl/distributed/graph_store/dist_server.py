"""
GiGL implementation of GLT DistServer.

Uses GiGL's DistSamplingProducer which supports neighbor sampling
and ABLP (anchor-based link prediction) via BaseGiGLSampler subclasses
(DistNeighborSampler for k-hop, DistPPRNeighborSampler for PPR).

Based on https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_server.py

Teardown protocol
-----------------
Cluster teardown is a strict three-phase sequence:

1. **Per-loader (compute side).** Each ``DistLoader.shutdown()`` issues
   ``DistServer.destroy_sampling_input(channel_id)`` against every
   storage server it registered with. This call performs the actual
   teardown work on the server: ``runtime.unregister_input`` for the
   channel and, when the channel is the last on its backend,
   ``runtime.shutdown`` for the backend.
2. **Per-compute-process.** After all loaders are torn down, the
   compute process calls
   ``gigl.distributed.graph_store.compute.shutdown_compute_process``,
   which calls ``glt.distributed.shutdown_client``
   and tears down the compute torch process group.
3. **Per-storage-process.** ``wait_and_shutdown_server`` blocks until
   ``DistServer.exit`` flips, then runs ``DistServer.shutdown()``
   (strict validation — see below), ``barrier()``, ``shutdown_rpc()``.

An example fo the teardown protocol:
```python
def compute_process():
    train_loader = ...
    val_loader = ...
    test_loader = ...

    loaders = [train_loader, val_loader, test_loader]
    for data in train_loader:
        train(data)
        if should_val():
            for data in val_loader:
                val(data)
    # Shutdown the loaders after training and validation.
    train_loader.shutdown()
    val_loader.shutdown()

    for data in test_loader:
        test(data)
    # Shutdown the loader after testing.
    test_loader.shutdown()


    # Step 2: Per-compute-process
    shutdown_compute_process()

# Step 3: Per-storage-process
wait_and_shutdown_server()
```

``DistServer.shutdown()`` does no teardown work itself. It validates
that phase 1 ran for every channel/backend, raises ``RuntimeError`` if
state remains, and drops residual tombstone/stats bookkeeping. The
actual ``runtime.shutdown()`` work happens in
``destroy_sampling_input`` when the last channel on a backend is
destroyed. ``wait_and_shutdown_server`` catches ``DistServer.shutdown``
exceptions so a buggy compute client cannot wedge healthy storage
peers on the barrier; the failing storage process re-raises after the
barrier so the orchestrator sees a non-zero exit.
"""

import threading
import time
from collections import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, TypeVar, Union

import graphlearn_torch.distributed.dist_server as glt_dist_server
import torch
from graphlearn_torch.channel import QueueTimeoutError, SampleMessage, ShmChannel
from graphlearn_torch.distributed import barrier, init_rpc, shutdown_rpc
from graphlearn_torch.partition import PartitionBook
from graphlearn_torch.sampler import RemoteSamplerInput

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
from gigl.distributed.sampler_options import PPRSamplerOptions
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import FeatureInfo, reverse_edge_type, select_label_edge_types
from gigl.utils.data_splitters import get_labels_for_anchor_nodes
from gigl.utils.share_memory import share_memory

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
        tombstoned: Terminal-state tombstone. Set to ``True`` once
            ``destroy_sampling_input`` has finished cleaning up this
            channel. The channel object may briefly outlive its
            registry entry — an in-flight ``fetch_one_sampled_message``
            on another RPC thread reads this flag to break out of its
            recv loop. Past-participle name: this is "done" state, not
            "in progress" state. Asymmetric with
            ``SamplingBackendState.tearing_down`` by design — backends
            are removed from the registry when teardown completes, so
            no terminal flag is needed there; channels are not removed
            until the in-flight fetch observes the tombstone, so the
            flag has to live on the object.
    """

    backend_id: int
    worker_key: str
    channel: ShmChannel
    epoch: int = -1
    lock: threading.RLock = field(default_factory=threading.RLock)
    tombstoned: bool = False


@dataclass
class SamplingBackendState:
    """Per-backend state for a shared sampling backend.

    Args:
        backend_id: The unique ID of this backend.
        backend_key: The key identifying this backend
            (e.g. ``"dist_neighbor_loader_0"``).
        runtime: The shared sampling backend runtime.
        active_channels: Set of channel IDs currently registered on this backend.
        lock: A reentrant lock guarding backend-level operations.
        init_complete: Whether ``runtime.init_backend()`` has completed successfully.
        init_error: If ``runtime.init_backend()`` raised, the exception;
            otherwise ``None``.
        tearing_down: In-progress marker. Set to ``True`` while
            ``destroy_sampling_input`` is tearing down this backend's
            runtime. A second-caller of ``init_sampling_backend`` that
            blocks on ``lock`` and observes this flag raises
            ``RuntimeError`` so it does not reuse a half-shutdown
            runtime. Present-participle name: this is "in progress"
            state, not "done" state. Asymmetric with
            ``ChannelState.tombstoned`` by design — the registry entry
            is removed once teardown completes, so no terminal flag is
            needed; after removal nothing can find this object.
    """

    backend_id: int
    backend_key: str
    runtime: SharedDistSamplingBackend
    active_channels: set[int] = field(default_factory=set)
    lock: threading.RLock = field(default_factory=threading.RLock)
    init_complete: bool = False
    init_error: Optional[BaseException] = None
    tearing_down: bool = False


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
        data and feature data, along with distributed partition books.
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
        self._tombstoned_channel_ids: set[int] = set()
        self._fetch_stats_by_channel_id: dict[int, _ChannelFetchStats] = {}
        self._log_every_n = log_every_n

    def shutdown(self) -> None:
        """Final post-teardown bookkeeping cleanup — does not perform any shutdown.

        Lenient contract: callers are expected to have torn down every
        channel and backend via ``destroy_sampling_input`` before
        invoking this method. The actual ``runtime.shutdown()`` work
        happens in ``destroy_sampling_input`` when the last channel on
        a backend is destroyed; this method only drops residual
        tombstone/stats bookkeeping and warns if any sampling state is
        still registered.
        """
        with self._lock:
            if (
                self._channel_state_by_channel_id
                or self._backend_state_by_backend_id
                or self._backend_id_by_backend_key
            ):
                logger.warning(
                    "DistServer.shutdown() called with live channels/backends; "
                    "leaving residual state for process exit. "
                    f"live_channels={sorted(self._channel_state_by_channel_id)} "
                    f"live_backends={sorted(self._backend_state_by_backend_id)} "
                    f"live_backend_keys={sorted(self._backend_id_by_backend_key)}"
                )
            # Always drop bookkeeping. Registry dicts (channels/backends)
            # are also dropped so a subsequent shutdown call is a clean
            # no-op rather than re-warning on the same residual state.
            self._channel_state_by_channel_id.clear()
            self._backend_state_by_backend_id.clear()
            self._backend_id_by_backend_key.clear()
            self._tombstoned_channel_ids.clear()
            self._fetch_stats_by_channel_id.clear()

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

    def get_edge_weights_registered(self) -> bool:
        """Return whether edge weights were registered in the dataset.

        Returns:
            True if edge weights were registered via ``DistPartitioner.register_edge_weights()``.
        """
        return self.dataset.has_edge_weights

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
            nodes = nodes[node_type]  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
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

        The caller always provides ``supervision_edge_type`` in outward form
        ``(anchor_node_type, relation, supervision_node_type)``, matching the
        taskMetadata convention.

        When the dataset is in-sampled (``edge_dir == "in"``) the stored positive/negative
        label edges are reversed relative to that outward form, so we reverse the supervision
        edge type before resolving the stored label edge types.
        This mirrors the colocated loader's handling in
        :meth:`~gigl.distributed.dist_ablp_neighborloader.DistABLPLoader._setup_for_colocated`.

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
        label_lookup_edge_type = (
            reverse_edge_type(request.supervision_edge_type)
            if self.dataset.edge_dir == "in"
            else request.supervision_edge_type
        )
        positive_label_edge_type, negative_label_edge_type = select_label_edge_types(
            label_lookup_edge_type, self.dataset.get_edge_types()
        )
        positive_labels, negative_labels = get_labels_for_anchor_nodes(
            self.dataset,
            anchors,
            positive_label_edge_type,
            negative_label_edge_type,
            max_labels_per_anchor_node=self.dataset.max_labels_per_anchor_node,
        )
        return anchors, positive_labels, negative_labels

    def init_sampling_backend(self, opts: InitSamplingBackendRequest) -> int:
        """Create or reuse a shared sampling backend for one loader instance.

        If a backend with the same ``backend_key`` already exists, blocks
        until the original initializer has finished, then returns its ID.
        If the original initializer failed, re-raises that failure so the
        second caller does not see a half-initialized backend.

        Args:
            opts: The initialization request containing the backend key,
                worker options, sampler options, and sampling config.

        Returns:
            The unique backend ID.

        Raises:
            RuntimeError: If a prior concurrent initialization for the same
                ``backend_key`` failed, or if the backend is currently being
                torn down by ``destroy_sampling_input``.
        """
        request_start_time = time.monotonic()
        is_first = False
        with self._lock:
            backend_id = self._backend_id_by_backend_key.get(opts.backend_key)
            if backend_id is not None:
                backend_state = self._backend_state_by_backend_id[backend_id]
            else:
                backend_id = self._next_backend_id
                self._next_backend_id += 1
                degree_tensors = None
                if isinstance(opts.sampler_options, PPRSamplerOptions):
                    degree_tensors = self.dataset.degree_tensor
                    share_memory(degree_tensors)
                backend_state = SamplingBackendState(
                    backend_id=backend_id,
                    backend_key=opts.backend_key,
                    runtime=SharedDistSamplingBackend(
                        data=self.dataset,
                        worker_options=opts.worker_options,
                        sampling_config=opts.sampling_config,
                        sampler_options=opts.sampler_options,
                        # We only need degree tensor for PPR sampling
                        degree_tensors=degree_tensors,
                    ),
                )
                self._backend_id_by_backend_key[opts.backend_key] = backend_id
                self._backend_state_by_backend_id[backend_id] = backend_state
                # Acquire backend_state.lock while still holding self._lock.
                # We're about to release self._lock and run init_backend()
                # outside it (init spawns workers and blocks on a Barrier;
                # holding self._lock across that would block every other
                # RPC). Without this overlap a second caller could arrive
                # after we release self._lock but before we acquire
                # backend_state.lock, win the lock, and observe an
                # uninitialized backend. Acquiring here guarantees the
                # first caller reaches backend_state.lock first.
                backend_state.lock.acquire()
                is_first = True

        if is_first:
            try:
                init_start_time = time.monotonic()
                backend_state.runtime.init_backend()
                backend_state.init_complete = True
                init_elapsed = time.monotonic() - init_start_time
                total_elapsed = time.monotonic() - request_start_time
                logger.info(
                    f"Initialized sampling backend backend_key={opts.backend_key} "
                    f"backend_id={backend_id} "
                    f"init_backend={init_elapsed:.2f}s total={total_elapsed:.2f}s"
                )
            except BaseException as e:
                backend_state.init_error = e
                with self._lock:
                    self._backend_id_by_backend_key.pop(opts.backend_key, None)
                    self._backend_state_by_backend_id.pop(backend_id, None)
                raise
            finally:
                # Manual release: this lock crosses the self._lock boundary
                # (acquired above while holding self._lock) so `with` cannot
                # be used.
                backend_state.lock.release()
        else:
            # Second-caller path. Block on backend_state.lock until the
            # first caller finishes init (or fails). Ordering is enforced
            # by the acquire-before-self._lock-release invariant above.
            with backend_state.lock:
                if backend_state.tearing_down:
                    raise RuntimeError(
                        f"init_sampling_backend: backend_key={opts.backend_key} "
                        f"is being torn down; a new backend cannot be "
                        f"initialized until teardown completes. This is an odd bug, please report it."
                    )
                if backend_state.init_error is not None:
                    raise RuntimeError(
                        f"init_sampling_backend: prior initialization failed "
                        f"for backend_key={opts.backend_key}"
                    ) from backend_state.init_error
                assert backend_state.init_complete
        return backend_id

    def register_sampling_input(self, opts: RegisterBackendRequest) -> int:
        """Register one compute-rank input channel on an existing backend.

        Args:
            opts: The registration request containing the backend ID,
                worker key, sampler input, sampling config, and buffer settings.

        Returns:
            The unique channel ID for this input.

        Raises:
            KeyError: If ``opts.backend_id`` does not refer to a registered
                backend (caller bug; backend must be registered first).
            Exception: Re-raises any failure from
                ``runtime.register_input`` after rolling back the partial
                channel state.
        """
        request_start_time = time.monotonic()
        sampler_input = opts.sampler_input

        if isinstance(sampler_input, RemoteSamplerInput):
            sampler_input = sampler_input.to_local_sampler_input(dataset=self.dataset)

        with self._lock:
            backend_state = self._backend_state_by_backend_id[opts.backend_id]
            channel_id = self._next_channel_id
            self._next_channel_id += 1
            # If the sampler input is empty, we create a channel with 1 slot and 1MB size
            # We do this to save on memory usage for empty inputs.
            # NOTE: We must keep creating these channels as we need to "register input" for
            # all nodes on the storage cluster, as they the `NeighborSampler` is responsible for
            # serving incoming sampling requests as well as sending them out.
            # TODO(kmonte): Look into either supporting truly empty channels or having a shared
            # DistSampler.
            if len(sampler_input) == 0:
                channel = ShmChannel(1, "1MB")
            else:
                channel = ShmChannel(opts.buffer_capacity, opts.buffer_size)
            channel_state = ChannelState(
                backend_id=opts.backend_id,
                worker_key=opts.worker_key,
                channel=channel,
            )
            self._channel_state_by_channel_id[channel_id] = channel_state
            backend_state.active_channels.add(channel_id)
            # Snapshot the count under self._lock so the log message
            # reflects state at the moment of registration, not a later
            # value that could be mutated by concurrent register/destroy.
            active_channels_at_register = len(backend_state.active_channels)

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
            f"active_channels={active_channels_at_register} "
            f"in {time.monotonic() - request_start_time:.2f}s"
        )
        return channel_id

    def destroy_sampling_input(self, channel_id: int) -> None:
        """Destroy one registered sampling channel and maybe its backend.

        If this is the last channel on the backend, the backend is shut
        down and removed.

        Args:
            channel_id: The ID of the channel to destroy.
        """
        request_start_time = time.monotonic()
        backend_id_for_log: Optional[int] = None
        triggered_backend_shutdown = False
        try:
            with self._lock:
                channel_state = self._channel_state_by_channel_id.get(channel_id)
                if channel_state is None:
                    # Idempotent re-destroy: the channel was already torn
                    # down by a prior destroy_sampling_input call, which
                    # popped the channel, popped its fetch stats, and set
                    # ``tombstoned=True``. Just make sure the tombstone is
                    # present so a concurrent start_new_epoch_sampling
                    # silent-no-ops rather than raising "unknown
                    # channel_id". No re-pop here — the prior destroy
                    # already did the work, and this branch must not
                    # duplicate the backend-still-registered cleanup
                    # below.
                    self._tombstoned_channel_ids.add(channel_id)
                    return
                backend_state = self._backend_state_by_backend_id.get(
                    channel_state.backend_id
                )
                backend_id_for_log = channel_state.backend_id
                if backend_state is None:
                    # Defensive: in current code this is unreachable. The
                    # invariant is that a channel's backend outlives the
                    # channel — the backend registry entry is popped only
                    # inside the ``should_shutdown_backend`` branch below,
                    # which fires only when
                    # ``backend_state.active_channels`` is empty. Reaching
                    # this branch would mean a future code path popped the
                    # backend without popping its channels, or test code
                    # set up the state by hand. Treat it as a partial
                    # teardown: tombstone the channel, drop bookkeeping,
                    # and bail.
                    self._tombstoned_channel_ids.add(channel_id)
                    self._channel_state_by_channel_id.pop(channel_id, None)
                    self._fetch_stats_by_channel_id.pop(channel_id, None)
                    channel_state.tombstoned = True
                    return

            # Two-lock dance:
            #
            # * ``backend_state.lock`` serializes this destroy against
            #   ``init_sampling_backend`` (second-caller),
            #   ``register_sampling_input``, and
            #   ``start_new_epoch_sampling`` for the same backend. We hold
            #   it across the runtime work
            #   (``unregister_input``, ``shutdown``) so a concurrent init
            #   for the same backend_key blocks until we have either
            #   finished or set ``tearing_down=True``.
            # * ``self._lock`` protects registry-dict mutations only. We
            #   acquire it under ``backend_state.lock`` for short
            #   re-validation windows and release it before the runtime
            #   calls so other RPCs (fetches, inits for *other* backends)
            #   are not blocked behind a worker join.
            #
            # Re-validate channel/backend state under ``self._lock``
            # because another call could have raced between the first
            # lock-drop and our acquire of ``backend_state.lock``.
            should_shutdown_backend = False
            with backend_state.lock:
                with self._lock:
                    current_channel_state = self._channel_state_by_channel_id.get(
                        channel_id
                    )
                    if current_channel_state is None:
                        return
                    if current_channel_state.backend_id != backend_state.backend_id:
                        return
                    current_channel_state.tombstoned = True
                    self._tombstoned_channel_ids.add(channel_id)
                    self._channel_state_by_channel_id.pop(channel_id, None)
                    self._fetch_stats_by_channel_id.pop(channel_id, None)
                # ``runtime.unregister_input`` joins the per-channel
                # worker. We release ``self._lock`` first (above) so
                # unrelated RPCs (fetches on other channels, inits for
                # other backends) are not blocked behind the worker join.
                # ``backend_state.lock`` is still held, so any concurrent
                # operation on *this* backend is serialized against us.
                # The channel is already popped from the registry and
                # tombstoned, so other callers see it as gone.
                backend_state.runtime.unregister_input(channel_id)
                with self._lock:
                    backend_state.active_channels.discard(channel_id)
                    if not backend_state.active_channels:
                        should_shutdown_backend = True
                # Shut the runtime down BEFORE popping the registry entries,
                # while still holding backend_state.lock. A concurrent
                # init_sampling_backend for the same backend_key will find
                # this still-registered backend, fall through to the
                # second-caller path, block on backend_state.lock, and see
                # tearing_down=True so it raises rather than reusing a
                # half-shutdown runtime.
                if should_shutdown_backend:
                    triggered_backend_shutdown = True
                    backend_state.tearing_down = True
                    backend_state.runtime.shutdown()
                    with self._lock:
                        self._backend_state_by_backend_id.pop(
                            backend_state.backend_id, None
                        )
                        self._backend_id_by_backend_key.pop(
                            backend_state.backend_key, None
                        )
        finally:
            # Slow-path warning: a healthy destroy is dominated by
            # ``runtime.unregister_input`` (an enqueue) and, on the last
            # channel, ``runtime.shutdown()`` (workers joined with a 5 s
            # timeout each, then terminated). >1 s on the non-last
            # destroy or >30 s on the last destroy is the smoking-gun
            # signal we are missing today.
            elapsed = time.monotonic() - request_start_time
            slow_threshold = 30.0 if triggered_backend_shutdown else 1.0
            if elapsed > slow_threshold:
                logger.warning(
                    f"destroy_sampling_input slow channel_id={channel_id} "
                    f"backend_id={backend_id_for_log} "
                    f"triggered_backend_shutdown={triggered_backend_shutdown} "
                    f"elapsed={elapsed:.3f}s"
                )

    def start_new_epoch_sampling(self, channel_id: int, epoch: int) -> None:
        """Start one new epoch on one registered channel.

        No-op if this channel has already started ``epoch`` or a later
        epoch (idempotent — safe to call repeatedly from retries).

        No-op if the channel is in the tombstoned set: this handles the
        legitimate destroy/start race window where a compute peer's
        start RPC arrives after destroy has landed.

        Args:
            channel_id: The ID of the channel to start the epoch on.
            epoch: The epoch number to start.

        Raises:
            RuntimeError: If ``channel_id`` was never registered on this
                server (vs. legitimately tombstoned — tombstoned ids
                are treated as a silent no-op).
        """
        with self._lock:
            channel_state = self._channel_state_by_channel_id.get(channel_id)
            if channel_state is None:
                # Tombstoned: legitimate destroy/start race where the
                # compute peer's start RPC arrived after destroy
                # already landed on the server. Silent no-op.
                if channel_id in self._tombstoned_channel_ids:
                    return
                # Never registered — caller bug. Loud.
                raise RuntimeError(
                    f"start_new_epoch_sampling: unknown channel_id={channel_id}"
                )
            backend_state = self._backend_state_by_backend_id.get(
                channel_state.backend_id
            )
        if backend_state is None:
            # Backend was torn down between the channel-lookup above
            # and the backend-lookup. The channel will be tombstoned
            # by the destroy that popped the backend. Treat as
            # tombstoned.
            return

        # Same two-lock dance as destroy_sampling_input (see comment
        # there). We hold ``backend_state.lock`` across the runtime
        # call so destroy/init for this backend serialize behind us,
        # and we acquire ``self._lock`` only for short re-validation /
        # epoch-update windows so other RPCs are not blocked behind
        # the runtime dispatch.
        with backend_state.lock:
            with self._lock:
                # Re-validate everything: state could have changed
                # between our first lookup (under self._lock) and our
                # acquire of backend_state.lock. In particular, a
                # destroy that ran in that window may have popped the
                # channel, swapped the backend, or set
                # ``tombstoned=True``.
                current_channel_state = self._channel_state_by_channel_id.get(
                    channel_id
                )
                current_backend_state = self._backend_state_by_backend_id.get(
                    channel_state.backend_id
                )
                if current_channel_state is None:
                    return
                if current_backend_state is not backend_state:
                    # Backend was rebuilt under the same backend_id
                    # (or the channel was reassigned) — bail; this
                    # RPC is for stale state.
                    return
                if current_channel_state.backend_id != backend_state.backend_id:
                    return
                if current_channel_state.tombstoned:
                    return
                if current_channel_state.epoch >= epoch:
                    # Idempotent: same or later epoch already running.
                    return
                worker_key = current_channel_state.worker_key
            logger.info(
                f"Starting epoch channel_id={channel_id} backend_id={channel_state.backend_id} "
                f"epoch={epoch}"
            )
            # Dispatch outside self._lock — runtime.start_new_epoch_sampling
            # may take work; same reasoning as the unregister_input
            # call in destroy_sampling_input.
            backend_state.runtime.start_new_epoch_sampling(channel_id, epoch)
            with self._lock:
                # Re-check tombstoned once more before bumping epoch:
                # a destroy that landed during the dispatch above must
                # not observe ``epoch`` advanced past it.
                post_dispatch_channel_state = self._channel_state_by_channel_id.get(
                    channel_id
                )
                if post_dispatch_channel_state is None:
                    return
                if post_dispatch_channel_state.tombstoned:
                    return
                post_dispatch_channel_state.epoch = epoch
            logger.debug(
                "start_new_epoch_sampling dispatched "
                f"channel_id={channel_id} worker_key={worker_key} epoch={epoch}"
            )

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
                    if channel_state.tombstoned:
                        self._log_fetch_stats_if_due(
                            channel_id,
                            channel_state.worker_key,
                            time.monotonic() - request_start_time,
                        )
                        return None, True
                    with self._lock:
                        current_backend_state = self._backend_state_by_backend_id.get(
                            channel_state.backend_id
                        )
                    if current_backend_state is not backend_state:
                        self._log_fetch_stats_if_due(
                            channel_id,
                            channel_state.worker_key,
                            time.monotonic() - request_start_time,
                        )
                        return None, True
                    # Timeout is expected whenever the producer has
                    # nothing ready; poll is_channel_epoch_done and loop.
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
        data and feature data, along with distributed partition book info.
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
    r"""Block until all clients have shut down, then shut down the
    server on the current process and destroy all RPC connections.

    Best-effort cluster liveness: if ``DistServer.shutdown`` raises
    (e.g. because a client crashed leaving state behind) we capture
    the exception, run ``barrier()`` + ``shutdown_rpc()`` so healthy
    storage peers do not hang on the barrier, then re-raise so the
    orchestrator sees a non-zero exit on the failing storage process.

    Step 3 of the three-phase teardown described in the module
    docstring.
    """
    current_context = glt_dist_server.get_context()
    if current_context is None:
        logger.warning(
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
    shutdown_error: Optional[BaseException] = None
    if _dist_server is not None:
        _dist_server.wait_for_exit()
        try:
            _dist_server.shutdown()
        except Exception as exc:
            # Capture-and-continue so healthy peers clear the barrier.
            # Re-raised after barrier + shutdown_rpc so the orchestrator
            # sees the failure as a non-zero exit. Broad except by
            # design — any failure during shutdown must not skip the
            # barrier (see commit 2eeb0c7d for the original wedge fix).
            shutdown_error = exc
            logger.exception(
                "DistServer.shutdown() failed during cluster teardown; "
                "continuing with barrier/shutdown_rpc, will re-raise after."
            )
        _dist_server = None
        # Also clear GLT's _dist_server (we set it in init_server so
        # GLT's RPC mechanism routes through us; leaving it would let
        # the next init_server call observe stale state).
        glt_dist_server._dist_server = None
    barrier()
    shutdown_rpc()
    if shutdown_error is not None:
        raise shutdown_error


def _call_func_on_server(func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
    r"""A callee entry for remote requests on the server side."""
    if not callable(func):
        logger.warning(
            f"'_call_func_on_server': receive a non-callable function target {func}"
        )
        return None  # type: ignore[return-value]  # ty: ignore[invalid-return-type]

    server = get_server()
    func_name = getattr(func, "__name__", None)
    if func_name is not None and hasattr(server, func_name):
        # NOTE: method does not respect inheritance.
        # `func` is the full name of the function, e.g. gigl.distributed.graph_store.dist_server.DistServer.get_edge_dir
        # And so if something subclasses DistServer, the *base* class method will be called, not the subclass method.
        return func(server, *args, **kwargs)

    return func(*args, **kwargs)
