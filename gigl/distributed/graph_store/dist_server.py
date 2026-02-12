"""
GiGL implementation of GLT DistServer.

Main change here is that we add create_dist_sampling_ablp_producer for use with GiGL ABLP tasks.

Based on https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_server.py
"""

import logging
import threading
import time
import warnings
from typing import Optional, Union

import graphlearn_torch.distributed.dist_server as glt_dist_server
import torch
from graphlearn_torch.channel import QueueTimeoutError, SampleMessage, ShmChannel
from graphlearn_torch.distributed import (
    RemoteDistSamplingWorkerOptions,
    barrier,
    init_rpc,
    shutdown_rpc,
)
from graphlearn_torch.distributed.dist_sampling_producer import DistMpSamplingProducer
from graphlearn_torch.distributed.dist_server import DistServer as GltDistServer
from graphlearn_torch.partition import PartitionBook
from graphlearn_torch.sampler import (
    EdgeSamplerInput,
    NodeSamplerInput,
    RemoteSamplerInput,
    SamplingConfig,
)

from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_sampling_producer import DistABLPSamplingProducer
from gigl.distributed.sampler import ABLPNodeSamplerInput
from gigl.src.common.types.graph_data import EdgeType, NodeType

SERVER_EXIT_STATUS_CHECK_INTERVAL = 5.0
r""" Interval (in seconds) to check exit status of server.
"""

FETCH_MESSAGE_TIMEOUT_SECONDS = 300.0
r""" Overall timeout (in seconds) for fetch_one_sampled_message polling.
If a producer does not yield any sampled message within this duration,
the call raises a TimeoutError instead of polling forever.
"""


# TODO(kmonte): Migrate graph_store/storage_utils to this class.
class DistServer(GltDistServer):
    r"""A server that supports launching remote sampling workers for
    training clients.

    Note that this server is enabled only when the distribution mode is a
    server-client framework, and the graph and feature store will be partitioned
    and managed by all server nodes.

    Args:
      dataset (DistDataset): The ``DistDataset`` object of a partition of graph
        data and feature data, along with distributed patition books.
    """

    def __init__(self, dataset: DistDataset) -> None:
        self.dataset = dataset
        self._lock = threading.RLock()
        self._exit = False
        self._cur_producer_idx = 0  # auto incremental index (same as producer count)
        # The mapping from the key in worker options (such as 'train', 'test')
        # to producer id
        self._worker_key2producer_id: dict[str, int] = {}
        self._producer_pool: dict[int, DistABLPSamplingProducer] = {}
        self._msg_buffer_pool: dict[int, ShmChannel] = {}
        self._epoch: dict[int, int] = {}  # last epoch for the producer
        # TODO(kmonte): Re-enable this once we always use GiGL dist server.
        # self._producer_lock: dict[int, threading.RLock] = {}

    def shutdown(self) -> None:
        for producer_id in list(self._producer_pool.keys()):
            self.destroy_sampling_producer(producer_id)
        assert len(self._producer_pool) == 0
        assert len(self._msg_buffer_pool) == 0

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

    def create_sampling_ablp_producer(
        self,
        sampler_input: Union[
            NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput, ABLPNodeSamplerInput
        ],
        sampling_config: SamplingConfig,
        worker_options: RemoteDistSamplingWorkerOptions,
    ) -> int:
        r"""Create and initialize an instance of ``DistABLPSamplingProducer`` with
        a group of subprocesses for distributed sampling.

        Args:
          sampler_input (NodeSamplerInput or EdgeSamplerInput): The input data
            for sampling.
          sampling_config (SamplingConfig): Configuration of sampling meta info.
          worker_options (RemoteDistSamplingWorkerOptions): Options for launching
            remote sampling workers by this server.

        Returns:
          A unique id of created sampling producer on this server.
        """

        if not isinstance(sampler_input, ABLPNodeSamplerInput):
            raise ValueError(
                f"Sampler input must be an instance of ABLPNodeSamplerInput. Received: {type(sampler_input)}"
            )

        return self._create_producer(
            sampler_input=sampler_input,
            sampling_config=sampling_config,
            worker_options=worker_options,
            producer_cls=DistABLPSamplingProducer,
        )

    def create_sampling_producer(
        self,
        sampler_input: Union[
            NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput, ABLPNodeSamplerInput
        ],
        sampling_config: SamplingConfig,
        worker_options: RemoteDistSamplingWorkerOptions,
    ) -> int:
        r"""Create and initialize an instance of ``DistSamplingProducer`` with
        a group of subprocesses for distributed sampling.

        Args:
          sampler_input (NodeSamplerInput or EdgeSamplerInput): The input data
            for sampling.
          sampling_config (SamplingConfig): Configuration of sampling meta info.
          worker_options (RemoteDistSamplingWorkerOptions): Options for launching
            remote sampling workers by this server.

        Returns:
          A unique id of created sampling producer on this server.
        """
        return self._create_producer(
            sampler_input=sampler_input,
            sampling_config=sampling_config,
            worker_options=worker_options,
            producer_cls=DistMpSamplingProducer,
        )

    def _create_producer(
        self,
        sampler_input: Union[
            NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput, ABLPNodeSamplerInput
        ],
        sampling_config: SamplingConfig,
        worker_options: RemoteDistSamplingWorkerOptions,
        producer_cls: type[Union[DistABLPSamplingProducer, DistMpSamplingProducer]],
    ) -> int:
        r"""Shared logic to create and initialize a sampling producer.

        Converts remote sampler inputs to local, creates a ``ShmChannel`` buffer,
        instantiates the given ``producer_cls``, and registers it in the internal pools.

        Args:
          sampler_input (NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput,
            or ABLPNodeSamplerInput): The input data for sampling.
          sampling_config (SamplingConfig): Configuration of sampling meta info.
          worker_options (RemoteDistSamplingWorkerOptions): Options for launching
            remote sampling workers by this server.
          producer_cls: The producer class to instantiate
            (``DistABLPSamplingProducer`` or ``DistMpSamplingProducer``).

        Returns:
          int: A unique id of created sampling producer on this server.
        """
        if isinstance(sampler_input, RemoteSamplerInput):
            sampler_input = sampler_input.to_local_sampler_input(dataset=self.dataset)

        with self._lock:
            producer_id = self._worker_key2producer_id.get(worker_options.worker_key)
            if producer_id is None:
                producer_id = self._cur_producer_idx
                self._worker_key2producer_id[worker_options.worker_key] = producer_id
                self._cur_producer_idx += 1
                buffer = ShmChannel(
                    worker_options.buffer_capacity, worker_options.buffer_size
                )
                producer = producer_cls(
                    self.dataset, sampler_input, sampling_config, worker_options, buffer
                )
                producer.init()
                # TODO(kmonte): Re-enable this once we always use GiGL dist server.
                # self._producer_lock[producer_id] = threading.RLock()
                self._producer_pool[producer_id] = producer
                self._msg_buffer_pool[producer_id] = buffer
                self._epoch[producer_id] = -1
        return producer_id

    def destroy_sampling_producer(self, producer_id: int) -> None:
        r"""Shutdown and destroy a sampling producer managed by this server with
        its producer id.
        """
        with self._lock:
            producer = self._producer_pool.get(producer_id, None)
            if producer is not None:
                producer.shutdown()
                self._producer_pool.pop(producer_id)
                self._msg_buffer_pool.pop(producer_id)
                self._epoch.pop(producer_id)
                # TODO(kmonte): Re-enable this once we always use GiGL dist server.
                # self._producer_lock.pop(producer_id)
                # Clean up worker_key -> producer_id mapping to avoid stale
                # cache entries if a loader with the same key is re-created.
                keys_to_remove = [
                    k
                    for k, v in self._worker_key2producer_id.items()
                    if v == producer_id
                ]
                for k in keys_to_remove:
                    self._worker_key2producer_id.pop(k)
                print(
                    f"Remaining producers: {len(self._producer_pool)}, ({sorted(self._producer_pool.keys())})"
                )

    def start_new_epoch_sampling(self, producer_id: int, epoch: int) -> None:
        r"""Start a new epoch sampling tasks for a specific sampling producer
        with its producer id.
        """
        with self._lock:
            cur_epoch = self._epoch[producer_id]
            if cur_epoch < epoch:
                self._epoch[producer_id] = epoch
                producer = self._producer_pool.get(producer_id, None)
                if producer is not None:
                    producer.produce_all()

    def fetch_one_sampled_message(
        self, producer_id: int
    ) -> tuple[Optional[SampleMessage], bool]:
        r"""Fetch a sampled message from the buffer of a specific sampling
        producer with its producer id.

        This method polls the producer's ShmChannel buffer with a 500ms
        per-iteration timeout. An overall timeout of
        ``FETCH_MESSAGE_TIMEOUT_SECONDS`` prevents indefinite blocking
        when sampling workers are unable to produce data (e.g. due to
        resource contention or worker failure).
        """
        producer = self._producer_pool.get(producer_id, None)
        if producer is None:
            warnings.warn(f"invalid producer_id {producer_id}")
            return None, False
        if producer.is_all_sampling_completed_and_consumed():
            return None, True
        buffer = self._msg_buffer_pool.get(producer_id, None)
        start_time = time.monotonic()
        while True:
            try:
                msg = buffer.recv(timeout_ms=500)
                return msg, False
            except QueueTimeoutError:
                if producer.is_all_sampling_completed():
                    return None, True
                elapsed = time.monotonic() - start_time
                if elapsed > FETCH_MESSAGE_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        f"fetch_one_sampled_message timed out after "
                        f"{elapsed:.1f}s for producer {producer_id}. "
                        f"Sampling workers may have failed or are under "
                        f"extreme resource contention."
                    )


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


def _call_func_on_server(func, *args, **kwargs):
    r"""A callee entry for remote requests on the server side."""
    if not callable(func):
        logging.warning(
            f"'_call_func_on_server': receive a non-callable " f"function target {func}"
        )
        return None

    server = get_server()
    if hasattr(server, func.__name__):
        return func(server, *args, **kwargs)

    return func(*args, **kwargs)
