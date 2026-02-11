"""
SamplingEngine abstraction for GiGL's distributed data loading.

Provides a clean interface over the two sampling modes GiGL uses:
- Colocated: graph data and compute on the same machines (MpDistSamplingWorkerOptions)
- Graph Store: graph data on separate storage nodes (RemoteDistSamplingWorkerOptions)
"""

import concurrent.futures
from abc import ABC, abstractmethod
from typing import Callable, List, Union

import torch
from graphlearn_torch.channel import SampleMessage, ShmChannel
from graphlearn_torch.distributed.dist_client import request_server
from graphlearn_torch.distributed.dist_sampling_producer import DistMpSamplingProducer
from graphlearn_torch.distributed.rpc import rpc_is_initialized
from graphlearn_torch.sampler import (
    EdgeSamplerInput,
    NodeSamplerInput,
    RemoteSamplerInput,
    SamplingConfig,
)

from gigl.common.logger import Logger
from gigl.distributed.graph_store.dist_server import DistServer

logger = Logger()


class SamplingEngine(ABC):
    """Abstracts the lifecycle of distributed sample production and consumption.

    Concrete implementations handle the two sampling modes used by GiGL:
    colocated (Mp workers) and graph store (remote workers).
    """

    @abstractmethod
    def start_epoch(self, epoch: int) -> None:
        """Signal the start of a new epoch. Implementations trigger sampling."""
        ...

    @abstractmethod
    def get_sample(self) -> SampleMessage:
        """Return the next sampled message (blocking)."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release all resources (channels, subprocesses, RPC connections)."""
        ...

    @property
    @abstractmethod
    def num_expected(self) -> Union[int, float]:
        """Number of batches expected per epoch. float('inf') for graph store mode."""
        ...


class ColocatedSamplingEngine(SamplingEngine):
    """Sampling engine for GiGL's colocated mode.

    Wraps a DistMpSamplingProducer (or any subclass, e.g. DistABLPSamplingProducer)
    and a ShmChannel. Used when graph data and compute live on the same machines.
    """

    def __init__(
        self,
        producer: DistMpSamplingProducer,
        channel: ShmChannel,
        input_len: int,
        batch_size: int,
        drop_last: bool,
    ):
        self._producer = producer
        self._channel = channel
        self._num_expected = input_len // batch_size
        if not drop_last and input_len % batch_size != 0:
            self._num_expected += 1

    def start_epoch(self, epoch: int) -> None:
        self._producer.produce_all()

    def get_sample(self) -> SampleMessage:
        return self._channel.recv()

    def shutdown(self) -> None:
        self._producer.shutdown()

    @property
    def num_expected(self) -> int:
        return self._num_expected


class GraphStoreSamplingEngine(SamplingEngine):
    """Sampling engine for GiGL's graph store mode.

    Manages server-side producer lifecycle via RPC. Used when graph data lives
    on separate storage nodes.

    The initialization is split into two phases:
    - ``__init__``: Stores configuration (safe to run without barriers).
    - ``setup_rpc()``: Dispatches RPCs to create producers on each server.
      This must be called inside the per-compute-node barrier loop to avoid
      race conditions in TensorPipe rendezvous.
    """

    def __init__(
        self,
        server_ranks: List[int],
        input_data_list: List[
            Union[NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput]
        ],
        sampling_config: SamplingConfig,
        worker_options,  # RemoteDistSamplingWorkerOptions
        server_create_fn: Callable = DistServer.create_sampling_producer,
    ):
        self._server_ranks = server_ranks
        self._input_data_list = input_data_list
        self._sampling_config = sampling_config
        self._worker_options = worker_options
        self._server_create_fn = server_create_fn
        self._producer_ids: List[int] = []
        self._channel = None  # Will be set by setup_rpc()
        self._shutdowned = False

        # Fetch dataset metadata from the first server
        (
            self.num_data_partitions,
            self.data_partition_idx,
            self.node_types,
            self.edge_types,
        ) = request_server(self._server_ranks[0], DistServer.get_dataset_meta)

        # Determine input_type from the first input data
        self._input_type = self._input_data_list[0].input_type

    def setup_rpc(self) -> None:
        """Dispatch RPCs to create producers on each server.

        This method must be called inside the per-compute-node barrier loop
        to avoid race conditions in TensorPipe rendezvous.
        """
        # Move input data to CPU for serialization
        for input_data in self._input_data_list:
            if not isinstance(input_data, RemoteSamplerInput):
                input_data = input_data.to(torch.device("cpu"))

        # Dispatch RPCs to all servers concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    request_server,
                    server_rank,
                    self._server_create_fn,
                    input_data,
                    self._sampling_config,
                    self._worker_options,
                )
                for server_rank, input_data in zip(
                    self._server_ranks, self._input_data_list
                )
            ]

        for future in futures:
            producer_id = future.result()
            self._producer_ids.append(producer_id)

        # Import here to avoid circular import issues with compiled GLT channel module
        from graphlearn_torch.channel import RemoteReceivingChannel

        self._channel = RemoteReceivingChannel(
            self._server_ranks,
            self._producer_ids,
            self._worker_options.prefetch_size,
        )

    def start_epoch(self, epoch: int) -> None:
        for server_rank, producer_id in zip(self._server_ranks, self._producer_ids):
            request_server(
                server_rank,
                DistServer.start_new_epoch_sampling,
                producer_id,
                epoch,
            )
        assert (
            self._channel is not None
        ), "setup_rpc() must be called before start_epoch()"
        self._channel.reset()

    def get_sample(self) -> SampleMessage:
        assert (
            self._channel is not None
        ), "setup_rpc() must be called before get_sample()"
        return self._channel.recv()

    def shutdown(self) -> None:
        if self._shutdowned:
            return
        if rpc_is_initialized() is True:
            for server_rank, producer_id in zip(self._server_ranks, self._producer_ids):
                request_server(
                    server_rank,
                    DistServer.destroy_sampling_producer,
                    producer_id,
                )
        self._shutdowned = True

    @property
    def num_expected(self) -> float:
        return float("inf")
