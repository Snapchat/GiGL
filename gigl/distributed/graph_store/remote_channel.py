"""GiGL-owned remote receiving channel for graph-store sampling.

This mirrors GLT's ``RemoteReceivingChannel`` behavior, but routes fetch RPCs
through GiGL server methods so channel-based sampling works with shared
producers.
"""

from __future__ import annotations

import queue
import time
from collections import abc
from typing import Callable, Optional, Union

import torch
from graphlearn_torch.channel import ChannelBase, SampleMessage

from gigl.common.logger import Logger
from gigl.distributed.graph_store.compute import async_request_server
from gigl.distributed.graph_store.dist_server import DistServer

logger = Logger()


class RemoteReceivingChannel(ChannelBase):
    """Pull-based receiving channel that fetches sampled messages from servers.

    Args:
        server_rank: Target storage server rank(s).
        channel_id: Sampling channel id(s), one per server rank.
        prefetch_size: Number of in-flight fetch requests per server.
        active_mask: Optional per-server mask indicating which channels can
            produce at least one batch this epoch. Inactive servers are treated
            as already finished and are never polled.
        pin_memory: If True, copy received tensors to CUDA-pinned host memory
            before returning from ``recv()``. Enables faster GPU transfers via
            DMA in the downstream collate function.
    """

    def __init__(
        self,
        server_rank: Union[int, list[int]],
        channel_id: Union[int, list[int]],
        prefetch_size: int = 2,
        active_mask: Optional[list[bool]] = None,
        pin_memory: bool = False,
    ) -> None:
        self.server_rank_list = (
            list(server_rank)
            if isinstance(server_rank, abc.Sequence)
            and not isinstance(server_rank, int)
            else [int(server_rank)]
        )
        self.channel_id_list = (
            list(channel_id)
            if isinstance(channel_id, abc.Sequence) and not isinstance(channel_id, int)
            else [int(channel_id)]
        )
        self.prefetch_size = prefetch_size

        if len(self.server_rank_list) != len(self.channel_id_list):
            raise ValueError(
                "server_rank and channel_id must have the same length, got "
                f"{len(self.server_rank_list)} and {len(self.channel_id_list)}"
            )
        if active_mask is None:
            self.active_mask = [True] * len(self.server_rank_list)
        else:
            if len(active_mask) != len(self.server_rank_list):
                raise ValueError(
                    "active_mask must have the same length as server_rank/channel_id, got "
                    f"{len(active_mask)} and {len(self.server_rank_list)}"
                )
            self.active_mask = list(active_mask)

        self.num_request_list = [0] * len(self.server_rank_list)
        self.num_received_list = [0] * len(self.server_rank_list)
        self.server_end_of_epoch = [not is_active for is_active in self.active_mask]
        self.global_end_of_epoch = all(self.server_end_of_epoch)
        self.queue: queue.Queue[
            tuple[Optional[SampleMessage], bool, int]
        ] = queue.Queue(maxsize=self.prefetch_size * len(self.server_rank_list))
        self._recv_count: int = 0
        self._log_every_n: int = 50
        self._pin_memory = pin_memory

    def reset(self) -> None:
        """Reset all state to start a new epoch."""
        while not self.queue.empty():
            _ = self.queue.get()
        self.server_end_of_epoch = [not is_active for is_active in self.active_mask]
        self.num_request_list = [0] * len(self.server_rank_list)
        self.num_received_list = [0] * len(self.server_rank_list)
        self.global_end_of_epoch = all(self.server_end_of_epoch)
        self._recv_count = 0

    def send(self, msg: SampleMessage, **kwargs: object) -> None:
        raise RuntimeError(
            f"'{self.__class__.__name__}': cannot send "
            "message with a receiving channel."
        )

    def recv(self, **kwargs: object) -> SampleMessage:
        request_some_elapsed = 0.0
        num_dispatched = 0
        if self.global_end_of_epoch:
            if self._all_received():
                raise StopIteration
        else:
            request_some_start = time.monotonic()
            num_dispatched = self._request_some()
            request_some_elapsed = time.monotonic() - request_some_start

        queue_depth = self.queue.qsize()
        queue_get_start = time.monotonic()
        msg, end_of_epoch, local_server_idx = self.queue.get()
        queue_get_elapsed = time.monotonic() - queue_get_start
        self.num_received_list[local_server_idx] += 1

        # Server guarantees that when end_of_epoch is true, msg is None.
        while end_of_epoch:
            self.server_end_of_epoch[local_server_idx] = True
            if sum(self.server_end_of_epoch) == len(self.server_rank_list):
                self.global_end_of_epoch = True
                if self._all_received():
                    raise StopIteration
            msg, end_of_epoch, local_server_idx = self.queue.get()
            self.num_received_list[local_server_idx] += 1

        if msg is None:
            raise RuntimeError(
                "Received unexpected None message when end_of_epoch is False."
            )

        if self._pin_memory:
            pin_start = time.monotonic()
            msg = self._pin_sample_message(msg)
            pin_elapsed = time.monotonic() - pin_start
        else:
            pin_elapsed = 0.0

        self._recv_count += 1
        if self._recv_count % self._log_every_n == 0:
            logger.info(
                "remote_channel_recv "
                f"recv_count={self._recv_count} "
                f"request_some_time={request_some_elapsed:.4f}s "
                f"num_rpcs_dispatched={num_dispatched} "
                f"queue_depth_before_get={queue_depth} "
                f"queue_get_time={queue_get_elapsed:.4f}s "
                f"pin_time={pin_elapsed:.4f}s"
            )

        return msg

    @staticmethod
    def _pin_sample_message(msg: SampleMessage) -> SampleMessage:
        """Copy all tensors in the message to CUDA-pinned host memory.

        This enables faster DMA transfers when subsequently calling
        ``.to(device)`` in the collate function.
        """
        pinned: SampleMessage = {}
        for k, v in msg.items():
            if isinstance(v, torch.Tensor) and not v.is_pinned():
                pinned[k] = v.pin_memory()
            else:
                pinned[k] = v
        return pinned

    def _all_received(self) -> bool:
        return sum(self.num_received_list) == sum(self.num_request_list)

    def _request_some(self) -> int:
        """Dispatch prefetch RPCs. Returns the number of new RPCs dispatched."""
        num_dispatched = 0

        def on_done(
            future: torch.futures.Future[tuple[Optional[SampleMessage], bool]],
            local_server_idx: int,
        ) -> None:
            try:
                msg, end_of_epoch = future.wait()
                self.queue.put((msg, end_of_epoch, local_server_idx))
            except Exception as exc:
                logger.error("broken future of receiving remote messages: %s", exc)

        def create_callback(
            local_server_idx: int,
        ) -> Callable[
            [torch.futures.Future[tuple[Optional[SampleMessage], bool]]], None
        ]:
            def callback(
                future: torch.futures.Future[tuple[Optional[SampleMessage], bool]],
            ) -> None:
                on_done(future, local_server_idx)

            return callback

        for local_server_idx, server_rank in enumerate(self.server_rank_list):
            if not self.active_mask[local_server_idx]:
                continue
            if self.server_end_of_epoch[local_server_idx]:
                continue
            missing = (
                self.num_received_list[local_server_idx]
                + self.prefetch_size
                - self.num_request_list[local_server_idx]
            )
            for _ in range(missing):
                future = async_request_server(
                    server_rank,
                    DistServer.fetch_one_sampled_message,
                    self.channel_id_list[local_server_idx],
                )
                future.add_done_callback(create_callback(local_server_idx))
                self.num_request_list[local_server_idx] += 1
                num_dispatched += 1

        return num_dispatched
