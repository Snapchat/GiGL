"""GiGL-owned remote receiving channel for graph-store sampling.

This mirrors GLT's ``RemoteReceivingChannel`` behavior, but routes fetch RPCs
through GiGL server methods so channel-based sampling works with shared
producers.
"""

from __future__ import annotations

import logging
import queue
from collections import abc
from typing import Callable, Optional, Union

import torch
from graphlearn_torch.channel import ChannelBase, SampleMessage

from gigl.distributed.graph_store.compute import async_request_server
from gigl.distributed.graph_store.dist_server import DistServer


class RemoteReceivingChannel(ChannelBase):
    """Pull-based receiving channel that fetches sampled messages from servers.

    Args:
        server_rank: Target storage server rank(s).
        channel_id: Sampling channel id(s), one per server rank.
        prefetch_size: Number of in-flight fetch requests per server.
    """

    def __init__(
        self,
        server_rank: Union[int, list[int]],
        channel_id: Union[int, list[int]],
        prefetch_size: int = 2,
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

        self.num_request_list = [0] * len(self.server_rank_list)
        self.num_received_list = [0] * len(self.server_rank_list)
        self.server_end_of_epoch = [False] * len(self.server_rank_list)
        self.global_end_of_epoch = False
        self.queue: queue.Queue[
            tuple[Optional[SampleMessage], bool, int]
        ] = queue.Queue(maxsize=self.prefetch_size * len(self.server_rank_list))

    def reset(self) -> None:
        """Reset all state to start a new epoch."""
        while not self.queue.empty():
            _ = self.queue.get()
        self.server_end_of_epoch = [False] * len(self.server_rank_list)
        self.num_request_list = [0] * len(self.server_rank_list)
        self.num_received_list = [0] * len(self.server_rank_list)
        self.global_end_of_epoch = False

    def send(self, msg: SampleMessage, **kwargs: object) -> None:
        raise RuntimeError(
            f"'{self.__class__.__name__}': cannot send "
            "message with a receiving channel."
        )

    def recv(self, **kwargs: object) -> SampleMessage:
        if self.global_end_of_epoch:
            if self._all_received():
                raise StopIteration
        else:
            self._request_some()

        msg, end_of_epoch, local_server_idx = self.queue.get()
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
        return msg

    def _all_received(self) -> bool:
        return sum(self.num_received_list) == sum(self.num_request_list)

    def _request_some(self) -> None:
        def on_done(
            future: torch.futures.Future[tuple[Optional[SampleMessage], bool]],
            local_server_idx: int,
        ) -> None:
            try:
                msg, end_of_epoch = future.wait()
                self.queue.put((msg, end_of_epoch, local_server_idx))
            except Exception as exc:
                logging.error("broken future of receiving remote messages: %s", exc)

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
