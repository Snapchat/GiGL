from unittest.mock import patch

import torch

from gigl.distributed.graph_store.remote_channel import RemoteReceivingChannel
from gigl.distributed.utils.sampling_errors import (
    SAMPLING_ERROR_KEY,
    encode_sampling_error,
    raise_if_sampling_error,
)
from tests.test_assets.test_case import TestCase


def _resolved_future(result) -> torch.futures.Future:
    future: torch.futures.Future = torch.futures.Future()
    future.set_result(result)
    return future


def _failed_future(exc: Exception) -> torch.futures.Future:
    """Create a future that raises on .wait()."""
    future: torch.futures.Future = torch.futures.Future()
    future.set_exception(exc)
    return future


class RemoteReceivingChannelTest(TestCase):
    def test_recv_prefetch_allows_buffered_message_before_terminal_fetch(self) -> None:
        channel = RemoteReceivingChannel(
            server_rank=[0],
            channel_id=[10],
            prefetch_size=2,
            active_mask=[True],
        )
        channel.reset()

        requested_server_ranks: list[int] = []
        responses = iter(
            [
                ({"seed": torch.tensor([1], dtype=torch.long)}, False),
                (None, True),
            ]
        )

        def _mock_async_request_server(server_rank, func, channel_id):
            requested_server_ranks.append(server_rank)
            return _resolved_future(next(responses))

        with patch(
            "gigl.distributed.graph_store.remote_channel.async_request_server",
            side_effect=_mock_async_request_server,
        ):
            msg = channel.recv()
            self.assertTrue(torch.equal(msg["seed"], torch.tensor([1])))
            with self.assertRaises(StopIteration):
                channel.recv()

        self.assertGreaterEqual(len(requested_server_ranks), 2)
        self.assertEqual(requested_server_ranks[:2], [0, 0])

    def test_recv_skips_inactive_servers(self) -> None:
        channel = RemoteReceivingChannel(
            server_rank=[0, 1],
            channel_id=[10, 11],
            prefetch_size=1,
            active_mask=[True, False],
        )
        channel.reset()

        requested_server_ranks: list[int] = []
        responses = iter(
            [
                ({"seed": torch.tensor([1], dtype=torch.long)}, False),
                (None, True),
            ]
        )

        def _mock_async_request_server(server_rank, func, channel_id):
            requested_server_ranks.append(server_rank)
            return _resolved_future(next(responses))

        with patch(
            "gigl.distributed.graph_store.remote_channel.async_request_server",
            side_effect=_mock_async_request_server,
        ):
            msg = channel.recv()
            self.assertTrue(torch.equal(msg["seed"], torch.tensor([1])))
            with self.assertRaises(StopIteration):
                channel.recv()

        self.assertEqual(requested_server_ranks, [0, 0])

    def test_recv_stops_immediately_when_all_servers_are_inactive(self) -> None:
        channel = RemoteReceivingChannel(
            server_rank=[0, 1],
            channel_id=[10, 11],
            prefetch_size=1,
            active_mask=[False, False],
        )
        channel.reset()

        with self.assertRaises(StopIteration):
            channel.recv()

    def test_recv_does_not_pin_memory_when_disabled(self) -> None:
        """Verify tensors are not pinned when pin_memory=False (default)."""
        channel = RemoteReceivingChannel(
            server_rank=[0],
            channel_id=[10],
            prefetch_size=1,
            active_mask=[True],
            pin_memory=False,
        )
        channel.reset()

        responses = iter(
            [
                ({"seed": torch.tensor([1], dtype=torch.long)}, False),
                (None, True),
            ]
        )

        def _mock_async_request_server(server_rank, func, channel_id):
            return _resolved_future(next(responses))

        with patch(
            "gigl.distributed.graph_store.remote_channel.async_request_server",
            side_effect=_mock_async_request_server,
        ):
            msg = channel.recv()
            self.assertFalse(msg["seed"].is_pinned())

    def test_recv_raises_on_failed_future(self) -> None:
        """Verify recv() raises RuntimeError instead of deadlocking when an RPC future fails."""
        channel = RemoteReceivingChannel(
            server_rank=[0],
            channel_id=[10],
            prefetch_size=1,
            active_mask=[True],
        )
        channel.reset()

        def _mock_async_request_server(server_rank, func, channel_id):
            return _failed_future(RuntimeError("RPC connection lost"))

        with patch(
            "gigl.distributed.graph_store.remote_channel.async_request_server",
            side_effect=_mock_async_request_server,
        ):
            with self.assertRaises(RuntimeError):
                channel.recv()

    def test_recv_forwards_sampling_error_pill_to_decoder(self) -> None:
        """A sampling-error poison pill survives graph-store transport and raises.

        The server-side sampler forwards the pill as an ordinary ``SampleMessage``;
        ``RemoteReceivingChannel.recv`` passes it through content-agnostically, and the
        loader-side ``raise_if_sampling_error`` decodes the embedded traceback.
        """
        channel = RemoteReceivingChannel(
            server_rank=[0],
            channel_id=[10],
            prefetch_size=1,
            active_mask=[True],
        )
        channel.reset()

        pill = {SAMPLING_ERROR_KEY: encode_sampling_error("boom")}
        # Responses are mocked returned messages from the server
        responses = iter([(pill, False), (None, True)])

        def _mock_async_request_server(server_rank, func, channel_id):
            return _resolved_future(next(responses))

        with patch(
            "gigl.distributed.graph_store.remote_channel.async_request_server",
            side_effect=_mock_async_request_server,
        ):
            msg = channel.recv()

        with self.assertRaises(RuntimeError) as ctx:
            raise_if_sampling_error(msg)
        self.assertIn("boom", str(ctx.exception))
