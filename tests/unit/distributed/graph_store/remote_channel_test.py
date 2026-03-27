import unittest
from unittest.mock import patch

import torch

from gigl.distributed.graph_store.remote_channel import RemoteReceivingChannel


def _resolved_future(result) -> torch.futures.Future:
    future: torch.futures.Future = torch.futures.Future()
    future.set_result(result)
    return future


class RemoteReceivingChannelTest(unittest.TestCase):
    def test_reset_reapplies_active_mask(self) -> None:
        channel = RemoteReceivingChannel(
            server_rank=[0, 1],
            channel_id=[10, 11],
            prefetch_size=2,
            active_mask=[True, False],
        )
        channel.server_end_of_epoch = [True, True]
        channel.global_end_of_epoch = True
        channel.num_request_list = [3, 4]
        channel.num_received_list = [1, 2]

        channel.reset()

        self.assertEqual(channel.server_end_of_epoch, [False, True])
        self.assertFalse(channel.global_end_of_epoch)
        self.assertEqual(channel.num_request_list, [0, 0])
        self.assertEqual(channel.num_received_list, [0, 0])

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


if __name__ == "__main__":
    unittest.main()
