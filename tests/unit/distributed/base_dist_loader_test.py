import unittest
from unittest.mock import Mock, patch

import torch
from graphlearn_torch.sampler import NodeSamplerInput

from gigl.distributed.base_dist_loader import BaseDistLoader


def _resolved_future(result=None) -> torch.futures.Future:
    future: torch.futures.Future = torch.futures.Future()
    future.set_result(result)
    return future


class BaseDistLoaderTest(unittest.TestCase):
    def test_sampler_input_has_batches_respects_drop_last(self) -> None:
        loader = BaseDistLoader.__new__(BaseDistLoader)
        loader.batch_size = 4
        loader.drop_last = True
        loader._shutdowned = True

        self.assertFalse(
            loader._sampler_input_has_batches(
                NodeSamplerInput(node=torch.empty(0, dtype=torch.long))
            )
        )
        self.assertFalse(
            loader._sampler_input_has_batches(
                NodeSamplerInput(node=torch.arange(3, dtype=torch.long))
            )
        )
        self.assertTrue(
            loader._sampler_input_has_batches(
                NodeSamplerInput(node=torch.arange(4, dtype=torch.long))
            )
        )

        loader.drop_last = False
        self.assertTrue(
            loader._sampler_input_has_batches(
                NodeSamplerInput(node=torch.arange(1, dtype=torch.long))
            )
        )

    def test_iter_starts_only_active_remote_channels(self) -> None:
        loader = BaseDistLoader.__new__(BaseDistLoader)
        loader._shutdowned = True
        loader._is_collocated_worker = False
        loader._is_mp_worker = False
        loader._server_rank_list = [0, 1, 2]
        loader._channel_id_list = [10, 11, 12]
        loader._remote_input_has_batches = [False, True, False]
        loader._channel = Mock()
        loader._epoch = 7

        rpc_calls: list[tuple[int, int, int]] = []

        def _mock_async_request_server(server_rank, func, channel_id, epoch):
            rpc_calls.append((server_rank, channel_id, epoch))
            return _resolved_future(None)

        with patch(
            "gigl.distributed.base_dist_loader.async_request_server",
            side_effect=_mock_async_request_server,
        ):
            result = BaseDistLoader.__iter__(loader)

        self.assertIs(result, loader)
        self.assertEqual(rpc_calls, [(1, 11, 7)])
        loader._channel.reset.assert_called_once_with()
        self.assertEqual(loader._epoch, 8)
        self.assertEqual(loader._num_recv, 0)


if __name__ == "__main__":
    unittest.main()
