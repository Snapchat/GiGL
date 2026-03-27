import unittest
from unittest.mock import Mock, patch

import torch
from graphlearn_torch.sampler import NodeSamplerInput

from gigl.distributed.base_dist_loader import BaseDistLoader, DistributedRuntimeInfo


def _resolved_future(result=None) -> torch.futures.Future:
    future: torch.futures.Future = torch.futures.Future()
    future.set_result(result)
    return future


class BaseDistLoaderTest(unittest.TestCase):
    @staticmethod
    def _runtime(rank: int, world_size: int) -> DistributedRuntimeInfo:
        return DistributedRuntimeInfo(
            node_world_size=world_size,
            node_rank=rank,
            rank=rank,
            world_size=world_size,
            local_rank=0,
            local_world_size=1,
            master_ip_address="127.0.0.1",
            should_cleanup_distributed_context=False,
        )

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

    def test_grouped_graph_store_phase_leader_preserves_stagger(self) -> None:
        loader = BaseDistLoader.__new__(BaseDistLoader)
        loader._shutdowned = True
        runtime = self._runtime(rank=2, world_size=3)
        gather_results = [
            ["alpha", "alpha", "beta"],
            [[], [], [101, 102]],
        ]
        issued_results: list[int] = []

        def _mock_all_gather_object(output, _value):
            output[:] = gather_results.pop(0)

        def _issue_phase_rpcs() -> list[int]:
            issued_results.extend([101, 102])
            return [101, 102]

        with (
            patch(
                "gigl.distributed.base_dist_loader.torch.distributed.all_gather_object",
                side_effect=_mock_all_gather_object,
            ),
            patch("gigl.distributed.base_dist_loader.time.sleep") as mock_sleep,
        ):
            result = loader._dispatch_grouped_graph_store_phase(
                runtime=runtime,
                my_key="beta",
                key_name="backend_key",
                process_start_gap_seconds=1.5,
                max_concurrent_producer_inits=1,
                issue_phase_rpcs=_issue_phase_rpcs,
            )

        self.assertEqual(result, [101, 102])
        self.assertEqual(issued_results, [101, 102])
        mock_sleep.assert_called_once_with(1.5)

    def test_grouped_graph_store_phase_follower_reuses_leader_results(self) -> None:
        loader = BaseDistLoader.__new__(BaseDistLoader)
        loader._shutdowned = True
        runtime = self._runtime(rank=1, world_size=3)
        gather_results = [
            ["alpha", "alpha", "beta"],
            [[11, 12], [], []],
        ]
        phase_called = False

        def _mock_all_gather_object(output, _value):
            output[:] = gather_results.pop(0)

        def _issue_phase_rpcs() -> list[int]:
            nonlocal phase_called
            phase_called = True
            return [99]

        with (
            patch(
                "gigl.distributed.base_dist_loader.torch.distributed.all_gather_object",
                side_effect=_mock_all_gather_object,
            ),
            patch("gigl.distributed.base_dist_loader.time.sleep") as mock_sleep,
        ):
            result = loader._dispatch_grouped_graph_store_phase(
                runtime=runtime,
                my_key="alpha",
                key_name="worker_key",
                process_start_gap_seconds=1.5,
                max_concurrent_producer_inits=1,
                issue_phase_rpcs=_issue_phase_rpcs,
            )

        self.assertEqual(result, [11, 12])
        self.assertFalse(phase_called)
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
