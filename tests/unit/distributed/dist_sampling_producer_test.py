import queue
from typing import cast
from unittest.mock import MagicMock, patch

import torch
import torch.multiprocessing as mp
from graphlearn_torch.sampler import NodeSamplerInput, SamplingConfig, SamplingType

from gigl.distributed.graph_store.shared_dist_sampling_producer import (
    EPOCH_DONE_EVENT,
    ActiveEpochState,
    SharedDistSamplingBackend,
    SharedMpCommand,
    StartEpochCmd,
    _compute_num_batches,
    _compute_worker_seeds_ranges,
    _epoch_batch_indices,
)
from gigl.distributed.sampler_options import KHopNeighborSamplerOptions
from tests.test_assets.test_case import TestCase


def _make_sampling_config(*, shuffle: bool = False) -> SamplingConfig:
    return SamplingConfig(
        sampling_type=SamplingType.NODE,
        num_neighbors=[2],
        batch_size=2,
        shuffle=shuffle,
        drop_last=False,
        with_edge=True,
        collect_features=True,
        with_neg=False,
        with_weight=False,
        edge_dir="out",
        seed=1234,
    )


class _FakeProcess:
    def __init__(self, *args, **kwargs) -> None:
        self.daemon = False

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        return None


class _FakeMpContext:
    def Barrier(self, parties: int):
        return MagicMock(wait=MagicMock())

    def Queue(self, maxsize: int = 0):
        return MagicMock()

    def Process(self, *args, **kwargs):
        return _FakeProcess(*args, **kwargs)


class DistSamplingProducerTest(TestCase):
    def test_compute_num_batches(self) -> None:
        self.assertEqual(_compute_num_batches(0, 2, False), 0)
        self.assertEqual(_compute_num_batches(1, 2, True), 0)
        self.assertEqual(_compute_num_batches(1, 2, False), 1)
        self.assertEqual(_compute_num_batches(5, 2, False), 3)
        self.assertEqual(_compute_num_batches(5, 2, True), 2)

    def test_epoch_batch_indices(self) -> None:
        active_state = ActiveEpochState(
            channel_id=0,
            epoch=0,
            input_len=6,
            batch_size=2,
            drop_last=False,
            seeds_index=torch.arange(6),
            total_batches=3,
            submitted_batches=1,
            cancelled=False,
        )
        result = _epoch_batch_indices(active_state)
        assert result is not None
        self.assert_tensor_equality(result, torch.tensor([2, 3]))

    def test_compute_worker_seeds_ranges(self) -> None:
        self.assertEqual(
            _compute_worker_seeds_ranges(input_len=7, batch_size=2, num_workers=3),
            [(0, 2), (2, 4), (4, 7)],
        )

    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.get_context")
    @patch("gigl.distributed.graph_store.shared_dist_sampling_producer.mp.get_context")
    @patch(
        "gigl.distributed.graph_store.shared_dist_sampling_producer._prepare_degree_tensors"
    )
    def test_init_backend_prepares_worker_options(
        self,
        mock_prepare_degree_tensors: MagicMock,
        mock_get_mp_context: MagicMock,
        mock_get_context: MagicMock,
    ) -> None:
        worker_options = MagicMock()
        worker_options.num_workers = 2
        worker_options.worker_concurrency = 1
        mock_get_context.return_value = MagicMock(
            is_server=MagicMock(return_value=True)
        )
        mock_get_mp_context.return_value = _FakeMpContext()
        backend = SharedDistSamplingBackend(
            data=MagicMock(),
            worker_options=worker_options,
            sampling_config=_make_sampling_config(),
            sampler_options=KHopNeighborSamplerOptions(num_neighbors=[2]),
        )

        backend.init_backend()

        worker_options._assign_worker_devices.assert_called_once()
        worker_options._set_worker_ranks.assert_called_once_with(
            mock_get_context.return_value
        )
        self.assertEqual(len(backend._task_queues), 2)
        self.assertEqual(len(backend._workers), 2)
        self.assertTrue(backend._initialized)
        mock_prepare_degree_tensors.assert_called_once()

    def test_start_new_epoch_sampling_shuffle_refreshes_per_epoch(self) -> None:
        worker_options = MagicMock()
        worker_options.num_workers = 2
        worker_options.worker_concurrency = 1
        backend = SharedDistSamplingBackend(
            data=MagicMock(),
            worker_options=worker_options,
            sampling_config=_make_sampling_config(shuffle=True),
            sampler_options=KHopNeighborSamplerOptions(num_neighbors=[2]),
        )
        backend._initialized = True
        recorded: list[tuple[int, SharedMpCommand, object]] = []
        backend._enqueue_worker_command = lambda worker_rank, command, payload: recorded.append(  # type: ignore[method-assign]
            (worker_rank, command, payload)
        )

        channel = MagicMock()
        input_tensor = torch.arange(6, dtype=torch.long)
        backend.register_input(
            channel_id=1,
            worker_key="loader_a_compute_rank_0",
            sampler_input=NodeSamplerInput(node=input_tensor.clone()),
            sampling_config=_make_sampling_config(shuffle=True),
            channel=channel,
        )
        backend.register_input(
            channel_id=2,
            worker_key="loader_b_compute_rank_0",
            sampler_input=NodeSamplerInput(node=input_tensor.clone()),
            sampling_config=_make_sampling_config(shuffle=True),
            channel=channel,
        )

        def _collect_epoch_indices(channel_id: int, epoch: int) -> torch.Tensor:
            recorded.clear()
            backend.start_new_epoch_sampling(channel_id, epoch)
            worker_payloads = {
                worker_rank: cast(StartEpochCmd, payload).seeds_index
                for worker_rank, command, payload in recorded
                if command == SharedMpCommand.START_EPOCH
            }
            assert all(
                seed_index is not None for seed_index in worker_payloads.values()
            )
            return torch.cat(
                [
                    cast(torch.Tensor, worker_payloads[worker_rank])
                    for worker_rank in sorted(worker_payloads)
                ]
            )

        channel_1_epoch_0 = _collect_epoch_indices(1, 0)
        channel_2_epoch_0 = _collect_epoch_indices(2, 0)
        channel_1_epoch_1 = _collect_epoch_indices(1, 1)

        self.assert_tensor_equality(channel_1_epoch_0, channel_2_epoch_0)
        self.assertNotEqual(
            channel_1_epoch_0.tolist(),
            channel_1_epoch_1.tolist(),
        )

    def test_describe_channel_reports_completed_workers(self) -> None:
        worker_options = MagicMock()
        worker_options.num_workers = 2
        worker_options.worker_concurrency = 1
        backend = SharedDistSamplingBackend(
            data=MagicMock(),
            worker_options=worker_options,
            sampling_config=_make_sampling_config(),
            sampler_options=KHopNeighborSamplerOptions(num_neighbors=[2]),
        )
        backend._initialized = True
        backend._event_queue = cast(mp.Queue, queue.Queue())
        backend._channel_input_sizes[1] = [4, 2]
        backend._channel_epoch[1] = 3
        cast(queue.Queue, backend._event_queue).put((EPOCH_DONE_EVENT, 1, 3, 0))

        description = backend.describe_channel(1)

        self.assertEqual(description["epoch"], 3)
        self.assertEqual(description["input_sizes"], [4, 2])
        self.assertEqual(description["completed_workers"], 1)
