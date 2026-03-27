import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from graphlearn_torch.sampler import NodeSamplerInput, SamplingConfig, SamplingType

from gigl.distributed.dist_sampling_producer import (
    ActiveEpochState,
    RegisterInputCmd,
    SharedMpCommand,
    StartEpochCmd,
    _compute_num_batches,
    _epoch_batch_indices,
    _shared_sampling_worker_loop,
)
from gigl.distributed.sampler_options import KHopNeighborSamplerOptions


class _FakeBarrier:
    def wait(self) -> int:
        return 0


class _FakeChannel:
    pass


class _RecordingSampler:
    instances: list["_RecordingSampler"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[tuple[int, torch.Tensor]] = []
        self.callbacks: list[object] = []
        self.registered_outputs: list[tuple[str, object]] = []
        self.unregistered_outputs: list[str] = []
        self.wait_all_called = False
        _RecordingSampler.instances.append(self)

    def start_loop(self) -> None:
        pass

    def shutdown_loop(self) -> None:
        pass

    def register_output(self, key: str, channel: object) -> None:
        self.registered_outputs.append((key, channel))

    def unregister_output(self, key: str) -> None:
        self.unregistered_outputs.append(key)

    def sample_from_nodes(self, sampler_input, key: str, callback) -> None:
        self.calls.append((int(key), sampler_input.node.clone()))
        self.callbacks.append(callback)

    def wait_all(self) -> None:
        self.wait_all_called = True


class DistSamplingProducerTest(unittest.TestCase):
    def setUp(self) -> None:
        _RecordingSampler.instances.clear()

    @staticmethod
    def _sampling_config(batch_size: int = 1) -> SamplingConfig:
        return SamplingConfig(
            sampling_type=SamplingType.NODE,
            num_neighbors=[2],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            with_edge=False,
            collect_features=False,
            with_neg=False,
            with_weight=False,
            edge_dir="in",
            seed=0,
        )

    def test_epoch_batch_indices_uses_seed_order(self) -> None:
        state = ActiveEpochState(
            channel_id=7,
            epoch=3,
            input_len=5,
            batch_size=2,
            drop_last=False,
            seeds_index=torch.tensor([4, 1, 3, 0, 2]),
            total_batches=_compute_num_batches(5, 2, False),
            submitted_batches=1,
        )

        batch_indices = _epoch_batch_indices(state)

        assert batch_indices is not None
        self.assertTrue(torch.equal(batch_indices, torch.tensor([3, 0])))

    def test_shared_sampling_worker_loop_round_robins_first_submissions(self) -> None:
        task_queue: queue.Queue = queue.Queue()
        event_queue: queue.Queue = queue.Queue()
        sampling_config = self._sampling_config(batch_size=1)

        task_queue.put(
            (
                SharedMpCommand.REGISTER_INPUT,
                RegisterInputCmd(
                    channel_id=0,
                    worker_key="worker-0",
                    sampler_input=NodeSamplerInput(
                        node=torch.tensor([0, 1, 2, 3]), input_type="user"
                    ),
                    sampling_config=sampling_config,
                    channel=_FakeChannel(),
                ),
            )
        )
        task_queue.put(
            (
                SharedMpCommand.REGISTER_INPUT,
                RegisterInputCmd(
                    channel_id=1,
                    worker_key="worker-0",
                    sampler_input=NodeSamplerInput(
                        node=torch.tensor([10, 11, 12, 13]), input_type="user"
                    ),
                    sampling_config=sampling_config,
                    channel=_FakeChannel(),
                ),
            )
        )
        task_queue.put(
            (
                SharedMpCommand.START_EPOCH,
                StartEpochCmd(channel_id=0, epoch=0, seeds_index=None),
            )
        )
        task_queue.put(
            (
                SharedMpCommand.START_EPOCH,
                StartEpochCmd(channel_id=1, epoch=0, seeds_index=None),
            )
        )

        worker_options = SimpleNamespace(
            worker_world_size=1,
            worker_ranks=[0],
            use_all2all=False,
            num_rpc_threads=1,
            master_addr="127.0.0.1",
            master_port=29500,
            rpc_timeout=30,
            worker_devices=["cpu"],
            worker_concurrency=2,
        )
        data = SimpleNamespace(num_partitions=1)

        with patch("gigl.distributed.dist_sampling_producer.init_worker_group"), patch(
            "gigl.distributed.dist_sampling_producer.init_rpc"
        ), patch("gigl.distributed.dist_sampling_producer.shutdown_rpc"), patch(
            "gigl.distributed.dist_sampling_producer._set_worker_signal_handlers"
        ), patch(
            "gigl.distributed.dist_sampling_producer.torch.set_num_threads"
        ), patch(
            "gigl.distributed.dist_sampling_producer.seed_everything"
        ), patch(
            "gigl.distributed.dist_sampling_producer.DistNeighborSampler",
            _RecordingSampler,
        ):
            worker_thread = threading.Thread(
                target=_shared_sampling_worker_loop,
                args=(
                    0,
                    data,
                    sampling_config,
                    worker_options,
                    task_queue,
                    event_queue,
                    _FakeBarrier(),
                    KHopNeighborSamplerOptions(num_neighbors=[2]),
                ),
                daemon=True,
            )
            worker_thread.start()

            deadline = time.time() + 2.0
            while time.time() < deadline:
                if (
                    _RecordingSampler.instances
                    and len(_RecordingSampler.instances[0].calls) >= 2
                ):
                    break
                time.sleep(0.01)

            self.assertTrue(_RecordingSampler.instances)
            sampler = _RecordingSampler.instances[0]
            self.assertGreaterEqual(len(sampler.calls), 2)
            self.assertEqual(
                [channel_id for channel_id, _ in sampler.calls[:2]], [0, 1]
            )
            self.assertTrue(torch.equal(sampler.calls[0][1], torch.tensor([0])))
            self.assertTrue(torch.equal(sampler.calls[1][1], torch.tensor([10])))

            task_queue.put((SharedMpCommand.STOP, None))
            worker_thread.join(timeout=2.0)

            self.assertFalse(worker_thread.is_alive())
            self.assertTrue(sampler.wait_all_called)


if __name__ == "__main__":
    unittest.main()
