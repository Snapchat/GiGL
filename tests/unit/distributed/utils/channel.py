import multiprocessing as mp
import pickle
from unittest.mock import MagicMock

import torch

from gigl.distributed.utils.channel import MonitoredShmChannel, SizedShmChannel
from gigl.src.common.utils import metrics_service_provider
from tests.test_assets.test_case import TestCase


def _producer_worker(channel, msg_list):
    for msg in msg_list:
        channel.send(msg)


class TestSizedShmChannel(TestCase):
    def setUp(self):
        self.msg_1 = {"foo": torch.rand(1), "bar": torch.rand(3)}
        self.msg_2 = {"baz": torch.rand(3), "bat": torch.rand(7)}

    def test_single_process_qsize(self):
        channel = SizedShmChannel()

        # Initially empty channel
        self.assertEqual(len(channel), 0)
        self.assertEqual(channel.qsize(), 0)
        self.assertTrue(channel.empty())

        # Send first message
        channel.send(self.msg_1)
        self.assertEqual(len(channel), 1)
        self.assertEqual(channel.qsize(), 1)
        self.assertFalse(channel.empty())

        # Send second message
        channel.send(self.msg_2)
        self.assertEqual(len(channel), 2)
        self.assertEqual(channel.qsize(), 2)

        # Receive first message
        recv_1 = channel.recv()
        self.assertEqual(len(channel), 1)
        self.assertEqual(channel.qsize(), 1)
        torch.testing.assert_close(recv_1, self.msg_1)

        # Receive second message
        recv_2 = channel.recv()
        self.assertEqual(len(channel), 0)
        self.assertEqual(channel.qsize(), 0)
        self.assertTrue(channel.empty())
        torch.testing.assert_close(recv_2, self.msg_2)

    def test_multiprocessing_qsize(self):
        channel = SizedShmChannel()

        # Note: We use `fork` because glt.ShmChannel (from which we inherit) relies on
        # page table duplication (fork) to inherit C++ memory maps. It does not support `spawn`.
        ctx = mp.get_context("fork")
        messages = [self.msg_1, self.msg_2]

        # Start producer process
        process = ctx.Process(target=_producer_worker, args=(channel, messages))
        process.start()
        process.join(timeout=5)

        # Verify parent process observes updated queue size from worker process
        self.assertEqual(channel.qsize(), 2)

        # Consume messages in parent
        _ = channel.recv()
        self.assertEqual(channel.qsize(), 1)

        _ = channel.recv()
        self.assertEqual(channel.qsize(), 0)

    def test_pickling_roundtrip(self):
        channel = SizedShmChannel()

        # Push a message
        channel.send(self.msg_1)
        self.assertEqual(channel.qsize(), 1)

        # Pickle and unpickle (simulates IPC)
        serialized = pickle.dumps(channel)
        unpickled_channel = pickle.loads(serialized)

        # Verify both instances read the exact same queue size correctly
        self.assertEqual(unpickled_channel.qsize(), 1)
        self.assertEqual(channel.qsize(), 1)

        # Mutate via original channel, verify unpickled copy reflects live state
        _ = channel.recv()
        self.assertEqual(unpickled_channel.qsize(), 0)


class MonitoredShmChannelTest(TestCase):
    def setUp(self) -> None:
        self._original_metrics_instance = metrics_service_provider._metrics_instance
        self.mock_metrics = metrics_service_provider._metrics_instance = MagicMock()
        self.msg_1 = {"foo": torch.rand(1), "bar": torch.rand(3)}
        self.msg_2 = {"baz": torch.rand(3), "bat": torch.rand(7)}

    def tearDown(self) -> None:
        metrics_service_provider._metrics_instance = self._original_metrics_instance

    def test_recv_publishes_qsize_gauge(self) -> None:
        for instance_count in range(3):
            channel_name = "test_channel"
            channel = MonitoredShmChannel(channel_name=channel_name)

            metric_name = f"{channel_name}_id{instance_count}_qsize"

            # Push 2 messages
            channel.send(self.msg_1)
            channel.send(self.msg_2)

            # First recv, qsize gauge before dequeue should be 2
            channel.recv()
            self.mock_metrics.add_gauge.assert_called_with(metric_name, 2)

            # Second recv, qsize gauge before dequeue should be 1
            channel.recv()
            self.mock_metrics.add_gauge.assert_called_with(metric_name, 1)
