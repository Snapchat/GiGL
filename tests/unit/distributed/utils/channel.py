import multiprocessing as mp
import pickle

import torch

from gigl.distributed.utils.channel import SizedShmChannel
from tests.test_assets.test_case import TestCase


def _producer_worker(channel, msg_list):
    for msg in msg_list:
        channel.send(msg)


class TestSizedShmChannel(TestCase):
    def setUp(self):
        self.channel = SizedShmChannel()
        self.msg_1 = {"foo": torch.rand(1), "bar": torch.rand(3)}
        self.msg_2 = {"baz": torch.rand(3), "bat": torch.rand(7)}

    def test_single_process_qsize(self):
        # Initially empty channel
        self.assertEqual(self.channel.qsize(), 0)
        self.assertTrue(self.channel.empty())

        # Send first message
        self.channel.send(self.msg_1)
        self.assertEqual(self.channel.qsize(), 1)
        self.assertFalse(self.channel.empty())

        # Send second message
        self.channel.send(self.msg_2)
        self.assertEqual(self.channel.qsize(), 2)

        # Receive first message
        recv_1 = self.channel.recv()
        self.assertEqual(self.channel.qsize(), 1)
        torch.testing.assert_close(recv_1, self.msg_1)

        # Receive second message
        recv_2 = self.channel.recv()
        self.assertEqual(self.channel.qsize(), 0)
        self.assertTrue(self.channel.empty())
        torch.testing.assert_close(recv_2, self.msg_2)

    def test_multiprocessing_qsize(self):
        # Note: We use `fork` because glt.ShmChannel (from which we inherit) relies on
        # page table duplication (fork) to inherit C++ memory maps. It does not support `spawn`.
        ctx = mp.get_context("fork")
        messages = [self.msg_1, self.msg_2]

        # Start producer process
        process = ctx.Process(target=_producer_worker, args=(self.channel, messages))
        process.start()
        process.join(timeout=5)

        # Verify parent process observes updated queue size from worker process
        self.assertEqual(self.channel.qsize(), 2)

        # Consume messages in parent
        _ = self.channel.recv()
        self.assertEqual(self.channel.qsize(), 1)

        _ = self.channel.recv()
        self.assertEqual(self.channel.qsize(), 0)

    def test_pickling_roundtrip(self):
        # Push a message
        self.channel.send(self.msg_1)
        self.assertEqual(self.channel.qsize(), 1)

        # Pickle and unpickle (simulates IPC)
        serialized = pickle.dumps(self.channel)
        unpickled_channel = pickle.loads(serialized)

        # Verify both instances read the exact same queue size correctly
        self.assertEqual(unpickled_channel.qsize(), 1)
        self.assertEqual(self.channel.qsize(), 1)

        # Mutate via original channel, verify unpickled copy reflects live state
        _ = self.channel.recv()
        self.assertEqual(unpickled_channel.qsize(), 0)
