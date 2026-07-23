import multiprocessing as mp

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
        ctx = mp.get_context("fork")
        messages = [self.msg_1, self.msg_2]

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
