import torch
from graphlearn_torch.channel import ShmChannel
from parameterized import param, parameterized

from gigl.distributed.utils.sampling_errors import (
    SAMPLING_ERROR_KEY,
    encode_sampling_error,
    raise_if_message_exceeds_channel,
    raise_if_sampling_error,
)
from tests.test_assets.test_case import TestCase


class TestSamplingErrors(TestCase):
    def test_roundtrip_ascii(self) -> None:
        tb = "Traceback (most recent call last):\n  KeyError: foo"
        msg = {SAMPLING_ERROR_KEY: encode_sampling_error(tb)}
        with self.assertRaises(RuntimeError) as ctx:
            raise_if_sampling_error(msg)
        self.assertIn(tb, str(ctx.exception))

    def test_roundtrip_non_ascii(self) -> None:
        tb = "boom – KeyError: user‑engages"
        msg = {SAMPLING_ERROR_KEY: encode_sampling_error(tb)}
        with self.assertRaises(RuntimeError) as ctx:
            raise_if_sampling_error(msg)
        self.assertIn(tb, str(ctx.exception))

    def test_empty_string_does_not_crash(self) -> None:
        # format_exc() is never empty in practice, but the codec must not crash.
        msg = {SAMPLING_ERROR_KEY: encode_sampling_error("")}
        with self.assertRaises(RuntimeError):
            raise_if_sampling_error(msg)

    def test_encoded_tensor_is_writable_uint8(self) -> None:
        t = encode_sampling_error("hi")
        self.assertEqual(t.dtype, torch.uint8)
        t[0] = t[0]  # must not raise (writable)

    def test_no_key_is_noop(self) -> None:
        raise_if_sampling_error({"ids": torch.zeros(3)})  # returns None, no raise


class _SizedChannel:
    """Stands in for a channel that knows its buffer size."""

    def __init__(self, max_message_bytes: int) -> None:
        self.max_message_bytes = max_message_bytes


class RaiseIfMessageExceedsChannelTest(TestCase):
    @parameterized.expand(
        [
            param("larger_than_the_buffer", num_bytes=2048, raises=True),
            param("exactly_the_buffer_size", num_bytes=1024, raises=True),
            param("smaller_than_the_buffer", num_bytes=1023, raises=False),
        ]
    )
    def test_a_message_is_rejected_when_it_is_at_least_the_buffer_size(
        self, _name: str, num_bytes: int, raises: bool
    ) -> None:
        message = {"ids": torch.zeros(num_bytes, dtype=torch.uint8)}
        if raises:
            with self.assertRaises(RuntimeError):
                raise_if_message_exceeds_channel(message, _SizedChannel(1024))
        else:
            raise_if_message_exceeds_channel(message, _SizedChannel(1024))

    def test_the_size_adds_up_every_tensor_by_its_element_size(self) -> None:
        message = {
            "ids": torch.zeros(100, dtype=torch.int64),
            "weights": torch.zeros(100, dtype=torch.float32),
        }
        raise_if_message_exceeds_channel(message, _SizedChannel(1201))
        with self.assertRaises(RuntimeError):
            raise_if_message_exceeds_channel(message, _SizedChannel(1200))

    def test_a_channel_without_a_known_size_is_not_checked(self) -> None:
        raise_if_message_exceeds_channel(
            {"ids": torch.zeros(10**6, dtype=torch.uint8)}, object()
        )

    def test_a_rejected_message_leaves_the_channel_usable(self) -> None:
        channel = ShmChannel(capacity=4, shm_size="1MB")
        channel.max_message_bytes = 1024 * 1024
        with self.assertRaises(RuntimeError):
            raise_if_message_exceeds_channel(
                {"ids": torch.zeros(2 * 1024 * 1024, dtype=torch.uint8)}, channel
            )
        error_message = {SAMPLING_ERROR_KEY: encode_sampling_error("boom")}
        channel.send(error_message)
        with self.assertRaises(RuntimeError):
            raise_if_sampling_error(channel.recv(timeout_ms=30_000))
