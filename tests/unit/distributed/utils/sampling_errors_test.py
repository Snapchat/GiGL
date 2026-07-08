import torch

from gigl.distributed.utils.sampling_errors import (
    SAMPLING_ERROR_KEY,
    encode_sampling_error,
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
