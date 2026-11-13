import pickle
import unittest
from typing import Optional
from unittest import mock

import torch
from absl.testing import absltest

from gigl.common.utils.feature_quantization import lossless
from gigl.common.utils.feature_quantization.lossless import (
    FeatureCodec,
    chunk_rows,
    narrowest_integer_encoding,
)
from tests.test_assets.test_case import TestCase

# A miniature feature set: real-valued columns around categorical id blocks of every width. Every
# id block has an out-of-vocabulary id of -1, like a vocabulary feature.
#   columns 0-4 dense | 5-7 ids in [-1, 70] | 8-9 dense | 10-13 ids in [-1, 210]
#   | 14-16 ids in [-1, 1247] | 17 dense | 18-19 ids in [-1, 265771] | 20-21 ids in [0, 300]
#   | 22-30 one-hot (0 or 1)
_FEATURES: list[tuple[str, int, Optional[tuple[int, int]]]] = [
    ("dense_a", 5, None),
    ("small_ids", 3, (-1, 70)),
    ("dense_b", 2, None),
    ("byte_ids", 4, (-1, 210)),
    ("short_ids", 3, (-1, 1247)),
    ("dense_c", 1, None),
    ("wide_ids", 2, (-1, 265771)),
    ("unsigned_ids", 2, (0, 300)),
    ("one_hot", 9, (0, 1)),
]
_NUM_COLUMNS = sum(width for _, width, _ in _FEATURES)
_EXPECTED_ENCODINGS = (
    ["float32"] * 5
    + ["int8"] * 3
    + ["float32"] * 2
    + ["uint8_bias1"] * 4
    + ["int16"] * 3
    + ["float32"]
    + ["int32"] * 2
    + ["int16"] * 2
    + ["bit"] * 9
)


def _features(num_rows: int, seed: int) -> torch.Tensor:
    """Float32 rows: gaussian dense columns (plus inf, NaN, -0.0) and ids that hit both ends of their range."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_rows, _NUM_COLUMNS, generator=generator)
    features[0, 0] = float("inf")
    features[1, 1] = float("nan")
    features[2, 2] = -0.0
    column = 0
    for _, width, domain in _FEATURES:
        if domain is not None:
            low, high = domain
            ids = torch.randint(
                low, high + 1, (num_rows, width), generator=generator
            ).float()
            ids[3], ids[4] = low, high
            features[:, column : column + width] = ids
        column += width
    return features


def _declared(dense_dtype: str = "float32") -> FeatureCodec:
    """Declares the narrowest encoding that holds each id block's domain, and ``dense_dtype`` for the rest."""
    encodings: list[str] = []
    for _, width, domain in _FEATURES:
        encoding = (
            dense_dtype if domain is None else narrowest_integer_encoding(*domain)
        )
        assert encoding is not None
        encodings += [encoding] * width
    return FeatureCodec(tuple(encodings))


def _same_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(left.view(torch.int32), right.view(torch.int32))


class FeatureCodecTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.features = _features(1000, seed=0)
        self.codec = _declared()

    def test_narrowest_encoding_is_chosen_per_column_and_rows_shrink(self) -> None:
        self.assertEqual(list(self.codec.column_encodings), _EXPECTED_ENCODINGS)
        # 8 float32 + 2 int32 = 40 B, then 5 int16 = 10 B, then 7 one-byte columns, then 9 bits in
        # 2 bytes, padded to 4.
        self.assertEqual(self.codec.row_bytes, 60)
        self.assertEqual(narrowest_integer_encoding(0, 1), "bit")
        self.assertEqual(narrowest_integer_encoding(0, 0), "bit")
        self.assertEqual(narrowest_integer_encoding(-1, 1), "int8")
        self.assertEqual(narrowest_integer_encoding(0, 200), "uint8")
        self.assertLess(self.codec.row_bytes, 4 * _NUM_COLUMNS)

    def test_round_trip_is_bit_exact_including_extremes_and_non_finite_dense(
        self,
    ) -> None:
        packed = self.codec.encode(self.features)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape, (1000, self.codec.row_bytes))
        decoded = self.codec.decode(packed)
        self.assertEqual(decoded.dtype, torch.float32)
        self.assertTrue(_same_bits(decoded, self.features))

    def test_chunks_target_a_byte_size_and_encoding_is_chunk_independent(self) -> None:
        # 8 MiB of float32 is 4096 rows of 512 columns; a row wider than a chunk still gets one row.
        self.assertEqual(chunk_rows(512, 4), 4096)
        self.assertEqual(chunk_rows(1 << 30, 4), 1)
        whole = self.codec.encode(self.features)
        with mock.patch.object(lossless, "_CHUNK_BYTES", 7 * 4 * _NUM_COLUMNS):
            self.assertEqual(chunk_rows(_NUM_COLUMNS, 4), 7)
            chunked = self.codec.encode(self.features)
        self.assertTrue(torch.equal(chunked, whole))

    def test_round_trip_covers_domain_edges(self) -> None:
        features = _features(8, seed=1)
        column = 5  # ids in [-1, 70]
        features[:, column] = torch.tensor(
            [-1.0, 0.0, 1.0, 70.0, -1.0, 70.0, 0.0, 69.0]
        )
        column = 18  # ids in [-1, 265771]
        features[:, column] = torch.tensor(
            [-1.0, 0.0, 255.0, 256.0, 65535.0, 65536.0, 265771.0, -1.0]
        )
        codec = _declared()
        self.assertTrue(_same_bits(codec.decode(codec.encode(features)), features))

    def test_decode_handles_gathered_strided_misaligned_and_column_strided_rows(
        self,
    ) -> None:
        packed = self.codec.encode(self.features)
        index = torch.tensor([999, 0, 17, 17, 500, 3])
        self.assertTrue(
            _same_bits(self.codec.decode(packed[index]), self.features[index])
        )
        # A strided view of the packed matrix (every other row) is not contiguous.
        self.assertTrue(_same_bits(self.codec.decode(packed[::2]), self.features[::2]))

        # Rows that start one byte into a larger buffer cannot be reinterpreted in place.
        buffer = torch.zeros(10 * self.codec.row_bytes + 1, dtype=torch.uint8)
        buffer[1:] = packed[:10].flatten()
        shifted = buffer[1:].view(10, self.codec.row_bytes)
        self.assertTrue(shifted.storage_offset() % 4 != 0)
        self.assertTrue(_same_bits(self.codec.decode(shifted), self.features[:10]))

        # Every packed byte of a row pair interleaved into one wider tensor: columns are not contiguous.
        interleaved = torch.stack([packed[:10], packed[:10]], dim=-1)[:, :, 0]
        self.assertNotEqual(interleaved.stride(1), 1)
        self.assertTrue(_same_bits(self.codec.decode(interleaved), self.features[:10]))

    def test_bit_columns_pack_eight_per_byte_least_significant_bit_first(self) -> None:
        for num_bit_columns in (1, 7, 8, 9, 17):
            codec = FeatureCodec(("float32", "int16") + ("bit",) * num_bit_columns)
            generator = torch.Generator().manual_seed(num_bit_columns)
            bits = torch.randint(0, 2, (50, num_bit_columns), generator=generator)
            features = torch.cat([torch.randn(50, 2), bits.float()], dim=1)
            features[:, 1] = torch.randint(-300, 300, (50,), generator=generator)
            packed = codec.encode(features)
            num_bit_bytes = (num_bit_columns + 7) // 8
            self.assertEqual(codec.row_bytes, (4 + 2 + num_bit_bytes + 3) // 4 * 4)
            self.assertTrue(_same_bits(codec.decode(packed), features))
            # The bit group sits after the 4-byte and 2-byte groups.
            bit_bytes = packed[:, 6 : 6 + num_bit_bytes]
            for column in range(num_bit_columns):
                stored = (bit_bytes[:, column // 8] >> (column % 8)) & 1
                self.assertTrue(torch.equal(stored.long(), bits[:, column]))
            # Padding bits of the last byte stay zero.
            if num_bit_columns % 8:
                self.assertTrue(
                    torch.equal(
                        bit_bytes[:, -1] >> (num_bit_columns % 8),
                        torch.zeros(50, dtype=torch.uint8),
                    )
                )

    def test_empty_batch_round_trips(self) -> None:
        empty = self.features[:0]
        decoded = self.codec.decode(self.codec.encode(empty))
        self.assertEqual(decoded.shape, (0, _NUM_COLUMNS))

    def test_hand_written_spec_round_trips_every_encoding(self) -> None:
        codec = FeatureCodec(("float32", "int8", "uint8", "int16", "uint8_bias1"))
        features = torch.tensor(
            [[0.5, -128.0, 255.0, -32768.0, -1.0], [-0.0, 127.0, 0.0, 32767.0, 254.0]]
        )
        self.assertEqual(
            codec.row_bytes, 12
        )  # 9 bytes of columns, padded to a multiple of 4
        self.assertTrue(_same_bits(codec.decode(codec.encode(features)), features))

    def test_bfloat16_dense_columns_round_to_bfloat16_and_ids_stay_exact(
        self,
    ) -> None:
        codec = _declared("bfloat16")
        decoded = codec.decode(codec.encode(self.features))
        dense = [
            c
            for c, encoding in enumerate(codec.column_encodings)
            if encoding == "bfloat16"
        ]
        ids = [c for c in range(_NUM_COLUMNS) if c not in dense]
        self.assertEqual(len(dense), 8)
        expected = self.features[:, dense].to(torch.bfloat16).float()
        self.assertTrue(torch.equal(decoded[:, dense].isnan(), expected.isnan()))
        self.assertTrue(
            torch.equal(torch.nan_to_num(decoded[:, dense]), torch.nan_to_num(expected))
        )
        self.assertTrue(torch.equal(decoded[:, ids], self.features[:, ids]))
        self.assertLess(codec.row_bytes, self.codec.row_bytes)

    def test_encode_and_decode_refuse_inputs_the_codec_cannot_represent(self) -> None:
        features = self.features.clone()
        features[0, 6] = 300.0  # the codec declares int8 for this column
        with self.assertRaises(ValueError):
            self.codec.encode(features)
        features = self.features.clone()
        features[0, 6] = 1.5  # not an integer
        with self.assertRaises(ValueError):
            self.codec.encode(features)
        for bad_bit in (0.5, 2.0, -1.0, float("nan")):
            features = self.features.clone()
            features[0, 25] = bad_bit  # a one-hot column
            with self.assertRaises(ValueError):
                self.codec.encode(features)
        with self.assertRaises(ValueError):
            self.codec.encode(self.features[:, :-1])
        with self.assertRaises(ValueError):
            self.codec.decode(
                torch.zeros(3, self.codec.row_bytes + 4, dtype=torch.uint8)
            )
        with self.assertRaises(ValueError):
            self.codec.decode(torch.zeros(3, self.codec.row_bytes))

    def test_skipping_validation_packs_valid_data_to_the_same_bytes(self) -> None:
        self.assertTrue(
            torch.equal(
                self.codec.encode(self.features, validate=False),
                self.codec.encode(self.features),
            )
        )

    def test_encode_checks_float64_ids_before_casting_to_float32(self) -> None:
        # 2**24 + 1 is a valid int32 id that float32 cannot hold; the stored id must not be rounded.
        codec = FeatureCodec(("int32",))
        packed = codec.encode(torch.tensor([[16777217.0]], dtype=torch.float64))
        self.assertEqual(int(packed.view(torch.int32)), 16777217)
        with self.assertRaises(ValueError):
            FeatureCodec(("bit",)).encode(
                torch.tensor([[1.0 + 2.0**-40]], dtype=torch.float64)
            )
        # Neither a float64 fraction nor an out-of-range value may be rounded into validity.
        with self.assertRaises(ValueError):
            FeatureCodec(("int8",)).encode(
                torch.tensor([[1.0 + 2.0**-40]], dtype=torch.float64)
            )
        with self.assertRaises(ValueError):
            FeatureCodec(("int8",)).encode(
                torch.tensor([[127.0 + 2.0**-40]], dtype=torch.float64)
            )

    def test_encode_rejects_non_float32_input_for_dense_encodings(self) -> None:
        float64_features = torch.tensor([[0.1, 1.0]], dtype=torch.float64)
        for dense_encoding in ("float32", "bfloat16"):
            with self.assertRaises(ValueError):
                FeatureCodec((dense_encoding, "int8")).encode(float64_features)
        # Integer encodings validate before casting, so float64 input stays allowed.
        FeatureCodec(("int8", "int8")).encode(float64_features.round())

    def test_codec_survives_pickle(self) -> None:
        restored = pickle.loads(pickle.dumps(self.codec))
        self.assertEqual(restored, self.codec)
        self.assertTrue(
            _same_bits(restored.decode(self.codec.encode(self.features)), self.features)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_decode_on_cuda_matches_cpu_decode(self) -> None:
        packed = self.codec.encode(self.features)
        decoded = self.codec.decode(packed.cuda())
        self.assertEqual(decoded.device.type, "cuda")
        self.assertTrue(_same_bits(decoded.cpu(), self.features))
        index = torch.tensor([5, 1, 999, 1], device="cuda")
        self.assertTrue(
            _same_bits(
                self.codec.decode(packed.cuda()[index]).cpu(),
                self.features[index.cpu()],
            )
        )


if __name__ == "__main__":
    absltest.main()
