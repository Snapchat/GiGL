import os
from typing import Optional
from unittest import mock

import numpy as np
import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import (
    barrier,
    init_rpc,
    init_worker_group,
    shutdown_rpc,
)
from parameterized import parameterized

from gigl.common.utils.feature_quantization.lossless import ColumnStats, FeatureCodec
from gigl.common.utils.feature_quantization.lossless_compression import (
    FEATURE_COMPRESSION_ENV,
    FeatureCompression,
    compress_loaded_features,
    compress_lossless_columns,
    resolve_feature_compression,
)
from gigl.common.utils.feature_quantization.numpy_ops import quantize_ndarray
from gigl.common.utils.feature_quantization.torch_ops import dequantize_torch_tensor
from gigl.distributed.utils import get_free_port
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE, FeatureQuantizationMetadata
from tests.test_assets.test_case import TestCase

_NUM_ROWS = 50


def _features(num_columns: int, encodings: dict[int, tuple[int, int]]) -> torch.Tensor:
    """Gaussian columns, except the ``encodings`` columns which hold ids of the given inclusive range (both ends present)."""
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(_NUM_ROWS, num_columns, generator=generator)
    for column, (low, high) in encodings.items():
        ids = torch.randint(low, high + 1, (_NUM_ROWS,), generator=generator).float()
        ids[0], ids[1] = low, high
        features[:, column] = ids
    return features


def _compress(
    features: torch.Tensor,
    quantized: Optional[torch.Tensor] = None,
    metadata: Optional[FeatureQuantizationMetadata] = None,
):
    stats = ColumnStats.from_features(features)
    return compress_lossless_columns(features, quantized, metadata, stats, "user")


def _reconstruct(
    raw: Optional[torch.Tensor],
    packed: torch.Tensor,
    metadata: FeatureQuantizationMetadata,
) -> torch.Tensor:
    """What a loader does: decode the packed columns and scatter them with the raw ones into logical order."""
    decoded = dequantize_torch_tensor(packed, metadata)
    scatter = metadata.scatter_index_tensors(torch.device("cpu"))
    out = decoded.new_empty(packed.shape[0], metadata.feature_dim)
    out[:, scatter.quantized] = decoded
    if raw is not None:
        out[:, scatter.raw] = raw
    return out


class CompressLosslessColumnsTest(TestCase):
    def test_integral_columns_move_into_the_sidecar_and_round_trip_exactly(
        self,
    ) -> None:
        features = _features(
            6, {1: (-1, 70), 2: (-1, 200), 3: (-1, 1000), 5: (0, 300000)}
        )

        raw, packed, metadata = _compress(features)

        assert packed is not None and metadata is not None and raw is not None
        self.assertEqual(metadata.lossless_feature_indices, (1, 2, 3, 5))
        self.assertEqual(metadata.raw_feature_indices, (0, 4))
        self.assertEqual(raw.shape, (_NUM_ROWS, 2))
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape, (_NUM_ROWS, metadata.packed_feature_dim))
        self.assertLess(packed.shape[1] + 4 * raw.shape[1], 4 * 6)
        self.assertTrue(torch.equal(_reconstruct(raw, packed, metadata), features))

    def test_lossless_columns_append_to_an_existing_quantized_sidecar(self) -> None:
        # Logical columns: 0 and 3 are 2-bit quantized by preprocessing, 1 and 4 are ids, 2 and 5 are real.
        logical = _features(6, {1: (-1, 70), 4: (0, 5000)})
        logical[:, [0, 3]] = (
            torch.tensor([0.0, 1.0, 2.0, 3.0])
            .repeat(_NUM_ROWS // 4 + 1)[:_NUM_ROWS]
            .unsqueeze(1)
            .repeat(1, 2)
        )
        quantized_metadata = FeatureQuantizationMetadata(
            bits=2,
            feature_dim=6,
            quantized_feature_indices=(0, 3),
            clip_min=0.0,
            clip_max=3.0,
        )
        quantized = torch.from_numpy(
            quantize_ndarray(
                logical[:, [0, 3]].numpy().astype(np.float32),
                bits=2,
                clip_min=0.0,
                clip_max=3.0,
            )
        )
        raw_before = logical[:, [1, 2, 4, 5]]

        raw, packed, metadata = _compress(raw_before, quantized, quantized_metadata)

        assert raw is not None and packed is not None and metadata is not None
        self.assertEqual(metadata.quantized_feature_indices, (0, 3))
        self.assertEqual(metadata.lossless_feature_indices, (1, 4))
        self.assertEqual(metadata.raw_feature_indices, (2, 5))
        # The quantized region is padded so the lossless region starts on a four-byte boundary.
        self.assertEqual(metadata.lossless_byte_offset, 4)
        self.assertTrue(torch.equal(packed[:, :1], quantized))
        before = _reconstruct(raw_before, quantized, quantized_metadata)
        self.assertTrue(torch.equal(_reconstruct(raw, packed, metadata), before))
        # Quantized columns come back at their quantized values, lossless ones exactly.
        self.assertTrue(torch.equal(before[:, [1, 4]], logical[:, [1, 4]]))

    def test_non_integral_columns_stay_raw(self) -> None:
        features = _features(6, {i: (0, 100) for i in range(1, 5)})
        features[7, 4] = 2.5  # not an id

        raw, packed, metadata = _compress(features)

        assert raw is not None and packed is not None and metadata is not None
        self.assertEqual(metadata.lossless_feature_indices, (1, 2, 3))
        self.assertEqual(metadata.raw_feature_indices, (0, 4, 5))
        self.assertTrue(torch.equal(_reconstruct(raw, packed, metadata), features))

    def test_one_hot_and_id_columns_get_their_narrowest_encoding(self) -> None:
        features = _features(4, {0: (0, 1), 1: (0, 1), 2: (0, 70000), 3: (-5, 5)})
        features[5, 3] = -0.0  # a -0.0 would decode as +0.0 in an integer encoding

        raw, packed, metadata = _compress(features)

        assert raw is not None and packed is not None and metadata is not None
        self.assertEqual(metadata.lossless_feature_indices, (0, 1, 2))
        self.assertEqual(metadata.lossless_column_encodings, ("bit", "bit", "int32"))
        self.assertTrue(torch.equal(_reconstruct(raw, packed, metadata), features))

    def test_all_columns_integral_leaves_no_raw_matrix(self) -> None:
        features = _features(4, {i: (0, 100) for i in range(4)})

        raw, packed, metadata = _compress(features)

        assert packed is not None and metadata is not None
        self.assertIsNone(raw)
        self.assertEqual(metadata.raw_feature_dim, 0)
        self.assertTrue(torch.equal(_reconstruct(None, packed, metadata), features))

    @parameterized.expand(
        [
            ("nothing_integral", {}),
            ("would_not_shrink", {0: (0, 100000)}),
        ]
    )
    def test_inputs_are_returned_untouched_when_compression_does_not_help(
        self, _: str, encodings: dict[int, tuple[int, int]]
    ) -> None:
        features = _features(2, encodings)

        raw, packed, metadata = _compress(features)

        self.assertIs(raw, features)
        self.assertIsNone(packed)
        self.assertIsNone(metadata)

    def test_a_second_lossless_region_or_a_wrong_width_is_rejected(self) -> None:
        features = _features(3, {})
        existing = FeatureQuantizationMetadata(
            bits=8,
            feature_dim=3,
            lossless_feature_indices=(0,),
            lossless_column_encodings=("int8",),
        )
        with self.assertRaises(ValueError):
            _compress(
                features[:, 1:],
                FeatureCodec(("int8",)).encode(features[:, :1]),
                existing,
            )
        quantized_metadata = FeatureQuantizationMetadata(
            bits=2,
            feature_dim=3,
            quantized_feature_indices=(0,),
            clip_min=0.0,
            clip_max=3.0,
        )
        with self.assertRaises(ValueError):
            _compress(
                features,
                torch.zeros(_NUM_ROWS, 1, dtype=torch.uint8),
                quantized_metadata,
            )

    def test_a_partition_without_rows_gets_an_empty_sidecar_of_the_merged_layout(
        self,
    ) -> None:
        features = _features(5, {i: (0, 100) for i in range(5)})
        stats = ColumnStats.from_features(features)

        raw, packed, metadata = compress_lossless_columns(
            None, None, None, stats, "user"
        )

        assert packed is not None and metadata is not None
        self.assertIsNone(raw)
        self.assertEqual(packed.shape, (0, metadata.packed_feature_dim))
        self.assertEqual(
            metadata,
            compress_lossless_columns(features, None, None, stats, "user")[2],
        )


def _compress_on_rank(
    rank: int,
    master_port: int,
    shards: dict[int, Optional[torch.Tensor]],
    output: dict,
) -> None:
    init_worker_group(world_size=2, rank=rank)
    init_rpc(master_addr="127.0.0.1", master_port=master_port, num_rpc_threads=4)
    raw, packed, metadata = compress_loaded_features(
        features=shards[rank],
        quantized_features=None,
        metadata=None,
        is_homogeneous=True,
        default_key=DEFAULT_HOMOGENEOUS_NODE_TYPE,
    )
    assert raw is None or isinstance(raw, torch.Tensor)
    assert packed is None or isinstance(packed, torch.Tensor)
    output[rank] = (
        metadata,
        None if raw is None else tuple(raw.shape),
        None if packed is None else tuple(packed.shape),
    )
    barrier()
    shutdown_rpc()


class CompressLoadedFeaturesAcrossRanksTest(TestCase):
    def test_ranks_agree_on_the_layout_when_one_has_an_unfit_value_and_one_no_rows(
        self,
    ) -> None:
        shard = _features(6, {i: (0, 100) for i in range(5)})
        shard[2, 3] = 2.5  # not integral, and only this rank sees it
        output = mp.Manager().dict()

        mp.spawn(
            _compress_on_rank,
            args=(get_free_port(), {0: shard, 1: None}, output),
            nprocs=2,
        )

        (metadata0, raw0, packed0), (metadata1, raw1, packed1) = output[0], output[1]
        self.assertEqual(metadata0, metadata1)
        # Column 3 holds a fraction and column 5 real values, so both stay raw on every rank.
        self.assertEqual(metadata0.lossless_feature_indices, (0, 1, 2, 4))
        self.assertEqual(raw0, (_NUM_ROWS, 2))
        self.assertEqual(packed0, (_NUM_ROWS, metadata0.packed_feature_dim))
        self.assertIsNone(raw1)
        self.assertEqual(packed1, (0, metadata0.packed_feature_dim))


class FeatureQuantizationMetadataLayoutTest(TestCase):
    def test_packed_row_layout_and_validation(self) -> None:
        metadata = FeatureQuantizationMetadata(
            bits=4,
            feature_dim=7,
            quantized_feature_indices=(0, 1, 2),
            clip_min=0.0,
            clip_max=1.0,
            lossless_feature_indices=(4, 6),
            lossless_column_encodings=("int16", "int8"),
        )
        # 3 four-bit columns take 2 bytes, padded to 4; the lossless region is 2 + 1 bytes padded to 4.
        self.assertEqual(metadata.quantized_packed_dim, 2)
        self.assertEqual(metadata.lossless_byte_offset, 4)
        self.assertEqual(metadata.packed_feature_dim, 8)
        self.assertEqual(metadata.packed_feature_indices, (0, 1, 2, 4, 6))
        self.assertEqual(metadata.raw_feature_indices, (3, 5))
        with self.assertRaises(ValueError):
            FeatureQuantizationMetadata(
                bits=4,
                feature_dim=7,
                quantized_feature_indices=(0,),
                lossless_feature_indices=(0,),
                lossless_column_encodings=("int8",),
            )
        with self.assertRaises(ValueError):
            FeatureQuantizationMetadata(
                bits=4,
                feature_dim=7,
                lossless_feature_indices=(1, 2),
                lossless_column_encodings=("int8",),
            )

    def test_metadata_without_a_lossless_region_keeps_its_old_layout(self) -> None:
        metadata = FeatureQuantizationMetadata(
            bits=2,
            feature_dim=5,
            quantized_feature_indices=(0, 2, 3),
            clip_min=0.0,
            clip_max=3.0,
        )
        self.assertEqual(metadata.packed_feature_dim, 1)
        self.assertEqual(metadata.raw_feature_indices, (1, 4))


class ResolveFeatureCompressionTest(TestCase):
    def test_environment_variable_selects_the_mode(self) -> None:
        for raw_value, expected in (
            ("", FeatureCompression.AUTO),
            ("auto", FeatureCompression.AUTO),
            ("OFF", FeatureCompression.OFF),
            ("Off", FeatureCompression.OFF),
        ):
            with mock.patch.dict(os.environ, {FEATURE_COMPRESSION_ENV: raw_value}):
                self.assertEqual(resolve_feature_compression(), expected)
        with mock.patch.dict(os.environ):
            os.environ.pop(FEATURE_COMPRESSION_ENV, None)
            self.assertEqual(resolve_feature_compression(), FeatureCompression.AUTO)

    def test_unknown_value_raises(self) -> None:
        with mock.patch.dict(os.environ, {FEATURE_COMPRESSION_ENV: "lossy"}):
            with self.assertRaises(ValueError):
                resolve_feature_compression()


if __name__ == "__main__":
    absltest.main()
