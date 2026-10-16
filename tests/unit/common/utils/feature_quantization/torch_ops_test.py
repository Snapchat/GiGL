import numpy as np
import torch

from gigl.common.utils.feature_quantization.lossless import FeatureCodec
from gigl.common.utils.feature_quantization.numpy_ops import quantize_ndarray
from gigl.common.utils.feature_quantization.torch_ops import dequantize_torch_tensor
from gigl.types.graph import FeatureQuantizationMetadata
from tests.test_assets.test_case import TestCase


class TorchFeatureQuantizationOpsTest(TestCase):
    def test_quantize_numpy_dequantize_torch_round_trip_single_bit(self) -> None:
        features = np.array([[-2.0, -0.5, 0.5, 3.0, -3.0]], dtype=np.float32)
        metadata = FeatureQuantizationMetadata(
            bits=1,
            feature_dim=features.shape[1],
            quantized_feature_indices=tuple(range(features.shape[1])),
            neg_mean=-1.25,
            pos_mean=1.75,
        )

        assert metadata.bits is not None
        packed = quantize_ndarray(features, bits=metadata.bits)
        actual = dequantize_torch_tensor(torch.from_numpy(packed), metadata=metadata)

        torch.testing.assert_close(
            actual, torch.tensor([[-1.25, -1.25, 1.75, 1.75, -1.25]])
        )

    def test_quantize_numpy_dequantize_torch_round_trip_multi_bit(self) -> None:
        features = np.array([[-1.0, 0.0, 1.0, 2.0, 4.0]], dtype=np.float32)
        clip_min = 0.0
        clip_max = 3.0
        metadata = FeatureQuantizationMetadata(
            bits=2,
            feature_dim=features.shape[1],
            quantized_feature_indices=tuple(range(features.shape[1])),
            clip_min=clip_min,
            clip_max=clip_max,
        )

        assert metadata.bits is not None
        packed = quantize_ndarray(
            features, bits=metadata.bits, clip_min=clip_min, clip_max=clip_max
        )
        actual = dequantize_torch_tensor(torch.from_numpy(packed), metadata=metadata)

        torch.testing.assert_close(actual, torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0]]))

    def test_dequantize_torch_tensor_single_bit_unpacks_full_byte(self) -> None:
        # 0b10101010 = 170 unpacks high-bits-first to [1, 0, 1, 0, 1, 0, 1, 0].
        # Code 1 maps to pos_mean and code 0 maps to neg_mean.
        metadata = FeatureQuantizationMetadata(
            bits=1,
            feature_dim=8,
            quantized_feature_indices=tuple(range(8)),
            neg_mean=-1.5,
            pos_mean=2.5,
        )
        actual = dequantize_torch_tensor(
            torch.tensor([[170]], dtype=torch.uint8), metadata=metadata
        )
        torch.testing.assert_close(
            actual,
            torch.tensor([[2.5, -1.5, 2.5, -1.5, 2.5, -1.5, 2.5, -1.5]]),
        )

    def test_dequantize_torch_tensor_single_bit_trims_padded_codes(self) -> None:
        # 0b10101000 = 168 unpacks to [1, 0, 1, 0, 1, 0, 0, 0].
        # The final three zeros are padding and are trimmed to five logical features.
        metadata = FeatureQuantizationMetadata(
            bits=1,
            feature_dim=5,
            quantized_feature_indices=tuple(range(5)),
            neg_mean=-1.5,
            pos_mean=2.5,
        )
        actual = dequantize_torch_tensor(
            torch.tensor([[168]], dtype=torch.uint8), metadata=metadata
        )
        torch.testing.assert_close(actual, torch.tensor([[2.5, -1.5, 2.5, -1.5, 2.5]]))

    def test_dequantize_torch_tensor_two_bit_unpacks_ascending_codes(self) -> None:
        # 27 = 0b00011011 unpacks high-bits-first into 2-bit codes [0, 1, 2, 3].
        # With clip range [0, 3], those codes dequantize exactly to the same values.
        metadata = FeatureQuantizationMetadata(
            bits=2,
            feature_dim=4,
            quantized_feature_indices=tuple(range(4)),
            clip_min=0.0,
            clip_max=3.0,
        )
        actual = dequantize_torch_tensor(
            torch.tensor([[27]], dtype=torch.uint8), metadata=metadata
        )
        torch.testing.assert_close(actual, torch.tensor([[0.0, 1.0, 2.0, 3.0]]))

    def test_dequantize_torch_tensor_two_bit_unpacks_descending_codes(self) -> None:
        # 228 = 0b11100100 unpacks high-bits-first into 2-bit codes [3, 2, 1, 0].
        # With clip range [0, 3], those codes dequantize exactly to the same values.
        metadata = FeatureQuantizationMetadata(
            bits=2,
            feature_dim=4,
            quantized_feature_indices=tuple(range(4)),
            clip_min=0.0,
            clip_max=3.0,
        )
        actual = dequantize_torch_tensor(
            torch.tensor([[228]], dtype=torch.uint8), metadata=metadata
        )
        torch.testing.assert_close(actual, torch.tensor([[3.0, 2.0, 1.0, 0.0]]))

    def test_dequantize_torch_tensor_two_bit_trims_padded_codes(self) -> None:
        # [27, 64] unpacks to [0, 1, 2, 3, 1, 0, 0, 0].
        # The final three zeros are padding and are trimmed to five logical features.
        metadata = FeatureQuantizationMetadata(
            bits=2,
            feature_dim=5,
            quantized_feature_indices=tuple(range(5)),
            clip_min=0.0,
            clip_max=3.0,
        )
        actual = dequantize_torch_tensor(
            torch.tensor([[27, 64]], dtype=torch.uint8), metadata=metadata
        )
        torch.testing.assert_close(actual, torch.tensor([[0.0, 1.0, 2.0, 3.0, 1.0]]))

    def test_dequantize_torch_tensor_four_bit_unpacks_ascending_codes(self) -> None:
        # 15 = 0x0F unpacks high-bits-first into 4-bit codes [0, 15].
        # With clip range [0, 15], those codes dequantize exactly to the same values.
        metadata = FeatureQuantizationMetadata(
            bits=4,
            feature_dim=2,
            quantized_feature_indices=tuple(range(2)),
            clip_min=0.0,
            clip_max=15.0,
        )
        actual = dequantize_torch_tensor(
            torch.tensor([[15]], dtype=torch.uint8), metadata=metadata
        )
        torch.testing.assert_close(actual, torch.tensor([[0.0, 15.0]]))

    def test_dequantize_torch_tensor_four_bit_unpacks_descending_codes(self) -> None:
        # 240 = 0xF0 unpacks high-bits-first into 4-bit codes [15, 0].
        # With clip range [0, 15], those codes dequantize exactly to the same values.
        metadata = FeatureQuantizationMetadata(
            bits=4,
            feature_dim=2,
            quantized_feature_indices=tuple(range(2)),
            clip_min=0.0,
            clip_max=15.0,
        )
        actual = dequantize_torch_tensor(
            torch.tensor([[240]], dtype=torch.uint8), metadata=metadata
        )
        torch.testing.assert_close(actual, torch.tensor([[15.0, 0.0]]))

    def test_dequantize_torch_tensor_four_bit_trims_padded_codes(self) -> None:
        # [15, 128] unpacks to 4-bit codes [0, 15, 8, 0].
        # The final zero is padding and is trimmed to three logical features.
        metadata = FeatureQuantizationMetadata(
            bits=4,
            feature_dim=3,
            quantized_feature_indices=tuple(range(3)),
            clip_min=0.0,
            clip_max=15.0,
        )
        actual = dequantize_torch_tensor(
            torch.tensor([[15, 128]], dtype=torch.uint8), metadata=metadata
        )
        torch.testing.assert_close(actual, torch.tensor([[0.0, 15.0, 8.0]]))

    def test_dequantize_torch_tensor_eight_bit_reads_one_code_per_column(self) -> None:
        # 8-bit quantization stores one code per uint8 column, so unpacking preserves order.
        metadata = FeatureQuantizationMetadata(
            bits=8,
            feature_dim=3,
            quantized_feature_indices=tuple(range(3)),
            clip_min=0.0,
            clip_max=255.0,
        )
        actual = dequantize_torch_tensor(
            torch.tensor([[0, 128, 255]], dtype=torch.uint8), metadata=metadata
        )
        torch.testing.assert_close(actual, torch.tensor([[0.0, 128.0, 255.0]]))

    def test_dequantize_torch_tensor_rejects_wrong_packed_feature_dim(self) -> None:
        # Five 2-bit features need two packed bytes: four codes in the first byte,
        # then one code plus padding in the second byte. One input byte is too short.
        metadata = FeatureQuantizationMetadata(
            bits=2,
            feature_dim=5,
            quantized_feature_indices=tuple(range(5)),
            clip_min=0.0,
            clip_max=3.0,
        )
        with self.assertRaises(ValueError):
            dequantize_torch_tensor(
                torch.tensor([[27]], dtype=torch.uint8), metadata=metadata
            )

    def test_dequantize_torch_tensor_requires_single_bit_neg_mean(self) -> None:
        metadata = FeatureQuantizationMetadata(
            bits=1,
            feature_dim=2,
            quantized_feature_indices=(0, 1),
            pos_mean=1.0,
        )
        with self.assertRaises(ValueError):
            dequantize_torch_tensor(
                torch.tensor([[128]], dtype=torch.uint8), metadata=metadata
            )

    def test_dequantize_torch_tensor_requires_single_bit_pos_mean(self) -> None:
        metadata = FeatureQuantizationMetadata(
            bits=1,
            feature_dim=2,
            quantized_feature_indices=(0, 1),
            neg_mean=-1.0,
        )
        with self.assertRaises(ValueError):
            dequantize_torch_tensor(
                torch.tensor([[128]], dtype=torch.uint8), metadata=metadata
            )

    def test_dequantize_torch_tensor_requires_multi_bit_clip_min(self) -> None:
        metadata = FeatureQuantizationMetadata(
            bits=4,
            feature_dim=2,
            quantized_feature_indices=(0, 1),
            clip_max=1.0,
        )
        with self.assertRaises(ValueError):
            dequantize_torch_tensor(
                torch.tensor([[15]], dtype=torch.uint8), metadata=metadata
            )

    def test_dequantize_torch_tensor_requires_multi_bit_clip_max(self) -> None:
        metadata = FeatureQuantizationMetadata(
            bits=4,
            feature_dim=2,
            quantized_feature_indices=(0, 1),
            clip_min=0.0,
        )
        with self.assertRaises(ValueError):
            dequantize_torch_tensor(
                torch.tensor([[15]], dtype=torch.uint8), metadata=metadata
            )

    def test_dequantize_torch_tensor_decodes_quantized_then_lossless_columns(
        self,
    ) -> None:
        # Logical columns 0 and 2 are 2-bit quantized, columns 1 and 4 are ids stored losslessly.
        quantized_features = np.array([[0.0, 3.0], [2.0, 1.0]], dtype=np.float32)
        ids = torch.tensor([[-1.0, 40000.0], [127.0, -7.0]])
        codec = FeatureCodec(("int8", "int32"))
        metadata = FeatureQuantizationMetadata(
            bits=2,
            feature_dim=5,
            quantized_feature_indices=(0, 2),
            clip_min=0.0,
            clip_max=3.0,
            lossless_feature_indices=(1, 4),
            lossless_column_kinds=codec.column_dtypes,
        )
        quantized = torch.from_numpy(
            quantize_ndarray(quantized_features, bits=2, clip_min=0.0, clip_max=3.0)
        )
        packed = torch.cat(
            [
                torch.nn.functional.pad(
                    quantized, (0, metadata.lossless_byte_offset - quantized.shape[1])
                ),
                codec.encode(ids),
            ],
            dim=1,
        )

        decoded = dequantize_torch_tensor(packed, metadata=metadata)

        self.assertEqual(packed.shape[1], metadata.packed_feature_dim)
        torch.testing.assert_close(
            decoded,
            torch.cat([torch.from_numpy(quantized_features), ids], dim=1),
            atol=0,
            rtol=0,
        )

    def test_dequantize_torch_tensor_decodes_lossless_only_rows(self) -> None:
        codec = FeatureCodec(("uint8_bias1", "int16"))
        ids = torch.tensor([[-1.0, -300.0], [254.0, 300.0]])
        metadata = FeatureQuantizationMetadata(
            bits=None,
            feature_dim=2,
            lossless_feature_indices=(0, 1),
            lossless_column_kinds=codec.column_dtypes,
        )

        decoded = dequantize_torch_tensor(codec.encode(ids), metadata=metadata)

        torch.testing.assert_close(decoded, ids, atol=0, rtol=0)

    def test_dequantize_torch_tensor_decodes_batched_leading_dimensions(self) -> None:
        codec = FeatureCodec(("int8", "int16"))
        ids = torch.tensor([[[-1.0, 5.0], [7.0, -9.0]]])
        metadata = FeatureQuantizationMetadata(
            bits=None,
            feature_dim=2,
            lossless_feature_indices=(0, 1),
            lossless_column_kinds=codec.column_dtypes,
        )
        packed = codec.encode(ids.reshape(-1, 2)).reshape(1, 2, -1)

        decoded = dequantize_torch_tensor(packed, metadata=metadata)

        torch.testing.assert_close(decoded, ids, atol=0, rtol=0)

    def test_dequantize_torch_tensor_rejects_a_row_of_the_wrong_width(self) -> None:
        codec = FeatureCodec(("int8",))
        metadata = FeatureQuantizationMetadata(
            bits=None,
            feature_dim=1,
            lossless_feature_indices=(0,),
            lossless_column_kinds=codec.column_dtypes,
        )
        with self.assertRaises(ValueError):
            dequantize_torch_tensor(
                torch.zeros(2, metadata.packed_feature_dim + 4, dtype=torch.uint8),
                metadata=metadata,
            )

    def test_dequantize_torch_tensor_of_metadata_without_packed_columns_is_empty(
        self,
    ) -> None:
        metadata = FeatureQuantizationMetadata(bits=None, feature_dim=3)

        decoded = dequantize_torch_tensor(
            torch.zeros(4, 0, dtype=torch.uint8), metadata=metadata
        )

        self.assertEqual(decoded.shape, (4, 0))
        self.assertEqual(decoded.dtype, torch.float32)
