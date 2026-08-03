import numpy as np

from gigl.common.utils.feature_quantization.numpy_ops import quantize_ndarray
from tests.test_assets.test_case import TestCase


class NumpyFeatureQuantizationOpsTest(TestCase):
    def test_quantize_ndarray_single_bit_packs_full_byte(self) -> None:
        # Values > 0 become 1 and the rest become 0:
        # [-1, 0, 0.5, 2, -0.5, 3, 4, -4] -> [0, 0, 1, 1, 0, 1, 1, 0].
        # High-bits-first packing gives 0b00110110 = 54.
        features = np.array([[-1.0, 0.0, 0.5, 2.0, -0.5, 3.0, 4.0, -4.0]])

        actual = quantize_ndarray(features, bits=1, stats={})

        np.testing.assert_array_equal(actual, np.array([[54]], dtype=np.uint8))

    def test_quantize_ndarray_single_bit_pads_final_byte(self) -> None:
        # Values > 0 become 1 and the rest become 0:
        # [1, -1, 2, -2, 3] becomes [1, 0, 1, 0, 1].
        # Padding fills the remaining bit slots with zeros: 0b10101000 = 168.
        features = np.array([[1.0, -1.0, 2.0, -2.0, 3.0]])

        actual = quantize_ndarray(features, bits=1, stats={})

        np.testing.assert_array_equal(actual, np.array([[168]], dtype=np.uint8))

    def test_quantize_ndarray_two_bit_packs_full_byte_ascending_codes(self) -> None:
        # With clip range [0, 3], these values equal their 2-bit codes.
        # [0, 1, 2, 3] packs as 00 01 10 11 = 0b00011011 = 27.
        features = np.array([[0.0, 1.0, 2.0, 3.0]])

        actual = quantize_ndarray(
            features, bits=2, stats={"clip_min": 0.0, "clip_max": 3.0}
        )

        np.testing.assert_array_equal(actual, np.array([[27]], dtype=np.uint8))

    def test_quantize_ndarray_two_bit_packs_full_byte_descending_codes(self) -> None:
        # With clip range [0, 3], these values equal their 2-bit codes.
        # [3, 2, 1, 0] packs as 11 10 01 00 = 0b11100100 = 228.
        features = np.array([[3.0, 2.0, 1.0, 0.0]])

        actual = quantize_ndarray(
            features, bits=2, stats={"clip_min": 0.0, "clip_max": 3.0}
        )

        np.testing.assert_array_equal(actual, np.array([[228]], dtype=np.uint8))

    def test_quantize_ndarray_two_bit_pads_final_byte(self) -> None:
        # The first four codes [0, 1, 2, 3] pack into byte 27.
        # The leftover code [1] starts the next byte as 01 00 00 00 = 64.
        features = np.array([[0.0, 1.0, 2.0, 3.0, 1.0]])

        actual = quantize_ndarray(
            features, bits=2, stats={"clip_min": 0.0, "clip_max": 3.0}
        )

        np.testing.assert_array_equal(actual, np.array([[27, 64]], dtype=np.uint8))

    def test_quantize_ndarray_four_bit_packs_full_byte_ascending_codes(self) -> None:
        # With clip range [0, 15], these values equal their 4-bit codes.
        # [0, 15] packs as 0000 1111 = 15.
        features = np.array([[0.0, 15.0]])

        actual = quantize_ndarray(
            features, bits=4, stats={"clip_min": 0.0, "clip_max": 15.0}
        )

        np.testing.assert_array_equal(actual, np.array([[15]], dtype=np.uint8))

    def test_quantize_ndarray_four_bit_packs_full_byte_descending_codes(self) -> None:
        # With clip range [0, 15], these values equal their 4-bit codes.
        # [15, 0] packs as 1111 0000 = 240.
        features = np.array([[15.0, 0.0]])

        actual = quantize_ndarray(
            features, bits=4, stats={"clip_min": 0.0, "clip_max": 15.0}
        )

        np.testing.assert_array_equal(actual, np.array([[240]], dtype=np.uint8))

    def test_quantize_ndarray_four_bit_pads_final_byte(self) -> None:
        # The first two codes [0, 15] pack into byte 15.
        # The leftover code [8] starts the next byte as 1000 0000 = 128.
        features = np.array([[0.0, 15.0, 8.0]])

        actual = quantize_ndarray(
            features, bits=4, stats={"clip_min": 0.0, "clip_max": 15.0}
        )

        np.testing.assert_array_equal(actual, np.array([[15, 128]], dtype=np.uint8))

    def test_quantize_ndarray_eight_bit_stores_one_code_per_column(self) -> None:
        # 8-bit quantization has one code per uint8 column, so no bit packing changes the order.
        features = np.array([[0.0, 128.0, 255.0]])

        actual = quantize_ndarray(
            features, bits=8, stats={"clip_min": 0.0, "clip_max": 255.0}
        )

        np.testing.assert_array_equal(actual, np.array([[0, 128, 255]], dtype=np.uint8))

    def test_quantize_ndarray_clips_multi_bit_values_before_packing(self) -> None:
        # Values are clipped before scaling: [-1, 0, 3, 4] over [0, 3]
        # becomes codes [0, 0, 3, 3], which packs as 00 00 11 11 = 15.
        features = np.array([[-1.0, 0.0, 3.0, 4.0]])

        actual = quantize_ndarray(
            features, bits=2, stats={"clip_min": 0.0, "clip_max": 3.0}
        )

        np.testing.assert_array_equal(actual, np.array([[15]], dtype=np.uint8))

    def test_quantize_ndarray_rejects_invalid_bit_width(self) -> None:
        with self.assertRaises(ValueError):
            quantize_ndarray(
                np.zeros((1, 1)), bits=3, stats={"clip_min": 0.0, "clip_max": 1.0}
            )

    def test_quantize_ndarray_rejects_non_2d_features(self) -> None:
        with self.assertRaises(ValueError):
            quantize_ndarray(
                np.zeros((1, 1, 1)), bits=2, stats={"clip_min": 0.0, "clip_max": 1.0}
            )
