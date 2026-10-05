import numpy as np
import pyarrow as pa
from parameterized import param, parameterized

from gigl.src.data_preprocessor.lib.transform.feature_quantization import (
    _build_feature_matrix,
)
from tests.test_assets.test_case import TestCase


class BuildFeatureMatrixTest(TestCase):
    @parameterized.expand(
        [
            param("list", list_type=pa.list_(pa.float32())),
            param("large_list", list_type=pa.large_list(pa.float32())),
            param("fixed_size_list_1", list_type=pa.list_(pa.float32(), 1)),
        ]
    )
    def test_unwraps_one_value_list_columns(
        self, _: str, list_type: pa.DataType
    ) -> None:
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([[1.5], [-2.0], [3.25]], type=list_type),
                pa.array([4.0, 5.0, 6.0], type=pa.float32()),
            ],
            names=["a", "b"],
        )

        feature_matrix = _build_feature_matrix(batch, ["b", "a"])

        np.testing.assert_array_equal(
            feature_matrix,
            np.array([[4.0, 1.5], [5.0, -2.0], [6.0, 3.25]], dtype=np.float32),
        )

    @parameterized.expand(
        [
            param(
                "multi_value_row",
                column=pa.array([[1.0], [2.0, 3.0]], type=pa.list_(pa.float32())),
            ),
            param(
                "empty_row",
                column=pa.array([[1.0], []], type=pa.list_(pa.float32())),
            ),
            param(
                "fixed_size_list_2",
                column=pa.array(
                    [[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32(), 2)
                ),
            ),
        ]
    )
    def test_rejects_rows_without_exactly_one_value(
        self, _: str, column: pa.Array
    ) -> None:
        batch = pa.RecordBatch.from_arrays([column], names=["a"])

        with self.assertRaisesRegex(ValueError, "exactly one value"):
            _build_feature_matrix(batch, ["a"])

    @parameterized.expand(
        [
            param("list", list_type=pa.list_(pa.float32())),
            param("fixed_size_list_1", list_type=pa.list_(pa.float32(), 1)),
        ]
    )
    def test_rejects_null_rows(self, _: str, list_type: pa.DataType) -> None:
        batch = pa.RecordBatch.from_arrays(
            [pa.array([[1.0], None], type=list_type)], names=["a"]
        )

        with self.assertRaisesRegex(ValueError, "1 null rows"):
            _build_feature_matrix(batch, ["a"])
