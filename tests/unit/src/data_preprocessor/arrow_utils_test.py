import numpy as np
import pyarrow as pa
from parameterized import param, parameterized

from gigl.src.data_preprocessor.lib.transform.arrow_utils import list_column_to_numpy
from tests.test_assets.test_case import TestCase


class ListColumnToNumpyTest(TestCase):
    @parameterized.expand(
        [
            param("list", list_type=pa.list_(pa.int64())),
            param("large_list", list_type=pa.large_list(pa.int64())),
            param("fixed_size_list", list_type=pa.list_(pa.int64(), 2)),
        ]
    )
    def test_sliced_column_returns_only_the_sliced_rows(
        self, _: str, list_type: pa.DataType
    ) -> None:
        column = pa.array([[0, 1], [2, 3], [4, 5], [6, 7]], type=list_type)

        values = list_column_to_numpy(column.slice(1, 2), num_rows=2)

        np.testing.assert_array_equal(values, [[2, 3], [4, 5]])

    def test_primitive_column_is_a_single_column_matrix(self) -> None:
        column = pa.array([1.0, 2.0, 3.0], type=pa.float32())

        values = list_column_to_numpy(column.slice(1, 2), num_rows=2)

        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_array_equal(values, [[2.0], [3.0]])

    def test_rejects_nulls(self) -> None:
        column = pa.array([[1], None], type=pa.list_(pa.int64()))

        with self.assertRaises(ValueError):
            list_column_to_numpy(column, num_rows=2)

    def test_rejects_value_count_not_divisible_by_rows(self) -> None:
        column = pa.array([[1], [2, 3]], type=pa.list_(pa.int64()))

        with self.assertRaises(ValueError):
            list_column_to_numpy(column, num_rows=2)
