"""Helpers for turning Arrow columns into numpy arrays without copying."""

import numpy as np
import pyarrow as pa


def list_column_to_numpy(column: pa.Array, num_rows: int) -> np.ndarray:
    """Returns a column holding one list per row as a ``[num_rows, width]`` numpy array.

    TFT emits every output as a list column with one fixed-length list per row. The values are returned as a view of
    the Arrow buffers whenever the value type allows it. A plain primitive column is returned as a width-1 matrix.

    The values are checked in total only: the number of values must divide evenly by ``num_rows``, but each row's
    length is not checked. This is enough for TFT dense outputs, whose shapes are fixed. Checking every row costs about
    as much as the conversion itself for batches with hundreds of scalar columns.

    Example:

        >>> column = pa.array([[1, 2], [3, 4], [5, 6]])
        >>> list_column_to_numpy(column.slice(1, 2), num_rows=2)
        array([[3, 4],
               [5, 6]])

    Args:
        column: A ``list``, ``large_list``, ``fixed_size_list`` or primitive array. May be a slice of a larger array.
        num_rows: Number of rows in the column.

    Returns:
        A ``[num_rows, width]`` array in the column's value type.

    Raises:
        ValueError: If the column length differs from ``num_rows``, the column or its values contain nulls, or the
            number of values is not a multiple of ``num_rows``.
    """
    if len(column) != num_rows:
        raise ValueError(f"Expected a column of {num_rows} rows, got {len(column)}.")
    if column.null_count:
        raise ValueError(f"Expected no null rows, got {column.null_count}.")

    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        # ``column.flatten()`` goes through pyarrow.compute for every column, which dominates conversion time. The
        # offsets are read directly instead. A list array's buffers are [validity bitmap, offsets, ...]: row ``i``
        # holds ``values[offsets[i]:offsets[i + 1]]``. ``column.offset`` is non-zero when the column is a slice of a
        # larger array, so the first row's offset is at ``offsets[column.offset]``.
        if pa.types.is_list(column.type):
            offsets_dtype = np.int32
        else:
            offsets_dtype = np.int64
        offsets = np.frombuffer(column.buffers()[1], dtype=offsets_dtype)
        start = int(offsets[column.offset])
        end = int(offsets[column.offset + num_rows])
        values = column.values.slice(start, end - start)
    elif pa.types.is_fixed_size_list(column.type):
        size = column.type.list_size
        values = column.values.slice(column.offset * size, num_rows * size)
    else:
        values = column

    if values.null_count:
        raise ValueError(f"Expected no null values, got {values.null_count}.")
    if num_rows == 0:
        return np.empty((0, 1), dtype=values.to_numpy(zero_copy_only=False).dtype)
    if len(values) % num_rows != 0:
        raise ValueError(
            f"Expected the same number of values in every row, got {len(values)} values for {num_rows} rows."
        )
    return values.to_numpy(zero_copy_only=False).reshape(num_rows, -1)
