"""Multi-threaded kernels for :func:`gigl.utils.csr.build_csr_from_coo`.

Each thread owns a range of rows, scans the whole edge list and writes only its own rows, so there
are no atomics or write collisions. Columns match the torch path exactly; edge ids of duplicate
edges keep input order, which the torch path does not guarantee. Ranges are balanced by edge count
through ``indptr``.

The kernels run on numba's OpenMP threading layer, which is not fork safe: a process that has run a
kernel must not ``fork()`` a child that runs one. GiGL builds datasets in ``mp.spawn`` children.
"""

from typing import Optional, no_type_check

import numba
import numpy as np
import torch
from numba import njit, prange

# @no_type_check keeps the unit launcher's jaxtyping hook from wrapping the kernels: numba's cache
# cannot pickle the wrapper's closure, so every cached call would fail under `make unit_test_py`.


@no_type_check
@njit(cache=True, parallel=True)
def _degrees_kernel(row: np.ndarray, num_rows: int, num_ranges: int) -> np.ndarray:
    # Equal-width row ranges: there is no indptr yet to balance by.
    counts = np.zeros(num_rows, dtype=np.int64)
    per_range = (num_rows + num_ranges - 1) // num_ranges
    for range_index in prange(num_ranges):
        low = range_index * per_range
        high = min(low + per_range, num_rows)
        for edge in range(row.shape[0]):
            row_id = row[edge]
            if low <= row_id < high:
                counts[row_id] += 1
    return counts


@no_type_check
@njit(cache=True, parallel=True)
def _scatter_kernel(
    row: np.ndarray,
    col: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_ids: np.ndarray,
    row_bounds: np.ndarray,
    write_edge_ids: bool,
) -> np.ndarray:
    # Returns edges placed per range, -1 where degrees and edges disagree. The check runs
    # before each write: an overfull row's cursor points into the next range's slice.
    num_ranges = row_bounds.shape[0] - 1
    placed = np.zeros(num_ranges, dtype=np.int64)
    for range_index in prange(num_ranges):
        low = row_bounds[range_index]
        high = row_bounds[range_index + 1]
        cursor = indptr[low:high].copy()
        count = 0
        valid = True
        for edge in range(row.shape[0]):
            row_id = row[edge]
            if valid and low <= row_id < high:
                slot = cursor[row_id - low]
                if slot >= indptr[row_id + 1]:
                    valid = False
                else:
                    indices[slot] = col[edge]
                    if write_edge_ids:
                        edge_ids[slot] = edge
                    cursor[row_id - low] += 1
                    count += 1
        for local_row in range(high - low):
            if cursor[local_row] != indptr[low + local_row + 1]:
                valid = False
        placed[range_index] = count if valid else -1
    return placed


@no_type_check
@njit(cache=True, parallel=True)
def _sort_rows_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_ids: np.ndarray,
    row_bounds: np.ndarray,
    write_edge_ids: bool,
) -> None:
    for range_index in prange(row_bounds.shape[0] - 1):
        for row_id in range(row_bounds[range_index], row_bounds[range_index + 1]):
            start = indptr[row_id]
            end = indptr[row_id + 1]
            if end - start < 2:
                continue
            if write_edge_ids:
                order = np.argsort(indices[start:end], kind="mergesort")
                indices[start:end] = indices[start:end][order]
                edge_ids[start:end] = edge_ids[start:end][order]
            else:
                indices[start:end].sort()


def _edge_balanced_row_bounds(indptr: np.ndarray, num_ranges: int) -> np.ndarray:
    # int64 keys: float keys would make searchsorted copy all of indptr to compare.
    keys = np.arange(num_ranges + 1, dtype=np.int64) * int(indptr[-1]) // num_ranges
    bounds = np.searchsorted(indptr, keys)
    bounds[0] = 0
    bounds[-1] = indptr.shape[0] - 1
    return np.unique(bounds)


def parallel_degrees(row: torch.Tensor, num_rows: int) -> torch.Tensor:
    """``torch.bincount(row, minlength=num_rows)`` across all cores, as int64."""
    return torch.from_numpy(
        _degrees_kernel(row.numpy(), num_rows, numba.get_num_threads())
    )


def parallel_scatter(
    row: torch.Tensor,
    col: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    edge_ids: Optional[torch.Tensor],
) -> None:
    """Place every edge in its row of ``indices`` (and its COO position in ``edge_ids``).

    Args:
        row: Source node id of each edge.
        col: Destination node id of each edge.
        indptr: CSR row pointers built from the degrees of ``row``.
        indices: Output column ids, one slot per edge; written in place.
        edge_ids: Output COO position of each edge, written in place, or ``None`` to skip them.

    Raises:
        ValueError: If the edges do not fill every row exactly.
    """
    indptr_np = indptr.numpy()
    placed = _scatter_kernel(
        row.numpy(),
        col.numpy(),
        indptr_np,
        indices.numpy(),
        np.empty(0, dtype=np.int64) if edge_ids is None else edge_ids.numpy(),
        _edge_balanced_row_bounds(indptr_np, numba.get_num_threads()),
        edge_ids is not None,
    )
    if int(placed.min()) < 0 or int(placed.sum()) != row.numel():
        raise ValueError("CSR scatter did not fill every row exactly; degrees disagree")


def parallel_sort_within_rows(
    indptr: torch.Tensor, indices: torch.Tensor, edge_ids: Optional[torch.Tensor]
) -> None:
    """Sort each row's columns ascending in place, permuting ``edge_ids`` alongside.

    Args:
        indptr: CSR row pointers.
        indices: Column ids to sort within each row; sorted in place.
        edge_ids: Edge ids to permute with ``indices``, or ``None``. A stable sort keeps
            duplicate edges in input order.
    """
    indptr_np = indptr.numpy()
    # More ranges than threads: long rows sort slower, and finer ranges balance them.
    _sort_rows_kernel(
        indptr_np,
        indices.numpy(),
        np.empty(0, dtype=np.int64) if edge_ids is None else edge_ids.numpy(),
        _edge_balanced_row_bounds(indptr_np, numba.get_num_threads() * 8),
        edge_ids is not None,
    )
