"""Multi-threaded kernels for :func:`gigl.utils.csr.build_csr_from_coo`.

Each thread owns a range of rows, scans the whole edge list and writes only its own rows, so there
are no atomics or write collisions. Columns match the torch path exactly; edge ids of duplicate
edges keep input order, which the torch path does not guarantee. Ranges are balanced by edge count
through ``indptr``.
"""

from functools import cache
from typing import Optional

import numba
import numpy as np
import torch
from numba import njit, prange


@njit(cache=True, parallel=True)
def _degrees_kernel(row, num_rows, num_ranges):
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


@njit(cache=True, parallel=True)
def _scatter_kernel(row, col, indptr, indices, edge_ids, row_bounds, write_edge_ids):
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


@njit(cache=True, parallel=True)
def _sort_rows_kernel(indptr, indices, edge_ids, row_bounds, write_edge_ids):
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


@cache
def _no_edge_ids() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def _edge_balanced_row_bounds(indptr: np.ndarray, num_ranges: int) -> np.ndarray:
    bounds = np.searchsorted(indptr, np.linspace(0, indptr[-1], num_ranges + 1))
    bounds[0] = 0
    bounds[-1] = indptr.shape[0] - 1
    return np.unique(bounds.astype(np.int64))


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

    Raises:
        ValueError: If the edges do not fill every row exactly.
    """
    indptr_np = indptr.numpy()
    placed = _scatter_kernel(
        row.numpy(),
        col.numpy(),
        indptr_np,
        indices.numpy(),
        _no_edge_ids() if edge_ids is None else edge_ids.numpy(),
        _edge_balanced_row_bounds(indptr_np, numba.get_num_threads()),
        edge_ids is not None,
    )
    if int(placed.min()) < 0 or int(placed.sum()) != row.numel():
        raise ValueError("CSR scatter did not fill every row exactly; degrees disagree")


def parallel_sort_within_rows(
    indptr: torch.Tensor, indices: torch.Tensor, edge_ids: Optional[torch.Tensor]
) -> None:
    """Sort each row's columns ascending in place, permuting ``edge_ids`` alongside."""
    indptr_np = indptr.numpy()
    # More ranges than threads: long rows sort slower, and finer ranges balance them.
    _sort_rows_kernel(
        indptr_np,
        indices.numpy(),
        _no_edge_ids() if edge_ids is None else edge_ids.numpy(),
        _edge_balanced_row_bounds(indptr_np, numba.get_num_threads() * 8),
        edge_ids is not None,
    )
