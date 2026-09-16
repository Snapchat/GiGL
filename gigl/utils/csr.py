"""Memory-lean COO -> CSR/CSC conversion.

``graphlearn_torch.utils.coo_to_csr`` delegates to ``torch_sparse.SparseTensor``, which builds a
composite sort key, sorts it, and gathers row and col through the permutation while the caller's
originals are still live. Seven full-size int64 arrays exist at the peak, measured at 7.25x one
int64 array on 400M edges. At a multi-billion-edge partition that is hundreds of GiB and the
conversion, not the graph, is what exhausts the host.

This implementation is a two-pass counting sort: degrees give the exact output layout up front,
so the output is allocated once and written in place, and the input is consumed in chunks so
transients are bounded by chunk size rather than by edge count::

    peak = row + col + indices + 2 x (num_rows + 1) + O(chunk + max_degree)

which is 2.0x one int64 array with int32 inputs. The ``max_degree`` term comes from the optional
within-row sort, which cannot split a single row across blocks; it only matters for a graph whose
largest row is a meaningful fraction of its edge count.
"""

import functools
import gc
from typing import Optional, Tuple

import torch

from gigl.common.logger import Logger
from gigl.utils.share_memory import allocate_preshared, is_disk_backed

logger = Logger()

# 16M edges of chunk transients is ~1 GB, small against any partition worth converting.
DEFAULT_CHUNK_SIZE = 1 << 24
# Reordering within a row holds four block-sized int64 arrays (expanded row ids, sort key,
# permutation, gathered result). 32M edges is ~1 GB; larger blocks cost memory for no speed gain.
DEFAULT_SORT_BLOCK_EDGES = 1 << 25
# Destination window for the banded scatter, used only when `indices` could not be placed in
# memory. Small enough that a band's pages stay cached while being written, large enough that the
# number of input passes stays low.
DEFAULT_BAND_BYTES = 4 * 2**30

_INT32_MAX_COLUMN_ID = 2**31 - 1


def _is_int32_dtype_rejection(error: BaseException) -> bool:
    """Whether ``error`` is an unpatched wheel declining an int32 column array.

    Upstream's ``init_cpu_from_csr`` calls ``data_ptr<int64_t>()``, so torch raises ``expected
    scalar type Long but found Int``. Matched on the message because that is the only thing
    separating this expected rejection from a genuinely broken install.
    """
    if not isinstance(error, RuntimeError):
        return False
    message = str(error)
    return "scalar type" in message and "Long" in message and "Int" in message


@functools.lru_cache(maxsize=1)
def glt_accepts_int32_indices() -> bool:
    """Whether the installed ``graphlearn_torch`` can consume an int32 CSR column array.

    Narrowing ``indices`` halves the dominant allocation here, but only a wheel built with
    ``gigl/scripts/patches/0001-glt-csr-col-count-and-int32-indices.patch`` can read it. The width
    is therefore a property of the installed binary, not of the data, and has to be asked: a
    released wheel raises on int32 input, and assuming otherwise would break every caller that
    composes this with ``Graph.lazy_init``.

    The probe samples as well as initializes, because the two halves of the patch fail
    independently and only one is loud: a sampler reading int32 storage as int64 returns plausible
    garbage rather than raising. Accepting int32 at init but sampling it wrongly means the binary
    is in an unknown state, so that case raises instead of falling back -- the int64 path shares
    the same dispatch and would be equally suspect.

    Returns:
        bool: True when an int32 column array can be both initialized and sampled correctly.

    Raises:
        RuntimeError: If the compiled graph accepts int32 columns but samples them incorrectly.
    """
    try:
        from graphlearn_torch import py_graphlearn_torch as pywrap
        from graphlearn_torch.data import Graph, Topology
    except ImportError as error:
        logger.warning(
            f"CSR indices will stay int64: graphlearn_torch is not importable ({error})"
        )
        return False

    # Row 0 -> [1, 2], row 1 -> [0], row 2 -> [1, 2].
    expected_neighbors = [1, 2, 0, 1, 2]
    expected_counts = [2, 1, 2]
    topology = Topology.__new__(Topology)
    topology._layout = "CSR"
    topology._indptr = torch.tensor([0, 2, 3, 5], dtype=torch.int64)
    topology._indices = torch.tensor(expected_neighbors, dtype=torch.int32)
    topology._edge_ids = None
    topology._edge_weights = None
    graph = Graph.__new__(Graph)
    graph.topo = topology
    graph.mode = "CPU"
    graph.device = None
    graph._graph = None
    try:
        graph.lazy_init()
    except RuntimeError as error:
        if not _is_int32_dtype_rejection(error):
            raise
        logger.info(
            f"CSR indices will stay int64: the installed graphlearn_torch does not accept an "
            f"int32 column array ({error}). Expected on the released wheel; in an image built by "
            f"install_glt.sh it means the CSR patch is missing."
        )
        return False

    # Full fanout: req_num above the max degree makes the sampler copy every neighbour, so this
    # is deterministic and comparable element-wise.
    neighbors, counts = pywrap.CPURandomSampler(graph.graph_handler).sample(
        torch.tensor([0, 1, 2], dtype=torch.int64), 8
    )
    if neighbors.tolist() != expected_neighbors or counts.tolist() != expected_counts:
        raise RuntimeError(
            f"The installed graphlearn_torch accepts an int32 CSR column array but samples it "
            f"incorrectly: expected {expected_neighbors} with counts {expected_counts}, got "
            f"{neighbors.tolist()} with {counts.tolist()}. This wheel would return plausible but "
            f"wrong neighbourhoods."
        )
    return True


def _degrees(keys: torch.Tensor, num_rows: int, chunk_size: int) -> torch.Tensor:
    """Count occurrences of each row id, chunked so no full-size int64 copy of ``keys`` exists."""
    degrees = torch.zeros(num_rows, dtype=torch.int64)
    for start in range(0, keys.numel(), chunk_size):
        chunk = keys[start : start + chunk_size].to(torch.int64)
        degrees.index_add_(0, chunk, torch.ones(chunk.numel(), dtype=torch.int64))
        del chunk
    return degrees


def _place_chunk(
    row_chunk: torch.Tensor,
    col_chunk: torch.Tensor,
    cursor: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Write one chunk's columns to their CSR destinations and advance ``cursor``.

    ``row_chunk`` must already be int64 and indexed relative to ``cursor``'s first row.
    """
    order = torch.argsort(row_chunk, stable=True)
    row_chunk = row_chunk[order]
    col_chunk = col_chunk[order]
    del order

    unique_rows, counts = torch.unique_consecutive(row_chunk, return_counts=True)
    del row_chunk
    run_starts = torch.cumsum(counts, dim=0) - counts
    offset_within_run = torch.arange(
        col_chunk.numel(), dtype=torch.int64
    ) - torch.repeat_interleave(run_starts, counts)
    del run_starts
    destination = (
        torch.repeat_interleave(cursor[unique_rows], counts) + offset_within_run
    )
    del offset_within_run
    indices[destination] = col_chunk.to(indices.dtype)
    # unique_rows are distinct, so this is a plain read-modify-write with no collisions.
    cursor[unique_rows] += counts
    del destination, col_chunk, unique_rows, counts


def _scatter_whole(
    row: torch.Tensor,
    col: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    chunk_size: int,
) -> None:
    """One pass over the input, writing each chunk wherever its rows land.

    Correct for any destination and optimal for one in memory. A file-backed destination needs
    :func:`_scatter_in_bands` instead, since the writes here are spread over the whole array.
    """
    num_edges = row.numel()
    # Where the next edge of each row goes. Advanced as chunks are placed, so the scatter needs no
    # global sort and stays order-preserving across chunks.
    cursor = indptr[:-1].clone()
    for start in range(0, num_edges, chunk_size):
        _place_chunk(
            row_chunk=row[start : start + chunk_size].to(torch.int64),
            col_chunk=col[start : start + chunk_size],
            cursor=cursor,
            indices=indices,
        )
    if not bool(torch.equal(cursor, indptr[1:])):
        raise ValueError("CSR scatter did not fill every row exactly; degrees disagree")
    del cursor


def _scatter_in_bands(
    row: torch.Tensor,
    col: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    chunk_size: int,
    band_bytes: int,
) -> None:
    """Fill ``indices`` one contiguous window at a time, for a destination that lives on disk.

    A single pass writes to offsets spread across the whole array, which for a file means every
    page is faulted in, dirtied by a few bytes, written back, and evicted, then faulted again on
    the next chunk -- billions of page touches at a billion-edge shape, which looks like a hang.

    Instead, take a band of rows whose destination slice fits ``band_bytes`` and scan the whole
    input once per band, writing only the edges belonging to it. Writes stay inside a window small
    enough to remain cached, so each page is faulted and written back once. The trade is
    ``num_bands`` cheap sequential passes over the in-memory input for one pass over the expensive
    output. ``cursor`` is per band rather than per row, so this holds a slice of ``indptr`` rather
    than a full clone.
    """
    num_rows = indptr.numel() - 1
    num_edges = row.numel()
    band_elements = max(band_bytes // indices.element_size(), 1)
    placed = 0
    bands = 0
    first_row = 0
    while first_row < num_rows:
        # The first row boundary past band_elements edges, clamped to at least one row so a row
        # with more edges than a band forms an oversized band of its own rather than looping
        # forever -- its destinations are contiguous anyway.
        last_row = int(
            torch.searchsorted(indptr, indptr[first_row] + band_elements).item()
        )
        last_row = min(max(last_row, first_row + 1), num_rows)
        if int(indptr[last_row].item()) > int(indptr[first_row].item()):
            cursor = indptr[first_row:last_row].clone()
            for start in range(0, num_edges, chunk_size):
                row_chunk = row[start : start + chunk_size]
                in_band = (row_chunk >= first_row) & (row_chunk < last_row)
                if not bool(in_band.any()):
                    del row_chunk, in_band
                    continue
                selected_rows = row_chunk[in_band].to(torch.int64) - first_row
                _place_chunk(
                    row_chunk=selected_rows,
                    col_chunk=col[start : start + chunk_size][in_band],
                    cursor=cursor,
                    indices=indices,
                )
                placed += int(selected_rows.numel())
                del row_chunk, in_band, selected_rows
            if not bool(torch.equal(cursor, indptr[first_row + 1 : last_row + 1])):
                raise ValueError(
                    f"CSR banded scatter did not fill rows [{first_row}, {last_row}) exactly; "
                    f"degrees disagree"
                )
            del cursor
        bands += 1
        first_row = last_row
    if placed != num_edges:
        raise ValueError(
            f"CSR banded scatter placed {placed:,} of {num_edges:,} edges; every edge must land "
            f"in exactly one band"
        )
    logger.info(f"CSR banded scatter used {bands} band(s) over the destination")


def _sort_within_rows(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    block_edges: int,
) -> None:
    """Sort each row's column slice ascending, in place.

    Gives parity with ``coo_to_csr``, which sorts on ``row * num_cols + col`` and so leaves
    columns ascending within each row.

    Deliberately does not build that composite key: ``row * num_cols`` overflows int64 once the
    row span times the column count exceeds 2**63, the row span of a block is not bounded by
    ``block_edges`` (a long run of empty rows lets one block cover arbitrarily many rows), and the
    failure is silent -- keys wrap, argsort succeeds, columns come out mis-ordered. Sorting by
    column and then stably by row preserves the column order established by the first pass, with
    no arithmetic on ids at all.

    ``block_edges`` bounds the transient working set except for a single row larger than a block,
    which is sorted whole: transients are then O(degree) for that row. Pass
    ``sort_within_row=False`` to skip this pass entirely if that is not affordable -- GLT's
    samplers do not require columns to be ordered within a row, only ``coo_to_csr`` parity does.
    """
    num_rows = indptr.numel() - 1
    row_start = 0
    while row_start < num_rows:
        row_end = int(
            torch.searchsorted(indptr, indptr[row_start] + block_edges).item()
        )
        row_end = min(max(row_end, row_start + 1), num_rows)
        edge_start = int(indptr[row_start].item())
        edge_end = int(indptr[row_end].item())
        if edge_end > edge_start:
            counts = indptr[row_start + 1 : row_end + 1] - indptr[row_start:row_end]
            rows_expanded = torch.repeat_interleave(
                torch.arange(row_end - row_start, dtype=torch.int64), counts
            )
            del counts
            by_column = torch.argsort(indices[edge_start:edge_end])
            columns = indices[edge_start:edge_end][by_column]
            rows_expanded = rows_expanded[by_column]
            del by_column
            by_row = torch.argsort(rows_expanded, stable=True)
            del rows_expanded
            indices[edge_start:edge_end] = columns[by_row]
            del columns, by_row
        row_start = row_end


def _indices_dtype(
    col: torch.Tensor, num_cols: int, num_edges: int, chunk_size: int
) -> torch.dtype:
    """The narrowest dtype the column array may use.

    Two independent conditions, both asked rather than assumed: whether the installed compiled
    graph can read int32 columns at all, and whether the values fit. An int32 input fits by
    construction; an int64 input is narrowed only after a chunked min/max confirms every value
    fits, so a wrong ``num_cols`` makes the output wider, never truncated. Both bounds are
    checked, since a column id below -2**31 wraps exactly as silently as one above 2**31-1.
    """
    if not glt_accepts_int32_indices():
        return torch.int64
    if col.dtype == torch.int32:
        return torch.int32
    if num_cols - 1 > _INT32_MAX_COLUMN_ID:
        return torch.int64
    observed_max = -1
    observed_min = 0
    for start in range(0, num_edges, chunk_size):
        chunk = col[start : start + chunk_size]
        observed_max = max(observed_max, int(chunk.max().item()))
        observed_min = min(observed_min, int(chunk.min().item()))
    fits = observed_max <= _INT32_MAX_COLUMN_ID and observed_min >= -(
        _INT32_MAX_COLUMN_ID + 1
    )
    return torch.int32 if fits else torch.int64


def build_csr_from_coo(
    row: torch.Tensor,
    col: torch.Tensor,
    num_rows: int,
    num_cols: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    sort_within_row: bool = True,
    sort_block_edges: int = DEFAULT_SORT_BLOCK_EDGES,
    band_bytes: int = DEFAULT_BAND_BYTES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a COO edge index to CSR without ever holding more than one int64 copy.

    Drop-in replacement for the ``(rowptr, col)`` half of
    ``graphlearn_torch.utils.coo_to_csr``, which is 7.25x more expensive at the peak.

    Edge ids and edge weights are unsupported: permuting them is a large part of what makes the
    upstream version expensive, and GiGL already declines to materialize edge ids when no edge
    type carries features.

    To build CSC instead, pass the column indices as ``row`` and the row indices as ``col``.

    Args:
        row (torch.Tensor): 1-D row indices, int32 or int64.
        col (torch.Tensor): 1-D column indices, same length as ``row``.
        num_rows (int): Size of the row dimension. ``indptr`` has ``num_rows + 1`` entries.
        num_cols (Optional[int]): Column-id domain bound, defaulting to ``num_rows``. Gates the
            int32 narrowing of ``indices``; the within-row sort needs no column bound.
        chunk_size (int): Edges per scatter chunk. Bounds the transient working set.
        sort_within_row (bool): Sort each row's columns ascending, matching ``coo_to_csr``. Costs
            one extra bounded-memory pass over the output.
        sort_block_edges (int): Edges per block of the within-row sort.
        band_bytes (int): Destination window for the banded scatter, used only when ``indices``
            could not be placed in memory. Ignored otherwise.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``indptr`` of shape ``[num_rows + 1]``, always int64,
        and ``indices`` of shape ``[num_edges]``, int32 when the installed graphlearn_torch
        accepts it and every column id fits, int64 otherwise.

    Raises:
        ValueError: If ``row`` and ``col`` disagree in length, a row id is out of range, or the
            scatter does not exactly fill every row.
    """
    if row.dim() != 1 or col.dim() != 1:
        raise ValueError(f"Expected 1-D row and col, got {row.shape} and {col.shape}")
    if row.numel() != col.numel():
        raise ValueError(
            f"row and col must be the same length, got {row.numel()} and {col.numel()}"
        )
    if num_cols is None:
        num_cols = num_rows
    num_edges = row.numel()
    indices_dtype = _indices_dtype(col, num_cols, num_edges, chunk_size)

    # Allocated so that a later Topology.share_memory_() cannot duplicate it -- see
    # allocate_preshared. Uninitialised rather than zeroed: element 0 is set below and [1:] is
    # entirely overwritten by the cumsum.
    indptr = allocate_preshared((num_rows + 1,), torch.int64)
    indptr[0] = 0
    if num_edges == 0:
        # The one path where the cumsum never runs, so [1:] would stay uninitialised. An all-zero
        # indptr is the correct CSR for a graph with no edges, and a rank owning no edges of some
        # edge type is normal under range partitioning.
        indptr.zero_()
        return indptr, torch.empty(0, dtype=indices_dtype)

    row_max = int(row.max().item())
    if row_max >= num_rows:
        raise ValueError(f"Row id {row_max} is out of range for num_rows={num_rows}")

    degrees = _degrees(row, num_rows, chunk_size)
    torch.cumsum(degrees, dim=0, out=indptr[1:])
    del degrees
    gc.collect()

    # random_access=True: the direct scatter writes to offsets spread across the whole array, so
    # memory is preferred. The banded path below is what runs if it lands on disk anyway.
    indices = allocate_preshared((num_edges,), indices_dtype, random_access=True)
    if is_disk_backed(indices):
        _scatter_in_bands(
            row=row,
            col=col,
            indptr=indptr,
            indices=indices,
            chunk_size=chunk_size,
            band_bytes=band_bytes,
        )
    else:
        _scatter_whole(
            row=row, col=col, indptr=indptr, indices=indices, chunk_size=chunk_size
        )
    gc.collect()

    if sort_within_row:
        _sort_within_rows(indptr, indices, sort_block_edges)

    logger.info(
        f"Built CSR for {num_edges:,} edges over {num_rows:,} rows "
        f"(indices {indices.numel() * indices.element_size() / 2**30:.1f} GiB as "
        f"{indices.dtype}, indptr {indptr.numel() * indptr.element_size() / 2**30:.1f} GiB)"
    )
    return indptr, indices
