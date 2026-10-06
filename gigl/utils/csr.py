"""Memory-lean COO -> CSR/CSC conversion.

``graphlearn_torch.utils.coo_to_csr`` delegates to ``torch_sparse.SparseTensor``, which builds a
composite sort key, sorts it, and gathers row and col through the permutation while the caller's
originals are still live. Seven full-size int64 arrays exist at the peak, measured at 7.25x one
int64 array on 400M edges. At a multi-billion-edge partition that is hundreds of GiB and the
conversion, not the graph, is what exhausts the host.

This implementation is a two-pass counting sort: degrees give the exact output layout up front,
so the output is allocated once and written in place, and the input is consumed in chunks so
transients are bounded by chunk size rather than by edge count::

    peak = row + col + indices + optional edge_ids + 2 x (num_rows + 1)
           + O(chunk + max_degree)

which is 2.0x one int64 array with int32 inputs when edge IDs are omitted, or 3.0x
when they are retained. The
``max_degree`` term comes from within-row sorting; it matters when one row owns a meaningful
fraction of the edges.

:class:`CompactTopology` wraps the result as a GLT ``Topology``.
"""

import gc
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from graphlearn_torch.data import Topology

from gigl.common.logger import Logger
from gigl.utils.share_memory import allocate_preshared, is_disk_backed

logger = Logger()

# 16M edges of chunk transients is ~1 GB, small against any partition worth converting.
DEFAULT_CHUNK_SIZE = 1 << 24
# Reordering within a row holds four block-sized int64 arrays (expanded row ids, sort key,
# permutation, gathered result). 32M edges is ~1 GB; larger blocks cost memory for no speed gain.
DEFAULT_SORT_BLOCK_EDGES = 1 << 25
# Destination window for the scatter when `indices` could not be placed in memory. Small enough that a band's pages stay cached while being written, large enough that the
# number of input passes stays low.
DEFAULT_BAND_BYTES = 4 * 2**30


@dataclass(frozen=True)
class CsrBuildResult:
    """CSR arrays and optional COO-position edge IDs in CSR order."""

    indptr: torch.Tensor
    indices: torch.Tensor
    edge_ids: Optional[torch.Tensor] = None


def _chunk_destinations(
    row_chunk: torch.Tensor,
    cursor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find each edge's CSR slot and advance ``cursor`` past the chunk.

    Args:
        row_chunk: ``[C]`` int64 row id of each edge, relative to ``cursor``'s first row.
        cursor: ``[R]`` the CSR slot where each row's next edge goes. Starts as
            ``indptr[:-1]`` and ends as ``indptr[1:]`` once every chunk is placed.

    Returns:
        CSR destinations and the stable row ordering to apply to columns and edge IDs.

    Example: with ``cursor = [0, 2, 3]`` (row 0 owns slots 0-1, row 1 slot 2, row 2 slots 3-4),
    the chunk ``row_chunk = [2, 0, 2]`` returns destinations ``[0, 3, 4]`` with
    order ``[1, 0, 2]``, and leaves ``cursor = [1, 2, 5]``.
    """
    order = torch.argsort(row_chunk, stable=True)
    row_chunk = row_chunk[order]

    # [U] distinct rows in this chunk, and [U] edges for each.
    unique_rows, counts = torch.unique_consecutive(row_chunk, return_counts=True)
    del row_chunk
    run_starts = torch.cumsum(counts, dim=0) - counts
    offset_within_run = torch.arange(
        order.numel(), dtype=torch.int64
    ) - torch.repeat_interleave(run_starts, counts)
    del run_starts
    # [C] CSR slot for each edge: its row's cursor plus its rank within the row.
    destination = (
        torch.repeat_interleave(cursor[unique_rows], counts) + offset_within_run
    )
    del offset_within_run
    # unique_rows are distinct, so this is a plain read-modify-write with no collisions.
    cursor[unique_rows] += counts
    return destination, order


def _scatter_in_bands(
    row: torch.Tensor,
    col: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    chunk_size: int,
    band_bytes: Optional[int],
    edge_ids_out: Optional[torch.Tensor] = None,
) -> None:
    """Fill ``indices`` one band of rows at a time, scanning the whole input once per band.

    ``band_bytes=None`` is a single band, right for an in-memory destination. On disk a single pass
    writes all over the file, so every page is faulted in, dirtied by a few bytes, written back and
    evicted, then faulted again on the next chunk -- at a billion-edge shape that looks like a
    hang. Bands whose destination slice fits ``band_bytes`` keep the writes inside a window that
    stays cached, trading cheap passes over the in-memory input for one pass over the output.
    """
    num_rows = indptr.numel() - 1
    num_edges = row.numel()
    bytes_per_edge = indices.element_size()
    if edge_ids_out is not None:
        bytes_per_edge += edge_ids_out.element_size()
    band_elements = num_edges
    if band_bytes is not None:
        band_elements = max(band_bytes // bytes_per_edge, 1)
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
                destinations, order = _chunk_destinations(
                    row_chunk=selected_rows,
                    cursor=cursor,
                )
                indices[destinations] = col[start : start + chunk_size][in_band][
                    order
                ].to(indices.dtype)
                if edge_ids_out is not None:
                    edge_ids_out[destinations] = (
                        torch.nonzero(in_band, as_tuple=True)[0][order] + start
                    )
                placed += int(selected_rows.numel())
                del row_chunk, in_band, selected_rows, destinations, order
            if not bool(torch.equal(cursor, indptr[first_row + 1 : last_row + 1])):
                raise ValueError(
                    f"CSR scatter did not fill rows [{first_row}, {last_row}) exactly; "
                    f"degrees disagree"
                )
            del cursor
        bands += 1
        first_row = last_row
    if placed != num_edges:
        raise ValueError(
            f"CSR scatter placed {placed:,} of {num_edges:,} edges; every edge must land in "
            f"exactly one band"
        )
    if band_bytes is not None:
        logger.info(f"CSR scatter used {bands} bands over the file-backed destination")


def _sort_within_rows(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    block_edges: int,
    edge_ids_out: Optional[torch.Tensor] = None,
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

    ``block_edges`` caps both the edges and the rows in a block, so transients are bounded except
    for a single row larger than a block, which is sorted whole: O(degree) for that row.
    """
    num_rows = indptr.numel() - 1
    row_start = 0
    while row_start < num_rows:
        row_end = int(
            torch.searchsorted(indptr, indptr[row_start] + block_edges).item()
        )
        # Capping rows too keeps `counts` and `rows_expanded` bounded across long runs of
        # empty rows, which the edge budget alone does not.
        row_end = min(max(row_end, row_start + 1), row_start + block_edges, num_rows)
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
            if edge_ids_out is not None:
                sorted_edge_ids = edge_ids_out[edge_start:edge_end][by_column]
            rows_expanded = rows_expanded[by_column]
            del by_column
            by_row = torch.argsort(rows_expanded, stable=True)
            del rows_expanded
            indices[edge_start:edge_end] = columns[by_row]
            if edge_ids_out is not None:
                edge_ids_out[edge_start:edge_end] = sorted_edge_ids[by_row]
                del sorted_edge_ids
            del columns, by_row
        row_start = row_end


def build_csr_from_coo(
    row: torch.Tensor,
    col: torch.Tensor,
    num_rows: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    sort_block_edges: int = DEFAULT_SORT_BLOCK_EDGES,
    band_bytes: int = DEFAULT_BAND_BYTES,
    retain_edge_ids: bool = True,
) -> CsrBuildResult:
    """Convert a COO edge index to CSR with bounded chunk and sort temporaries.

    Produces the same CSR arrays as ``graphlearn_torch.utils.coo_to_csr`` without its
    full-size sorting temporaries.

    When ``retain_edge_ids`` is true, write original COO positions in CSR order alongside
    columns. The final int64 vector is allocated directly; no full-size input ID vector is needed.
    Explicit edge IDs and edge weights are unsupported.

    To build CSC instead, pass the column indices as ``row`` and the row indices as ``col``.

    Args:
        row (torch.Tensor): 1-D row indices, int32 or int64.
        col (torch.Tensor): 1-D column indices, same length as ``row``.
        num_rows (Optional[int]): Size of the row dimension; ``indptr`` has ``num_rows + 1``
            entries. Defaults to ``max(row) + 1``, as ``coo_to_csr`` does.
        chunk_size (int): Edges per scatter chunk. Bounds the transient working set.
        sort_block_edges (int): Edges per block of the within-row sort, which orders each row's
            columns ascending as ``coo_to_csr`` does.
        band_bytes (int): Destination window for the scatter, used only when ``indices`` could
            not be placed in memory.
        retain_edge_ids (bool): Retain original COO positions in CSR order by default.
            The int64 IDs cost 8 bytes per edge. Set to ``False`` when callers do not
            sample edge IDs.

    Returns:
        CsrBuildResult: ``indptr`` of shape ``[num_rows + 1]`` and ``indices`` of shape
        ``[num_edges]``, both int64, plus optional int64 ``edge_ids`` of shape
        ``[num_edges]`` in CSR order. Empty input returns an empty ID tensor when
        retention is enabled and ``None`` otherwise.

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
    num_edges = row.numel()
    row_max = -1
    if num_edges:
        row_max = int(row.max().item())
    if num_rows is None:
        num_rows = row_max + 1
    elif row_max >= num_rows:
        raise ValueError(f"Row id {row_max} is out of range for num_rows={num_rows}")

    # Allocated so that a later Topology.share_memory_() cannot duplicate it -- see
    # allocate_preshared.
    indptr = allocate_preshared((num_rows + 1,), torch.int64)
    if num_edges == 0:
        # A rank can own no edges of some edge type.
        indptr.zero_()
        edge_ids = None
        if retain_edge_ids:
            edge_ids = torch.empty(0, dtype=torch.int64)
        return CsrBuildResult(
            indptr=indptr,
            indices=torch.empty(0, dtype=torch.int64),
            edge_ids=edge_ids,
        )
    indptr[0] = 0
    torch.cumsum(torch.bincount(row, minlength=num_rows), dim=0, out=indptr[1:])
    gc.collect()

    # random_access=True: the scatter writes all over the array, so memory is preferred; bands
    # take over if it lands on disk anyway.
    indices = allocate_preshared((num_edges,), torch.int64, random_access=True)
    edge_ids = None
    if retain_edge_ids:
        edge_ids = allocate_preshared((num_edges,), torch.int64, random_access=True)
    scatter_band_bytes = None
    if is_disk_backed(indices) or (edge_ids is not None and is_disk_backed(edge_ids)):
        scatter_band_bytes = band_bytes
    _scatter_in_bands(
        row=row,
        col=col,
        indptr=indptr,
        indices=indices,
        chunk_size=chunk_size,
        band_bytes=scatter_band_bytes,
        edge_ids_out=edge_ids,
    )
    gc.collect()
    _sort_within_rows(indptr, indices, sort_block_edges, edge_ids)

    logger.info(
        f"Built CSR for {num_edges:,} edges over {num_rows:,} rows "
        f"(indices {indices.numel() * indices.element_size() / 2**30:.1f} GiB as "
        f"{indices.dtype}, indptr {indptr.numel() * indptr.element_size() / 2**30:.1f} GiB)"
    )
    return CsrBuildResult(indptr=indptr, indices=indices, edge_ids=edge_ids)


class CompactTopology(Topology):
    """A GLT ``Topology`` built by :func:`build_csr_from_coo`.

    ``Topology.__init__`` is not called: it would fabricate a full ``torch.arange(num_edges)``
    and convert with ``coo_to_csr``. Instead, the counting sort writes original COO positions
    directly into final CSR order when retained. Edge features need explicit IDs and cannot use
    this class.

    Args:
        edge_index (torch.Tensor): ``[2, num_edges]`` COO. int32 and int64 are used as they are;
            anything else is cast to int64, as GLT's ``Topology`` does.
        layout (Literal["CSR", "CSC"]): The layout GLT samples from.
        num_rows (Optional[int]): Size of the compressed dimension. Defaults to the largest
            compressed id + 1, as GLT's ``Topology`` does.
        retain_edge_ids (bool): Retain COO-position IDs for edge sampling by default,
            at a cost of 8 bytes per edge. Use ``False`` only when callers sample
            without edge IDs.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        layout: Literal["CSR", "CSC"],
        num_rows: Optional[int] = None,
        *,
        retain_edge_ids: bool = True,
    ) -> None:
        # GLT's Topology casts any input to int64; integer inputs keep their width here.
        if edge_index.is_floating_point():
            edge_index = edge_index.to(torch.int64)
        # CSC compresses destinations, so it is the CSR of the reversed edges.
        if layout == "CSR":
            compressed, other = 0, 1
        else:
            compressed, other = 1, 0
        self._layout = layout
        csr = build_csr_from_coo(
            row=edge_index[compressed],
            col=edge_index[other],
            num_rows=num_rows,
            retain_edge_ids=retain_edge_ids,
        )
        self._indptr, self._indices, self._edge_ids = (
            csr.indptr,
            csr.indices,
            csr.edge_ids,
        )
        self._edge_weights = None
