import os
import tempfile
from typing import Literal
from unittest import mock

import torch
from graphlearn_torch.utils import coo_to_csr
from parameterized import param, parameterized

from gigl.utils.csr import (
    CompactTopology,
    _first_row_at_or_past,
    _scatter_in_bands,
    _sort_within_rows,
    build_csr_from_coo,
    select_indptr_dtype,
)
from gigl.utils.csr_parallel import _edge_balanced_row_bounds, parallel_scatter
from gigl.utils.share_memory import allocate_preshared, is_disk_backed
from tests.test_assets.test_case import TestCase


def _reference_csr(
    row: torch.Tensor, col: torch.Tensor, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (indptr, indices) that graphlearn_torch would produce for the same input.

    ``coo_to_csr`` returns an int64 indptr; it is cast to the dtype ``build_csr_from_coo`` picks so the
    values can be compared exactly.
    """
    indptr, indices, _, _ = coo_to_csr(
        row.to(torch.int64), col.to(torch.int64), node_sizes=(num_rows, num_rows)
    )
    return indptr.to(select_indptr_dtype(row.numel())), indices


class BuildCsrFromCooTest(TestCase):
    """``build_csr_from_coo`` replaces ``graphlearn_torch.utils.coo_to_csr``.

    Equivalence is the whole contract: the output is handed to the compiled
    ``Graph.init_cpu_from_csr``, which segfaults rather than raises on malformed input, so these
    compare against the upstream implementation directly instead of asserting hand-written
    expectations.

    """

    @parameterized.expand(
        [
            param("dense_random_int64", dtype=torch.int64, chunk_size=1 << 24),
            param("dense_random_int32", dtype=torch.int32, chunk_size=1 << 24),
            # A chunk size coprime with the row count splits rows across chunk boundaries, which
            # is what the per-row cursor exists to handle.
            param("chunk_splits_rows_int64", dtype=torch.int64, chunk_size=997),
            param("chunk_splits_rows_int32", dtype=torch.int32, chunk_size=997),
        ]
    )
    def test_matches_upstream(
        self, _name: str, dtype: torch.dtype, chunk_size: int
    ) -> None:
        generator = torch.Generator().manual_seed(7)
        num_rows = 1000
        row = torch.randint(0, num_rows, (50_000,), generator=generator).to(dtype)
        col = torch.randint(0, num_rows, (50_000,), generator=generator).to(dtype)

        expected_indptr, expected_indices = _reference_csr(row, col, num_rows)
        result = build_csr_from_coo(
            row, col, num_rows=num_rows, chunk_size=chunk_size, retain_edge_ids=True
        )

        torch.testing.assert_close(result.indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(
            result.indices.long(), expected_indices, rtol=0, atol=0
        )
        self.assertEqual(result.indptr.dtype, torch.int32)
        self.assertEqual(result.indices.dtype, dtype)
        edge_ids = result.edge_ids
        assert edge_ids is not None
        self.assertEqual(edge_ids.dtype, torch.int64)
        self.assertEqual(edge_ids.numel(), row.numel())

    def test_edge_ids_are_original_coo_positions_in_csr_order(self) -> None:
        row = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
        col = torch.tensor([3, 5, 1, 2], dtype=torch.int64)

        result = build_csr_from_coo(row, col, num_rows=2, retain_edge_ids=True)

        self.assertIsNotNone(result.edge_ids)
        torch.testing.assert_close(
            result.indptr, torch.tensor([0, 2, 4], dtype=torch.int32)
        )
        torch.testing.assert_close(result.indices, torch.tensor([2, 5, 1, 3]))
        torch.testing.assert_close(result.edge_ids, torch.tensor([3, 1, 2, 0]))

        omitted = build_csr_from_coo(row, col, num_rows=2)
        self.assertIsNone(omitted.edge_ids)
        torch.testing.assert_close(omitted.indptr, result.indptr)
        torch.testing.assert_close(omitted.indices, result.indices)

    def test_matches_upstream_with_mostly_empty_rows(self) -> None:
        generator = torch.Generator().manual_seed(11)
        num_rows = 500_000
        row = torch.randint(0, 5, (200,), generator=generator) * 100_000
        col = torch.randint(0, num_rows, (200,), generator=generator)

        expected_indptr, expected_indices = _reference_csr(row, col, num_rows)
        result = build_csr_from_coo(row, col, num_rows=num_rows)

        torch.testing.assert_close(result.indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(result.indices, expected_indices, rtol=0, atol=0)

    def test_matches_upstream_with_row_larger_than_sort_block(self) -> None:
        """A single row exceeding ``sort_block_edges`` must still terminate and be correct.

        The within-row sort walks blocks with ``searchsorted``, which returns the same boundary
        for a row that alone exceeds the block budget; the loop clamps forward by one row so it
        cannot stall.
        """
        generator = torch.Generator().manual_seed(13)
        num_rows = 50
        row = torch.zeros(20_000, dtype=torch.int64)
        row[10_000:] = torch.randint(1, num_rows, (10_000,), generator=generator)
        col = torch.randint(0, num_rows, (20_000,), generator=generator)

        expected_indptr, expected_indices = _reference_csr(row, col, num_rows)
        result = build_csr_from_coo(
            row, col, num_rows=num_rows, chunk_size=512, sort_block_edges=64
        )

        torch.testing.assert_close(result.indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(result.indices, expected_indices, rtol=0, atol=0)

    def test_matches_upstream_when_sort_blocks_are_capped_by_rows(self) -> None:
        """Blocks crossing long runs of empty rows are cut by row count as well as by edges."""
        generator = torch.Generator().manual_seed(17)
        num_rows = 2_000
        row = torch.randint(0, 20, (400,), generator=generator) * 100
        col = torch.randint(0, num_rows, (400,), generator=generator)

        expected_indptr, expected_indices = _reference_csr(row, col, num_rows)
        result = build_csr_from_coo(row, col, num_rows=num_rows, sort_block_edges=7)

        torch.testing.assert_close(result.indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(result.indices, expected_indices, rtol=0, atol=0)

    def test_matches_upstream_with_duplicate_edges(self) -> None:
        generator = torch.Generator().manual_seed(17)
        row = torch.randint(0, 4, (10_000,), generator=generator)
        col = torch.randint(0, 3, (10_000,), generator=generator)

        expected_indptr, expected_indices = _reference_csr(row, col, 4)
        result = build_csr_from_coo(
            row, col, num_rows=4, chunk_size=333, retain_edge_ids=True
        )

        torch.testing.assert_close(result.indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(result.indices, expected_indices, rtol=0, atol=0)
        # Duplicates may take any order, but every edge id must name its own edge.
        edge_ids = result.edge_ids
        assert edge_ids is not None
        self.assertEqual(sorted(edge_ids.tolist()), list(range(row.numel())))
        csr_rows = torch.repeat_interleave(torch.arange(4), result.indptr.diff())
        torch.testing.assert_close(row[edge_ids], csr_rows, rtol=0, atol=0)
        torch.testing.assert_close(col[edge_ids], result.indices, rtol=0, atol=0)

    def test_trailing_rows_beyond_max_row_id_are_empty(self) -> None:
        generator = torch.Generator().manual_seed(19)
        row = torch.randint(0, 100, (5_000,), generator=generator)
        col = torch.randint(0, 100, (5_000,), generator=generator)

        result = build_csr_from_coo(row, col, num_rows=4096)

        self.assertEqual(result.indptr.numel(), 4097)
        self.assertEqual(int(result.indptr[-1]), 5_000)
        # Every row above the largest observed row id has zero degree.
        self.assertEqual(int((result.indptr[101:] - result.indptr[100:-1]).sum()), 0)
        self.assertEqual(result.indices.numel(), 5_000)

    def test_columns_ascend_within_every_row_across_huge_id_ranges(self) -> None:
        """Columns must ascend within each row even when one sort block spans many rows.

        The within-row sort used to build the key ``row * num_cols + col``, which overflows int64
        once the block's row span times the column count exceeds 2**63 -- and a block's row span
        is NOT bounded by ``sort_block_edges``, because a long run of empty rows lets one block
        cover arbitrarily many rows. That failure was silent: keys wrap, argsort still succeeds,
        and columns come out mis-ordered.

        The implementation no longer does arithmetic on ids (it sorts by column, then stably by
        row), so the failure class is gone by construction rather than by a bound check. A graph
        big enough to have overflowed cannot be allocated in a unit test, so this asserts the
        ordering property directly on a graph whose ids are spread over a wide range and whose
        empty runs force a single block to span the whole row space.
        """
        generator = torch.Generator().manual_seed(29)
        num_rows = 2_000_000
        occupied = torch.tensor([0, 500_000, 1_500_000, num_rows - 1])
        row = occupied[
            torch.randint(0, occupied.numel(), (4_000,), generator=generator)
        ]
        col = torch.randint(0, num_rows, (4_000,), generator=generator)

        result = build_csr_from_coo(
            row, col, num_rows=num_rows, sort_block_edges=1 << 30
        )

        for row_id in occupied.tolist():
            start, end = int(result.indptr[row_id]), int(result.indptr[row_id + 1])
            self.assertGreater(end, start, f"row {row_id} should have edges")
            slice_ = result.indices[start:end]
            self.assertTrue(
                bool(torch.all(slice_[1:] >= slice_[:-1])),
                f"columns not ascending within row {row_id}",
            )
            # Every column recorded for this row is one that actually appeared with it.
            self.assertEqual(
                sorted(slice_.tolist()), sorted(col[row == row_id].tolist())
            )

    def test_csc_is_csr_of_the_transpose(self) -> None:
        """CSC is requested by swapping the arguments, so it must equal CSR of the transpose."""
        generator = torch.Generator().manual_seed(23)
        num_rows = 800
        row = torch.randint(0, num_rows, (20_000,), generator=generator)
        col = torch.randint(0, num_rows, (20_000,), generator=generator)

        expected_indptr, expected_indices = _reference_csr(col, row, num_rows)
        result = build_csr_from_coo(col, row, num_rows=num_rows)

        torch.testing.assert_close(result.indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(result.indices, expected_indices, rtol=0, atol=0)

    def test_empty_input(self) -> None:
        empty = torch.empty(0, dtype=torch.int64)

        result = build_csr_from_coo(empty, empty, num_rows=10, retain_edge_ids=True)

        self.assertEqual(result.indptr.numel(), 11)
        self.assertEqual(int(result.indptr.sum()), 0)
        self.assertEqual(result.indices.numel(), 0)
        self.assertEqual(result.indices.dtype, torch.int64)
        edge_ids = result.edge_ids
        assert edge_ids is not None
        self.assertEqual(edge_ids.numel(), 0)
        self.assertIsNone(build_csr_from_coo(empty, empty).edge_ids)

    def test_row_id_beyond_num_rows_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_csr_from_coo(torch.tensor([5]), torch.tensor([0]), num_rows=3)

    def test_parallel_scatter_raises_when_degrees_disagree_with_edges(self) -> None:
        # indptr claims row 0 holds one edge, but two edges name row 0.
        with self.assertRaises(ValueError):
            parallel_scatter(
                row=torch.tensor([0, 0]),
                col=torch.tensor([1, 2]),
                indptr=torch.tensor([0, 1, 2]),
                indices=torch.empty(2, dtype=torch.int64),
                edge_ids=None,
            )

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            build_csr_from_coo(torch.tensor([0, 1]), torch.tensor([0]), num_rows=2)

    def test_non_one_dimensional_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_csr_from_coo(
                torch.zeros((2, 4), dtype=torch.int64),
                torch.zeros((2, 4), dtype=torch.int64),
                num_rows=4,
            )


class ScatterPlacementTest(TestCase):
    """Where `indices` lands, and whether several bands produce the same CSR as one.

    Bands exist because a single scatter pass over a file-backed destination took over an hour on a
    31 GiB array, re-dirtying every page on every chunk.
    """

    def setUp(self) -> None:
        self._spill_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._spill_dir.cleanup)

    def _spilling(self, **overrides: str):
        env = {
            "GIGL_TENSOR_SPILL_DIR": self._spill_dir.name,
            "GIGL_TENSOR_SPILL_MIN_BYTES": str(4 * 1024),
        }
        env.update(overrides)
        return mock.patch.dict(os.environ, env)

    @staticmethod
    def _random_coo(num_rows: int, num_edges: int, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        row = torch.randint(0, num_rows, (num_edges,), generator=generator)
        col = torch.randint(0, num_rows, (num_edges,), generator=generator)
        return row, col

    def _scatter(self, row, col, indptr, chunk_size, band_bytes):
        indices = torch.empty(row.numel(), dtype=torch.int64)
        _scatter_in_bands(
            row=row,
            col=col,
            indptr=indptr,
            indices=indices,
            chunk_size=chunk_size,
            band_bytes=band_bytes,
        )
        return indices

    @parameterized.expand(
        [
            # (rows, edges, band_bytes) -- band sizes chosen to force 1, several, and many bands.
            # Both input dtypes, because PRODUCTION rows/cols are int32 (the int32-edge-index
            # lever): the banded path compares an int32 tensor against Python int bounds and
            # upcasts with `.to(torch.int64) - first_row`, none of which int64-only tests touch.
            param(
                "one band", rows=50, edges=400, band_bytes=1 << 20, dtype=torch.int64
            ),
            param(
                "several bands", rows=50, edges=400, band_bytes=256, dtype=torch.int64
            ),
            param(
                "one row per band", rows=50, edges=400, band_bytes=8, dtype=torch.int64
            ),
            param(
                "empty rows present",
                rows=500,
                edges=120,
                band_bytes=128,
                dtype=torch.int64,
            ),
            param(
                "int32 one band",
                rows=50,
                edges=400,
                band_bytes=1 << 20,
                dtype=torch.int32,
            ),
            param(
                "int32 several bands",
                rows=50,
                edges=400,
                band_bytes=256,
                dtype=torch.int32,
            ),
            param(
                "int32 one row per band",
                rows=50,
                edges=400,
                band_bytes=8,
                dtype=torch.int32,
            ),
            param(
                "int32 empty rows",
                rows=500,
                edges=120,
                band_bytes=128,
                dtype=torch.int32,
            ),
        ]
    )
    def test_several_bands_match_a_single_band(
        self, _, rows: int, edges: int, band_bytes: int, dtype: torch.dtype
    ):
        row, col = self._random_coo(rows, edges)
        row, col = row.to(dtype), col.to(dtype)
        indptr = build_csr_from_coo(row=row, col=col, num_rows=rows).indptr

        torch.testing.assert_close(
            self._scatter(row, col, indptr, chunk_size=7, band_bytes=band_bytes),
            self._scatter(row, col, indptr, chunk_size=7, band_bytes=None),
        )

    def test_bands_agree_on_a_row_larger_than_a_band(self):
        """A supernode cannot be split, so its band is oversized by design."""
        row = torch.cat([torch.zeros(300, dtype=torch.int64), torch.arange(1, 20)])
        col = torch.arange(row.numel(), dtype=torch.int64) % 19
        indptr = build_csr_from_coo(row=row, col=col, num_rows=20).indptr

        torch.testing.assert_close(
            # A couple of elements at most: forces a band per row.
            self._scatter(row, col, indptr, chunk_size=11, band_bytes=8),
            self._scatter(row, col, indptr, chunk_size=11, band_bytes=None),
        )

    def _shm_fits(self):
        """Pretend the shared-memory mount has room, whatever this HOST's /dev/shm says.

        Without this the placement tests assert an environmental accident: even a 1.6 MB tensor
        needs n_bytes + the 2 GiB reserve free on the mount, so on a container with Docker's
        default 64 MiB /dev/shm every one of them silently takes the disk branch and the policy
        under test never runs. The cgroup-headroom check stays real (or explicitly mocked) --
        this only removes the mount-size dependence.
        """
        return mock.patch(
            "gigl.utils.share_memory._shared_memory_shortfall", return_value=None
        )

    def test_a_scattered_destination_prefers_memory_even_when_spilling_is_on(self):
        """Disk-first is right for a streamed write, wrong for a scatter."""
        with (
            self._spilling(),
            self._shm_fits(),
            mock.patch(
                "gigl.utils.share_memory.available_memory_bytes", return_value=1 << 50
            ),
        ):
            streamed = allocate_preshared((200_000,), torch.int64)
            scattered = allocate_preshared((200_000,), torch.int64, random_access=True)

        self.assertTrue(is_disk_backed(streamed), "default should still prefer disk")
        self.assertFalse(
            is_disk_backed(scattered), "a scattered destination should be in memory"
        )
        self.assertTrue(
            scattered.is_shared(), "and still pre-shared, or GLT duplicates it"
        )

    def test_compact_edge_ids_are_not_copied_when_shared(self):
        coo = torch.stack([torch.arange(2_000) % 20, torch.arange(2_000) % 30])
        with (
            self._spilling(GIGL_TENSOR_SPILL_MIN_BYTES="1024"),
            self._shm_fits(),
            mock.patch(
                "gigl.utils.share_memory.available_memory_bytes", return_value=1 << 50
            ),
        ):
            topology = CompactTopology(
                coo, layout="CSR", retain_edge_ids=True, compressed_id_offset=0
            )
            edge_ids_ptr = topology.edge_ids.data_ptr()
            self.assertTrue(topology.edge_ids.is_shared())
            topology.share_memory_()
            topology.share_memory_()
        self.assertEqual(topology.edge_ids.data_ptr(), edge_ids_ptr)

    def test_it_falls_back_to_disk_when_memory_cannot_hold_it(self):
        with (
            self._spilling(),
            # The mount says yes, so the fallback below is attributable to the CGROUP check
            # specifically -- without this, a small host /dev/shm would make it pass for the
            # wrong reason.
            self._shm_fits(),
            mock.patch(
                "gigl.utils.share_memory.available_memory_bytes", return_value=1024
            ),
        ):
            scattered = allocate_preshared((200_000,), torch.int64, random_access=True)

        self.assertTrue(
            is_disk_backed(scattered),
            "with no headroom the file is the only option left",
        )

    def test_a_build_whose_destination_lands_on_disk_still_matches_upstream(self):
        row, col = self._random_coo(64, 500)
        with (
            # The destination is 500 int64 = 4000 B, so the threshold has to be under that for
            # the file path to be taken at all.
            self._spilling(GIGL_TENSOR_SPILL_MIN_BYTES="1024"),
            mock.patch(
                "gigl.utils.share_memory.available_memory_bytes", return_value=1024
            ),
        ):
            result = build_csr_from_coo(row=row, col=col, num_rows=64, band_bytes=256)

        self.assertTrue(is_disk_backed(result.indices))
        reference_indptr, reference_indices = _reference_csr(row, col, 64)
        torch.testing.assert_close(
            result.indptr, reference_indptr.to(result.indptr.dtype)
        )
        torch.testing.assert_close(result.indices, reference_indices)

    def test_banded_disk_outputs_keep_edge_ids_aligned(self):
        order = torch.randperm(128, generator=torch.Generator().manual_seed(23))
        row = (torch.arange(128) // 16)[order]
        col = (torch.arange(128) % 16)[order]
        with (
            self._spilling(GIGL_TENSOR_SPILL_MIN_BYTES="512"),
            mock.patch(
                "gigl.utils.share_memory.available_memory_bytes", return_value=128
            ),
        ):
            result = build_csr_from_coo(
                row,
                col,
                num_rows=8,
                chunk_size=17,
                band_bytes=128,
                retain_edge_ids=True,
            )

        edge_ids = result.edge_ids
        assert edge_ids is not None
        self.assertTrue(is_disk_backed(result.indices))
        self.assertTrue(is_disk_backed(edge_ids))
        expected_ptr, expected_col, expected_ids, _ = coo_to_csr(
            row, col, edge_id=torch.arange(128), node_sizes=(8, 16)
        )
        torch.testing.assert_close(result.indptr, expected_ptr.to(result.indptr.dtype))
        torch.testing.assert_close(result.indices, expected_col)
        torch.testing.assert_close(edge_ids, expected_ids)


class IndptrDtypeTest(TestCase):
    """The indptr is int32 while the edge count fits and int64 beyond it, in every code path.

    A graph past 2**31 edges cannot be allocated in a unit test, so the overflow boundary is
    exercised by lowering ``INT32_INDPTR_MAX_EDGES``; the default is asserted separately.
    """

    @parameterized.expand(
        [
            param("no_edges", num_edges=0, expected=torch.int32),
            param("one_edge", num_edges=1, expected=torch.int32),
            param(
                "the_largest_count_int32_holds",
                num_edges=2**31 - 1,
                expected=torch.int32,
            ),
            param("one_edge_past_int32", num_edges=2**31, expected=torch.int64),
            param("far_past_int32", num_edges=2**33, expected=torch.int64),
        ]
    )
    def test_the_default_limit_switches_to_int64_one_edge_past_int32(
        self, _name: str, num_edges: int, expected: torch.dtype
    ) -> None:
        self.assertEqual(select_indptr_dtype(num_edges), expected)

    @parameterized.expand(
        [
            param("below_the_limit", num_edges=99, expected=torch.int32),
            param("at_the_limit", num_edges=100, expected=torch.int32),
            param("past_the_limit", num_edges=101, expected=torch.int64),
        ]
    )
    def test_the_lowered_limit_moves_the_boundary(
        self, _name: str, num_edges: int, expected: torch.dtype
    ) -> None:
        with mock.patch("gigl.utils.csr.INT32_INDPTR_MAX_EDGES", 100):
            self.assertEqual(select_indptr_dtype(num_edges), expected)

    @parameterized.expand(
        [
            param("int64_columns_stay_int64", column_dtype=torch.int64),
            param("int32_columns_stay_int32", column_dtype=torch.int32),
        ]
    )
    def test_int32_and_int64_indptr_hold_the_same_offsets_and_leave_indices_alone(
        self, _name: str, column_dtype: torch.dtype
    ) -> None:
        generator = torch.Generator().manual_seed(3)
        num_rows = 300
        row = torch.randint(0, num_rows, (4_000,), generator=generator)
        col = torch.randint(0, num_rows, (4_000,), generator=generator).to(column_dtype)

        narrow = build_csr_from_coo(row, col, num_rows=num_rows, retain_edge_ids=True)
        with mock.patch("gigl.utils.csr.INT32_INDPTR_MAX_EDGES", 100):
            wide = build_csr_from_coo(row, col, num_rows=num_rows, retain_edge_ids=True)

        self.assertEqual(narrow.indptr.dtype, torch.int32)
        self.assertEqual(wide.indptr.dtype, torch.int64)
        torch.testing.assert_close(narrow.indptr.long(), wide.indptr, rtol=0, atol=0)
        torch.testing.assert_close(narrow.indices, wide.indices, rtol=0, atol=0)
        torch.testing.assert_close(narrow.edge_ids, wide.edge_ids, rtol=0, atol=0)
        self.assertEqual(narrow.indices.dtype, column_dtype)

    def test_an_empty_build_uses_the_selected_dtype(self) -> None:
        empty = torch.empty(0, dtype=torch.int64)
        self.assertEqual(
            build_csr_from_coo(empty, empty, num_rows=4).indptr.dtype, torch.int32
        )
        with mock.patch("gigl.utils.csr.INT32_INDPTR_MAX_EDGES", -1):
            self.assertEqual(
                build_csr_from_coo(empty, empty, num_rows=4).indptr.dtype, torch.int64
            )

    @parameterized.expand(
        [
            param(
                "csr_with_a_row_for_every_id_below_the_range_end",
                layout="CSR",
                rank_local=False,
            ),
            param("csr_with_only_the_rank_local_rows", layout="CSR", rank_local=True),
            param(
                "csc_with_a_row_for_every_id_below_the_range_end",
                layout="CSC",
                rank_local=False,
            ),
            param("csc_with_only_the_rank_local_rows", layout="CSC", rank_local=True),
        ]
    )
    def test_both_row_layouts_narrow_the_indptr_and_round_trip_through_to_coo(
        self, _name: str, layout: Literal["CSR", "CSC"], rank_local: bool
    ) -> None:
        """A rank owning global ids [40, 100): rank-local stores 60 rows, full-range stores 100."""
        range_start, range_rows = 40, 60
        generator = torch.Generator().manual_seed(5)
        compressed = torch.randint(
            range_start, range_start + range_rows, (500,), generator=generator
        )
        other = torch.randint(0, 1000, (500,), generator=generator)
        compressed_dimension = 0 if layout == "CSR" else 1
        coo = torch.stack(
            [compressed, other] if layout == "CSR" else [other, compressed]
        )
        compressed_id_offset = 0
        num_rows = range_start + range_rows
        if rank_local:
            # What dist_dataset does before the build: shift into the rank's own row range.
            coo[compressed_dimension] -= range_start
            compressed_id_offset = range_start
            num_rows = range_rows

        def build() -> CompactTopology:
            return CompactTopology(
                coo.clone(),
                layout=layout,
                num_rows=num_rows,
                compressed_id_offset=compressed_id_offset,
            )

        narrow_topology = build()
        with mock.patch("gigl.utils.csr.INT32_INDPTR_MAX_EDGES", 100):
            wide_topology = build()

        self.assertEqual(narrow_topology.indptr.dtype, torch.int32)
        self.assertEqual(wide_topology.indptr.dtype, torch.int64)
        self.assertEqual(narrow_topology.indptr.numel(), num_rows + 1)
        torch.testing.assert_close(
            narrow_topology.indptr.long(), wide_topology.indptr, rtol=0, atol=0
        )
        torch.testing.assert_close(
            narrow_topology.indices, wide_topology.indices, rtol=0, atol=0
        )
        # to_coo runs GLT's ptr2ind over the indptr and adds the offset back.
        for narrow_part, wide_part in zip(
            narrow_topology.to_coo()[:2], wide_topology.to_coo()[:2]
        ):
            torch.testing.assert_close(narrow_part, wide_part, rtol=0, atol=0)
        original_compressed = narrow_topology.to_coo()[compressed_dimension]
        self.assertEqual(
            sorted(original_compressed.tolist()), sorted(compressed.tolist())
        )

    def test_the_banded_scatter_and_sort_give_the_same_columns_for_either_indptr_width(
        self,
    ) -> None:
        generator = torch.Generator().manual_seed(9)
        num_rows = 200
        row = torch.randint(0, num_rows, (3_000,), generator=generator)
        col = torch.randint(0, num_rows, (3_000,), generator=generator)
        indptr64 = torch.zeros(num_rows + 1, dtype=torch.int64)
        torch.cumsum(torch.bincount(row, minlength=num_rows), dim=0, out=indptr64[1:])

        results = []
        for indptr in (indptr64, indptr64.to(torch.int32)):
            indices = torch.empty(row.numel(), dtype=torch.int64)
            _scatter_in_bands(
                row=row,
                col=col,
                indptr=indptr,
                indices=indices,
                chunk_size=211,
                band_bytes=512,
            )
            _sort_within_rows(indptr, indices, block_edges=97)
            results.append(indices)

        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
        expected_indptr, expected_indices = _reference_csr(row, col, num_rows)
        torch.testing.assert_close(results[1], expected_indices, rtol=0, atol=0)

    @parameterized.expand(
        [
            param("an_offset_some_row_starts_at", edge_offset=3, expected=3),
            param("an_offset_between_two_row_starts", edge_offset=4, expected=3),
            param("the_final_offset", edge_offset=5, expected=3),
            param("an_offset_past_every_row_start", edge_offset=6, expected=4),
            # Past int32 itself: must not overflow or widen an int32 indptr.
            param(
                "an_offset_beyond_what_int32_holds", edge_offset=2**31 + 7, expected=4
            ),
        ]
    )
    def test_the_first_row_at_or_past_an_offset_is_the_same_for_int32_and_int64_indptr(
        self, _name: str, edge_offset: int, expected: int
    ) -> None:
        indptr = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
        self.assertEqual(_first_row_at_or_past(indptr, edge_offset), expected)
        self.assertEqual(
            _first_row_at_or_past(indptr.to(torch.int64), edge_offset), expected
        )

    def test_edge_balanced_bounds_are_the_same_for_int32_and_int64_indptr(self) -> None:
        indptr = torch.tensor([0, 5, 5, 9, 30, 31, 31, 60], dtype=torch.int64)
        self.assertEqual(
            _edge_balanced_row_bounds(indptr.to(torch.int32).numpy(), 3).tolist(),
            _edge_balanced_row_bounds(indptr.numpy(), 3).tolist(),
        )


if __name__ == "__main__":
    from absl.testing import absltest

    absltest.main()
