import os
import tempfile
from unittest import mock

import torch
from graphlearn_torch.utils import coo_to_csr
from parameterized import param, parameterized

from gigl.utils.csr import _scatter_in_bands, _scatter_whole, build_csr_from_coo
from gigl.utils.share_memory import allocate_preshared, is_disk_backed
from tests.test_assets.test_case import TestCase


def _reference_csr(
    row: torch.Tensor, col: torch.Tensor, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (indptr, indices) that graphlearn_torch would produce for the same input."""
    indptr, indices, _, _ = coo_to_csr(
        row.to(torch.int64), col.to(torch.int64), node_sizes=(num_rows, num_rows)
    )
    return indptr, indices


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
        indptr, indices = build_csr_from_coo(
            row, col, num_rows=num_rows, chunk_size=chunk_size
        )

        torch.testing.assert_close(indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
        self.assertEqual(indptr.dtype, torch.int64)
        self.assertEqual(indices.dtype, torch.int64)

    def test_matches_upstream_with_mostly_empty_rows(self) -> None:
        generator = torch.Generator().manual_seed(11)
        num_rows = 500_000
        row = torch.randint(0, 5, (200,), generator=generator) * 100_000
        col = torch.randint(0, num_rows, (200,), generator=generator)

        expected_indptr, expected_indices = _reference_csr(row, col, num_rows)
        indptr, indices = build_csr_from_coo(row, col, num_rows=num_rows)

        torch.testing.assert_close(indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)

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
        indptr, indices = build_csr_from_coo(
            row, col, num_rows=num_rows, chunk_size=512, sort_block_edges=64
        )

        torch.testing.assert_close(indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)

    def test_matches_upstream_when_sort_blocks_are_capped_by_rows(self) -> None:
        """Blocks crossing long runs of empty rows are cut by row count as well as by edges."""
        generator = torch.Generator().manual_seed(17)
        num_rows = 2_000
        row = torch.randint(0, 20, (400,), generator=generator) * 100
        col = torch.randint(0, num_rows, (400,), generator=generator)

        expected_indptr, expected_indices = _reference_csr(row, col, num_rows)
        indptr, indices = build_csr_from_coo(
            row, col, num_rows=num_rows, sort_block_edges=7
        )

        torch.testing.assert_close(indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)

    def test_matches_upstream_with_duplicate_edges(self) -> None:
        generator = torch.Generator().manual_seed(17)
        row = torch.randint(0, 4, (10_000,), generator=generator)
        col = torch.randint(0, 3, (10_000,), generator=generator)

        expected_indptr, expected_indices = _reference_csr(row, col, 4)
        indptr, indices = build_csr_from_coo(row, col, num_rows=4, chunk_size=333)

        torch.testing.assert_close(indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)

    def test_trailing_rows_beyond_max_row_id_are_empty(self) -> None:
        generator = torch.Generator().manual_seed(19)
        row = torch.randint(0, 100, (5_000,), generator=generator)
        col = torch.randint(0, 100, (5_000,), generator=generator)

        indptr, indices = build_csr_from_coo(row, col, num_rows=4096)

        self.assertEqual(indptr.numel(), 4097)
        self.assertEqual(int(indptr[-1]), 5_000)
        # Every row above the largest observed row id has zero degree.
        self.assertEqual(int((indptr[101:] - indptr[100:-1]).sum()), 0)
        self.assertEqual(indices.numel(), 5_000)

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

        indptr, indices = build_csr_from_coo(
            row, col, num_rows=num_rows, sort_block_edges=1 << 30
        )

        for row_id in occupied.tolist():
            start, end = int(indptr[row_id]), int(indptr[row_id + 1])
            self.assertGreater(end, start, f"row {row_id} should have edges")
            slice_ = indices[start:end]
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
        indptr, indices = build_csr_from_coo(col, row, num_rows=num_rows)

        torch.testing.assert_close(indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)

    def test_empty_input(self) -> None:
        empty = torch.empty(0, dtype=torch.int64)

        indptr, indices = build_csr_from_coo(empty, empty, num_rows=10)

        self.assertEqual(indptr.numel(), 11)
        self.assertEqual(int(indptr.sum()), 0)
        self.assertEqual(indices.numel(), 0)
        self.assertEqual(indices.dtype, torch.int64)

    def test_row_id_beyond_num_rows_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_csr_from_coo(torch.tensor([5]), torch.tensor([0]), num_rows=3)

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
    """Where `indices` lands, and whether the banded path agrees with the direct one.

    The banded path exists because job 3301592358876872704 spent over an hour in the direct scatter
    against a file-backed destination: 251 passes over 8.2M pages of a 31.3 GiB array. Confining
    writes to a window makes each page dirty once -- but only matters if it produces the same CSR.
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
    def test_banded_scatter_matches_the_direct_scatter(
        self, _, rows: int, edges: int, band_bytes: int, dtype: torch.dtype
    ):
        row, col = self._random_coo(rows, edges)
        row, col = row.to(dtype), col.to(dtype)
        expected_indptr, expected_indices = build_csr_from_coo(
            row=row, col=col, num_rows=rows, chunk_size=7, sort_within_row=False
        )

        # Same layout and dtype as production would allocate, filled by the banded path instead.
        banded = torch.empty(edges, dtype=expected_indices.dtype)
        _scatter_in_bands(
            row=row,
            col=col,
            indptr=expected_indptr,
            indices=banded,
            chunk_size=7,
            band_bytes=band_bytes,
        )

        torch.testing.assert_close(banded, expected_indices)

    def test_the_two_paths_agree_on_a_row_larger_than_a_band(self):
        """A supernode cannot be split, so its band is oversized by design."""
        row = torch.cat([torch.zeros(300, dtype=torch.int64), torch.arange(1, 20)])
        col = torch.arange(row.numel(), dtype=torch.int64) % 19
        indptr, expected = build_csr_from_coo(
            row=row, col=col, num_rows=20, chunk_size=11, sort_within_row=False
        )

        banded = torch.empty(row.numel(), dtype=expected.dtype)
        _scatter_in_bands(
            row=row,
            col=col,
            indptr=indptr,
            indices=banded,
            chunk_size=11,
            band_bytes=8,  # a couple of elements at most: forces a band per row
        )

        torch.testing.assert_close(banded, expected)

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
        with self._spilling(), self._shm_fits():
            streamed = allocate_preshared((200_000,), torch.int64)
            scattered = allocate_preshared((200_000,), torch.int64, random_access=True)

        self.assertTrue(is_disk_backed(streamed), "default should still prefer disk")
        self.assertFalse(
            is_disk_backed(scattered), "a scattered destination should be in memory"
        )
        self.assertTrue(
            scattered.is_shared(), "and still pre-shared, or GLT duplicates it"
        )

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

    @parameterized.expand(
        # int32 is what production feeds this path (the int32-edge-index lever).
        [param("int64", dtype=torch.int64), param("int32", dtype=torch.int32)]
    )
    def test_build_uses_the_banded_path_for_a_disk_backed_destination(
        self, _, dtype: torch.dtype
    ):
        """Placement must actually select the algorithm, not just be logged."""
        row, col = self._random_coo(64, 500)
        row, col = row.to(dtype), col.to(dtype)
        with (
            # The destination is 500 int64 = 4000 B, so the threshold has to be under that for
            # the file path to be taken at all.
            self._spilling(GIGL_TENSOR_SPILL_MIN_BYTES="1024"),
            mock.patch(
                "gigl.utils.share_memory.available_memory_bytes", return_value=1024
            ),
            mock.patch(
                "gigl.utils.csr._scatter_in_bands", wraps=_scatter_in_bands
            ) as banded,
            mock.patch("gigl.utils.csr._scatter_whole", wraps=_scatter_whole) as whole,
        ):
            indptr, indices = build_csr_from_coo(row=row, col=col, num_rows=64)

        banded.assert_called_once()
        whole.assert_not_called()
        # And the result is still right, not merely produced by the intended function.
        reference_indptr, reference_indices = _reference_csr(row, col, 64)
        torch.testing.assert_close(indptr, reference_indptr)
        torch.testing.assert_close(indices.to(torch.int64), reference_indices)

    def test_build_uses_the_direct_path_when_the_destination_is_in_memory(self):
        row, col = self._random_coo(64, 500)
        with (
            # Threshold under the 2000 B destination, so it is the random_access POLICY choosing
            # memory rather than the tensor merely being too small to spill.
            self._spilling(GIGL_TENSOR_SPILL_MIN_BYTES="1024"),
            self._shm_fits(),
            mock.patch(
                "gigl.utils.csr._scatter_in_bands", wraps=_scatter_in_bands
            ) as banded,
            mock.patch("gigl.utils.csr._scatter_whole", wraps=_scatter_whole) as whole,
        ):
            indptr, indices = build_csr_from_coo(row=row, col=col, num_rows=64)

        self.assertFalse(is_disk_backed(indices), "policy should have chosen memory")
        self.assertTrue(
            indices.is_shared(),
            "and pre-shared, so GLT's share_memory_() has nothing to duplicate",
        )
        # indptr placement is not asserted here: at num_rows=64 it is 520 B, below any threshold,
        # so this fixture cannot say anything about its policy either way.
        self.assertEqual(indptr.numel(), 65)

        whole.assert_called_once()
        banded.assert_not_called()


if __name__ == "__main__":
    from absl.testing import absltest

    absltest.main()
