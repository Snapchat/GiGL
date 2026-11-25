"""make unit_test_py PY_TEST_FILES="glt_int32_indptr_test.py"

The row-pointer half of ``gigl/scripts/patches/0003-glt-int32-indptr.patch``: the COMPILED CPU
samplers must read int32 indptr and produce byte-identical results to the int64 graph. This
mirrors ``glt_int32_indices_test.py`` (the column-id half, from 0001) structurally; see that
file's module docstring for why this lives in the suite rather than a probe script, and for the
determinism trick that makes element-wise comparison possible.

These tests SKIP when the installed graphlearn_torch has no int32-indptr support, so they pass on
an unpatched wheel instead of failing confusingly; ``test_which_mode_this_wheel_runs_in`` records
which mode ran, so a silently-unpatched image shows up as a skip and not as green.
"""

import unittest
from unittest import mock

import torch
from graphlearn_torch import py_graphlearn_torch as pywrap
from graphlearn_torch.data import Graph, Topology
from parameterized import param, parameterized

from gigl.scripts import verify_glt_patches
from gigl.scripts.verify_glt_patches import build_cpu_csr_graph
from gigl.utils.csr import CompactTopology, to_local_compressed_ids
from tests.test_assets.test_case import TestCase


def _random_csr(
    num_rows: int, num_edges: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A valid CSR (indptr, int64 indices, degrees) with rows in ascending order."""
    generator = torch.Generator().manual_seed(seed)
    row = torch.randint(
        0, num_rows, (num_edges,), generator=generator, dtype=torch.int64
    )
    col = torch.randint(
        0, num_rows, (num_edges,), generator=generator, dtype=torch.int64
    )
    order = torch.argsort(row)
    row, col = row[order], col[order]
    degrees = torch.bincount(row, minlength=num_rows)
    indptr = torch.zeros(num_rows + 1, dtype=torch.int64)
    torch.cumsum(degrees, dim=0, out=indptr[1:])
    return indptr, col.contiguous(), degrees


def _wheel_supports_int32_indptr() -> bool:
    """Whether the installed compiled extension accepts an int32 CSR indptr."""
    try:
        build_cpu_csr_graph(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int64),
        )
    except RuntimeError:
        return False
    return True


_HAS_INT32_INDPTR = _wheel_supports_int32_indptr()
_SKIP_REASON = "installed graphlearn_torch has no int32 indptr support: the CSR patch is not in this wheel"


class Int32IndptrSupportTest(TestCase):
    def test_which_mode_this_wheel_runs_in(self) -> None:
        """Records whether the patch is present, so an all-skip run is self-explaining.

        See ``glt_int32_indices_test.py``'s identically-named test: the same "suite ran nothing"
        trap applies here. On an unpatched wheel every test in this file skips; the signal to
        read is the skip COUNT, and ``gigl/scripts/verify_glt_patches.py`` is what actually gates
        the image build.
        """
        if not _HAS_INT32_INDPTR:
            self.skipTest(
                "int32-indptr patch support absent: int32 row pointers rejected by the compiled "
                "extension, so every parity test in this file skipped. Expected on a wheel that "
                "predates the patch; NOT expected inside an image whose wheel install_glt.sh "
                "built with gigl/scripts/patches/0003-glt-int32-indptr.patch."
            )
        self.assertTrue(_HAS_INT32_INDPTR)


@unittest.skipUnless(_HAS_INT32_INDPTR, _SKIP_REASON)
class Int32IndptrSamplingParityTest(TestCase):
    """The int32-row graph must sample IDENTICALLY to the int64 graph over the same CSR."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.num_rows = 20_000
        cls.indptr64, cls.indices, cls.degrees = _random_csr(
            cls.num_rows, 200_000, seed=7
        )
        cls.indptr32 = cls.indptr64.to(torch.int32)
        cls.graph64 = build_cpu_csr_graph(cls.indptr64, cls.indices)
        cls.graph_row32 = build_cpu_csr_graph(cls.indptr32, cls.indices)
        cls.graph_both32 = build_cpu_csr_graph(
            cls.indptr32, cls.indices.to(torch.int32)
        )
        generator = torch.Generator().manual_seed(11)
        cls.seeds = torch.randint(
            0, cls.num_rows, (4_000,), generator=generator, dtype=torch.int64
        )
        cls.max_degree = int(cls.degrees.max().item())

    def test_int32_rows_with_a_full_fanout_sample_the_same_neighbors_as_int64_rows(
        self,
    ) -> None:
        """req_num > max degree makes UniformSample copy, not draw -- so this is exact."""
        sampler64 = pywrap.CPURandomSampler(self.graph64.graph_handler)
        sampler_row32 = pywrap.CPURandomSampler(self.graph_row32.graph_handler)

        neighbors64, counts64 = sampler64.sample(self.seeds, self.max_degree + 1)
        neighbors_row32, counts_row32 = sampler_row32.sample(
            self.seeds, self.max_degree + 1
        )

        self.assertTrue(torch.equal(counts64, counts_row32), "degree counts diverged")
        self.assertTrue(
            torch.equal(neighbors64, neighbors_row32),
            "sampled neighbor ids diverged between the int64 and int32-row graphs",
        )
        self.assertGreater(neighbors_row32.numel(), 0, "fixture sampled nothing")

    def test_int32_rows_with_int32_columns_sample_the_same_neighbors_as_int64(
        self,
    ) -> None:
        """Row and column narrowing are independent flags; the combination the GiGL
        partitioners actually build (both int32) must also sample identically."""
        sampler64 = pywrap.CPURandomSampler(self.graph64.graph_handler)
        sampler_both32 = pywrap.CPURandomSampler(self.graph_both32.graph_handler)

        neighbors64, counts64 = sampler64.sample(self.seeds, self.max_degree + 1)
        neighbors_both32, counts_both32 = sampler_both32.sample(
            self.seeds, self.max_degree + 1
        )

        self.assertTrue(torch.equal(counts64, counts_both32))
        self.assertTrue(torch.equal(neighbors64, neighbors_both32))

    def test_neighbors_and_counts_sampled_from_int32_rows_are_int64(self) -> None:
        """The narrowing is at REST only; see the identically-named test in
        ``glt_int32_indices_test.py`` for why this matters downstream."""
        sampler_row32 = pywrap.CPURandomSampler(self.graph_row32.graph_handler)
        neighbors, counts = sampler_row32.sample(self.seeds, 5)
        self.assertEqual(neighbors.dtype, torch.int64)
        self.assertEqual(counts.dtype, torch.int64)

    def test_a_fanout_below_the_degree_draws_true_neighbors_with_exact_counts(
        self,
    ) -> None:
        """A draw cannot be compared element-wise, so check membership and cardinality."""
        sampler_row32 = pywrap.CPURandomSampler(self.graph_row32.graph_handler)
        requested = 5
        neighbors, counts = sampler_row32.sample(self.seeds, requested)
        offsets = torch.zeros(self.seeds.numel() + 1, dtype=torch.int64)
        torch.cumsum(counts, dim=0, out=offsets[1:])

        for index in range(0, self.seeds.numel(), 173):
            node = int(self.seeds[index])
            true_neighbors = set(
                self.indices[self.indptr64[node] : self.indptr64[node + 1]].tolist()
            )
            drawn = neighbors[offsets[index] : offsets[index + 1]].tolist()
            self.assertEqual(
                len(drawn),
                min(requested, int(self.degrees[node])),
                f"wrong number of neighbors drawn for node {node}",
            )
            for neighbor in drawn:
                self.assertIn(
                    neighbor, true_neighbors, f"node {node} drew a non-neighbor"
                )


@unittest.skipUnless(_HAS_INT32_INDPTR, _SKIP_REASON)
class Int32IndptrRejectionTest(TestCase):
    """Paths NOT taught the int32 row-pointer layout must fail loudly, never sample wrongly.

    ``row_ptr_`` is nullptr on an int32-row graph, so an untaught consumer that kept reading it
    would dereference null (a crash at best, garbage at worst). This asserts the ``TORCH_CHECK``
    fires instead -- see the identically-structured class in ``glt_int32_indices_test.py``.
    """

    def setUp(self) -> None:
        super().setUp()
        self.indptr64, indices, _ = _random_csr(500, 4_000, seed=13)
        self.graph_row32 = build_cpu_csr_graph(self.indptr64.to(torch.int32), indices)
        self.seeds = torch.arange(10, dtype=torch.int64)

    def test_an_unsupported_indptr_dtype_is_rejected(self) -> None:
        """Only int32 and int64 are accepted; anything else must not be reinterpreted."""
        with self.assertRaises(RuntimeError):
            build_cpu_csr_graph(
                torch.tensor([0, 1], dtype=torch.int16),
                torch.tensor([0], dtype=torch.int64),
            )

    def test_a_non_contiguous_int32_row_array_is_rejected(self) -> None:
        """Samplers read the row-pointer array as flat storage, so a strided view must not be
        taken."""
        indptr = torch.arange(8, dtype=torch.int32)[::2]
        self.assertFalse(indptr.is_contiguous())
        with self.assertRaises(RuntimeError):
            build_cpu_csr_graph(indptr, torch.zeros(3, dtype=torch.int64))

    @unittest.skipUnless(
        torch.cuda.is_available() and hasattr(pywrap.Graph(), "init_cuda_from_csr"),
        "needs a GPU and a WITH_CUDA=ON graphlearn_torch build",
    )
    def test_cuda_init_rejects_an_int32_row_topology(self) -> None:
        """The CUDA path was deliberately NOT taught int32 row pointers.

        ``Graph::InitCUDAGraphFromCSR`` calls ``indptr.data_ptr<int64_t>()`` unconditionally --
        unlike the CPU path, it has no ``TORCH_CHECK`` of its own for this. torch's accessor
        throws on a dtype mismatch, so the failure is loud for free, but that is a property of
        torch's accessor that a future refactor could remove -- asserted rather than assumed, for
        the same reason as the column-id analog of this test in ``glt_int32_indices_test.py``.

        Skipped on a CPU-only wheel, where the entry point does not exist at all: that is the
        local build, NOT the artifact that ships. This assertion only means something inside the
        base image, which is where the in-image re-verification runs it.
        """
        indptr, indices, _ = _random_csr(100, 500, seed=17)
        topology = Topology.__new__(Topology)
        topology._layout = "CSR"
        topology._indptr = indptr.to(torch.int32)
        topology._indices = indices
        topology._edge_ids = None
        topology._edge_weights = None
        graph = Graph.__new__(Graph)
        graph.topo = topology
        graph.mode = "CUDA"
        graph.device = 0
        graph._graph = None
        with self.assertRaises(RuntimeError):
            graph.lazy_init()


@unittest.skipUnless(_HAS_INT32_INDPTR, _SKIP_REASON)
class Int32IndptrEmptyGraphTest(TestCase):
    """An edgeless rank is normal under range partitioning, and it still narrows to int32."""

    def test_an_empty_int32_row_graph_initializes_and_samples_nothing(self) -> None:
        graph = build_cpu_csr_graph(
            torch.zeros(11, dtype=torch.int32), torch.empty(0, dtype=torch.int64)
        )
        self.assertEqual(graph.col_count, 0)

        sampler = pywrap.CPURandomSampler(graph.graph_handler)
        neighbors, counts = sampler.sample(torch.arange(5, dtype=torch.int64), 3)

        self.assertEqual(neighbors.numel(), 0)
        self.assertEqual(int(counts.sum()), 0)
        self.assertEqual(neighbors.dtype, torch.int64)


def _build_compact_graph(
    coo: torch.Tensor, num_rows: int, compressed_id_offset: int, wide_indptr: bool
) -> Graph:
    """A CPU graph over ``CompactTopology``, with the indptr forced to int64 when ``wide_indptr``."""
    limit = -1 if wide_indptr else torch.iinfo(torch.int32).max
    with mock.patch("gigl.utils.csr.INT32_INDPTR_MAX_EDGES", limit):
        topology = CompactTopology(
            coo,
            layout="CSR",
            num_rows=num_rows,
            retain_edge_ids=True,
            compressed_id_offset=compressed_id_offset,
        )
    graph = Graph(topology, "CPU", None)
    graph.lazy_init()
    return graph


@unittest.skipUnless(_HAS_INT32_INDPTR, _SKIP_REASON)
class RowLayoutSamplingParityTest(TestCase):
    """int32 and int64 indptr sample identically in the rank-local and the full-range row layout.

    A rank owning global ids [range_start, range_end) stores either only those rows (rank-local,
    seeds translated with ``to_local_compressed_ids``) or every row below ``range_end``
    (``GIGL_RANK_LOCAL_CSR_ROWS=off``, global ids index the indptr directly).
    """

    range_start = 300
    range_end = 500

    def _coo(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(21)
        compressed = torch.randint(
            self.range_start, self.range_end, (6_000,), generator=generator
        )
        other = torch.randint(0, 100_000, (6_000,), generator=generator)
        return torch.stack([compressed, other])

    @parameterized.expand(
        [
            param("a_rank_that_stores_only_its_own_rows", rank_local=True),
            param("a_rank_that_stores_every_row_below_its_range_end", rank_local=False),
        ]
    )
    def test_int32_and_int64_indptr_sample_identically_and_unowned_ids_sample_nothing(
        self, _name: str, rank_local: bool
    ) -> None:
        coo = self._coo()
        if rank_local:
            coo[0] -= self.range_start
            offset, num_rows = self.range_start, self.range_end - self.range_start
        else:
            offset, num_rows = 0, self.range_end
        narrow = _build_compact_graph(coo.clone(), num_rows, offset, wide_indptr=False)
        wide = _build_compact_graph(coo.clone(), num_rows, offset, wide_indptr=True)
        self.assertEqual(narrow.topo.indptr.dtype, torch.int32)
        self.assertEqual(wide.topo.indptr.dtype, torch.int64)

        # Includes ids below and past the stored range, which must sample nothing.
        seeds = torch.arange(self.range_start - 20, self.range_end + 20)
        local_seeds = to_local_compressed_ids(seeds, offset, num_rows)
        max_degree = int(wide.topo.degrees.max())
        for fanout in (3, max_degree + 1):
            narrow_sampler = pywrap.CPURandomSampler(narrow.graph_handler)
            wide_sampler = pywrap.CPURandomSampler(wide.graph_handler)
            if fanout > max_degree:
                # Copies every neighbor, so the outputs can be compared element-wise.
                for narrow_part, wide_part in zip(
                    narrow_sampler.sample(local_seeds, fanout),
                    wide_sampler.sample(local_seeds, fanout),
                ):
                    self.assert_tensor_equality(narrow_part, wide_part)
                for narrow_part, wide_part in zip(
                    narrow_sampler.sample_with_edge(local_seeds, fanout),
                    wide_sampler.sample_with_edge(local_seeds, fanout),
                ):
                    self.assert_tensor_equality(narrow_part, wide_part)
            else:
                narrow_neighbors, narrow_counts = narrow_sampler.sample(
                    local_seeds, fanout
                )
                _, wide_counts = wide_sampler.sample(local_seeds, fanout)
                self.assert_tensor_equality(narrow_counts, wide_counts)
                self.assertEqual(narrow_neighbors.dtype, torch.int64)
        # Seeds outside the stored range contribute no neighbors.
        outside_counts = pywrap.CPURandomSampler(narrow.graph_handler).sample(
            local_seeds[:20], 3
        )[1]
        self.assertEqual(int(outside_counts.sum()), 0)


@unittest.skipUnless(_HAS_INT32_INDPTR, _SKIP_REASON)
class UntaughtSamplerRejectionTest(TestCase):
    """Every CPU reader of the row pointer either supports int32 or rejects it with a clear error."""

    def setUp(self) -> None:
        super().setUp()
        indptr, indices, _ = _random_csr(200, 1_500, seed=31)
        self.graph = build_cpu_csr_graph(indptr.to(torch.int32), indices)

    def test_the_random_negative_sampler_rejects_int32_rows(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "int32 row-pointer"):
            pywrap.CPURandomNegativeSampler(self.graph.graph_handler).sample(4, 3, True)

    def test_the_weighted_sampler_rejects_int32_rows_naming_the_row_pointer_layout(
        self,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "int32 row-pointer"):
            pywrap.CPUWeightedSampler(self.graph.graph_handler).sample(
                torch.arange(5, dtype=torch.int64), 2
            )

    def test_the_subgraph_op_rejects_int32_rows(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "int32 row-pointer"):
            pywrap.CPUSubGraphOp(self.graph.graph_handler).node_subgraph(
                torch.arange(5, dtype=torch.int64), True
            )

    def test_degrees_and_to_coo_work_on_an_int32_indptr(self) -> None:
        topology = self.graph.topo
        self.assertEqual(int(topology.degrees.sum()), topology.indices.numel())
        self.assertEqual(topology.to_coo()[0].numel(), topology.indices.numel())


class VerifyInt32IndptrGateTest(TestCase):
    """The build-time gate: it must fail the image when 0003 is absent, and pass when it is live."""

    def test_the_gate_fails_when_the_wheel_rejects_an_int32_indptr(self) -> None:
        with mock.patch.object(
            verify_glt_patches,
            "build_cpu_csr_graph",
            side_effect=RuntimeError("expected scalar type Long"),
        ):
            self.assertFalse(verify_glt_patches.verify_int32_indptr())

    def test_main_fails_when_the_int32_indptr_gate_fails(self) -> None:
        passing = mock.Mock(return_value=True)
        with (
            mock.patch.object(verify_glt_patches, "verify_bitmap_col_count", passing),
            mock.patch.object(verify_glt_patches, "verify_int32_indices", passing),
            mock.patch.object(
                verify_glt_patches, "verify_cpu_random_missing_edge_ids", passing
            ),
            mock.patch.object(verify_glt_patches, "verify_queue_teardown", passing),
            mock.patch.object(
                verify_glt_patches, "verify_int32_indptr", return_value=False
            ) as indptr_gate,
        ):
            self.assertNotEqual(verify_glt_patches.main(), 0)
        indptr_gate.assert_called_once()

    @unittest.skipUnless(_HAS_INT32_INDPTR, _SKIP_REASON)
    def test_the_gate_passes_on_a_patched_wheel(self) -> None:
        self.assertTrue(verify_glt_patches.verify_int32_indptr())


if __name__ == "__main__":
    from absl.testing import absltest

    absltest.main()
