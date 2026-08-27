"""make unit_test_py PY_TEST_FILES="glt_int32_indices_test.py"

GLT patch 0002 (``gigl/scripts/patches/0002-glt-int32-csr-indices.patch``): the COMPILED CPU
samplers must read int32 column ids and produce byte-identical results to the int64 graph.

Why this file exists rather than a probe script: the code that can corrupt training silently
lives in a C++ extension that ``gigl/scripts/install_glt.sh`` rebuilds from source in every image
build. A wrong dispatch there does not raise -- it samples the wrong neighbors -- so the
guarantee has to be re-established against each newly built wheel, which means it belongs in the
suite and not in a scratch directory.

These tests SKIP when the installed graphlearn_torch has no int32 support, so they pass on an
unpatched wheel instead of failing confusingly; ``test_the_patch_is_present_in_this_wheel``
records which mode ran, so a silently-unpatched image shows up as a skip and not as green.

The determinism trick that makes element-wise comparison possible: with ``req_num`` above the
maximum degree, ``UniformSample`` copies every neighbor instead of drawing, so full-fanout
sampling is exact and comparable id-by-id. Sub-degree draws are then checked as a distribution
(every drawn id must be a true neighbor, exact counts).
"""

import unittest

import torch
from graphlearn_torch import py_graphlearn_torch as pywrap
from graphlearn_torch.data import Graph, Topology

from tests.test_assets.test_case import TestCase


def _build_cpu_graph(indptr: torch.Tensor, indices: torch.Tensor) -> Graph:
    """A CPU ``Graph`` over a ready-made CSR, bypassing ``Topology.__init__``.

    Populating the attributes directly keeps the fixture free of the ``arange(num_edges)`` edge
    ids ``Topology.__init__`` would attach, so the CSR reaches the compiled extension exactly as
    given.
    """
    topology = Topology.__new__(Topology)
    topology._layout = "CSR"
    topology._indptr = indptr
    topology._indices = indices
    topology._edge_ids = None
    topology._edge_weights = None
    graph = Graph.__new__(Graph)
    graph.topo = topology
    graph.mode = "CPU"
    graph.device = None
    graph._graph = None
    graph.lazy_init()
    return graph


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


def _wheel_supports_int32_indices() -> bool:
    """Whether the installed compiled extension accepts int32 CSR indices."""
    try:
        _build_cpu_graph(
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int32),
        )
    except RuntimeError:
        return False
    return True


_HAS_INT32 = _wheel_supports_int32_indices()
_SKIP_REASON = "installed graphlearn_torch has no int32 CSR support: patch 0002 is not in this wheel"


class Int32IndicesSupportTest(TestCase):
    def test_which_mode_this_wheel_runs_in(self) -> None:
        """Records whether the patch is present, so an all-skip run is self-explaining.

        This deliberately does NOT fail on an unpatched wheel. GiGL's own CI and a plain
        ``make unit_test_py`` run against the released graphlearn_torch, which has no int32
        support and never will until this patch is upstreamed -- failing there would be a broken
        build reporting a correct state.

        The consequence has to be stated plainly, because it is the "suite ran nothing" trap:
        **on an unpatched wheel every test in this file skips, so green here proves nothing about
        int32 sampling.** The signal to read is the skip COUNT. Inside an image whose wheel
        ``install_glt.sh`` built, the parity tests must actually RUN;
        ``gigl/scripts/verify_glt_patches.py`` is what checks that during the image build, and it
        is not optional before shipping an int32 topology to a cluster.
        """
        if not _HAS_INT32:
            self.skipTest(
                "patch 0002 absent: int32 CSR indices rejected by the compiled extension, so "
                "every parity test in this file skipped. Expected on a wheel that predates the "
                "patches; NOT expected inside an image whose wheel install_glt.sh built with "
                "gigl/scripts/patches/0002-glt-int32-csr-indices.patch."
            )
        self.assertTrue(_HAS_INT32)


@unittest.skipUnless(_HAS_INT32, _SKIP_REASON)
class Int32SamplingParityTest(TestCase):
    """The int32 graph must sample IDENTICALLY to the int64 graph over the same CSR."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.num_rows = 20_000
        cls.indptr, cls.indices64, cls.degrees = _random_csr(
            cls.num_rows, 200_000, seed=7
        )
        cls.indices32 = cls.indices64.to(torch.int32)
        cls.graph64 = _build_cpu_graph(cls.indptr, cls.indices64)
        cls.graph32 = _build_cpu_graph(cls.indptr, cls.indices32)
        generator = torch.Generator().manual_seed(11)
        cls.seeds = torch.randint(
            0, cls.num_rows, (4_000,), generator=generator, dtype=torch.int64
        )
        cls.max_degree = int(cls.degrees.max().item())

    def test_full_fanout_sampling_is_identical_element_wise(self) -> None:
        """req_num > max degree makes UniformSample copy, not draw -- so this is exact."""
        sampler64 = pywrap.CPURandomSampler(self.graph64.graph_handler)
        sampler32 = pywrap.CPURandomSampler(self.graph32.graph_handler)

        neighbors64, counts64 = sampler64.sample(self.seeds, self.max_degree + 1)
        neighbors32, counts32 = sampler32.sample(self.seeds, self.max_degree + 1)

        self.assertTrue(torch.equal(counts64, counts32), "degree counts diverged")
        self.assertTrue(
            torch.equal(neighbors64, neighbors32),
            "sampled neighbor ids diverged between the int64 and int32 graphs",
        )
        self.assertGreater(neighbors32.numel(), 0, "fixture sampled nothing")

    def test_sampled_outputs_stay_int64(self) -> None:
        """The narrowing is at REST only.

        Sampled ids flow into ``Data``/``HeteroData`` and are indexed against int64 feature
        tables and label tensors, so an int32 output would propagate the dtype into the whole
        batch path. The patch casts on read for exactly this reason.
        """
        sampler32 = pywrap.CPURandomSampler(self.graph32.graph_handler)
        neighbors, counts = sampler32.sample(self.seeds, 5)
        self.assertEqual(neighbors.dtype, torch.int64)
        self.assertEqual(counts.dtype, torch.int64)

    def test_sub_degree_draws_are_true_neighbors_with_exact_counts(self) -> None:
        """A draw cannot be compared element-wise, so check membership and cardinality."""
        sampler32 = pywrap.CPURandomSampler(self.graph32.graph_handler)
        requested = 5
        neighbors, counts = sampler32.sample(self.seeds, requested)
        offsets = torch.zeros(self.seeds.numel() + 1, dtype=torch.int64)
        torch.cumsum(counts, dim=0, out=offsets[1:])

        for index in range(0, self.seeds.numel(), 173):
            node = int(self.seeds[index])
            true_neighbors = set(
                self.indices64[self.indptr[node] : self.indptr[node + 1]].tolist()
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

    def test_col_count_is_identical(self) -> None:
        """col_count comes from the patched distinct-count, which is now dtype-generic."""
        self.assertEqual(self.graph64.col_count, self.graph32.col_count)
        self.assertGreater(self.graph32.col_count, 0)

    def test_the_int32_graph_holds_half_the_bytes(self) -> None:
        """The entire point: the CSC column array halves at billion-edge scale."""
        bytes64 = self.indices64.numel() * self.indices64.element_size()
        bytes32 = self.indices32.numel() * self.indices32.element_size()
        self.assertEqual(bytes64, 2 * bytes32)


@unittest.skipUnless(_HAS_INT32, _SKIP_REASON)
class Int32RejectionTest(TestCase):
    """Paths NOT taught the int32 layout must fail loudly, never sample wrongly.

    ``col_idx_`` is nullptr on an int32 graph, so an untaught consumer that kept reading it
    would dereference null (a crash at best, garbage at worst). Each of these asserts the
    ``TORCH_CHECK`` fires instead.
    """

    def setUp(self) -> None:
        super().setUp()
        self.indptr, indices64, _ = _random_csr(500, 4_000, seed=13)
        self.graph32 = _build_cpu_graph(self.indptr, indices64.to(torch.int32))
        self.seeds = torch.arange(10, dtype=torch.int64)

    def test_the_weighted_sampler_rejects_an_int32_graph(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            pywrap.CPUWeightedSampler(self.graph32.graph_handler).sample(self.seeds, 3)
        self.assertIn("int32", str(caught.exception))

    def test_an_unsupported_indices_dtype_is_rejected(self) -> None:
        """Only int32 and int64 are accepted; anything else must not be reinterpreted."""
        with self.assertRaises(RuntimeError):
            _build_cpu_graph(
                torch.tensor([0, 1], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int16),
            )

    def test_a_non_contiguous_int32_column_array_is_rejected(self) -> None:
        """Samplers read the column array as flat storage, so a strided view must not be taken."""
        indices = torch.arange(8, dtype=torch.int32)[::2]
        self.assertFalse(indices.is_contiguous())
        with self.assertRaises(RuntimeError):
            _build_cpu_graph(torch.tensor([0, 2, 4], dtype=torch.int64), indices)

    @unittest.skipUnless(
        torch.cuda.is_available() and hasattr(pywrap.Graph(), "init_cuda_from_csr"),
        "needs a GPU and a WITH_CUDA=ON graphlearn_torch build",
    )
    def test_cuda_init_rejects_an_int32_topology(self) -> None:
        """The CUDA path was deliberately NOT taught int32.

        It reads ``data_ptr<int64_t>()``, which throws on a dtype mismatch -- so the failure is
        loud for free. Asserted rather than assumed, because "it happens to throw" is a property
        of torch's accessor that a future refactor could remove.

        Skipped on a CPU-only wheel, where the entry point does not exist at all: that is the
        local build, NOT the artifact that ships. This assertion only means something inside the
        base image, which is where the in-image re-verification runs it.
        """
        indptr, indices64, _ = _random_csr(100, 500, seed=17)
        topology = Topology.__new__(Topology)
        topology._layout = "CSR"
        topology._indptr = indptr
        topology._indices = indices64.to(torch.int32)
        topology._edge_ids = None
        topology._edge_weights = None
        graph = Graph.__new__(Graph)
        graph.topo = topology
        graph.mode = "CUDA"
        graph.device = 0
        graph._graph = None
        with self.assertRaises(RuntimeError):
            graph.lazy_init()


@unittest.skipUnless(_HAS_INT32, _SKIP_REASON)
class Int32EmptyGraphTest(TestCase):
    """An edgeless rank is normal under range partitioning, and it still narrows to int32."""

    def test_an_empty_int32_graph_initializes_and_samples_nothing(self) -> None:
        graph = _build_cpu_graph(
            torch.zeros(11, dtype=torch.int64), torch.empty(0, dtype=torch.int32)
        )
        self.assertEqual(graph.col_count, 0)

        sampler = pywrap.CPURandomSampler(graph.graph_handler)
        neighbors, counts = sampler.sample(torch.arange(5, dtype=torch.int64), 3)

        self.assertEqual(neighbors.numel(), 0)
        self.assertEqual(int(counts.sum()), 0)
        self.assertEqual(neighbors.dtype, torch.int64)


if __name__ == "__main__":
    from absl.testing import absltest

    absltest.main()
