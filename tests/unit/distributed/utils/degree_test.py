import inspect
from collections import Counter
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.data import Topology
from parameterized import param, parameterized

from gigl.distributed.utils.degree import (
    _INT32_MAX,
    _accumulate_degrees_in_place,
    _all_reduce_single_degree_tensor,
    _clamp_to_int32,
    _compute_degrees_from_topology,
    _get_degree_reduce_context,
    _pad_to_size,
    compute_and_broadcast_degree_tensor,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.utils.csr import CompactTopology
from tests.test_assets.distributed.test_dataset import (
    DEFAULT_HETEROGENEOUS_EDGE_INDICES,
    DEFAULT_HOMOGENEOUS_EDGE_INDEX,
    STORY,
    STORY_TO_USER,
    USER,
    USER_TO_STORY,
    create_heterogeneous_dataset,
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    create_test_process_group,
    get_process_group_init_method,
)
from tests.test_assets.test_case import TestCase


def _compute_expected_degrees_from_edge_index(
    edge_index: torch.Tensor, num_nodes: int, node_axis: int = 0
) -> torch.Tensor:
    """Compute expected degrees from a COO edge index along one endpoint axis."""
    nodes = edge_index[node_axis]
    degrees = torch.zeros(num_nodes, dtype=torch.int32)
    for node in nodes:
        degrees[node] += 1
    return degrees


class TestDegreeComputation(TestCase):
    """Tests for degree computation with torch.distributed initialized.

    These tests verify that the all-reduce path works correctly.
    Uses a single-node process group.
    """

    def setUp(self):
        """Set up distributed process group before each test."""
        super().setUp()
        create_test_process_group()

    def tearDown(self):
        """Clean up distributed process group after each test."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_homogeneous_graph(self):
        """Test degree computation for a homogeneous graph."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        assert dataset.graph is not None
        result = compute_and_broadcast_degree_tensor(dataset.graph, dataset.edge_dir)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assertEqual(result.shape[0], num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_heterogeneous_graph(self):
        """Test degree computation for a heterogeneous graph."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        assert dataset.graph is not None
        result = compute_and_broadcast_degree_tensor(dataset.graph, dataset.edge_dir)

        assert not isinstance(result, torch.Tensor)
        expected = {
            USER: torch.ones(5, dtype=torch.int32),
            STORY: torch.ones(5, dtype=torch.int32),
        }
        self.assertEqual(set(result.keys()), set(expected.keys()))

        for node_type, expected_degrees in expected.items():
            self.assert_tensor_equality(result[node_type], expected_degrees)

    def test_heterogeneous_graph_with_missing_topology(self):
        """Test that edge types with missing topology get empty tensors.

        This test creates a real heterogeneous dataset, then manually sets one
        edge type's topology to None to simulate the edge case where topology
        is unavailable.
        """
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        assert dataset.graph is not None
        assert isinstance(dataset.graph, dict)

        original_graph = dataset.graph[USER_TO_STORY]
        assert original_graph.topo is not None

        # Manually set one graph's topology to None to test the edge case
        dataset.graph[STORY_TO_USER].topo = None

        result = compute_and_broadcast_degree_tensor(dataset.graph, dataset.edge_dir)

        assert not isinstance(result, torch.Tensor)
        self.assertEqual(set(result.keys()), {USER, STORY})

        # Edge type with topology should have computed degrees
        self.assert_tensor_equality(result[USER], torch.ones(5, dtype=torch.int32))

        # Edge type without topology should have empty tensor
        self.assertEqual(result[STORY].numel(), 0)


def _run_local_world_size_correction_homogeneous(
    rank: int,
    world_size: int,
    init_method: str,
    edge_index: torch.Tensor,
    expected_degrees: torch.Tensor,
) -> None:
    """Worker function for multi-process local_world_size correction test (homogeneous)."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    try:
        dataset = create_homogeneous_dataset(edge_index=edge_index)
        assert dataset.graph is not None
        result = compute_and_broadcast_degree_tensor(dataset.graph, dataset.edge_dir)

        assert isinstance(result, torch.Tensor)
        assert_tensor_equality(result, expected_degrees)
    finally:
        dist.destroy_process_group()


def _run_local_world_size_correction_heterogeneous(
    rank: int,
    world_size: int,
    init_method: str,
    edge_indices: dict[EdgeType, torch.Tensor],
    expected_degrees: dict[NodeType, torch.Tensor],
) -> None:
    """Worker function for multi-process local_world_size correction test (heterogeneous)."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    try:
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)
        assert dataset.graph is not None
        result = compute_and_broadcast_degree_tensor(dataset.graph, dataset.edge_dir)

        assert not isinstance(result, torch.Tensor)
        assert set(result.keys()) == set(expected_degrees.keys())
        for node_type, expected in expected_degrees.items():
            assert_tensor_equality(result[node_type], expected)
    finally:
        dist.destroy_process_group()


class TestLocalWorldSizeCorrection(TestCase):
    """Tests for over-counting correction with multiple processes on the same machine.

    These tests spawn 2 real processes that share the same graph data, simulating
    the scenario where multiple training processes on one machine share a partition.
    The all-reduce should correctly divide by local_world_size=2.
    """

    def test_local_world_size_correction_homogeneous(self):
        """Test over-counting correction with 2 processes sharing the same data."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        raw_degrees = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        expected_degrees = raw_degrees  # After correction: (2*raw) / 2 = raw

        init_method = get_process_group_init_method()
        mp.spawn(
            fn=_run_local_world_size_correction_homogeneous,
            args=(2, init_method, edge_index, expected_degrees),
            nprocs=2,
        )

    def test_local_world_size_correction_heterogeneous(self):
        """Test over-counting correction for heterogeneous graphs with 2 processes."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES

        expected_degrees = {
            USER: torch.ones(5, dtype=torch.int32),
            STORY: torch.ones(5, dtype=torch.int32),
        }

        init_method = get_process_group_init_method()
        mp.spawn(
            fn=_run_local_world_size_correction_heterogeneous,
            args=(2, init_method, edge_indices, expected_degrees),
            nprocs=2,
        )


def _sum_and_saturate_reference(degrees_per_source: list[torch.Tensor]) -> torch.Tensor:
    """Sum degree tensors of possibly different lengths in int64, then saturate to int32."""
    total = torch.zeros(
        max((len(degrees) for degrees in degrees_per_source), default=0),
        dtype=torch.int64,
    )
    for degrees in degrees_per_source:
        total[: len(degrees)] += degrees.to(torch.int64)
    return total.clamp(max=_INT32_MAX).to(torch.int32)


def _random_degrees(
    length: int, seed: int, high: int = 1000, num_saturated: int = 0
) -> torch.Tensor:
    """Random int32 degrees in ``[0, high)``, with the first entries set to int32 max."""
    generator = torch.Generator().manual_seed(seed)
    degrees = torch.randint(0, high, (length,), generator=generator, dtype=torch.int32)
    degrees[:num_saturated] = _INT32_MAX
    return degrees


class TestAccumulateDegreesInPlace(TestCase):
    """Tests for summing per-edge-type degrees in bounded int64 slices."""

    @parameterized.expand(
        [
            param("single_source", lengths=[10], chunk_size=4),
            param("size_not_divisible_by_chunk", lengths=[10, 10, 10], chunk_size=3),
            param("size_divisible_by_chunk", lengths=[12, 12], chunk_size=4),
            param("size_smaller_than_one_chunk", lengths=[5, 5, 5], chunk_size=100),
            param("chunk_of_one", lengths=[7, 7], chunk_size=1),
            param("later_source_longer", lengths=[4, 9, 13], chunk_size=5),
            param("later_source_shorter", lengths=[13, 9, 4], chunk_size=5),
        ]
    )
    def test_summing_edge_types_matches_an_int64_reference_for_any_chunk_size(
        self, _, lengths: list[int], chunk_size: int
    ) -> None:
        sources = [
            _random_degrees(length, seed=seed) for seed, length in enumerate(lengths)
        ]

        total = sources[0].clone()
        for source in sources[1:]:
            total = _accumulate_degrees_in_place(total, source, chunk_size=chunk_size)

        self.assertEqual(total.dtype, torch.int32)
        self.assert_tensor_equality(total, _sum_and_saturate_reference(sources))

    @parameterized.expand(
        [
            param("both_at_max", existing=[_INT32_MAX, 5], added=[_INT32_MAX, 5]),
            param("max_plus_one", existing=[_INT32_MAX, 0], added=[1, 0]),
            param("two_halves_overflow", existing=[2**30 + 1], added=[2**30 + 1]),
            param("below_max_is_exact", existing=[_INT32_MAX - 3], added=[3]),
        ]
    )
    def test_a_sum_above_int32_max_saturates_instead_of_wrapping(
        self, _, existing: list[int], added: list[int]
    ) -> None:
        existing_degrees = torch.tensor(existing, dtype=torch.int32)
        added_degrees = torch.tensor(added, dtype=torch.int32)

        total = _accumulate_degrees_in_place(
            existing_degrees, added_degrees, chunk_size=1
        )

        self.assertTrue(bool((total >= 0).all()))
        expected = torch.tensor(
            [min(left + right, _INT32_MAX) for left, right in zip(existing, added)],
            dtype=torch.int32,
        )
        self.assert_tensor_equality(total, expected)

    def test_saturating_across_many_edge_types_never_goes_negative(self) -> None:
        sources = [_random_degrees(9, seed=seed, num_saturated=4) for seed in range(5)]

        total = sources[0].clone()
        for source in sources[1:]:
            total = _accumulate_degrees_in_place(total, source, chunk_size=2)

        self.assertTrue(bool((total >= 0).all()))
        self.assert_tensor_equality(total[:4], torch.full((4,), _INT32_MAX).int())
        self.assert_tensor_equality(total, _sum_and_saturate_reference(sources))

    def test_a_running_total_that_is_long_enough_is_updated_in_place(self) -> None:
        existing = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

        total = _accumulate_degrees_in_place(
            existing, torch.tensor([10, 10], dtype=torch.int32), chunk_size=1
        )

        self.assertEqual(total.data_ptr(), existing.data_ptr())
        self.assert_tensor_equality(
            existing, torch.tensor([11, 12, 3, 4], dtype=torch.int32)
        )

    def test_longer_degrees_extend_the_running_total_with_zeros(self) -> None:
        total = _accumulate_degrees_in_place(
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([10, 10, 10, 10, 10], dtype=torch.int32),
            chunk_size=2,
        )

        self.assert_tensor_equality(
            total, torch.tensor([11, 12, 10, 10, 10], dtype=torch.int32)
        )

    @parameterized.expand(
        [
            param("both_empty", existing=[], added=[], expected=[]),
            param("empty_existing", existing=[], added=[4, 5], expected=[4, 5]),
            param("empty_degrees", existing=[4, 5], added=[], expected=[4, 5]),
        ]
    )
    def test_an_empty_side_leaves_the_other_side_unchanged(
        self, _, existing: list[int], added: list[int], expected: list[int]
    ) -> None:
        total = _accumulate_degrees_in_place(
            torch.tensor(existing, dtype=torch.int32),
            torch.tensor(added, dtype=torch.int32),
            chunk_size=3,
        )

        self.assertEqual(total.dtype, torch.int32)
        self.assert_tensor_equality(total, torch.tensor(expected, dtype=torch.int32))

    def test_default_chunk_sizes_are_unchanged(self) -> None:
        self.assertEqual(
            inspect.signature(_accumulate_degrees_in_place)
            .parameters["chunk_size"]
            .default,
            16_000_000,
        )
        self.assertEqual(
            inspect.signature(_all_reduce_single_degree_tensor)
            .parameters["chunk_size"]
            .default,
            64_000_000,
        )


class TestMultiEdgeTypeDegreeAccumulation(TestCase):
    """End-to-end degree summation across edge types sharing an anchor node type."""

    def setUp(self):
        super().setUp()
        create_test_process_group()

    def tearDown(self):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_degrees_sum_across_edge_types_that_share_an_anchor_type(self) -> None:
        # Three edge types; two of them anchor on "user" when edge_dir is "out".
        edge_rows = {
            ("user", "to", "story"): torch.tensor([0, 0, 1, 3, 3, 3]),
            ("user", "follows", "user"): torch.tensor([0, 2, 2, 2, 5]),
            ("story", "to", "user"): torch.tensor([0, 1, 1]),
        }
        graph = {}
        per_anchor_degrees: dict[str, list[torch.Tensor]] = {}
        for edge_type, rows in edge_rows.items():
            coo = torch.stack([rows, torch.arange(rows.numel())])
            topology = CompactTopology(coo, layout="CSR", compressed_id_offset=0)
            graph[edge_type] = mock.Mock(topo=topology)
            per_anchor_degrees.setdefault(edge_type[0], []).append(
                _compute_degrees_from_topology(topology)
            )

        result = compute_and_broadcast_degree_tensor(graph, "out")

        assert not isinstance(result, torch.Tensor)
        self.assertEqual(set(result.keys()), {"user", "story"})
        for anchor_type, degrees in per_anchor_degrees.items():
            self.assert_tensor_equality(
                result[anchor_type], _sum_and_saturate_reference(degrees)
            )


def _make_topology_stub(indptr: torch.Tensor) -> Topology:
    """A topology exposing only ``indptr``, so the indptr dtype can be chosen freely."""
    topology = mock.MagicMock(spec=Topology)
    topology.indptr = indptr
    return topology


class TestIndptrDtype(TestCase):
    @parameterized.expand(
        [
            param("int32_indptr", dtype=torch.int32),
            param("int64_indptr", dtype=torch.int64),
        ]
    )
    def test_degrees_are_exact_int32_for_either_indptr_dtype(
        self, _, dtype: torch.dtype
    ) -> None:
        indptr = torch.tensor([0, 3, 3, 8, 9], dtype=dtype)

        degrees = _compute_degrees_from_topology(_make_topology_stub(indptr))

        self.assertEqual(degrees.dtype, torch.int32)
        self.assert_tensor_equality(
            degrees, torch.tensor([3, 0, 5, 1], dtype=torch.int32)
        )

    def test_an_int32_indptr_difference_at_the_limit_is_kept(self) -> None:
        indptr = torch.tensor([0, _INT32_MAX, _INT32_MAX], dtype=torch.int32)

        degrees = _compute_degrees_from_topology(_make_topology_stub(indptr))

        self.assert_tensor_equality(
            degrees, torch.tensor([_INT32_MAX, 0], dtype=torch.int32)
        )

    def test_an_int64_indptr_difference_above_int32_max_saturates(self) -> None:
        indptr = torch.tensor([0, 2**31 + 5, 2**31 + 6], dtype=torch.int64)

        degrees = _compute_degrees_from_topology(_make_topology_stub(indptr))

        self.assertEqual(degrees.dtype, torch.int32)
        self.assert_tensor_equality(
            degrees, torch.tensor([_INT32_MAX, 1], dtype=torch.int32)
        )

    @parameterized.expand(
        [
            param("int32_indptr", dtype=torch.int32),
            param("int64_indptr", dtype=torch.int64),
        ]
    )
    def test_an_empty_indptr_gives_no_degrees(self, _, dtype: torch.dtype) -> None:
        degrees = _compute_degrees_from_topology(
            _make_topology_stub(torch.zeros(1, dtype=dtype))
        )

        self.assertEqual(degrees.numel(), 0)
        self.assertEqual(degrees.dtype, torch.int32)


def _run_all_reduce_single_degree_tensor(
    rank: int,
    world_size: int,
    init_method: str,
    per_rank_degrees: list[torch.Tensor],
    per_rank_machine: list[str],
    chunk_size: int,
    expected_degrees: torch.Tensor,
) -> None:
    """All-reduces this rank's degrees with the machine layout in ``per_rank_machine``."""
    dist.init_process_group(
        backend="gloo", init_method=init_method, world_size=world_size, rank=rank
    )
    try:
        with mock.patch(
            "gigl.distributed.utils.degree.get_internal_ip_from_all_ranks",
            return_value=per_rank_machine,
        ):
            local_world_size, device = _get_degree_reduce_context()
        assert local_world_size == Counter(per_rank_machine)[per_rank_machine[rank]]

        result = _all_reduce_single_degree_tensor(
            per_rank_degrees[rank].clone(), local_world_size, device, chunk_size
        )

        assert result.dtype == torch.int32
        assert bool((result >= 0).all()), f"Negative degree in {result}"
        assert_tensor_equality(result, expected_degrees)
    finally:
        dist.destroy_process_group()


class TestChunkedAllReduce(TestCase):
    """Multi-process tests for the chunked all-reduce and over-count correction.

    Each rank claims a machine id, so tests can place several ranks on one
    machine (``local_world_size`` > 1) or one rank per machine.
    """

    @parameterized.expand(
        [
            param(
                "one_rank_per_machine_sums_without_correction",
                per_rank_degrees=[[3, 5, 2, 7, 1], [1, 4, 6, 0, 2]],
                per_rank_machine=["a", "b"],
                chunk_size=2,
                expected=[4, 9, 8, 7, 3],
            ),
            param(
                "two_ranks_on_one_machine_are_not_double_counted",
                per_rank_degrees=[[3, 5, 2, 7, 1], [3, 5, 2, 7, 1]],
                per_rank_machine=["a", "a"],
                chunk_size=2,
                expected=[3, 5, 2, 7, 1],
            ),
            param(
                "four_ranks_on_one_machine_are_not_double_counted",
                per_rank_degrees=[[3, 5, 2, 7, 1]] * 4,
                per_rank_machine=["a"] * 4,
                chunk_size=3,
                expected=[3, 5, 2, 7, 1],
            ),
            param(
                "two_machines_with_two_ranks_each_sum_once_per_machine",
                per_rank_degrees=[[3, 5, 2], [3, 5, 2], [1, 4, 6], [1, 4, 6]],
                per_rank_machine=["a", "a", "b", "b"],
                chunk_size=2,
                expected=[4, 9, 8],
            ),
            param(
                "odd_sum_on_a_shared_machine_floors",
                # Machine "a" has 2 ranks, so each of its ranks divides by 2;
                # the odd sum 3 + 4 floors to 3 instead of rounding up.
                per_rank_degrees=[[3], [4]],
                per_rank_machine=["a", "a"],
                chunk_size=1,
                expected=[3],
            ),
            param(
                "ranks_with_different_lengths_are_zero_padded",
                per_rank_degrees=[[1, 2, 3, 4, 5], [10, 20, 30]],
                per_rank_machine=["a", "b"],
                chunk_size=2,
                expected=[11, 22, 33, 4, 5],
            ),
            param(
                "size_smaller_than_one_chunk",
                per_rank_degrees=[[1, 2, 3], [4, 5, 6]],
                per_rank_machine=["a", "b"],
                chunk_size=1_000,
                expected=[5, 7, 9],
            ),
            param(
                "size_divisible_by_chunk",
                per_rank_degrees=[[1, 2, 3, 4], [4, 3, 2, 1]],
                per_rank_machine=["a", "b"],
                chunk_size=2,
                expected=[5, 5, 5, 5],
            ),
            param(
                "size_not_divisible_by_chunk",
                per_rank_degrees=[list(range(10)), list(range(10))],
                per_rank_machine=["a", "b"],
                chunk_size=4,
                expected=[2 * value for value in range(10)],
            ),
            param(
                "int32_max_degrees_saturate_instead_of_wrapping",
                per_rank_degrees=[
                    [_INT32_MAX, _INT32_MAX - 1, 5],
                    [_INT32_MAX, _INT32_MAX - 1, 5],
                ],
                per_rank_machine=["a", "b"],
                chunk_size=2,
                expected=[_INT32_MAX, _INT32_MAX, 10],
            ),
            param(
                "int32_max_degrees_on_a_shared_machine_are_preserved",
                per_rank_degrees=[[_INT32_MAX, 1]] * 4,
                per_rank_machine=["a"] * 4,
                chunk_size=1,
                expected=[_INT32_MAX, 1],
            ),
            param(
                "all_ranks_empty",
                per_rank_degrees=[[], []],
                per_rank_machine=["a", "b"],
                chunk_size=4,
                expected=[],
            ),
            param(
                "one_rank_empty",
                per_rank_degrees=[[], [7, 8]],
                per_rank_machine=["a", "b"],
                chunk_size=1,
                expected=[7, 8],
            ),
        ]
    )
    def test_chunked_all_reduce_sums_ranks_and_corrects_the_over_count(
        self,
        _,
        per_rank_degrees: list[list[int]],
        per_rank_machine: list[str],
        chunk_size: int,
        expected: list[int],
    ) -> None:
        world_size = len(per_rank_degrees)
        mp.spawn(
            fn=_run_all_reduce_single_degree_tensor,
            args=(
                world_size,
                get_process_group_init_method(),
                [
                    torch.tensor(degrees, dtype=torch.int32)
                    for degrees in per_rank_degrees
                ],
                per_rank_machine,
                chunk_size,
                torch.tensor(expected, dtype=torch.int32),
            ),
            nprocs=world_size,
        )


class TestDatasetDegreeProperty(TestCase):
    """Tests for DistDataset.degree_tensor property."""

    def setUp(self):
        """Set up distributed process group before each test."""
        super().setUp()
        create_test_process_group()

    def tearDown(self):
        """Clean up distributed process group after each test."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_degree_tensor_homogeneous(self):
        """Test degree_tensor property for a homogeneous graph."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = dataset.degree_tensor

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_degree_tensor_caches_result(self):
        """Test that degree_tensor property caches the result."""
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)

        result1 = dataset.degree_tensor
        result2 = dataset.degree_tensor

        self.assertIs(result1, result2)

    def test_degree_tensor_heterogeneous(self):
        """Test degree_tensor property for a heterogeneous graph."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = dataset.degree_tensor

        assert not isinstance(result, torch.Tensor)
        expected = {
            USER: torch.ones(5, dtype=torch.int32),
            STORY: torch.ones(5, dtype=torch.int32),
        }
        self.assertEqual(set(result.keys()), set(expected.keys()))

        for node_type, expected_degrees in expected.items():
            self.assert_tensor_equality(result[node_type], expected_degrees)


class TestHelperFunctions(TestCase):
    """Tests for internal helper functions."""

    @parameterized.expand(
        [
            param(
                "pad_smaller_tensor",
                tensor=torch.tensor([1, 2, 3], dtype=torch.int32),
                target_size=5,
                expected=torch.tensor([1, 2, 3, 0, 0], dtype=torch.int32),
            ),
            param(
                "no_padding_needed",
                tensor=torch.tensor([1, 2, 3], dtype=torch.int32),
                target_size=3,
                expected=torch.tensor([1, 2, 3], dtype=torch.int32),
            ),
            param(
                "tensor_larger_than_target",
                tensor=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
                target_size=3,
                expected=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
            ),
        ]
    )
    def test_pad_to_size(self, _, tensor, target_size, expected):
        """Test _pad_to_size helper function."""
        result = _pad_to_size(tensor, target_size)
        self.assert_tensor_equality(result, expected)

    def test_compute_degrees_from_topology(self):
        """Test _compute_degrees_from_topology helper function."""
        rows = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3])
        coo = torch.stack([rows, torch.arange(rows.numel())])
        topology = CompactTopology(coo, layout="CSR", compressed_id_offset=0)
        expected = torch.tensor([3, 2, 5, 2], dtype=torch.int32)
        result = _compute_degrees_from_topology(topology)
        self.assert_tensor_equality(result, expected)

    @parameterized.expand(
        [
            param("middle_rank", row_start=4, num_rows=9),
            # A last rank whose final nodes have no edges keeps their zero degrees.
            param("last_rank_with_trailing_zero_degrees", row_start=4, num_rows=12),
        ]
    )
    def test_degrees_with_a_compressed_id_offset_are_indexed_by_global_id(
        self, _, row_start: int, num_rows: int
    ):
        coo = torch.tensor([[4, 4, 6, 8, 8, 8], [0, 1, 2, 0, 1, 3]])
        full = CompactTopology(
            coo, layout="CSR", num_rows=num_rows, compressed_id_offset=0
        )
        shifted = coo.clone()
        shifted[0] -= row_start
        offset_topology = CompactTopology(
            shifted,
            layout="CSR",
            num_rows=num_rows - row_start,
            compressed_id_offset=row_start,
        )

        self.assert_tensor_equality(
            _compute_degrees_from_topology(offset_topology),
            _compute_degrees_from_topology(full),
        )

    def test_clamp_to_int32(self):
        """Test that large degree values clamp before conversion."""
        int32_max = torch.iinfo(torch.int32).max
        tensor = torch.tensor([0, int32_max, int32_max + 1], dtype=torch.int64)
        expected = torch.tensor([0, int32_max, int32_max], dtype=torch.int32)

        result = _clamp_to_int32(tensor)

        self.assert_tensor_equality(result, expected)

    def test_compute_degrees_from_topology_all_zeros(self):
        """Test _compute_degrees_from_topology with all-zero indptr (no edges)."""
        # No edges for any of the 4 nodes
        topology = CompactTopology(
            torch.empty((2, 0), dtype=torch.int64),
            layout="CSR",
            num_rows=4,
            compressed_id_offset=0,
        )
        expected = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        result = _compute_degrees_from_topology(topology)
        self.assert_tensor_equality(result, expected)

    def test_compute_degrees_from_topology_empty(self):
        """Test _compute_degrees_from_topology with empty indptr (no nodes)."""
        # An indptr of [0] means 0 nodes
        topology = CompactTopology(
            torch.empty((2, 0), dtype=torch.int64),
            layout="CSR",
            num_rows=0,
            compressed_id_offset=0,
        )
        expected = torch.empty(0, dtype=torch.int32)
        result = _compute_degrees_from_topology(topology)
        self.assert_tensor_equality(result, expected)
        self.assertEqual(result.numel(), 0)


if __name__ == "__main__":
    absltest.main()
