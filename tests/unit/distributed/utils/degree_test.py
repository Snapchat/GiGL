import torch
from absl.testing import absltest
from parameterized import param, parameterized

from gigl.distributed.utils.degree import (
    _clamp_to_int16,
    _compute_degrees_from_indptr,
    _pad_to_size,
    compute_and_broadcast_degree_tensor,
)
from tests.test_assets.distributed.test_dataset import (
    DEFAULT_HETEROGENEOUS_EDGE_INDICES,
    DEFAULT_HOMOGENEOUS_EDGE_INDEX,
    create_heterogeneous_dataset,
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase


def _compute_expected_degrees_from_edge_index(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Compute expected out-degrees from COO edge index."""
    src_nodes = edge_index[0]
    degrees = torch.zeros(num_nodes, dtype=torch.int16)
    for src in src_nodes:
        degrees[src] += 1
    return degrees


class TestLocalDegreeComputation(TestCase):
    """Tests for local DistDataset degree computation."""

    def test_homogeneous_graph(self):
        """Test degree computation for a homogeneous graph using real DistDataset."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = compute_and_broadcast_degree_tensor(dataset)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assertEqual(result.shape[0], num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_homogeneous_graph_custom_edges(self):
        """Test degree computation with a custom edge index."""
        edge_index = torch.tensor([[0, 0, 1, 2, 2, 2, 3], [1, 2, 2, 0, 1, 3, 0]])
        num_nodes = int(edge_index[0].max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = compute_and_broadcast_degree_tensor(dataset)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_heterogeneous_graph(self):
        """Test degree computation for a heterogeneous graph using real DistDataset."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = compute_and_broadcast_degree_tensor(dataset)

        assert isinstance(result, dict)
        self.assertEqual(len(result), len(edge_indices))

        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
            self.assertIn(edge_type, result)
            self.assert_tensor_equality(result[edge_type], expected)


class TestDistributedDegreeComputation(TestCase):
    """Tests for degree computation with torch.distributed initialized.

    These tests verify that the all-reduce path works correctly when
    torch.distributed is initialized. Uses a single-node process group.
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

    def test_homogeneous_graph_distributed(self):
        """Test degree computation with distributed initialized."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = compute_and_broadcast_degree_tensor(dataset)

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assertEqual(result.shape[0], num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_heterogeneous_graph_distributed(self):
        """Test heterogeneous degree computation with distributed initialized."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = compute_and_broadcast_degree_tensor(dataset)

        assert isinstance(result, dict)
        self.assertEqual(len(result), len(edge_indices))

        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
            self.assertIn(edge_type, result)
            self.assert_tensor_equality(result[edge_type], expected)

    def test_num_shared_data_processes_divides_degrees(self):
        """Test that num_shared_data_processes correctly divides degrees for over-counting correction.

        In colocated mode, multiple processes on the same machine share the same data.
        When they all compute and all-reduce their degrees, the result is over-counted
        by the number of local processes. The num_shared_data_processes parameter corrects this.

        With a single-process test group, passing num_shared_data_processes=2 should halve
        the degrees (simulating 2 processes that would have over-counted).
        """
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)

        # Compute with num_shared_data_processes=1 (no correction)
        result_no_correction = compute_and_broadcast_degree_tensor(
            dataset, num_shared_data_processes=1
        )

        # Reset the cached degree tensor to compute again
        dataset._degree_tensor = None

        # Compute with num_shared_data_processes=2 (simulates 2 processes sharing data)
        result_with_correction = compute_and_broadcast_degree_tensor(
            dataset, num_shared_data_processes=2
        )

        # The result with correction should be half of the uncorrected result
        # (integer division, so may have rounding)
        assert isinstance(result_no_correction, torch.Tensor)
        assert isinstance(result_with_correction, torch.Tensor)
        expected_corrected = result_no_correction // 2
        self.assert_tensor_equality(result_with_correction, expected_corrected)

    def test_num_shared_data_processes_heterogeneous(self):
        """Test num_shared_data_processes correction for heterogeneous graphs."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        # Compute with num_shared_data_processes=1 (no correction)
        result_no_correction = compute_and_broadcast_degree_tensor(
            dataset, num_shared_data_processes=1
        )

        # Reset the cached degree tensor to compute again
        dataset._degree_tensor = None

        # Compute with num_shared_data_processes=2 (simulates 2 processes sharing data)
        result_with_correction = compute_and_broadcast_degree_tensor(
            dataset, num_shared_data_processes=2
        )

        assert isinstance(result_no_correction, dict)
        assert isinstance(result_with_correction, dict)

        for edge_type in result_no_correction:
            expected_corrected = result_no_correction[edge_type] // 2
            self.assert_tensor_equality(
                result_with_correction[edge_type], expected_corrected
            )


class TestDatasetDegreeProperty(TestCase):
    """Tests for DistDataset.compute_degree_tensor() and degree_tensor property."""

    def test_degree_tensor_initially_none(self):
        """Test that degree_tensor is None before compute_degree_tensor is called."""
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)
        self.assertIsNone(dataset.degree_tensor)

    def test_compute_degree_tensor_homogeneous(self):
        """Test compute_degree_tensor for a homogeneous graph."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        result = dataset.compute_degree_tensor()

        assert isinstance(result, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(result, expected)

    def test_compute_degree_tensor_caches_result(self):
        """Test that compute_degree_tensor caches the result."""
        dataset = create_homogeneous_dataset(edge_index=DEFAULT_HOMOGENEOUS_EDGE_INDEX)

        result1 = dataset.compute_degree_tensor()
        result2 = dataset.compute_degree_tensor()

        self.assertIs(result1, result2)

    def test_degree_tensor_property_after_compute(self):
        """Test that degree_tensor property returns cached result after compute."""
        edge_index = DEFAULT_HOMOGENEOUS_EDGE_INDEX
        num_nodes = int(edge_index.max().item() + 1)

        dataset = create_homogeneous_dataset(edge_index=edge_index)
        computed = dataset.compute_degree_tensor()
        from_property = dataset.degree_tensor

        self.assertIs(computed, from_property)
        assert isinstance(from_property, torch.Tensor)
        expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
        self.assert_tensor_equality(from_property, expected)

    def test_compute_degree_tensor_heterogeneous(self):
        """Test compute_degree_tensor for a heterogeneous graph."""
        edge_indices = DEFAULT_HETEROGENEOUS_EDGE_INDICES
        dataset = create_heterogeneous_dataset(edge_indices=edge_indices)

        result = dataset.compute_degree_tensor()

        assert isinstance(result, dict)
        self.assertEqual(len(result), len(edge_indices))

        for edge_type, edge_index in edge_indices.items():
            num_nodes = int(edge_index[0].max().item() + 1)
            expected = _compute_expected_degrees_from_edge_index(edge_index, num_nodes)
            self.assertIn(edge_type, result)
            self.assert_tensor_equality(result[edge_type], expected)


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

    def test_compute_degrees_from_indptr(self):
        """Test _compute_degrees_from_indptr helper function."""
        indptr = torch.tensor([0, 3, 5, 10, 12], dtype=torch.int64)
        expected = torch.tensor([3, 2, 5, 2], dtype=torch.int16)
        result = _compute_degrees_from_indptr(indptr)
        self.assert_tensor_equality(result, expected)

    def test_clamp_to_int16(self):
        """Test _clamp_to_int16 helper function."""
        max_int16 = torch.iinfo(torch.int16).max
        tensor = torch.tensor([1, max_int16 + 100, 5], dtype=torch.int64)
        expected = torch.tensor([1, max_int16, 5], dtype=torch.int16)
        result = _clamp_to_int16(tensor)
        self.assert_tensor_equality(result, expected)


if __name__ == "__main__":
    absltest.main()
