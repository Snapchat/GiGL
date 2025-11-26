"""Tests for distributed environment utilities."""

import unittest
from unittest import mock

from parameterized import param, parameterized

from gigl.env.distributed import GraphStoreInfo


class TestGraphStoreInfo(unittest.TestCase):
    """Test suite for GraphStoreInfo properties."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.graph_store_info = GraphStoreInfo(
            num_storage_nodes=4,
            num_compute_nodes=8,
            cluster_master_ip="10.0.0.1",
            storage_cluster_master_ip="10.0.0.2",
            compute_cluster_master_ip="10.0.0.3",
            cluster_master_port=1234,
            storage_cluster_master_port=1235,
            compute_cluster_master_port=1236,
            num_processes_per_compute=2,
        )

    def test_num_cluster_nodes(self):
        """Test num_cluster_nodes returns sum of storage and compute nodes."""
        expected = 12  # 4 storage + 8 compute
        self.assertEqual(self.graph_store_info.num_cluster_nodes, expected)

    def test_compute_cluster_world_size(self):
        """Test compute_cluster_world_size returns correct calculation."""
        expected = 16  # 8 compute nodes * 2 processes per compute
        self.assertEqual(self.graph_store_info.compute_cluster_world_size, expected)

    @parameterized.expand(
        [
            param("first_storage_node", rank="8", expected_storage_rank=0),
            param("middle_storage_node", rank="10", expected_storage_rank=2),
            param("last_storage_node", rank="11", expected_storage_rank=3),
        ]
    )
    def test_storage_node_rank_valid(self, _, rank, expected_storage_rank):
        """Test storage_node_rank returns correct rank for valid storage nodes."""
        with mock.patch.dict("os.environ", {"RANK": rank}):
            self.assertEqual(
                self.graph_store_info.storage_node_rank, expected_storage_rank
            )

    @parameterized.expand(
        [
            param("first_compute_node", rank="0"),
            param("middle_compute_node", rank="5"),
            param("last_compute_node", rank="7"),
        ]
    )
    def test_storage_node_rank_invalid(self, _, rank):
        """Test storage_node_rank raises ValueError for compute node ranks."""
        with mock.patch.dict("os.environ", {"RANK": rank}):
            with self.assertRaises(ValueError) as context:
                _ = self.graph_store_info.storage_node_rank
            self.assertIn("is not a server rank", str(context.exception))
            self.assertIn(f"Global rank {rank}", str(context.exception))

    @parameterized.expand(
        [
            param("first_compute_node", rank="0", expected_compute_rank=0),
            param("middle_compute_node", rank="5", expected_compute_rank=5),
            param("last_compute_node", rank="7", expected_compute_rank=7),
        ]
    )
    def test_compute_node_rank_valid(self, _, rank, expected_compute_rank):
        """Test compute_node_rank returns correct rank for valid compute nodes."""
        with mock.patch.dict("os.environ", {"RANK": rank}):
            self.assertEqual(
                self.graph_store_info.compute_node_rank, expected_compute_rank
            )

    @parameterized.expand(
        [
            param("first_storage_node", rank="8"),
            param("middle_storage_node", rank="10"),
            param("last_storage_node", rank="11"),
        ]
    )
    def test_compute_node_rank_invalid(self, _, rank):
        """Test compute_node_rank raises ValueError for storage node ranks."""
        with mock.patch.dict("os.environ", {"RANK": rank}):
            with self.assertRaises(ValueError) as context:
                _ = self.graph_store_info.compute_node_rank
            self.assertIn("is not a compute rank", str(context.exception))
            self.assertIn(f"Global rank {rank}", str(context.exception))


if __name__ == "__main__":
    unittest.main()
