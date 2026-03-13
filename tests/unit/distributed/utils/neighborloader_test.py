from typing import Optional, Union

import torch
from absl.testing import absltest
from parameterized import param, parameterized
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

from gigl.distributed.utils.neighborloader import (
    ServerSlice,
    ShardStrategy,
    compute_server_assignments,
    labeled_to_homogeneous,
    patch_fanout_for_sampling,
    set_missing_features,
    shard_nodes_by_process,
    strip_label_edges,
)
from gigl.types.graph import FeatureInfo, message_passing_to_positive_label
from tests.test_assets.test_case import TestCase

_U2U_EDGE_TYPE = ("user", "to", "user")
_U2I_EDGE_TYPE = ("user", "to", "item")
_I2U_EDGE_TYPE = ("item", "to", "user")
_LABELED_EDGE_TYPE = message_passing_to_positive_label(_U2I_EDGE_TYPE)


class LoaderUtilsTest(TestCase):
    def setUp(self):
        self._device = torch.device("cpu")
        super().setUp()

    @parameterized.expand(
        [
            param(
                "Test shard_nodes_by_process on 0 rank",
                local_process_rank=0,
                local_process_world_size=2,
                expected_sharded_tensor=torch.tensor([1, 3, 5, 7]),
            ),
            param(
                "Test shard_nodes_by_process on 1 rank",
                local_process_rank=1,
                local_process_world_size=2,
                expected_sharded_tensor=torch.tensor([9, 11, 13, 15, 17]),
            ),
        ]
    )
    def test_shard_nodes_by_process(
        self,
        _,
        local_process_rank: int,
        local_process_world_size: int,
        expected_sharded_tensor: torch.Tensor,
    ):
        sharded_tensor = shard_nodes_by_process(
            input_nodes=torch.tensor([1, 3, 5, 7, 9, 11, 13, 15, 17]),
            local_process_rank=local_process_rank,
            local_process_world_size=local_process_world_size,
        )
        self.assert_tensor_equality(sharded_tensor, expected_sharded_tensor)

    @parameterized.expand(
        [
            param(
                "Test patch_fanout_for_sampling on num_neighbors dict with labeled edge type in dataset",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE, _LABELED_EDGE_TYPE],
                num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7],
                    _I2U_EDGE_TYPE: [3, 4],
                },
                expected_num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7],
                    _I2U_EDGE_TYPE: [3, 4],
                    _LABELED_EDGE_TYPE: [0, 0],
                },
            ),
            param(
                "Test patch_fanout_for_sampling on num_neighbors dict with labeled edge type in dataset and fanout",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE, _LABELED_EDGE_TYPE],
                num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7],
                    _I2U_EDGE_TYPE: [3, 4],
                    # If labeled edge type fanout is provided by the user, we assume it was by accident, since users shouldn't be aware of this injected edge type,
                    # and still set the fanout of it to be 0.
                    _LABELED_EDGE_TYPE: [2, 2],
                },
                expected_num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7],
                    _I2U_EDGE_TYPE: [3, 4],
                    _LABELED_EDGE_TYPE: [0, 0],
                },
            ),
            param(
                "Test patch_fanout_for_sampling on num_neighbors list",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE, _LABELED_EDGE_TYPE],
                num_neighbors=[1, 3],
                expected_num_neighbors={
                    _U2I_EDGE_TYPE: [1, 3],
                    _I2U_EDGE_TYPE: [1, 3],
                    _LABELED_EDGE_TYPE: [0, 0],
                },
            ),
            param(
                "Test patch_fanout_for_sampling on homogeneous dataset",
                edge_types=None,
                num_neighbors=[1, 3],
                expected_num_neighbors=[1, 3],
            ),
        ]
    )
    def test_patch_neighbors_with_zero_fanout(
        self,
        _,
        edge_types: Optional[list[EdgeType]],
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
        expected_num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
    ):
        num_neighbors = patch_fanout_for_sampling(edge_types, num_neighbors)
        self.assertEqual(num_neighbors, expected_num_neighbors)

    @parameterized.expand(
        [
            param(
                "Test when homogeneous dataset has heterogeneous num_neighbors",
                edge_types=None,
                num_neighbors={_U2I_EDGE_TYPE: [2, 7]},
            ),
            param(
                "Test when heterogeneous dataset and num_neighbors has different number of hops per edge type",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE],
                num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7, 10],
                    _I2U_EDGE_TYPE: [3, 4],
                },
            ),
            param(
                "Test for heterogeneous dataset and num_neighbors when there is a missing edge type in num_neighbors from the dataset",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE],
                num_neighbors={_U2I_EDGE_TYPE: [2, 7]},
            ),
            param(
                "Test for heterogeneous dataset and num_neighbors when there is an extra edge type in num_neighbors from the dataset",
                edge_types=[_I2U_EDGE_TYPE],
                num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7, 10],
                    _I2U_EDGE_TYPE: [3, 4],
                },
            ),
            param(
                "Test for homogeneous dataset with negative fanout",
                edge_types=None,
                num_neighbors=[-1, 3],
            ),
            param(
                "Test for heterogeneous dataset with negative fanout",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE],
                num_neighbors={
                    _U2I_EDGE_TYPE: [2, -7],
                    _I2U_EDGE_TYPE: [3, 4],
                },
            ),
        ]
    )
    def test_patch_neighbors_failure(
        self,
        _,
        edge_types: Optional[list[EdgeType]],
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
    ):
        with self.assertRaises(ValueError):
            num_neighbors = patch_fanout_for_sampling(edge_types, num_neighbors)

    def test_pyg_to_homogeneous(self):
        data = HeteroData()
        data[_U2I_EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 0]])
        data[_I2U_EDGE_TYPE].edge_index = torch.tensor([[1, 0], [0, 1]])
        data["user"].x = torch.tensor([[1.0], [2.0]])
        data["item"].x = torch.tensor([[3.0], [4.0]])
        data.num_sampled_edges = {
            _U2I_EDGE_TYPE: torch.tensor([1, 1]),
            _I2U_EDGE_TYPE: torch.tensor([2, 2]),
        }
        data.num_sampled_nodes = {
            "user": torch.tensor([1, 1]),
            "item": torch.tensor([2, 2]),
        }

        data.batch = torch.tensor([0, 1])

        homogeneous_data = labeled_to_homogeneous(_U2I_EDGE_TYPE, data)
        self.assertIsInstance(homogeneous_data, Data)
        self.assertTrue(hasattr(homogeneous_data, "edge_index"))
        self.assertTrue(hasattr(homogeneous_data, "batch"))
        self.assertTrue(hasattr(homogeneous_data, "batch_size"))
        self.assert_tensor_equality(
            homogeneous_data.num_sampled_nodes, torch.tensor([1, 1])
        )
        self.assert_tensor_equality(
            homogeneous_data.num_sampled_edges, torch.tensor([1, 1])
        )
        self.assert_tensor_equality(homogeneous_data.batch, torch.tensor([0, 1]))
        self.assertEqual(homogeneous_data.batch_size, 2)

    def test_strip_label_edges(self):
        data = HeteroData()
        data[_U2I_EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 0]])
        data[_I2U_EDGE_TYPE].edge_index = torch.tensor([[1, 0], [0, 1]])
        data[_LABELED_EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 0]])
        data["user"].x = torch.tensor([[1.0], [2.0]])
        data["item"].x = torch.tensor([[3.0], [4.0]])
        data.num_sampled_edges = {
            _U2I_EDGE_TYPE: torch.tensor([1, 1]),
            _I2U_EDGE_TYPE: torch.tensor([2, 2]),
            _LABELED_EDGE_TYPE: torch.tensor([1, 1]),
        }

        stripped_data = strip_label_edges(data)
        self.assertIsInstance(stripped_data, HeteroData)
        self.assertFalse(_LABELED_EDGE_TYPE in stripped_data.edge_types)
        self.assertTrue(_U2I_EDGE_TYPE in stripped_data.edge_types)
        self.assertTrue(_I2U_EDGE_TYPE in stripped_data.edge_types)

        self.assertFalse(_LABELED_EDGE_TYPE in stripped_data.num_sampled_edges)
        self.assertTrue(_U2I_EDGE_TYPE in stripped_data.num_sampled_edges)
        self.assertTrue(_I2U_EDGE_TYPE in stripped_data.num_sampled_edges)

    @parameterized.expand(
        [
            param(
                "No node features, no edge features, float dtype",
                num_node_features=0,
                num_edge_features=0,
                dtype=torch.float32,
            ),
            param(
                "Node features, no edge features, float dtype",
                num_node_features=20,
                num_edge_features=0,
                dtype=torch.float32,
            ),
            param(
                "Node features, edge features, float dtype",
                num_node_features=20,
                num_edge_features=40,
                dtype=torch.float32,
            ),
            param(
                "No node features, no edge features, int dtype",
                num_node_features=0,
                num_edge_features=0,
                dtype=torch.int32,
            ),
            param(
                "Node features, no edge features, int dtype",
                num_node_features=20,
                num_edge_features=0,
                dtype=torch.int32,
            ),
            param(
                "Node features, edge features, int dtype",
                num_node_features=20,
                num_edge_features=40,
                dtype=torch.int32,
            ),
        ]
    )
    def test_homogeneous_set_missing_features(
        self, _, num_node_features: int, num_edge_features: int, dtype: torch.dtype
    ):
        data = Data()
        if num_node_features != 0:
            data.x = torch.zeros((num_node_features, 2), dtype=dtype)
        if num_edge_features != 0:
            data.edge_attr = torch.zeros((num_edge_features, 4), dtype=dtype)
        data = set_missing_features(
            data=data,
            node_feature_info=FeatureInfo(dim=2, dtype=dtype),
            edge_feature_info=FeatureInfo(dim=4, dtype=dtype),
            device=self._device,
        )
        self.assert_tensor_equality(
            data.x,
            torch.zeros((num_node_features, 2), device=self._device, dtype=dtype),
        )
        self.assert_tensor_equality(
            data.edge_attr,
            torch.zeros((num_edge_features, 4), device=self._device, dtype=dtype),
        )

    @parameterized.expand(
        [
            param(
                "No node features, no edge features, float dtype",
                user_num_node_features=0,
                u2u_num_edge_features=0,
                dtype=torch.float32,
            ),
            param(
                "Node features, no edge features, float dtype",
                user_num_node_features=30,
                u2u_num_edge_features=0,
                dtype=torch.float32,
            ),
            param(
                "Node features, edge features, float dtype",
                user_num_node_features=30,
                u2u_num_edge_features=60,
                dtype=torch.float32,
            ),
            param(
                "No node features, no edge features, int dtype",
                user_num_node_features=0,
                u2u_num_edge_features=0,
                dtype=torch.int32,
            ),
            param(
                "Node features, no edge features, int dtype",
                user_num_node_features=30,
                u2u_num_edge_features=0,
                dtype=torch.int32,
            ),
            param(
                "Node features, edge features, int dtype",
                user_num_node_features=30,
                u2u_num_edge_features=60,
                dtype=torch.int32,
            ),
        ]
    )
    def test_heterogeneous_set_missing_features(
        self,
        _,
        user_num_node_features: int,
        u2u_num_edge_features: int,
        dtype: torch.dtype,
    ):
        hetero_data = HeteroData()
        if user_num_node_features != 0:
            hetero_data["user"].x = torch.zeros(
                (user_num_node_features, 3), dtype=dtype
            )
        if u2u_num_edge_features != 0:
            hetero_data[_U2U_EDGE_TYPE].edge_attr = torch.zeros(
                (u2u_num_edge_features, 6), dtype=dtype
            )
        hetero_data = set_missing_features(
            data=hetero_data,
            node_feature_info={
                "user": FeatureInfo(dim=3, dtype=dtype),
                "item": FeatureInfo(dim=4, dtype=dtype),
            },
            edge_feature_info={
                _U2U_EDGE_TYPE: FeatureInfo(dim=6, dtype=dtype),
                _I2U_EDGE_TYPE: FeatureInfo(dim=7, dtype=dtype),
            },
            device=self._device,
        )

        self.assert_tensor_equality(
            hetero_data["user"].x,
            torch.zeros(
                (user_num_node_features, 3),
                device=self._device,
                dtype=dtype,
            ),
        )
        self.assert_tensor_equality(
            hetero_data["item"].x,
            torch.zeros((0, 4), device=self._device, dtype=dtype),
        )

        self.assert_tensor_equality(
            hetero_data[_U2U_EDGE_TYPE].edge_attr,
            torch.zeros(
                (u2u_num_edge_features, 6),
                device=self._device,
                dtype=dtype,
            ),
        )
        self.assert_tensor_equality(
            hetero_data[_I2U_EDGE_TYPE].edge_attr,
            torch.zeros((0, 7), device=self._device, dtype=dtype),
        )

    def test_set_missing_features_no_feats(self):
        data = set_missing_features(
            data=Data(),
            node_feature_info=None,
            edge_feature_info=None,
            device=self._device,
        )
        self.assertIsNone(data.x)
        self.assertIsNone(data.edge_attr)

        hetero_data = HeteroData()
        hetero_data["user", "to", "item"] = torch.tensor([[0, 1], [1, 2]])

        hetero_data = set_missing_features(
            data=HeteroData(),
            node_feature_info=None,
            edge_feature_info=None,
            device=self._device,
        )
        self.assertFalse(hasattr(hetero_data["user"], "x"))
        self.assertFalse(hasattr(hetero_data["item"], "x"))
        self.assertFalse(hasattr(hetero_data["user"], "edge_attr"))
        self.assertFalse(hasattr(hetero_data["item"], "edge_attr"))

    def test_set_missing_features_failure(self):
        with self.assertRaises(ValueError):
            set_missing_features(
                data=Data(),
                node_feature_info={
                    "user": FeatureInfo(dim=3, dtype=torch.float32),
                    "item": FeatureInfo(dim=4, dtype=torch.float32),
                },
                edge_feature_info={
                    _U2U_EDGE_TYPE: FeatureInfo(dim=6, dtype=torch.float32),
                    _I2U_EDGE_TYPE: FeatureInfo(dim=7, dtype=torch.float32),
                },
                device=self._device,
            )
        with self.assertRaises(ValueError):
            set_missing_features(
                data=HeteroData(),
                node_feature_info=FeatureInfo(dim=2, dtype=torch.float32),
                edge_feature_info=FeatureInfo(dim=2, dtype=torch.float32),
                device=self._device,
            )

    def test_set_custom_features_homogeneous(self):
        data = Data()
        input_node_feats = torch.tensor(
            [[3.0, 5.0], [4.0, 6.0], [-7.0, -10.0]], dtype=torch.float64
        )
        data.x = input_node_feats
        data = set_missing_features(
            data=data,
            node_feature_info=FeatureInfo(dim=2, dtype=torch.float64),
            edge_feature_info=FeatureInfo(dim=4, dtype=torch.int32),
            device=self._device,
        )
        # Assert we did not override the value or data type of the node features
        self.assert_tensor_equality(
            data.x,
            input_node_feats,
        )
        # Assert we set the edge type features and the appropriate data type
        self.assert_tensor_equality(
            data.edge_attr,
            torch.zeros((0, 4), device=self._device, dtype=torch.int32),
        )

    def test_set_custom_features_heterogeneous(self):
        data = HeteroData()
        input_user_node_feats = torch.tensor(
            [[3.0, 5.0], [4.0, 6.0], [-7.0, -10.0]], dtype=torch.float64
        )
        input_item_node_feats = torch.tensor([[13], [14], [-17]], dtype=torch.int16)

        data["user"].x = input_user_node_feats
        data["item"].x = input_item_node_feats
        data = set_missing_features(
            data=data,
            node_feature_info={
                "user": FeatureInfo(dim=2, dtype=torch.float64),
                "item": FeatureInfo(dim=1, dtype=torch.int16),
            },
            edge_feature_info={
                _U2I_EDGE_TYPE: FeatureInfo(dim=4, dtype=torch.int32),
                _I2U_EDGE_TYPE: FeatureInfo(dim=8, dtype=torch.uint8),
            },
            device=self._device,
        )
        # Assert we did not override the value or data type of the node features
        self.assert_tensor_equality(
            data["user"].x,
            input_user_node_feats,
        )
        self.assert_tensor_equality(
            data["item"].x,
            input_item_node_feats,
        )
        # Assert we set the edge type features and the appropriate data type
        self.assert_tensor_equality(
            data[_U2I_EDGE_TYPE].edge_attr,
            torch.zeros((0, 4), device=self._device, dtype=torch.int32),
        )
        self.assert_tensor_equality(
            data[_I2U_EDGE_TYPE].edge_attr,
            torch.zeros((0, 8), device=self._device, dtype=torch.uint8),
        )


class TestShardStrategy(TestCase):
    """Tests for ShardStrategy enum values."""

    def test_enum_values(self):
        self.assertEqual(ShardStrategy.ROUND_ROBIN.value, "round_robin")
        self.assertEqual(ShardStrategy.CONTIGUOUS.value, "contiguous")


class TestComputeServerAssignments(TestCase):
    """Tests for compute_server_assignments and ServerSlice."""

    def test_even_split_4_servers_2_compute(self):
        """4 servers, 2 compute nodes: each gets 2 full servers."""
        # Rank 0: servers 0, 1
        assignments_0 = compute_server_assignments(
            num_servers=4, num_compute_nodes=2, compute_rank=0
        )
        self.assertEqual(set(assignments_0.keys()), {0, 1})
        self.assertEqual(
            assignments_0[0],
            ServerSlice(server_rank=0, start_num=0, start_den=2, end_num=2, end_den=2),
        )
        self.assertEqual(
            assignments_0[1],
            ServerSlice(server_rank=1, start_num=0, start_den=2, end_num=2, end_den=2),
        )

        # Rank 1: servers 2, 3
        assignments_1 = compute_server_assignments(
            num_servers=4, num_compute_nodes=2, compute_rank=1
        )
        self.assertEqual(set(assignments_1.keys()), {2, 3})

    def test_fractional_split_3_servers_2_compute(self):
        """3 servers, 2 compute nodes: server 1 is split at boundary."""
        # Rank 0: [0, 1.5) → server 0 fully, server 1 first half
        assignments_0 = compute_server_assignments(
            num_servers=3, num_compute_nodes=2, compute_rank=0
        )
        self.assertEqual(set(assignments_0.keys()), {0, 1})
        # Server 0: full (start_num=0, end_num=2, den=2)
        self.assertEqual(assignments_0[0].start_num, 0)
        self.assertEqual(assignments_0[0].end_num, 2)
        # Server 1: first half (start_num=0, end_num=1, den=2)
        self.assertEqual(assignments_0[1].start_num, 0)
        self.assertEqual(assignments_0[1].end_num, 1)

        # Rank 1: [1.5, 3) → server 1 second half, server 2 fully
        assignments_1 = compute_server_assignments(
            num_servers=3, num_compute_nodes=2, compute_rank=1
        )
        self.assertEqual(set(assignments_1.keys()), {1, 2})
        # Server 1: second half (start_num=1, end_num=2, den=2)
        self.assertEqual(assignments_1[1].start_num, 1)
        self.assertEqual(assignments_1[1].end_num, 2)
        # Server 2: full
        self.assertEqual(assignments_1[2].start_num, 0)
        self.assertEqual(assignments_1[2].end_num, 2)

    def test_1_server_2_compute(self):
        """1 server, 2 compute nodes: both share one server."""
        assignments_0 = compute_server_assignments(
            num_servers=1, num_compute_nodes=2, compute_rank=0
        )
        self.assertEqual(set(assignments_0.keys()), {0})
        self.assertEqual(assignments_0[0].start_num, 0)
        self.assertEqual(assignments_0[0].end_num, 1)
        self.assertEqual(assignments_0[0].start_den, 2)

        assignments_1 = compute_server_assignments(
            num_servers=1, num_compute_nodes=2, compute_rank=1
        )
        self.assertEqual(set(assignments_1.keys()), {0})
        self.assertEqual(assignments_1[0].start_num, 1)
        self.assertEqual(assignments_1[0].end_num, 2)

    def test_more_compute_than_servers(self):
        """2 servers, 5 compute nodes: some compute nodes share a server."""
        all_assignments: list[dict[int, ServerSlice]] = []
        for rank in range(5):
            assignments = compute_server_assignments(
                num_servers=2, num_compute_nodes=5, compute_rank=rank
            )
            all_assignments.append(assignments)
            # Each rank should have at most 2 servers
            self.assertLessEqual(len(assignments), 2)

        # Verify recombination invariant for both servers
        for server in range(2):
            tensor = torch.arange(100)
            slices: list[torch.Tensor] = []
            for rank_assignments in all_assignments:
                if server in rank_assignments:
                    slices.append(rank_assignments[server].slice_tensor(tensor))
            combined = torch.cat(slices)
            self.assert_tensor_equality(combined, tensor)

    def test_single_compute_gets_all_servers(self):
        """1 compute node should get all servers fully."""
        assignments = compute_server_assignments(
            num_servers=3, num_compute_nodes=1, compute_rank=0
        )
        self.assertEqual(set(assignments.keys()), {0, 1, 2})
        for s in range(3):
            self.assertEqual(assignments[s].start_num, 0)
            self.assertEqual(assignments[s].end_num, 1)
            self.assertEqual(assignments[s].end_den, 1)

    def test_recombination_invariant_even(self):
        """Concatenating all ranks' slices for a server reproduces the original tensor."""
        tensor = torch.arange(20)
        for server in range(4):
            slices: list[torch.Tensor] = []
            for rank in range(2):
                assignments = compute_server_assignments(
                    num_servers=4, num_compute_nodes=2, compute_rank=rank
                )
                if server in assignments:
                    slices.append(assignments[server].slice_tensor(tensor))
            combined = torch.cat(slices) if slices else torch.empty(0, dtype=torch.long)
            # Each server is fully owned by exactly one rank in the even case
            if slices:
                self.assert_tensor_equality(combined, tensor)

    def test_recombination_invariant_fractional(self):
        """Fractional split: concatenating all ranks' slices reproduces the original tensor."""
        tensor = torch.arange(10)
        for server in range(3):
            slices: list[torch.Tensor] = []
            for rank in range(2):
                assignments = compute_server_assignments(
                    num_servers=3, num_compute_nodes=2, compute_rank=rank
                )
                if server in assignments:
                    slices.append(assignments[server].slice_tensor(tensor))
            combined = torch.cat(slices)
            self.assert_tensor_equality(combined, tensor)

    def test_validation_negative_servers(self):
        with self.assertRaises(ValueError):
            compute_server_assignments(
                num_servers=-1, num_compute_nodes=2, compute_rank=0
            )

    def test_validation_zero_servers(self):
        with self.assertRaises(ValueError):
            compute_server_assignments(
                num_servers=0, num_compute_nodes=2, compute_rank=0
            )

    def test_validation_negative_compute_nodes(self):
        with self.assertRaises(ValueError):
            compute_server_assignments(
                num_servers=2, num_compute_nodes=-1, compute_rank=0
            )

    def test_validation_zero_compute_nodes(self):
        with self.assertRaises(ValueError):
            compute_server_assignments(
                num_servers=2, num_compute_nodes=0, compute_rank=0
            )

    def test_validation_rank_too_large(self):
        with self.assertRaises(ValueError):
            compute_server_assignments(
                num_servers=2, num_compute_nodes=2, compute_rank=2
            )

    def test_validation_negative_rank(self):
        with self.assertRaises(ValueError):
            compute_server_assignments(
                num_servers=2, num_compute_nodes=2, compute_rank=-1
            )


class TestServerSlice(TestCase):
    """Tests for ServerSlice.slice_tensor."""

    def test_full_tensor_no_clone(self):
        """Full tensor (start=0, end=total) returns the same object, no clone."""
        tensor = torch.arange(10)
        server_slice = ServerSlice(
            server_rank=0, start_num=0, start_den=1, end_num=1, end_den=1
        )
        result = server_slice.slice_tensor(tensor)
        self.assertTrue(result.data_ptr() == tensor.data_ptr())

    def test_partial_slice_clones(self):
        """Partial slice returns a clone (different data_ptr)."""
        tensor = torch.arange(10)
        server_slice = ServerSlice(
            server_rank=0, start_num=0, start_den=2, end_num=1, end_den=2
        )
        result = server_slice.slice_tensor(tensor)
        self.assert_tensor_equality(result, torch.arange(5))
        self.assertNotEqual(result.data_ptr(), tensor.data_ptr())

    def test_second_half_slice(self):
        """Second half slice works correctly."""
        tensor = torch.arange(10)
        server_slice = ServerSlice(
            server_rank=0, start_num=1, start_den=2, end_num=2, end_den=2
        )
        result = server_slice.slice_tensor(tensor)
        self.assert_tensor_equality(result, torch.arange(5, 10))

    def test_empty_tensor(self):
        """Slicing an empty tensor returns an empty tensor."""
        tensor = torch.empty(0, dtype=torch.long)
        server_slice = ServerSlice(
            server_rank=0, start_num=0, start_den=2, end_num=1, end_den=2
        )
        result = server_slice.slice_tensor(tensor)
        self.assertEqual(len(result), 0)

    def test_odd_sized_tensor_fractional(self):
        """Odd-sized tensor with fractional split uses integer division correctly."""
        tensor = torch.arange(7)  # 7 elements
        # First half: 7 * 0 // 2 = 0, 7 * 1 // 2 = 3
        first_half = ServerSlice(
            server_rank=0, start_num=0, start_den=2, end_num=1, end_den=2
        )
        # Second half: 7 * 1 // 2 = 3, 7 * 2 // 2 = 7
        second_half = ServerSlice(
            server_rank=0, start_num=1, start_den=2, end_num=2, end_den=2
        )
        self.assert_tensor_equality(first_half.slice_tensor(tensor), torch.arange(3))
        self.assert_tensor_equality(
            second_half.slice_tensor(tensor), torch.arange(3, 7)
        )
        # Recombination
        combined = torch.cat(
            [
                first_half.slice_tensor(tensor),
                second_half.slice_tensor(tensor),
            ]
        )
        self.assert_tensor_equality(combined, tensor)


if __name__ == "__main__":
    absltest.main()
