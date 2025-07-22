import unittest
from typing import Optional, Union

import torch
from parameterized import param, parameterized
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

from gigl.distributed.utils.neighborloader import (
    labeled_to_homogeneous,
    patch_fanout_for_sampling,
    set_missing_features,
    shard_nodes_by_process,
    strip_label_edges,
)
from gigl.types.graph import message_passing_to_positive_label
from tests.test_assets.distributed.utils import assert_tensor_equality

_U2U_EDGE_TYPE = ("user", "to", "user")
_U2I_EDGE_TYPE = ("user", "to", "item")
_I2U_EDGE_TYPE = ("item", "to", "user")
_LABELED_EDGE_TYPE = message_passing_to_positive_label(_U2I_EDGE_TYPE)


class LoaderUtilsTest(unittest.TestCase):
    def setUp(self):
        self._device = torch.device("cpu")
        self._feature_dtype = torch.float32
        return super().setUp()

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
        assert_tensor_equality(sharded_tensor, expected_sharded_tensor)

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
        assert_tensor_equality(homogeneous_data.num_sampled_nodes, torch.tensor([1, 1]))
        assert_tensor_equality(homogeneous_data.num_sampled_edges, torch.tensor([1, 1]))
        assert_tensor_equality(homogeneous_data.batch, torch.tensor([0, 1]))
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
                "No node features, no edge features",
                num_node_features=0,
                num_edge_features=0,
            ),
            param(
                "Node features, no edge features",
                num_node_features=20,
                num_edge_features=0,
            ),
            param(
                "Node features, edge features",
                num_node_features=20,
                num_edge_features=40,
            ),
        ]
    )
    def test_homogeneous_set_missing_features(
        self, _, num_node_features: int, num_edge_features: int
    ):
        data = Data()
        if num_node_features != 0:
            data.x = torch.ones((num_node_features, 2), dtype=self._feature_dtype)
        if num_edge_features != 0:
            data.edge_attr = torch.ones(
                (num_edge_features, 4), dtype=self._feature_dtype
            )
        data = set_missing_features(
            data=data,
            node_feature_dim=2,
            edge_feature_dim=4,
            device=self._device,
            dtype=self._feature_dtype,
        )
        assert_tensor_equality(
            data.x,
            torch.ones(
                (num_node_features, 2), device=self._device, dtype=self._feature_dtype
            ),
        )
        assert_tensor_equality(
            data.edge_attr,
            torch.ones(
                (num_edge_features, 4), device=self._device, dtype=self._feature_dtype
            ),
        )

    @parameterized.expand(
        [
            param(
                "No node features, no edge features",
                user_num_node_features=0,
                u2u_num_edge_features=0,
            ),
            param(
                "Node features, no edge features",
                user_num_node_features=30,
                u2u_num_edge_features=0,
            ),
            param(
                "Node features, edge features",
                user_num_node_features=30,
                u2u_num_edge_features=60,
            ),
        ]
    )
    def test_heterogeneous_set_missing_features(
        self, _, user_num_node_features: int, u2u_num_edge_features: int
    ):
        hetero_data = HeteroData()
        if user_num_node_features != 0:
            hetero_data["user"].x = torch.ones(
                (user_num_node_features, 3), dtype=self._feature_dtype
            )
        if u2u_num_edge_features != 0:
            hetero_data[_U2U_EDGE_TYPE].edge_attr = torch.ones(
                (u2u_num_edge_features, 6), dtype=self._feature_dtype
            )
        hetero_data = set_missing_features(
            data=hetero_data,
            node_feature_dim={"user": 3, "item": 4},
            edge_feature_dim={_U2U_EDGE_TYPE: 6, _I2U_EDGE_TYPE: 7},
            device=self._device,
            dtype=torch.float32,
        )

        assert_tensor_equality(
            hetero_data["user"].x,
            torch.ones(
                (user_num_node_features, 3),
                device=self._device,
                dtype=self._feature_dtype,
            ),
        )
        assert_tensor_equality(
            hetero_data["item"].x,
            torch.ones((0, 4), device=self._device, dtype=self._feature_dtype),
        )

        assert_tensor_equality(
            hetero_data[_U2U_EDGE_TYPE].edge_attr,
            torch.ones(
                (u2u_num_edge_features, 6),
                device=self._device,
                dtype=self._feature_dtype,
            ),
        )
        assert_tensor_equality(
            hetero_data[_I2U_EDGE_TYPE].edge_attr,
            torch.ones((0, 7), device=self._device, dtype=self._feature_dtype),
        )

    def test_set_missing_features_failure(self):
        with self.assertRaises(ValueError):
            set_missing_features(
                data=Data(),
                node_feature_dim={"user": 3, "item": 4},
                edge_feature_dim={_U2U_EDGE_TYPE: 6, _I2U_EDGE_TYPE: 7},
                device=self._device,
                dtype=self._feature_dtype,
            )
        with self.assertRaises(ValueError):
            set_missing_features(
                data=HeteroData(),
                node_feature_dim=3,
                edge_feature_dim=2,
                device=self._device,
                dtype=self._feature_dtype,
            )
