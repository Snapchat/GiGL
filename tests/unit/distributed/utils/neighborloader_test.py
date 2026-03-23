from typing import Optional, Union

import torch
from absl.testing import absltest
from parameterized import param, parameterized
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

from gigl.distributed.sampler import (
    NEGATIVE_LABEL_METADATA_KEY,
    POSITIVE_LABEL_METADATA_KEY,
)
from gigl.distributed.utils.neighborloader import (
    ServerSlice,
    compute_server_assignments,
    extract_edge_type_metadata,
    extract_metadata,
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


class TestComputeServerAssignments(TestCase):
    @parameterized.expand(
        [
            param(
                "rank_0",
                compute_rank=0,
                expected_assignments={
                    0: ServerSlice(
                        server_rank=0,
                        start_num=0,
                        start_den=2,
                        end_num=2,
                        end_den=2,
                    ),
                    1: ServerSlice(
                        server_rank=1,
                        start_num=0,
                        start_den=2,
                        end_num=1,
                        end_den=2,
                    ),
                },
            ),
            param(
                "rank_1",
                compute_rank=1,
                expected_assignments={
                    1: ServerSlice(
                        server_rank=1,
                        start_num=1,
                        start_den=2,
                        end_num=2,
                        end_den=2,
                    ),
                    2: ServerSlice(
                        server_rank=2,
                        start_num=0,
                        start_den=2,
                        end_num=2,
                        end_den=2,
                    ),
                },
            ),
        ]
    )
    def test_fractional_boundary_assignment(
        self, _, compute_rank: int, expected_assignments: dict[int, ServerSlice]
    ) -> None:
        assignments = compute_server_assignments(
            num_servers=3, num_compute_nodes=2, compute_rank=compute_rank
        )

        self.assertEqual(assignments, expected_assignments)

    def test_assignments_recombine_server_data(self) -> None:
        tensor = torch.arange(7)
        all_assignments = [
            compute_server_assignments(
                num_servers=2, num_compute_nodes=5, compute_rank=rank
            )
            for rank in range(5)
        ]

        for server_rank in range(2):
            combined = torch.cat(
                [
                    assignments[server_rank].slice_tensor(tensor)
                    for assignments in all_assignments
                    if server_rank in assignments
                ]
            )
            self.assert_tensor_equality(combined, tensor)

    @parameterized.expand(
        [
            param(
                "negative_servers",
                num_servers=-1,
                num_compute_nodes=2,
                compute_rank=0,
            ),
            param(
                "zero_servers",
                num_servers=0,
                num_compute_nodes=2,
                compute_rank=0,
            ),
            param(
                "negative_compute_nodes",
                num_servers=2,
                num_compute_nodes=-1,
                compute_rank=0,
            ),
            param(
                "zero_compute_nodes",
                num_servers=2,
                num_compute_nodes=0,
                compute_rank=0,
            ),
            param(
                "rank_too_large",
                num_servers=2,
                num_compute_nodes=2,
                compute_rank=2,
            ),
            param(
                "negative_rank",
                num_servers=2,
                num_compute_nodes=2,
                compute_rank=-1,
            ),
        ]
    )
    def test_validates_arguments(
        self, _, num_servers: int, num_compute_nodes: int, compute_rank: int
    ) -> None:
        with self.assertRaises(ValueError):
            compute_server_assignments(
                num_servers=num_servers,
                num_compute_nodes=num_compute_nodes,
                compute_rank=compute_rank,
            )


class TestServerSlice(TestCase):
    def test_full_tensor_returns_same_object(self) -> None:
        tensor = torch.arange(10)
        server_slice = ServerSlice(
            server_rank=0, start_num=0, start_den=1, end_num=1, end_den=1
        )
        result = server_slice.slice_tensor(tensor)
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

    def test_partial_slice_clones_requested_range(self) -> None:
        tensor = torch.arange(10)
        server_slice = ServerSlice(
            server_rank=0, start_num=0, start_den=2, end_num=1, end_den=2
        )
        result = server_slice.slice_tensor(tensor)
        self.assert_tensor_equality(result, torch.arange(5))
        self.assertNotEqual(result.data_ptr(), tensor.data_ptr())


class ExtractMetadataTest(TestCase):
    def setUp(self):
        self._device = torch.device("cpu")
        super().setUp()

    def test_separates_metadata_from_sampling_data(self):
        msg = {
            "#META.ppr_scores": torch.tensor([1.0, 2.0]),
            "#META.custom_key": torch.tensor([3]),
            "user.ids": torch.tensor([10, 20]),
            "user__to__item.rows": torch.tensor([0, 1]),
        }
        metadata, stripped_msg = extract_metadata(msg, self._device)

        self.assertEqual(set(metadata.keys()), {"ppr_scores", "custom_key"})
        self.assert_tensor_equality(metadata["ppr_scores"], torch.tensor([1.0, 2.0]))
        self.assert_tensor_equality(metadata["custom_key"], torch.tensor([3]))

        self.assertEqual(set(stripped_msg.keys()), {"user.ids", "user__to__item.rows"})
        self.assert_tensor_equality(stripped_msg["user.ids"], torch.tensor([10, 20]))

    def test_no_metadata_keys(self):
        msg = {
            "user.ids": torch.tensor([10, 20]),
            "#IS_HETERO": torch.tensor([1]),
        }
        metadata, stripped_msg = extract_metadata(msg, self._device)

        self.assertEqual(metadata, {})
        self.assertEqual(set(stripped_msg.keys()), {"user.ids", "#IS_HETERO"})

    def test_only_metadata_keys(self):
        msg = {
            "#META.scores": torch.tensor([1.0]),
        }
        metadata, stripped_msg = extract_metadata(msg, self._device)

        self.assertEqual(set(metadata.keys()), {"scores"})
        self.assertEqual(stripped_msg, {})

    def test_does_not_modify_original_message(self):
        original_tensor = torch.tensor([1.0, 2.0])
        msg = {
            "#META.scores": original_tensor,
            "user.ids": torch.tensor([10]),
        }
        original_keys = set(msg.keys())

        extract_metadata(msg, self._device)

        self.assertEqual(set(msg.keys()), original_keys)
        self.assertIn("#META.scores", msg)

    def test_empty_message(self):
        metadata, stripped_msg = extract_metadata({}, self._device)
        self.assertEqual(metadata, {})
        self.assertEqual(stripped_msg, {})


class ExtractEdgeTypeMetadataTest(TestCase):
    def test_matching_keys_extracted_and_parsed(self):
        pos_label_edge_type = message_passing_to_positive_label(_U2I_EDGE_TYPE)
        metadata = {
            f"{POSITIVE_LABEL_METADATA_KEY}{repr(pos_label_edge_type)}": torch.tensor(
                [[0, 1], [2, 3]]
            ),
            "other_key": torch.tensor([99]),
        }
        matched, remaining = extract_edge_type_metadata(
            metadata, [POSITIVE_LABEL_METADATA_KEY]
        )

        self.assertEqual(
            set(matched[POSITIVE_LABEL_METADATA_KEY].keys()), {pos_label_edge_type}
        )
        self.assert_tensor_equality(
            matched[POSITIVE_LABEL_METADATA_KEY][pos_label_edge_type],
            torch.tensor([[0, 1], [2, 3]]),
        )
        self.assertEqual(set(remaining.keys()), {"other_key"})
        self.assert_tensor_equality(remaining["other_key"], torch.tensor([99]))

    def test_no_matching_keys_returns_empty_matched(self):
        neg_label_edge_type = message_passing_to_positive_label(_U2I_EDGE_TYPE)
        metadata = {
            f"{NEGATIVE_LABEL_METADATA_KEY}{repr(neg_label_edge_type)}": torch.tensor(
                [[4, 5]]
            ),
        }
        matched, remaining = extract_edge_type_metadata(
            metadata, [POSITIVE_LABEL_METADATA_KEY]
        )

        self.assertEqual(matched[POSITIVE_LABEL_METADATA_KEY], {})
        self.assertEqual(
            set(remaining.keys()),
            {f"{NEGATIVE_LABEL_METADATA_KEY}{repr(neg_label_edge_type)}"},
        )

    def test_positive_and_negative_labels_extracted_in_single_call(self):
        """Typical usage: call once with both positive and negative label prefixes."""
        pos_label_edge_type = message_passing_to_positive_label(_U2I_EDGE_TYPE)
        neg_label_edge_type = message_passing_to_positive_label(_U2I_EDGE_TYPE)
        metadata = {
            f"{POSITIVE_LABEL_METADATA_KEY}{repr(pos_label_edge_type)}": torch.tensor(
                [[0, 1]]
            ),
            f"{NEGATIVE_LABEL_METADATA_KEY}{repr(neg_label_edge_type)}": torch.tensor(
                [[4, 5]]
            ),
            "extra": torch.tensor([42]),
        }
        matched, remaining = extract_edge_type_metadata(
            metadata, [POSITIVE_LABEL_METADATA_KEY, NEGATIVE_LABEL_METADATA_KEY]
        )
        positive_labels = matched[POSITIVE_LABEL_METADATA_KEY]
        negative_labels = matched[NEGATIVE_LABEL_METADATA_KEY]

        self.assertEqual(set(positive_labels.keys()), {pos_label_edge_type})
        self.assert_tensor_equality(
            positive_labels[pos_label_edge_type], torch.tensor([[0, 1]])
        )
        self.assertEqual(set(negative_labels.keys()), {neg_label_edge_type})
        self.assert_tensor_equality(
            negative_labels[neg_label_edge_type], torch.tensor([[4, 5]])
        )
        self.assertEqual(set(remaining.keys()), {"extra"})


if __name__ == "__main__":
    absltest.main()
