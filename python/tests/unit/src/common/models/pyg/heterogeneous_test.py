import unittest

import torch
import torch_geometric
from torch_geometric.data import HeteroData

from gigl.src.common.models.pyg.heterogeneous import HGT
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation


class TestHGT(unittest.TestCase):
    def setUp(self):
        self._default_use_segment_matmul = torch_geometric.backend.use_segment_matmul
        # Set this field to True so that it will be able to raise error if the HGTConv forward pass does not behave as expected
        torch_geometric.backend.use_segment_matmul = True
        return super().setUp()

    def tearDown(self):
        torch_geometric.backend.use_segment_matmul = self._default_use_segment_matmul
        return super().tearDown()

    def test_hgt_forward(self):
        user_node_type = NodeType("user")
        item_node_type = NodeType("item")
        location_node_type = NodeType("location")
        user_to_user_edge_type = EdgeType(
            user_node_type, Relation("to"), user_node_type
        )
        item_to_user_edge_type = EdgeType(
            item_node_type, Relation("to"), user_node_type
        )
        location_to_item_edge_type = EdgeType(
            location_node_type, Relation("to"), item_node_type
        )

        node_type_to_feat_dim_map = {
            user_node_type: 2,
            item_node_type: 3,
            location_node_type: 4,
        }
        edge_type_to_feat_dim_map = {
            user_to_user_edge_type: 0,
            item_to_user_edge_type: 0,
            location_to_item_edge_type: 0,
        }
        encoder = HGT(
            node_type_to_feat_dim_map=node_type_to_feat_dim_map,
            edge_type_to_feat_dim_map=edge_type_to_feat_dim_map,
            hid_dim=16,
            out_dim=16,
            num_layers=2,
            num_heads=2,
            should_l2_normalize_embedding_layer_output=True,
            feature_embedding_layers=None,
        )
        data = HeteroData()
        # We intentionally initialize the data in the HeteroData object in different orders than the `node_type_to_feat_dim_map` and `edge_type_to_feat_dim_map` to
        # ensure that we can still forward pass in this scenario

        data.x_dict = {
            item_node_type: torch.rand((3, 3)),
            user_node_type: torch.rand((2, 2)),
            location_node_type: torch.rand((4, 4)),
        }
        data.edge_index_dict = {
            location_to_item_edge_type: torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]]),
            user_to_user_edge_type: torch.tensor([[0, 1], [1, 0]]),
            item_to_user_edge_type: torch.tensor([[0, 1, 2], [0, 1, 0]]),
        }

        # Validate that we can forward pass using the mocked data
        result = encoder(
            data=data,
            output_node_types=[user_node_type, item_node_type, location_node_type],
            device=torch.device("cpu"),
        )

        # Assert all requested node types are in the output
        self.assertIn(user_node_type, result)
        self.assertIn(item_node_type, result)
        self.assertIn(location_node_type, result)

        # Assert error if we forward with an edge index dictionary which has different keys than the HGT constructor
        with self.assertRaises(ValueError):
            data.edge_index_dict = {
                location_to_item_edge_type: torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]]),
                user_to_user_edge_type: torch.tensor([[0, 1], [1, 0]]),
            }
            encoder(
                data=data,
                output_node_types=[user_node_type, item_node_type, location_node_type],
                device=torch.device("cpu"),
            )
