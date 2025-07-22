import unittest
from typing import Optional, Union

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, HeteroData

from gigl.module.models import LinkPredictionGNN
from gigl.src.common.models.pyg.heterogeneous import HGT
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    get_process_group_init_method,
)


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Add some dummy params so DDP can work with it.
        # Otherwise, we see the below:
        # RuntimeError: DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
        self._lin = nn.Linear(2, 2)

    def forward(
        self,
        data: Union[Data, HeteroData],
        device: torch.device,
        output_node_types: Optional[list[NodeType]] = None,
    ) -> Union[torch.Tensor, dict[NodeType, torch.Tensor]]:
        if isinstance(data, HeteroData):
            if not output_node_types:
                raise ValueError(
                    "Output node types must be specified for heterogeneous data"
                )
            return {
                node_type: torch.tensor([1.0, 2.0]) for node_type in output_node_types
            }
        else:
            return torch.tensor([1.0, 2.0])


class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Add some dummy params so DDP can work with it.
        # Otherwise, we see the below:
        # RuntimeError: DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
        self._lin = nn.Linear(2, 2)

    def forward(
        self, query_embeddings: torch.Tensor, candidate_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return query_embeddings + candidate_embeddings


class TestLinkPredictionGNN(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_forward_homogeneous(self):
        encoder = DummyEncoder()
        decoder = DummyDecoder()
        model = LinkPredictionGNN(encoder, decoder)
        data = Data()
        result = model.forward(data, self.device)
        assert isinstance(result, torch.Tensor)
        assert_tensor_equality(result, torch.tensor([1.0, 2.0]))

    def test_forward_heterogeneous_with_node_types(self):
        encoder = DummyEncoder()
        decoder = DummyDecoder()
        model = LinkPredictionGNN(encoder, decoder)
        data = HeteroData()
        output_node_types = [NodeType("type1"), NodeType("type2")]
        result = model.forward(data, self.device, output_node_types)
        assert isinstance(result, dict)
        self.assertEqual(set(result.keys()), set(output_node_types))
        for node_type in output_node_types:
            assert_tensor_equality(result[node_type], torch.tensor([1.0, 2.0]))

    def test_forward_heterogeneous_missing_node_types(self):
        encoder = DummyEncoder()
        decoder = DummyDecoder()
        model = LinkPredictionGNN(encoder, decoder)
        data = HeteroData()
        with self.assertRaises(ValueError):
            model.forward(data, self.device)

    def test_decode(self):
        encoder = DummyEncoder()
        decoder = DummyDecoder()
        model = LinkPredictionGNN(encoder, decoder)
        q = torch.tensor([1.0, 2.0])
        c = torch.tensor([3.0, 4.0])
        result = model.decode(q, c)
        assert_tensor_equality(result, torch.tensor([4.0, 6.0]))

    def test_encoder_property(self):
        encoder = DummyEncoder()
        decoder = DummyDecoder()
        model = LinkPredictionGNN(encoder, decoder)
        self.assertIs(model.encoder, encoder)

    def test_decoder_property(self):
        encoder = DummyEncoder()
        decoder = DummyDecoder()
        model = LinkPredictionGNN(encoder, decoder)
        self.assertIs(model.decoder, decoder)

    def test_for_ddp(self):
        torch.distributed.init_process_group(
            rank=0, world_size=1, init_method=get_process_group_init_method()
        )
        self.addCleanup(torch.distributed.destroy_process_group)
        encoder = DummyEncoder()
        decoder = DummyDecoder()
        model = LinkPredictionGNN(encoder, decoder)
        ddp_model = model.to_ddp(self.device, find_unused_encoder_parameters=True)
        self.assertIsInstance(ddp_model, LinkPredictionGNN)
        self.assertIsInstance(ddp_model.encoder, nn.parallel.DistributedDataParallel)
        self.assertIsInstance(ddp_model.decoder, nn.parallel.DistributedDataParallel)
        self.assertTrue(hasattr(ddp_model.encoder, "module"))
        self.assertTrue(hasattr(ddp_model.decoder, "module"))

    def test_unwrap_from_ddp(self):
        torch.distributed.init_process_group(
            rank=0, world_size=1, init_method=get_process_group_init_method()
        )
        self.addCleanup(torch.distributed.destroy_process_group)
        encoder = DummyEncoder()
        decoder = DummyDecoder()
        model = LinkPredictionGNN(encoder, decoder)
        ddp_model = model.to_ddp(self.device, find_unused_encoder_parameters=True)
        unwrapped = ddp_model.unwrap_from_ddp()
        self.assertIs(unwrapped.encoder, encoder)
        self.assertIs(unwrapped.decoder, decoder)


class TestHGT(unittest.TestCase):
    def setUp(self):
        # Set this field to True so that it will be able to raise error if the HGTConv forward pass does not behave as expected
        torch_geometric.backend.use_segment_matmul = True

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


if __name__ == "__main__":
    unittest.main()
