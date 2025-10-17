import unittest
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn.models import LightGCN as PyGLightGCN
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP

from gigl.module.models import LightGCN, LinkPredictionGNN
from gigl.src.common.types.graph_data import NodeType
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


# TODO(swong3): Move create model and graph data in individual tests, rather than using a method to do so
class TestLightGCN(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

        # Test parameters
        self.num_nodes = 4
        self.embedding_dim = 4
        self.num_layers = 2

        # Create test edge index (undirected graph)
        # Graph diagram: https://is.gd/1QuU4J
        self.edge_index = torch.tensor(
            [
                [0, 0, 1, 2, 3, 3],
                [2, 3, 3, 0, 0, 1],
            ],
            dtype=torch.long,
        )

        # Create test data
        self.data = Data(edge_index=self.edge_index, num_nodes=self.num_nodes)
        self.data.node = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        # Fixed embedding weights for reproducible testing
        self.test_embeddings = torch.tensor(
            [
                [0.2, 0.5, 0.1, 0.4],  # Node 0
                [0.6, 0.1, 0.2, 0.5],  # Node 1
                [0.9, 0.4, 0.1, 0.4],  # Node 2
                [0.3, 0.8, 0.3, 0.6],  # Node 3
            ],
            dtype=torch.float32,
        )

        self.expected_output = torch.tensor(
            [
                [0.4495, 0.5311, 0.1555, 0.4865],  # Node 0
                [0.3943, 0.2975, 0.1825, 0.4386],  # Node 1
                [0.5325, 0.4121, 0.1089, 0.3650],  # Node 2
                [0.4558, 0.6207, 0.2506, 0.5817],  # Node 3
            ],
            dtype=torch.float32,
        )

    def tearDown(self):
        """Clean up distributed process group after each test."""
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()

    def _create_lightgcn_model(
        self, node_type_to_num_nodes: Union[int, dict[NodeType, int]]
    ) -> LightGCN:
        """Create a LightGCN model with the specified configuration."""
        return LightGCN(
            node_type_to_num_nodes=node_type_to_num_nodes,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            device=self.device,
        )

    def _set_embeddings(self, model: LightGCN, node_type: str):
        """Set the embedding weights for the model to match test data."""
        with torch.no_grad():
            table = model._embedding_bag_collection.embedding_bags[
                f"node_embedding_{node_type}"
            ]
            table.weight[:] = self.test_embeddings

    def _create_pyg_reference(self) -> PyGLightGCN:
        ref = PyGLightGCN(
            num_nodes=self.num_nodes,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
        ).to(
            self.device
        )  # <<< move model to device

        with torch.no_grad():
            ref.embedding.weight[:] = self.test_embeddings.to(
                self.device
            )  # <<< set on device
        return ref

    def test_forward_homogeneous(self):
        """Test forward pass with homogeneous graph."""
        node_type_to_num_nodes = self.num_nodes
        model = self._create_lightgcn_model(node_type_to_num_nodes)
        self._set_embeddings(model, "default_homogeneous_node_type")

        # Forward pass
        output = model(self.data, self.device)

        # Check output shape
        self.assertEqual(output.shape, (self.num_nodes, self.embedding_dim))

    def test_forward_homogeneous_with_anchor_node_ids(self):
        """Test forward pass with homogeneous graph and anchor node ids."""
        node_type_to_num_nodes = self.num_nodes
        model = self._create_lightgcn_model(node_type_to_num_nodes)
        self._set_embeddings(model, "default_homogeneous_node_type")
        anchor_node_ids = torch.tensor([0, 1], dtype=torch.long)

        output = model(self.data, self.device, anchor_node_ids=anchor_node_ids)

        self.assertEqual(output.shape, (2, self.embedding_dim))

    def test_compare_with_pyg_reference(self):
        """Test that our implementation matches PyG LightGCN output."""
        # Create our model
        node_type_to_num_nodes = self.num_nodes
        our_model = self._create_lightgcn_model(node_type_to_num_nodes)
        self._set_embeddings(our_model, "default_homogeneous_node_type")

        # Create PyG reference model
        pyg_model = self._create_pyg_reference()
        with torch.no_grad():
            our_output = our_model(self.data, self.device)
            pyg_output = pyg_model.get_embedding(
                self.edge_index.to(self.device)
            )  # <<< edge_index on device
        assert_tensor_equality(our_output, pyg_output)

    def test_compare_with_math(self):
        """Test that our implementation matches the mathematical formulation of LightGCN."""
        node_type_to_num_nodes = self.num_nodes
        our_model = self._create_lightgcn_model(node_type_to_num_nodes)
        self._set_embeddings(our_model, "default_homogeneous_node_type")
        output = our_model(self.data, self.device)

        self.assertTrue(
            torch.allclose(output, self.expected_output, atol=1e-4, rtol=1e-4)
        )

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        node_type_to_num_nodes = self.num_nodes
        model = self._create_lightgcn_model(node_type_to_num_nodes)

        model.train()
        output = model(self.data, self.device)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for embedding parameters
        embedding_table = model._embedding_bag_collection.embedding_bags[
            "node_embedding_default_homogeneous_node_type"
        ]
        self.assertIsNotNone(embedding_table.weight.grad)
        self.assertTrue(torch.any(embedding_table.weight.grad != 0))

    def test_dmp_wrapped_model_produces_correct_output(self):
        """
        Test that DMP-wrapped LightGCN produces the same output as non-wrapped model. Note: We only test with a single process for unit test.
        """
        # Initialize distributed
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method=get_process_group_init_method(),
                rank=0,
                world_size=1,
            )

        # Create model
        model = self._create_lightgcn_model(self.num_nodes)

        # Wrap with DMP
        dmp_model = DMP(
            module=model,
            device=self.device,
        )

        # Set embeddings AFTER DMP wrapping (required for CPU/Gloo)
        self._set_embeddings(model, "default_homogeneous_node_type")

        # Run forward pass on DMP-wrapped model
        with torch.no_grad():
            output = dmp_model(data=self.data, device=self.device)

        # Verify output matches expected values
        self.assertTrue(
            torch.allclose(output, self.expected_output, atol=1e-4, rtol=1e-4),
            f"DMP output doesn't match expected.\nGot:\n{output}\nExpected:\n{self.expected_output}",
        )

    def test_dmp_gradient_flow(self):
        """
        Test that gradients flow properly through DMP-wrapped model.
        """
        from torchrec.distributed.model_parallel import DistributedModelParallel as DMP

        # Initialize distributed
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method=get_process_group_init_method(),
                rank=0,
                world_size=1,
            )

        # Create and wrap model
        model = self._create_lightgcn_model(self.num_nodes)

        dmp_model = DMP(
            module=model,
            device=self.device,
        )

        self._set_embeddings(model, "default_homogeneous_node_type")

        model.train()

        # Forward and backward pass
        output = dmp_model(data=self.data, device=self.device)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        embedding_table = model._embedding_bag_collection.embedding_bags[
            "node_embedding_default_homogeneous_node_type"
        ]
        self.assertIsNotNone(
            embedding_table.weight.grad,
            "Gradients should exist after backward pass",
        )
        self.assertTrue(
            torch.any(embedding_table.weight.grad != 0),
            "Gradients should be non-zero",
        )

if __name__ == "__main__":
    unittest.main()
