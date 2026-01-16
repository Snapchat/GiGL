import unittest
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn.models import LightGCN as PyGLightGCN
from torchrec.distributed.model_parallel import (
    DistributedModelParallel as DistributedModelParallel,
)

from gigl.nn.models import LightGCN, LinkPredictionGNN
from gigl.src.common.types.graph_data import NodeType
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    create_test_process_group,
    destroy_test_process_group,
    get_process_group_init_method,
)

# Embedding table name for default homogeneous node type
# Constructed as f"node_embedding_{DEFAULT_HOMOGENEOUS_NODE_TYPE}" in LightGCN
DEFAULT_EMBEDDING_TABLE_NAME = f"node_embedding_{DEFAULT_HOMOGENEOUS_NODE_TYPE}"


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

    def tearDown(self):
        # Ensure the process group is destroyed after each test
        # to avoid interference with subsequent tests
        destroy_test_process_group()
        super().tearDown()

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
        create_test_process_group()
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
        create_test_process_group()
        self.addCleanup(destroy_test_process_group)
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
        destroy_test_process_group()
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

    def _set_embeddings(self, model: LightGCN, node_type: NodeType):
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
        self._set_embeddings(model, DEFAULT_HOMOGENEOUS_NODE_TYPE)

        # Forward pass
        output = model(self.data, self.device)

        # Check output shape
        self.assertEqual(output.shape, (self.num_nodes, self.embedding_dim))

    def test_forward_homogeneous_with_anchor_node_ids(self):
        """Test forward pass with homogeneous graph and anchor node ids."""
        node_type_to_num_nodes = self.num_nodes
        model = self._create_lightgcn_model(node_type_to_num_nodes)
        self._set_embeddings(model, DEFAULT_HOMOGENEOUS_NODE_TYPE)
        anchor_node_ids = torch.tensor([0, 1], dtype=torch.long)

        output = model(self.data, self.device, anchor_node_ids=anchor_node_ids)

        self.assertEqual(output.shape, (2, self.embedding_dim))

    def test_compare_with_pyg_reference(self):
        """Test that our implementation matches PyG LightGCN output."""
        # Create our model
        node_type_to_num_nodes = self.num_nodes
        our_model = self._create_lightgcn_model(node_type_to_num_nodes)
        self._set_embeddings(our_model, DEFAULT_HOMOGENEOUS_NODE_TYPE)

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
        self._set_embeddings(our_model, DEFAULT_HOMOGENEOUS_NODE_TYPE)
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
            DEFAULT_EMBEDDING_TABLE_NAME
        ]
        self.assertIsNotNone(embedding_table.weight.grad)
        self.assertTrue(torch.any(embedding_table.weight.grad != 0))

    def test_dmp_multiprocess(self):
        """
        Test DMP with multiple processes to verify embedding sharding works correctly.

        Tests both forward pass (output correctness) and backward pass (gradient flow).

        Note: Uses CPU/Gloo backend for unit testing.
        """
        world_size = 2
        process_group_init_method = get_process_group_init_method()

        # Spawn world_size processes
        mp.spawn(
            fn=_run_dmp_multiprocess_test,
            args=(
                world_size,  # total number of processes
                process_group_init_method,  # initialization method for process group
                self.num_nodes,  # number of nodes in test graph
                self.embedding_dim,  # dimension of embeddings
                self.num_layers,  # number of LightGCN layers
                self.edge_index,  # edge connectivity
                self.test_embeddings,  # test embedding values
                self.expected_output,  # expected model output
            ),
            nprocs=world_size,
        )


def _run_dmp_multiprocess_test(
    rank: int,
    world_size: int,
    process_group_init_method: str,
    num_nodes: int,
    embedding_dim: int,
    num_layers: int,
    edge_index: torch.Tensor,
    test_embeddings: torch.Tensor,
    expected_output: torch.Tensor,
):
    """
    Helper function that runs in each spawned process for multi-process DMP testing.

    Tests both forward pass (output correctness) and backward pass (gradient flow)
    in a multi-process distributed environment.

    Args:
        rank: Rank of this process (0, 1, 2, ...)
        world_size: Total number of processes
        process_group_init_method: Initialization method for process group
        num_nodes: Number of nodes in test graph
        embedding_dim: Dimension of embeddings
        num_layers: Number of LightGCN layers
        edge_index: Edge connectivity
        test_embeddings: Test embedding values
        expected_output: Expected model output
    """
    try:
        # Initialize process group for this rank
        dist.init_process_group(
            backend="gloo",  # Use Gloo for CPU testing
            init_method=process_group_init_method,
            rank=rank,
            world_size=world_size,
        )

        device = torch.device("cpu")  # Use CPU for unit tests
        # Create model
        model = LightGCN(
            node_type_to_num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            device=device,
        )

        # Wrap with DMP - this will shard embeddings across ranks
        dmp_model = DistributedModelParallel(
            module=model,
            device=device,
        )

        # Set embeddings AFTER DMP wrapping
        # Note: On CPU/Gloo, DMP doesn't actually shard - it replicates the full table
        # So we need to set ALL embeddings on each rank
        with torch.no_grad():
            table = model._embedding_bag_collection.embedding_bags[
                DEFAULT_EMBEDDING_TABLE_NAME
            ]
            # Set all embeddings (CPU/Gloo replicates, doesn't shard)
            table.weight[:] = test_embeddings.to(device)

        # Create test data
        data = Data(edge_index=edge_index.to(device), num_nodes=num_nodes)
        data.node = torch.arange(num_nodes, dtype=torch.long, device=device)

        # Test 1: Forward pass - DMP will fetch embeddings across ranks as needed
        with torch.no_grad():
            output = dmp_model(data=data, device=device)

        # Verify output matches expected (all ranks should get same result)
        if not torch.allclose(output, expected_output.to(device), atol=1e-4, rtol=1e-4):
            raise AssertionError(
                f"Rank {rank}: DMP multi-process output doesn't match expected.\n"
                f"Got:\n{output}\nExpected:\n{expected_output.to(device)}"
            )

        # Test 2: Backward pass - verify gradients flow correctly in multi-process DMP
        dmp_model.train()
        output = dmp_model(data=data, device=device)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        embedding_table = model._embedding_bag_collection.embedding_bags[
            DEFAULT_EMBEDDING_TABLE_NAME
        ]
        if embedding_table.weight.grad is None:
            raise AssertionError(
                f"Rank {rank}: Gradients should exist after backward pass"
            )
        if not torch.any(embedding_table.weight.grad != 0):
            raise AssertionError(
                f"Rank {rank}: Gradients should be non-zero after backward pass"
            )

    finally:
        # Cleanup process group for this spawned process
        destroy_test_process_group()


if __name__ == "__main__":
    unittest.main()
