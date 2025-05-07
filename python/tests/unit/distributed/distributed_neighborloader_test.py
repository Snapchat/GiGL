import unittest
from collections import abc
from typing import MutableMapping

import graphlearn_torch as glt
import torch
import torch.distributed.rpc
from torch.multiprocessing import Manager
from torch_geometric.data import Data, HeteroData

from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.src.common.types.graph_data import NodeType
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
    DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    GraphPartitionData,
    PartitionOutput,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    to_heterogeneous_node,
)
from tests.test_assets.distributed.run_distributed_dataset import (
    run_distributed_dataset,
)
from tests.test_assets.distributed.utils import assert_tensor_equality


class DistributedNeighborLoaderTest(unittest.TestCase):
    def setUp(self):
        self._master_ip_address = "localhost"
        self._world_size = 1
        self._num_rpc_threads = 4

        self._context = DistributedContext(
            main_worker_ip_address=self._master_ip_address,
            global_rank=0,
            global_world_size=self._world_size,
        )

    def test_distributed_neighbor_loader(self):
        master_port = glt.utils.get_free_port(self._master_ip_address)
        manager = Manager()
        output_dict: MutableMapping[int, DistLinkPredictionDataset] = manager.dict()

        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
            output_dict=output_dict,
            should_load_tensors_in_parallel=True,
            master_ip_address=self._master_ip_address,
            master_port=master_port,
        )

        loader = DistNeighborLoader(
            dataset=dataset,
            num_neighbors=[2, 2],
            context=self._context,
            local_process_rank=0,
            local_process_world_size=1,
            pin_memory_device=torch.device("cpu"),
        )

        count = 0
        for datum in loader:
            self.assertIsInstance(datum, Data)
            count += 1

        # Cora has 2708 nodes, make sure we go over all of them.
        # https://paperswithcode.com/dataset/cora
        self.assertEqual(count, 2708)

    def test_distributed_neighbor_loader_batched(self):
        node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
        positive_edge_type = message_passing_to_positive_label(
            DEFAULT_HOMOGENEOUS_EDGE_TYPE
        )
        negative_edge_type = message_passing_to_negative_label(
            DEFAULT_HOMOGENEOUS_EDGE_TYPE
        )
        edge_index = {
            DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor(
                [
                    [10, 10, 11, 11, 12, 12, 13, 13],
                    [11, 12, 12, 13, 14, 11, 14, 11],
                ]
            ),
        }
        partition_output = PartitionOutput(
            node_partition_book=to_heterogeneous_node(torch.zeros(14)),
            edge_partition_book={
                DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.zeros(6),
                positive_edge_type: torch.zeros(3),
                negative_edge_type: torch.zeros(3),
            },
            partitioned_edge_index={
                etype: GraphPartitionData(
                    edge_index=idx, edge_ids=torch.arange(idx.size(1))
                )
                for etype, idx in edge_index.items()
            },
            partitioned_edge_features=None,
            partitioned_node_features=None,
            partitioned_negative_labels=None,
            partitioned_positive_labels=None,
        )
        dataset = DistLinkPredictionDataset(rank=0, world_size=1, edge_dir="out")
        dataset.build(partition_output=partition_output)

        loader = DistNeighborLoader(
            dataset=dataset,
            num_neighbors=[2],
            input_nodes=(node_type, torch.tensor([[10, 12]])),
            context=self._context,
            local_process_rank=0,
            local_process_world_size=1,
        )
        count = 0
        for datum in loader:
            self.assertIsInstance(datum, HeteroData)
            count += 1

        self.assertEqual(count, 1)
        assert_tensor_equality(
            datum[node_type].node, torch.tensor([10, 11, 12, 14]), dim=0
        )
        assert_tensor_equality(datum[node_type].batch, torch.tensor([10, 12]), dim=0)

    # TODO: (svij) - Figure out why this test is failing on Google Cloud Build
    @unittest.skip("Failing on Google Cloud Build - skiping for now")
    def test_distributed_neighbor_loader_heterogeneous(self):
        master_port = glt.utils.get_free_port(self._master_ip_address)
        manager = Manager()
        output_dict: MutableMapping[int, DistLinkPredictionDataset] = manager.dict()

        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            output_dict=output_dict,
            should_load_tensors_in_parallel=True,
            master_ip_address=self._master_ip_address,
            master_port=master_port,
        )

        assert isinstance(dataset.node_ids, abc.Mapping)
        loader = DistNeighborLoader(
            dataset=dataset,
            input_nodes=(NodeType("author"), dataset.node_ids[NodeType("author")]),
            num_neighbors=[2, 2],
            context=self._context,
            local_process_rank=0,
            local_process_world_size=1,
            pin_memory_device=torch.device("cpu"),
        )

        count = 0
        for datum in loader:
            self.assertIsInstance(datum, HeteroData)
            count += 1

        self.assertEqual(count, 4057)


if __name__ == "__main__":
    unittest.main()
