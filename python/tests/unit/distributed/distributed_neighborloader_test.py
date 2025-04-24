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
from tests.test_assets.distributed.run_distributed_dataset import (
    run_distributed_dataset,
)


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
