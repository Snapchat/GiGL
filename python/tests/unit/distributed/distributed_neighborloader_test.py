import unittest
from collections import abc
from typing import MutableMapping

import graphlearn_torch as glt
import torch
import torch.distributed.rpc
from parameterized import param, parameterized
from torch.multiprocessing import Manager
from torch_geometric.data import Data, HeteroData

from gigl.distributed.dataset_factory import build_dataset_from_task_config_uri
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.distributed.distributed_neighborloader import (
    DistABLPLoader,
    DistNeighborLoader,
)
from gigl.src.common.types.graph_data import NodeType
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
    DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    GraphPartitionData,
    PartitionOutput,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    to_heterogeneous_node,
    to_homogeneous,
)
from tests.test_assets.distributed.run_distributed_dataset import (
    run_distributed_dataset,
)
from tests.test_assets.distributed.utils import assert_tensor_equality

_POSITIVE_EDGE_TYPE = message_passing_to_positive_label(DEFAULT_HOMOGENEOUS_EDGE_TYPE)
_NEGATIVE_EDGE_TYPE = message_passing_to_negative_label(DEFAULT_HOMOGENEOUS_EDGE_TYPE)


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

    @parameterized.expand(
        [
            param(
                "Positive and Negative edges",
                labeled_edges={
                    _POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]]),
                    _NEGATIVE_EDGE_TYPE: torch.tensor(
                        [[10, 10, 11, 15], [13, 16, 14, 17]]
                    ),
                },
                expected_node=torch.tensor([10, 11, 12, 13, 14, 15, 16, 17]),
                expected_srcs=torch.tensor([10, 10, 15, 15, 16, 16, 11, 11]),
                expected_dsts=torch.tensor([11, 12, 13, 14, 12, 14, 13, 17]),
                expected_positive_labels={
                    10: torch.tensor([15]),
                    15: torch.tensor([16]),
                },
                expected_negative_labels={
                    10: torch.tensor([13, 16]),
                    15: torch.tensor([17]),
                },
            ),
            param(
                "Positive edges",
                labeled_edges={_POSITIVE_EDGE_TYPE: torch.tensor([[10, 15], [15, 16]])},
                expected_node=torch.tensor([10, 11, 12, 13, 14, 15, 16, 17]),
                expected_srcs=torch.tensor([10, 10, 15, 15, 16, 16, 11, 11]),
                expected_dsts=torch.tensor([11, 12, 13, 14, 12, 14, 13, 17]),
                expected_positive_labels={
                    10: torch.tensor([15]),
                    15: torch.tensor([16]),
                },
                expected_negative_labels=None,
            ),
        ]
    )
    # TODO: (mkolodner-sc) - Re-enable this test once ports are dynamically inferred
    @unittest.skip("Failing due to ports being already allocated - skiping for now")
    def test_ablp_dataloader(
        self,
        _,
        labeled_edges,
        expected_node,
        expected_srcs,
        expected_dsts,
        expected_positive_labels,
        expected_negative_labels,
    ):
        # Graph looks like https://is.gd/w2oEVp:
        # Message passing
        # 10 -> {11, 12}
        # 11 -> {13, 17}
        # 15 -> {13, 14}
        # 16 -> {12, 14}
        # Positive labels
        # 10 -> 15
        # 15 -> 16
        # Negative labels
        # 10 -> {13, 16}
        # 11 -> 14

        edge_index = {
            DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor(
                [
                    [10, 10, 11, 11, 15, 15, 16, 16],
                    [11, 12, 13, 17, 13, 14, 12, 14],
                ]
            ),
        }
        edge_index.update(labeled_edges)

        partition_output = PartitionOutput(
            node_partition_book=to_heterogeneous_node(torch.zeros(18)),
            edge_partition_book={
                e_type: torch.zeros(int(e_idx.max().item() + 1))
                for e_type, e_idx in edge_index.items()
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

        loader = DistABLPLoader(
            dataset=dataset,
            num_neighbors=[2, 2],
            input_nodes=torch.tensor([10, 15]),
            batch_size=2,
            context=self._context,
            local_process_rank=0,
            local_process_world_size=1,
        )

        count = 0
        for datum in loader:
            self.assertIsInstance(datum, Data)
            count += 1

        self.assertEqual(count, 1)
        dsts, srcs, *_ = datum.coo()
        assert_tensor_equality(
            datum.node,
            expected_node,
            dim=0,
        )
        self.assertEqual(datum.y_positive.keys(), expected_positive_labels.keys())
        for anchor in expected_positive_labels.keys():
            assert_tensor_equality(
                datum.y_positive[anchor],
                expected_positive_labels[anchor],
            )
        if expected_negative_labels is not None:
            self.assertEqual(datum.y_negative.keys(), expected_negative_labels.keys())
            for anchor in expected_negative_labels.keys():
                assert_tensor_equality(
                    datum.y_negative[anchor],
                    expected_negative_labels[anchor],
                )
        else:
            self.assertFalse(hasattr(datum, "y_negative"))
        dsts, srcs, *_ = datum.coo()
        assert_tensor_equality(datum.node[srcs], expected_srcs)
        assert_tensor_equality(datum.node[dsts], expected_dsts)

    # TODO: (mkolodner-sc) - Re-enable this test once ports are dynamically inferred
    @unittest.skip("Failing due to ports being already allocated - skiping for now")
    def test_cora_supervised(self):
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        dataset = build_dataset_from_task_config_uri(
            task_config_uri=cora_supervised_info.frozen_gbml_config_uri.uri,
            distributed_context=self._context,
            is_inference=False,
        )
        loader = DistABLPLoader(
            dataset=dataset,
            num_neighbors=[2, 2],
            input_nodes=to_homogeneous(dataset.train_node_ids),
            context=self._context,
            local_process_rank=0,
            local_process_world_size=1,
        )
        count = 0
        for datum in loader:
            self.assertIsInstance(datum, Data)
            self.assertTrue(hasattr(datum, "y_positive"))
            self.assertIsInstance(datum.y_positive, dict)
            self.assertTrue(hasattr(datum, "y_negative"))
            self.assertIsInstance(datum.y_negative, dict)
            self.assertEqual(datum.y_positive.keys(), datum.y_negative.keys())
            count += 1
        self.assertEqual(count, 2161)


if __name__ == "__main__":
    unittest.main()
