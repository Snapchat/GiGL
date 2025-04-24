import unittest
from collections import abc, defaultdict
from typing import MutableMapping, Optional

import graphlearn_torch as glt
import torch
import torch.multiprocessing as mp
from graphlearn_torch.data import Feature, Graph
from parameterized import param, parameterized
from torch.multiprocessing import Manager
from torch.testing import assert_close

from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
    TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
    TOY_GRAPH_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
)
from gigl.utils.data_splitters import (
    HashedNodeAnchorLinkSplitter,
    NodeAnchorLinkSplitter,
)
from tests.test_assets.distributed.run_distributed_dataset import (
    run_distributed_dataset,
)


class DistDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self._master_ip_address = "localhost"
        self._world_size = 2
        self._num_rpc_threads = 4

    @parameterized.expand(
        [
            param(
                "Test GLT Dataset Load in sequence with homogeneous toy NABLP dataset",
                mocked_dataset_info=TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
                should_load_tensors_in_parallel=False,
                is_heterogeneous=False,
            ),
            param(
                "Test GLT Dataset Load in sequence with homogeneous toy dataset with user defined labels",
                mocked_dataset_info=TOY_GRAPH_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
                should_load_tensors_in_parallel=False,
                is_heterogeneous=False,
            ),
            param(
                "Test GLT Dataset Load in sequence with heterogeneous toy dataset",
                mocked_dataset_info=HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
                should_load_tensors_in_parallel=False,
                is_heterogeneous=True,
            ),
            param(
                "Test GLT Dataset Load in parallel with heterogeneous toy dataset",
                mocked_dataset_info=HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
                should_load_tensors_in_parallel=True,
                is_heterogeneous=True,
            ),
        ]
    )
    def test_dataset_correctness(
        self,
        _,
        mocked_dataset_info: MockedDatasetInfo,
        should_load_tensors_in_parallel: bool,
        is_heterogeneous: bool,
    ) -> None:
        master_port = glt.utils.get_free_port(self._master_ip_address)
        manager = Manager()
        output_dict: MutableMapping[int, DistLinkPredictionDataset] = manager.dict()

        mp.spawn(
            run_distributed_dataset,
            args=(
                self._world_size,
                mocked_dataset_info,
                output_dict,
                should_load_tensors_in_parallel,
                self._master_ip_address,
                master_port,
            ),
            nprocs=self._world_size,
            join=True,
        )

        for dataset in output_dict.values():
            graph = dataset.graph
            node_ids = dataset.node_ids
            node_features = dataset.node_features
            edge_features = dataset.edge_features

            if is_heterogeneous:
                assert isinstance(node_features, abc.Mapping)
                assert isinstance(node_ids, abc.Mapping)
                assert isinstance(graph, abc.Mapping)
                assert isinstance(edge_features, abc.Mapping)
            else:
                assert isinstance(node_features, Feature)
                assert isinstance(node_ids, torch.Tensor)
                assert isinstance(graph, Graph)
                assert isinstance(edge_features, Feature)
                node_features = {DEFAULT_HOMOGENEOUS_NODE_TYPE: node_features}
                node_ids = {DEFAULT_HOMOGENEOUS_NODE_TYPE: node_ids}
                edge_features = {DEFAULT_HOMOGENEOUS_EDGE_TYPE: edge_features}
                graph = {DEFAULT_HOMOGENEOUS_EDGE_TYPE: graph}

            # id2index is a tensor that is used to get the map the global node/edge ids to the local indices in the feature tensors.
            # As a result, we ensure that all global ids are indexable in the id2index tensor and that all local indices are indexable from the id2index tensor.

            # Validating Node Correctness
            for node_type in node_features:
                # We use lazy_init_with_ipc_handle() to populate the feature_tensor and id2index fields
                node_features[node_type].lazy_init_with_ipc_handle()
                # The max node id + 1 should be equal to the shape of the id2index tensor. We add one since ids are 0-indexed.
                self.assertEqual(
                    torch.max(node_ids[node_type]).item() + 1,
                    node_features[node_type].id2index.size(0),
                )
                # We ensure that each local index in node_features is indexable from id2index
                for local_index in range(node_features[node_type].size(0)):
                    self.assertTrue(local_index in node_features[node_type].id2index)

            # Validating Edge Correctness
            for edge_type in edge_features:
                # We use lazy_init_with_ipc_handle() to populate the feature_tensor and id2index fields
                edge_features[edge_type].lazy_init_with_ipc_handle()
                # The max edge id + 1 should be equal to the shape of the id2index tensor. We add one since ids are 0-indexed.
                self.assertEqual(
                    torch.max(graph[edge_type].topo.edge_ids).item() + 1,
                    edge_features[edge_type].id2index.size(0),
                )
                # We ensure that each local index in edge_features is indexable from id2index
                for local_index in range(edge_features[edge_type].size(0)):
                    self.assertTrue(local_index in edge_features[edge_type].id2index)

    @parameterized.expand(
        [
            param(
                "Test GLT Dataset Split with heterogeneous toy dataset",
                mocked_dataset_info=TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
                is_heterogeneous=False,
                split_fn=HashedNodeAnchorLinkSplitter(sampling_direction="out"),
            ),
            param(
                "Test GLT Dataset Load in parallel with homogeneous toy NABLP dataset",
                mocked_dataset_info=HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
                is_heterogeneous=True,
                split_fn=HashedNodeAnchorLinkSplitter(
                    sampling_direction="out",
                    edge_types=EdgeType(
                        NodeType("story"), Relation("to"), NodeType("user")
                    ),
                ),
            ),
            param(
                "Test GLT Dataset Load in parallel with homogeneous toy NABLP dataset - two supervision edge types",
                mocked_dataset_info=HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
                is_heterogeneous=True,
                split_fn=HashedNodeAnchorLinkSplitter(
                    sampling_direction="out",
                    num_test=1,
                    num_val=1,
                    edge_types=[
                        EdgeType(NodeType("story"), Relation("to"), NodeType("user")),
                        EdgeType(NodeType("user"), Relation("to"), NodeType("story")),
                    ],
                ),
            ),
        ]
    )
    def test_split_dataset_correctness(
        self,
        _,
        mocked_dataset_info: MockedDatasetInfo,
        is_heterogeneous: bool,
        split_fn: Optional[NodeAnchorLinkSplitter],
    ) -> None:
        master_port = glt.utils.get_free_port(self._master_ip_address)
        manager = Manager()
        output_dict: MutableMapping[int, DistLinkPredictionDataset] = manager.dict()

        mp.spawn(
            run_distributed_dataset,
            args=(
                self._world_size,
                mocked_dataset_info,
                output_dict,
                True,  # should_load_tensors_in_parallel
                self._master_ip_address,
                master_port,
                None,  # partitioner
                None,  # dataset
                split_fn,
            ),
            nprocs=self._world_size,
            join=True,
        )

        node_ids_by_rank_by_type: dict[NodeType, dict[int, torch.Tensor]] = defaultdict(
            dict
        )
        for rank, dataset in output_dict.items():
            node_ids = dataset.node_ids
            if is_heterogeneous:
                assert isinstance(node_ids, abc.Mapping)
            else:
                assert isinstance(node_ids, torch.Tensor)
                node_ids = {DEFAULT_HOMOGENEOUS_NODE_TYPE: node_ids}

            for node_type, node_ids_tensor in node_ids.items():
                node_ids_by_rank_by_type[node_type][rank] = node_ids_tensor

        # Assert that the node ids are disjoint across all ranks:
        for node_type, node_ids_by_rank in node_ids_by_rank_by_type.items():
            all_node_ids = torch.cat(list(node_ids_by_rank.values()))
            unique_node_ids = torch.unique(all_node_ids)
            with self.subTest(f"Node type disjointness for {node_type}"):
                # Check that all node ids are unique across ranks
                assert_close(all_node_ids.msort(), unique_node_ids.msort())


if __name__ == "__main__":
    unittest.main()
