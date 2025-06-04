from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType, Relation
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.mocking.dataset_asset_mocker import DatasetAssetMocker
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.pyg_datasets_forks import CoraFromGCS, DBLPFromGCS
from gigl.src.mocking.lib.versioning import (
    MockedDatasetArtifactMetadata,
    update_mocked_dataset_artifact_metadata,
)

logger = Logger()

_HOMOGENEOUS_TOY_GRAPH_CONFIG = "gigl/src/mocking/mocking_assets/toy_graph_data.yaml"
_BIPARTITE_TOY_GRAPH_CONFIG = (
    "gigl/src/mocking/mocking_assets/bipartite_toy_graph_data.yaml"
)


class DatasetAssetMockingSuite:
    """
    This class houses functions which are used to mock datasets for testing purposes,
    e.g. `mock_cora_homogeneous_supervised_node_classification_dataset`.
    To add a mocking task, create a new function which starts with `mock` and returns
    a MockedDatasetInfo instance.
    """

    @dataclass
    class ToyGraphData:
        node_types: Dict[str, NodeType]
        edge_types: Dict[str, EdgeType]
        node_feats: Dict[str, torch.Tensor]
        edge_indices: Dict[str, torch.Tensor]
        node_labels: Optional[Dict[str, torch.Tensor]] = None
        edge_feats: Optional[Dict[str, torch.Tensor]] = None

    @dataclass
    class UserDefinedLabels:
        pos_edge_index: torch.Tensor
        neg_edge_index: torch.Tensor
        pos_edge_feats: torch.Tensor
        neg_edge_feats: torch.Tensor

    @staticmethod
    def _get_pyg_cora_dataset(
        store_at: str = "/tmp/Cora",
    ) -> Tuple[CoraFromGCS, NodeType, EdgeType]:
        """Cora graph is the graph in the first index in the returned dataset
        i.e. the Planetoid object is subscriptable, data = dataset[0]
        Train and tests masks are defined by `train_mask` and `test_mask`` properties on data.
        Returns:
            torch_geometric.datasets.planetoid.Planetoid
        """
        # Fetch the dataset
        dataset = CoraFromGCS(root=store_at, name="Cora")
        node_type = NodeType("paper")
        edge_type = EdgeType(node_type, Relation("cites"), node_type)
        return dataset[0], node_type, edge_type

    @staticmethod
    def _get_pyg_dblp_dataset(
        store_at: str = "/tmp/DBLP",
    ) -> Tuple[DBLPFromGCS, Dict[str, NodeType], Dict[str, EdgeType]]:
        """DBLP graph is the graph in the first index in the returned dataset.
        https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.DBLP.html
        Detailed description of the dataset:
        HeteroData(
            author={
                x=[4057, 334],
                y=[4057],
                train_mask=[4057],
                val_mask=[4057],
                test_mask=[4057]
            },
            paper={ x=[14328, 4231] },
            term={ x=[7723, 50] },
            conference={ num_nodes=20 },
            (author, to, paper)={ edge_index=[2, 19645] },
            (paper, to, author)={ edge_index=[2, 19645] },
            (paper, to, term)={ edge_index=[2, 85810] },
            (paper, to, conference)={ edge_index=[2, 14328] },
            (term, to, paper)={ edge_index=[2, 85810] },
            (conference, to, paper)={ edge_index=[2, 14328] }
        )
        """
        # Fetch the dataset
        dataset = DBLPFromGCS(root=store_at)[0]
        # here we only use certain node/edge types to simplify the graph.
        node_types = {
            "author": NodeType("author"),
            "paper": NodeType("paper"),
            "term": NodeType("term"),
        }
        edge_types = {
            "author_to_paper": EdgeType(
                node_types["author"], Relation("to"), node_types["paper"]
            ),
            "paper_to_author": EdgeType(
                node_types["paper"], Relation("to"), node_types["author"]
            ),
            "term_to_paper": EdgeType(
                node_types["term"], Relation("to"), node_types["paper"]
            ),
        }
        # add dummy edge features for the edge types we use
        dataset[("author", "to", "paper")].edge_attr = torch.FloatTensor(
            [1, 2, 3, 4, 5]
        ).repeat(dataset[("author", "to", "paper")].num_edges, 1)
        dataset[("paper", "to", "author")].edge_attr = torch.FloatTensor(
            [6, 5, 4, 3, 2, 1]
        ).repeat(dataset[("paper", "to", "author")].num_edges, 1)
        dataset[("term", "to", "paper")].edge_attr = torch.FloatTensor([1, 2]).repeat(
            dataset[("term", "to", "paper")].num_edges, 1
        )
        return dataset, node_types, edge_types

    @staticmethod
    def _generate_mock_pos_neg_edge_indices_and_feats(
        main_edge_indices: torch.Tensor,
        num_pos_per_node: int = 1,
        num_neg_per_node: int = 3,
        is_edge_list_bipartite: bool = False,
    ) -> UserDefinedLabels:
        """
        Sample given number of non-overlapping positive and negative edges
        per anchor (src) node in the given edge index.
        """

        if is_edge_list_bipartite:
            num_anchor_nodes = int(main_edge_indices[0, :].max() + 1)
            num_target_nodes = int(main_edge_indices[1, :].max() + 1)
        else:
            num_anchor_nodes = int(main_edge_indices.max() + 1)
            num_target_nodes = num_anchor_nodes

        pos_edge_index = torch.zeros(
            2, num_pos_per_node * num_anchor_nodes, dtype=torch.long
        )
        neg_edge_index = torch.zeros(
            2, num_neg_per_node * num_anchor_nodes, dtype=torch.long
        )

        pos_idx_counter = 0
        neg_idx_counter = 0
        for anchor_node_id in range(num_anchor_nodes):
            target_nodes = np.random.choice(
                num_target_nodes,
                size=num_pos_per_node + num_neg_per_node,
                replace=False,
            )
            for pos_target_node in target_nodes[:num_pos_per_node]:
                pos_edge_index[0, pos_idx_counter] = anchor_node_id
                pos_edge_index[1, pos_idx_counter] = pos_target_node
                pos_idx_counter += 1
            for neg_target_node in target_nodes[num_pos_per_node:]:
                neg_edge_index[0, neg_idx_counter] = anchor_node_id
                neg_edge_index[1, neg_idx_counter] = neg_target_node
                neg_idx_counter += 1

        pos_edge_feats = torch.FloatTensor([0, 2, 4]).repeat(pos_edge_index.shape[1], 1)
        neg_edge_feats = torch.FloatTensor([1, 3, 5]).repeat(neg_edge_index.shape[1], 1)

        return DatasetAssetMockingSuite.UserDefinedLabels(
            pos_edge_index=pos_edge_index,
            neg_edge_index=neg_edge_index,
            pos_edge_feats=pos_edge_feats,
            neg_edge_feats=neg_edge_feats,
        )

    def mock_cora_homogeneous_supervised_node_classification_dataset(
        self,
    ) -> MockedDatasetInfo:
        data, node_type, edge_type = self._get_pyg_cora_dataset()
        mocked_dataset_info = MockedDatasetInfo(
            name="cora_homogeneous_supervised_node_classification",  # TODO: (svij-sc) These can prolly be enums
            task_metadata_type=TaskMetadataType.NODE_BASED_TASK,
            edge_index={edge_type: data.edge_index},
            node_feats={node_type: data.x},
            node_labels={node_type: data.y},
            sample_node_type=node_type,
        )
        return mocked_dataset_info

    def mock_cora_homogeneous_supervised_node_classification_dataset_with_edge_features(
        self,
    ) -> MockedDatasetInfo:
        data, node_type, edge_type = self._get_pyg_cora_dataset()
        data.edge_attr = torch.FloatTensor([1, 2, 3, 4]).repeat(data.num_edges, 1)
        mocked_dataset_info = MockedDatasetInfo(
            name="cora_homogeneous_supervised_node_classification_edge_features",
            task_metadata_type=TaskMetadataType.NODE_BASED_TASK,
            edge_index={edge_type: data.edge_index},
            node_feats={node_type: data.x},
            edge_feats={edge_type: data.edge_attr},
            node_labels={node_type: data.y},
            sample_node_type=node_type,
        )
        return mocked_dataset_info

    def mock_cora_homogeneous_node_anchor_based_link_prediction_dataset(
        self,
    ) -> MockedDatasetInfo:
        data, node_type, edge_type = self._get_pyg_cora_dataset()
        mocked_dataset_info = MockedDatasetInfo(
            name="cora_homogeneous_node_anchor",
            task_metadata_type=TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK,
            edge_index={edge_type: data.edge_index},
            node_feats={node_type: data.x},
            sample_edge_type=edge_type,
        )
        return mocked_dataset_info

    def mock_cora_homogeneous_node_anchor_based_link_prediction_dataset_with_edge_features(
        self,
    ) -> MockedDatasetInfo:
        data, node_type, edge_type = self._get_pyg_cora_dataset()
        data.edge_attr = torch.FloatTensor([1, 2, 3, 4]).repeat(data.num_edges, 1)
        mocked_dataset_info = MockedDatasetInfo(
            name="cora_homogeneous_node_anchor_edge_features",
            task_metadata_type=TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK,
            edge_index={edge_type: data.edge_index},
            node_feats={node_type: data.x},
            edge_feats={edge_type: data.edge_attr},
            sample_edge_type=edge_type,
        )
        return mocked_dataset_info

    # TODO: (svij-sc) Opportunity to reduce some replication
    # across mocking functions.
    def mock_cora_homogeneous_node_anchor_based_link_prediction_dataset_with_user_defined_labels(
        self,
    ) -> MockedDatasetInfo:
        data, node_type, edge_type = self._get_pyg_cora_dataset()
        data.edge_attr = torch.FloatTensor([1, 2, 3, 4]).repeat(data.num_edges, 1)
        udl = DatasetAssetMockingSuite._generate_mock_pos_neg_edge_indices_and_feats(
            main_edge_indices=data.edge_index,
            num_pos_per_node=3,
            num_neg_per_node=3,
        )
        mocked_dataset_info = MockedDatasetInfo(
            name="cora_homogeneous_node_anchor_edge_features_user_defined_labels",
            task_metadata_type=TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK,
            edge_index={edge_type: data.edge_index},
            node_feats={node_type: data.x},
            edge_feats={edge_type: data.edge_attr},
            sample_edge_type=edge_type,
            user_defined_edge_index={
                edge_type: {
                    EdgeUsageType.POSITIVE: udl.pos_edge_index,
                    EdgeUsageType.NEGATIVE: udl.neg_edge_index,
                }
            },
            user_defined_edge_feats={
                edge_type: {
                    EdgeUsageType.POSITIVE: udl.pos_edge_feats,
                    EdgeUsageType.NEGATIVE: udl.neg_edge_feats,
                }
            },
        )
        return mocked_dataset_info

    def mock_dblp_node_anchor_based_link_prediction_dataset(
        self,
    ) -> MockedDatasetInfo:
        data, node_types, edge_types = self._get_pyg_dblp_dataset()
        mocked_dataset_info = MockedDatasetInfo(
            name="dblp_node_anchor_edge_features_lp",
            task_metadata_type=TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK,  # type: ignore
            edge_index={
                edge_types["author_to_paper"]: data[
                    edge_types["author_to_paper"].tuple_repr()
                ].edge_index,
                edge_types["paper_to_author"]: data[
                    edge_types["paper_to_author"].tuple_repr()
                ].edge_index,
                edge_types["term_to_paper"]: data[
                    edge_types["term_to_paper"].tuple_repr()
                ].edge_index,
            },
            node_feats={
                node_types["author"]: data[node_types["author"]].x,
                node_types["paper"]: data[node_types["paper"]].x,
                node_types["term"]: data[node_types["term"]].x,
            },
            edge_feats={
                edge_types["author_to_paper"]: data[
                    edge_types["author_to_paper"].tuple_repr()
                ].edge_attr,
                edge_types["paper_to_author"]: data[
                    edge_types["paper_to_author"].tuple_repr()
                ].edge_attr,
                edge_types["term_to_paper"]: data[
                    edge_types["term_to_paper"].tuple_repr()
                ].edge_attr,
            },
            sample_edge_type=edge_types["paper_to_author"],
        )
        return mocked_dataset_info

    def mock_dblp_node_anchor_based_link_prediction_dataset_with_user_defined_labels(
        self,
    ) -> MockedDatasetInfo:
        data, node_types, edge_types = self._get_pyg_dblp_dataset()
        udl = DatasetAssetMockingSuite._generate_mock_pos_neg_edge_indices_and_feats(
            main_edge_indices=data[
                edge_types["paper_to_author"].tuple_repr()
            ].edge_index,
            num_pos_per_node=2,
            num_neg_per_node=3,
            is_edge_list_bipartite=True,
        )
        mocked_dataset_info = MockedDatasetInfo(
            name="dblp_node_anchor_edge_features_user_defined_labels",
            task_metadata_type=TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK,  # type: ignore
            edge_index={
                edge_types["author_to_paper"]: data[
                    edge_types["author_to_paper"].tuple_repr()
                ].edge_index,
                edge_types["paper_to_author"]: data[
                    edge_types["paper_to_author"].tuple_repr()
                ].edge_index,
                edge_types["term_to_paper"]: data[
                    edge_types["term_to_paper"].tuple_repr()
                ].edge_index,
            },
            node_feats={
                node_types["author"]: data[node_types["author"]].x,
                node_types["paper"]: data[node_types["paper"]].x,
                node_types["term"]: data[node_types["term"]].x,
            },
            edge_feats={
                edge_types["author_to_paper"]: data[
                    edge_types["author_to_paper"].tuple_repr()
                ].edge_attr,
                edge_types["paper_to_author"]: data[
                    edge_types["paper_to_author"].tuple_repr()
                ].edge_attr,
                edge_types["term_to_paper"]: data[
                    edge_types["term_to_paper"].tuple_repr()
                ].edge_attr,
            },
            sample_edge_type=edge_types["paper_to_author"],
            user_defined_edge_index={
                edge_types["paper_to_author"]: {
                    EdgeUsageType.POSITIVE: udl.pos_edge_index,
                    EdgeUsageType.NEGATIVE: udl.neg_edge_index,
                }
            },
            user_defined_edge_feats={
                edge_types["paper_to_author"]: {
                    EdgeUsageType.POSITIVE: udl.pos_edge_feats,
                    EdgeUsageType.NEGATIVE: udl.neg_edge_feats,
                }
            },
        )
        return mocked_dataset_info

    def _create_custom_toy_graph(self, graph_config):
        with open(graph_config, "r") as f:
            graph_config = yaml.safe_load(f)

        node_config = graph_config["graph"]["node_types"]
        node_types = {node_type: NodeType(node_type) for node_type in node_config}

        edge_config = graph_config["graph"]["edge_types"]
        edge_types = {
            edge_type: EdgeType(
                NodeType(edge_config[edge_type]["src_node_type"]),
                Relation(edge_config[edge_type]["relation_type"]),
                NodeType(edge_config[edge_type]["dst_node_type"]),
            )
            for edge_type in edge_config.keys()
        }

        edge_indices_dict = {}
        for edge_type in edge_config:
            edge_index_list = []
            for adj in graph_config["adj_list"][edge_type]:
                dst_list = adj["dst"]
                edge_index_list.extend([(adj["src"], dst) for dst in dst_list])
            edge_indices_dict[edge_type] = (
                torch.tensor(edge_index_list).t().contiguous()
            )

        node_feats_dict = {}
        for node_type in node_config:
            node_feats_list: List[str] = []
            for node in graph_config["nodes"][node_type]:
                features = node["features"]
                node_feats_list.append(features)
            node_feats_dict[node_type] = torch.tensor(node_feats_list)

        edge_feat_dict = {
            edge_type: edge_indices_dict[edge_type].t() * 0.1
            for edge_type in edge_config
        }  # dummy edge features, st they're just edge_index * 0.1

        return DatasetAssetMockingSuite.ToyGraphData(
            node_types=node_types,
            edge_types=edge_types,
            node_feats=node_feats_dict,
            edge_indices=edge_indices_dict,
            edge_feats=edge_feat_dict,
        )

    def mock_toy_graph_homogeneous_node_anchor_based_link_prediction_dataset(
        self,
    ) -> MockedDatasetInfo:
        toy_data = self._create_custom_toy_graph(
            graph_config=_HOMOGENEOUS_TOY_GRAPH_CONFIG
        )
        mocked_dataset_info = MockedDatasetInfo(
            name="toy_graph_homogeneous_node_anchor_lp",
            task_metadata_type=TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK,
            edge_index={
                edge_type: toy_data.edge_indices[edge_type_str]
                for edge_type_str, edge_type in toy_data.edge_types.items()
            },
            node_feats={
                node_type: toy_data.node_feats[node_type_str]
                for node_type_str, node_type in toy_data.node_types.items()
            },
            edge_feats={
                edge_type: toy_data.edge_feats[edge_type_str]
                for edge_type_str, edge_type in toy_data.edge_types.items()
            },
            sample_edge_type=list(toy_data.edge_types.values())[0],
        )
        return mocked_dataset_info

    def mock_toy_graph_homogeneous_node_anchor_based_link_prediction_with_user_def_labels_dataset(
        self,
    ) -> MockedDatasetInfo:
        toy_data = self._create_custom_toy_graph(
            graph_config=_HOMOGENEOUS_TOY_GRAPH_CONFIG
        )

        pos_edge_index = torch.tensor(
            [
                [1, 2, 4, 5, 6, 10, 11, 15, 20, 22, 23],
                [0, 24, 11, 20, 14, 16, 8, 18, 5, 24, 20],
            ]
        )
        neg_edge_index = torch.tensor(
            [
                [0, 1, 2, 4, 6, 10, 12, 13, 16, 16, 18, 20, 22, 23],
                [7, 2, 14, 14, 24, 23, 9, 15, 11, 14, 21, 3, 7, 9],
            ]
        )

        pos_edge_feats = torch.FloatTensor([0, 2, 4]).repeat(11, 1)
        neg_edge_feats = torch.FloatTensor([1, 3, 5]).repeat(14, 1)

        udl = DatasetAssetMockingSuite.UserDefinedLabels(
            pos_edge_index=pos_edge_index,
            neg_edge_index=neg_edge_index,
            pos_edge_feats=pos_edge_feats,
            neg_edge_feats=neg_edge_feats,
        )
        mocked_dataset_info = MockedDatasetInfo(
            name="toy_graph_homogeneous_node_anchor_lp_user_defined_edges",
            task_metadata_type=TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK,
            edge_index={
                edge_type: toy_data.edge_indices[edge_type_str]
                for edge_type_str, edge_type in toy_data.edge_types.items()
            },
            node_feats={
                node_type: toy_data.node_feats[node_type_str]
                for node_type_str, node_type in toy_data.node_types.items()
            },
            edge_feats={
                edge_type: toy_data.edge_feats[edge_type_str]
                for edge_type_str, edge_type in toy_data.edge_types.items()
            },
            sample_edge_type=list(toy_data.edge_types.values())[0],
            user_defined_edge_index={
                list(toy_data.edge_types.values())[0]: {
                    EdgeUsageType.POSITIVE: udl.pos_edge_index,
                    EdgeUsageType.NEGATIVE: udl.neg_edge_index,
                }
            },
            user_defined_edge_feats={
                list(toy_data.edge_types.values())[0]: {
                    EdgeUsageType.POSITIVE: udl.pos_edge_feats,
                    EdgeUsageType.NEGATIVE: udl.neg_edge_feats,
                }
            },
        )
        return mocked_dataset_info

    def mock_toy_graph_heterogeneous_node_anchor_based_link_prediction_dataset(
        self,
    ) -> MockedDatasetInfo:
        toy_data = self._create_custom_toy_graph(
            graph_config=_BIPARTITE_TOY_GRAPH_CONFIG
        )
        mocked_dataset_info = MockedDatasetInfo(
            name="toy_graph_heterogeneous_node_anchor_lp",
            task_metadata_type=TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK,
            edge_index={
                edge_type: toy_data.edge_indices[edge_type_str]
                for edge_type_str, edge_type in toy_data.edge_types.items()
            },
            node_feats={
                node_type: toy_data.node_feats[node_type_str]
                for node_type_str, node_type in toy_data.node_types.items()
            },
            edge_feats={
                edge_type: toy_data.edge_feats[edge_type_str]
                for edge_type_str, edge_type in toy_data.edge_types.items()
            },
            sample_edge_type=toy_data.edge_types["user_to_story"],
        )
        return mocked_dataset_info

    def __init__(self):
        self.mocked_datasets: Dict[str, MockedDatasetInfo] = dict()
        mocking_func_names: List[str] = [
            attr
            for attr in dir(self)
            if callable(getattr(self, attr)) and attr.startswith("mock")
        ]
        mocking_funcs = [getattr(self, attr) for attr in mocking_func_names]
        logger.debug("Registering mocked datasets...")

        mocked_dataset_info: MockedDatasetInfo
        for mocking_func in mocking_funcs:
            logger.debug(f"\t- {mocking_func.__name__}")
            mocked_dataset_info = mocking_func()
            self.mocked_datasets[mocked_dataset_info.name] = mocked_dataset_info


mocked_datasets = DatasetAssetMockingSuite().mocked_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allows mocking of dataset assets.")
    parser.add_argument(
        "--select",
        help=f"The name attribute of individual {MockedDatasetInfo.__name__} instances",
        required=False,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--resource_config_uri",
        help="resource config is needed to run",
        required=True,
    )
    parser.add_argument(
        "--version",
        help="version identifier for the mocked dataset",
        required=False,
        default=current_formatted_datetime(),
    )
    args, _ = parser.parse_known_args()

    if args.select:
        mocked_datasets = {k: v for k, v in mocked_datasets.items() if k in args.select}

    logger.info(f"Will generate mocked data with version {args.version}")
    logger.info(f"Will run {len(mocked_datasets)} mocking funcs:")

    mocker = DatasetAssetMocker()
    for mocked_dataset_name, mocked_dataset_info in mocked_datasets.items():
        logger.info(f"Mocking {mocked_dataset_name}...")
        mocked_dataset_info.version = args.version
        frozen_gbml_config_uri = mocker.mock_assets(
            mocked_dataset_info=mocked_dataset_info
        )
        logger.info(f"Completed mocking {mocked_dataset_name}.")

        # Update version in the mocked dataset version tracker.
        artifact_metadata = MockedDatasetArtifactMetadata(
            version=args.version, frozen_gbml_config_uri=frozen_gbml_config_uri
        )
        logger.info(f"Updating version of {mocked_dataset_name} to {args.version}...")
        update_mocked_dataset_artifact_metadata(
            task_name_to_artifact_metadata={mocked_dataset_name: artifact_metadata}
        )
