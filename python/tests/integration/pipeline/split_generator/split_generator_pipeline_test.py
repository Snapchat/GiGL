from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from typing import List, Set, Tuple, Type, TypeVar, cast

import numpy as np

from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.constants import SPARK_35_TFRECORD_JAR_LOCAL_PATH
from gigl.common.logger import Logger
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.local_fs import get_project_root_directory
from gigl.src.common.graph_builder.graph_builder_factory import GraphBuilderFactory
from gigl.src.common.graph_builder.pyg_graph_builder import PygGraphBuilder
from gigl.src.common.graph_builder.pyg_graph_data import PygGraphData
from gigl.src.common.types.graph_data import EdgeUsageType
from gigl.src.common.types.model import GraphBackend
from gigl.src.common.types.pb_wrappers.dataset_metadata_utils import _get_tfrecord_uris
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.task_metadata import TaskMetadataPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
    TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.src.training.v1.lib.data_loaders.tf_records_iterable_dataset import (
    TfRecordsIterableDataset,
)
from snapchat.research.gbml import (
    dataset_metadata_pb2,
    gbml_config_pb2,
    training_samples_schema_pb2,
)
from tests.integration.pipeline.split_generator.lib import (
    node_anchor_based_link_prediction,
    supervised_node_classification,
)
from tests.integration.pipeline.utils import (
    get_gcs_assets_dir_from_frozen_gbml_config_uri,
)

logger = Logger()

TSample = TypeVar(
    "TSample",
    training_samples_schema_pb2.SupervisedNodeClassificationSample,
    training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
    training_samples_schema_pb2.RootedNodeNeighborhood,
    training_samples_schema_pb2.SupervisedLinkBasedTaskSample,
)  # Must be exactly one of the protos above


def are_dataset_split_sets_disjoint(train: Set, val: Set, test: Set) -> bool:
    are_sets_disjoint = (
        train.isdisjoint(val) and val.isdisjoint(test) and train.isdisjoint(test)
    )
    return are_sets_disjoint


class SplitGeneratorPipelineTest(unittest.TestCase):
    """
    This test checks the completion of split generator pipeline with the Cora dataset.
    Test will error out if it takes too long, or output files missing.
    """

    def setUp(self) -> None:
        self.__proto_utils = ProtoUtils()

    def __generate_gbml_config_helper(
        self, task_name: str
    ) -> gbml_config_pb2.GbmlConfig:
        artifact_metadata = get_mocked_dataset_artifact_metadata()[task_name]
        gbml_config_uri = artifact_metadata.frozen_gbml_config_uri
        gbml_config_pb = self.__proto_utils.read_proto_from_yaml(
            uri=gbml_config_uri, proto_cls=gbml_config_pb2.GbmlConfig
        )
        task_metadata_type = TaskMetadataPbWrapper(
            task_metadata_pb=gbml_config_pb.task_metadata
        ).task_metadata_type

        # Pre-load files locally since Spark running locally cant read from GCS
        gcs_assets_dir = get_gcs_assets_dir_from_frozen_gbml_config_uri(
            gbml_config_uri=gbml_config_uri
        )
        tmp_local_assets_download_path = LocalUri(f"{tempfile.mkdtemp()}")
        file_loader = FileLoader()
        file_loader.load_directory(
            dir_uri_src=gcs_assets_dir, dir_uri_dst=tmp_local_assets_download_path
        )
        self.__overwrite_local_downloaded_assets_paths(
            gbml_config_pb=gbml_config_pb,
            gcs_dir=gcs_assets_dir,
            local_dir=tmp_local_assets_download_path,
        )

        if task_metadata_type == TaskMetadataType.NODE_BASED_TASK:
            self.__overwrite_sgs_output_paths_node_classification(
                gbml_config_pb=gbml_config_pb,
                gcs_dir=gcs_assets_dir,
                local_dir=tmp_local_assets_download_path,
            )
        elif (
            task_metadata_type
            == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
        ):
            self.__overwrite_sgs_output_paths_link_prediction(
                gbml_config_pb=gbml_config_pb,
                gcs_dir=gcs_assets_dir,
                local_dir=tmp_local_assets_download_path,
            )
        else:
            raise NotImplementedError(
                f"Task metadata type {task_metadata_type} not yet supported"
            )

        return gbml_config_pb

    def __generate_cora_homogeneous_supervised_node_classification_gbml_config_pb(
        self,
        split_generator_config: gbml_config_pb2.GbmlConfig.DatasetConfig.SplitGeneratorConfig,
    ) -> LocalUri:
        """
        Overwrites the output paths for a mocked gbml config to ensure each test gets a unique write destination.
        """

        task_name = CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO.name

        gbml_config_pb = self.__generate_gbml_config_helper(task_name=task_name)

        # update output paths
        frozen_config_uri = self.__overwrite_splitgen_output_paths_node_classification(
            gbml_config_pb=gbml_config_pb, split_generator_config=split_generator_config
        )

        return frozen_config_uri

    def __generate_cora_homogeneous_node_anchor_based_link_prediction_gbml_config_pb(
        self,
        split_generator_config: gbml_config_pb2.GbmlConfig.DatasetConfig.SplitGeneratorConfig,
    ) -> LocalUri:
        """
        Overwrites the output paths for a mocked gbml config to ensure each test gets a unique write destination.
        """

        task_name = TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO.name

        gbml_config_pb = self.__generate_gbml_config_helper(task_name=task_name)

        frozen_config_uri = self.__overwrite_splitgen_output_paths_link_prediction(
            gbml_config_pb=gbml_config_pb, split_generator_config=split_generator_config
        )

        return frozen_config_uri

    def __overwrite_splitgen_output_paths_node_classification(
        self,
        gbml_config_pb: gbml_config_pb2.GbmlConfig,
        split_generator_config: gbml_config_pb2.GbmlConfig.DatasetConfig.SplitGeneratorConfig,
    ):
        # Override the frozen config with configurable test params
        gbml_config_pb.dataset_config.split_generator_config.assigner_cls_path = (
            split_generator_config.assigner_cls_path
        )
        gbml_config_pb.dataset_config.split_generator_config.split_strategy_cls_path = (
            split_generator_config.split_strategy_cls_path
        )
        # Overwrite output paths.
        tmp_dir_path = tempfile.mkdtemp()
        tmp_split_generator_dir = LocalUri.join(tmp_dir_path, "split_generator")

        outputPaths = (
            gbml_config_pb.shared_config.dataset_metadata.supervised_node_classification_dataset
        )
        outputPaths.train_data_uri = LocalUri.join(
            tmp_split_generator_dir, "train/" "samples/"
        ).uri
        outputPaths.val_data_uri = LocalUri.join(
            tmp_split_generator_dir, "val/" "samples/"
        ).uri
        outputPaths.test_data_uri = LocalUri.join(
            tmp_split_generator_dir, "test/" "samples/"
        ).uri
        frozen_gbml_config_uri = LocalUri.join(
            tmp_split_generator_dir, "splitgen_supervised_node_classification.yaml"
        )

        self.__proto_utils.write_proto_to_yaml(
            proto=gbml_config_pb, uri=frozen_gbml_config_uri
        )
        return frozen_gbml_config_uri

    def __overwrite_splitgen_output_paths_link_prediction(
        self,
        gbml_config_pb: gbml_config_pb2.GbmlConfig,
        split_generator_config: gbml_config_pb2.GbmlConfig.DatasetConfig.SplitGeneratorConfig,
    ):
        # Override the frozen config with configurable test params
        gbml_config_pb.dataset_config.split_generator_config.assigner_cls_path = (
            split_generator_config.assigner_cls_path
        )
        gbml_config_pb.dataset_config.split_generator_config.split_strategy_cls_path = (
            split_generator_config.split_strategy_cls_path
        )

        # adjust split ratios so that each bucket gets some edges for small graphs like Toy Graph
        gbml_config_pb.dataset_config.split_generator_config.assigner_args[
            "train_split"
        ] = "0.5"
        gbml_config_pb.dataset_config.split_generator_config.assigner_args[
            "val_split"
        ] = "0.25"
        gbml_config_pb.dataset_config.split_generator_config.assigner_args[
            "test_split"
        ] = "0.25"

        # Overwrite output paths.
        tmp_dir_path = tempfile.mkdtemp()
        tmp_split_generator_dir = LocalUri.join(tmp_dir_path, "split_generator")

        outputPaths = (
            gbml_config_pb.shared_config.dataset_metadata.node_anchor_based_link_prediction_dataset
        )
        outputPaths.train_main_data_uri = LocalUri.join(
            tmp_split_generator_dir, "train/" "main_samples/"
        ).uri
        outputPaths.val_main_data_uri = LocalUri.join(
            tmp_split_generator_dir, "val/" "main_samples/"
        ).uri
        outputPaths.test_main_data_uri = LocalUri.join(
            tmp_split_generator_dir, "test/" "main_samples/"
        ).uri
        task_metadata_pb_wrapper = TaskMetadataPbWrapper(
            task_metadata_pb=gbml_config_pb.task_metadata
        )
        random_negative_node_types = (
            task_metadata_pb_wrapper.get_supervision_edge_node_types(
                should_include_src_nodes=False,
                should_include_dst_nodes=True,
            )
        )

        for node_type in random_negative_node_types:
            outputPaths.train_node_type_to_random_negative_data_uri[
                node_type
            ] = LocalUri.join(
                tmp_split_generator_dir,
                "train/",
                "random_negative_samples/",
                f"{node_type}/",
            ).uri
            outputPaths.val_node_type_to_random_negative_data_uri[
                node_type
            ] = LocalUri.join(
                tmp_split_generator_dir,
                "val/",
                "random_negative_samples/",
                f"{node_type}/",
            ).uri
            outputPaths.test_node_type_to_random_negative_data_uri[
                node_type
            ] = LocalUri.join(
                tmp_split_generator_dir,
                "test/",
                "random_negative_samples/",
                f"{node_type}/",
            ).uri

        frozen_gbml_config_uri = LocalUri.join(
            tmp_split_generator_dir,
            "splitgen_transductive_node_anchor_link_prediction.yaml",
        )

        self.__proto_utils.write_proto_to_yaml(
            proto=gbml_config_pb, uri=frozen_gbml_config_uri
        )
        return frozen_gbml_config_uri

    def __overwrite_local_downloaded_assets_paths(
        self,
        gbml_config_pb: gbml_config_pb2.GbmlConfig,
        gcs_dir: GcsUri,
        local_dir: LocalUri,
    ):
        """
        First overwrites the preprocessed_metadata_uri referenced in gbml_config_pb to local.
        Then, opens the file at preprocessed_metadata_uri and overwrites those paths to local too.
        overwrites the SGS output referenced in gbml_config_pb to local.
        """

        preprocessed_metadata_uri = (
            gbml_config_pb.shared_config.preprocessed_metadata_uri
        )
        gbml_config_pb.shared_config.preprocessed_metadata_uri = (
            preprocessed_metadata_uri.replace(gcs_dir.uri, local_dir.uri)
        )
        with open(gbml_config_pb.shared_config.preprocessed_metadata_uri, "r+") as f:
            content = f.read()
            f.seek(0)
            new_content = content.replace(gcs_dir.uri, local_dir.uri)
            f.write(new_content)
            f.truncate()

    def __overwrite_sgs_output_paths_link_prediction(
        self,
        gbml_config_pb: gbml_config_pb2.GbmlConfig,
        gcs_dir: GcsUri,
        local_dir: LocalUri,
    ):
        flattened_output_dataset = (
            gbml_config_pb.shared_config.flattened_graph_metadata.node_anchor_based_link_prediction_output
        )

        flattened_output_dataset.tfrecord_uri_prefix = (
            flattened_output_dataset.tfrecord_uri_prefix.replace(
                gcs_dir.uri, local_dir.uri
            )
        )
        for (
            node_type,
            random_negative_tfrecord_uri_prefix,
        ) in (
            flattened_output_dataset.node_type_to_random_negative_tfrecord_uri_prefix.items()
        ):
            flattened_output_dataset.node_type_to_random_negative_tfrecord_uri_prefix[
                node_type
            ] = random_negative_tfrecord_uri_prefix.replace(gcs_dir.uri, local_dir.uri)

    def __overwrite_sgs_output_paths_node_classification(
        self,
        gbml_config_pb: gbml_config_pb2.GbmlConfig,
        gcs_dir: GcsUri,
        local_dir: LocalUri,
    ):
        flattened_output_dataset = (
            gbml_config_pb.shared_config.flattened_graph_metadata.supervised_node_classification_output
        )

        flattened_output_dataset.labeled_tfrecord_uri_prefix = (
            flattened_output_dataset.labeled_tfrecord_uri_prefix.replace(
                gcs_dir.uri, local_dir.uri
            )
        )

    def _compile_and_run_splitgen_pipeline_locally(
        self,
        frozen_gbml_config_uri: LocalUri,
        resource_config_uri: LocalUri,
    ):
        _PATH_TO_SPLIT_GEN_SCALA_ROOT = os.path.join(
            get_project_root_directory(), "scala_spark35"
        )
        _SCALA_TOOLS_PATH = os.path.join(get_project_root_directory(), "tools", "scala")
        _SPLIT_GEN_SCALA_PROJECT_NAME = "split_generator"

        commands: List[str] = [
            f"cd {_PATH_TO_SPLIT_GEN_SCALA_ROOT} && sbt {_SPLIT_GEN_SCALA_PROJECT_NAME}/assembly",
            f"""{_SCALA_TOOLS_PATH}/spark-3.5.0-bin-hadoop3/bin/spark-submit \\
                --class Main \\
                --master local \\
                --jars {SPARK_35_TFRECORD_JAR_LOCAL_PATH} \\
                {_PATH_TO_SPLIT_GEN_SCALA_ROOT}/{_SPLIT_GEN_SCALA_PROJECT_NAME}/target/scala-2.12/split_generator-assembly-1.0.jar \\
                splitgen_integration_test_{current_formatted_datetime()} \\
                {frozen_gbml_config_uri.uri} \\
                {resource_config_uri.uri}""",
        ]

        for command in commands:
            logger.info(f"Running following command: {command}")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            output, error = process.communicate()
            output = output or b""
            error = error or b""

            logger.info(f"STDOUT:")
            logger.info(output.decode(encoding="utf-8"))
            logger.info(f"STDERR:")
            logger.info(error.decode(encoding="utf-8"))

    @staticmethod
    def __read_training_sample_protos_from_tfrecords(
        uri_prefix: Uri, proto_cls: Type[TSample]
    ) -> List[TSample]:
        def parse_training_sample_pb(byte_str: bytes) -> TSample:
            pb = proto_cls()
            pb.ParseFromString(byte_str)
            return pb

        tf_record_uris = _get_tfrecord_uris(uri_prefix=uri_prefix)
        dataset = TfRecordsIterableDataset(
            tf_record_uris=tf_record_uris,
            process_raw_sample_fn=parse_training_sample_pb,
        )
        sample_pbs = list(dataset)
        return sample_pbs

    def __build_node_classification_data_splits(
        self,
        node_classification_dataset_metadata_pb: dataset_metadata_pb2.SupervisedNodeClassificationDataset,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    ) -> Tuple[
        supervised_node_classification.NodeClassificationSplitData,
        supervised_node_classification.NodeClassificationSplitData,
        supervised_node_classification.NodeClassificationSplitData,
    ]:
        # Read train, val, test samples from tfrecords.
        train_split_samples = self.__read_training_sample_protos_from_tfrecords(
            uri_prefix=UriFactory.create_uri(
                uri=node_classification_dataset_metadata_pb.train_data_uri
            ),
            proto_cls=training_samples_schema_pb2.SupervisedNodeClassificationSample,
        )

        assert (
            len(train_split_samples) > 0
        ), "We should have more than 0 training samples"

        val_split_samples = self.__read_training_sample_protos_from_tfrecords(
            uri_prefix=UriFactory.create_uri(
                uri=node_classification_dataset_metadata_pb.val_data_uri
            ),
            proto_cls=training_samples_schema_pb2.SupervisedNodeClassificationSample,
        )

        test_split_samples = self.__read_training_sample_protos_from_tfrecords(
            uri_prefix=UriFactory.create_uri(
                uri=node_classification_dataset_metadata_pb.test_data_uri
            ),
            proto_cls=training_samples_schema_pb2.SupervisedNodeClassificationSample,
        )

        # Build train/val/test NodeClassificationSplitData splits.
        train_split = supervised_node_classification.build_single_data_split_subgraph_from_dataset_samples(
            split_samples=train_split_samples,
            graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            graph_builder=GraphBuilderFactory.get_graph_builder(
                backend_name=GraphBackend.PYG
            ),
        )

        val_split = supervised_node_classification.build_single_data_split_subgraph_from_dataset_samples(
            split_samples=val_split_samples,
            graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            graph_builder=GraphBuilderFactory.get_graph_builder(
                backend_name=GraphBackend.PYG
            ),
        )

        test_split = supervised_node_classification.build_single_data_split_subgraph_from_dataset_samples(
            split_samples=test_split_samples,
            graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            graph_builder=GraphBuilderFactory.get_graph_builder(
                backend_name=GraphBackend.PYG
            ),
        )

        return train_split, val_split, test_split

    def __build_node_anchor_based_link_prediction_data_splits(
        self,
        node_anchor_dataset_metadata_pb: dataset_metadata_pb2.NodeAnchorBasedLinkPredictionDataset,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    ) -> Tuple[
        node_anchor_based_link_prediction.NodeAnchorBasedLinkPredictionSplitData,
        node_anchor_based_link_prediction.NodeAnchorBasedLinkPredictionSplitData,
        node_anchor_based_link_prediction.NodeAnchorBasedLinkPredictionSplitData,
    ]:
        # Read main train, val, test samples from tfrecords.
        main_train_split_samples = self.__read_training_sample_protos_from_tfrecords(
            uri_prefix=UriFactory.create_uri(
                uri=node_anchor_dataset_metadata_pb.train_main_data_uri
            ),
            proto_cls=training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
        )

        main_val_split_samples = self.__read_training_sample_protos_from_tfrecords(
            uri_prefix=UriFactory.create_uri(
                uri=node_anchor_dataset_metadata_pb.val_main_data_uri
            ),
            proto_cls=training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
        )

        main_test_split_samples = self.__read_training_sample_protos_from_tfrecords(
            uri_prefix=UriFactory.create_uri(
                uri=node_anchor_dataset_metadata_pb.test_main_data_uri
            ),
            proto_cls=training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
        )

        # Read random negative train, val, test samples from tfrecords.
        task_metadata_pb = (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_pb
        )
        # TODO: (tzhao-sc) the following supports on homogeneous graphs, it might need to load from multiple
        #  folders at the same time for heterogeneous graph support.
        node_type = task_metadata_pb.node_anchor_based_link_prediction_task_metadata.supervision_edge_types[
            0
        ].src_node_type
        random_neg_train_split_samples = self.__read_training_sample_protos_from_tfrecords(
            uri_prefix=UriFactory.create_uri(
                uri=node_anchor_dataset_metadata_pb.train_node_type_to_random_negative_data_uri[
                    node_type
                ]
            ),
            proto_cls=training_samples_schema_pb2.RootedNodeNeighborhood,
        )

        random_neg_val_split_samples = self.__read_training_sample_protos_from_tfrecords(
            uri_prefix=UriFactory.create_uri(
                uri=node_anchor_dataset_metadata_pb.val_node_type_to_random_negative_data_uri[
                    node_type
                ]
            ),
            proto_cls=training_samples_schema_pb2.RootedNodeNeighborhood,
        )

        random_neg_test_split_samples = self.__read_training_sample_protos_from_tfrecords(
            uri_prefix=UriFactory.create_uri(
                uri=node_anchor_dataset_metadata_pb.test_node_type_to_random_negative_data_uri[
                    node_type
                ]
            ),
            proto_cls=training_samples_schema_pb2.RootedNodeNeighborhood,
        )

        # Build train/val/test NodeAnchorBasedLinkPredictionSplitData splits.
        train_split = node_anchor_based_link_prediction.build_single_data_split_subgraph_from_samples(
            split_main_samples=main_train_split_samples,
            split_random_negatives=random_neg_train_split_samples,
            graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            graph_builder=GraphBuilderFactory.get_graph_builder(
                backend_name=GraphBackend.PYG
            ),
        )

        val_split = node_anchor_based_link_prediction.build_single_data_split_subgraph_from_samples(
            split_main_samples=main_val_split_samples,
            split_random_negatives=random_neg_val_split_samples,
            graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            graph_builder=GraphBuilderFactory.get_graph_builder(
                backend_name=GraphBackend.PYG
            ),
        )

        test_split = node_anchor_based_link_prediction.build_single_data_split_subgraph_from_samples(
            split_main_samples=main_test_split_samples,
            split_random_negatives=random_neg_test_split_samples,
            graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            graph_builder=GraphBuilderFactory.get_graph_builder(
                backend_name=GraphBackend.PYG
            ),
        )

        return train_split, val_split, test_split

    def are_edge_features_included(self, composed_graph: PygGraphData) -> None:
        # assert edge features exist
        self.assertTrue(composed_graph.edge_attr_dict != {})
        # if there are edge feats, for all relations assert edge feats dim is non zero, ie edge features are not empty
        edge_feats_list = [
            edge_feature
            for relation, edge_feature in composed_graph.edge_attr_dict.items()
        ]
        all_edge_feats_dim = [edge_feats.shape[1] for edge_feats in edge_feats_list]
        self.assertTrue(np.all(all_edge_feats_dim))

    def __validate_node_classification_split(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        node_classification_split_type: supervised_node_classification.NodeClassificationSettingType,
        mocked_dataset_info: MockedDatasetInfo,
    ):
        training_sample_cls = (
            gbml_config_pb_wrapper.dataset_metadata_pb_wrapper.training_sample_type
        )
        self.assertTrue(
            training_sample_cls
            == training_samples_schema_pb2.SupervisedNodeClassificationSample
        )

        node_classification_dataset_pb = (
            gbml_config_pb_wrapper.dataset_metadata_pb_wrapper.dataset_metadata_pb.supervised_node_classification_dataset
        )
        (
            train_split,
            val_split,
            test_split,
        ) = self.__build_node_classification_data_splits(
            node_classification_dataset_metadata_pb=node_classification_dataset_pb,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )

        supervised_node_classification.log_node_classification_split_details(
            train_split=train_split, val_split=val_split, test_split=test_split
        )

        # Check that splits are disjoint in terms of labeled nodes.
        self.assertTrue(
            are_dataset_split_sets_disjoint(
                train=train_split.labeled_nodes,
                val=val_split.labeled_nodes,
                test=test_split.labeled_nodes,
            )
        )

        train_graph: PygGraphData = cast(PygGraphData, train_split.graph)
        val_graph: PygGraphData = cast(PygGraphData, val_split.graph)
        test_graph: PygGraphData = cast(PygGraphData, test_split.graph)

        graph_builder = cast(
            PygGraphBuilder,
            GraphBuilderFactory.get_graph_builder(backend_name=GraphBackend.PYG),
        )
        for graph_data in (train_graph, val_graph, test_graph):
            graph_builder.add_graph_data(graph_data=graph_data)
        composed_graph = graph_builder.build()

        logger.info(
            f"Composed graph: {composed_graph.num_nodes, composed_graph.num_edges}"
        )

        # Ensure that the composed graph has at most as many edges as the input graph (no new edges were created)
        self.assertLessEqual(
            composed_graph.num_edges,
            sum(
                [
                    mocked_dataset_info.get_num_edges(
                        edge_type=edge_type, edge_usage_type=EdgeUsageType.MAIN
                    )
                    for edge_type in mocked_dataset_info.edge_types
                ]
            ),
        )

        # Ensure that the composed graph has at most as many nodes as the input graph (no new nodes were created)
        self.assertLessEqual(
            composed_graph.num_nodes, sum(mocked_dataset_info.num_nodes.values())
        )

        if (
            node_classification_split_type
            == supervised_node_classification.NodeClassificationSettingType.TRANSDUCTIVE
        ):
            # We cannot really ensure specific about this split type, since it depends on graph structure.
            pass
        elif (
            node_classification_split_type
            == supervised_node_classification.NodeClassificationSettingType.INDUCTIVE
        ):
            # All edge sets across train/val/test splits must be disjoint.
            self.assertEquals(
                train_graph.num_edges + val_graph.num_edges + test_graph.num_edges,
                composed_graph.num_edges,
            )

        # asserts edge_feats exist and are non empty
        self.are_edge_features_included(composed_graph=composed_graph)

    def __validate_transductive_node_anchor_based_link_prediction_split(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    ):
        training_sample_cls = (
            gbml_config_pb_wrapper.dataset_metadata_pb_wrapper.training_sample_type
        )
        self.assertTrue(
            training_sample_cls
            == training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample
        )

        node_anchor_dataset_metadata_pb = (
            gbml_config_pb_wrapper.dataset_metadata_pb_wrapper.output_metadata
        )

        assert isinstance(
            node_anchor_dataset_metadata_pb,
            dataset_metadata_pb2.NodeAnchorBasedLinkPredictionDataset,
        )

        (
            train_split,
            val_split,
            test_split,
        ) = self.__build_node_anchor_based_link_prediction_data_splits(
            node_anchor_dataset_metadata_pb=node_anchor_dataset_metadata_pb,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )

        node_anchor_based_link_prediction.log_node_anchor_based_link_prediction_split_details(
            train_split=train_split, val_split=val_split, test_split=test_split
        )

        # Check that splits are disjoint in terms of labeled edges.
        self.assertTrue(
            are_dataset_split_sets_disjoint(
                train=train_split.pos_edges,
                val=val_split.pos_edges,
                test=test_split.pos_edges,
            )
        )
        self.assertTrue(
            are_dataset_split_sets_disjoint(
                train=train_split.hard_neg_edges,
                val=val_split.hard_neg_edges,
                test=test_split.hard_neg_edges,
            )
        )

        train_graph: PygGraphData = cast(PygGraphData, train_split.graph)
        val_graph: PygGraphData = cast(PygGraphData, val_split.graph)
        test_graph: PygGraphData = cast(PygGraphData, test_split.graph)

        graph_builder = GraphBuilderFactory.get_graph_builder(
            backend_name=GraphBackend.PYG
        )
        for graph_data in (train_graph, val_graph, test_graph):
            graph_builder.add_graph_data(graph_data=graph_data)
        composed_graph = graph_builder.build()

        # Train graph and val graph should be equivalent in terms of number of edges in this setting.
        self.assertEqual(
            train_graph.num_edges,
            val_graph.num_edges,
        )

        # Test graph should have more edges than val/train graphs in this setting.
        self.assertGreater(
            test_graph.num_edges,
            val_graph.num_edges,
        )

        # asserts edge_feats exist and are non empty
        self.are_edge_features_included(composed_graph=composed_graph)

    def test_split_generator_transductive_node_anchor_link_prediction(
        self,
    ):
        # Transductive node anchor based link prediction SplitStrategy constants
        transductive_node_anchor_based_link_prediction_assigner_cls_path = (
            "splitgenerator."
            "lib."
            "assigners."
            "TransductiveEdgeToLinkSplitHashingAssigner"
        )
        transductive_node_anchor_based_link_prediction_split_strategy_cls_path = (
            "splitgenerator."
            "lib."
            "split_strategies."
            "TransductiveNodeAnchorBasedLinkPredictionSplitStrategy"
        )

        transductive_node_anchor_based_link_prediction_split_generator_pb = gbml_config_pb2.GbmlConfig.DatasetConfig.SplitGeneratorConfig(
            split_strategy_cls_path=transductive_node_anchor_based_link_prediction_split_strategy_cls_path,
            assigner_cls_path=transductive_node_anchor_based_link_prediction_assigner_cls_path,
        )

        frozen_gbml_config_uri = self.__generate_cora_homogeneous_node_anchor_based_link_prediction_gbml_config_pb(
            split_generator_config=transductive_node_anchor_based_link_prediction_split_generator_pb
        )

        logger.info(
            f"Running Split Generator pipeline using GbmlConfig at {frozen_gbml_config_uri}"
        )
        resource_config_uri = UriFactory.create_uri(
            uri=get_resource_config().get_resource_config_uri
        )
        assert isinstance(resource_config_uri, LocalUri)

        self._compile_and_run_splitgen_pipeline_locally(
            frozen_gbml_config_uri=frozen_gbml_config_uri,
            resource_config_uri=resource_config_uri,
        )

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=frozen_gbml_config_uri
            )
        )

        self.__validate_transductive_node_anchor_based_link_prediction_split(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )

    def test_split_generator_transductive_supervised_node_classification(
        self,
    ):
        # Transductive Supervised Node Classification SplitStrategy constants
        transductive_supervised_node_classification_assigner_cls_path = (
            "splitgenerator." "lib." "assigners." "NodeToDatasetSplitHashingAssigner"
        )
        transductive_supervised_node_classification_split_strategy_cls_path = (
            "splitgenerator."
            "lib."
            "split_strategies."
            "TransductiveSupervisedNodeClassificationSplitStrategy"
        )

        transductive_supervised_node_classification_split_generator_pb = gbml_config_pb2.GbmlConfig.DatasetConfig.SplitGeneratorConfig(
            split_strategy_cls_path=transductive_supervised_node_classification_split_strategy_cls_path,
            assigner_cls_path=transductive_supervised_node_classification_assigner_cls_path,
        )

        frozen_gbml_config_uri = self.__generate_cora_homogeneous_supervised_node_classification_gbml_config_pb(
            split_generator_config=transductive_supervised_node_classification_split_generator_pb
        )

        logger.info(
            f"Running Transductive Node Classification Split Generator pipeline using GbmlConfig at {frozen_gbml_config_uri}"
        )

        resource_config_uri = UriFactory.create_uri(
            uri=get_resource_config().get_resource_config_uri
        )
        assert isinstance(resource_config_uri, LocalUri)

        self._compile_and_run_splitgen_pipeline_locally(
            frozen_gbml_config_uri=frozen_gbml_config_uri,
            resource_config_uri=resource_config_uri,
        )

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=frozen_gbml_config_uri
            )
        )

        self.__validate_node_classification_split(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            node_classification_split_type=supervised_node_classification.NodeClassificationSettingType.TRANSDUCTIVE,
            mocked_dataset_info=CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
        )

    def test_split_generator_inductive_supervised_node_classification(
        self,
    ):
        # Inductive Supervised Node Classification SplitStrategy constants
        inductive_supervised_node_classification_assigner_cls_path = (
            "splitgenerator." "lib." "assigners." "NodeToDatasetSplitHashingAssigner"
        )
        inductive_supervised_node_classification_split_strategy_cls_path = (
            "splitgenerator."
            "lib."
            "split_strategies."
            "InductiveSupervisedNodeClassificationSplitStrategy"
        )

        inductive_supervised_node_classification_split_generator_pb = gbml_config_pb2.GbmlConfig.DatasetConfig.SplitGeneratorConfig(
            split_strategy_cls_path=inductive_supervised_node_classification_split_strategy_cls_path,
            assigner_cls_path=inductive_supervised_node_classification_assigner_cls_path,
        )

        frozen_gbml_config_uri = self.__generate_cora_homogeneous_supervised_node_classification_gbml_config_pb(
            split_generator_config=inductive_supervised_node_classification_split_generator_pb
        )

        logger.info(
            f"Running inductive Node Classification Split Generator pipeline using GbmlConfig at {frozen_gbml_config_uri}"
        )

        resource_config_uri = UriFactory.create_uri(
            uri=get_resource_config().get_resource_config_uri
        )
        assert isinstance(resource_config_uri, LocalUri)

        self._compile_and_run_splitgen_pipeline_locally(
            frozen_gbml_config_uri=frozen_gbml_config_uri,
            resource_config_uri=resource_config_uri,
        )

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=frozen_gbml_config_uri
            )
        )

        self.__validate_node_classification_split(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            node_classification_split_type=supervised_node_classification.NodeClassificationSettingType.INDUCTIVE,
            mocked_dataset_info=CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
        )
