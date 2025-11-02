import unittest

from gigl.common.logger import Logger
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.dataset_metadata import DatasetMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.config_populator.config_populator import ConfigPopulator
from parameterized import param, parameterized
from snapchat.research.gbml import (
    dataset_metadata_pb2,
    flattened_graph_metadata_pb2,
    gbml_config_pb2,
    inference_metadata_pb2,
    trained_model_metadata_pb2,
)

logger = Logger()

_TEMPLATE_CONFIG_FOR_SGS = gbml_config_pb2.GbmlConfig(
    task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata(
        node_based_task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata.NodeBasedTaskMetadata()
    ),
)

_TEMPLATE_CONFIG_FOR_GLT = gbml_config_pb2.GbmlConfig(
    task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata(
        node_based_task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata.NodeBasedTaskMetadata()
    ),
    feature_flags={"should_run_glt_backend": "True"},
)


class ConfigPopulatorUnitTest(unittest.TestCase):
    """
    This test checks the completion of the ConfigPopulator step.
    """

    def test_sgs_config_population_is_accurate(
        self,
    ):
        config_populator = ConfigPopulator()

        frozen_gbml_config_pb = config_populator._populate_frozen_gbml_config_pb(
            applied_task_identifier=self.applied_task_identifier,
            template_gbml_config_pb=_TEMPLATE_CONFIG_FOR_SGS,
        )
        gbml_config_pb_wrapper = GbmlConfigPbWrapper(
            gbml_config_pb=frozen_gbml_config_pb
        )

        # Check that preprocessed metadata uri is set.
        self.assertNotEqual(
            gbml_config_pb_wrapper.shared_config.preprocessed_metadata_uri, ""
        )

        # Assert the right flattened graph metadata was set.
        self.assertTrue(
            gbml_config_pb_wrapper.shared_config.HasField("flattened_graph_metadata")
        )
        flattened_output_pb = (
            gbml_config_pb_wrapper.flattened_graph_metadata_pb_wrapper.output_metadata
        )
        if isinstance(
            flattened_output_pb,
            flattened_graph_metadata_pb2.SupervisedNodeClassificationOutput,
        ):
            self.assertNotEqual(flattened_output_pb.labeled_tfrecord_uri_prefix, "")
            self.assertNotEqual(flattened_output_pb.unlabeled_tfrecord_uri_prefix, "")
        # Assert the right dataset metadata was set
        self.assertTrue(
            gbml_config_pb_wrapper.shared_config.HasField("dataset_metadata")
        )
        dataset_metadata_pb = DatasetMetadataPbWrapper(
            dataset_metadata_pb=gbml_config_pb_wrapper.shared_config.dataset_metadata
        ).output_metadata
        if isinstance(
            dataset_metadata_pb,
            dataset_metadata_pb2.SupervisedNodeClassificationDataset,
        ):
            self.assertNotEqual(dataset_metadata_pb.train_data_uri, "")
            self.assertNotEqual(dataset_metadata_pb.val_data_uri, "")
            self.assertNotEqual(dataset_metadata_pb.test_data_uri, "")

        # GBML Config Pb Wrapper Checks
        self.assertFalse(gbml_config_pb_wrapper.should_use_glt_backend)
        # We should expect both of the below calls to happen successfully without error when they are set
        gbml_config_pb_wrapper.flattened_graph_metadata_pb_wrapper
        gbml_config_pb_wrapper.dataset_metadata_pb_wrapper

        # Assert trainer metadata assets were set
        trained_model_metadata_pb: trained_model_metadata_pb2.TrainedModelMetadata = (
            gbml_config_pb_wrapper.shared_config.trained_model_metadata
        )
        self.assertTrue(
            isinstance(
                trained_model_metadata_pb,
                trained_model_metadata_pb2.TrainedModelMetadata,
            )
        )
        self.assertNotEqual(trained_model_metadata_pb.trained_model_uri, "")
        self.assertNotEqual(trained_model_metadata_pb.scripted_model_uri, "")

        # Assert inference metadata assets were set
        inference_metadata_pb: inference_metadata_pb2.InferenceMetadata = (
            gbml_config_pb_wrapper.shared_config.inference_metadata
        )
        self.assertTrue(
            isinstance(inference_metadata_pb, inference_metadata_pb2.InferenceMetadata)
        )
        for node_type in inference_metadata_pb.node_type_to_inferencer_output_info_map:
            self.assertNotEqual(
                inference_metadata_pb.node_type_to_inferencer_output_info_map[
                    node_type
                ].embeddings_path,
                "",
            )
            if (
                gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_type
                == TaskMetadataType.NODE_BASED_TASK
            ):
                self.assertNotEqual(
                    inference_metadata_pb.node_type_to_inferencer_output_info_map[
                        node_type
                    ].predictions_path,
                    "",
                )

    @parameterized.expand(
        [
            param(
                should_populate_embeddings_path=True,
            ),
            param(
                should_populate_embeddings_path=False,
            ),
        ]
    )
    def test_glt_config_population_is_accurate(
        self, should_populate_embeddings_path: bool
    ):
        config_populator = ConfigPopulator()
        frozen_gbml_config_pb = config_populator._populate_frozen_gbml_config_pb(
            applied_task_identifier=self.applied_task_identifier,
            template_gbml_config_pb=_TEMPLATE_CONFIG_FOR_GLT,
        )
        frozen_gbml_config_pb.feature_flags["should_populate_embeddings_path"] = str(
            should_populate_embeddings_path
        )
        gbml_config_pb_wrapper = GbmlConfigPbWrapper(
            gbml_config_pb=frozen_gbml_config_pb
        )

        # Check that preprocessed metadata uri is set.
        self.assertNotEqual(
            gbml_config_pb_wrapper.shared_config.preprocessed_metadata_uri, ""
        )

        self.assertFalse(
            gbml_config_pb_wrapper.shared_config.HasField("flattened_graph_metadata")
        )
        self.assertFalse(
            gbml_config_pb_wrapper.shared_config.HasField("dataset_metadata")
        )

        # GBML Config Pb Wrapper Checks
        self.assertTrue(gbml_config_pb_wrapper.should_use_glt_backend)
        self.assertEqual(
            gbml_config_pb_wrapper.should_populate_embeddings_path,
            should_populate_embeddings_path,
        )
        with self.assertRaises(ValueError):
            # We should expect to throw an error when accessing the flattened graph metadata pb wrapper when it is not set
            gbml_config_pb_wrapper.flattened_graph_metadata_pb_wrapper
        with self.assertRaises(ValueError):
            # We should expect to throw an error when accessing the dataset metadata pb wrapper when it is not set
            gbml_config_pb_wrapper.dataset_metadata_pb_wrapper

        # Assert trainer metadata assets were set
        trained_model_metadata_pb: trained_model_metadata_pb2.TrainedModelMetadata = (
            gbml_config_pb_wrapper.shared_config.trained_model_metadata
        )
        self.assertTrue(
            isinstance(
                trained_model_metadata_pb,
                trained_model_metadata_pb2.TrainedModelMetadata,
            )
        )
        self.assertNotEqual(trained_model_metadata_pb.trained_model_uri, "")
        self.assertNotEqual(trained_model_metadata_pb.scripted_model_uri, "")

        # Assert inference metadata assets were set
        inference_metadata_pb: inference_metadata_pb2.InferenceMetadata = (
            gbml_config_pb_wrapper.shared_config.inference_metadata
        )
        self.assertTrue(
            isinstance(inference_metadata_pb, inference_metadata_pb2.InferenceMetadata)
        )
        for node_type in inference_metadata_pb.node_type_to_inferencer_output_info_map:
            if should_populate_embeddings_path:
                self.assertNotEqual(
                    inference_metadata_pb.node_type_to_inferencer_output_info_map[
                        node_type
                    ].embeddings_path,
                    "",
                )
            else:
                self.assertEqual(
                    inference_metadata_pb.node_type_to_inferencer_output_info_map[
                        node_type
                    ].embeddings_path,
                    "",
                )
            if (
                gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_type
                == TaskMetadataType.NODE_BASED_TASK
            ):
                self.assertNotEqual(
                    inference_metadata_pb.node_type_to_inferencer_output_info_map[
                        node_type
                    ].predictions_path,
                    "",
                )

    def setUp(self) -> None:
        self.applied_task_identifier = AppliedTaskIdentifier(
            f"test_config_populator_functionality_{current_formatted_datetime()}"
        )

    def tearDown(self) -> None:
        logger.info("Config Populator pipeline completion test tear down.")
        return super().tearDown()
