import unittest

import torch

from gigl.common.logger import Logger
from gigl.src.common.modeling_task_specs.node_anchor_based_link_prediction_modeling_task_spec import (
    NodeAnchorBasedLinkPredictionModelingTaskSpec,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.versioning import (
    MockedDatasetArtifactMetadata,
    get_mocked_dataset_artifact_metadata,
)
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
    DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)

logger = Logger()


class NodeAnchorBasedLinkPredictionModelingTaskSpecPygTrainingTest(unittest.TestCase):
    """
    Tests functionality of being able to train a model with a node anchor based link
    prediction modeling task spec.
    """

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def run_training_on_mocked_data(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        gbml_trainer: NodeAnchorBasedLinkPredictionModelingTaskSpec,
    ) -> None:
        """
        Run training given a gbml config pb wrapper and gbml_trainer
        gbml_config_pb_wrapper (GbmlConfigPbWrapper): GBML Config pb wrapper for mocked training job
        gbml_trainer (NodeAnchorBasedLinkPredictionModelingTaskSpec): Initialized Task Spec for mocked training job
        """
        gbml_trainer.init_model(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )
        gbml_trainer.setup_for_training()

        # Next train our gbml pipeline model
        gbml_trainer.train(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=torch.device("cpu"),
        )
        model_eval_metrics = gbml_trainer.eval(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=torch.device("cpu"),
        )
        gbml_test_mrr = model_eval_metrics.metrics["mrr"]

        logger.info(f"gbml_test_mrr: {gbml_test_mrr}")

    def test_homogeneous_pyg_training(self):
        """
        Test that we can train with homogeneous data on Cora.
        """
        metadata: (
            MockedDatasetArtifactMetadata
        ) = get_mocked_dataset_artifact_metadata()[
            CORA_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=metadata.frozen_gbml_config_uri
            )
        )
        gbml_trainer = NodeAnchorBasedLinkPredictionModelingTaskSpec(
            gnn_model_class_path="gigl.src.common.models.pyg.homogeneous.GraphSAGE",
            main_sample_batch_size=8,
            random_negative_sample_batch_size=8,
            random_negative_sample_batch_size_for_evaluation=8,
            early_stop_patience=1,
            val_every_num_batches=5,
            num_val_batches=5,
            num_test_batches=5,
        )
        self.run_training_on_mocked_data(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper, gbml_trainer=gbml_trainer
        )

    def test_heterogeneous_pyg_training(self):
        """
        Test that we can train with heterogeneous data on DBLP.
        """
        metadata: (
            MockedDatasetArtifactMetadata
        ) = get_mocked_dataset_artifact_metadata()[
            DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=metadata.frozen_gbml_config_uri
            )
        )
        gbml_trainer = NodeAnchorBasedLinkPredictionModelingTaskSpec(
            gnn_model_class_path="gigl.src.common.models.pyg.heterogeneous.HGT",
            main_sample_batch_size=8,
            random_negative_sample_batch_size=8,
            random_negative_sample_batch_size_for_evaluation=8,
            early_stop_patience=1,
            val_every_num_batches=5,
            num_val_batches=5,
            num_test_batches=5,
        )
        self.run_training_on_mocked_data(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper, gbml_trainer=gbml_trainer
        )
