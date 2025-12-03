import unittest

import torch

from gigl.common.logger import Logger
from gigl.src.common.modeling_task_specs.node_classification_modeling_task_spec import (
    NodeClassificationModelingTaskSpec,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.versioning import (
    MockedDatasetArtifactMetadata,
    get_mocked_dataset_artifact_metadata,
)
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
)

metadata: MockedDatasetArtifactMetadata = get_mocked_dataset_artifact_metadata()[
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO.name
]
gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
    gbml_config_uri=metadata.frozen_gbml_config_uri
)

logger = Logger()


class NodeClassificationSpecPygTrainingTest(unittest.TestCase):
    """
    Tests functionality of being able to train a model with a node classification modeling task spec.
    """

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_pyg_training(self):
        """
        Test that we can train on Cora.
        """

        gbml_trainer = NodeClassificationModelingTaskSpec()
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
        gbml_test_acc = model_eval_metrics.metrics["acc"]
        logger.info(f"gbml_test_acc: {gbml_test_acc}")
