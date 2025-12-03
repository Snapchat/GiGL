import unittest

import torch

from gigl.common.logger import Logger
from gigl.src.common.modeling_task_specs.graphsage_template_modeling_spec import (
    GraphSageTemplateTrainerSpec,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.versioning import (
    MockedDatasetArtifactMetadata,
    get_mocked_dataset_artifact_metadata,
)
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
)

metadata: MockedDatasetArtifactMetadata = get_mocked_dataset_artifact_metadata()[
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO.name
]
gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
    gbml_config_uri=metadata.frozen_gbml_config_uri
)

logger = Logger()


class GraphSageTemplateTrainerSpecTrainingTest(unittest.TestCase):
    """
    Tests training functionality for GraphSageTemplateTrainerSpec.
    """

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_graphsage_training(self):
        """
        Test that the GraphSageTemplateTrainerSpec can perform training and evaluation.
        """
        trainer = GraphSageTemplateTrainerSpec(
            main_sample_batch_size=8,
            random_negative_batch_size=8,
            early_stop_patience=1,
            validate_every_n_batches=5,
            num_val_batches=5,
            num_epochs=1,
            optim_lr=0.005,
        )
        trainer.init_model(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=torch.device("cpu"),
        )
        trainer.setup_for_training()

        trainer.train(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=torch.device("cpu"),
        )

        model_eval_metrics = trainer.eval(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=torch.device("cpu"),
        )
        gbml_test_mrr = model_eval_metrics.metrics["mrr"]

        self.assertIsNotNone(
            gbml_test_mrr, "MRR should not be None after training and evaluation."
        )
