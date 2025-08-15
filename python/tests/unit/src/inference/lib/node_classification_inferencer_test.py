import tempfile
import unittest
from typing import Optional, OrderedDict

import torch
import yaml

from gigl.common import LocalUri
from gigl.common.logger import Logger
from gigl.src.common.graph_builder.graph_builder_factory import GraphBuilderFactory
from gigl.src.common.types.model import GraphBackend
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.inference.v1.lib.base_inferencer import (
    InferBatchResults,
    SupervisedNodeClassificationBaseInferencer,
)
from gigl.src.inference.v1.lib.inference_output_schema import (
    DEFAULT_EMBEDDING_FIELD,
    DEFAULT_NODE_ID_FIELD,
    DEFAULT_PREDICTION_FIELD,
)
from gigl.src.inference.v1.lib.node_classification_inferencer import (
    NodeClassificationInferenceBlueprint,
)
from gigl.src.training.v1.lib.data_loaders.supervised_node_classification_data_loader import (
    SupervisedNodeClassificationBatch,
)
from snapchat.research.gbml import flattened_graph_metadata_pb2, gbml_config_pb2
from tests.test_assets.celeb_test_graph.assets import (
    get_celeb_graph_metadata_pb2,
    get_celeb_preprocessed_metadata,
    get_celeb_supervised_sample,
)
from tests.test_assets.models.pass_through import PassThroughNet

logger = Logger()


class NodeClassificationInferencerTest(unittest.TestCase):
    class _SimpleInferer(SupervisedNodeClassificationBaseInferencer):
        @property
        def model(self) -> torch.nn.Module:
            return self._model

        @model.setter
        def model(self, model: torch.nn.Module) -> None:
            self._model = model

        def __init__(self) -> None:
            super().__init__()
            self._model = self.init_model(None)

        def init_model(
            self,
            gbml_config_pb_wrapper: Optional[GbmlConfigPbWrapper],
            state_dict: Optional[OrderedDict[str, torch.Tensor]] = None,
        ) -> torch.nn.Module:
            return PassThroughNet()

        def infer_batch(
            self,
            batch: SupervisedNodeClassificationBatch,
            device: torch.device = torch.device("cpu"),
        ) -> InferBatchResults:
            embeddings = torch.Tensor(
                [i for i in range(len(batch.root_nodes))]
            ).reshape(
                -1, 1
            )  # [[1], [2]]
            return InferBatchResults(
                embeddings=embeddings, predictions=batch.root_node_labels
            )

    def test_workflow(self):
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".yaml") as tfh:
            yaml.dump(get_celeb_preprocessed_metadata(), tfh)
            tfh_name = tfh.name
        preprocessed_metadata_uri = LocalUri(tfh_name)
        gbml_config_pb = gbml_config_pb2.GbmlConfig(
            graph_metadata=get_celeb_graph_metadata_pb2(),
            shared_config=gbml_config_pb2.GbmlConfig.SharedConfig(
                preprocessed_metadata_uri=preprocessed_metadata_uri.uri,
                flattened_graph_metadata=flattened_graph_metadata_pb2.FlattenedGraphMetadata(
                    supervised_node_classification_output=flattened_graph_metadata_pb2.SupervisedNodeClassificationOutput(
                        labeled_tfrecord_uri_prefix="", unlabeled_tfrecord_uri_prefix=""
                    ),
                ),
            ),
        )
        gbml_config_pb_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)
        inferencer = NodeClassificationInferencerTest._SimpleInferer()

        inference_blueprint = NodeClassificationInferenceBlueprint(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            inferencer=inferencer,
            graph_builder=GraphBuilderFactory.get_graph_builder(
                backend_name=GraphBackend.PYG
            ),
        )

        batch_generator_fn = inference_blueprint.get_batch_generator_fn()
        batch_for_inference = batch_generator_fn(
            batch=[get_celeb_supervised_sample(), get_celeb_supervised_sample()]
        )
        self.assertIsInstance(batch_for_inference, SupervisedNodeClassificationBatch)
        self.assertEqual(len(batch_for_inference.root_nodes), 2)
        inferer = inference_blueprint.get_inferer()
        tagged_outputs = [
            tagged_output for tagged_output in inferer(batch_for_inference)
        ]
        self.assertEqual(len(tagged_outputs), 4)  # two predictions and two embeddings

        embedding_sum = 0.0
        for tagged_output in tagged_outputs:
            self.assertEqual(tagged_output.value[DEFAULT_NODE_ID_FIELD], 1)
            if tagged_output.tag == "predictions":
                self.assertEqual(tagged_output.value[DEFAULT_PREDICTION_FIELD], 1)
            elif tagged_output.tag == "embeddings":
                self.assertTrue(
                    tagged_output.value[DEFAULT_EMBEDDING_FIELD] == [0.0]
                    or tagged_output.value[DEFAULT_EMBEDDING_FIELD] == [1.0]
                )
                embedding_sum += tagged_output.value[DEFAULT_EMBEDDING_FIELD][0]

        self.assertEqual(
            embedding_sum, 1.0
        )  # First embedding is [0.0] and second is [1.0]
