import os
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
import unittest

from gigl.common import LocalUri
from gigl.common.logger import Logger
from gigl.common.utils.proto_utils import ProtoUtils
from snapchat.research.gbml import gbml_config_pb2

logger = Logger()

TEST_TASK_CONFIG = """
sharedConfig:
    isGraphDirected: true
datasetConfig:
    dataPreprocessorConfig:
        dataPreprocessorArgs:
            bq_edges_table_name: "project.dataset.bq_edges_table_name_${now:%Y%m%d}"
            positive_label_date_range: "${now:%Y%m%d,days-10}:${now:%Y%m%d,days-1}"
    subgraphSamplerConfig:
        numHops: 1
        numNeighborsToSample: 2
        numPositiveSamples: 1
    splitGeneratorConfig:
        assignerArgs:
            test_split: '0.2'
            train_split: '0.7'
            val_split: '0.1'
        assignerClsPath: splitgenerator.lib.assigners.TransductiveEdgeToLinkSplitHashingAssigner
        splitStrategyClsPath: splitgenerator.lib.split_strategies.TransductiveNodeAnchorBasedLinkPredictionSplitStrategy
graphMetadata:
    edgeTypes:
    - dstNodeType: user
      relation: is_friends_with
      srcNodeType: user
    nodeTypes:
    - user
inferencerConfig:
    inferencerClsPath: gigl.src.common.modeling_task_specs.node_anchor_based_link_prediction_modeling_task_spec.NodeAnchorBasedLinkPredictionModelingTaskSpec
    inferencerArgs:
        num_layers: '2'
        hid_dim: '128'
        out_dim: '128'
taskMetadata:
    nodeAnchorBasedLinkPredictionTaskMetadata:
        supervisionEdgeTypes:
        - srcNodeType: user
          relation: is_friends_with
          dstNodeType: user
trainerConfig:
    trainerArgs:
        margin: '0.3'
        optim_lr: '0.005'
    trainerClsPath: gigl.src.common.modeling_task_specs.node_anchor_based_link_prediction_modeling_task_spec.NodeAnchorBasedLinkPredictionModelingTaskSpec
"""
class ProtoUtilsTest(unittest.TestCase):
    def setUp(self):
        self.proto_utils = ProtoUtils()


        tmp_file = NamedTemporaryFile(delete=False)
        logger.info(f"Writing test config to {tmp_file.name}")
        tmp_file.write(TEST_TASK_CONFIG.encode())
        tmp_file.close()
        self.test_task_config_path = tmp_file.name

    def tearDown(self):
        # delete the temporary file
        pass
        # os.remove(self.test_task_config_path)

    def test_can_read_gbml_config_from_yaml(self):
        task_config = self.proto_utils.read_proto_from_yaml(
            uri=LocalUri(self.test_task_config_path),
            proto_cls=gbml_config_pb2.GbmlConfig,
        )

        self.assertEqual(task_config.shared_config.is_graph_directed, True)
        expected_bq_edges_table_name = (
            f"project.dataset.bq_edges_table_name_{datetime.now().strftime('%Y%m%d')}"
        )
        self.assertEqual(
            task_config.dataset_config.data_preprocessor_config.data_preprocessor_args["bq_edges_table_name"],
            expected_bq_edges_table_name,
        )
        expected_positive_label_date_range_start = (
            datetime.now() - timedelta(days=10)
        ).strftime("%Y%m%d")
        expected_positive_label_date_range_end = (
            datetime.now() - timedelta(days=1)
        ).strftime("%Y%m%d")
        self.assertEqual(
            task_config.dataset_config.data_preprocessor_config.data_preprocessor_args["positive_label_date_range"],
            f"{expected_positive_label_date_range_start}:{expected_positive_label_date_range_end}",
        )
