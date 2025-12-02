import os
import shutil
import tempfile
import unittest
from typing import Optional, Type

from parameterized import param, parameterized

import gigl.env.pipelines_config
from gigl.common import UriFactory
from gigl.src.validation_check.config_validator import kfp_validation_checks

# Helper functions for creating mock config YAML strings


def _create_valid_offline_subgraph_sampling_task_config() -> str:
    """Create a task config YAML for offline subgraph sampling (without live SGS backend flag)."""
    return """
graphMetadata:
  edgeTypes:
  - dstNodeType: paper
    relation: cites
    srcNodeType: paper
  nodeTypes:
  - paper
taskMetadata:
  nodeAnchorBasedLinkPredictionTaskMetadata:
    supervisionEdgeTypes:
      - srcNodeType: paper
        relation: cites
        dstNodeType: paper
datasetConfig:
  dataPreprocessorConfig:
    dataPreprocessorConfigClsPath: gigl.src.mocking.mocking_assets.passthrough_preprocessor_config_for_mocked_assets.PassthroughPreprocessorConfigForMockedAssets
    dataPreprocessorArgs:
      mocked_dataset_name: 'cora_homogeneous_node_anchor_edge_features'
  subgraphSamplerConfig:
    numHops: 2
    numNeighborsToSample: 10
    numPositiveSamples: 1
  splitGeneratorConfig:
    assignerArgs:
      seed: '42'
      test_split: '0.2'
      train_split: '0.7'
      val_split: '0.1'
    assignerClsPath: splitgenerator.lib.assigners.TransductiveEdgeToLinkSplitHashingAssigner
    splitStrategyClsPath: splitgenerator.lib.split_strategies.TransductiveNodeAnchorBasedLinkPredictionSplitStrategy
inferencerConfig:
  inferencerClsPath: gigl.src.common.modeling_task_specs.node_anchor_based_link_prediction_modeling_task_spec.NodeAnchorBasedLinkPredictionModelingTaskSpec
trainerConfig:
  trainerClsPath: gigl.src.common.modeling_task_specs.node_anchor_based_link_prediction_modeling_task_spec.NodeAnchorBasedLinkPredictionModelingTaskSpec
"""


def _create_valid_live_subgraph_sampling_task_config() -> str:
    """Create a task config YAML for live subgraph sampling (with GLT backend flag enabled)."""
    return """
graphMetadata:
  edgeTypes:
  - dstNodeType: paper
    relation: cites
    srcNodeType: paper
  nodeTypes:
  - paper
datasetConfig:
  dataPreprocessorConfig:
    dataPreprocessorConfigClsPath: gigl.src.mocking.mocking_assets.passthrough_preprocessor_config_for_mocked_assets.PassthroughPreprocessorConfigForMockedAssets
    dataPreprocessorArgs:
      mocked_dataset_name: 'cora_homogeneous_node_anchor_edge_features_user_defined_labels'
trainerConfig:
  command: python -m examples.link_prediction.homogeneous_training
inferencerConfig:
  command: python -m examples.link_prediction.homogeneous_inference
taskMetadata:
  nodeAnchorBasedLinkPredictionTaskMetadata:
    supervisionEdgeTypes:
    - dstNodeType: paper
      relation: cites
      srcNodeType: paper
featureFlags:
  should_run_glt_backend: 'True'
"""


def _create_valid_live_subgraph_sampling_resource_config() -> str:
    """Create a resource config YAML for GLT/live backend (without subgraph sampler and split generator configs)."""
    return """
sharedResourceConfig:
  commonComputeConfig:
    project: "test-project"
    region: "us-central1"
    tempAssetsBucket: "gs://test-temp"
    tempRegionalAssetsBucket: "gs://test-temp-regional"
    permAssetsBucket: "gs://test-perm"
    tempAssetsBqDatasetName: "test_dataset"
    embeddingBqDatasetName: "test_embeddings"
    gcpServiceAccountEmail: "test@test-project.iam.gserviceaccount.com"
    dataflowRunner: "DataflowRunner"
preprocessorConfig:
  nodePreprocessorConfig:
    numWorkers: 10
    maxNumWorkers: 20
    diskSizeGb: 100
    machineType: "n1-standard-4"
  edgePreprocessorConfig:
    numWorkers: 15
    maxNumWorkers: 25
    diskSizeGb: 150
    machineType: "n1-standard-8"
trainerResourceConfig:
  vertexAiTrainerConfig:
    machineType: "n1-standard-16"
    gpuType: "NVIDIA_TESLA_T4"
    gpuLimit: 2
    numReplicas: 3
inferencerResourceConfig:
  vertexAiInferencerConfig:
    machineType: "n1-standard-4"
    gpuType: "NVIDIA_TESLA_T4"
    gpuLimit: 2
    numReplicas: 3
"""


def _create_valid_offline_subgraph_sampling_resource_config() -> str:
    """Create a resource config YAML for offline subgraph sampling (with all component configs)."""
    return """
sharedResourceConfig:
  commonComputeConfig:
    project: "test-project"
    region: "us-central1"
    tempAssetsBucket: "gs://test-temp"
    tempRegionalAssetsBucket: "gs://test-temp-regional"
    permAssetsBucket: "gs://test-perm"
    tempAssetsBqDatasetName: "test_dataset"
    embeddingBqDatasetName: "test_embeddings"
    gcpServiceAccountEmail: "test@test-project.iam.gserviceaccount.com"
    dataflowRunner: "DataflowRunner"
preprocessorConfig:
  nodePreprocessorConfig:
    numWorkers: 10
    maxNumWorkers: 20
    diskSizeGb: 100
    machineType: "n1-standard-4"
  edgePreprocessorConfig:
    numWorkers: 15
    maxNumWorkers: 25
    diskSizeGb: 150
    machineType: "n1-standard-8"
subgraphSamplerConfig:
  machineType: "n2d-highmem-16"
  numLocalSsds: 2
  numReplicas: 4
splitGeneratorConfig:
  machineType: "n2d-standard-16"
  numLocalSsds: 2
  numReplicas: 4
trainerResourceConfig:
  vertexAiTrainerConfig:
    machineType: "n1-highmem-8"
    gpuType: "NVIDIA_TESLA_P100"
    gpuLimit: 1
    numReplicas: 2
inferencerResourceConfig:
  dataflowInferencerConfig:
    numWorkers: 1
    maxNumWorkers: 256
    machineType: "c2d-highmem-32"
    diskSizeGb: 100
"""


class TestConfigValidationPerSGSBackends(unittest.TestCase):
    """Test suite for config validation with different SGS backends (live subgraph sampling, offline subgraph sampling)."""

    def setUp(self):
        """Set up temporary directory for test config files."""
        self._temp_dir = tempfile.mkdtemp()

        # Create task config files
        self._live_task_config_uri = UriFactory.create_uri(
            self._write_temp_file(
                _create_valid_live_subgraph_sampling_task_config(),
                "live_task_config.yaml",
            )
        )
        self._offline_task_config_uri = UriFactory.create_uri(
            self._write_temp_file(
                _create_valid_offline_subgraph_sampling_task_config(),
                "offline_task_config.yaml",
            )
        )

        # Create resource config files
        self._live_resource_config_uri = UriFactory.create_uri(
            self._write_temp_file(
                _create_valid_live_subgraph_sampling_resource_config(),
                "live_resource_config.yaml",
            )
        )
        self._offline_resource_config_uri = UriFactory.create_uri(
            self._write_temp_file(
                _create_valid_offline_subgraph_sampling_resource_config(),
                "offline_resource_config.yaml",
            )
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._temp_dir, ignore_errors=True)
        # Clear the cached resource config to ensure each test loads its own config. Otherwise, the resource config will be cached, leading to
        # incorrect resource configs being used for tests (i.e. live SGS resource config being used for offline SGS tests).
        gigl.env.pipelines_config._resource_config = None

    def _write_temp_file(self, content: str, filename: str) -> str:
        """Write content to a temporary file and return its path."""
        filepath = os.path.join(self._temp_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    @parameterized.expand(
        [
            param(
                "Test that live subgraph sampling resource config passes when we are doing live subgraph sampling",
                should_use_live_sgs_backend=True,
                should_use_live_sgs_resource_config=True,
                expected_exception=None,
            ),
            param(
                "Test that live subgraph sampling resource config fails when we are doing offline subgraph sampling",
                # This test is expected to fail since we are missing required fields to specify how to do offline subgraph sampling.
                should_use_live_sgs_backend=False,
                should_use_live_sgs_resource_config=True,
                expected_exception=AssertionError,
            ),
            param(
                "Test that offline subgraph sampling resource config passes when we are doing offline subgraph sampling",
                should_use_live_sgs_backend=False,
                should_use_live_sgs_resource_config=False,
                expected_exception=None,
            ),
            param(
                "Test that offline subgraph sampling resource config passes when we are doing live subgraph sampling",
                # This test is expected to pass because an offline SGS resource config is still valid when live SGS is enabled.
                should_use_live_sgs_backend=True,
                should_use_live_sgs_resource_config=False,
                expected_exception=None,
            ),
        ]
    )
    def test_resource_config_validation_with_mock_configs(
        self,
        _,
        should_use_live_sgs_backend: bool,
        should_use_live_sgs_resource_config: bool,
        expected_exception: Optional[Type[Exception]] = None,
    ) -> None:
        task_config_uri = (
            self._live_task_config_uri
            if should_use_live_sgs_backend
            else self._offline_task_config_uri
        )
        resource_config_uri = (
            self._live_resource_config_uri
            if should_use_live_sgs_resource_config
            else self._offline_resource_config_uri
        )

        if expected_exception is None:
            kfp_validation_checks(
                job_name="resource_config_validation_test",
                task_config_uri=task_config_uri,
                start_at="config_populator",
                resource_config_uri=resource_config_uri,
            )
        else:
            with self.assertRaises(expected_exception):
                kfp_validation_checks(
                    job_name="resource_config_validation_test",
                    task_config_uri=task_config_uri,
                    start_at="config_populator",
                    resource_config_uri=resource_config_uri,
                )


if __name__ == "__main__":
    unittest.main()
