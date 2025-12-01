import os
import shutil
import tempfile
import unittest
from typing import Optional, Type

from parameterized import param, parameterized

from gigl.common import Uri, UriFactory
from gigl.src.validation_check.config_validator import kfp_validation_checks

# Task config URIs (using real configs from the codebase)
OFFLINE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI = UriFactory.create_uri(
    "gigl/src/mocking/configs/e2e_node_anchor_based_link_prediction_template_gbml_config.yaml"
)
LIVE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI = UriFactory.create_uri(
    "examples/link_prediction/configs/e2e_hom_cora_sup_task_config.yaml"
)


# Helper functions for creating mock resource config YAML strings
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
  dataflowInferencerConfig:
    numWorkers: 10
    maxNumWorkers: 20
    diskSizeGb: 100
    machineType: "n1-standard-4"
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
  machineType: "n1-standard-8"
  numLocalSsds: 2
  numReplicas: 5
splitGeneratorConfig:
  machineType: "n1-standard-8"
  numLocalSsds: 2
  numReplicas: 5
trainerResourceConfig:
  vertexAiTrainerConfig:
    machineType: "n1-standard-16"
    gpuType: "NVIDIA_TESLA_T4"
    gpuLimit: 2
    numReplicas: 3
inferencerResourceConfig:
  dataflowInferencerConfig:
    numWorkers: 10
    maxNumWorkers: 20
    diskSizeGb: 100
    machineType: "n1-standard-4"
"""


class TestConfigValidationPerSGSBackends(unittest.TestCase):
    """Test suite for config validation with different SGS backends (live subgraph sampling, offline subgraph sampling)."""

    def setUp(self):
        """Set up temporary directory for test config files."""
        self.temp_dir = tempfile.mkdtemp()

        # Create resource config files
        self._live_resource_config_path = self._write_temp_file(
            _create_valid_live_subgraph_sampling_resource_config(),
            "live_resource_config.yaml",
        )
        self._offline_resource_config_path = self._write_temp_file(
            _create_valid_offline_subgraph_sampling_resource_config(),
            "offline_resource_config.yaml",
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_temp_file(self, content: str, filename: str) -> str:
        """Write content to a temporary file and return its path."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    @parameterized.expand(
        [
            param(
                "Test that live subgraph sampling resource config passes when we are doing live subgraph sampling",
                use_live_resource_config=True,
                task_config_uri=LIVE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI,
                expected_exception=None,
            ),
            param(
                "Test that live subgraph sampling resource config fails when we are doing offline subgraph sampling",
                # This test is expected to fail since we are missing required fields to specify how to do offline subgraph sampling.
                use_live_resource_config=True,
                task_config_uri=OFFLINE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI,
                expected_exception=AssertionError,
            ),
            param(
                "Test that offline subgraph sampling resource config passes when we are doing offline subgraph sampling",
                use_live_resource_config=False,
                task_config_uri=OFFLINE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI,
                expected_exception=None,
            ),
            param(
                "Test that offline subgraph sampling resource config passes when we are doing live subgraph sampling",
                # This test is expected to pass because an offline SGS resource config is still valid when live SGS is enabled.
                use_live_resource_config=False,
                task_config_uri=LIVE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI,
                expected_exception=None,
            ),
        ]
    )
    def test_resource_config_validation_with_mock_configs(
        self,
        _,
        use_live_resource_config: bool,
        task_config_uri: Uri,
        expected_exception: Optional[Type[Exception]] = None,
    ) -> None:
        # Select the appropriate resource config based on the parameter
        resource_config_path = (
            self._live_resource_config_path
            if use_live_resource_config
            else self._offline_resource_config_path
        )

        # Act & Assert
        if expected_exception is None:
            kfp_validation_checks(
                job_name="resource_config_validation_test",
                task_config_uri=task_config_uri,
                # Start at config populator since the task configs are template configs
                start_at="config_populator",
                resource_config_uri=UriFactory.create_uri(resource_config_path),
            )
        else:
            with self.assertRaises(expected_exception):
                kfp_validation_checks(
                    job_name="resource_config_validation_test",
                    task_config_uri=task_config_uri,
                    # Start at config populator since the task configs are template configs
                    start_at="config_populator",
                    resource_config_uri=UriFactory.create_uri(resource_config_path),
                )


if __name__ == "__main__":
    unittest.main()
