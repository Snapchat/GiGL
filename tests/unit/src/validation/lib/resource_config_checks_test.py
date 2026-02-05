from absl.testing import absltest

from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.validation_check.libs.resource_config_checks import (
    _check_if_dataflow_resource_config_valid,
    _check_if_spark_resource_config_valid,
    _validate_accelerator_type,
    _validate_machine_config,
    check_if_inferencer_graph_store_storage_command_valid,
    check_if_inferencer_resource_config_valid,
    check_if_preprocessor_resource_config_valid,
    check_if_shared_resource_config_valid,
    check_if_split_generator_resource_config_valid,
    check_if_subgraph_sampler_resource_config_valid,
    check_if_trainer_graph_store_storage_command_valid,
    check_if_trainer_resource_config_valid,
)
from snapchat.research.gbml import gbml_config_pb2, gigl_resource_config_pb2
from tests.test_assets.test_case import TestCase

# Helper functions for creating valid configurations


def _create_valid_dataflow_config() -> gigl_resource_config_pb2.DataflowResourceConfig:
    """Create a valid Dataflow resource configuration."""
    config = gigl_resource_config_pb2.DataflowResourceConfig()
    config.num_workers = 10
    config.max_num_workers = 20
    config.disk_size_gb = 100
    config.machine_type = "n1-standard-4"
    return config


def _create_valid_spark_config() -> gigl_resource_config_pb2.SparkResourceConfig:
    """Create a valid Spark resource configuration."""
    config = gigl_resource_config_pb2.SparkResourceConfig()
    config.machine_type = "n1-standard-8"
    config.num_local_ssds = 2
    config.num_replicas = 5
    return config


def _create_valid_shared_resource_config() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    """Create a valid GiglResourceConfig with SharedResourceConfig."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    config.shared_resource_config.common_compute_config.project = "test-project"
    config.shared_resource_config.common_compute_config.region = "us-central1"
    config.shared_resource_config.common_compute_config.temp_assets_bucket = (
        "gs://test-temp"
    )
    config.shared_resource_config.common_compute_config.temp_regional_assets_bucket = (
        "gs://test-temp-regional"
    )
    config.shared_resource_config.common_compute_config.perm_assets_bucket = (
        "gs://test-perm"
    )
    config.shared_resource_config.common_compute_config.temp_assets_bq_dataset_name = (
        "test_dataset"
    )
    config.shared_resource_config.common_compute_config.embedding_bq_dataset_name = (
        "test_embeddings"
    )
    config.shared_resource_config.common_compute_config.gcp_service_account_email = (
        "test@test-project.iam.gserviceaccount.com"
    )
    config.shared_resource_config.common_compute_config.dataflow_runner = (
        "DataflowRunner"
    )
    return config


def _create_valid_preprocessor_config() -> gigl_resource_config_pb2.GiglResourceConfig:
    """Create a valid GiglResourceConfig with preprocessor config."""
    config = gigl_resource_config_pb2.GiglResourceConfig()

    # Node preprocessor config
    config.preprocessor_config.node_preprocessor_config.num_workers = 10
    config.preprocessor_config.node_preprocessor_config.max_num_workers = 20
    config.preprocessor_config.node_preprocessor_config.disk_size_gb = 100
    config.preprocessor_config.node_preprocessor_config.machine_type = "n1-standard-4"

    # Edge preprocessor config
    config.preprocessor_config.edge_preprocessor_config.num_workers = 15
    config.preprocessor_config.edge_preprocessor_config.max_num_workers = 25
    config.preprocessor_config.edge_preprocessor_config.disk_size_gb = 150
    config.preprocessor_config.edge_preprocessor_config.machine_type = "n1-standard-8"

    return config


def _create_valid_subgraph_sampler_config() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    """Create a valid GiglResourceConfig with subgraph sampler config."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    config.subgraph_sampler_config.machine_type = "n1-standard-8"
    config.subgraph_sampler_config.num_local_ssds = 2
    config.subgraph_sampler_config.num_replicas = 5
    return config


def _create_valid_split_generator_config() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    """Create a valid GiglResourceConfig with split generator config."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    config.split_generator_config.machine_type = "n1-standard-8"
    config.split_generator_config.num_local_ssds = 2
    config.split_generator_config.num_replicas = 5
    return config


def _create_valid_local_trainer_config() -> gigl_resource_config_pb2.GiglResourceConfig:
    """Create a valid GiglResourceConfig with local trainer config."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    config.trainer_resource_config.local_trainer_config.num_workers = 4
    return config


def _create_valid_vertex_ai_trainer_config() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    """Create a valid GiglResourceConfig with Vertex AI trainer config."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    config.trainer_resource_config.vertex_ai_trainer_config.machine_type = (
        "n1-standard-16"
    )
    config.trainer_resource_config.vertex_ai_trainer_config.gpu_type = "NVIDIA_TESLA_T4"
    config.trainer_resource_config.vertex_ai_trainer_config.gpu_limit = 2
    config.trainer_resource_config.vertex_ai_trainer_config.num_replicas = 3
    return config


def _create_valid_kfp_trainer_config() -> gigl_resource_config_pb2.GiglResourceConfig:
    """Create a valid GiglResourceConfig with KFP trainer config."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    config.trainer_resource_config.kfp_trainer_config.cpu_request = "4"
    config.trainer_resource_config.kfp_trainer_config.memory_request = "16Gi"
    config.trainer_resource_config.kfp_trainer_config.gpu_type = "NVIDIA_TESLA_V100"
    config.trainer_resource_config.kfp_trainer_config.gpu_limit = 1
    config.trainer_resource_config.kfp_trainer_config.num_replicas = 2
    return config


def _create_valid_vertex_ai_graph_store_trainer_config() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    """Create a valid GiglResourceConfig with Vertex AI Graph Store trainer config."""
    config = gigl_resource_config_pb2.GiglResourceConfig()

    # Graph store pool config
    config.trainer_resource_config.vertex_ai_graph_store_trainer_config.graph_store_pool.machine_type = (
        "n1-highmem-8"
    )
    config.trainer_resource_config.vertex_ai_graph_store_trainer_config.graph_store_pool.gpu_type = (
        "ACCELERATOR_TYPE_UNSPECIFIED"
    )
    config.trainer_resource_config.vertex_ai_graph_store_trainer_config.graph_store_pool.gpu_limit = (
        0
    )
    config.trainer_resource_config.vertex_ai_graph_store_trainer_config.graph_store_pool.num_replicas = (
        2
    )

    # Compute pool config
    config.trainer_resource_config.vertex_ai_graph_store_trainer_config.compute_pool.machine_type = (
        "n1-standard-16"
    )
    config.trainer_resource_config.vertex_ai_graph_store_trainer_config.compute_pool.gpu_type = (
        "NVIDIA_TESLA_T4"
    )
    config.trainer_resource_config.vertex_ai_graph_store_trainer_config.compute_pool.gpu_limit = (
        2
    )
    config.trainer_resource_config.vertex_ai_graph_store_trainer_config.compute_pool.num_replicas = (
        3
    )

    return config


def _create_valid_dataflow_inferencer_config() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    """Create a valid GiglResourceConfig with Dataflow inferencer config."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    config.inferencer_resource_config.dataflow_inferencer_config.num_workers = 10
    config.inferencer_resource_config.dataflow_inferencer_config.max_num_workers = 20
    config.inferencer_resource_config.dataflow_inferencer_config.disk_size_gb = 100
    config.inferencer_resource_config.dataflow_inferencer_config.machine_type = (
        "n1-standard-4"
    )
    return config


def _create_valid_vertex_ai_inferencer_config() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    """Create a valid GiglResourceConfig with Vertex AI inferencer config."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    config.inferencer_resource_config.vertex_ai_inferencer_config.machine_type = (
        "n1-standard-16"
    )
    config.inferencer_resource_config.vertex_ai_inferencer_config.gpu_type = (
        "NVIDIA_TESLA_T4"
    )
    config.inferencer_resource_config.vertex_ai_inferencer_config.gpu_limit = 2
    config.inferencer_resource_config.vertex_ai_inferencer_config.num_replicas = 3
    return config


def _create_valid_local_inferencer_config() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    """Create a valid GiglResourceConfig with local inferencer config."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    config.inferencer_resource_config.local_inferencer_config.num_workers = 4
    return config


def _create_valid_vertex_ai_config() -> gigl_resource_config_pb2.VertexAiResourceConfig:
    """Create a valid Vertex AI resource configuration."""
    config = gigl_resource_config_pb2.VertexAiResourceConfig()
    config.machine_type = "n1-standard-16"
    return config


def _create_vertex_ai_config_with_gpu() -> (
    gigl_resource_config_pb2.VertexAiResourceConfig
):
    """Create a Vertex AI config with GPU."""
    config = gigl_resource_config_pb2.VertexAiResourceConfig()
    config.machine_type = "n1-standard-16"
    config.gpu_type = "NVIDIA_TESLA_T4"
    config.gpu_limit = 2
    return config


def _create_vertex_ai_config_with_cpu() -> (
    gigl_resource_config_pb2.VertexAiResourceConfig
):
    """Create a Vertex AI config with CPU only."""
    config = gigl_resource_config_pb2.VertexAiResourceConfig()
    config.machine_type = "n1-standard-16"
    config.gpu_type = "ACCELERATOR_TYPE_UNSPECIFIED"
    config.gpu_limit = 0
    return config


def _create_kfp_config_with_gpu() -> gigl_resource_config_pb2.KFPResourceConfig:
    """Create a KFP config with GPU."""
    config = gigl_resource_config_pb2.KFPResourceConfig()
    config.cpu_request = "4"
    config.memory_request = "16Gi"
    config.gpu_type = "NVIDIA_TESLA_V100"
    config.gpu_limit = 1
    return config


# Test Classes


class TestDataflowResourceConfig(TestCase):
    """Test suite for Dataflow resource configuration validation."""

    def test_valid_dataflow_config(self):
        """Test that a valid Dataflow configuration passes validation."""
        config = _create_valid_dataflow_config()
        # Should not raise any exception
        _check_if_dataflow_resource_config_valid(config)

    def test_missing_num_workers(self):
        """Test that missing num_workers raises an assertion error."""
        config = _create_valid_dataflow_config()
        config.num_workers = 0
        with self.assertRaises(AssertionError):
            _check_if_dataflow_resource_config_valid(config)

    def test_missing_max_num_workers(self):
        """Test that missing max_num_workers raises an assertion error."""
        config = _create_valid_dataflow_config()
        config.max_num_workers = 0
        with self.assertRaises(AssertionError):
            _check_if_dataflow_resource_config_valid(config)

    def test_missing_disk_size_gb(self):
        """Test that missing disk_size_gb raises an assertion error."""
        config = _create_valid_dataflow_config()
        config.disk_size_gb = 0
        with self.assertRaises(AssertionError):
            _check_if_dataflow_resource_config_valid(config)

    def test_missing_machine_type(self):
        """Test that missing machine_type raises an assertion error."""
        config = _create_valid_dataflow_config()
        config.machine_type = ""
        with self.assertRaises(AssertionError):
            _check_if_dataflow_resource_config_valid(config)


class TestSparkResourceConfig(TestCase):
    """Test suite for Spark resource configuration validation."""

    def test_valid_spark_config(self):
        """Test that a valid Spark configuration passes validation."""
        config = _create_valid_spark_config()
        # Should not raise any exception
        _check_if_spark_resource_config_valid(config)

    def test_missing_machine_type(self):
        """Test that missing machine_type raises an assertion error."""
        config = _create_valid_spark_config()
        config.machine_type = ""
        with self.assertRaises(AssertionError):
            _check_if_spark_resource_config_valid(config)

    def test_missing_num_local_ssds(self):
        """Test that missing num_local_ssds raises an assertion error."""
        config = _create_valid_spark_config()
        config.num_local_ssds = 0
        with self.assertRaises(AssertionError):
            _check_if_spark_resource_config_valid(config)

    def test_missing_num_replicas(self):
        """Test that missing num_replicas raises an assertion error."""
        config = _create_valid_spark_config()
        config.num_replicas = 0
        with self.assertRaises(AssertionError):
            _check_if_spark_resource_config_valid(config)


class TestSharedResourceConfig(TestCase):
    """Test suite for shared resource configuration validation."""

    def test_valid_shared_resource_config(self):
        """Test that a valid shared resource configuration passes validation."""
        config = _create_valid_shared_resource_config()
        # Should not raise any exception
        check_if_shared_resource_config_valid(config)

    def test_missing_project(self):
        """Test that missing project raises an assertion error."""
        config = _create_valid_shared_resource_config()
        config.shared_resource_config.common_compute_config.project = ""
        with self.assertRaises(AssertionError):
            check_if_shared_resource_config_valid(config)

    def test_missing_region(self):
        """Test that missing region raises an assertion error."""
        config = _create_valid_shared_resource_config()
        config.shared_resource_config.common_compute_config.region = ""
        with self.assertRaises(AssertionError):
            check_if_shared_resource_config_valid(config)

    def test_missing_temp_assets_bucket(self):
        """Test that missing temp_assets_bucket raises an assertion error."""
        config = _create_valid_shared_resource_config()
        config.shared_resource_config.common_compute_config.temp_assets_bucket = ""
        with self.assertRaises(AssertionError):
            check_if_shared_resource_config_valid(config)

    def test_missing_temp_regional_assets_bucket(self):
        """Test that missing temp_regional_assets_bucket raises an assertion error."""
        config = _create_valid_shared_resource_config()
        config.shared_resource_config.common_compute_config.temp_regional_assets_bucket = (
            ""
        )
        with self.assertRaises(AssertionError):
            check_if_shared_resource_config_valid(config)

    def test_missing_perm_assets_bucket(self):
        """Test that missing perm_assets_bucket raises an assertion error."""
        config = _create_valid_shared_resource_config()
        config.shared_resource_config.common_compute_config.perm_assets_bucket = ""
        with self.assertRaises(AssertionError):
            check_if_shared_resource_config_valid(config)

    def test_missing_temp_assets_bq_dataset_name(self):
        """Test that missing temp_assets_bq_dataset_name raises an assertion error."""
        config = _create_valid_shared_resource_config()
        config.shared_resource_config.common_compute_config.temp_assets_bq_dataset_name = (
            ""
        )
        with self.assertRaises(AssertionError):
            check_if_shared_resource_config_valid(config)

    def test_missing_embedding_bq_dataset_name(self):
        """Test that missing embedding_bq_dataset_name raises an assertion error."""
        config = _create_valid_shared_resource_config()
        config.shared_resource_config.common_compute_config.embedding_bq_dataset_name = (
            ""
        )
        with self.assertRaises(AssertionError):
            check_if_shared_resource_config_valid(config)

    def test_missing_gcp_service_account_email(self):
        """Test that missing gcp_service_account_email raises an assertion error."""
        config = _create_valid_shared_resource_config()
        config.shared_resource_config.common_compute_config.gcp_service_account_email = (
            ""
        )
        with self.assertRaises(AssertionError):
            check_if_shared_resource_config_valid(config)

    def test_missing_dataflow_runner(self):
        """Test that missing dataflow_runner raises an assertion error."""
        config = _create_valid_shared_resource_config()
        config.shared_resource_config.common_compute_config.dataflow_runner = ""
        with self.assertRaises(AssertionError):
            check_if_shared_resource_config_valid(config)


class TestPreprocessorResourceConfig(TestCase):
    """Test suite for preprocessor resource configuration validation."""

    def test_valid_preprocessor_config(self):
        """Test that a valid preprocessor configuration passes validation."""
        config = _create_valid_preprocessor_config()
        # Should not raise any exception
        check_if_preprocessor_resource_config_valid(config)

    def test_invalid_node_preprocessor_config(self):
        """Test that invalid node preprocessor config raises an assertion error."""
        config = _create_valid_preprocessor_config()
        config.preprocessor_config.node_preprocessor_config.num_workers = 0
        with self.assertRaises(AssertionError):
            check_if_preprocessor_resource_config_valid(config)

    def test_invalid_edge_preprocessor_config(self):
        """Test that invalid edge preprocessor config raises an assertion error."""
        config = _create_valid_preprocessor_config()
        config.preprocessor_config.edge_preprocessor_config.machine_type = ""
        with self.assertRaises(AssertionError):
            check_if_preprocessor_resource_config_valid(config)


class TestSubgraphSamplerResourceConfig(TestCase):
    """Test suite for subgraph sampler resource configuration validation."""

    def test_valid_subgraph_sampler_config(self):
        """Test that a valid subgraph sampler configuration passes validation."""
        config = _create_valid_subgraph_sampler_config()
        # Should not raise any exception
        check_if_subgraph_sampler_resource_config_valid(config)

    def test_invalid_subgraph_sampler_config(self):
        """Test that invalid subgraph sampler config raises an assertion error."""
        config = _create_valid_subgraph_sampler_config()
        config.subgraph_sampler_config.num_replicas = 0
        with self.assertRaises(AssertionError):
            check_if_subgraph_sampler_resource_config_valid(config)


class TestSplitGeneratorResourceConfig(TestCase):
    """Test suite for split generator resource configuration validation."""

    def test_valid_split_generator_config(self):
        """Test that a valid split generator configuration passes validation."""
        config = _create_valid_split_generator_config()
        # Should not raise any exception
        check_if_split_generator_resource_config_valid(config)

    def test_invalid_split_generator_config(self):
        """Test that invalid split generator config raises an assertion error."""
        config = _create_valid_split_generator_config()
        config.split_generator_config.machine_type = ""
        with self.assertRaises(AssertionError):
            check_if_split_generator_resource_config_valid(config)


class TestTrainerResourceConfig(TestCase):
    """Test suite for trainer resource configuration validation."""

    def test_valid_local_trainer_config(self):
        """Test that a valid local trainer configuration passes validation."""
        config = _create_valid_local_trainer_config()
        # Should not raise any exception
        check_if_trainer_resource_config_valid(config)

    def test_invalid_local_trainer_config_missing_num_workers(self):
        """Test that missing num_workers in local trainer config raises an assertion error."""
        config = _create_valid_local_trainer_config()
        config.trainer_resource_config.local_trainer_config.num_workers = 0
        with self.assertRaises(AssertionError):
            check_if_trainer_resource_config_valid(config)

    def test_valid_vertex_ai_trainer_config(self):
        """Test that a valid Vertex AI trainer configuration passes validation."""
        config = _create_valid_vertex_ai_trainer_config()
        # Should not raise any exception
        check_if_trainer_resource_config_valid(config)

    def test_invalid_vertex_ai_trainer_config_missing_machine_type(self):
        """Test that missing machine_type in Vertex AI trainer config raises an assertion error."""
        config = _create_valid_vertex_ai_trainer_config()
        config.trainer_resource_config.vertex_ai_trainer_config.machine_type = ""
        with self.assertRaises(AssertionError):
            check_if_trainer_resource_config_valid(config)

    def test_invalid_vertex_ai_trainer_config_missing_gpu_type(self):
        """Test that missing gpu_type in Vertex AI trainer config raises an assertion error."""
        config = _create_valid_vertex_ai_trainer_config()
        config.trainer_resource_config.vertex_ai_trainer_config.gpu_type = ""
        with self.assertRaises(AssertionError):
            check_if_trainer_resource_config_valid(config)

    def test_invalid_vertex_ai_trainer_config_missing_num_replicas(self):
        """Test that missing num_replicas in Vertex AI trainer config raises an assertion error."""
        config = _create_valid_vertex_ai_trainer_config()
        config.trainer_resource_config.vertex_ai_trainer_config.num_replicas = 0
        with self.assertRaises(AssertionError):
            check_if_trainer_resource_config_valid(config)

    def test_invalid_vertex_ai_trainer_config_cpu_with_gpu_limit(self):
        """Test that CPU training with gpu_limit > 0 raises an assertion error."""
        config = _create_valid_vertex_ai_trainer_config()
        config.trainer_resource_config.vertex_ai_trainer_config.gpu_type = (
            "ACCELERATOR_TYPE_UNSPECIFIED"
        )
        config.trainer_resource_config.vertex_ai_trainer_config.gpu_limit = (
            1  # Should be 0 for CPU
        )
        with self.assertRaises(AssertionError):
            check_if_trainer_resource_config_valid(config)

    def test_invalid_vertex_ai_trainer_config_gpu_with_zero_limit(self):
        """Test that GPU training with gpu_limit = 0 raises an assertion error."""
        config = _create_valid_vertex_ai_trainer_config()
        config.trainer_resource_config.vertex_ai_trainer_config.gpu_limit = (
            0  # Should be > 0 for GPU
        )
        with self.assertRaises(AssertionError):
            check_if_trainer_resource_config_valid(config)

    def test_valid_kfp_trainer_config(self):
        """Test that a valid KFP trainer configuration passes validation."""
        config = _create_valid_kfp_trainer_config()
        # Should not raise any exception
        check_if_trainer_resource_config_valid(config)

    def test_invalid_kfp_trainer_config_missing_cpu_request(self):
        """Test that missing cpu_request in KFP trainer config raises an assertion error."""
        config = _create_valid_kfp_trainer_config()
        config.trainer_resource_config.kfp_trainer_config.cpu_request = ""
        with self.assertRaises(AssertionError):
            check_if_trainer_resource_config_valid(config)

    def test_invalid_kfp_trainer_config_missing_memory_request(self):
        """Test that missing memory_request in KFP trainer config raises an assertion error."""
        config = _create_valid_kfp_trainer_config()
        config.trainer_resource_config.kfp_trainer_config.memory_request = ""
        with self.assertRaises(AssertionError):
            check_if_trainer_resource_config_valid(config)

    def test_invalid_kfp_trainer_config_cpu_with_gpu_limit(self):
        """Test that CPU training with gpu_limit > 0 in KFP config raises an assertion error."""
        config = _create_valid_kfp_trainer_config()
        config.trainer_resource_config.kfp_trainer_config.gpu_type = (
            "ACCELERATOR_TYPE_UNSPECIFIED"
        )
        config.trainer_resource_config.kfp_trainer_config.gpu_limit = 1
        with self.assertRaises(AssertionError):
            check_if_trainer_resource_config_valid(config)

    def test_valid_vertex_ai_graph_store_trainer_config(self):
        """Test that a valid Vertex AI Graph Store trainer configuration passes validation."""
        config = _create_valid_vertex_ai_graph_store_trainer_config()
        # Should not raise any exception
        check_if_trainer_resource_config_valid(config)

    def test_invalid_vertex_ai_graph_store_trainer_config_graph_store_pool(self):
        """Test that invalid graph store pool config raises an assertion error."""
        config = _create_valid_vertex_ai_graph_store_trainer_config()
        config.trainer_resource_config.vertex_ai_graph_store_trainer_config.graph_store_pool.machine_type = (
            ""
        )
        with self.assertRaises(AssertionError):
            check_if_trainer_resource_config_valid(config)

    def test_invalid_vertex_ai_graph_store_trainer_config_compute_pool(self):
        """Test that invalid compute pool config raises an assertion error."""
        config = _create_valid_vertex_ai_graph_store_trainer_config()
        config.trainer_resource_config.vertex_ai_graph_store_trainer_config.compute_pool.gpu_limit = (
            0  # Should be > 0 for GPU
        )
        with self.assertRaises(AssertionError):
            check_if_trainer_resource_config_valid(config)


class TestInferencerResourceConfig(TestCase):
    """Test suite for inferencer resource configuration validation."""

    def test_valid_dataflow_inferencer_config(self):
        """Test that a valid Dataflow inferencer configuration passes validation."""
        config = _create_valid_dataflow_inferencer_config()
        # Should not raise any exception
        check_if_inferencer_resource_config_valid(config)

    def test_invalid_dataflow_inferencer_config(self):
        """Test that invalid Dataflow inferencer config raises an assertion error."""
        config = _create_valid_dataflow_inferencer_config()
        config.inferencer_resource_config.dataflow_inferencer_config.num_workers = 0
        with self.assertRaises(AssertionError):
            check_if_inferencer_resource_config_valid(config)

    def test_valid_vertex_ai_inferencer_config(self):
        """Test that a valid Vertex AI inferencer configuration passes validation."""
        config = _create_valid_vertex_ai_inferencer_config()
        # Should not raise any exception
        check_if_inferencer_resource_config_valid(config)

    def test_invalid_vertex_ai_inferencer_config_missing_machine_type(self):
        """Test that missing machine_type in Vertex AI inferencer config raises an assertion error."""
        config = _create_valid_vertex_ai_inferencer_config()
        config.inferencer_resource_config.vertex_ai_inferencer_config.machine_type = ""
        with self.assertRaises(AssertionError):
            check_if_inferencer_resource_config_valid(config)

    def test_invalid_vertex_ai_inferencer_config_missing_gpu_type(self):
        """Test that missing gpu_type in Vertex AI inferencer config raises an assertion error."""
        config = _create_valid_vertex_ai_inferencer_config()
        config.inferencer_resource_config.vertex_ai_inferencer_config.gpu_type = ""
        with self.assertRaises(AssertionError):
            check_if_inferencer_resource_config_valid(config)

    def test_invalid_vertex_ai_inferencer_config_missing_num_replicas(self):
        """Test that missing num_replicas in Vertex AI inferencer config raises an assertion error."""
        config = _create_valid_vertex_ai_inferencer_config()
        config.inferencer_resource_config.vertex_ai_inferencer_config.num_replicas = 0
        with self.assertRaises(AssertionError):
            check_if_inferencer_resource_config_valid(config)

    def test_invalid_vertex_ai_inferencer_config_cpu_with_gpu_limit(self):
        """Test that CPU inference with gpu_limit > 0 raises an assertion error."""
        config = _create_valid_vertex_ai_inferencer_config()
        config.inferencer_resource_config.vertex_ai_inferencer_config.gpu_type = (
            "ACCELERATOR_TYPE_UNSPECIFIED"
        )
        config.inferencer_resource_config.vertex_ai_inferencer_config.gpu_limit = 1
        with self.assertRaises(AssertionError):
            check_if_inferencer_resource_config_valid(config)

    def test_invalid_vertex_ai_inferencer_config_gpu_with_zero_limit(self):
        """Test that GPU inference with gpu_limit = 0 raises an assertion error."""
        config = _create_valid_vertex_ai_inferencer_config()
        config.inferencer_resource_config.vertex_ai_inferencer_config.gpu_limit = 0
        with self.assertRaises(AssertionError):
            check_if_inferencer_resource_config_valid(config)

    def test_valid_local_inferencer_config(self):
        """Test that a valid local inferencer configuration passes validation."""
        config = _create_valid_local_inferencer_config()
        # Should not raise any exception
        check_if_inferencer_resource_config_valid(config)

    def test_invalid_local_inferencer_config(self):
        """Test that invalid local inferencer config raises an assertion error."""
        config = _create_valid_local_inferencer_config()
        config.inferencer_resource_config.local_inferencer_config.num_workers = 0
        with self.assertRaises(AssertionError):
            check_if_inferencer_resource_config_valid(config)


class TestAcceleratorTypeValidation(TestCase):
    """Test suite for accelerator type validation helper."""

    def test_valid_gpu_config_vertex_ai(self):
        """Test that a valid GPU configuration for Vertex AI passes validation."""
        config = _create_vertex_ai_config_with_gpu()
        # Should not raise any exception
        _validate_accelerator_type(config)

    def test_valid_cpu_config_vertex_ai(self):
        """Test that a valid CPU configuration for Vertex AI passes validation."""
        config = _create_vertex_ai_config_with_cpu()
        # Should not raise any exception
        _validate_accelerator_type(config)

    def test_invalid_cpu_config_with_gpu_limit_vertex_ai(self):
        """Test that CPU config with gpu_limit > 0 raises an assertion error."""
        config = _create_vertex_ai_config_with_cpu()
        config.gpu_limit = 1
        with self.assertRaises(AssertionError):
            _validate_accelerator_type(config)

    def test_invalid_gpu_config_with_zero_limit_vertex_ai(self):
        """Test that GPU config with gpu_limit = 0 raises an assertion error."""
        config = _create_vertex_ai_config_with_gpu()
        config.gpu_limit = 0
        with self.assertRaises(AssertionError):
            _validate_accelerator_type(config)

    def test_valid_gpu_config_kfp(self):
        """Test that a valid GPU configuration for KFP passes validation."""
        config = _create_kfp_config_with_gpu()
        # Should not raise any exception
        _validate_accelerator_type(config)

    def test_invalid_gpu_config_with_zero_limit_kfp(self):
        """Test that GPU config with gpu_limit = 0 in KFP raises an assertion error."""
        config = _create_kfp_config_with_gpu()
        config.gpu_limit = 0
        with self.assertRaises(AssertionError):
            _validate_accelerator_type(config)


class TestValidateMachineConfig(TestCase):
    """Test suite for _validate_machine_config error handling."""

    def test_valid_local_config(self):
        """Test that a valid LocalResourceConfig passes validation."""
        gigl_config = _create_valid_local_trainer_config()
        config = gigl_config.trainer_resource_config.local_trainer_config
        # Should not raise any exception
        _validate_machine_config(config)

    def test_invalid_local_config_missing_num_workers(self):
        """Test that LocalResourceConfig without num_workers raises an assertion error."""
        gigl_config = _create_valid_local_trainer_config()
        config = gigl_config.trainer_resource_config.local_trainer_config
        config.num_workers = 0
        with self.assertRaises(AssertionError):
            _validate_machine_config(config)

    def test_valid_dataflow_config(self):
        """Test that a valid DataflowResourceConfig passes validation."""
        config = _create_valid_dataflow_config()
        # Should not raise any exception
        _validate_machine_config(config)

    def test_valid_kfp_config(self):
        """Test that a valid KFPResourceConfig passes validation."""
        gigl_config = _create_valid_kfp_trainer_config()
        config = gigl_config.trainer_resource_config.kfp_trainer_config
        # Should not raise any exception
        _validate_machine_config(config)

    def test_valid_vertex_ai_config(self):
        """Test that a valid VertexAiResourceConfig passes validation."""
        gigl_config = _create_valid_vertex_ai_trainer_config()
        config = gigl_config.trainer_resource_config.vertex_ai_trainer_config
        # Should not raise any exception
        _validate_machine_config(config)

    def test_valid_vertex_ai_graph_store_config(self):
        """Test that a valid VertexAiGraphStoreConfig passes validation."""
        gigl_config = _create_valid_vertex_ai_graph_store_trainer_config()
        config = (
            gigl_config.trainer_resource_config.vertex_ai_graph_store_trainer_config
        )
        # Should not raise any exception
        _validate_machine_config(config)


# Helper functions for creating GbmlConfig configurations


def _create_gbml_config_with_both_graph_stores(
    storage_command: str = "python -m gigl.distributed.graph_store.storage_main",
) -> GbmlConfigPbWrapper:
    """Create a GbmlConfig with graph_store_storage_config set for both trainer and inferencer."""
    gbml_config = gbml_config_pb2.GbmlConfig()
    gbml_config.trainer_config.graph_store_storage_config.command = storage_command
    gbml_config.inferencer_config.graph_store_storage_config.command = storage_command
    return GbmlConfigPbWrapper(gbml_config_pb=gbml_config)


def _create_gbml_config_without_graph_stores() -> GbmlConfigPbWrapper:
    """Create a GbmlConfig without graph_store_storage_config for trainer or inferencer."""
    gbml_config = gbml_config_pb2.GbmlConfig()
    gbml_config.trainer_config.trainer_args["some_arg"] = "some_value"
    gbml_config.inferencer_config.inferencer_args["some_arg"] = "some_value"
    return GbmlConfigPbWrapper(gbml_config_pb=gbml_config)


class TestTrainerGraphStoreStorageCommand(TestCase):
    """Test suite for trainer graph store storage_command validation."""

    def test_valid_storage_command(self):
        """Test that a valid storage_command passes validation."""
        gbml_config = _create_gbml_config_with_both_graph_stores()
        # Should not raise any exception
        check_if_trainer_graph_store_storage_command_valid(gbml_config)

    def test_missing_storage_command(self):
        """Test that missing storage_command raises an assertion error."""
        gbml_config = _create_gbml_config_with_both_graph_stores(storage_command="")
        with self.assertRaises(AssertionError):
            check_if_trainer_graph_store_storage_command_valid(gbml_config)

    def test_no_graph_store_config(self):
        """Test that no graph store config passes validation (nothing to check)."""
        gbml_config = _create_gbml_config_without_graph_stores()
        # Should not raise any exception - no graph store means nothing to validate
        check_if_trainer_graph_store_storage_command_valid(gbml_config)


class TestInferencerGraphStoreStorageCommand(TestCase):
    """Test suite for inferencer graph store storage_command validation."""

    def test_valid_storage_command(self):
        """Test that a valid storage_command passes validation."""
        gbml_config = _create_gbml_config_with_both_graph_stores()
        # Should not raise any exception
        check_if_inferencer_graph_store_storage_command_valid(gbml_config)

    def test_missing_storage_command(self):
        """Test that missing storage_command raises an assertion error."""
        gbml_config = _create_gbml_config_with_both_graph_stores(storage_command="")
        with self.assertRaises(AssertionError):
            check_if_inferencer_graph_store_storage_command_valid(gbml_config)

    def test_no_graph_store_config(self):
        """Test that no graph store config passes validation (nothing to check)."""
        gbml_config = _create_gbml_config_without_graph_stores()
        # Should not raise any exception - no graph store means nothing to validate
        check_if_inferencer_graph_store_storage_command_valid(gbml_config)


if __name__ == "__main__":
    absltest.main()
