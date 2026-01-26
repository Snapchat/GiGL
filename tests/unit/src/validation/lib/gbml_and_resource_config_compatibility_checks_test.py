import unittest

from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.validation_check.libs.gbml_and_resource_config_compatibility_checks import (
    check_inferencer_graph_store_compatibility,
    check_trainer_graph_store_compatibility,
)
from snapchat.research.gbml import gbml_config_pb2, gigl_resource_config_pb2

# Helper functions for creating VertexAiGraphStoreConfig


def _create_vertex_ai_graph_store_config() -> (
    gigl_resource_config_pb2.VertexAiGraphStoreConfig
):
    """Create a valid VertexAiGraphStoreConfig."""
    config = gigl_resource_config_pb2.VertexAiGraphStoreConfig()
    # Graph store pool
    config.graph_store_pool.machine_type = "n1-highmem-8"
    config.graph_store_pool.gpu_type = "ACCELERATOR_TYPE_UNSPECIFIED"
    config.graph_store_pool.gpu_limit = 0
    config.graph_store_pool.num_replicas = 2
    # Compute pool
    config.compute_pool.machine_type = "n1-standard-16"
    config.compute_pool.gpu_type = "NVIDIA_TESLA_T4"
    config.compute_pool.gpu_limit = 2
    config.compute_pool.num_replicas = 3
    return config


def _create_vertex_ai_resource_config() -> (
    gigl_resource_config_pb2.VertexAiResourceConfig
):
    """Create a valid VertexAiResourceConfig (non-graph store)."""
    config = gigl_resource_config_pb2.VertexAiResourceConfig()
    config.machine_type = "n1-standard-16"
    config.gpu_type = "NVIDIA_TESLA_T4"
    config.gpu_limit = 2
    config.num_replicas = 3
    return config


def _create_shared_resource_config(
    config: gigl_resource_config_pb2.GiglResourceConfig,
) -> None:
    """Populate shared resource config fields."""
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


# Helper functions for creating GbmlConfig configurations


def _create_gbml_config_with_trainer_graph_store(
    storage_command: str = "python -m gigl.distributed.graph_store.storage_main",
) -> GbmlConfigPbWrapper:
    """Create a GbmlConfig with graph_store_storage_config set for trainer."""
    gbml_config = gbml_config_pb2.GbmlConfig()
    gbml_config.trainer_config.graph_store_storage_config.command = storage_command
    return GbmlConfigPbWrapper(gbml_config_pb=gbml_config)


def _create_gbml_config_without_trainer_graph_store() -> GbmlConfigPbWrapper:
    """Create a GbmlConfig without graph_store_storage_config for trainer."""
    gbml_config = gbml_config_pb2.GbmlConfig()
    gbml_config.trainer_config.trainer_args["some_arg"] = "some_value"
    return GbmlConfigPbWrapper(gbml_config_pb=gbml_config)


def _create_gbml_config_with_inferencer_graph_store(
    storage_command: str = "python -m gigl.distributed.graph_store.storage_main",
) -> GbmlConfigPbWrapper:
    """Create a GbmlConfig with graph_store_storage_config set for inferencer."""
    gbml_config = gbml_config_pb2.GbmlConfig()
    gbml_config.inferencer_config.graph_store_storage_config.command = storage_command
    return GbmlConfigPbWrapper(gbml_config_pb=gbml_config)


def _create_gbml_config_without_inferencer_graph_store() -> GbmlConfigPbWrapper:
    """Create a GbmlConfig without graph_store_storage_config for inferencer."""
    gbml_config = gbml_config_pb2.GbmlConfig()
    gbml_config.inferencer_config.inferencer_args["some_arg"] = "some_value"
    return GbmlConfigPbWrapper(gbml_config_pb=gbml_config)


def _create_gbml_config_with_both_graph_stores(
    storage_command: str = "python -m gigl.distributed.graph_store.storage_main",
) -> GbmlConfigPbWrapper:
    """Create a GbmlConfig with graph_store_storage_config for both trainer and inferencer."""
    gbml_config = gbml_config_pb2.GbmlConfig()
    gbml_config.trainer_config.graph_store_storage_config.command = storage_command
    gbml_config.inferencer_config.graph_store_storage_config.command = storage_command
    return GbmlConfigPbWrapper(gbml_config_pb=gbml_config)


def _create_gbml_config_without_graph_stores() -> GbmlConfigPbWrapper:
    """Create a GbmlConfig without graph_store_storage_config for both trainer and inferencer."""
    gbml_config = gbml_config_pb2.GbmlConfig()
    gbml_config.trainer_config.trainer_args["some_arg"] = "some_value"
    gbml_config.inferencer_config.inferencer_args["some_arg"] = "some_value"
    return GbmlConfigPbWrapper(gbml_config_pb=gbml_config)


# Helper functions for creating GiglResourceConfig configurations


def _create_resource_config_with_trainer_graph_store() -> GiglResourceConfigWrapper:
    """Create a GiglResourceConfig with VertexAiGraphStoreConfig for trainer."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    _create_shared_resource_config(config)

    # Trainer with VertexAiGraphStoreConfig
    config.trainer_resource_config.vertex_ai_graph_store_trainer_config.CopyFrom(
        _create_vertex_ai_graph_store_config()
    )
    # Inferencer with standard config
    config.inferencer_resource_config.vertex_ai_inferencer_config.CopyFrom(
        _create_vertex_ai_resource_config()
    )
    return GiglResourceConfigWrapper(resource_config=config)


def _create_resource_config_without_trainer_graph_store() -> GiglResourceConfigWrapper:
    """Create a GiglResourceConfig without VertexAiGraphStoreConfig for trainer."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    _create_shared_resource_config(config)

    # Trainer with standard config
    config.trainer_resource_config.vertex_ai_trainer_config.CopyFrom(
        _create_vertex_ai_resource_config()
    )
    # Inferencer with standard config
    config.inferencer_resource_config.vertex_ai_inferencer_config.CopyFrom(
        _create_vertex_ai_resource_config()
    )
    return GiglResourceConfigWrapper(resource_config=config)


def _create_resource_config_with_inferencer_graph_store() -> GiglResourceConfigWrapper:
    """Create a GiglResourceConfig with VertexAiGraphStoreConfig for inferencer."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    _create_shared_resource_config(config)

    # Trainer with standard config
    config.trainer_resource_config.vertex_ai_trainer_config.CopyFrom(
        _create_vertex_ai_resource_config()
    )
    # Inferencer with VertexAiGraphStoreConfig
    config.inferencer_resource_config.vertex_ai_graph_store_inferencer_config.CopyFrom(
        _create_vertex_ai_graph_store_config()
    )
    return GiglResourceConfigWrapper(resource_config=config)


def _create_resource_config_without_inferencer_graph_store() -> (
    GiglResourceConfigWrapper
):
    """Create a GiglResourceConfig without VertexAiGraphStoreConfig for inferencer."""
    return _create_resource_config_without_trainer_graph_store()


def _create_resource_config_with_both_graph_stores() -> GiglResourceConfigWrapper:
    """Create a GiglResourceConfig with VertexAiGraphStoreConfig for both trainer and inferencer."""
    config = gigl_resource_config_pb2.GiglResourceConfig()
    _create_shared_resource_config(config)

    # Trainer with VertexAiGraphStoreConfig
    config.trainer_resource_config.vertex_ai_graph_store_trainer_config.CopyFrom(
        _create_vertex_ai_graph_store_config()
    )
    # Inferencer with VertexAiGraphStoreConfig
    config.inferencer_resource_config.vertex_ai_graph_store_inferencer_config.CopyFrom(
        _create_vertex_ai_graph_store_config()
    )
    return GiglResourceConfigWrapper(resource_config=config)


def _create_resource_config_without_graph_stores() -> GiglResourceConfigWrapper:
    """Create a GiglResourceConfig without VertexAiGraphStoreConfig for both trainer and inferencer."""
    return _create_resource_config_without_trainer_graph_store()


# Test Classes


class TestTrainerGraphStoreCompatibility(unittest.TestCase):
    """Test suite for trainer graph store compatibility checks."""

    def test_both_have_trainer_graph_store(self):
        """Test that both configs having trainer graph store passes validation."""
        gbml_config = _create_gbml_config_with_trainer_graph_store()
        resource_config = _create_resource_config_with_trainer_graph_store()
        # Should not raise any exception
        check_trainer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_neither_has_trainer_graph_store(self):
        """Test that neither config having trainer graph store passes validation."""
        gbml_config = _create_gbml_config_without_trainer_graph_store()
        resource_config = _create_resource_config_without_trainer_graph_store()
        # Should not raise any exception
        check_trainer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_template_has_trainer_graph_store_resource_does_not(self):
        """Test that template having graph store but resource not raises an assertion error."""
        gbml_config = _create_gbml_config_with_trainer_graph_store()
        resource_config = _create_resource_config_without_trainer_graph_store()
        with self.assertRaises(AssertionError):
            check_trainer_graph_store_compatibility(
                gbml_config_pb_wrapper=gbml_config,
                resource_config_wrapper=resource_config,
            )

    def test_resource_has_trainer_graph_store_template_does_not(self):
        """Test that resource having graph store but template not raises an assertion error."""
        gbml_config = _create_gbml_config_without_trainer_graph_store()
        resource_config = _create_resource_config_with_trainer_graph_store()
        with self.assertRaises(AssertionError):
            check_trainer_graph_store_compatibility(
                gbml_config_pb_wrapper=gbml_config,
                resource_config_wrapper=resource_config,
            )


class TestInferencerGraphStoreCompatibility(unittest.TestCase):
    """Test suite for inferencer graph store compatibility checks."""

    def test_both_have_inferencer_graph_store(self):
        """Test that both configs having inferencer graph store passes validation."""
        gbml_config = _create_gbml_config_with_inferencer_graph_store()
        resource_config = _create_resource_config_with_inferencer_graph_store()
        # Should not raise any exception
        check_inferencer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_neither_has_inferencer_graph_store(self):
        """Test that neither config having inferencer graph store passes validation."""
        gbml_config = _create_gbml_config_without_inferencer_graph_store()
        resource_config = _create_resource_config_without_inferencer_graph_store()
        # Should not raise any exception
        check_inferencer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_template_has_inferencer_graph_store_resource_does_not(self):
        """Test that template having graph store but resource not raises an assertion error."""
        gbml_config = _create_gbml_config_with_inferencer_graph_store()
        resource_config = _create_resource_config_without_inferencer_graph_store()
        with self.assertRaises(AssertionError):
            check_inferencer_graph_store_compatibility(
                gbml_config_pb_wrapper=gbml_config,
                resource_config_wrapper=resource_config,
            )

    def test_resource_has_inferencer_graph_store_template_does_not(self):
        """Test that resource having graph store but template not raises an assertion error."""
        gbml_config = _create_gbml_config_without_inferencer_graph_store()
        resource_config = _create_resource_config_with_inferencer_graph_store()
        with self.assertRaises(AssertionError):
            check_inferencer_graph_store_compatibility(
                gbml_config_pb_wrapper=gbml_config,
                resource_config_wrapper=resource_config,
            )


class TestMixedGraphStoreConfigurations(unittest.TestCase):
    """Test suite for mixed graph store configuration scenarios."""

    def test_both_have_all_graph_stores(self):
        """Test that both configs having all graph stores passes validation."""
        gbml_config = _create_gbml_config_with_both_graph_stores()
        resource_config = _create_resource_config_with_both_graph_stores()
        # Should not raise any exception for trainer
        check_trainer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )
        # Should not raise any exception for inferencer
        check_inferencer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_neither_has_any_graph_stores(self):
        """Test that neither config having any graph stores passes validation."""
        gbml_config = _create_gbml_config_without_graph_stores()
        resource_config = _create_resource_config_without_graph_stores()
        # Should not raise any exception for trainer
        check_trainer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )
        # Should not raise any exception for inferencer
        check_inferencer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_trainer_graph_store_only_compatible(self):
        """Test trainer graph store only configuration is compatible."""
        gbml_config = _create_gbml_config_with_trainer_graph_store()
        resource_config = _create_resource_config_with_trainer_graph_store()
        # Should not raise any exception for trainer
        check_trainer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )
        # Should not raise any exception for inferencer (neither has it)
        check_inferencer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_inferencer_graph_store_only_compatible(self):
        """Test inferencer graph store only configuration is compatible."""
        gbml_config = _create_gbml_config_with_inferencer_graph_store()
        resource_config = _create_resource_config_with_inferencer_graph_store()
        # Should not raise any exception for trainer (neither has it)
        check_trainer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )
        # Should not raise any exception for inferencer
        check_inferencer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_template_has_both_resource_has_trainer_only(self):
        """Test that template having both but resource having only trainer raises an error for inferencer."""
        gbml_config = _create_gbml_config_with_both_graph_stores()
        resource_config = _create_resource_config_with_trainer_graph_store()
        # Should not raise any exception for trainer
        check_trainer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )
        # Should raise an assertion error for inferencer
        with self.assertRaises(AssertionError):
            check_inferencer_graph_store_compatibility(
                gbml_config_pb_wrapper=gbml_config,
                resource_config_wrapper=resource_config,
            )

    def test_template_has_both_resource_has_inferencer_only(self):
        """Test that template having both but resource having only inferencer raises an error for trainer."""
        gbml_config = _create_gbml_config_with_both_graph_stores()
        resource_config = _create_resource_config_with_inferencer_graph_store()
        # Should raise an assertion error for trainer
        with self.assertRaises(AssertionError):
            check_trainer_graph_store_compatibility(
                gbml_config_pb_wrapper=gbml_config,
                resource_config_wrapper=resource_config,
            )
        # Should not raise any exception for inferencer
        check_inferencer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )


if __name__ == "__main__":
    unittest.main()
