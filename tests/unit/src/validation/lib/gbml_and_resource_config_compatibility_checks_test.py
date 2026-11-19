from absl.testing import absltest

from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.validation_check.config_validator import (
    _run_gbml_and_resource_config_compatibility_checks,
)
from gigl.src.validation_check.libs.gbml_and_resource_config_compatibility_checks import (
    check_inferencer_graph_store_compatibility,
    check_trainer_graph_store_compatibility,
)
from snapchat.research.gbml import gbml_config_pb2, gigl_resource_config_pb2
from tests.test_assets.test_case import TestCase

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
    """Create a GiglResourceConfig without VertexAiGraphStoreConfig for trainer or inferencer."""
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


class TestTrainerGraphStoreCompatibility(TestCase):
    """Test suite for trainer graph store compatibility checks."""

    def test_both_have_trainer_graph_store(self):
        """Test that both configs having trainer graph store passes validation."""
        gbml_config = _create_gbml_config_with_both_graph_stores()
        resource_config = _create_resource_config_with_both_graph_stores()
        # Should not raise any exception
        check_trainer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_neither_has_trainer_graph_store(self):
        """Test that neither config having trainer graph store passes validation."""
        gbml_config = _create_gbml_config_without_graph_stores()
        resource_config = _create_resource_config_without_graph_stores()
        # Should not raise any exception
        check_trainer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_template_has_trainer_graph_store_resource_does_not(self):
        """Test that template having graph store but resource not raises an assertion error."""
        gbml_config = _create_gbml_config_with_both_graph_stores()
        resource_config = _create_resource_config_without_graph_stores()
        with self.assertRaises(AssertionError):
            check_trainer_graph_store_compatibility(
                gbml_config_pb_wrapper=gbml_config,
                resource_config_wrapper=resource_config,
            )

    def test_resource_has_trainer_graph_store_template_does_not(self):
        """Test that resource having graph store but template not raises an assertion error."""
        gbml_config = _create_gbml_config_without_graph_stores()
        resource_config = _create_resource_config_with_both_graph_stores()
        with self.assertRaises(AssertionError):
            check_trainer_graph_store_compatibility(
                gbml_config_pb_wrapper=gbml_config,
                resource_config_wrapper=resource_config,
            )


class TestInferencerGraphStoreCompatibility(TestCase):
    """Test suite for inferencer graph store compatibility checks."""

    def test_both_have_inferencer_graph_store(self):
        """Test that both configs having inferencer graph store passes validation."""
        gbml_config = _create_gbml_config_with_both_graph_stores()
        resource_config = _create_resource_config_with_both_graph_stores()
        # Should not raise any exception
        check_inferencer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_neither_has_inferencer_graph_store(self):
        """Test that neither config having inferencer graph store passes validation."""
        gbml_config = _create_gbml_config_without_graph_stores()
        resource_config = _create_resource_config_without_graph_stores()
        # Should not raise any exception
        check_inferencer_graph_store_compatibility(
            gbml_config_pb_wrapper=gbml_config,
            resource_config_wrapper=resource_config,
        )

    def test_template_has_inferencer_graph_store_resource_does_not(self):
        """Test that template having graph store but resource not raises an assertion error."""
        gbml_config = _create_gbml_config_with_both_graph_stores()
        resource_config = _create_resource_config_without_graph_stores()
        with self.assertRaises(AssertionError):
            check_inferencer_graph_store_compatibility(
                gbml_config_pb_wrapper=gbml_config,
                resource_config_wrapper=resource_config,
            )

    def test_resource_has_inferencer_graph_store_template_does_not(self):
        """Test that resource having graph store but template not raises an assertion error."""
        gbml_config = _create_gbml_config_without_graph_stores()
        resource_config = _create_resource_config_with_both_graph_stores()
        with self.assertRaises(AssertionError):
            check_inferencer_graph_store_compatibility(
                gbml_config_pb_wrapper=gbml_config,
                resource_config_wrapper=resource_config,
            )


class TestGraphStoreCompatibilityDispatcher(TestCase):
    """Test graph store compatibility through the pipeline dispatcher."""

    def test_custom_launchers_own_graph_store_compatibility(self) -> None:
        resource_config = _create_resource_config_without_graph_stores()
        resource_config.resource_config.trainer_resource_config.custom_trainer_config.command = "train"
        resource_config.resource_config.inferencer_resource_config.custom_inferencer_config.command = "infer"
        for gbml_config, has_graph_store in (
            (_create_gbml_config_with_both_graph_stores(), True),
            (_create_gbml_config_without_graph_stores(), False),
        ):
            with self.subTest(has_graph_store=has_graph_store):
                _run_gbml_and_resource_config_compatibility_checks(
                    start_at="config_populator",
                    stop_after=None,
                    gbml_config_pb_wrapper=gbml_config,
                    resource_config_wrapper=resource_config,
                )

    def test_custom_launcher_does_not_skip_other_component_validation(self) -> None:
        gbml_config = _create_gbml_config_with_both_graph_stores()
        for component in ("trainer", "inferencer"):
            with self.subTest(component=component):
                resource_config = _create_resource_config_without_graph_stores()
                if component == "trainer":
                    resource_config.resource_config.trainer_resource_config.custom_trainer_config.command = "train"
                else:
                    resource_config.resource_config.inferencer_resource_config.custom_inferencer_config.command = "infer"
                with self.assertRaises(AssertionError):
                    _run_gbml_and_resource_config_compatibility_checks(
                        start_at="config_populator",
                        stop_after=None,
                        gbml_config_pb_wrapper=gbml_config,
                        resource_config_wrapper=resource_config,
                    )


if __name__ == "__main__":
    absltest.main()
