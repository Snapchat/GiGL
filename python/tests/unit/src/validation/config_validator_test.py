import os
import shutil
import tempfile
import unittest

import google.protobuf.message
from parameterized import param, parameterized

import gigl.env.pipelines_config
from gigl.common import Uri, UriFactory
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.validation_check.config_validator import kfp_validation_checks
from snapchat.research.gbml import (
    gbml_config_pb2,
    gigl_resource_config_pb2,
    graph_schema_pb2,
)

# Shared helper functions for creating proto components


def _create_paper_cites_paper_edge_type() -> graph_schema_pb2.EdgeType:
    """Create an EdgeType proto for paper-cites-paper edges."""
    return graph_schema_pb2.EdgeType(
        src_node_type="paper",
        relation="cites",
        dst_node_type="paper",
    )


def _create_test_graph_metadata() -> graph_schema_pb2.GraphMetadata:
    """Create a GraphMetadata proto for testing."""
    return graph_schema_pb2.GraphMetadata(
        node_types=["paper"],
        edge_types=[_create_paper_cites_paper_edge_type()],
    )


def _create_test_task_metadata() -> gbml_config_pb2.GbmlConfig.TaskMetadata:
    """Create a TaskMetadata proto for node anchor based link prediction."""
    return gbml_config_pb2.GbmlConfig.TaskMetadata(
        node_anchor_based_link_prediction_task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata(
            supervision_edge_types=[_create_paper_cites_paper_edge_type()]
        )
    )


def _create_data_preprocessor_config() -> (
    gbml_config_pb2.GbmlConfig.DatasetConfig.DataPreprocessorConfig
):
    """Create a DataPreprocessorConfig proto for testing."""
    return gbml_config_pb2.GbmlConfig.DatasetConfig.DataPreprocessorConfig(
        data_preprocessor_config_cls_path="gigl.src.mocking.mocking_assets.passthrough_preprocessor_config_for_mocked_assets.PassthroughPreprocessorConfigForMockedAssets",
        data_preprocessor_args={
            "mocked_dataset_name": "cora_homogeneous_node_anchor_edge_features"
        },
    )


def _create_test_common_compute_config() -> (
    gigl_resource_config_pb2.SharedResourceConfig.CommonComputeConfig
):
    """Create a CommonComputeConfig proto for testing."""
    return gigl_resource_config_pb2.SharedResourceConfig.CommonComputeConfig(
        project="test-project",
        region="us-central1",
        temp_assets_bucket="gs://test-temp",
        temp_regional_assets_bucket="gs://test-temp-regional",
        perm_assets_bucket="gs://test-perm",
        temp_assets_bq_dataset_name="test_dataset",
        embedding_bq_dataset_name="test_embeddings",
        gcp_service_account_email="test@test-project.iam.gserviceaccount.com",
        dataflow_runner="DataflowRunner",
    )


def _create_test_shared_resource_config() -> (
    gigl_resource_config_pb2.SharedResourceConfig
):
    """Create a SharedResourceConfig proto for testing."""
    return gigl_resource_config_pb2.SharedResourceConfig(
        common_compute_config=_create_test_common_compute_config(),
    )


def _create_test_preprocessor_config() -> (
    gigl_resource_config_pb2.DataPreprocessorConfig
):
    """Create a DataPreprocessorConfig proto for testing."""
    return gigl_resource_config_pb2.DataPreprocessorConfig(
        node_preprocessor_config=gigl_resource_config_pb2.DataflowResourceConfig(
            num_workers=10,
            max_num_workers=20,
            disk_size_gb=100,
            machine_type="n1-standard-4",
        ),
        edge_preprocessor_config=gigl_resource_config_pb2.DataflowResourceConfig(
            num_workers=15,
            max_num_workers=25,
            disk_size_gb=150,
            machine_type="n1-standard-8",
        ),
    )


def _create_test_trainer_resource_config() -> (
    gigl_resource_config_pb2.TrainerResourceConfig
):
    """Create a TrainerResourceConfig proto for testing."""
    return gigl_resource_config_pb2.TrainerResourceConfig(
        vertex_ai_trainer_config=gigl_resource_config_pb2.VertexAiResourceConfig(
            machine_type="n1-standard-16",
            gpu_type="NVIDIA_TESLA_T4",
            gpu_limit=2,
            num_replicas=3,
        ),
    )


# Helper functions for creating mock config protos


def _create_valid_offline_subgraph_sampling_task_config() -> gbml_config_pb2.GbmlConfig:
    """Create a task config proto for offline subgraph sampling (without live SGS backend flag)."""
    # Dataset config
    dataset_config = gbml_config_pb2.GbmlConfig.DatasetConfig(
        data_preprocessor_config=_create_data_preprocessor_config(),
        subgraph_sampler_config=gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig(
            num_hops=2,
            num_neighbors_to_sample=10,
            num_positive_samples=1,
        ),
        split_generator_config=gbml_config_pb2.GbmlConfig.DatasetConfig.SplitGeneratorConfig(
            assigner_cls_path="splitgenerator.lib.assigners.TransductiveEdgeToLinkSplitHashingAssigner",
            split_strategy_cls_path="splitgenerator.lib.split_strategies.TransductiveNodeAnchorBasedLinkPredictionSplitStrategy",
            assigner_args={
                "seed": "42",
                "test_split": "0.2",
                "train_split": "0.7",
                "val_split": "0.1",
            },
        ),
    )

    # Trainer config
    trainer_config = gbml_config_pb2.GbmlConfig.TrainerConfig(
        trainer_cls_path="gigl.src.common.modeling_task_specs.node_anchor_based_link_prediction_modeling_task_spec.NodeAnchorBasedLinkPredictionModelingTaskSpec",
    )

    # Inferencer config
    inferencer_config = gbml_config_pb2.GbmlConfig.InferencerConfig(
        inferencer_cls_path="gigl.src.common.modeling_task_specs.node_anchor_based_link_prediction_modeling_task_spec.NodeAnchorBasedLinkPredictionModelingTaskSpec",
    )

    return gbml_config_pb2.GbmlConfig(
        graph_metadata=_create_test_graph_metadata(),
        task_metadata=_create_test_task_metadata(),
        dataset_config=dataset_config,
        trainer_config=trainer_config,
        inferencer_config=inferencer_config,
    )


def _create_valid_live_subgraph_sampling_task_config() -> gbml_config_pb2.GbmlConfig:
    """Create a task config proto for live subgraph sampling (with GLT backend flag enabled)."""
    # Dataset config
    dataset_config = gbml_config_pb2.GbmlConfig.DatasetConfig(
        data_preprocessor_config=_create_data_preprocessor_config(),
    )
    # Trainer config
    trainer_config = gbml_config_pb2.GbmlConfig.TrainerConfig(
        command="python -m examples.link_prediction.homogeneous_training",
    )

    # Inferencer config
    inferencer_config = gbml_config_pb2.GbmlConfig.InferencerConfig(
        command="python -m examples.link_prediction.homogeneous_inference",
    )

    return gbml_config_pb2.GbmlConfig(
        graph_metadata=_create_test_graph_metadata(),
        task_metadata=_create_test_task_metadata(),
        dataset_config=dataset_config,
        trainer_config=trainer_config,
        inferencer_config=inferencer_config,
        feature_flags={"should_run_glt_backend": "True"},
    )


def _create_valid_live_subgraph_sampling_resource_config() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    """Create a resource config proto for GLT/live backend (without subgraph sampler and split generator configs)."""
    # Inferencer resource config
    inferencer_resource_config = gigl_resource_config_pb2.InferencerResourceConfig(
        vertex_ai_inferencer_config=gigl_resource_config_pb2.VertexAiResourceConfig(
            machine_type="n1-standard-4",
            gpu_type="NVIDIA_TESLA_T4",
            gpu_limit=2,
            num_replicas=3,
        ),
    )

    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=_create_test_shared_resource_config(),
        preprocessor_config=_create_test_preprocessor_config(),
        trainer_resource_config=_create_test_trainer_resource_config(),
        inferencer_resource_config=inferencer_resource_config,
    )


def _create_valid_offline_subgraph_sampling_resource_config() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    """Create a resource config proto for offline subgraph sampling (with all component configs)."""
    # Subgraph sampler config
    subgraph_sampler_config = gigl_resource_config_pb2.SparkResourceConfig(
        machine_type="n2d-highmem-16",
        num_local_ssds=2,
        num_replicas=4,
    )

    # Split generator config
    split_generator_config = gigl_resource_config_pb2.SparkResourceConfig(
        machine_type="n2d-standard-16",
        num_local_ssds=2,
        num_replicas=4,
    )

    # Inferencer resource config
    inferencer_resource_config = gigl_resource_config_pb2.InferencerResourceConfig(
        dataflow_inferencer_config=gigl_resource_config_pb2.DataflowResourceConfig(
            num_workers=1,
            max_num_workers=256,
            machine_type="c2d-highmem-32",
            disk_size_gb=100,
        ),
    )

    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=_create_test_shared_resource_config(),
        preprocessor_config=_create_test_preprocessor_config(),
        subgraph_sampler_config=subgraph_sampler_config,
        split_generator_config=split_generator_config,
        trainer_resource_config=_create_test_trainer_resource_config(),
        inferencer_resource_config=inferencer_resource_config,
    )


class TestConfigValidationPerSGSBackends(unittest.TestCase):
    """Test suite for config validation with different SGS backends (live subgraph sampling, offline subgraph sampling)."""

    def setUp(self):
        """Set up temporary directory for test config files."""
        self._temp_dir = tempfile.mkdtemp()
        self._proto_utils = ProtoUtils()

        # Create task config files
        self._live_task_config_uri = self._write_proto_to_file(
            _create_valid_live_subgraph_sampling_task_config(),
            "live_task_config.yaml",
        )
        self._offline_task_config_uri = self._write_proto_to_file(
            _create_valid_offline_subgraph_sampling_task_config(),
            "offline_task_config.yaml",
        )

        # Create resource config files
        self._live_resource_config_uri = self._write_proto_to_file(
            _create_valid_live_subgraph_sampling_resource_config(),
            "live_resource_config.yaml",
        )
        self._offline_resource_config_uri = self._write_proto_to_file(
            _create_valid_offline_subgraph_sampling_resource_config(),
            "offline_resource_config.yaml",
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._temp_dir, ignore_errors=True)
        # Clear the cached resource config to ensure each test loads its own config. Otherwise, the resource config will be cached, leading to
        # incorrect resource configs being used for tests (i.e. live SGS resource config being used for offline SGS tests).
        gigl.env.pipelines_config._resource_config = None

    def _write_proto_to_file(
        self, proto: google.protobuf.message.Message, filename: str
    ) -> Uri:
        """Write proto to a temporary file and return its URI."""
        filepath = os.path.join(self._temp_dir, filename)
        uri = UriFactory.create_uri(filepath)
        self._proto_utils.write_proto_to_yaml(proto, uri)
        return uri

    @parameterized.expand(
        [
            param(
                "Test that live subgraph sampling resource config passes when we are doing live subgraph sampling",
                should_use_live_sgs_backend=True,
                should_use_live_sgs_resource_config=True,
            ),
            param(
                "Test that offline subgraph sampling resource config passes when we are doing offline subgraph sampling",
                should_use_live_sgs_backend=False,
                should_use_live_sgs_resource_config=False,
            ),
            param(
                "Test that offline subgraph sampling resource config passes when we are doing live subgraph sampling",
                # This test is expected to pass because an offline SGS resource config is still valid when live SGS is enabled.
                should_use_live_sgs_backend=True,
                should_use_live_sgs_resource_config=False,
            ),
        ]
    )
    def test_resource_config_validation_success_with_mock_configs(
        self,
        _,
        should_use_live_sgs_backend: bool,
        should_use_live_sgs_resource_config: bool,
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

        kfp_validation_checks(
            job_name="resource_config_validation_test",
            task_config_uri=task_config_uri,
            start_at="config_populator",
            resource_config_uri=resource_config_uri,
        )

    def test_resource_config_validation_failure_with_mock_configs(
        self,
    ) -> None:
        # For this setting, we should expect failure since the live SGS resource config does not have
        # sufficient fields to specify how to do offline subgraph sampling.
        with self.assertRaises(AssertionError):
            kfp_validation_checks(
                job_name="resource_config_validation_test",
                task_config_uri=self._offline_task_config_uri,
                start_at="config_populator",
                resource_config_uri=self._live_resource_config_uri,
            )


if __name__ == "__main__":
    unittest.main()
