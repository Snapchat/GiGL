import tempfile
import unittest
from pathlib import Path

from gigl.common import UriFactory
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    COMPONENT_TO_SHORTENED_COST_LABEL_MAP,
    GiglResourceConfigWrapper,
)
from snapchat.research.gbml import gigl_resource_config_pb2


class TestGiglResourceConfigWrapper(unittest.TestCase):
    """Test suite for GiglResourceConfigWrapper."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.proto_utils = ProtoUtils()

    def _create_shared_resource_config(
        self,
    ) -> gigl_resource_config_pb2.SharedResourceConfig:
        """Helper to create a valid SharedResourceConfig."""
        config = gigl_resource_config_pb2.SharedResourceConfig()
        config.resource_labels["env"] = "test"
        config.resource_labels["cost_resource_group_tag"] = "unittest_COMPONENT"
        config.resource_labels["cost_resource_group"] = "gigl_test"
        config.common_compute_config.project = "test-project"
        config.common_compute_config.region = "us-central1"
        config.common_compute_config.temp_assets_bucket = "gs://test-temp-bucket"
        config.common_compute_config.temp_regional_assets_bucket = (
            "gs://test-temp-regional-bucket"
        )
        config.common_compute_config.perm_assets_bucket = "gs://test-perm-bucket"
        config.common_compute_config.temp_assets_bq_dataset_name = "test_temp_dataset"
        config.common_compute_config.embedding_bq_dataset_name = (
            "test_embeddings_dataset"
        )
        config.common_compute_config.gcp_service_account_email = (
            "test-sa@test-project.iam.gserviceaccount.com"
        )
        config.common_compute_config.dataflow_runner = "DataflowRunner"
        return config

    def _create_gigl_resource_config_with_direct_shared_config(
        self,
    ) -> gigl_resource_config_pb2.GiglResourceConfig:
        """Helper to create a GiglResourceConfig with direct SharedResourceConfig."""
        config = gigl_resource_config_pb2.GiglResourceConfig()
        config.shared_resource_config.CopyFrom(self._create_shared_resource_config())
        return config

    def _create_gigl_resource_config_with_shared_config_uri(
        self, uri: str
    ) -> gigl_resource_config_pb2.GiglResourceConfig:
        """Helper to create a GiglResourceConfig with SharedResourceConfig URI."""
        config = gigl_resource_config_pb2.GiglResourceConfig()
        config.shared_resource_config_uri = uri
        return config

    def test_shared_resource_config_direct(self):
        """Test loading SharedResourceConfig directly."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        shared_config = wrapper.shared_resource_config
        self.assertEqual(shared_config.common_compute_config.project, "test-project")
        self.assertEqual(shared_config.common_compute_config.region, "us-central1")
        self.assertEqual(shared_config.resource_labels["env"], "test")

    def test_shared_resource_config_from_uri(self):
        """Test loading SharedResourceConfig from URI using a temp file."""
        shared_config = self._create_shared_resource_config()

        # Create a temporary file and write the config
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            uri = UriFactory.create_uri(str(temp_path))
            self.proto_utils.write_proto_to_yaml(shared_config, uri)

        try:
            # Create config with URI
            config = self._create_gigl_resource_config_with_shared_config_uri(
                str(temp_path)
            )
            wrapper = GiglResourceConfigWrapper(resource_config=config)

            loaded_config = wrapper.shared_resource_config
            self.assertEqual(
                loaded_config.common_compute_config.project, "test-project"
            )
            self.assertEqual(loaded_config.common_compute_config.region, "us-central1")
            self.assertEqual(loaded_config.resource_labels["env"], "test")
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)

    def test_shared_resource_config_missing(self):
        """Test that ValueError is raised when no SharedResourceConfig is provided."""
        config = gigl_resource_config_pb2.GiglResourceConfig()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        with self.assertRaises(ValueError) as context:
            _ = wrapper.shared_resource_config

        self.assertIn("SharedResourceConfig", str(context.exception))

    def test_get_resource_labels_no_component(self):
        """Test getting resource labels without component replacement."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        labels = wrapper.get_resource_labels()
        self.assertEqual(labels["env"], "test")
        self.assertEqual(labels["cost_resource_group_tag"], "unittest_na")
        self.assertEqual(labels["cost_resource_group"], "gigl_test")

    def test_get_resource_labels_with_component(self):
        """Test getting resource labels with component replacement."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        labels = wrapper.get_resource_labels(component=GiGLComponents.DataPreprocessor)
        self.assertEqual(labels["cost_resource_group_tag"], "unittest_pre")

        labels = wrapper.get_resource_labels(component=GiGLComponents.SubgraphSampler)
        self.assertEqual(labels["cost_resource_group_tag"], "unittest_sgs")

        labels = wrapper.get_resource_labels(component=GiGLComponents.Trainer)
        self.assertEqual(labels["cost_resource_group_tag"], "unittest_tra")

    def test_get_resource_labels_custom_replacement_key(self):
        """Test getting resource labels with custom replacement key."""
        config = gigl_resource_config_pb2.GiglResourceConfig()
        config.shared_resource_config.CopyFrom(self._create_shared_resource_config())
        config.shared_resource_config.resource_labels["custom_key"] = "value_CUSTOM"

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        labels = wrapper.get_resource_labels(
            component=GiGLComponents.DataPreprocessor, replacement_key="CUSTOM"
        )
        self.assertEqual(labels["custom_key"], "value_pre")

    def test_get_resource_labels_formatted_for_dataflow(self):
        """Test getting resource labels formatted for Dataflow."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        labels = wrapper.get_resource_labels_formatted_for_dataflow(
            component=GiGLComponents.DataPreprocessor
        )

        self.assertIsInstance(labels, list)
        self.assertIn("env=test", labels)
        self.assertIn("cost_resource_group_tag=unittest_pre", labels)
        self.assertIn("cost_resource_group=gigl_test", labels)

    def test_project_property(self):
        """Test project property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        self.assertEqual(wrapper.project, "test-project")

    def test_service_account_email_property(self):
        """Test service_account_email property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        self.assertEqual(
            wrapper.service_account_email,
            "test-sa@test-project.iam.gserviceaccount.com",
        )

    def test_temp_assets_bucket_path_property(self):
        """Test temp_assets_bucket_path property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        bucket_path = wrapper.temp_assets_bucket_path
        self.assertEqual(str(bucket_path), "gs://test-temp-bucket")

    def test_temp_assets_regional_bucket_path_property(self):
        """Test temp_assets_regional_bucket_path property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        bucket_path = wrapper.temp_assets_regional_bucket_path
        self.assertEqual(str(bucket_path), "gs://test-temp-regional-bucket")

    def test_perm_assets_bucket_path_property(self):
        """Test perm_assets_bucket_path property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        bucket_path = wrapper.perm_assets_bucket_path
        self.assertEqual(str(bucket_path), "gs://test-perm-bucket")

    def test_temp_assets_bq_dataset_name_property(self):
        """Test temp_assets_bq_dataset_name property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        self.assertEqual(wrapper.temp_assets_bq_dataset_name, "test_temp_dataset")

    def test_embedding_bq_dataset_name_property(self):
        """Test embedding_bq_dataset_name property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        self.assertEqual(wrapper.embedding_bq_dataset_name, "test_embeddings_dataset")

    def test_dataflow_runner_property(self):
        """Test dataflow_runner property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        self.assertEqual(wrapper.dataflow_runner, "DataflowRunner")

    def test_region_property(self):
        """Test region property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        self.assertEqual(wrapper.region, "us-central1")

    def test_trainer_config_vertex_ai(self):
        """Test trainer_config with Vertex AI configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.trainer_resource_config.vertex_ai_trainer_config.machine_type = (
            "n1-standard-8"
        )
        config.trainer_resource_config.vertex_ai_trainer_config.gpu_type = (
            "NVIDIA_TESLA_V100"
        )
        config.trainer_resource_config.vertex_ai_trainer_config.gpu_limit = 2
        config.trainer_resource_config.vertex_ai_trainer_config.num_replicas = 4

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        trainer_config = wrapper.trainer_config

        assert isinstance(
            trainer_config, gigl_resource_config_pb2.VertexAiResourceConfig
        )
        self.assertEqual(trainer_config.machine_type, "n1-standard-8")
        self.assertEqual(trainer_config.gpu_type, "NVIDIA_TESLA_V100")
        self.assertEqual(trainer_config.gpu_limit, 2)
        self.assertEqual(trainer_config.num_replicas, 4)

    def test_trainer_config_kfp(self):
        """Test trainer_config with KFP configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.trainer_resource_config.kfp_trainer_config.cpu_request = "4"
        config.trainer_resource_config.kfp_trainer_config.memory_request = "16Gi"
        config.trainer_resource_config.kfp_trainer_config.gpu_type = "nvidia.com/gpu"
        config.trainer_resource_config.kfp_trainer_config.gpu_limit = 1
        config.trainer_resource_config.kfp_trainer_config.num_replicas = 2

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        trainer_config = wrapper.trainer_config

        assert isinstance(trainer_config, gigl_resource_config_pb2.KFPResourceConfig)
        self.assertEqual(trainer_config.cpu_request, "4")
        self.assertEqual(trainer_config.memory_request, "16Gi")
        self.assertEqual(trainer_config.num_replicas, 2)

    def test_trainer_config_local(self):
        """Test trainer_config with Local configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.trainer_resource_config.local_trainer_config.num_workers = 4

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        trainer_config = wrapper.trainer_config

        assert isinstance(trainer_config, gigl_resource_config_pb2.LocalResourceConfig)
        self.assertEqual(trainer_config.num_workers, 4)

    def test_trainer_config_vertex_ai_graph_store(self):
        """Test trainer_config with Vertex AI Graph Store configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.trainer_resource_config.vertex_ai_graph_store_trainer_config.compute_pool.machine_type = (
            "n1-highmem-8"
        )
        config.trainer_resource_config.vertex_ai_graph_store_trainer_config.graph_store_pool.machine_type = (
            "n1-standard-4"
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        trainer_config = wrapper.trainer_config

        assert isinstance(
            trainer_config, gigl_resource_config_pb2.VertexAiGraphStoreConfig
        )
        self.assertEqual(trainer_config.compute_pool.machine_type, "n1-highmem-8")
        self.assertEqual(trainer_config.graph_store_pool.machine_type, "n1-standard-4")

    def test_trainer_config_deprecated_vertex_ai(self):
        """Test deprecated trainer_config with Vertex AI configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.trainer_config.vertex_ai_trainer_config.machine_type = "n1-standard-4"
        config.trainer_config.vertex_ai_trainer_config.gpu_type = "NVIDIA_TESLA_T4"
        config.trainer_config.vertex_ai_trainer_config.gpu_limit = 1
        config.trainer_config.vertex_ai_trainer_config.num_replicas = 2

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        trainer_config = wrapper.trainer_config

        assert isinstance(
            trainer_config, gigl_resource_config_pb2.VertexAiResourceConfig
        )
        self.assertEqual(trainer_config.machine_type, "n1-standard-4")
        self.assertEqual(trainer_config.gpu_type, "NVIDIA_TESLA_T4")

    def test_trainer_config_deprecated_kfp(self):
        """Test deprecated trainer_config with KFP configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.trainer_config.kfp_trainer_config.cpu_request = "2"
        config.trainer_config.kfp_trainer_config.memory_request = "8Gi"
        config.trainer_config.kfp_trainer_config.num_replicas = 1

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        trainer_config = wrapper.trainer_config

        assert isinstance(trainer_config, gigl_resource_config_pb2.KFPResourceConfig)
        self.assertEqual(trainer_config.cpu_request, "2")
        self.assertEqual(trainer_config.memory_request, "8Gi")

    def test_trainer_config_deprecated_local(self):
        """Test deprecated trainer_config with Local configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.trainer_config.local_trainer_config.num_workers = 2

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        trainer_config = wrapper.trainer_config

        assert isinstance(trainer_config, gigl_resource_config_pb2.LocalResourceConfig)
        self.assertEqual(trainer_config.num_workers, 2)

    def test_trainer_config_missing(self):
        """Test that ValueError is raised when trainer config is missing."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        with self.assertRaises(ValueError) as context:
            _ = wrapper.trainer_config

        self.assertIn("Trainer config not found", str(context.exception))

    def test_vertex_ai_trainer_region_default(self):
        """Test vertex_ai_trainer_region returns default region."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.trainer_resource_config.vertex_ai_trainer_config.machine_type = (
            "n1-standard-4"
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.vertex_ai_trainer_region, "us-central1")

    def test_vertex_ai_trainer_region_override(self):
        """Test vertex_ai_trainer_region with region override."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.trainer_resource_config.vertex_ai_trainer_config.machine_type = (
            "n1-standard-4"
        )
        config.trainer_resource_config.vertex_ai_trainer_config.gcp_region_override = (
            "us-west1"
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.vertex_ai_trainer_region, "us-west1")

    def test_vertex_ai_trainer_region_non_vertex_ai_error(self):
        """Test vertex_ai_trainer_region raises error for non-Vertex AI trainer."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.trainer_resource_config.local_trainer_config.num_workers = 2

        wrapper = GiglResourceConfigWrapper(resource_config=config)

        with self.assertRaises(ValueError) as context:
            _ = wrapper.vertex_ai_trainer_region

        self.assertIn("only supported for Vertex AI trainers", str(context.exception))

    def test_inferencer_config_dataflow(self):
        """Test inferencer_config with Dataflow configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.inferencer_resource_config.dataflow_inferencer_config.num_workers = 10
        config.inferencer_resource_config.dataflow_inferencer_config.max_num_workers = (
            20
        )
        config.inferencer_resource_config.dataflow_inferencer_config.machine_type = (
            "n1-standard-4"
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        inferencer_config = wrapper.inferencer_config

        assert isinstance(
            inferencer_config, gigl_resource_config_pb2.DataflowResourceConfig
        )
        self.assertEqual(inferencer_config.num_workers, 10)
        self.assertEqual(inferencer_config.max_num_workers, 20)

    def test_inferencer_config_vertex_ai(self):
        """Test inferencer_config with Vertex AI configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.inferencer_resource_config.vertex_ai_inferencer_config.machine_type = (
            "n1-standard-8"
        )
        config.inferencer_resource_config.vertex_ai_inferencer_config.num_replicas = 3

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        inferencer_config = wrapper.inferencer_config

        assert isinstance(
            inferencer_config, gigl_resource_config_pb2.VertexAiResourceConfig
        )
        self.assertEqual(inferencer_config.machine_type, "n1-standard-8")
        self.assertEqual(inferencer_config.num_replicas, 3)

    def test_inferencer_config_local(self):
        """Test inferencer_config with Local configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.inferencer_resource_config.local_inferencer_config.num_workers = 4

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        inferencer_config = wrapper.inferencer_config

        assert isinstance(
            inferencer_config, gigl_resource_config_pb2.LocalResourceConfig
        )
        self.assertEqual(inferencer_config.num_workers, 4)

    def test_inferencer_config_vertex_ai_graph_store(self):
        """Test inferencer_config with Vertex AI Graph Store configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.inferencer_resource_config.vertex_ai_graph_store_inferencer_config.compute_pool.machine_type = (
            "n1-highmem-4"
        )
        config.inferencer_resource_config.vertex_ai_graph_store_inferencer_config.graph_store_pool.machine_type = (
            "n1-standard-2"
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        inferencer_config = wrapper.inferencer_config

        assert isinstance(
            inferencer_config, gigl_resource_config_pb2.VertexAiGraphStoreConfig
        )
        self.assertEqual(inferencer_config.compute_pool.machine_type, "n1-highmem-4")
        self.assertEqual(
            inferencer_config.graph_store_pool.machine_type, "n1-standard-2"
        )

    def test_inferencer_config_deprecated(self):
        """Test deprecated inferencer_config with Dataflow configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.inferencer_config.num_workers = 15
        config.inferencer_config.max_num_workers = 30
        config.inferencer_config.machine_type = "n1-standard-2"

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        inferencer_config = wrapper.inferencer_config

        assert isinstance(
            inferencer_config, gigl_resource_config_pb2.DataflowResourceConfig
        )
        self.assertEqual(inferencer_config.num_workers, 15)
        self.assertEqual(inferencer_config.max_num_workers, 30)

    def test_inferencer_config_missing(self):
        """Test that ValueError is raised when inferencer config is missing."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        with self.assertRaises(ValueError) as context:
            _ = wrapper.inferencer_config

        self.assertIn("Inferencer config not found", str(context.exception))

    def test_vertex_ai_inferencer_region_default(self):
        """Test vertex_ai_inferencer_region returns default region."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.inferencer_resource_config.vertex_ai_inferencer_config.machine_type = (
            "n1-standard-4"
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.vertex_ai_inferencer_region, "us-central1")

    def test_vertex_ai_inferencer_region_override(self):
        """Test vertex_ai_inferencer_region with region override."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.inferencer_resource_config.vertex_ai_inferencer_config.machine_type = (
            "n1-standard-4"
        )
        config.inferencer_resource_config.vertex_ai_inferencer_config.gcp_region_override = (
            "us-east1"
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.vertex_ai_inferencer_region, "us-east1")

    def test_vertex_ai_inferencer_region_non_vertex_ai_error(self):
        """Test vertex_ai_inferencer_region raises error for non-Vertex AI inferencer."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.inferencer_resource_config.local_inferencer_config.num_workers = 2

        wrapper = GiglResourceConfigWrapper(resource_config=config)

        with self.assertRaises(ValueError) as context:
            _ = wrapper.vertex_ai_inferencer_region

        self.assertIn(
            "only supported for Vertex AI inferencers", str(context.exception)
        )

    def test_preprocessor_config(self):
        """Test preprocessor_config property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.preprocessor_config.edge_preprocessor_config.num_workers = 5
        config.preprocessor_config.edge_preprocessor_config.machine_type = (
            "n1-standard-2"
        )
        config.preprocessor_config.node_preprocessor_config.num_workers = 3
        config.preprocessor_config.node_preprocessor_config.machine_type = (
            "n1-standard-1"
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        preprocessor_config = wrapper.preprocessor_config

        assert isinstance(
            preprocessor_config, gigl_resource_config_pb2.DataPreprocessorConfig
        )
        self.assertEqual(preprocessor_config.edge_preprocessor_config.num_workers, 5)
        self.assertEqual(preprocessor_config.node_preprocessor_config.num_workers, 3)

    def test_subgraph_sampler_config(self):
        """Test subgraph_sampler_config property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.subgraph_sampler_config.machine_type = "n1-standard-16"
        config.subgraph_sampler_config.num_replicas = 10
        config.subgraph_sampler_config.num_local_ssds = 4

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        sampler_config = wrapper.subgraph_sampler_config

        assert isinstance(sampler_config, gigl_resource_config_pb2.SparkResourceConfig)
        self.assertEqual(sampler_config.machine_type, "n1-standard-16")
        self.assertEqual(sampler_config.num_replicas, 10)

    def test_split_generator_config(self):
        """Test split_generator_config property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        config.split_generator_config.machine_type = "n1-standard-8"
        config.split_generator_config.num_replicas = 5

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        split_gen_config = wrapper.split_generator_config

        assert isinstance(
            split_gen_config, gigl_resource_config_pb2.SparkResourceConfig
        )
        self.assertEqual(split_gen_config.machine_type, "n1-standard-8")
        self.assertEqual(split_gen_config.num_replicas, 5)

    def test_component_to_shortened_cost_label_map(self):
        """Test that COMPONENT_TO_SHORTENED_COST_LABEL_MAP contains expected mappings."""
        self.assertEqual(
            COMPONENT_TO_SHORTENED_COST_LABEL_MAP[GiGLComponents.DataPreprocessor],
            "pre",
        )
        self.assertEqual(
            COMPONENT_TO_SHORTENED_COST_LABEL_MAP[GiGLComponents.SubgraphSampler], "sgs"
        )
        self.assertEqual(
            COMPONENT_TO_SHORTENED_COST_LABEL_MAP[GiGLComponents.SplitGenerator], "spl"
        )
        self.assertEqual(
            COMPONENT_TO_SHORTENED_COST_LABEL_MAP[GiGLComponents.Trainer], "tra"
        )
        self.assertEqual(
            COMPONENT_TO_SHORTENED_COST_LABEL_MAP[GiGLComponents.Inferencer], "inf"
        )
        self.assertEqual(
            COMPONENT_TO_SHORTENED_COST_LABEL_MAP[GiGLComponents.PostProcessor], "pos"
        )


if __name__ == "__main__":
    unittest.main()
