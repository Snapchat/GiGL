import copy
import tempfile
from pathlib import Path
from typing import Final

from absl.testing import absltest

from gigl.common import UriFactory
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from snapchat.research.gbml import gigl_resource_config_pb2
from tests.test_assets.test_case import TestCase

_ENV: Final[str] = "test"
_COST_RESOURCE_GROUP_TAG: Final[str] = "unittest_COMPONENT"
_COST_RESOURCE_GROUP: Final[str] = "gigl_test"


def _create_shared_resource_config() -> gigl_resource_config_pb2.SharedResourceConfig:
    """Helper to create a valid SharedResourceConfig."""
    common_compute_config = (
        gigl_resource_config_pb2.SharedResourceConfig.CommonComputeConfig(
            project="test-project",
            region="us-central1",
            temp_assets_bucket="gs://test-temp-bucket",
            temp_regional_assets_bucket="gs://test-temp-regional-bucket",
            perm_assets_bucket="gs://test-perm-bucket",
            temp_assets_bq_dataset_name="test_temp_dataset",
            embedding_bq_dataset_name="test_embeddings_dataset",
            gcp_service_account_email="test-sa@test-project.iam.gserviceaccount.com",
            dataflow_runner="DataflowRunner",
        )
    )

    config = gigl_resource_config_pb2.SharedResourceConfig(
        resource_labels={
            "env": _ENV,
            "cost_resource_group_tag": _COST_RESOURCE_GROUP_TAG,
            "cost_resource_group": _COST_RESOURCE_GROUP,
        },
        common_compute_config=common_compute_config,
    )
    return config


class TestGiglResourceConfigWrapper(TestCase):
    """Test suite for GiglResourceConfigWrapper."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.proto_utils = ProtoUtils()

    def _create_gigl_resource_config_with_direct_shared_config(
        self,
    ) -> gigl_resource_config_pb2.GiglResourceConfig:
        """Helper to create a GiglResourceConfig with direct SharedResourceConfig."""
        config = gigl_resource_config_pb2.GiglResourceConfig()
        config.shared_resource_config.CopyFrom(_create_shared_resource_config())
        return config

    def test_shared_resource_config_direct(self):
        """Test loading SharedResourceConfig directly."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        shared_config = wrapper.shared_resource_config
        self.assertEqual(shared_config, _create_shared_resource_config())

    def test_shared_resource_config_from_uri(self):
        """Test loading SharedResourceConfig from URI using a temp file."""
        shared_config = _create_shared_resource_config()

        # Create a temporary file and write the config
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            uri = UriFactory.create_uri(str(temp_path))
            self.proto_utils.write_proto_to_yaml(shared_config, uri)

        try:
            # Create config with URI
            config = gigl_resource_config_pb2.GiglResourceConfig(
                shared_resource_config_uri=str(temp_path)
            )
            wrapper = GiglResourceConfigWrapper(resource_config=config)

            loaded_config = wrapper.shared_resource_config
            self.assertEqual(loaded_config, _create_shared_resource_config())
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

    def test_get_resource_labels(self):
        """Test getting resource labels without component replacement."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        wrapper = GiglResourceConfigWrapper(resource_config=config)

        # Test without component
        labels = wrapper.get_resource_labels()
        expected_labels = {
            "env": _ENV,
            "cost_resource_group_tag": "unittest_na",
            "cost_resource_group": _COST_RESOURCE_GROUP,
        }
        self.assertEqual(labels, expected_labels)

        # Test with DataPreprocessor component
        labels = wrapper.get_resource_labels(component=GiGLComponents.DataPreprocessor)
        expected_labels = {
            "env": _ENV,
            "cost_resource_group_tag": "unittest_pre",
            "cost_resource_group": _COST_RESOURCE_GROUP,
        }
        self.assertEqual(labels, expected_labels)

        # Test with SubgraphSampler component
        labels = wrapper.get_resource_labels(component=GiGLComponents.SubgraphSampler)
        expected_labels = {
            "env": _ENV,
            "cost_resource_group_tag": "unittest_sgs",
            "cost_resource_group": _COST_RESOURCE_GROUP,
        }
        self.assertEqual(labels, expected_labels)

        # Test with Trainer component
        labels = wrapper.get_resource_labels(component=GiGLComponents.Trainer)
        expected_labels = {
            "env": _ENV,
            "cost_resource_group_tag": "unittest_tra",
            "cost_resource_group": _COST_RESOURCE_GROUP,
        }
        self.assertEqual(labels, expected_labels)

    def test_get_resource_labels_custom_replacement_key(self):
        """Test getting resource labels with custom replacement key."""
        config = gigl_resource_config_pb2.GiglResourceConfig()
        config.shared_resource_config.CopyFrom(_create_shared_resource_config())
        config.shared_resource_config.resource_labels["custom_key"] = "value_CUSTOM"

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        labels = wrapper.get_resource_labels(
            component=GiGLComponents.DataPreprocessor, replacement_key="CUSTOM"
        )
        expected_labels = {
            "env": "test",
            "cost_resource_group_tag": "unittest_COMPONENT",
            "cost_resource_group": "gigl_test",
            "custom_key": "value_pre",
        }
        self.assertEqual(labels, expected_labels)

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

    def test_trainer_config_vertex_ai(self):
        """Test trainer_config with Vertex AI configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        trainer_config = gigl_resource_config_pb2.VertexAiResourceConfig(
            machine_type="n1-standard-8",
            gpu_type="NVIDIA_TESLA_V100",
            gpu_limit=2,
            num_replicas=4,
        )
        config.trainer_resource_config.vertex_ai_trainer_config.CopyFrom(
            copy.deepcopy(trainer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.trainer_config, trainer_config)

    def test_trainer_config_kfp(self):
        """Test trainer_config with KFP configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        trainer_config = gigl_resource_config_pb2.KFPResourceConfig(
            cpu_request="4",
            memory_request="16Gi",
            gpu_type="nvidia.com/gpu",
            gpu_limit=1,
            num_replicas=2,
        )
        config.trainer_resource_config.kfp_trainer_config.CopyFrom(
            copy.deepcopy(trainer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.trainer_config, trainer_config)

    def test_trainer_config_local(self):
        """Test trainer_config with Local configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        trainer_config = gigl_resource_config_pb2.LocalResourceConfig(
            num_workers=4,
        )
        config.trainer_resource_config.local_trainer_config.CopyFrom(
            copy.deepcopy(trainer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.trainer_config, trainer_config)

    def test_trainer_config_vertex_ai_graph_store(self):
        """Test trainer_config with Vertex AI Graph Store configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        trainer_config = gigl_resource_config_pb2.VertexAiGraphStoreConfig(
            compute_pool=gigl_resource_config_pb2.VertexAiResourceConfig(
                machine_type="n1-highmem-8",
            ),
            graph_store_pool=gigl_resource_config_pb2.VertexAiResourceConfig(
                machine_type="n1-standard-4",
            ),
        )
        config.trainer_resource_config.vertex_ai_graph_store_trainer_config.CopyFrom(
            copy.deepcopy(trainer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.trainer_config, trainer_config)

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
        trainer_config = gigl_resource_config_pb2.VertexAiResourceConfig(
            machine_type="n1-standard-4",
        )
        config.trainer_resource_config.vertex_ai_trainer_config.CopyFrom(
            copy.deepcopy(trainer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.vertex_ai_trainer_region, "us-central1")

    def test_vertex_ai_trainer_region_override(self):
        """Test vertex_ai_trainer_region with region override."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        trainer_config = gigl_resource_config_pb2.VertexAiResourceConfig(
            machine_type="n1-standard-4",
            gcp_region_override="us-west1",
        )
        config.trainer_resource_config.vertex_ai_trainer_config.CopyFrom(
            copy.deepcopy(trainer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.vertex_ai_trainer_region, "us-west1")

    def test_vertex_ai_trainer_region_non_vertex_ai_error(self):
        """Test vertex_ai_trainer_region raises error for non-Vertex AI trainer."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        trainer_config = gigl_resource_config_pb2.LocalResourceConfig(
            num_workers=2,
        )
        config.trainer_resource_config.local_trainer_config.CopyFrom(
            copy.deepcopy(trainer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)

        with self.assertRaises(ValueError) as context:
            _ = wrapper.vertex_ai_trainer_region

        self.assertIn("only supported for Vertex AI trainers", str(context.exception))

    def test_inferencer_config_dataflow(self):
        """Test inferencer_config with Dataflow configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        inferencer_config = gigl_resource_config_pb2.DataflowResourceConfig(
            num_workers=10,
            max_num_workers=20,
            machine_type="n1-standard-4",
        )
        config.inferencer_resource_config.dataflow_inferencer_config.CopyFrom(
            copy.deepcopy(inferencer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.inferencer_config, inferencer_config)

    def test_inferencer_config_vertex_ai(self):
        """Test inferencer_config with Vertex AI configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        inferencer_config = gigl_resource_config_pb2.VertexAiResourceConfig(
            machine_type="n1-standard-8",
            num_replicas=3,
        )
        config.inferencer_resource_config.vertex_ai_inferencer_config.CopyFrom(
            copy.deepcopy(inferencer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.inferencer_config, inferencer_config)

    def test_inferencer_config_local(self):
        """Test inferencer_config with Local configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        inferencer_config = gigl_resource_config_pb2.LocalResourceConfig(
            num_workers=4,
        )
        config.inferencer_resource_config.local_inferencer_config.CopyFrom(
            copy.deepcopy(inferencer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.inferencer_config, inferencer_config)

    def test_inferencer_config_vertex_ai_graph_store(self):
        """Test inferencer_config with Vertex AI Graph Store configuration."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        inferencer_config = gigl_resource_config_pb2.VertexAiGraphStoreConfig(
            compute_pool=gigl_resource_config_pb2.VertexAiResourceConfig(
                machine_type="n1-highmem-4",
            ),
            graph_store_pool=gigl_resource_config_pb2.VertexAiResourceConfig(
                machine_type="n1-standard-2",
            ),
        )
        config.inferencer_resource_config.vertex_ai_graph_store_inferencer_config.CopyFrom(
            copy.deepcopy(inferencer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.inferencer_config, inferencer_config)

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
        inferencer_config = gigl_resource_config_pb2.VertexAiResourceConfig(
            machine_type="n1-standard-4",
        )
        config.inferencer_resource_config.vertex_ai_inferencer_config.CopyFrom(
            copy.deepcopy(inferencer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.vertex_ai_inferencer_region, "us-central1")

    def test_vertex_ai_inferencer_region_override(self):
        """Test vertex_ai_inferencer_region with region override."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        inferencer_config = gigl_resource_config_pb2.VertexAiResourceConfig(
            machine_type="n1-standard-4",
            gcp_region_override="us-east1",
        )
        config.inferencer_resource_config.vertex_ai_inferencer_config.CopyFrom(
            copy.deepcopy(inferencer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.vertex_ai_inferencer_region, "us-east1")

    def test_vertex_ai_inferencer_region_non_vertex_ai_error(self):
        """Test vertex_ai_inferencer_region raises error for non-Vertex AI inferencer."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        inferencer_config = gigl_resource_config_pb2.LocalResourceConfig(
            num_workers=2,
        )
        config.inferencer_resource_config.local_inferencer_config.CopyFrom(
            copy.deepcopy(inferencer_config)
        )

        wrapper = GiglResourceConfigWrapper(resource_config=config)

        with self.assertRaises(ValueError) as context:
            _ = wrapper.vertex_ai_inferencer_region

        self.assertIn(
            "only supported for Vertex AI inferencers", str(context.exception)
        )

    def test_preprocessor_config(self):
        """Test preprocessor_config property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        preprocessor_config = gigl_resource_config_pb2.DataPreprocessorConfig(
            edge_preprocessor_config=gigl_resource_config_pb2.DataflowResourceConfig(
                num_workers=5,
                machine_type="n1-standard-2",
            ),
            node_preprocessor_config=gigl_resource_config_pb2.DataflowResourceConfig(
                num_workers=3,
                machine_type="n1-standard-1",
            ),
        )
        config.preprocessor_config.CopyFrom(copy.deepcopy(preprocessor_config))

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.preprocessor_config, preprocessor_config)

    def test_subgraph_sampler_config(self):
        """Test subgraph_sampler_config property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        sampler_config = gigl_resource_config_pb2.SparkResourceConfig(
            machine_type="n1-standard-16",
            num_replicas=10,
            num_local_ssds=4,
        )
        config.subgraph_sampler_config.CopyFrom(copy.deepcopy(sampler_config))

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.subgraph_sampler_config, sampler_config)

    def test_split_generator_config(self):
        """Test split_generator_config property."""
        config = self._create_gigl_resource_config_with_direct_shared_config()
        split_gen_config = gigl_resource_config_pb2.SparkResourceConfig(
            machine_type="n1-standard-8",
            num_replicas=5,
        )
        config.split_generator_config.CopyFrom(copy.deepcopy(split_gen_config))

        wrapper = GiglResourceConfigWrapper(resource_config=config)
        self.assertEqual(wrapper.split_generator_config, split_gen_config)


if __name__ == "__main__":
    absltest.main()
