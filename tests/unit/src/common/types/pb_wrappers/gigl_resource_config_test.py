from absl.testing import absltest

from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from snapchat.research.gbml.gigl_resource_config_pb2 import GiglResourceConfig
from tests.test_assets.test_case import TestCase


class GiglResourceConfigTest(TestCase):
    def test_vertex_ai_trainer_region_default(self):
        resource_config = GiglResourceConfig()
        resource_config.shared_resource_config.common_compute_config.region = (
            "us-central1"
        )
        resource_config.trainer_resource_config.vertex_ai_trainer_config.SetInParent()
        resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
        self.assertEqual(
            resource_config_wrapper.vertex_ai_trainer_region, "us-central1"
        )

    def test_vertex_ai_trainer_region_override(self):
        resource_config = GiglResourceConfig()
        resource_config.shared_resource_config.common_compute_config.region = (
            "us-central1"
        )
        resource_config.trainer_resource_config.vertex_ai_trainer_config.gcp_region_override = (
            "us-east1"
        )
        resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
        self.assertEqual(resource_config_wrapper.vertex_ai_trainer_region, "us-east1")

    def test_vertex_ai_inferencer_region_default(self):
        resource_config = GiglResourceConfig()
        resource_config.shared_resource_config.common_compute_config.region = (
            "us-central1"
        )
        resource_config.inferencer_resource_config.vertex_ai_inferencer_config.SetInParent()
        resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
        self.assertEqual(
            resource_config_wrapper.vertex_ai_inferencer_region, "us-central1"
        )

    def test_vertex_ai_inferencer_region_override(self):
        resource_config = GiglResourceConfig()
        resource_config.shared_resource_config.common_compute_config.region = (
            "us-central1"
        )
        resource_config.inferencer_resource_config.vertex_ai_inferencer_config.gcp_region_override = (
            "us-east1"
        )
        resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
        self.assertEqual(
            resource_config_wrapper.vertex_ai_inferencer_region, "us-east1"
        )

    def test_vertex_ai_trainer_region_error_no_trainer_config(self):
        resource_config = GiglResourceConfig()
        resource_config.shared_resource_config.common_compute_config.region = (
            "us-central1"
        )
        resource_config.trainer_resource_config.SetInParent()
        resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
        with self.assertRaises(ValueError):
            resource_config_wrapper.vertex_ai_trainer_region

    def test_vertex_ai_trainer_region_error_local_trainer_config(self):
        resource_config = GiglResourceConfig()
        resource_config.shared_resource_config.common_compute_config.region = (
            "us-central1"
        )
        resource_config.trainer_resource_config.local_trainer_config.SetInParent()
        resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
        with self.assertRaises(ValueError):
            resource_config_wrapper.vertex_ai_trainer_region

    def test_vertex_ai_trainer_region_error_kfp_trainer_config(self):
        resource_config = GiglResourceConfig()
        resource_config.shared_resource_config.common_compute_config.region = (
            "us-central1"
        )
        resource_config.trainer_resource_config.kfp_trainer_config.SetInParent()
        resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
        with self.assertRaises(ValueError):
            resource_config_wrapper.vertex_ai_trainer_region

    def test_vertex_ai_inferencer_region_error_no_inferencer_config(self):
        resource_config = GiglResourceConfig()
        resource_config.shared_resource_config.common_compute_config.region = (
            "us-central1"
        )
        resource_config.inferencer_resource_config.SetInParent()
        resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
        with self.assertRaises(ValueError):
            resource_config_wrapper.vertex_ai_inferencer_region

    def test_vertex_ai_inferencer_region_error_local_inferencer_config(self):
        resource_config = GiglResourceConfig()
        resource_config.shared_resource_config.common_compute_config.region = (
            "us-central1"
        )
        resource_config.inferencer_resource_config.local_inferencer_config.SetInParent()
        resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
        with self.assertRaises(ValueError):
            resource_config_wrapper.vertex_ai_inferencer_region

    def test_vertex_ai_inferencer_region_error_dataflow_inferencer_config(self):
        resource_config = GiglResourceConfig()
        resource_config.shared_resource_config.common_compute_config.region = (
            "us-central1"
        )
        resource_config.inferencer_resource_config.dataflow_inferencer_config.SetInParent()
        resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
        with self.assertRaises(ValueError):
            resource_config_wrapper.vertex_ai_inferencer_region


if __name__ == "__main__":
    absltest.main()
