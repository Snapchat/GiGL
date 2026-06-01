import os
import tempfile
from unittest.mock import patch

from gigl.common import LocalUri, Uri
from gigl.common.constants import (
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
)
from gigl.common.metrics.base_metrics import NopMetricsPublisher
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.env.constants import (
    GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY,
    GIGL_COMPONENT_ENV_KEY,
    GIGL_CPU_DOCKER_URI_ENV_KEY,
    GIGL_CUDA_DOCKER_URI_ENV_KEY,
    GIGL_RESOURCE_CONFIG_URI_ENV_KEY,
    GIGL_TASK_CONFIG_URI_ENV_KEY,
)
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.utils import metrics_service_provider
from gigl.src.common.utils.gigl_env import get_gigl_runtime_env_vars
from gigl.src.common.utils.gigl_runtime import initialize_gigl_runtime
from gigl.src.common.utils.metrics_service_provider import (
    JOB_NAME_GROUPING_ENV_KEY,
    get_metrics_service_instance,
)
from snapchat.research.gbml import gbml_config_pb2
from tests.test_assets.test_case import TestCase


class GiGLRuntimeTest(TestCase):
    def setUp(self) -> None:
        self.proto_utils = ProtoUtils()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self._original_metrics_instance = metrics_service_provider._metrics_instance

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()
        metrics_service_provider._metrics_instance = self._original_metrics_instance

    def _write_task_config(self, config: gbml_config_pb2.GbmlConfig) -> LocalUri:
        uri = LocalUri.join(self.tmp_dir.name, "task_config.yaml")
        self.proto_utils.write_proto_to_yaml(proto=config, uri=uri)
        return uri

    def test_get_gigl_runtime_env_vars_sets_expected_values(self) -> None:
        env_vars = get_gigl_runtime_env_vars(
            applied_task_identifier="job-42",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            component=GiGLComponents.DataPreprocessor,
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
        )

        self.assertEqual(env_vars[GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY], "job-42")
        self.assertEqual(
            env_vars[GIGL_TASK_CONFIG_URI_ENV_KEY], "gs://bucket/task.yaml"
        )
        self.assertEqual(
            env_vars[GIGL_RESOURCE_CONFIG_URI_ENV_KEY],
            "gs://bucket/resource.yaml",
        )
        self.assertEqual(env_vars[GIGL_COMPONENT_ENV_KEY], "DataPreprocessor")
        self.assertEqual(env_vars[GIGL_CPU_DOCKER_URI_ENV_KEY], "gcr.io/p/cpu:tag")
        self.assertEqual(env_vars[GIGL_CUDA_DOCKER_URI_ENV_KEY], "gcr.io/p/cuda:tag")

    def test_get_gigl_runtime_env_vars_defaults_empty_image_uris(self) -> None:
        env_vars = get_gigl_runtime_env_vars(
            applied_task_identifier="job-42",
            task_config_uri=Uri("gs://bucket/task.yaml"),
            resource_config_uri=Uri("gs://bucket/resource.yaml"),
            component=GiGLComponents.DataPreprocessor,
            cpu_docker_uri="",
            cuda_docker_uri=None,
        )

        self.assertEqual(
            env_vars[GIGL_CPU_DOCKER_URI_ENV_KEY],
            DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
        )
        self.assertEqual(
            env_vars[GIGL_CUDA_DOCKER_URI_ENV_KEY],
            DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA,
        )

    def test_initialize_gigl_runtime_sets_env_and_initializes_metrics(self) -> None:
        task_config_uri = self._write_task_config(gbml_config_pb2.GbmlConfig())

        with patch.dict(os.environ, {}, clear=False):
            initialize_gigl_runtime(
                applied_task_identifier="job-42",
                task_config_uri=task_config_uri,
                resource_config_uri=Uri("gs://bucket/resource.yaml"),
                service_name="data-preprocessor-service",
                component=GiGLComponents.DataPreprocessor,
                cpu_docker_uri="gcr.io/p/cpu:tag",
                cuda_docker_uri="gcr.io/p/cuda:tag",
            )

            self.assertEqual(
                os.environ[GIGL_APPLIED_TASK_IDENTIFIER_ENV_KEY],
                "job-42",
            )
            self.assertEqual(
                os.environ[GIGL_TASK_CONFIG_URI_ENV_KEY],
                task_config_uri.uri,
            )
            self.assertEqual(
                os.environ[GIGL_RESOURCE_CONFIG_URI_ENV_KEY],
                "gs://bucket/resource.yaml",
            )
            self.assertEqual(
                os.environ[GIGL_COMPONENT_ENV_KEY],
                "DataPreprocessor",
            )
            self.assertEqual(
                os.environ[GIGL_CPU_DOCKER_URI_ENV_KEY],
                "gcr.io/p/cpu:tag",
            )
            self.assertEqual(
                os.environ[GIGL_CUDA_DOCKER_URI_ENV_KEY],
                "gcr.io/p/cuda:tag",
            )
            self.assertEqual(
                os.environ[JOB_NAME_GROUPING_ENV_KEY],
                "data-preprocessor-service",
            )
            self.assertIsInstance(get_metrics_service_instance(), NopMetricsPublisher)
