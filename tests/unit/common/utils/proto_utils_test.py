import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch

from absl.testing import absltest
from hydra import initialize_config_dir
from hydra.errors import MissingConfigException
from omegaconf import OmegaConf

from gigl.common import LocalUri
from gigl.common.logger import Logger
from gigl.common.utils.hydra_config import compose_yaml_config
from gigl.common.utils.proto_utils import ProtoUtils
from snapchat.research.gbml import gbml_config_pb2
from snapchat.research.gbml.gigl_resource_config_pb2 import GiglResourceConfig
from tests.test_assets.test_case import TestCase

logger = Logger()

TEST_TASK_CONFIG = """
sharedConfig:
    isGraphDirected: true
datasetConfig:
    dataPreprocessorConfig:
        dataPreprocessorArgs:
            bq_edges_table_name: project.dataset.bq_edges_table_name_${now:%Y%m%d}
            positive_label_date_range: ${now:%Y%m%d,days-10}:${now:%Y%m%d,days-1}
"""


class ProtoUtilsTest(TestCase):
    def setUp(self):
        super().setUp()
        self.proto_utils = ProtoUtils()

        tmp_file = NamedTemporaryFile(delete=False)
        logger.info(f"Writing test config to {tmp_file.name}")
        tmp_file.write(TEST_TASK_CONFIG.encode())
        tmp_file.close()
        self.test_task_config_path = tmp_file.name

    def tearDown(self):
        super().tearDown()
        # delete the temporary file
        os.remove(self.test_task_config_path)

    def test_can_read_gbml_config_from_yaml(self):
        task_config = self.proto_utils.read_proto_from_yaml(
            uri=LocalUri(self.test_task_config_path),
            proto_cls=gbml_config_pb2.GbmlConfig,
        )

        self.assertEqual(task_config.shared_config.is_graph_directed, True)
        expected_bq_edges_table_name = (
            f"project.dataset.bq_edges_table_name_{datetime.now().strftime('%Y%m%d')}"
        )
        self.assertEqual(
            task_config.dataset_config.data_preprocessor_config.data_preprocessor_args[
                "bq_edges_table_name"
            ],
            expected_bq_edges_table_name,
        )
        expected_positive_label_date_range_start = (
            datetime.now() - timedelta(days=10)
        ).strftime("%Y%m%d")
        expected_positive_label_date_range_end = (
            datetime.now() - timedelta(days=1)
        ).strftime("%Y%m%d")
        self.assertEqual(
            task_config.dataset_config.data_preprocessor_config.data_preprocessor_args[
                "positive_label_date_range"
            ],
            f"{expected_positive_label_date_range_start}:{expected_positive_label_date_range_end}",
        )

    def test_read_proto_from_yaml_raises_typeerror_when_root_is_not_a_mapping(self):
        list_yaml = "- a\n- b\n- c\n"
        tmp_file = NamedTemporaryFile(delete=False)
        tmp_file.write(list_yaml.encode())
        tmp_file.close()
        try:
            with self.assertRaises(TypeError):
                self.proto_utils.read_proto_from_yaml(
                    uri=LocalUri(tmp_file.name),
                    proto_cls=gbml_config_pb2.GbmlConfig,
                )
        finally:
            os.remove(tmp_file.name)

    def test_can_compose_task_config_from_primary_parent(self):
        with TemporaryDirectory() as temp_directory:
            config_root = Path(temp_directory)
            (config_root / "shared").mkdir()
            (config_root / "task.yaml").write_text(
                "defaults:\n"
                "  - shared@sharedConfig: directed\n"
                "  - _self_\n"
                "datasetConfig:\n"
                "  dataPreprocessorConfig:\n"
                "    dataPreprocessorArgs:\n"
                '      yesterday: "${now:%Y%m%d,days-1}"\n'
            )
            (config_root / "shared" / "directed.yaml").write_text(
                "isGraphDirected: true\n"
            )

            task_config = self.proto_utils.compose_proto_from_yaml(
                uri=LocalUri(config_root / "task.yaml"),
                proto_cls=gbml_config_pb2.GbmlConfig,
            )

            self.assertTrue(task_config.shared_config.is_graph_directed)
            self.assertEqual(
                task_config.dataset_config.data_preprocessor_config.data_preprocessor_args[
                    "yesterday"
                ],
                (datetime.now() - timedelta(days=1)).strftime("%Y%m%d"),
            )

    def test_can_compose_resource_config_from_primary_parent(self):
        with TemporaryDirectory() as temp_directory:
            config_root = Path(temp_directory)
            (config_root / "shared").mkdir()
            (config_root / "resource.yaml").write_text(
                "defaults:\n  - shared@shared_resource_config: local\n  - _self_\n"
            )
            (config_root / "shared" / "local.yaml").write_text(
                "common_compute_config:\n"
                "  project: example-project\n"
                "  region: us-central1\n"
                "  temp_regional_assets_bucket: gs://example-bucket\n"
            )

            resource_config = self.proto_utils.compose_proto_from_yaml(
                uri=LocalUri(config_root / "resource.yaml"),
                proto_cls=GiglResourceConfig,
            )

            self.assertEqual(
                resource_config.shared_resource_config.common_compute_config.project,
                "example-project",
            )

    def test_resource_config_resolves_dynamic_interpolation(self):
        with TemporaryDirectory() as temp_directory:
            config_path = Path(temp_directory) / "resource.yaml"
            config_path.write_text(
                "shared_resource_config:\n"
                "  common_compute_config:\n"
                '    project: "${oc.env:PROJECT_ID}"\n'
            )

            with patch.dict(os.environ, {"PROJECT_ID": "example-project"}):
                resource_config = self.proto_utils.read_proto_from_yaml(
                    uri=LocalUri(config_path),
                    proto_cls=GiglResourceConfig,
                )

            self.assertEqual(
                resource_config.shared_resource_config.common_compute_config.project,
                "example-project",
            )

    def test_resource_config_allows_deterministic_value_interpolation(self):
        with TemporaryDirectory() as temp_directory:
            config_path = Path(temp_directory) / "resource.yaml"
            config_path.write_text(
                "shared_resource_config:\n"
                "  common_compute_config:\n"
                "    project: example-project\n"
                '    region: "${.project}"\n'
            )

            resource_config = self.proto_utils.read_proto_from_yaml(
                uri=LocalUri(config_path),
                proto_cls=GiglResourceConfig,
            )

            self.assertEqual(
                resource_config.shared_resource_config.common_compute_config.region,
                "example-project",
            )

    def test_resource_composition_ignores_dynamic_resolver_in_unselected_file(self):
        with TemporaryDirectory() as temp_directory:
            config_root = Path(temp_directory)
            (config_root / "shared").mkdir()
            config_path = config_root / "resource.yaml"
            config_path.write_text(
                "defaults:\n  - shared@shared_resource_config: local\n  - _self_\n"
            )
            (config_root / "shared" / "local.yaml").write_text(
                "common_compute_config:\n"
                "  project: example-project\n"
                "  region: us-central1\n"
            )
            (config_root / "unselected_task.yaml").write_text(
                'run_name: "${now:%Y%m%d}"\n'
            )

            resource_config = self.proto_utils.compose_proto_from_yaml(
                uri=LocalUri(config_path),
                proto_cls=GiglResourceConfig,
            )

            self.assertEqual(
                resource_config.shared_resource_config.common_compute_config.project,
                "example-project",
            )

    def test_resource_composition_resolves_dynamic_resolver_in_selected_file(self):
        with TemporaryDirectory() as temp_directory:
            config_root = Path(temp_directory)
            (config_root / "shared").mkdir()
            config_path = config_root / "resource.yaml"
            config_path.write_text(
                "defaults:\n  - shared@shared_resource_config: dynamic\n  - _self_\n"
            )
            (config_root / "shared" / "dynamic.yaml").write_text(
                'common_compute_config:\n  project: "${oc.env:PROJECT_ID}"\n'
            )

            with patch.dict(os.environ, {"PROJECT_ID": "example-project"}):
                resource_config = self.proto_utils.compose_proto_from_yaml(
                    uri=LocalUri(config_path),
                    proto_cls=GiglResourceConfig,
                )

            self.assertEqual(
                resource_config.shared_resource_config.common_compute_config.project,
                "example-project",
            )

    def test_resource_composition_resolves_dynamic_nested_default(self):
        with TemporaryDirectory() as temp_directory:
            config_root = Path(temp_directory)
            (config_root / "base").mkdir()
            (config_root / "compute").mkdir()
            config_path = config_root / "resource.yaml"
            config_path.write_text(
                "defaults:\n  - base@_global_: resource\n  - _self_\n"
            )
            (config_root / "base" / "resource.yaml").write_text(
                "defaults:\n"
                "  - /compute@shared_resource_config: "
                "${oc.env:RESOURCE_PROFILE}\n"
                "  - _self_\n"
            )
            (config_root / "compute" / "local.yaml").write_text(
                "common_compute_config:\n"
                "  project: example-project\n"
                "  region: us-central1\n"
            )

            with patch.dict(os.environ, {"RESOURCE_PROFILE": "local"}):
                resource_config = self.proto_utils.compose_proto_from_yaml(
                    uri=LocalUri(config_path),
                    proto_cls=GiglResourceConfig,
                )

            self.assertEqual(
                resource_config.shared_resource_config.common_compute_config.project,
                "example-project",
            )

    def test_composition_is_thread_safe(self):
        with TemporaryDirectory() as temp_directory:
            config_root = Path(temp_directory)
            (config_root / "shared").mkdir()
            config_path = config_root / "task.yaml"
            config_path.write_text(
                "defaults:\n  - shared@sharedConfig: directed\n  - _self_\n"
            )
            (config_root / "shared" / "directed.yaml").write_text(
                "isGraphDirected: true\n"
            )

            def read_config(_: int) -> bool:
                return (
                    ProtoUtils()
                    .compose_proto_from_yaml(
                        uri=LocalUri(config_path),
                        proto_cls=gbml_config_pb2.GbmlConfig,
                    )
                    .shared_config.is_graph_directed
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(read_config, range(32)))

            self.assertEqual(results, [True] * 32)

    def test_composition_preserves_foreign_hydra_context(self):
        with TemporaryDirectory() as temp_directory:
            config_root = Path(temp_directory)
            config_path = config_root / "task.yaml"
            config_path.write_text("defaults:\n  - _self_\n")

            with initialize_config_dir(
                config_dir=str(config_root),
                version_base="1.3",
            ):
                with self.assertRaises(RuntimeError):
                    self.proto_utils.compose_proto_from_yaml(
                        uri=LocalUri(config_path),
                        proto_cls=gbml_config_pb2.GbmlConfig,
                    )

    def test_composition_restores_gigl_resolvers_after_failure(self):
        with TemporaryDirectory() as temp_directory:
            config_path = Path(temp_directory) / "task.yaml"
            config_path.write_text("defaults:\n  - missing\n")

            with self.assertRaises(MissingConfigException):
                compose_yaml_config(LocalUri(config_path))

            config = OmegaConf.create({"tomorrow": "${now:%Y-%m-%d,days+1}"})
            self.assertRegex(config.tomorrow, r"\d{4}-\d{2}-\d{2}")


if __name__ == "__main__":
    absltest.main()
