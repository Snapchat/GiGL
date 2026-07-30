import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch

from absl.testing import absltest
from hydra import initialize_config_dir
from omegaconf import OmegaConf

from gigl.common import GcsUri, LocalUri
from gigl.common.logger import Logger
from gigl.common.omegaconf_resolvers import now_resolver
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

            task_config = self.proto_utils.read_proto_from_yaml(
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

            resource_config = self.proto_utils.read_proto_from_yaml(
                uri=LocalUri(config_root / "resource.yaml"),
                proto_cls=GiglResourceConfig,
            )

            self.assertEqual(
                resource_config.shared_resource_config.common_compute_config.project,
                "example-project",
            )

    def test_resource_config_rejects_interpolation(self):
        with TemporaryDirectory() as temp_directory:
            config_path = Path(temp_directory) / "resource.yaml"
            config_path.write_text(
                "shared_resource_config:\n"
                "  common_compute_config:\n"
                '    project: "${oc.env:PROJECT_ID}"\n'
            )

            with self.assertRaises(ValueError):
                self.proto_utils.read_proto_from_yaml(
                    uri=LocalUri(config_path),
                    proto_cls=GiglResourceConfig,
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

            resource_config = self.proto_utils.read_proto_from_yaml(
                uri=LocalUri(config_path),
                proto_cls=GiglResourceConfig,
            )

            self.assertEqual(
                resource_config.shared_resource_config.common_compute_config.project,
                "example-project",
            )

    def test_resource_composition_rejects_dynamic_resolver_in_selected_file(self):
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

            with self.assertRaises(ValueError):
                self.proto_utils.read_proto_from_yaml(
                    uri=LocalUri(config_path),
                    proto_cls=GiglResourceConfig,
                )

    def test_resource_composition_rejects_dynamic_nested_default(self):
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
                "  - compute@shared_resource_config: "
                "${oc.env:RESOURCE_PROFILE}\n"
                "  - _self_\n"
            )
            (config_root / "compute" / "local.yaml").write_text(
                "common_compute_config:\n"
                "  project: example-project\n"
                "  region: us-central1\n"
            )

            with (
                patch.dict(os.environ, {"RESOURCE_PROFILE": "local"}),
                self.assertRaises(ValueError),
            ):
                self.proto_utils.read_proto_from_yaml(
                    uri=LocalUri(config_path),
                    proto_cls=GiglResourceConfig,
                )

    def test_composed_primary_requires_yaml_extension(self):
        with TemporaryDirectory() as temp_directory:
            config_path = Path(temp_directory) / "task.yml"
            config_path.write_text("defaults:\n  - _self_\n")

            with self.assertRaises(ValueError):
                self.proto_utils.read_proto_from_yaml(
                    uri=LocalUri(config_path),
                    proto_cls=gbml_config_pb2.GbmlConfig,
                )

    def test_gcs_composition_is_not_supported(self):
        with self.assertRaisesRegex(
            TypeError,
            "Hydra composition is not supported for GcsUri",
        ):
            compose_yaml_config(GcsUri("gs://example-bucket/configs/task.yaml"))

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
                    .read_proto_from_yaml(
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
                    self.proto_utils.read_proto_from_yaml(
                        uri=LocalUri(config_path),
                        proto_cls=gbml_config_pb2.GbmlConfig,
                    )

    def test_composition_restores_foreign_omegaconf_resolver(self):
        with TemporaryDirectory() as temp_directory:
            config_path = Path(temp_directory) / "task.yaml"
            config_path.write_text("defaults:\n  - _self_\n")
            OmegaConf.register_new_resolver(
                "now", lambda *_: "foreign-now", replace=True
            )
            try:
                self.proto_utils.read_proto_from_yaml(
                    uri=LocalUri(config_path),
                    proto_cls=gbml_config_pb2.GbmlConfig,
                )

                config = OmegaConf.create({"value": "${now:any-format}"})
                self.assertEqual(config.value, "foreign-now")
            finally:
                OmegaConf.register_new_resolver("now", now_resolver, replace=True)

    def test_composition_rejects_parent_traversal(self):
        for selector in (
            "../outside@_global_",
            "optional ../outside@_global_",
            "override optional ../outside@_global_",
        ):
            with self.subTest(selector=selector):
                with TemporaryDirectory() as temp_directory:
                    config_root = Path(temp_directory) / "bundle"
                    config_root.mkdir()
                    config_path = config_root / "task.yaml"
                    config_path.write_text(f"defaults:\n  - {selector}\n")
                    (Path(temp_directory) / "outside.yaml").write_text(
                        "sharedConfig:\n  isGraphDirected: true\n"
                    )

                    with self.assertRaises(ValueError):
                        self.proto_utils.read_proto_from_yaml(
                            uri=LocalUri(config_path),
                            proto_cls=gbml_config_pb2.GbmlConfig,
                        )

    def test_composition_rejects_interpolated_default_selector(self):
        with TemporaryDirectory() as temp_directory:
            config_path = Path(temp_directory) / "task.yaml"
            config_path.write_text(
                "defaults:\n  - group@sharedConfig: ${oc.env:TASK_PROFILE}\n"
            )

            with (
                patch.dict(os.environ, {"TASK_PROFILE": "../outside"}),
                self.assertRaises(ValueError),
            ):
                self.proto_utils.read_proto_from_yaml(
                    uri=LocalUri(config_path),
                    proto_cls=gbml_config_pb2.GbmlConfig,
                )

    def test_composition_rejects_symlinked_config_group(self):
        with TemporaryDirectory() as temp_directory:
            temp_root = Path(temp_directory)
            config_root = temp_root / "bundle"
            outside_group = temp_root / "outside"
            config_root.mkdir()
            outside_group.mkdir()
            config_path = config_root / "task.yaml"
            config_path.write_text(
                "defaults:\n  - escaped@sharedConfig: directed\n  - _self_\n"
            )
            (outside_group / "directed.yaml").write_text("isGraphDirected: true\n")
            (config_root / "escaped").symlink_to(
                outside_group, target_is_directory=True
            )

            with self.assertRaises(ValueError):
                self.proto_utils.read_proto_from_yaml(
                    uri=LocalUri(config_path),
                    proto_cls=gbml_config_pb2.GbmlConfig,
                )


if __name__ == "__main__":
    absltest.main()
