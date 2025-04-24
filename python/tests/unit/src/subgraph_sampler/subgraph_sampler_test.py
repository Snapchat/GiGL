import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import yaml
from google.protobuf.json_format import MessageToDict

import gigl.env.dep_constants as dep_constants
import gigl.src.common.constants.gcs as gcs_constants
from gigl.common import GcsUri, LocalUri, UriFactory
from gigl.common.constants import (
    SPARK_31_TFRECORD_JAR_GCS_PATH,
    SPARK_35_TFRECORD_JAR_GCS_PATH,
)
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils import metrics_service_provider
from gigl.src.subgraph_sampler import subgraph_sampler
from gigl.src.subgraph_sampler.lib.ingestion_protocol import BaseIngestion
from snapchat.research.gbml import gbml_config_pb2, gigl_resource_config_pb2


# Class that's used as a dummy to be injected and then mocked out
# For testing the ingestion part of the subgraph sampler.
class _Ingestor(BaseIngestion):
    pass


# Fully quyalified name of the class to be mocked.
_INGESTOR_FQN = f"{_Ingestor.__module__}.{_Ingestor.__name__}"


class SubgraphSamplerTest(unittest.TestCase):
    test_dir: Path
    _tmp_dir: tempfile.TemporaryDirectory
    main_jar_local_path: Path
    sidecar_jar_local_path: Path
    mock_file_loader: MagicMock
    mock_gcs_utils: MagicMock
    mock_dataproc_helper: MagicMock
    config_file_dir: Path
    resource_config: gigl_resource_config_pb2.GiglResourceConfig
    resource_config_path_local_path: Path
    gbml_config: gbml_config_pb2.GbmlConfig
    gbml_config_path_local_path: Path
    task_identifier = AppliedTaskIdentifier("test_task")

    def setUp(self):
        super().setUp()

        # Setup temp dir.
        # Structured as:
        # temp_dir
        #     ├─jars
        #     │  ├─subgraph_sampler.jar
        #     │  └─not_a_jar.txt
        #     ├─sidecar
        #     │  ├─sidecar.jar
        #     └─configs
        #         ├─resource_config.yaml    # gigl_resource_config_pb2.GiglResourceConfig
        #         └─gbml_config.yaml        # gbml_config_pb2.GbmlConfig
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self._tmp_dir.name)

        jar_file_dir = self.test_dir / "jars"
        jar_file_dir.mkdir()
        self.main_jar_local_path = jar_file_dir / "subgraph_sampler.jar"
        self.main_jar_local_path.touch()
        (jar_file_dir / "not_a_jar.txt").touch()
        sidecar_dir = self.test_dir / "sidecar"
        sidecar_dir.mkdir()
        self.sidecar_jar_local_path = sidecar_dir / "sidecar.jar"
        self.sidecar_jar_local_path.touch()

        self.mock_file_loader = MagicMock()
        self.mock_file_loader.load_file = MagicMock()
        self.mock_file_loader.delete_files = MagicMock()
        self.mock_file_loader.load_files = MagicMock()

        self.mock_gcs_utils = MagicMock()
        self.mock_gcs_utils.upload_files_to_gcs = MagicMock()

        self.mock_dataproc_helper = MagicMock()
        self.mock_dataproc_helper.create_dataproc_cluster = MagicMock()
        self.mock_dataproc_helper.submite_and_wait_scala_spark_job = MagicMock()

        self.config_file_dir = self.test_dir / "configs"
        self.config_file_dir.mkdir()

        self.resource_config = gigl_resource_config_pb2.GiglResourceConfig()
        self.resource_config.shared_resource_config.common_compute_config.project = (
            "test_project"
        )
        self.resource_config.shared_resource_config.common_compute_config.region = (
            "test_region"
        )
        self.resource_config.shared_resource_config.common_compute_config.gcp_service_account_email = (
            "test@mail.com"
        )
        self.resource_config.shared_resource_config.common_compute_config.temp_assets_bucket = (
            "gs://test_bucket/foo"
        )
        self.resource_config.shared_resource_config.common_compute_config.temp_regional_assets_bucket = (
            "gs://test_bucket/bar"
        )
        self.resource_config.subgraph_sampler_config.num_replicas = 1
        self.resource_config.subgraph_sampler_config.num_local_ssds = 2

        self.resource_config_path_local_path = (
            self.config_file_dir / "resource_config.yaml"
        )
        with open(self.resource_config_path_local_path, "w") as f:
            f.write(yaml.dump(MessageToDict(self.resource_config)))

        self.gbml_config = gbml_config_pb2.GbmlConfig()
        self.gbml_config.shared_config.flattened_graph_metadata.node_anchor_based_link_prediction_output.tfrecord_uri_prefix = (
            "gs://test_tfrecord"
        )
        self.gbml_config.shared_config.flattened_graph_metadata.node_anchor_based_link_prediction_output.node_type_to_random_negative_tfrecord_uri_prefix[
            "foo_node"
        ] = "gs://test_tfrecord/foo_node"
        self.gbml_config_path_local_path = self.config_file_dir / "gbml_config.yaml"
        with open(self.gbml_config_path_local_path, "w") as f:
            f.write(yaml.dump(MessageToDict(self.gbml_config)))

    def tearDown(self):
        super().tearDown()
        self._tmp_dir.cleanup()

    @patch("gigl.src.subgraph_sampler.subgraph_sampler.FileLoader", autospec=True)
    @patch("gigl.src.subgraph_sampler.subgraph_sampler.GcsUtils", autospec=True)
    @patch.object(dep_constants, "get_jar_file_uri", autospec=True)
    @patch("gigl.src.subgraph_sampler.subgraph_sampler.SparkJobManager", autospec=True)
    def test_subgraph_sampler_for_spark(
        self,
        mock_spark_job_manager,
        mock_get_subgraph_sampler_root_dir,
        mock_gcs_utils_factory,
        mock_file_loader_factory,
    ):
        mock_get_subgraph_sampler_root_dir.return_value = LocalUri(
            self.main_jar_local_path
        )

        mock_file_loader_factory.return_value = self.mock_file_loader
        mock_gcs_utils_factory.return_value = self.mock_gcs_utils
        mock_spark_job_manager.return_value = self.mock_dataproc_helper

        # If metrics aren't initiated then get_metrics_service_instance will throw
        metrics_service_provider.initialize_metrics(
            task_config_uri=LocalUri(self.gbml_config_path_local_path),
            service_name="test_service",
        )
        sampler = subgraph_sampler.SubgraphSampler()
        sampler.run(
            applied_task_identifier=self.task_identifier,
            resource_config_uri=LocalUri(self.resource_config_path_local_path),
            task_config_uri=LocalUri(self.gbml_config_path_local_path),
            additional_spark35_jar_file_uris=[
                LocalUri("/does/not/exist/should/not/be/passed/in")
            ],
        )
        subgraph_sampler_root = gcs_constants.get_subgraph_sampler_root_dir(
            applied_task_identifier=self.task_identifier
        )
        applied_task_root = gcs_constants.get_applied_task_temp_gcs_path(
            applied_task_identifier=self.task_identifier
        )
        with self.subTest("ensure SGS GCS bucket is setup"):
            self.mock_file_loader.load_file.assert_has_calls(
                [
                    call(
                        file_uri_src=LocalUri(self.gbml_config_path_local_path),
                        file_uri_dst=GcsUri.join(
                            applied_task_root,
                            "task_config.yaml",
                        ),
                    ),
                    call(
                        file_uri_src=LocalUri(self.resource_config_path_local_path),
                        file_uri_dst=GcsUri.join(
                            applied_task_root,
                            "resource_config.yaml",
                        ),
                    ),
                ],
                any_order=True,
            )
            self.mock_file_loader.delete_files.assert_called_once_with(
                uris=[
                    subgraph_sampler_root,
                    GcsUri("gs://test_tfrecord/foo_node"),
                    GcsUri("gs://test_tfrecord"),
                ]
            )

        with self.subTest("ensure main jar and sidecar jars are uploaded"):
            self.mock_file_loader.load_files.assert_called_once_with(
                source_to_dest_file_uri_map={
                    LocalUri(self.main_jar_local_path): GcsUri.join(
                        subgraph_sampler_root,
                        "subgraph_sampler.jar",
                    ),
                },
            )

        with self.subTest("correct jar file uris are passed to dataproc"):
            self.mock_dataproc_helper.submit_and_wait_scala_spark_job.assert_called_once_with(
                main_jar_file_uri=GcsUri.join(
                    subgraph_sampler_root,
                    "subgraph_sampler.jar",
                ).uri,
                max_job_duration=ANY,
                runtime_args=ANY,
                extra_jar_file_uris=[
                    SPARK_31_TFRECORD_JAR_GCS_PATH,
                ],
                use_spark35=False,
            )

    @patch(_INGESTOR_FQN)
    @patch("gigl.src.subgraph_sampler.subgraph_sampler.FileLoader", autospec=True)
    @patch("gigl.src.subgraph_sampler.subgraph_sampler.GcsUtils", autospec=True)
    @patch.object(dep_constants, "get_jar_file_uri", autospec=True)
    @patch("gigl.src.subgraph_sampler.subgraph_sampler.SparkJobManager", autospec=True)
    def test_subgraph_sampler_for_spark35_and_graphdb_ingestion(
        self,
        mock_spark_job_manager,
        mock_get_subgraph_sampler_root_dir,
        mock_gcs_utils_factory,
        mock_file_loader_factory,
        mock_ingestor_factory,
    ):
        self.gbml_config.dataset_config.subgraph_sampler_config.graph_db_config.graph_db_args[
            "foo_graph_arg"
        ] = "foo"
        self.gbml_config.dataset_config.subgraph_sampler_config.graph_db_config.graph_db_ingestion_args[
            "bar_graph_arg"
        ] = "bar"
        self.gbml_config.dataset_config.subgraph_sampler_config.graph_db_config.graph_db_ingestion_cls_path = (
            _INGESTOR_FQN
        )
        # Overwrite with the new config
        with open(self.gbml_config_path_local_path, "w") as f:
            f.write(yaml.dump(MessageToDict(self.gbml_config)))
        mock_get_subgraph_sampler_root_dir.return_value = UriFactory.create_uri(
            str(self.main_jar_local_path)
        )

        mock_file_loader_factory.return_value = self.mock_file_loader
        mock_gcs_utils_factory.return_value = self.mock_gcs_utils
        mock_spark_job_manager.return_value = self.mock_dataproc_helper

        mock_ingestor = MagicMock()
        mock_ingestor_factory.return_value = mock_ingestor

        # If metrics aren't initiated then get_metrics_service_instance will throw
        metrics_service_provider.initialize_metrics(
            task_config_uri=LocalUri(self.gbml_config_path_local_path),
            service_name="test_service",
        )
        sampler = subgraph_sampler.SubgraphSampler()
        sampler.run(
            applied_task_identifier=self.task_identifier,
            custom_worker_image_uri="gcr.io/test_project/test_image:latest",
            resource_config_uri=LocalUri(self.resource_config_path_local_path),
            task_config_uri=LocalUri(self.gbml_config_path_local_path),
            additional_spark35_jar_file_uris=[LocalUri(self.sidecar_jar_local_path)],
        )
        subgraph_sampler_root = gcs_constants.get_subgraph_sampler_root_dir(
            applied_task_identifier=self.task_identifier
        )
        applied_task_root = gcs_constants.get_applied_task_temp_gcs_path(
            applied_task_identifier=self.task_identifier
        )
        with self.subTest("ensure SGS GCS bucket is setup"):
            self.mock_file_loader.load_file.assert_has_calls(
                [
                    call(
                        file_uri_src=LocalUri(self.gbml_config_path_local_path),
                        file_uri_dst=GcsUri.join(
                            applied_task_root,
                            "task_config.yaml",
                        ),
                    ),
                    call(
                        file_uri_src=LocalUri(self.resource_config_path_local_path),
                        file_uri_dst=GcsUri.join(
                            applied_task_root,
                            "resource_config.yaml",
                        ),
                    ),
                ],
                any_order=True,
            )
            self.mock_file_loader.delete_files.assert_called_once_with(
                uris=[
                    subgraph_sampler_root,
                    GcsUri("gs://test_tfrecord/foo_node"),
                    GcsUri("gs://test_tfrecord"),
                ]
            )
        with self.subTest("ensure main jar and sidecar jars are uploaded"):
            self.mock_file_loader.load_files.assert_called_once_with(
                source_to_dest_file_uri_map={
                    LocalUri(self.main_jar_local_path): GcsUri.join(
                        subgraph_sampler_root,
                        "subgraph_sampler.jar",
                    ),
                    LocalUri(self.sidecar_jar_local_path): GcsUri.join(
                        subgraph_sampler_root,
                        "sidecar.jar",
                    ),
                },
            )

        with self.subTest("correct jar file uris are passed to dataproc"):
            self.mock_dataproc_helper.submit_and_wait_scala_spark_job.assert_called_once_with(
                main_jar_file_uri=GcsUri.join(
                    subgraph_sampler_root,
                    "subgraph_sampler.jar",
                ).uri,
                max_job_duration=ANY,
                runtime_args=ANY,
                extra_jar_file_uris=[
                    GcsUri.join(
                        subgraph_sampler_root,
                        "sidecar.jar",
                    ).uri,
                    SPARK_35_TFRECORD_JAR_GCS_PATH,
                ],
                use_spark35=True,
            )
        with self.subTest("ingestor"):
            mock_ingestor_factory.assert_called_once_with(
                bar_graph_arg="bar", foo_graph_arg="foo"
            )
            mock_ingestor.ingest.assert_called_once()
            # We do all this hoopla for checking the call args because the gbml wrapper is not fully constructed
            # (e.g. no graph_metadata) and thus can't be printed for errors
            # (AttributeError: 'GbmlConfigPbWrapper' object has no attribute '_graph_metadata_pb_wrapper')
            ingest_call = mock_ingestor.ingest.call_args
            self.assertEqual(len(ingest_call.args), 0)
            self.assertEqual(len(ingest_call.kwargs), 4)
            self.assertEqual(
                ingest_call.kwargs["resource_config_uri"],
                UriFactory.create_uri(str(self.resource_config_path_local_path)),
            )
            self.assertEqual(
                ingest_call.kwargs["applied_task_identifier"],
                AppliedTaskIdentifier("test_task"),
            )
            self.assertEqual(
                ingest_call.kwargs["custom_worker_image_uri"],
                "gcr.io/test_project/test_image:latest",
            )
            wrapper = ingest_call.kwargs["gbml_config_pb_wrapper"]
            self.assertEqual(wrapper.gbml_config_pb, self.gbml_config)
            mock_ingestor.clean_up.assert_called_once()
