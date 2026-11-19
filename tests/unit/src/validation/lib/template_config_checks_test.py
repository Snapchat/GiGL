import tempfile
from typing import Optional

from absl.testing import absltest
from parameterized import param, parameterized

from gigl.common import LocalUri
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.validation_check.libs.template_config_checks import (
    check_data_preprocessor_output_format_supported_by_backend,
    check_if_preprocessed_metadata_valid,
)
from snapchat.research.gbml import (
    gbml_config_pb2,
    graph_schema_pb2,
    preprocessed_metadata_pb2,
)
from tests.test_assets.test_case import TestCase


def _create_gbml_config(
    should_run_glt_backend: bool, output_format: Optional[str]
) -> GbmlConfigPbWrapper:
    feature_flags: dict[str, str] = {
        "should_run_glt_backend": str(should_run_glt_backend)
    }
    if output_format is not None:
        feature_flags["data_preprocessor_output_format"] = output_format
    return GbmlConfigPbWrapper(
        gbml_config_pb=gbml_config_pb2.GbmlConfig(feature_flags=feature_flags)
    )


class DataPreprocessorOutputFormatCheckTest(TestCase):
    @parameterized.expand(
        [
            param("glt_default", should_run_glt_backend=True, output_format=None),
            param("glt_parquet", should_run_glt_backend=True, output_format="parquet"),
            param(
                "glt_tfrecord", should_run_glt_backend=True, output_format="tfrecord"
            ),
            param(
                "non_glt_tfrecord",
                should_run_glt_backend=False,
                output_format="tfrecord",
            ),
        ]
    )
    def test_accepts_supported_formats(
        self, _, should_run_glt_backend: bool, output_format: Optional[str]
    ) -> None:
        check_data_preprocessor_output_format_supported_by_backend(
            gbml_config_wrapper=_create_gbml_config(
                should_run_glt_backend=should_run_glt_backend,
                output_format=output_format,
            )
        )

    @parameterized.expand(
        [
            param("non_glt_default", should_run_glt_backend=False, output_format=None),
            param(
                "non_glt_parquet",
                should_run_glt_backend=False,
                output_format="parquet",
            ),
            param("glt_unknown", should_run_glt_backend=True, output_format="bogus"),
        ]
    )
    def test_rejects_unsupported_formats(
        self, _, should_run_glt_backend: bool, output_format: Optional[str]
    ) -> None:
        with self.assertRaises(ValueError):
            check_data_preprocessor_output_format_supported_by_backend(
                gbml_config_wrapper=_create_gbml_config(
                    should_run_glt_backend=should_run_glt_backend,
                    output_format=output_format,
                )
            )


class PreprocessedMetadataCheckTest(TestCase):
    def _check_metadata(
        self, tfrecord_uri_prefix: str, parquet_uri_prefix: str
    ) -> None:
        """Writes metadata with the given data prefixes on its node and edge, then validates it."""
        preprocessed_metadata_pb = preprocessed_metadata_pb2.PreprocessedMetadata()
        node_metadata = (
            preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata[0]
        )
        node_metadata.node_id_key = "node_id"
        node_metadata.schema_uri = "gs://bucket/node/schema.pbtxt"
        node_metadata.transform_fn_assets_uri = "gs://bucket/node/transform_fn"
        node_metadata.enumerated_node_ids_bq_table = "project.dataset.node_ids"
        node_metadata.enumerated_node_data_bq_table = "project.dataset.node_data"
        node_metadata.tfrecord_uri_prefix = tfrecord_uri_prefix
        node_metadata.parquet_uri_prefix = parquet_uri_prefix
        edge_metadata = (
            preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata[0]
        )
        edge_metadata.src_node_id_key = "src"
        edge_metadata.dst_node_id_key = "dst"
        main_edge_info = edge_metadata.main_edge_info
        main_edge_info.schema_uri = "gs://bucket/edge/schema.pbtxt"
        main_edge_info.transform_fn_assets_uri = "gs://bucket/edge/transform_fn"
        main_edge_info.enumerated_edge_data_bq_table = "project.dataset.edge_data"
        main_edge_info.tfrecord_uri_prefix = tfrecord_uri_prefix
        main_edge_info.parquet_uri_prefix = parquet_uri_prefix

        metadata_file = tempfile.NamedTemporaryFile(suffix=".yaml")
        self.addCleanup(metadata_file.close)
        ProtoUtils().write_proto_to_yaml(
            proto=preprocessed_metadata_pb, uri=LocalUri(metadata_file.name)
        )
        gbml_config_pb = gbml_config_pb2.GbmlConfig(
            graph_metadata=graph_schema_pb2.GraphMetadata(
                condensed_node_type_map={0: "paper"},
                condensed_edge_type_map={
                    0: graph_schema_pb2.EdgeType(
                        src_node_type="paper", relation="cites", dst_node_type="paper"
                    )
                },
            ),
            shared_config=gbml_config_pb2.GbmlConfig.SharedConfig(
                preprocessed_metadata_uri=metadata_file.name
            ),
        )
        check_if_preprocessed_metadata_valid(gbml_config_pb=gbml_config_pb)

    def test_accepts_tfrecord_output(self) -> None:
        self._check_metadata(
            tfrecord_uri_prefix="gs://bucket/features/", parquet_uri_prefix=""
        )

    def test_accepts_parquet_output(self) -> None:
        self._check_metadata(
            tfrecord_uri_prefix="", parquet_uri_prefix="gs://bucket/parquet/"
        )

    def test_rejects_output_without_data_prefix(self) -> None:
        with self.assertRaises(AssertionError):
            self._check_metadata(tfrecord_uri_prefix="", parquet_uri_prefix="")


if __name__ == "__main__":
    absltest.main()
