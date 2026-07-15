import tempfile
from uuid import uuid4

from gigl.common import LocalUri
from gigl.common.utils.file_loader import FileLoader
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from snapchat.research.gbml import gbml_config_pb2
from tests.test_assets.graph_metadata_constants import (
    EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES,
    EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES,
    EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB,
    EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER,
    EXAMPLE_HETEROGENEOUS_PREPROCESSED_METADATA_PB,
)
from tests.test_assets.test_case import TestCase


class GbmlConfigTest(TestCase):
    def setUp(self) -> None:
        self.file_loader = FileLoader()
        self.proto_utils = ProtoUtils()
        self.gbml_config_test_run_id = str(uuid4())
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.target_proto_uri = LocalUri.join(
            self.tmp_dir.name, f"{self.gbml_config_test_run_id}.proto"
        )
        self.target_yaml_uri = LocalUri.join(
            self.tmp_dir.name, f"{self.gbml_config_test_run_id}.yaml"
        )

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_gbml_config_read_and_write_proto(self):
        obj = gbml_config_pb2.GbmlConfig()
        obj.dataset_config.data_preprocessor_config.data_preprocessor_config_cls_path = self.gbml_config_test_run_id

        self.proto_utils.write_proto_to_binary(proto=obj, uri=self.target_proto_uri)
        obj2 = self.proto_utils.read_proto_from_binary(
            uri=self.target_proto_uri, proto_cls=gbml_config_pb2.GbmlConfig
        )
        self.assertEqual(obj, obj2)

    def test_gbml_config_read_and_write_yaml(self):
        obj = gbml_config_pb2.GbmlConfig()
        obj.dataset_config.data_preprocessor_config.data_preprocessor_config_cls_path = self.gbml_config_test_run_id

        self.proto_utils.write_proto_to_yaml(proto=obj, uri=self.target_yaml_uri)
        obj2 = self.proto_utils.read_proto_from_yaml(
            uri=self.target_yaml_uri, proto_cls=gbml_config_pb2.GbmlConfig
        )
        self.assertEqual(obj, obj2)

    def test_typed_keyed_feature_maps_match_condensed_maps(self):
        """`*_to_feature_*_map` properties are typed-key views of the condensed maps.

        Builds a GbmlConfigPbWrapper with both graph_metadata and
        preprocessed_metadata populated (heterogeneous example fixture), then
        verifies each typed-keyed property contains the same values as the
        underlying condensed-keyed map after joining through the
        graph_metadata's condensed→typed map.
        """
        # Stage the heterogeneous preprocessed metadata on disk so
        # __load_preprocessed_metadata_pb_wrapper can read it via the URI.
        preprocessed_metadata_uri = LocalUri.join(
            self.tmp_dir.name,
            f"{self.gbml_config_test_run_id}_preprocessed_metadata.yaml",
        )
        self.proto_utils.write_proto_to_yaml(
            proto=EXAMPLE_HETEROGENEOUS_PREPROCESSED_METADATA_PB,
            uri=preprocessed_metadata_uri,
        )

        gbml_config_pb = gbml_config_pb2.GbmlConfig(
            graph_metadata=EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB,
        )
        gbml_config_pb.shared_config.preprocessed_metadata_uri = (
            preprocessed_metadata_uri.uri
        )

        gbml_config_pb_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)

        condensed_to_node_type = EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER.condensed_node_type_to_node_type_map
        condensed_to_edge_type = EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER.condensed_edge_type_to_edge_type_map
        expected_node_types = {
            condensed_to_node_type[c]
            for c in EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES
        }
        expected_edge_types = {
            condensed_to_edge_type[c]
            for c in EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES
        }

        # Keysets reflect every condensed type rekeyed via the type map.
        self.assertEqual(
            set(gbml_config_pb_wrapper.node_type_to_feature_dim_map.keys()),
            expected_node_types,
        )
        self.assertEqual(
            set(gbml_config_pb_wrapper.node_type_to_feature_schema_map.keys()),
            expected_node_types,
        )
        self.assertEqual(
            set(gbml_config_pb_wrapper.edge_type_to_feature_dim_map.keys()),
            expected_edge_types,
        )

        # Values agree with the condensed-keyed source for every entry.
        preprocessed = gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper
        for (
            condensed_node_type,
            expected_dim,
        ) in preprocessed.condensed_node_type_to_feature_dim_map.items():
            node_type = condensed_to_node_type[condensed_node_type]
            self.assertEqual(
                gbml_config_pb_wrapper.node_type_to_feature_dim_map[node_type],
                expected_dim,
            )
        for (
            condensed_node_type,
            expected_schema,
        ) in preprocessed.condensed_node_type_to_feature_schema_map.items():
            node_type = condensed_to_node_type[condensed_node_type]
            self.assertEqual(
                gbml_config_pb_wrapper.node_type_to_feature_schema_map[node_type],
                expected_schema,
            )
        for (
            condensed_edge_type,
            expected_dim,
        ) in preprocessed.condensed_edge_type_to_feature_dim_map.items():
            edge_type = condensed_to_edge_type[condensed_edge_type]
            self.assertEqual(
                gbml_config_pb_wrapper.edge_type_to_feature_dim_map[edge_type],
                expected_dim,
            )

    def test_typed_keyed_feature_maps_default_empty_when_metadata_missing(self):
        """When neither graph_metadata nor preprocessed_metadata are populated
        on the input GbmlConfig, the typed-keyed maps default to empty dicts.

        This matches the behavior of the underlying loaders, which silently
        skip population when their inputs are absent.
        """
        gbml_config_pb = gbml_config_pb2.GbmlConfig()
        gbml_config_pb_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)

        self.assertEqual(gbml_config_pb_wrapper.node_type_to_feature_dim_map, {})
        self.assertEqual(gbml_config_pb_wrapper.node_type_to_feature_schema_map, {})
        self.assertEqual(gbml_config_pb_wrapper.edge_type_to_feature_dim_map, {})
