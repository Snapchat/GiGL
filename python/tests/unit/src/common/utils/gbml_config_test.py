import tempfile
import unittest
from uuid import uuid4

from gigl.common import LocalUri
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.utils.file_loader import FileLoader
from snapchat.research.gbml import gbml_config_pb2


class GbmlConfigTest(unittest.TestCase):
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
        obj.dataset_config.data_preprocessor_config.data_preprocessor_config_cls_path = (
            self.gbml_config_test_run_id
        )

        self.proto_utils.write_proto_to_binary(proto=obj, uri=self.target_proto_uri)
        obj2 = self.proto_utils.read_proto_from_binary(
            uri=self.target_proto_uri, proto_cls=gbml_config_pb2.GbmlConfig
        )
        self.assertEqual(obj, obj2)

    def test_gbml_config_read_and_write_yaml(self):
        obj = gbml_config_pb2.GbmlConfig()
        obj.dataset_config.data_preprocessor_config.data_preprocessor_config_cls_path = (
            self.gbml_config_test_run_id
        )

        self.proto_utils.write_proto_to_yaml(proto=obj, uri=self.target_yaml_uri)
        obj2 = self.proto_utils.read_proto_from_yaml(
            uri=self.target_yaml_uri, proto_cls=gbml_config_pb2.GbmlConfig
        )
        self.assertEqual(obj, obj2)
