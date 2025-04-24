from tempfile import NamedTemporaryFile
from typing import Optional, Type, TypeVar

import yaml
from google.protobuf import message
from google.protobuf.json_format import MessageToDict, ParseDict

from gigl.common import LocalUri, Uri
from gigl.common.logger import Logger
from gigl.src.common.utils.file_loader import FileLoader

logger = Logger()

T = TypeVar("T", bound=message.Message)


class ProtoUtils:
    def __init__(self, project: Optional[str] = None) -> None:
        self.__file_loader = FileLoader(project=project)

    def read_proto_from_yaml(self, uri: Uri, proto_cls: Type[T]) -> T:
        tfh = self.__file_loader.load_to_temp_file(file_uri_src=uri, delete=False)
        with open(tfh.name, "r") as file:
            obj_dict = yaml.load(file, Loader=yaml.FullLoader)
        tfh.close()
        proto = ParseDict(js_dict=obj_dict, message=proto_cls())
        return proto

    def read_proto_from_binary(self, uri: Uri, proto_cls: Type[T]) -> T:
        tfh = self.__file_loader.load_to_temp_file(file_uri_src=uri, delete=False)
        with open(tfh.name, "rb") as file:
            proto_bytes = file.read()
        tfh.close()
        proto = proto_cls()
        proto.ParseFromString(proto_bytes)
        return proto

    def write_proto_to_yaml(self, proto: message.Message, uri: Uri) -> None:
        proto_dict = MessageToDict(message=proto)
        tfh = NamedTemporaryFile(delete=False)
        with open(tfh.name, "w") as file:
            yaml_str = yaml.dump(proto_dict, default_flow_style=False)
            file.write(yaml_str)
        tfh.close()
        self.__file_loader.load_file(file_uri_src=LocalUri(tfh.name), file_uri_dst=uri)

    def write_proto_to_binary(self, proto: message.Message, uri: Uri) -> None:
        tfh = NamedTemporaryFile(delete=False)
        with open(tfh.name, "wb") as file:
            proto_bytes = proto.SerializeToString()
            file.write(proto_bytes)
        tfh.close()
        self.__file_loader.load_file(file_uri_src=LocalUri(tfh.name), file_uri_dst=uri)
