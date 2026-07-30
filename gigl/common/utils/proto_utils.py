from tempfile import NamedTemporaryFile
from typing import Optional, Type, TypeVar, cast

import yaml
from google.protobuf import message
from google.protobuf.json_format import MessageToDict, ParseDict
from omegaconf import OmegaConf

from gigl.common import LocalUri, Uri
from gigl.common.logger import Logger
from gigl.common.omegaconf_resolvers import register_resolvers
from gigl.common.utils.hydra_config import compose_yaml_config
from gigl.src.common.utils.file_loader import FileLoader

logger = Logger()

T = TypeVar("T", bound=message.Message)


def proto_to_yaml(proto: message.Message) -> str:
    """Serialize a protobuf message to canonical YAML.

    Args:
        proto: Protobuf message to serialize.

    Returns:
        YAML containing the protobuf JSON representation.
    """
    proto_dict = MessageToDict(message=proto)
    return yaml.safe_dump(proto_dict, default_flow_style=False, sort_keys=True)


class ProtoUtils:
    def __init__(self, project: Optional[str] = None) -> None:
        self.__file_loader = FileLoader(project=project)
        register_resolvers()

    def read_proto_from_yaml(self, uri: Uri, proto_cls: Type[T]) -> T:
        tfh = self.__file_loader.load_to_temp_file(file_uri_src=uri, delete=False)
        with open(tfh.name, "r") as file:
            raw_data = yaml.safe_load(file)
            omega_conf_obj = OmegaConf.create(raw_data)
        tfh.close()
        obj_dict = OmegaConf.to_object(omega_conf_obj)
        if not isinstance(obj_dict, dict):
            raise TypeError(
                f"ProtoUtils.read_proto_from_yaml expected a mapping at the YAML root for "
                f"{uri}, got {type(obj_dict).__name__}."
            )
        proto = ParseDict(js_dict=cast(dict, obj_dict), message=proto_cls())
        return proto

    def compose_proto_from_yaml(self, uri: LocalUri, proto_cls: Type[T]) -> T:
        """Compose a local YAML config with Hydra and parse it as a protobuf."""
        obj_dict = compose_yaml_config(uri=uri)
        proto = ParseDict(js_dict=cast(dict, obj_dict), message=proto_cls())
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
