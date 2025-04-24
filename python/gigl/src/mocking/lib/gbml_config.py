from gigl.common import UriFactory
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from snapchat.research.gbml import gbml_config_pb2, preprocessed_metadata_pb2


def load_preprocessed_metadata_pb_wrapper_from_gbml_config_pb(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> PreprocessedMetadataPbWrapper:
    proto_utils = ProtoUtils()
    preprocessed_metadata_pb = proto_utils.read_proto_from_yaml(
        uri=UriFactory.create_uri(
            uri=gbml_config_pb.shared_config.preprocessed_metadata_uri
        ),
        proto_cls=preprocessed_metadata_pb2.PreprocessedMetadata,
    )
    preprocessed_metadata_pb_wrapper = PreprocessedMetadataPbWrapper(
        preprocessed_metadata_pb=preprocessed_metadata_pb
    )
    return preprocessed_metadata_pb_wrapper
