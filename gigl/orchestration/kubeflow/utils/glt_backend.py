from gigl.common import Uri
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper


def resolve_should_use_glt_backend(task_config_uri: Uri) -> bool:
    config = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=task_config_uri
    )
    return config.should_use_glt_backend
