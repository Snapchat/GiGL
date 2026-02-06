import argparse
import json
import os
from typing import Optional

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from snapchat.research.gbml.gigl_resource_config_pb2 import GiglResourceConfig

logger = Logger()


def _try_loading_resource_config_uri_from_pipeline_options() -> Optional[str]:
    """
    Tries to load the resource config URI from the pipeline options.
    Returns the resource config path if found, otherwise returns None.
    """
    logger.info(
        "Could not find resource config path from parsed args... Assuming running Dataflow job"
    )
    try:
        display_data = json.loads(os.environ.get("PIPELINE_OPTIONS", "{}")).get(
            "display_data", []
        )
        resource_config_path = next(
            (
                item["value"]
                for item in display_data
                if item.get("key") == "resource_config_uri"
            ),
            None,
        )
        logger.info(f"Found resource config path: {resource_config_path}")
    except json.JSONDecodeError:
        logger.error("Failed to decode PIPELINE_OPTIONS as JSON.")
        resource_config_path = None

    return resource_config_path


_resource_config: Optional[GiglResourceConfigWrapper] = None


def get_resource_config(
    resource_config_uri: Optional[Uri] = None,
) -> GiglResourceConfigWrapper:
    """
    Function call to return a resource config wrapper object
    Usage:
        resource_config = get_resource_config()
        print(resource_config.trainer_config)
    Args:
        resource_config_uri: Optional[Uri] = None
            The URI of the resource config file. If None, the function will try to load the resource config from the
            command-line argument --resource_config_uri or the environment variable RESOURCE_CONFIG_PATH. If these are
            not set, the function will try to load the resource config from the pipeline options.

    Returns:
        resource_config: GiglResourceConfigWrapper
            The resource config wrapper object
    """
    global _resource_config
    if _resource_config is not None:
        return _resource_config

    resource_config_str = None
    if resource_config_uri is not None:
        resource_config_str = str(resource_config_uri)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--resource_config_uri",
            type=str,
            required=False,
        )
        args, _ = parser.parse_known_args()
        resource_config_str = args.resource_config_uri or os.getenv(
            "RESOURCE_CONFIG_PATH"
        )

        if resource_config_str is None:
            resource_config_str = (
                _try_loading_resource_config_uri_from_pipeline_options()
            )
            if resource_config_str is None:
                raise ValueError(
                    "No resource config provided, either via command-line argument or environment variable."
                )

    os.environ["RESOURCE_CONFIG_PATH"] = resource_config_str
    resource_config_path = UriFactory.create_uri(uri=resource_config_str)

    from gigl.common.utils.proto_utils import ProtoUtils

    proto_utils = ProtoUtils()
    _resource_config = GiglResourceConfigWrapper(
        proto_utils.read_proto_from_yaml(
            resource_config_path, proto_cls=GiglResourceConfig
        )
    )

    return _resource_config


def is_resource_config_loaded() -> bool:
    """
    Checks if the resource config has been loaded.
    Returns True if the resource config has been loaded, False otherwise.
    """
    return _resource_config is not None


if __name__ == "__main__":
    resource_config = get_resource_config()
    print(resource_config)
