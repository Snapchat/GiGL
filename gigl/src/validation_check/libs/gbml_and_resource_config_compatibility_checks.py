"""
Compatibility checks between GbmlConfig (template config) and GiglResourceConfig (resource config).

These checks ensure that graph store mode configurations are consistent across both configs.
If graph store mode is set up for trainer or inferencer in one config, it must be set up in the other.
"""

from typing import Literal

from google.protobuf.message import Message

from gigl.common.logger import Logger
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from snapchat.research.gbml import gigl_resource_config_pb2

logger = Logger()


def _gbml_config_has_graph_store(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    component: Literal["trainer", "inferencer"],
) -> bool:
    """
    Check if the GbmlConfig has graph_store_storage_config set for inferencer.

    Args:
        gbml_config_pb_wrapper: The GbmlConfig wrapper to check.

    Returns:
        True if graph_store_storage_config is set for inferencer, False otherwise.
    """
    if component == "inferencer":
        config: Message = gbml_config_pb_wrapper.gbml_config_pb.inferencer_config
    elif component == "trainer":
        config = gbml_config_pb_wrapper.gbml_config_pb.trainer_config
    else:
        raise ValueError(
            f"Invalid component: {component}. Must be 'inferencer' or 'trainer'."
        )
    return config.HasField("graph_store_storage_config")


def _resource_config_has_graph_store(
    resource_config_wrapper: GiglResourceConfigWrapper,
    component: Literal["trainer", "inferencer"],
) -> bool:
    """
    Check if the GiglResourceConfig has VertexAiGraphStoreConfig set for the given component.

    Args:
        resource_config_wrapper: The resource config wrapper to check.

    Returns:
        True if VertexAiGraphStoreConfig is set for trainer, False otherwise.
    """
    if component == "trainer":
        config: Message = resource_config_wrapper.trainer_config
    elif component == "inferencer":
        config = resource_config_wrapper.inferencer_config
    else:
        raise ValueError(
            f"Invalid component: {component}. Must be 'trainer' or 'inferencer'."
        )
    return isinstance(config, gigl_resource_config_pb2.VertexAiGraphStoreConfig)


def check_trainer_graph_store_compatibility(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    resource_config_wrapper: GiglResourceConfigWrapper,
) -> None:
    """
    Check that trainer graph store mode is consistently configured across both configs.

    If graph_store_storage_config is set in GbmlConfig.trainer_config, then
    VertexAiGraphStoreConfig must be set in GiglResourceConfig.trainer_resource_config,
    and vice versa. Also validates that storage_command is set when graph store mode is enabled.

    Args:
        gbml_config_pb_wrapper: The GbmlConfig wrapper (template config).
        resource_config_wrapper: The GiglResourceConfig wrapper (resource config).

    Raises:
        AssertionError: If graph store configurations are not compatible or storage_command is missing.
    """
    logger.info(
        "Config validation check: trainer graph store compatibility between template and resource configs."
    )

    gbml_has_graph_store = _gbml_config_has_graph_store(
        gbml_config_pb_wrapper, "trainer"
    )
    resource_has_graph_store = _resource_config_has_graph_store(
        resource_config_wrapper, "trainer"
    )

    if gbml_has_graph_store ^ resource_has_graph_store:
        raise AssertionError(
            f"If one of GbmlConfig.trainer_config.graph_store_storage_config or GiglResourceConfig.trainer_resource_config is set, the other must also be set. GbmlConfig.trainer_config.graph_store_storage_config is set: {gbml_has_graph_store}, GiglResourceConfig.trainer_resource_config is set: {resource_has_graph_store}."
        )


def check_inferencer_graph_store_compatibility(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    resource_config_wrapper: GiglResourceConfigWrapper,
) -> None:
    """
    Check that inferencer graph store mode is consistently configured across both configs.

    If graph_store_storage_config is set in GbmlConfig.inferencer_config, then
    VertexAiGraphStoreConfig must be set in GiglResourceConfig.inferencer_resource_config,
    and vice versa. Also validates that storage_command is set when graph store mode is enabled.

    Args:
        gbml_config_pb_wrapper: The GbmlConfig wrapper (template config).
        resource_config_wrapper: The GiglResourceConfig wrapper (resource config).

    Raises:
        AssertionError: If graph store configurations are not compatible or storage_command is missing.
    """
    logger.info(
        "Config validation check: inferencer graph store compatibility between template and resource configs."
    )

    gbml_has_graph_store = _gbml_config_has_graph_store(
        gbml_config_pb_wrapper, "inferencer"
    )
    resource_has_graph_store = _resource_config_has_graph_store(
        resource_config_wrapper, "inferencer"
    )

    if gbml_has_graph_store and not resource_has_graph_store:
        raise AssertionError(
            f"If one of GbmlConfig.inferencer_config.graph_store_storage_config or GiglResourceConfig.inferencer_resource_config is set, the other must also be set. GbmlConfig.inferencer_config.graph_store_storage_config is set: {gbml_has_graph_store}, GiglResourceConfig.inferencer_resource_config is set: {resource_has_graph_store}."
        )
