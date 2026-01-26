"""
Compatibility checks between GbmlConfig (template config) and GiglResourceConfig (resource config).

These checks ensure that graph store mode configurations are consistent across both configs.
If graph store mode is set up for trainer or inferencer in one config, it must be set up in the other.
"""

from gigl.common.logger import Logger
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from snapchat.research.gbml import gigl_resource_config_pb2

logger = Logger()


def _gbml_config_has_trainer_graph_store(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> bool:
    """
    Check if the GbmlConfig has graph_store_storage_config set for trainer.

    Args:
        gbml_config_pb_wrapper: The GbmlConfig wrapper to check.

    Returns:
        True if graph_store_storage_config is set for trainer, False otherwise.
    """
    trainer_config = gbml_config_pb_wrapper.gbml_config_pb.trainer_config
    return trainer_config.HasField("graph_store_storage_config")


def _gbml_config_has_inferencer_graph_store(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> bool:
    """
    Check if the GbmlConfig has graph_store_storage_config set for inferencer.

    Args:
        gbml_config_pb_wrapper: The GbmlConfig wrapper to check.

    Returns:
        True if graph_store_storage_config is set for inferencer, False otherwise.
    """
    inferencer_config = gbml_config_pb_wrapper.gbml_config_pb.inferencer_config
    return inferencer_config.HasField("graph_store_storage_config")


def _resource_config_has_trainer_graph_store(
    resource_config_wrapper: GiglResourceConfigWrapper,
) -> bool:
    """
    Check if the GiglResourceConfig has VertexAiGraphStoreConfig set for trainer.

    Args:
        resource_config_wrapper: The resource config wrapper to check.

    Returns:
        True if VertexAiGraphStoreConfig is set for trainer, False otherwise.
    """
    trainer_config = resource_config_wrapper.trainer_config
    return isinstance(trainer_config, gigl_resource_config_pb2.VertexAiGraphStoreConfig)


def _resource_config_has_inferencer_graph_store(
    resource_config_wrapper: GiglResourceConfigWrapper,
) -> bool:
    """
    Check if the GiglResourceConfig has VertexAiGraphStoreConfig set for inferencer.

    Args:
        resource_config_wrapper: The resource config wrapper to check.

    Returns:
        True if VertexAiGraphStoreConfig is set for inferencer, False otherwise.
    """
    inferencer_config = resource_config_wrapper.inferencer_config
    return isinstance(
        inferencer_config, gigl_resource_config_pb2.VertexAiGraphStoreConfig
    )


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

    gbml_has_graph_store = _gbml_config_has_trainer_graph_store(gbml_config_pb_wrapper)
    resource_has_graph_store = _resource_config_has_trainer_graph_store(
        resource_config_wrapper
    )

    if gbml_has_graph_store and not resource_has_graph_store:
        raise AssertionError(
            "GbmlConfig.trainer_config.graph_store_storage_config is set, but "
            "GiglResourceConfig.trainer_resource_config does not use VertexAiGraphStoreConfig. "
            "Both configs must use graph store mode for trainer, or neither should."
        )

    if resource_has_graph_store and not gbml_has_graph_store:
        raise AssertionError(
            "GiglResourceConfig.trainer_resource_config uses VertexAiGraphStoreConfig, but "
            "GbmlConfig.trainer_config.graph_store_storage_config is not set. "
            "Both configs must use graph store mode for trainer, or neither should."
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

    gbml_has_graph_store = _gbml_config_has_inferencer_graph_store(
        gbml_config_pb_wrapper
    )
    resource_has_graph_store = _resource_config_has_inferencer_graph_store(
        resource_config_wrapper
    )

    if gbml_has_graph_store and not resource_has_graph_store:
        raise AssertionError(
            "GbmlConfig.inferencer_config.graph_store_storage_config is set, but "
            "GiglResourceConfig.inferencer_resource_config does not use VertexAiGraphStoreConfig. "
            "Both configs must use graph store mode for inferencer, or neither should."
        )

    if resource_has_graph_store and not gbml_has_graph_store:
        raise AssertionError(
            "GiglResourceConfig.inferencer_resource_config uses VertexAiGraphStoreConfig, but "
            "GbmlConfig.inferencer_config.graph_store_storage_config is not set. "
            "Both configs must use graph store mode for inferencer, or neither should."
        )
