from concurrent.futures import Future
from typing import Optional, Union

import torch
import torch.nn as nn
from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.experimental.knowledge_graph_embedding.common.dist_checkpoint import (
    AppState,
    load_checkpoint_from_uri,
    save_checkpoint_to_uri,
)
from gigl.experimental.knowledge_graph_embedding.lib.config.training import (
    CheckpointingConfig,
)

logger = Logger()


def maybe_load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpointing_config: CheckpointingConfig,
) -> bool:
    """
    Load the model and optimizer checkpoints if they exist.

    Args:
        model: The model to load the checkpoint into.
        optimizer: The optimizer to load the checkpoint into.
        checkpointing_config: The training configuration containing the checkpointing paths.

    Returns:
        bool: True if the model and optimizer were loaded successfully, False otherwise.
    """

    if not checkpointing_config.load_from_path:
        logger.info(
            f"No checkpoint specified to load from. Skipping loading checkpoints."
        )
        return False

    load_from_checkpoint_path: Uri = UriFactory.create_uri(
        checkpointing_config.load_from_path
    )
    logger.info(
        f"Loading model and optimizer from checkpoint path: {load_from_checkpoint_path}"
    )
    app_state = AppState(model=model, optimizer=optimizer)
    load_checkpoint_from_uri(
        state_dict=app_state.to_state_dict(),
        checkpoint_id=load_from_checkpoint_path,
    )
    return True


def maybe_save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpointing_config: CheckpointingConfig,
    checkpoint_id: str = "",
) -> Optional[Union[Future[Uri], Uri]]:
    """
    Save the model and optimizer checkpoints if specified in the training configuration.

    Args:
        model: The model to save the checkpoint for.
        optimizer: The optimizer to save the checkpoint for.
        checkpointing_config: The training configuration containing the checkpointing paths.
        checkpoint_id: An optional identifier for the checkpoint, used to differentiate between checkpoints if needed.

    Returns:
        Optional[Union[Future[Uri], Uri]]: The URI where the checkpoint was saved, or a Future object if saved asynchronously.
        If no checkpointing path is specified, returns None.
    """

    # Set up the checkpoint saving paths.
    should_save_checkpoint_async = checkpointing_config.should_save_async
    logger.info(f"Got saving condition: {should_save_checkpoint_async}")
    if not checkpointing_config.save_to_path:
        logger.info(f"No checkpoint specified to save to. Skipping saving checkpoint.")
        return None

    save_to_checkpoint_path = UriFactory.create_uri(checkpointing_config.save_to_path)
    checkpoint_id_uri = Uri.join(save_to_checkpoint_path, checkpoint_id)
    app_state = AppState(model=model, optimizer=optimizer)
    return save_checkpoint_to_uri(
        state_dict=app_state.to_state_dict(),
        checkpoint_id=checkpoint_id_uri,
        should_save_asynchronously=should_save_checkpoint_async,
    )
