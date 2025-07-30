"""
This module provides functions to load and save distributed checkpoints
using the Torch Distributed Checkpointing API.
"""

import tempfile
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor, Future
import torch.nn as nn
import torch.optim as optim
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE


from gigl.common import GcsUri, LocalUri, Uri
from gigl.common.logger import Logger
from gigl.src.common.utils.file_loader import FileLoader

logger = Logger()

class AppState(Stateful):
    """
    This is a useful wrapper for checkpointing an application state. Since this
    object is compliant with the Stateful protocol, DCP will automatically
    call state_dict/load_state_dict as needed in the dcp.save/load APIs.

    We take advantage of this wrapper to hande calling distributed state dict
    methods on the model and optimizer.

    See https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html
    for more details.
    """

    MODEL_KEY = "model"
    OPTIMIZER_KEY = "optimizer"
    APP_STATE_KEY = "app"


    def __init__(self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict() if self.optimizer else None
        return {
            self.MODEL_KEY: model_state_dict,
            self.OPTIMIZER_KEY: optimizer_state_dict,
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        self.model.load_state_dict(state_dict[self.MODEL_KEY])
        if self.optimizer and state_dict.get(self.OPTIMIZER_KEY):
            self.optimizer.load_state_dict(state_dict[self.OPTIMIZER_KEY])

    def to_state_dict(self) -> STATE_DICT_TYPE:
        """
        Converts the AppState to a state dict that can be used with DCP.
        """
        return {
            self.APP_STATE_KEY: self,
        }

def load_checkpoint_from_uri(
    state_dict: STATE_DICT_TYPE,
    checkpoint_id: Uri,
):
    assert isinstance(checkpoint_id, LocalUri) or isinstance(
        checkpoint_id, GcsUri
    ), "checkpoint_id must be a LocalUri or GcsUri."
    local_uri = (
        checkpoint_id
        if isinstance(checkpoint_id, LocalUri)
        else LocalUri(tempfile.mkdtemp(prefix="checkpoint"))
    )
    if isinstance(checkpoint_id, GcsUri):
        # If the URI is a GCS URI, we need to download it first
        file_loader = FileLoader()
        file_loader.load_directory(dir_uri_src=checkpoint_id, dir_uri_dst=local_uri)
        logger.info(f"Downloaded checkpoint from GCS: {checkpoint_id} to {local_uri}")

    reader = dcp.FileSystemReader(path=local_uri.uri)
    dcp.load(state_dict=state_dict, storage_reader=reader)
    logger.info(f"Loaded checkpoint from {checkpoint_id}")


def save_checkpoint_to_uri(
    state_dict: STATE_DICT_TYPE,
    checkpoint_id: Uri,
    should_save_asynchronously: bool = False,
) -> Union[Future[Uri], Uri]:
    """
    Saves the state_dict to a specified checkpoint_id URI using the Torch Distributed Checkpointing API.

    If the checkpoint_id is a GCS URI, it will first save the checkpoint
    locally and then upload it to GCS.

    If `should_save_asynchronously` is True, the save operation will be
    performed asynchronously, returning a Future object. Otherwise, it will
    block until the save operation is complete.

    Args:
        state_dict (STATE_DICT_TYPE): The state dictionary to save.
        checkpoint_id (Uri): The URI where the checkpoint will be saved.
        should_save_asynchronously (bool): If True, saves the checkpoint asynchronously.
    Returns:
        Union[Future[Uri], Uri]: The URI where the checkpoint was saved, or
        a Future object if saved asynchronously.
    Raises:
        AssertionError: If checkpoint_id is not a LocalUri or GcsUri.
    """

    def _save_checkpoint(
        checkpoint_id: Uri, local_uri: LocalUri, checkpoint_future: Optional[Future] = None
    ) -> Uri:
        # If we have a checkpoint future, we will wait for it to complete (async save)
        if checkpoint_future:
            checkpoint_future.result()

        if isinstance(checkpoint_id, GcsUri):
            # If the URI is a GCS URI, we need to ensure the file is uploaded
            # to GCS after saving it locally.
            file_loader = FileLoader()
            file_loader.load_directory(dir_uri_src=local_uri, dir_uri_dst=checkpoint_id)
            logger.info(f"Uploaded checkpoint to GCS: {checkpoint_id}")

        return checkpoint_id

    assert isinstance(checkpoint_id, LocalUri) or isinstance(
        checkpoint_id, GcsUri
    ), "checkpoint_id must be a LocalUri or GcsUri."
    local_uri = (
        checkpoint_id
        if isinstance(checkpoint_id, LocalUri)
        else LocalUri(tempfile.mkdtemp(prefix="checkpoint"))
    )

    writer = dcp.FileSystemWriter(path=local_uri.uri)

    if should_save_asynchronously:
        logger.info(f"Saving checkpoint asynchronously to {checkpoint_id}")
        checkpoint_future = dcp.async_save(state_dict, storage_writer=writer)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            _save_checkpoint, checkpoint_id, local_uri, checkpoint_future
        )
        return future
    else:
        logger.info(f"Saving checkpoint synchronously to {checkpoint_id}")
        dcp.save(state_dict, storage_writer=writer)
        return _save_checkpoint(checkpoint_id, local_uri, None)
