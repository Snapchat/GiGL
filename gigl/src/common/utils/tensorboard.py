"""Shared TensorBoard helpers for GiGL training entrypoints."""

import os
from typing import Any, Optional

import tensorflow as tf

from gigl.common import Uri

VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY = "AIP_TENSORBOARD_LOG_DIR"


def resolve_tensorboard_log_dir(
    configured_tensorboard_log_uri: Optional[Uri],
) -> Optional[str]:
    """Resolve the TensorBoard log directory for the current runtime.

    Vertex AI sets ``AIP_TENSORBOARD_LOG_DIR`` when ``baseOutputDirectory`` is
    configured on a CustomJob. Outside Vertex AI, GiGL falls back to the
    TensorBoard URI stored in the task config.

    Args:
        configured_tensorboard_log_uri: The TensorBoard URI from GiGL config.

    Returns:
        The resolved log directory, or ``None`` when no directory is available.
    """
    vertex_tensorboard_log_dir = os.environ.get(VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY)
    if vertex_tensorboard_log_dir:
        return vertex_tensorboard_log_dir

    if configured_tensorboard_log_uri is None:
        return None

    return configured_tensorboard_log_uri.uri


def create_tensorboard_writer(
    should_log_to_tensorboard: bool,
    configured_tensorboard_log_uri: Optional[Uri],
    should_write_events: bool,
) -> Optional[Any]:
    """Create a TensorBoard summary writer when logging is enabled.

    Args:
        should_log_to_tensorboard: Whether TensorBoard logging is enabled.
        configured_tensorboard_log_uri: The TensorBoard URI from GiGL config.
        should_write_events: Whether the current process should emit events.

    Returns:
        A TensorBoard writer, or ``None`` when logging should be skipped.
    """
    if not should_log_to_tensorboard or not should_write_events:
        return None

    tensorboard_log_dir = resolve_tensorboard_log_dir(
        configured_tensorboard_log_uri=configured_tensorboard_log_uri
    )
    if tensorboard_log_dir is None:
        return None

    return tf.summary.create_file_writer(tensorboard_log_dir)


def write_tensorboard_scalar(
    writer: Optional[Any],
    tag: str,
    value: float,
    step: int,
) -> None:
    """Write a scalar TensorBoard event when a writer is available.

    Args:
        writer: TensorBoard writer created by ``create_tensorboard_writer``.
        tag: The TensorBoard series name.
        value: Scalar value to log.
        step: TensorBoard step for the event.
    """
    if writer is None:
        return

    with writer.as_default():
        tf.summary.scalar(tag, value, step=step)
        writer.flush()


def close_tensorboard_writer(writer: Optional[Any]) -> None:
    """Close a TensorBoard writer when one exists.

    Args:
        writer: TensorBoard writer created by ``create_tensorboard_writer``.
    """
    if writer is not None:
        writer.close()
