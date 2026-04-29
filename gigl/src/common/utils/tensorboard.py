"""TensorBoard writer for GiGL training entrypoints."""

import os
from typing import Any, Optional

import tensorflow as tf

from gigl.common import Uri

_VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY = "AIP_TENSORBOARD_LOG_DIR"


def _resolve_log_dir(configured_uri: Optional[Uri]) -> Optional[str]:
    """Resolve the TensorBoard log directory.

    Vertex AI populates ``AIP_TENSORBOARD_LOG_DIR`` when ``baseOutputDirectory``
    is configured on a CustomJob. Outside Vertex AI, GiGL falls back to the
    URI from the task config.

    Args:
        configured_uri: The TensorBoard URI from GiGL config.

    Returns:
        The resolved log directory, or ``None`` when no directory is available.
    """
    vertex_log_dir = os.environ.get(_VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY)
    if vertex_log_dir:
        return vertex_log_dir
    if configured_uri is None:
        return None
    return configured_uri.uri


class TensorBoardWriter:
    """Writes scalar metrics to TensorBoard.

    No-ops when disabled or when no log directory is available, so callers
    never see ``Optional[TensorBoardWriter]`` plumbing.

    The writer flushes after every ``log()`` call so that Vertex's TensorBoard
    UI sees events live as training progresses.

    Example:
        >>> with TensorBoardWriter.from_uri(uri, enabled=is_chief and should_log) as tb:
        ...     tb.log({"Loss/train": loss, "Loss/val": vloss}, step=batch_idx)
    """

    def __init__(self, log_dir: Optional[str]) -> None:
        """Initialize the writer.

        Args:
            log_dir: Destination directory for TensorBoard events. When
                ``None``, the writer is a no-op and allocates no TF resources.
        """
        self._writer: Optional[Any] = (
            tf.summary.create_file_writer(log_dir) if log_dir else None
        )
        self._closed = False

    @classmethod
    def from_uri(
        cls,
        configured_uri: Optional[Uri],
        *,
        enabled: bool = True,
    ) -> "TensorBoardWriter":
        """Build a writer with Vertex AI env-var precedence.

        When ``enabled`` is ``False``, returns a no-op writer without reading
        the environment or the configured URI.

        Args:
            configured_uri: The TensorBoard URI from GiGL config. Used only
                when ``AIP_TENSORBOARD_LOG_DIR`` is unset.
            enabled: Whether this caller is responsible for writing events.
                Typically ``should_log_to_tensorboard and is_chief_process``.

        Returns:
            A ``TensorBoardWriter`` instance — real if enabled and a log
            directory was resolved, no-op otherwise.
        """
        if not enabled:
            return cls(log_dir=None)
        return cls(log_dir=_resolve_log_dir(configured_uri))

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Write each metric scalar at ``step`` and flush.

        No-ops when the writer is disabled or already closed.

        Args:
            metrics: Mapping of TensorBoard tag to scalar value. All entries
                are written at the same step.
            step: TensorBoard step for the events.
        """
        if self._writer is None or self._closed:
            return
        with self._writer.as_default():
            for tag, value in metrics.items():
                tf.summary.scalar(tag, value, step=step)
            self._writer.flush()

    def close(self) -> None:
        """Close the underlying TF writer.

        Idempotent; safe to call multiple times and on no-op writers.
        """
        if self._writer is not None and not self._closed:
            self._writer.close()
            self._closed = True

    def __enter__(self) -> "TensorBoardWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
