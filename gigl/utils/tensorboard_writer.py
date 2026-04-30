"""TensorBoard writer for GiGL training entrypoints."""

import os
from typing import Any, Optional

import tensorflow as tf

# Vertex AI sets this env var to ``<baseOutputDirectory>/logs/`` (or
# ``<baseOutputDirectory>/<trial_id>/logs/`` for HyperparameterTuningJob trials)
# when ``CustomJobSpec.baseOutputDirectory`` is configured. GiGL's launcher
# derives ``baseOutputDirectory`` from the GbmlConfig's ``tensorboardLogsUri``
# (see ``gigl/src/common/vertex_ai_launcher.py``), so within a GiGL-launched
# trainer this env var is the authoritative log directory.
#
# References:
#   https://cloud.google.com/vertex-ai/docs/training/code-requirements
#   https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec#FIELDS.base_output_directory
_VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY = "AIP_TENSORBOARD_LOG_DIR"


class TensorBoardWriter:
    """Writes scalar metrics to TensorBoard.

    No-ops when disabled, so callers never see ``Optional[TensorBoardWriter]``
    plumbing across chief / non-chief ranks.

    The writer flushes after every ``log()`` call so that Vertex's TensorBoard
    UI sees events live as training progresses.

    Example:
        >>> with TensorBoardWriter.from_env(enabled=is_chief_process) as tb:
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
    def from_env(cls, *, enabled: bool = True) -> "TensorBoardWriter":
        """Build a writer from Vertex AI's ``AIP_TENSORBOARD_LOG_DIR`` env var.

        When ``enabled`` is ``False``, returns a no-op writer without reading
        the environment. This is the path non-chief ranks take so they can
        share the same call sites as the chief.

        When ``enabled`` is ``True``, the env var must be set; otherwise this
        raises ``RuntimeError`` rather than silently no-op'ing. The env var is
        populated by Vertex AI from ``CustomJobSpec.baseOutputDirectory`` (see
        the references in this module's header).

        Args:
            enabled: Whether this caller is responsible for writing events.
                Typically ``is_chief_process``.

        Returns:
            A ``TensorBoardWriter`` instance — real if enabled, no-op otherwise.

        Raises:
            RuntimeError: If ``enabled`` is True and ``AIP_TENSORBOARD_LOG_DIR``
                is not set in the environment.
        """
        if not enabled:
            return cls(log_dir=None)
        log_dir = os.environ.get(_VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY)
        if not log_dir:
            raise RuntimeError(
                f"{_VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY} is not set. "
                "TensorBoardWriter.from_env() requires the trainer to run as "
                "a Vertex AI CustomJob with baseOutputDirectory configured. "
                "See https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec#FIELDS.base_output_directory."
            )
        return cls(log_dir=log_dir)

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
