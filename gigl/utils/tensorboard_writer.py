"""TensorBoard writer for GiGL training entrypoints."""

import os
import re
from typing import Any, Final, Optional

import tensorflow as tf
from google.cloud import aiplatform

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
_VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY: Final[str] = "AIP_TENSORBOARD_LOG_DIR"

# Set by GiGL's launcher (``gigl/src/common/vertex_ai_launcher.py``) when the
# user requested a stable Vertex AI ``TensorboardExperiment`` for cross-job
# comparison. When all three are set on the chief rank, the writer also
# starts a background uploader (``aiplatform.start_upload_tb_log``) that
# streams events from the parent log dir to that experiment under the
# configured ``Tensorboard`` instance, with the run-name subdir surfacing
# as a distinct ``TensorboardRun``. Without these, the writer just writes
# files to ``AIP_TENSORBOARD_LOG_DIR`` and only Vertex's built-in
# auto-uploader (gated on ``jobSpec.tensorboard``) ingests them.
_GIGL_TENSORBOARD_RESOURCE_NAME_ENV_KEY: Final[str] = "GIGL_TENSORBOARD_RESOURCE_NAME"
_GIGL_TENSORBOARD_EXPERIMENT_NAME_ENV_KEY: Final[str] = (
    "GIGL_TENSORBOARD_EXPERIMENT_NAME"
)
_GIGL_TENSORBOARD_RUN_NAME_ENV_KEY: Final[str] = "GIGL_TENSORBOARD_RUN_NAME"

_TENSORBOARD_RESOURCE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^projects/(?P<project>[^/]+)"
    r"/locations/(?P<location>[^/]+)"
    r"/tensorboards/(?P<tensorboard_id>[^/]+)$"
)


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

    def __init__(
        self,
        log_dir: Optional[str],
        *,
        upload_started: bool = False,
    ) -> None:
        """Initialize the writer.

        Args:
            log_dir: Destination directory for TensorBoard events. When
                ``None``, the writer is a no-op and allocates no TF resources.
            upload_started: Whether ``aiplatform.start_upload_tb_log`` has
                been called and needs a paired ``end_upload_tb_log`` on
                ``close()``.
        """
        self._writer: Optional[Any] = (
            tf.summary.create_file_writer(log_dir) if log_dir else None
        )
        self._closed = False
        self._upload_started = upload_started

    @classmethod
    def from_env(cls, *, enabled: bool = True) -> "TensorBoardWriter":
        """Build a writer from Vertex AI's ``AIP_TENSORBOARD_LOG_DIR`` env var.

        When ``enabled`` is ``False``, returns a no-op writer without reading
        the environment. This is the path non-chief ranks take so they can
        share the same call sites as the chief.

        When ``enabled`` is ``True``:

        - ``AIP_TENSORBOARD_LOG_DIR`` must be set; otherwise this raises
          ``RuntimeError`` rather than silently no-op'ing. The env var is
          populated by Vertex AI from ``CustomJobSpec.baseOutputDirectory``
          (see the references in this module's header).
        - If ``GIGL_TENSORBOARD_RUN_NAME`` is set, events are written to
          ``<AIP_TENSORBOARD_LOG_DIR>/<run_name>/`` so the SDK uploader's
          ``LogdirLoader`` discovers the subdir as a distinct
          ``TensorboardRun`` (instead of merging into the SDK's hardcoded
          ``DEFAULT_RUN_NAME = "default"``). The launcher injects this env
          var when the user opts into ``tensorboard_experiment_name``.
        - If ``GIGL_TENSORBOARD_RESOURCE_NAME`` and
          ``GIGL_TENSORBOARD_EXPERIMENT_NAME`` are also set, this also starts
          a background ``aiplatform`` uploader that streams events from the
          PARENT log dir (so the run-name subdir surfaces as a run) to the
          named ``TensorboardExperiment`` under the configured
          ``Tensorboard`` instance. The uploader is shut down on
          :meth:`close`.

        Args:
            enabled: Whether this caller is responsible for writing events.
                Typically ``is_chief_process``.

        Returns:
            A ``TensorBoardWriter`` instance — real if enabled, no-op otherwise.

        Raises:
            RuntimeError: If ``enabled`` is True and ``AIP_TENSORBOARD_LOG_DIR``
                is not set in the environment.
            ValueError: If ``GIGL_TENSORBOARD_RESOURCE_NAME`` is set but does
                not match ``projects/.../locations/.../tensorboards/...``.
        """
        if not enabled:
            return cls(log_dir=None)
        parent_log_dir = os.environ.get(_VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY)
        if not parent_log_dir:
            raise RuntimeError(
                f"{_VERTEX_TENSORBOARD_LOG_DIR_ENV_KEY} is not set. "
                "TensorBoardWriter.from_env() requires the trainer to run as "
                "a Vertex AI CustomJob with baseOutputDirectory configured. "
                "See https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec#FIELDS.base_output_directory."
            )
        run_name = os.environ.get(_GIGL_TENSORBOARD_RUN_NAME_ENV_KEY)
        effective_log_dir = (
            os.path.join(parent_log_dir, run_name) if run_name else parent_log_dir
        )

        # Construct the file writer FIRST. If TF construction fails we don't
        # want a leaked uploader thread keeping the (non-daemon) process
        # alive. See codex review round 2, issue 6.
        instance = cls(log_dir=effective_log_dir, upload_started=False)
        try:
            if _maybe_start_uploader(parent_log_dir=parent_log_dir):
                instance._upload_started = True
        except BaseException:
            instance.close()
            raise
        return instance

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
        """Close the underlying TF writer and stop the uploader if running.

        Idempotent; safe to call multiple times and on no-op writers.
        """
        if self._closed:
            return
        if self._writer is not None:
            self._writer.close()
        if self._upload_started:
            aiplatform.end_upload_tb_log()
        self._closed = True

    def __enter__(self) -> "TensorBoardWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _maybe_start_uploader(*, parent_log_dir: str) -> bool:
    """Start the aiplatform TB uploader iff the GiGL env vars are present.

    Watches ``parent_log_dir`` (not the run-name subdir under it), so the
    SDK's ``LogdirLoader`` discovers each run via
    ``os.path.relpath(subdir, parent_log_dir)``. The Vertex AI TensorBoard
    data model (``Tensorboard`` → ``TensorboardExperiment`` → ``TensorboardRun``
    → ``TensorboardTimeSeries``) is documented at
    https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview.

    Returns ``True`` if the uploader was started (caller must arrange for
    ``aiplatform.end_upload_tb_log`` on shutdown), ``False`` otherwise.

    Args:
        parent_log_dir: The ``AIP_TENSORBOARD_LOG_DIR`` value — i.e. the
            directory whose children are run-name subdirectories.

    Raises:
        ValueError: If ``GIGL_TENSORBOARD_RESOURCE_NAME`` is set but does not
            match the expected resource-name format.
    """
    tb_resource_name = os.environ.get(_GIGL_TENSORBOARD_RESOURCE_NAME_ENV_KEY)
    experiment_name = os.environ.get(_GIGL_TENSORBOARD_EXPERIMENT_NAME_ENV_KEY)
    if not tb_resource_name or not experiment_name:
        return False

    match = _TENSORBOARD_RESOURCE_NAME_PATTERN.match(tb_resource_name)
    if not match:
        raise ValueError(
            f"{_GIGL_TENSORBOARD_RESOURCE_NAME_ENV_KEY}={tb_resource_name!r} "
            "does not match projects/.../locations/.../tensorboards/...; "
            "the GiGL launcher should set this to the same resource name "
            "configured on GiglResourceConfig."
        )

    aiplatform.init(
        project=match["project"],
        location=match["location"],
    )
    aiplatform.start_upload_tb_log(
        tensorboard_id=match["tensorboard_id"],
        tensorboard_experiment_name=experiment_name,
        logdir=parent_log_dir,
    )
    return True
