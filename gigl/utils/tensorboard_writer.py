"""TensorBoard writer for GiGL training and inference entrypoints.

Writes scalars to Vertex AI's TensorboardService via the synchronous
``aiplatform.log_time_series_metrics`` API. The writer attaches to a Vertex
AI ``Experiment`` + ``ExperimentRun`` whose backing ``Tensorboard`` resource
the caller supplies explicitly.

Vertex AI TensorBoard data model:
    Tensorboard -> TensorboardExperiment -> TensorboardRun -> TensorboardTimeSeries
    https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview

Configuration is plumbed through the trainer/inferencer's argparse interface
(typically populated from ``GbmlConfig.trainerConfig.trainerArgs`` or
``inferencerConfig.inferencerArgs``), not through env vars or proto fields on
``GiglResourceConfig``. Construct the writer with
:meth:`TensorBoardWriter.create` and let chief / non-chief ranks share the
same call sites:

    >>> is_chief_process = args.machine_rank == 0 and local_rank == 0
    >>> with TensorBoardWriter.create(
    ...     resource_name=args.tensorboard_resource_name,
    ...     experiment_name=args.tensorboard_experiment_name,
    ...     experiment_run_name=args.job_name,
    ...     enabled=is_chief_process,
    ... ) as tb:
    ...     tb.log({"Loss/train": loss}, step=batch_idx)
"""

import re
from typing import Final, Optional

from google.cloud import aiplatform

from gigl.common.logger import Logger

logger = Logger()

# Vertex AI Tensorboard resource name format.
_TENSORBOARD_RESOURCE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^projects/(?P<project>[^/]+)"
    r"/locations/(?P<location>[^/]+)"
    r"/tensorboards/(?P<tensorboard_id>[^/]+)$"
)


class TensorBoardWriter:
    """Writes scalar metrics to a Vertex AI ``ExperimentRun``.

    No-ops when disabled, so callers never see ``Optional[TensorBoardWriter]``
    plumbing across chief / non-chief ranks.

    Each :meth:`log` call issues a synchronous ``WriteTensorboardRunData`` RPC
    via ``aiplatform.log_time_series_metrics``. On first sight of any new
    metric key the SDK also issues a ``CreateTensorboardTimeSeries`` RPC.
    Failures propagate to the caller rather than being absorbed in a
    background uploader thread.
    """

    def __init__(self, *, active: bool) -> None:
        """Initialize the writer.

        Callers should use :meth:`create` rather than constructing directly.

        Args:
            active: When ``False``, the writer is a no-op (no SDK calls).
                When ``True``, :meth:`create` has already called
                ``aiplatform.init`` and ``aiplatform.start_run`` on this
                process.
        """
        self._active = active
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        resource_name: Optional[str],
        experiment_name: Optional[str],
        experiment_run_name: str,
        enabled: bool,
    ) -> "TensorBoardWriter":
        """Construct a writer from explicit configuration.

        When ``enabled`` is ``False`` (non-chief ranks), returns a no-op
        writer without touching the aiplatform SDK regardless of the other
        arguments.

        When ``enabled`` is ``True``, all three of ``resource_name``,
        ``experiment_name``, and ``experiment_run_name`` must be non-empty.
        Missing any of them raises ``RuntimeError`` so config gaps surface
        immediately. ``resource_name`` must additionally match
        ``projects/.../locations/.../tensorboards/...``.

        Side effects when ``enabled`` is ``True`` and all args are valid:

        - Calls ``aiplatform.init(project=..., location=..., experiment=...,
          experiment_tensorboard=...)`` with project + location parsed from
          ``resource_name``.
        - Calls ``aiplatform.start_run(experiment_run_name, resume=False)``.
          Callers are expected to pass a launch-unique run name (typically
          the trainer's ``job_name``).
        - Logs the human-readable TensorBoard UI URL so engineers can find
          the cross-job experiment page from trainer stdout.

        Args:
            resource_name: Fully-qualified Vertex AI ``Tensorboard`` resource
                name (``projects/.../locations/.../tensorboards/<id>``).
            experiment_name: Vertex AI ``TensorboardExperiment`` ID under
                ``resource_name``. Multiple jobs that share this value
                surface as comparable runs on a single TensorBoard page.
            experiment_run_name: Vertex AI ``TensorboardRun`` ID under
                ``experiment_name``. Must be unique per launch (use
                ``args.job_name``).
            enabled: Whether this caller is responsible for writing events
                (typically ``is_chief_process``).

        Returns:
            A ``TensorBoardWriter`` — real if ``enabled``, no-op otherwise.

        Raises:
            RuntimeError: ``enabled`` is True and any required argument is
                missing.
            ValueError: ``resource_name`` doesn't match the Vertex AI
                Tensorboard resource-name format.
        """
        if not enabled:
            return cls(active=False)

        missing = [
            name
            for name, value in (
                ("resource_name", resource_name),
                ("experiment_name", experiment_name),
                ("experiment_run_name", experiment_run_name),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "TensorBoardWriter.create(enabled=True) requires "
                f"{', '.join(missing)} to be set. The trainer/inferencer "
                "entrypoint plumbs these through argparse from "
                "GbmlConfig.trainerArgs / inferencerArgs."
            )

        assert resource_name is not None  # narrowed by the missing check above
        assert experiment_name is not None
        assert experiment_run_name is not None
        match = _TENSORBOARD_RESOURCE_NAME_PATTERN.match(resource_name)
        if not match:
            raise ValueError(
                f"resource_name {resource_name!r} does not match "
                "projects/.../locations/.../tensorboards/...; pass the "
                "Tensorboard resource name from GCP, not the display name."
            )

        aiplatform.init(
            project=match["project"],
            location=match["location"],
            experiment=experiment_name,
            experiment_tensorboard=resource_name,
        )
        aiplatform.start_run(experiment_run_name, resume=False)
        experiment_url = (
            f"https://{match['location']}.tensorboard.googleusercontent.com/experiment/"
            f"projects+{match['project']}"
            f"+locations+{match['location']}"
            f"+tensorboards+{match['tensorboard_id']}"
            f"+experiments+{experiment_name}"
        )
        logger.info(
            f"View TensorBoard (cross-job comparison, experiment={experiment_name!r}): "
            f"{experiment_url}"
        )
        return cls(active=True)

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Write each metric scalar at ``step`` via Vertex AI Experiments.

        No-ops when the writer is inactive or already closed. All entries
        in ``metrics`` are written under the hood in a single
        ``WriteTensorboardRunData`` RPC.

        Args:
            metrics: Mapping of TensorBoard tag to scalar value. All entries
                are written at the same step.
            step: TensorBoard step for the data points.
        """
        if not self._active or self._closed:
            return
        aiplatform.log_time_series_metrics(metrics, step=step)

    def close(self) -> None:
        """End the backing ``ExperimentRun``.

        Idempotent; safe to call multiple times and on no-op writers.
        """
        if self._closed:
            return
        if self._active:
            aiplatform.end_run()
        self._closed = True

    def __enter__(self) -> "TensorBoardWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
