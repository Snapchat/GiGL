"""Submit a tiny Vertex AI CustomJob that exercises GiGL's TensorBoard wiring.

Goal: <2 min from "I changed launcher / writer code" to "I see whether TB
shows up." Bypasses ConfigPopulator and the full pipeline; uses the
production launcher path (``launch_single_pool_job``) so the same env-var
injection and submit logic runs as in real training.

Required CLI flags:
    --project              GCP project (e.g. ``snap-umap-dev``).
    --region               Vertex AI region (e.g. ``us-central1``).
    --service-account      Service account email used by the CustomJob.
    --staging-bucket       Regional GCS bucket Vertex stages artifacts under.
    --tensorboard          Full TensorBoard resource name
                           (``projects/.../locations/.../tensorboards/...``).
    --container-uri        Container image to use. REQUIRED — must contain the
                           branch under test. Pointing at a released image
                           would test stale code; codex review explicitly
                           flagged defaulting to ``DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU``
                           as wrong (round-2 issue 2).

Optional:
    --experiment-name      Vertex AI ``TensorboardExperiment`` name. Leave
                           unset to test the per-job auto-upload path (R3).
                           Set to opt into multi-job comparison (R1+R2).
    --job-name             CustomJob display name. Defaults to a timestamped
                           ``gigl-tb-smoke-...``.
    --dry-run              Print the constructed VertexAiJobConfig and exit
                           without submitting.

Verification:
    On real (non-dry-run) submission, after the CustomJob completes the
    script polls the TensorBoard API surfaces and asserts:

    - The per-job ``TensorboardExperiment`` (named after the CustomJob's
      numeric ID) exists, has a run, and that run has at least one
      ``TensorboardTimeSeries`` for the ``smoke/value`` tag.
    - When ``--experiment-name`` was passed, the user-named experiment also
      exists with a run named after the launch-unique ``GIGL_TENSORBOARD_RUN_NAME``,
      and that run has at least one time series.

    Both TB UI URLs are printed for manual inspection.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import time
from typing import Optional

from google.cloud import aiplatform

from gigl.common import GcsUri, Uri
from gigl.common.logger import Logger
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.vertex_ai_launcher import launch_single_pool_job
from snapchat.research.gbml import gigl_resource_config_pb2

logger = Logger()

_TENSORBOARD_RESOURCE_NAME_PATTERN = re.compile(
    r"^projects/(?P<project>[^/]+)"
    r"/locations/(?P<location>[^/]+)"
    r"/tensorboards/(?P<tensorboard_id>[^/]+)$"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument(
        "--staging-bucket",
        required=True,
        help="Regional GCS bucket (e.g. gs://gigl-dev-temp-assets).",
    )
    parser.add_argument(
        "--tensorboard",
        required=True,
        help="Full TensorBoard resource name.",
    )
    parser.add_argument(
        "--container-uri",
        required=True,
        help=(
            "Container image with the branch code. Required; pointing at a "
            "released image would test stale code."
        ),
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_resource_config(
    *,
    project: str,
    region: str,
    service_account: str,
    staging_bucket: str,
    tensorboard_resource_name: str,
) -> gigl_resource_config_pb2.GiglResourceConfig:
    """Minimal GiglResourceConfig wired for a 1-replica CPU CustomJob."""
    common = gigl_resource_config_pb2.SharedResourceConfig.CommonComputeConfig(
        project=project,
        region=region,
        # The launcher reads ``temp_regional_assets_bucket`` as the Vertex
        # AI staging bucket (see VertexAIService construction in
        # gigl/src/common/vertex_ai_launcher.py).
        temp_regional_assets_bucket=staging_bucket,
        temp_assets_bucket=staging_bucket,
        perm_assets_bucket=staging_bucket,
        temp_assets_bq_dataset_name="not_used_by_smoke",
        embedding_bq_dataset_name="not_used_by_smoke",
        gcp_service_account_email=service_account,
        dataflow_runner="DataflowRunner",
    )
    shared = gigl_resource_config_pb2.SharedResourceConfig(
        common_compute_config=common,
        resource_labels={"cost_resource_group": "gigl_dev_smoke"},
    )
    trainer = gigl_resource_config_pb2.VertexAiResourceConfig(
        # n1-standard-2 is rejected by Vertex AI training in this project;
        # n1-standard-16 is the smallest spec we've confirmed accepted.
        machine_type="n1-standard-16",
        gpu_type="ACCELERATOR_TYPE_UNSPECIFIED",
        gpu_limit=0,
        num_replicas=1,
        timeout=600,
        tensorboard_resource_name=tensorboard_resource_name,
    )
    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=shared,
        trainer_resource_config=gigl_resource_config_pb2.TrainerResourceConfig(
            vertex_ai_trainer_config=trainer,
        ),
    )


def _verify_per_job_experiment(
    *,
    tensorboard_resource_name: str,
    job_id: str,
) -> None:
    """The auto-uploader names its TensorboardExperiment after the job's numeric ID."""
    experiment_resource_name = (
        f"{tensorboard_resource_name}/experiments/{job_id}"
    )
    runs = aiplatform.TensorboardRun.list(
        tensorboard_experiment_name=experiment_resource_name,
    )
    if not runs:
        raise RuntimeError(
            f"Per-job TensorboardExperiment {experiment_resource_name} has no "
            "TensorboardRuns; the auto-uploader did not ingest any events."
        )
    for run in runs:
        time_series = aiplatform.TensorboardTimeSeries.list(
            tensorboard_run_name=run.resource_name,
        )
        if not time_series:
            raise RuntimeError(
                f"Run {run.resource_name} has no TensorboardTimeSeries; "
                "events did not reach the API."
            )
    logger.info(
        f"Per-job experiment OK: {len(runs)} run(s) under {experiment_resource_name}"
    )


def _verify_named_experiment(
    *,
    tensorboard_resource_name: str,
    experiment_name: str,
) -> None:
    """The chief-rank uploader names its TensorboardExperiment after the user flag."""
    experiment_resource_name = (
        f"{tensorboard_resource_name}/experiments/{experiment_name}"
    )
    runs = aiplatform.TensorboardRun.list(
        tensorboard_experiment_name=experiment_resource_name,
    )
    if not runs:
        raise RuntimeError(
            f"Named TensorboardExperiment {experiment_resource_name} has no "
            "TensorboardRuns; the chief-rank uploader did not ingest events."
        )
    for run in runs:
        time_series = aiplatform.TensorboardTimeSeries.list(
            tensorboard_run_name=run.resource_name,
        )
        if not time_series:
            raise RuntimeError(
                f"Run {run.resource_name} has no TensorboardTimeSeries; "
                "events did not reach the API."
            )
    run_names = sorted(r.display_name for r in runs)
    logger.info(
        f"Named experiment OK: {len(runs)} run(s) under {experiment_resource_name}: "
        f"{run_names}"
    )


def _print_tb_urls(
    *,
    region: str,
    project: str,
    tensorboard_id: str,
    job_id: str,
    experiment_name: Optional[str],
) -> None:
    base = f"https://{region}.tensorboard.googleusercontent.com/experiment"
    qualifier = (
        f"projects+{project}+locations+{region}+tensorboards+{tensorboard_id}"
    )
    per_job = f"{base}/{qualifier}+experiments+{job_id}"
    logger.info(f"Per-job TB URL: {per_job}")
    if experiment_name:
        named = f"{base}/{qualifier}+experiments+{experiment_name}"
        logger.info(f"Named TB URL:   {named}")


def main() -> int:
    args = _parse_args()

    tb_match = _TENSORBOARD_RESOURCE_NAME_PATTERN.match(args.tensorboard)
    if not tb_match:
        logger.error(
            f"--tensorboard must be projects/.../locations/.../tensorboards/...; "
            f"got {args.tensorboard!r}."
        )
        return 2

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"gigl-tb-smoke-{timestamp}"
    tensorboard_logs_uri = GcsUri(
        f"{args.staging_bucket.rstrip('/')}/tb-smoke/{timestamp}/logs/"
    )

    resource_config = _build_resource_config(
        project=args.project,
        region=args.region,
        service_account=args.service_account,
        staging_bucket=args.staging_bucket,
        tensorboard_resource_name=args.tensorboard,
    )
    resource_wrapper = GiglResourceConfigWrapper(resource_config=resource_config)

    if args.dry_run:
        logger.info(
            "Dry run — would submit a CustomJob with:\n"
            f"  job_name              = {job_name}\n"
            f"  container_uri         = {args.container_uri}\n"
            f"  tensorboard_logs_uri  = {tensorboard_logs_uri}\n"
            f"  tensorboard_resource  = {args.tensorboard}\n"
            f"  experiment_name       = {args.experiment_name!r}\n"
        )
        return 0

    aiplatform.init(project=args.project, location=args.region)
    custom_job = launch_single_pool_job(
        vertex_ai_resource_config=resource_config.trainer_resource_config.vertex_ai_trainer_config,
        job_name=job_name,
        task_config_uri=Uri("gs://unused/by/smoke.yaml"),
        resource_config_uri=Uri("gs://unused/by/smoke.yaml"),
        process_command="python -m gigl.utils.dev.tb_smoke_main",
        process_runtime_args={},
        resource_config_wrapper=resource_wrapper,
        cpu_docker_uri=args.container_uri,
        cuda_docker_uri=args.container_uri,
        component=GiGLComponents.Trainer,
        vertex_ai_region=args.region,
        tensorboard_logs_uri=tensorboard_logs_uri,
        tensorboard_experiment_name=args.experiment_name,
    )
    job_id = custom_job.name  # trailing segment of resource_name == numeric job ID
    logger.info(f"Submitted CustomJob: {custom_job.resource_name}")
    logger.info(
        f"Job UI: https://console.cloud.google.com/ai/platform/locations/"
        f"{args.region}/training/{job_id}?project={args.project}"
    )

    # CustomJob.submit blocks until completion in this code path (see
    # VertexAIService._submit_job: job.wait_for_completion). Give the
    # uploader thread a brief grace period in case the trainer's sleep
    # was tight.
    time.sleep(5)

    _verify_per_job_experiment(
        tensorboard_resource_name=args.tensorboard,
        job_id=job_id,
    )
    if args.experiment_name:
        _verify_named_experiment(
            tensorboard_resource_name=args.tensorboard,
            experiment_name=args.experiment_name,
        )

    _print_tb_urls(
        region=args.region,
        project=args.project,
        tensorboard_id=tb_match["tensorboard_id"],
        job_id=job_id,
        experiment_name=args.experiment_name,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
