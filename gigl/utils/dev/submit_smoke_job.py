"""Submit a tiny Vertex AI CustomJob that exercises GiGL's TensorBoard wiring.

Goal: <2 min from "I changed launcher / writer code" to "I see whether TB
shows up." Bypasses ConfigPopulator and the full pipeline; uses the
production launcher path (``launch_single_pool_job``) so the same submit
logic runs as in real training.

Required CLI flags:
    --project              GCP project (e.g. ``external-snap-ci-github-gigl``).
    --region               Vertex AI region (e.g. ``us-central1``).
    --service-account      Service account email used by the CustomJob.
    --staging-bucket       Regional GCS bucket Vertex stages artifacts under.
    --tensorboard          Full TensorBoard resource name
                           (``projects/.../locations/.../tensorboards/...``).
    --experiment-name      Vertex AI ``TensorboardExperiment`` name. The
                           tb_smoke_main entry point will pass this and the
                           --tensorboard value to ``TensorBoardWriter.create``.
    --container-uri        Container image to use. REQUIRED — must contain the
                           branch under test.

Optional:
    --job-name             CustomJob display name. Defaults to a timestamped
                           ``gigl-tb-smoke-...``.
    --dry-run              Print the constructed submission parameters and
                           exit without submitting.

Verification:
    After the CustomJob completes the script polls the TensorBoard API
    surface and asserts the user-named ``TensorboardExperiment`` exists
    with at least one ``TensorboardRun`` containing time series data.

    The TB UI URL is printed for manual inspection.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import time

from google.cloud import aiplatform

from gigl.common import Uri
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
        help="Regional GCS bucket (e.g. gs://gigl-cicd-temp).",
    )
    parser.add_argument(
        "--tensorboard",
        required=True,
        help="Full TensorBoard resource name.",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help=(
            "TensorboardExperiment name. Passed to tb_smoke_main, which "
            "creates the run under this experiment."
        ),
    )
    parser.add_argument(
        "--container-uri",
        required=True,
        help=(
            "Container image with the branch code. Required; pointing at a "
            "released image would test stale code."
        ),
    )
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_resource_config(
    *,
    project: str,
    region: str,
    service_account: str,
    staging_bucket: str,
) -> gigl_resource_config_pb2.GiglResourceConfig:
    """Minimal GiglResourceConfig wired for a 1-replica CPU CustomJob."""
    common = gigl_resource_config_pb2.SharedResourceConfig.CommonComputeConfig(
        project=project,
        region=region,
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
        # n1-standard-2 is rejected by Vertex AI; n1-standard-16 is the
        # smallest spec we've confirmed accepted in dev.
        machine_type="n1-standard-16",
        gpu_type="ACCELERATOR_TYPE_UNSPECIFIED",
        gpu_limit=0,
        num_replicas=1,
        timeout=600,
    )
    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=shared,
        trainer_resource_config=gigl_resource_config_pb2.TrainerResourceConfig(
            vertex_ai_trainer_config=trainer,
        ),
    )


def _verify_named_experiment(
    *,
    tensorboard_resource_name: str,
    experiment_name: str,
) -> None:
    """Confirm the chief-rank writer ingested events into the named experiment."""
    experiment_resource_name = (
        f"{tensorboard_resource_name}/experiments/{experiment_name}"
    )
    runs = aiplatform.TensorboardRun.list(
        tensorboard_experiment_name=experiment_resource_name,
    )
    if not runs:
        raise RuntimeError(
            f"Named TensorboardExperiment {experiment_resource_name} has no "
            "TensorboardRuns; the writer did not ingest events."
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


def _print_tb_url(
    *,
    region: str,
    project: str,
    tensorboard_id: str,
    experiment_name: str,
) -> None:
    base = f"https://{region}.tensorboard.googleusercontent.com/experiment"
    qualifier = f"projects+{project}+locations+{region}+tensorboards+{tensorboard_id}"
    named = f"{base}/{qualifier}+experiments+{experiment_name}"
    logger.info(f"Named TB URL: {named}")


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

    resource_config = _build_resource_config(
        project=args.project,
        region=args.region,
        service_account=args.service_account,
        staging_bucket=args.staging_bucket,
    )
    resource_wrapper = GiglResourceConfigWrapper(resource_config=resource_config)

    process_runtime_args = {
        "tensorboard_resource_name": args.tensorboard,
        "tensorboard_experiment_name": args.experiment_name,
    }

    if args.dry_run:
        logger.info(
            "Dry run — would submit a CustomJob with:\n"
            f"  job_name              = {job_name}\n"
            f"  container_uri         = {args.container_uri}\n"
            f"  tensorboard_resource  = {args.tensorboard}\n"
            f"  experiment_name       = {args.experiment_name!r}\n"
            f"  process_runtime_args  = {process_runtime_args}\n"
        )
        return 0

    aiplatform.init(project=args.project, location=args.region)
    launch_single_pool_job(
        vertex_ai_resource_config=resource_config.trainer_resource_config.vertex_ai_trainer_config,
        job_name=job_name,
        task_config_uri=Uri("gs://unused/by/smoke.yaml"),
        resource_config_uri=Uri("gs://unused/by/smoke.yaml"),
        process_command="python -m gigl.utils.dev.tb_smoke_main",
        process_runtime_args=process_runtime_args,
        resource_config_wrapper=resource_wrapper,
        cpu_docker_uri=args.container_uri,
        cuda_docker_uri=args.container_uri,
        component=GiGLComponents.Trainer,
        vertex_ai_region=args.region,
    )
    logger.info(f"Submitted CustomJob: {job_name}")

    # CustomJob.submit blocks until completion inside launch_single_pool_job
    # (see VertexAIService._submit_job: job.wait_for_completion). Give the
    # backing TensorboardExperiment a short grace period for any final RPCs.
    time.sleep(5)

    _verify_named_experiment(
        tensorboard_resource_name=args.tensorboard,
        experiment_name=args.experiment_name,
    )
    _print_tb_url(
        region=args.region,
        project=args.project,
        tensorboard_id=tb_match["tensorboard_id"],
        experiment_name=args.experiment_name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
