"""Cancel the GCP resources a GitHub Actions run started.

Cloud Builds are found by the tag ``gigl-ci-run-<run id>`` that
``.github/actions/run-cloud-run-command-on-active-checkout`` sets.
Vertex AI pipeline runs are found by the label ``github_run_id=<run id>`` that
``tests/e2e_tests/e2e_test.py`` sets when ``GITHUB_RUN_ID`` is in the environment.
Each cancelled pipeline run then runs its exit task, which cancels that run's child jobs.

Runs on a GitHub runner that has gcloud and uv but not the GiGL environment, so the
action invokes it with ``uv run --frozen --only-group gcp`` (the GCP client libraries
from pyproject.toml) and ``PYTHONPATH`` set to the checkout. Both resource kinds are
attempted even if the other one fails.

Usage:
    python .github/scripts/cancel_ci_gcp_resources.py \
        --project=my-project --region=us-central1 --github_run_id=123456
"""

import argparse
import subprocess
import sys

from gigl.common.logger import Logger
from gigl.common.services.vertex_ai import VertexAIService

logger = Logger()


def cancel_cloud_builds(project: str, github_run_id: str) -> list[str]:
    """Cancel the queued or working Cloud Builds tagged with this run.

    Uses the gcloud CLI, which the runner already has authenticated.

    Args:
        project: Google Cloud project the builds were submitted to.
        github_run_id: GitHub Actions run id in the build tag.

    Returns:
        The ids of the builds that were cancelled.
    """
    tag = f"gigl-ci-run-{github_run_id}"
    listed = subprocess.run(
        [
            "gcloud",
            "builds",
            "list",
            f"--project={project}",
            f'--filter=tags="{tag}" AND (status=WORKING OR status=QUEUED)',
            "--format=value(id)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    build_ids = listed.stdout.split()
    for build_id in build_ids:
        logger.info(f"Cancelling Cloud Build {build_id} (tag {tag}).")
        subprocess.run(
            ["gcloud", "builds", "cancel", build_id, f"--project={project}"],
            check=True,
        )
    return build_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Google Cloud project.")
    parser.add_argument(
        "--region", required=True, help="Vertex AI region of the pipeline runs."
    )
    parser.add_argument(
        "--github_run_id", required=True, help="GitHub Actions run id to clean up."
    )
    args = parser.parse_args()

    failures: list[str] = []
    try:
        cancelled_builds = cancel_cloud_builds(args.project, args.github_run_id)
        logger.info(f"Cancelled Cloud Builds: {cancelled_builds or 'none'}")
    except Exception as e:
        failures.append(f"Cloud Builds: {e!r}")

    try:
        cancelled_runs = VertexAIService.cancel_pipeline_runs_with_label(
            project=args.project,
            location=args.region,
            label_key="github_run_id",
            label_value=args.github_run_id,
        )
        logger.info(f"Cancelled pipeline runs: {cancelled_runs or 'none'}")
    except Exception as e:
        failures.append(f"pipeline runs: {e!r}")

    if failures:
        logger.error(f"Cancel of run {args.github_run_id} failed for: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
