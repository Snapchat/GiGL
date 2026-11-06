"""Cancel the GCP resources a GitHub Actions run started.

Cloud Builds are found by the tag ``gigl-ci-run-<run id>`` that
``.github/actions/run-cloud-run-command-on-active-checkout`` sets, and cancelled through
``CloudBuildService``.
Vertex AI pipeline runs are found by the label ``github_run_id=<run id>`` that
``tests/e2e_tests/e2e_test.py`` sets when ``GITHUB_RUN_ID`` is in the environment.
Both identifiers are defined in ``gigl/src/common/constants/ci.py``.
Each cancelled pipeline run then runs its exit task, which cancels that run's child jobs.

The build submits pipeline runs one at a time, so a run can appear after the first
list call. The script waits for the cancelled builds to stop, then lists and cancels
pipeline runs again until a pass finds nothing new.

Runs on a GitHub runner that has uv but not the GiGL environment, so the
action invokes it with ``uv run --frozen --only-group gh-actions`` (the dependency group
in pyproject.toml that lists what this script imports) and ``PYTHONPATH`` set to the
repo copy the action came from, not the PR checkout. Both resource kinds are attempted
even if the other one fails.

Usage:
    python .github/scripts/cancel_ci_gcp_resources.py \
        --project=my-project --region=us-central1 --github_run_id=123456
"""

import argparse
import sys
import time
from typing import Final

from gigl.common.logger import Logger
from gigl.common.services.cloud_build import CloudBuildService
from gigl.common.services.vertex_ai import VertexAIService
from gigl.src.common.constants.ci import (
    GITHUB_RUN_ID_LABEL_KEY,
    cloud_build_tag_for_github_run,
)

logger = Logger()

# How long to wait for a cancelled build to stop before the pipeline passes start anyway.
_BUILD_STOP_TIMEOUT_S: Final[float] = 5 * 60
# Seconds between pipeline discovery passes, and the most passes to make.
# A pass after the first that cancels nothing ends the loop.
_DISCOVERY_INTERVAL_S: Final[float] = 30
_MAX_DISCOVERY_PASSES: Final[int] = 5


def cancel_pipeline_runs_until_quiet(
    project: str, region: str, github_run_id: str
) -> tuple[list[str], list[Exception]]:
    """List and cancel the labelled pipeline runs until a pass finds nothing new.

    The loop ends at the first pass after pass one that cancels nothing, or after
    ``_MAX_DISCOVERY_PASSES`` passes.
    The extra passes catch a run whose submission was in flight during an earlier pass.

    Example:
        >>> cancel_pipeline_runs_until_quiet("my-project", "us-central1", "123456")
        (['projects/my-project/locations/us-central1/pipelineJobs/run-1'], [])

    Args:
        project: Google Cloud project of the runs.
        region: Vertex AI region of the runs.
        github_run_id: GitHub Actions run id in the run label.

    Returns:
        The resource names of the cancelled runs, and the errors from failed passes.
    """
    cancelled: list[str] = []
    errors: list[Exception] = []
    for pass_number in range(_MAX_DISCOVERY_PASSES):
        if pass_number > 0:
            time.sleep(_DISCOVERY_INTERVAL_S)
        try:
            cancelled_in_pass = VertexAIService.cancel_pipeline_runs_with_label(
                project=project,
                location=region,
                label_key=GITHUB_RUN_ID_LABEL_KEY,
                label_value=github_run_id,
            )
        except Exception as e:
            logger.warning(f"Pipeline discovery pass {pass_number} failed: {e!r}")
            errors.append(e)
            continue
        cancelled.extend(cancelled_in_pass)
        if pass_number > 0 and not cancelled_in_pass:
            break
    return cancelled, errors


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
        cloud_build_service = CloudBuildService(args.project)
        cancelled_builds = cloud_build_service.cancel_builds_with_tag(
            cloud_build_tag_for_github_run(args.github_run_id)
        )
        logger.info(f"Cancelled Cloud Builds: {cancelled_builds or 'none'}")
        # A build that is still running can still submit pipeline runs.
        cloud_build_service.wait_for_builds_to_stop(
            cancelled_builds, timeout_s=_BUILD_STOP_TIMEOUT_S
        )
    except Exception as e:
        failures.append(f"Cloud Builds: {e!r}")

    cancelled_runs, pipeline_errors = cancel_pipeline_runs_until_quiet(
        args.project, args.region, args.github_run_id
    )
    logger.info(f"Cancelled pipeline runs: {cancelled_runs or 'none'}")
    for error in pipeline_errors:
        failures.append(f"pipeline runs: {error!r}")

    if failures:
        logger.error(f"Cancel of run {args.github_run_id} failed for: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
