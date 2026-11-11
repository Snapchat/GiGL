"""Identifiers that tie GCP resources back to the GitHub Actions run that created them."""

from typing import Final

# Label key on Vertex AI pipeline runs. Set by tests/e2e_tests/e2e_test.py when GITHUB_RUN_ID is
# in the environment; read by .github/scripts/cancel_ci_gcp_resources.py.
GITHUB_RUN_ID_LABEL_KEY: Final[str] = "github_run_id"

# Prefix of the Cloud Build tag. .github/cloud_builder/run_command_on_active_checkout.yaml must spell the
# same tag as `gigl-ci-run-${_GITHUB_RUN_ID}` because YAML cannot import this module; a unit test checks
# that the two agree.
_CLOUD_BUILD_RUN_TAG_PREFIX: Final[str] = "gigl-ci-run-"


def cloud_build_tag_for_github_run(github_run_id: str) -> str:
    """Return the Cloud Build tag that marks a build as started by one GitHub Actions run.

    Cloud Build runs started for a CI run carry this tag, so the cancel script can find them.

    Example:
        >>> cloud_build_tag_for_github_run("123456")
        'gigl-ci-run-123456'

    Args:
        github_run_id: The GitHub Actions run id.

    Returns:
        The tag string.
    """
    return f"{_CLOUD_BUILD_RUN_TAG_PREFIX}{github_run_id}"
