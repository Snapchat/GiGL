"""Cancel Cloud Build builds by tag through the ``google-cloud-build`` library."""

from google.cloud.devtools import cloudbuild_v1

from gigl.common.logger import Logger

logger = Logger()


class CloudBuildService:
    """Lists and cancels Cloud Build builds in one project.

    Example:
        >>> service = CloudBuildService(project="p")
        >>> service.cancel_builds_with_tag("gigl-ci-run-123456")
        ['6c1d0e5a-0000-0000-0000-000000000000']
    """

    def __init__(self, project: str) -> None:
        """
        Args:
            project: GCP project that owns the builds.
        """
        self._project = project
        self._client = cloudbuild_v1.CloudBuildClient()

    def cancel_builds_with_tag(self, tag: str) -> list[str]:
        """Request a cancel for every queued or running build that carries ``tag``.

        Builds that already finished are left alone.

        A failed cancel does not stop the loop.
        Every matching build gets a cancel request before the errors are raised.

        Example:
            >>> CloudBuildService(project="p").cancel_builds_with_tag("gigl-ci-run-123456")
            ['6c1d0e5a-0000-0000-0000-000000000000']

        Args:
            tag: Cloud Build tag to match.

        Returns:
            Ids of the builds a cancel was requested for, in listing order.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the list call fails.
            ExceptionGroup: If one or more cancels failed.
                It holds one error for each failed build.
        """
        active_builds = self._client.list_builds(
            request=cloudbuild_v1.ListBuildsRequest(
                project_id=self._project,
                filter=f'tags="{tag}" AND (status=WORKING OR status=QUEUED)',
            )
        )
        cancelled: list[str] = []
        errors: list[Exception] = []
        for build in active_builds:
            logger.info(f"Cancelling Cloud Build {build.id} (tag {tag}).")
            try:
                self._client.cancel_build(
                    request=cloudbuild_v1.CancelBuildRequest(
                        project_id=self._project, id=build.id
                    )
                )
            except Exception as e:
                logger.warning(f"Could not cancel Cloud Build {build.id}: {e!r}")
                errors.append(e)
                continue
            cancelled.append(build.id)
        if errors:
            raise ExceptionGroup(
                f"{len(errors)} Cloud Build cancel(s) failed for tag {tag}", errors
            )
        return cancelled
