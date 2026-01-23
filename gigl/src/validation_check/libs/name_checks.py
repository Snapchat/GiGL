"""Checks for if assorted strings are valid."""
import re

from gigl.common.logger import Logger

logger = Logger()


def check_if_kfp_pipeline_job_name_valid(job_name: str) -> None:
    """
    Check if kfp pipeline job name valid. It is used to start spark cluster and must match pattern.
    The kfp pipeline job name is also used to generate AppliedTaskIdentifier for each component.
    """
    # TODO(mkolodner, kmonte): Check if our max length should be shorter.
    logger.info(f"Config validation check: if job_name: {job_name} is valid.")
    if not bool(re.match(r"^(?:[a-z](?:[_a-z0-9]{0,49}[a-z0-9])?)$", job_name)):
        raise ValueError(
            f"Invalid 'job_name'. Only lowercase letters, numbers, and underscores are allowed. "
            f"The name must start with lowercase letter or number and end with a lowercase letter or number. "
            "The name must be between 1 and 52 characters long. "
            f"'job_name' provided: {job_name} ."
        )
