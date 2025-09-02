"""OmegaConf custom resolvers for GiGL.

This module contains custom OmegaConf resolvers that can be used in YAML configuration
files to provide dynamic values during configuration loading.
"""

import subprocess
from datetime import datetime, timedelta

from omegaconf import OmegaConf

from gigl.common.logger import Logger

logger = Logger()


_SUPPORTED_UNITS = ("weeks", "days", "seconds", "minutes", "hours")


def now_resolver(*args: str) -> str:
    """Resolver that creates a string representing the current time (with optional offset) using strftime.

    This resolver supports both time formatting and time offsets with explicit named parameters
    that are compatible with OmegaConf's resolver parameter limitations.

    Args:
        *args: Variable arguments where:
            - First argument (optional): datetime.datetime compatible strftime format string.
                If not provided, defaults to "%Y%m%d_%H%M%S"
            - Subsequent arguments: datetime.timedelta compatible time offset specifications in format "unit±value" where:
                * unit can be: days, seconds, minutes, hours, weeks
                * ± can be + or - (+ can be omitted for positive values)
                * Examples: "days+1", "hours-2", "minutes30", "seconds-15"

    Returns:
        Current time (with optional offset) formatted as a string.

    Example:
        In YAML config:
        ```yaml
        name: "exp_${now:%Y%m%d_%H%M%S}"
        start_time: "${now:%Y-%m-%d %H:%M:%S}"
        log_file: "logs/run_${now:%H-%M-%S}.log"
        timestamp: "${now:}"  # Uses default format
        short_date: "${now:%m-%d}"

        tomorrow: "${now:%Y-%m-%d, days+1}"
        yesterday: "${now:%Y-%m-%d, days-1}"
        tomorrow_plus_5_hours_30_min_15_sec: "${now:%Y-%m-%d %H:%M:%S,hours+5,days+1,minutes+30,seconds+15}"
        next_week: "${now:%Y-%m-%d, weeks+1}"
        multiple_args: "${now:%Y%m%d, days-15}:${now:%Y%m%d, days-1}"

        This would resolve to something like:
        ```yaml
        name: "exp_20231215_143022"
        start_time: "2023-12-15 14:30:22"
        log_file: "logs/run_14-30-22.log"
        timestamp: "20231215_143022"
        short_date: "12-15"

        tomorrow: "2023-12-16"
        yesterday: "2023-12-14"
        tomorrow_plus_5_hours_30_min_15_sec: "2023-12-16 20:00:37"
        next_week: "2023-12-22"
        multiple_args: "20231201:20231214"
        ```
    """
    # Default values
    format_str = "%Y%m%d_%H%M%S"
    weeks = days = hours = minutes = seconds = 0

    for i, arg in enumerate(
        args
    ):  # i.e. args = ["%Y%m%d_%H%M%S", "days+1", "hours-2", "minutes30", "seconds-15"]
        if i == 0 and not any(unit in arg for unit in _SUPPORTED_UNITS):
            # First argument is format string if it doesn't contain time units
            format_str = arg

        else:
            # Parse time offset specifications like "days+1", "hours-2", etc.
            arg = arg.strip()
            error_could_not_parse_msg = (
                f"Could not parse time offset '{arg}', it should be of form days+1, hours-2, etc. "
                f"Provided: {args}. See docs for more details."
            )

            # Try to parse each known unit
            for unit in _SUPPORTED_UNITS:
                if arg.startswith(unit):
                    value_str = arg[len(unit) :]
                    if not value_str or not (
                        value_str.startswith("+") or value_str.startswith("-")
                    ):
                        raise ValueError(error_could_not_parse_msg)

                    try:
                        value = int(value_str)
                    except ValueError:
                        # Cleaner error message
                        raise ValueError(error_could_not_parse_msg)

                    if unit == "weeks":
                        weeks = value
                    if unit == "days":
                        days = value
                    elif unit == "hours":
                        hours = value
                    elif unit == "minutes":
                        minutes = value
                    elif unit == "seconds":
                        seconds = value
                    break  # Exit the unit loop once we find a match

            else:  # If loop completes without breaking, then no unit found in this argument
                raise ValueError(error_could_not_parse_msg)

    # Calculate the target time
    target_time = datetime.now() + timedelta(
        weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds
    )
    return target_time.strftime(format_str)


def git_hash_resolver() -> str:
    """Resolver that returns the current git hash.

    This resolver returns the current git hash if one is available.
    Takes no arguments and returns the git hash as a string.

    Returns:
        Current git hash as a string, or empty string if not available.

    Example:
        In YAML config:
        ```yaml
        model_version: "model_${git_hash}"
        experiment_id: "exp_${git_hash}_${now:%Y%m%d}"
        ```
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        logger.warning(
            "Could not retrieve git hash - git command failed or not in a git repository"
        )
        return ""


def register_resolvers() -> None:
    """Register all custom OmegaConf resolvers.

    This function should be called at application startup to register
    all custom resolvers with OmegaConf.
    """
    logger.info("Registering OmegaConf resolvers")
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver("now", now_resolver)
    else:
        logger.info(
            "OmegaConf resolver 'now' already registered, skipping registration"
        )

    if not OmegaConf.has_resolver("git_hash"):
        OmegaConf.register_new_resolver("git_hash", git_hash_resolver)
    else:
        logger.info(
            "OmegaConf resolver 'git_hash' already registered, skipping registration"
        )
