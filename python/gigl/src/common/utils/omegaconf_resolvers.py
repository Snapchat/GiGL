"""OmegaConf custom resolvers for GiGL.

This module contains custom OmegaConf resolvers that can be used in YAML configuration
files to provide dynamic values during configuration loading.
"""

from datetime import datetime, timedelta

from omegaconf import OmegaConf

from gigl.common.logger import Logger

logger = Logger()


def now_resolver(*args) -> str:
    """Resolver that creates a string representing the current time (with optional offset) using strftime.

    This resolver supports both time formatting and time offsets with explicit named parameters
    that are compatible with OmegaConf's resolver parameter limitations.

    Args:
        *args: Variable arguments where:
               - First argument (optional): strftime format string. If not provided, defaults to "%Y%m%d_%H%M%S"
               - Subsequent arguments: Time offset specifications in format "unit±value" where:
                 * unit can be: days, hours, minutes, seconds
                 * ± can be + or - (+ can be omitted for positive values)
                 * Examples: "days+1", "hours-2", "minutes30", "seconds-15"

    Returns:
        Current time (with optional offset) formatted as a string.

    Example:
        In YAML config:
        ```yaml
        # Basic usage
        timestamp: "${now}"  # Uses default format
        formatted: "${now:%Y-%m-%d %H:%M:%S}"  # Custom format only

        # With explicit named time offsets
        tomorrow: "${now:%Y-%m-%d,days+1}"
        yesterday: "${now:%Y-%m-%d,days-1}"
        future_time: "${now:%Y-%m-%d %H:%M:%S,days+1,hours-1}"
        complex: "${now:%Y%m%d_%H%M%S,days+7,hours-2,minutes+30,seconds-15}"

        # Default format with offsets (no format string)
        simple_tomorrow: "${now:days+1}"
        two_hours_ago: "${now:hours-2}"
        next_week: "${now:days+7}"

        # Multiple offsets
        complex_default: "${now:days+1,hours-3,minutes+45}"
        ```

        This would resolve to something like:
        ```yaml
        timestamp: "20231215_143022"
        formatted: "2023-12-15 14:30:22"
        tomorrow: "2023-12-16"
        yesterday: "2023-12-14"
        future_time: "2023-12-16 13:30:22"
        ```
    """
    # Default values
    format_str = "%Y%m%d_%H%M%S"
    days = hours = minutes = seconds = 0

    for i, arg in enumerate(args):
        if i == 0 and not any(
            unit in arg for unit in ["days", "hours", "minutes", "seconds"]
        ):
            # First argument is format string if it doesn't contain time units
            format_str = arg
        else:
            # Parse time offset specifications like "days+1", "hours-2", etc.
            arg = arg.strip()

            # Try to parse each known unit
            for unit in ["days", "hours", "minutes", "seconds"]:
                if arg.startswith(unit):
                    value_str = arg[len(unit) :]

                    # Handle the sign - remove + if present, keep - for negatives
                    if value_str.startswith("+"):
                        value_str = value_str[1:]

                    try:
                        value = int(value_str)
                        if unit == "days":
                            days = value
                        elif unit == "hours":
                            hours = value
                        elif unit == "minutes":
                            minutes = value
                        elif unit == "seconds":
                            seconds = value
                        break  # Exit the unit loop once we find a match
                    except ValueError:
                        logger.warning(
                            f"Could not parse time offset '{arg}': invalid value '{value_str}'"
                        )
                        break
            else:
                # No unit found in this argument
                if arg:  # Only warn if the argument is not empty
                    logger.warning(
                        f"Could not parse time offset '{arg}': unknown format"
                    )

    # Calculate the target time
    target_time = datetime.now() + timedelta(
        days=days, hours=hours, minutes=minutes, seconds=seconds
    )
    return target_time.strftime(format_str)


def register_resolvers() -> None:
    """Register all custom OmegaConf resolvers.

    This function should be called once at application startup to register
    all custom resolvers with OmegaConf.
    """
    logger.info("Registering OmegaConf resolvers")
    OmegaConf.register_new_resolver("now", now_resolver)
