import datetime
import uuid

from gigl.src.common.constants.time import (
    DEFAULT_DATETIME_FORMAT,
    NODASH_DATE_UNDERSCORE_TIME_FORMAT,
)


def parse_formatted_datetime(
    stringified_date: str, fmt: str = DEFAULT_DATETIME_FORMAT
) -> datetime.datetime:
    return datetime.datetime.strptime(stringified_date, fmt)


def format_datetime(dt: datetime.datetime, fmt: str = DEFAULT_DATETIME_FORMAT) -> str:
    return dt.strftime(fmt)


def current_formatted_datetime(fmt: str = DEFAULT_DATETIME_FORMAT) -> str:
    """
    Returns:
        str: current timezone aware utc datetime formatted as a string
        as per provided date format string.
    """
    return current_datetime().strftime(fmt)


def new_test_run_id(prefix: str) -> str:
    """Returns an ID for naming the resources of one test run.

    Each call returns a new ID, so call it once per test run and reuse the result.
    The ID is ``<prefix>_<UTC YYYYmmdd_HHMMSS>_<6 random hex chars>``. The random
    suffix keeps runs that start in the same second from sharing GCS paths,
    BigQuery tables, or Dataflow job names. When ``prefix`` is lowercase
    ``[a-z0-9_]``, so is the ID, which makes it valid as a KFP job name, BigQuery
    table name, GCS path component, and (after Dataflow's own sanitizing) Dataflow
    job name.

    Args:
        prefix: Leading text that identifies the test.

    Returns:
        str: The unique ID.
    """
    timestamp = current_formatted_datetime(fmt=NODASH_DATE_UNDERSCORE_TIME_FORMAT)
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:6]}"


def current_datetime() -> datetime.datetime:
    """
    Returns:
        str: current timezone aware utc datetime.
    """
    return datetime.datetime.now(datetime.timezone.utc)


def convert_days_to_ms(days: int) -> int:
    return days * 24 * 60 * 60 * 1000


def is_datetime_str_format_valid(datetime_str: str, datetime_format: str) -> bool:
    """
    Validates a datetime string with specified format. ex "%Y%m%d" for "YYYYmmdd"
    Return True/False so users can decide to raise error or reformat string

    :param datetime_str:
    :param datetime_format:
    :return:
    """
    try:
        datetime.datetime.strptime(datetime_str, datetime_format)
        return True
    except ValueError:
        return False
    except Exception as e:
        return False
