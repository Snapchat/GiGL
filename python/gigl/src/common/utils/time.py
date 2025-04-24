import datetime

from gigl.src.common.constants.time import DEFAULT_DATETIME_FORMAT


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
