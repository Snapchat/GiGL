"""Deprecated shim - implementation moved to :mod:`gigl.common.utils.time`.

Importing from here continues to work but logs a deprecation warning. This shim
will be removed in a future release.
"""

from gigl.common.logger import Logger
from gigl.common.utils.time import (  # noqa: F401  (re-export for backwards compat)
    convert_days_to_ms,
    current_datetime,
    current_formatted_datetime,
    format_datetime,
    is_datetime_str_format_valid,
    parse_formatted_datetime,
)

logger = Logger()
logger.warning(
    "gigl.common.utils.time is deprecated and will be removed in a future release. "
    "Please import from `gigl.common.utils.time` instead."
)
