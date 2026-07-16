"""Deprecated shim - datetime format constants moved to :mod:`gigl.common.constants`.

Importing from here continues to work but logs a deprecation warning. This shim
will be removed in a future release.
"""

from gigl.common.constants import (  # noqa: F401  (re-export for backwards compat)
    DASH_DATE_FMT,
    DEFAULT_DATE_FORMAT,
    DEFAULT_DATETIME_FORMAT,
    NODASH_DATE_FMT,
    NODASH_DATETIME_FORMAT,
)
from gigl.common.logger import Logger

logger = Logger()
logger.warning(
    "gigl.common.constants is deprecated and will be removed in a future "
    "release. Please import from `gigl.common.constants` instead."
)
