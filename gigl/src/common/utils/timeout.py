"""Deprecated shim - implementation moved to :mod:`gigl.common.utils.timeout`.

Importing from here continues to work but logs a deprecation warning. This shim
will be removed in a future release.
"""

from gigl.common.logger import Logger
from gigl.common.utils.timeout import (  # noqa: F401  (re-export for backwards compat)
    TimedOutException,
    timeout,
)

logger = Logger()
logger.warning(
    "gigl.common.utils.timeout is deprecated and will be removed in a future release. "
    "Please import from `gigl.common.utils.timeout` instead."
)
