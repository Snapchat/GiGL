"""Deprecated shim - implementation moved to :mod:`gigl.common.utils.bq`.

Importing from here continues to work but logs a deprecation warning. This shim
will be removed in a future release.
"""

from gigl.common.logger import Logger
from gigl.common.utils.bq import (  # noqa: F401  (re-export for backwards compat)
    BqUtils,
)

logger = Logger()
logger.warning(
    "gigl.common.utils.bq is deprecated and will be removed in a future release. "
    "Please import from `gigl.common.utils.bq` instead."
)
