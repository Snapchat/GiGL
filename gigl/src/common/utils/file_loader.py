"""Deprecated shim - implementation moved to :mod:`gigl.common.utils.file_loader`.

Importing from here continues to work but logs a deprecation warning. This shim
will be removed in a future release.
"""

from gigl.common.logger import Logger
from gigl.common.utils.file_loader import (  # noqa: F401  (re-export for backwards compat)
    FileLoader,
)

logger = Logger()
logger.warning(
    "gigl.common.utils.file_loader is deprecated and will be removed in a future release. "
    "Please import from `gigl.common.utils.file_loader` instead."
)
