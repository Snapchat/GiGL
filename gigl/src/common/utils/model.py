"""Deprecated shim - implementation moved to :mod:`gigl.common.utils.model`.

Importing from here continues to work but logs a deprecation warning. This shim
will be removed in a future release.
"""

from gigl.common.logger import Logger
from gigl.common.utils.model import (  # noqa: F401  (re-export for backwards compat)
    load_scripted_model_from_uri,
    load_state_dict_from_uri,
    save_scripted_model,
    save_state_dict,
)

logger = Logger()
logger.warning(
    "gigl.common.utils.model is deprecated and will be removed in a future release. "
    "Please import from `gigl.common.utils.model` instead."
)
