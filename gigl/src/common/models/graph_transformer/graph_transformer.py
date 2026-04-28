"""Deprecated module - kept for backwards compatibility.

The implementation now lives in :mod:`gigl.nn.graph_transformer`. Importing
classes from here will continue to work but logs a deprecation warning at
instantiation time. This shim will be removed in a future release.
"""

from typing import Any

from gigl.common.logger import Logger
from gigl.nn.graph_transformer import FeedForwardNetwork as _FeedForwardNetwork
from gigl.nn.graph_transformer import (
    GraphTransformerEncoder as _GraphTransformerEncoder,
)
from gigl.nn.graph_transformer import (
    GraphTransformerEncoderLayer as _GraphTransformerEncoderLayer,
)

logger = Logger()

_DEPRECATION_MSG = (
    "gigl.src.common.models.graph_transformer.graph_transformer is deprecated "
    "and will be removed in a future release. Please import from "
    "`gigl.nn.graph_transformer` instead."
)


class FeedForwardNetwork(_FeedForwardNetwork):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(_DEPRECATION_MSG)
        super().__init__(*args, **kwargs)


class GraphTransformerEncoderLayer(_GraphTransformerEncoderLayer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(_DEPRECATION_MSG)
        super().__init__(*args, **kwargs)


class GraphTransformerEncoder(_GraphTransformerEncoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(_DEPRECATION_MSG)
        super().__init__(*args, **kwargs)
