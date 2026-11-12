"""Backwards-compatible re-exports.

The shared loader types live in ``gigl.common.data.loader_types`` and ``TFRecordDataLoader`` in
``gigl.common.data.tf_record_loader``. Importing this module imports TensorFlow.
"""

from gigl.common.data.loader_types import (
    LoadedEntityTensors,
    SerializedTFRecordInfo,
    TFDatasetOptions,
)
from gigl.common.data.tf_record_loader import TFRecordDataLoader

__all__ = [
    "LoadedEntityTensors",
    "SerializedTFRecordInfo",
    "TFDatasetOptions",
    "TFRecordDataLoader",
]
