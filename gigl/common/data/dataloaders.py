"""Re-exports the TFRecord loader and its types from their new modules.

TODO: Remove this module once downstream imports use ``loader_types`` and ``tf_record_loader`` directly.
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
