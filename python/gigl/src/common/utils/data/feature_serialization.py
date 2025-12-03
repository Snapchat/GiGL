from typing import Sequence

import numpy as np

from gigl.common.utils.compute.serialization.serialize_np import NumpyCoder


class FeatureSerializationUtils:
    coder = NumpyCoder()

    @classmethod
    def deserialize_node_features(
        cls, serialized_features: Sequence[float]
    ) -> Sequence[float]:
        return serialized_features

    @classmethod
    def serialize_node_features(cls, features: np.ndarray) -> np.ndarray:
        return features

    @classmethod
    def deserialize_edge_features(
        cls, serialized_features: Sequence[float]
    ) -> Sequence[float]:
        return serialized_features

    @classmethod
    def serialize_edge_features(cls, features: np.ndarray) -> np.ndarray:
        return features
