from typing import List

import torch

from gigl.src.data_preprocessor.lib.types import FeatureSchema


def filter_features(
    feature_schema: FeatureSchema,
    feature_names: List[str],
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Returns tensor with features from x based on feature_names
    """
    indices = []
    for feature in feature_names:
        assert feature in feature_schema.feature_index, f"feature {feature} not found"
        start, end = feature_schema.feature_index[feature]
        indices.extend(list(range(start, end)))
    return x[:, indices].view(-1, len(indices))
