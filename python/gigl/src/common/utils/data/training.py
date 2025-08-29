import torch

from gigl.src.data_preprocessor.lib.types import FeatureSchema
from gigl.common.logger import Logger

logger = Logger()

def filter_features(
    feature_schema: FeatureSchema,
    feature_names: list[str],
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

    try:
        return x[:, indices].view(-1, len(indices))
    except Exception as e:
        logger.error(f"filter_feature func: Filtering features: {feature_names}, indices: {indices}")
        logger.error(f"filter_feature func: x shape: {x.shape}")
        logger.error(f"filter_feature func: feature_schema.feature_index: {feature_schema.feature_index}")
