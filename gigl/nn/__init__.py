"""GiGL NN Module"""

from gigl.nn.loss import RetrievalLoss
from gigl.nn.models import LightGCN, LinkPredictionGNN

__all__ = [
    "LightGCN",
    "LinkPredictionGNN",
    "RetrievalLoss",
]
