from typing import List

import torch
import torch.nn as nn
import torchrec

from gigl.common.logger import Logger

logger = Logger()


class LargeEmbeddingLookup(nn.Module):
    def __init__(self, embeddings_config: List[torchrec.EmbeddingBagConfig]):
        super().__init__()
        self.ebc = torchrec.EmbeddingBagCollection(
            tables=embeddings_config,
            device=torch.device("meta"),
        )

        logger.info(
            f"EmbeddingBagCollection named parameters: {list(self.ebc.named_parameters())}"
        )

    def forward(
        self, sparse_features: torchrec.KeyedJaggedTensor
    ) -> torchrec.KeyedTensor:
        # Forward pass through the embedding bag collection
        return self.ebc(sparse_features)
