from enum import Enum
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP

from gigl.src.common.models.layers.feature_interaction import DCNv2


class CombinationMode(Enum):
    parallel = "parallel"
    stacked = "stacked"


class FeatureInteraction(nn.Module):
    def __init__(
        self,
        in_dim: int,
        use_dcnv2_feats_interaction: bool = False,
        dcnv2_kwargs: Dict[str, Any] = {},
        use_mlp_feats_interaction: bool = False,
        mlp_feats_kwargs: Dict[str, Any] = {},
        activation: Callable = F.relu,
        combination_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert (
            use_dcnv2_feats_interaction or use_mlp_feats_interaction
        ), f"At least one type of the feature interaction layer should be enabled"
        self.in_dim = in_dim
        self.use_dcnv2_feats_interaction = use_dcnv2_feats_interaction
        self.dcnv2 = None
        self.use_mlp_feats_interaction = use_mlp_feats_interaction
        self.mlp = None
        self.combination_mode = None
        if use_dcnv2_feats_interaction and use_mlp_feats_interaction:
            if not combination_mode:
                raise ValueError(
                    f"combination_mode must be provided if both DCN and MLP layers are enabled"
                )
            try:
                self.combination_mode = CombinationMode[combination_mode]
            except:
                raise ValueError(
                    f"provided combination_mode={combination_mode} is not supported"
                )
        if use_dcnv2_feats_interaction:
            self.dcnv2 = DCNv2(
                in_dim=in_dim,
                **dcnv2_kwargs,
            )
        if use_mlp_feats_interaction:
            self.mlp = MLP(
                in_channels=in_dim,
                hidden_channels=in_dim,
                out_channels=in_dim,
                act=activation,
                **mlp_feats_kwargs,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.combination_mode and self.combination_mode == CombinationMode.parallel:
            assert isinstance(self.dcnv2, nn.Module) and isinstance(self.mlp, nn.Module)
            x_cross = self.dcnv2(x)
            x_deep = self.mlp(x)
            return torch.cat((x_cross, x_deep), dim=-1)
        elif self.combination_mode and self.combination_mode == CombinationMode.stacked:
            assert isinstance(self.dcnv2, nn.Module) and isinstance(self.mlp, nn.Module)
            x_cross = self.dcnv2(x)
            return self.mlp(x_cross)
        else:
            if self.dcnv2:
                assert isinstance(self.dcnv2, nn.Module)
                return self.dcnv2(x)
            assert isinstance(self.mlp, nn.Module)
            return self.mlp(x)

    @property
    def output_dim(self):
        if self.combination_mode and self.combination_mode == CombinationMode.parallel:
            return self.in_dim * 2
        return self.in_dim

    def reset_parameters(self):
        if self.dcnv2:
            self.dcnv2.reset_parameters()
        if self.mlp:
            self.mlp.reset_parameters()
