from enum import Enum
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP


class DecoderType(Enum):
    hadamard_MLP = "hadamard_MLP"
    inner_product = "inner_product"

    @classmethod
    def get_all_criteria(cls) -> List[str]:
        return [m.name for m in cls]


class LinkPredictionDecoder(nn.Module):
    def __init__(
        self,
        decoder_type: DecoderType = DecoderType.inner_product,
        decoder_channel_list: Optional[List[int]] = None,
        act: Union[str, Callable, None] = F.relu,
        act_first: bool = False,
        bias: Union[bool, List[bool]] = False,
        plain_last: bool = False,
        norm: Optional[Union[str, Callable]] = None,
    ):
        super(LinkPredictionDecoder, self).__init__()
        self.decoder_type = decoder_type
        self.decoder_channel_list = decoder_channel_list

        if self.decoder_type.value == "hadamard_MLP" and not isinstance(
            self.decoder_channel_list, List
        ):
            raise ValueError(
                f"The decoder channel list must be provided when using 'hadamard_MLP' decoder, however you provided {self.decoder_channel_list}"
            )
        if (
            isinstance(self.decoder_channel_list, List)
            and len(self.decoder_channel_list) <= 1
        ):
            raise ValueError(
                f"The decoder channel list must have length at least 2, however you provided a list of length {len(self.decoder_channel_list)}"
            )
        if (
            isinstance(self.decoder_channel_list, List)
            and self.decoder_channel_list[-1] != 1
        ):
            raise ValueError(
                f"The last element in decoder channel list must be equal to 1, however you provided {self.decoder_channel_list[-1]}"
            )
        if self.decoder_type.value == "hadamard_MLP":
            self.mlp_decoder = MLP(
                channel_list=self.decoder_channel_list,
                act=act,
                act_first=act_first,
                bias=bias,
                plain_last=plain_last,
                norm=norm,
            )

    def forward(self, query_embeddings, candidate_embeddings) -> torch.Tensor:
        if self.decoder_type.value == "inner_product":
            scores = torch.mm(query_embeddings, candidate_embeddings.T)
        elif self.decoder_type.value == "hadamard_MLP":
            hadamard_scores = query_embeddings.unsqueeze(dim=1) * candidate_embeddings
            scores = self.mlp_decoder(hadamard_scores).sum(dim=-1)
        return scores
