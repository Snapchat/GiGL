from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Data, HeteroData
from typing_extensions import Self

from gigl.src.common.types.graph_data import NodeType


class LinkPredictionGNN(nn.Module):
    """
    Link Prediction GNN model for both homogeneous and heterogeneous use cases
    Args:
        encoder (nn.Module): Either BasicGNN or Heterogeneous GNN for generating embeddings
        decoder (nn.Module): Decoder for transforming embeddings into scores.
            Recommended to use `gigl.src.common.models.pyg.link_prediction.LinkPredictionDecoder`
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(
        self,
        data: Union[Data, HeteroData],
        device: torch.device,
        output_node_types: Optional[list[NodeType]] = None,
    ) -> Union[torch.Tensor, dict[NodeType, torch.Tensor]]:
        if isinstance(data, HeteroData):
            if output_node_types is None:
                raise ValueError(
                    "Output node types must be specified in forward() pass for heterogeneous model"
                )
            return self._encoder(
                data=data, output_node_types=output_node_types, device=device
            )
        else:
            return self._encoder(data=data, device=device)

    def decode(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self._decoder(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
        )

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    def to_ddp(
        self, device: torch.device, find_unused_encoder_parameters: bool = False
    ) -> Self:
        """
        Converts the model to DistributedDataParallel (DDP) mode.
        We expose this function so that both the encoder and decoder can be safely used in a distributed manner.
        If we do not do this, then calling forward() on the individual modules may not work correctly.
        e.g. this makes it safe to do: `LinkPredictionGNN.encoder(data, device)`
        Without this method, DDP may not correctly handle the forward pass as it only wraps the model as a whole.
        See:
            https://discuss.pytorch.org/t/using-model-functions-in-distributeddataparallel/160214
            and DistributedDataParallel.forward calling _pre_forward:
            https://github.com/pytorch/pytorch/blob/26807dcf277feb2d99ab88d7b6da526488baea93/torch/nn/parallel/distributed.py#L1657
        Args:
            device (torch.device): The device to which the model should be moved.
            find_unused_encoder_parameters (bool): Whether to find unused parameters in the model.
                This should be set to True if the model has parameters that are not used in the forward pass.
        Returns:
            LinkPredictionGNN: A new instance of LinkPredictionGNN for use with DDP.
        """

        ddp_encoder = DistributedDataParallel(
            self._encoder,
            device_ids=[device] if device.type != "cpu" else None,
            output_device=device if device.type != "cpu" else None,
            find_unused_parameters=find_unused_encoder_parameters,
        )
        # Do this "backwards" so the we can define "ddp_decoder" as a nn.Module first...
        if not any(p.requires_grad for p in self._decoder.parameters()):
            # If the decoder has no trainable parameters, we can just use it as is
            ddp_decoder = self._decoder.to(device)
        else:
            # Only wrap the decoder in DDP if it has parameters that require gradients
            # Otherwise DDP will complain about no parameters to train.
            ddp_decoder = DistributedDataParallel(
                self._decoder.to(device),
                device_ids=[device] if device.type != "cpu" else None,
                output_device=device if device.type != "cpu" else None,
            )
        self._encoder = ddp_encoder
        self._decoder = ddp_decoder
        return self

    def unwrap_from_ddp(self) -> "LinkPredictionGNN":
        """
        Unwraps the model from DistributedDataParallel if it is wrapped.

        Returns:
            LinkPredictionGNN: A new instance of LinkPredictionGNN with the original encoder and decoder.
        """
        if isinstance(self._encoder, DistributedDataParallel):
            encoder = self._encoder.module
        else:
            encoder = self._encoder

        if isinstance(self._decoder, DistributedDataParallel):
            decoder = self._decoder.module
        else:
            decoder = self._decoder

        return LinkPredictionGNN(encoder=encoder, decoder=decoder)
