from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Data, HeteroData

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

    @classmethod
    def for_ddp(
        cls,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        dummy_data: Union[Data, HeteroData],
        output_node_types: Optional[list[NodeType]] = None,
    ) -> "LinkPredictionGNN":
        """
        Created for DistributedDataParallel (DDP) training.
        This method wraps the encoder and decoder in DDP.
        Args:
            encoder (nn.Module): The GNN encoder to be wrapped.
            decoder (nn.Module): The decoder for link prediction.
            device (torch.device): The device to which the model should be moved.
            dummy_data (Union[Data, HeteroData]): Dummy data to initialize the encoder.
            output_node_types (Optional[list[NodeType]]): Node types for the output, required for heterogeneous data.
        Returns:
            LinkPredictionGNN: A new instance of LinkPredictionGNN for use with DDP.
        """

        # Dummy forward pass to initialize the encoder
        # This is necessary to ensure that the model is properly set up for DDP
        if output_node_types is None:
            output_node_types = []
        was_train = encoder.training
        encoder.eval()
        encoder(dummy_data, device, output_node_types)
        if was_train:
            encoder.train()
        ddp_encoder = DistributedDataParallel(
            encoder.to(device),
            device_ids=[device] if device.type != "cpu" else None,
            output_device=device if device.type != "cpu" else None,
        )
        ddp_decoder = DistributedDataParallel(
            decoder.to(device),
            device_ids=[device] if device.type != "cpu" else None,
            output_device=device if device.type != "cpu" else None,
        )
        return cls(
            encoder=ddp_encoder,
            decoder=ddp_decoder,
        )

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
