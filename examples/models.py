from typing import Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData

from gigl.src.common.models.pyg.homogeneous import GraphSAGE
from gigl.src.common.models.pyg.link_prediction import LinkPredictionDecoder
from gigl.src.common.types.graph_data import NodeType


class LinkPredictionGNN(nn.Module):
    """
    Link Prediction GNN model for both homogeneous and heterogeneous use cases
    Args:
        encoder (nn.Module): Either BasicGNN or Heterogeneous GNN for generating embeddings
        decoder (nn.Module): LinkPredictionDecoder for transforming embeddings into scores
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: LinkPredictionDecoder,
    ) -> None:
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(
        self,
        data: Union[Data, HeteroData],
        device: torch.device,
        output_node_types: Optional[list[NodeType]] = None,
    ) -> dict[NodeType, torch.Tensor]:
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


def init_example_gigl_homogeneous_cora_model(
    node_feature_dim: int,
    edge_feature_dim: int,
    args: dict[str, str],
    device: Optional[torch.device] = None,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> nn.Module:
    """
    Initializes a homogeneous GiGL LinkPredictionGNN model, which inherits from `nn.Module`. Note that this is just an example --
    any `nn.Module` subclass can work with GiGL training or inference.

    Args:
        node_feature_dim (int): Input node feature dimension for the model
        edge_feature_dim (int): Input edge feature dimension for the model
        args (Dict[str, str]): Arguments for training or inference
        device (Optional[torch.device]): Torch device of the model, if None defaults to CPU
        state_dict (Optional[Dict[str, torch.Tensor]]): State dictionary for pretrained model
    Returns:
        LinkPredictionGNN: Link Prediction model for training or inference
    """
    encoder_model = GraphSAGE(
        in_dim=node_feature_dim,
        hid_dim=int(args.get("hid_dim", 16)),
        out_dim=int(args.get("out_dim", 16)),
        edge_dim=edge_feature_dim if edge_feature_dim > 0 else None,
        num_layers=int(args.get("num_layers", 2)),
        conv_kwargs={},  # Use default conv args for this model type
        should_l2_normalize_embedding_layer_output=True,
    )

    decoder_model = LinkPredictionDecoder()  # Defaults to inner product decoder

    model: LinkPredictionGNN = LinkPredictionGNN(
        encoder=encoder_model,
        decoder=decoder_model,
    )

    # Push the model to the specified device.
    if device is None:
        device = torch.device("cpu")
    model.to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model
