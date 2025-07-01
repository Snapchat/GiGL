from typing import Optional

import torch

from gigl.module.models import LinkPredictionGNN
from gigl.src.common.models.pyg.homogeneous import GraphSAGE
from gigl.src.common.models.pyg.link_prediction import LinkPredictionDecoder


def init_example_gigl_homogeneous_cora_model(
    node_feature_dim: int,
    edge_feature_dim: int,
    args: dict[str, str],
    device: Optional[torch.device] = None,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> LinkPredictionGNN:
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
