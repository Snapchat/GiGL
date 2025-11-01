from typing import Optional

import torch
from gigl.module.models import LinkPredictionGNN
from gigl.src.common.models.pyg.heterogeneous import HGT
from gigl.src.common.models.pyg.homogeneous import GraphSAGE
from gigl.src.common.models.pyg.link_prediction import LinkPredictionDecoder
from gigl.src.common.types.graph_data import EdgeType, NodeType


def init_example_gigl_homogeneous_model(
    node_feature_dim: int,
    edge_feature_dim: int,
    hid_dim: int = 16,
    out_dim: int = 16,
    num_layers: int = 2,
    device: Optional[torch.device] = None,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
    wrap_with_ddp: bool = False,
    find_unused_encoder_parameters: bool = False,
) -> LinkPredictionGNN:
    """
    Initializes a homogeneous GiGL LinkPredictionGNN model, which inherits from `nn.Module`. Note that this is just an example --
    any `nn.Module` subclass can work with GiGL training or inference.

    Args:
        node_feature_dim (int): Input node feature dimension for the model
        edge_feature_dim (int): Input edge feature dimension for the model
        hid_dim (int): Hidden dimension of model
        out_dim (int): Output dimension of model
        num_layers (int): Number of layers to include in model
        device (Optional[torch.device]): Torch device of the model, if None defaults to CPU
        state_dict (Optional[dict[str, torch.Tensor]]): State dictionary for pretrained model
        wrap_with_ddp (bool): Whether to initialize the model for DistributedDataParallel (DDP) training.
            If True, the model will be setup for `DistributedDataParallel`.
        find_unused_encoder_parameters (bool): If when `find_unused_parameters` should be passed in when initializing the encoder for DDP training.
            See the docs at https://docs.pytorch.org/docs/stable/notes/ddp.html for more details.
            At a high level - this should only be True if the model has parameters that are not used in the forward pass,
            e.g. when the model has been initialized with all edge types in the graph, but the training task only uses a subset of them.

    Returns:
        LinkPredictionGNN: Link Prediction model for training or inference
    """
    # We use the GiGL GraphSAGE encoder for this example instead of the base PyG GraphSAGE model since the base model doesn't
    # have support for edge features or additional utilities like `should_l2_normalize_embedding_layer_output`,
    # which normalizes the output embeddings from the encoder. However, a base GraphSAGE encoder would also work in this case,
    # it would just have less modeling levers.
    encoder_model = GraphSAGE(
        in_dim=node_feature_dim,
        hid_dim=hid_dim,
        out_dim=out_dim,
        edge_dim=edge_feature_dim if edge_feature_dim > 0 else None,
        num_layers=num_layers,
        conv_kwargs={},  # Use default conv args for this model type
        should_l2_normalize_embedding_layer_output=True,
    )

    decoder_model = LinkPredictionDecoder()  # Defaults to inner product decoder
    model = LinkPredictionGNN(
        encoder=encoder_model,
        decoder=decoder_model,
    )
    if wrap_with_ddp:
        # Initialize the model for DDP training.
        model = model.to_ddp(
            device=device,
            find_unused_encoder_parameters=find_unused_encoder_parameters,
        )
    if device is None:
        device = torch.device("cpu")
    model.to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def init_example_gigl_heterogeneous_model(
    node_type_to_feature_dim: dict[NodeType, int],
    edge_type_to_feature_dim: dict[EdgeType, int],
    hid_dim: int = 16,
    out_dim: int = 16,
    num_layers: int = 2,
    num_heads: int = 2,
    device: Optional[torch.device] = None,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
    wrap_with_ddp: bool = False,
    find_unused_encoder_parameters: bool = False,
) -> LinkPredictionGNN:
    """
    Initializes a heterogeneous GiGL LinkPredictionGNN model, which inherits from `nn.Module`. Note that this is just an example --
    any `nn.Module` subclass can work with GiGL training or inference.

    Args:
        node_type_to_feature_dim (dict[NodeType, int]): Input node feature dimension for the model per node type
        edge_type_to_feature_dim (dict[EdgeType, int]): Input edge feature dimension for the model per edge type
        hid_dim (int): Hidden dimension of model
        out_dim (int): Output dimension of model
        num_layers (int): Number of layers to include in model
        num_heads (int): Number of attention heads to include in model
        device (Optional[torch.device]): Torch device of the model, if None defaults to CPU
        state_dict (Optional[dict[str, torch.Tensor]]): State dictionary for pretrained model
        wrap_with_ddp (bool): Whether to initialize the model for DistributedDataParallel (DDP) training.
            If True, the model will be setup for `DistributedDataParallel`.
        find_unused_encoder_parameters (bool): If when `find_unused_parameters` should be passed in when initializing the encoder for DDP training.
            See the docs at https://docs.pytorch.org/docs/stable/notes/ddp.html for more details.
            At a high level - this should only be True if the model has parameters that are not used in the forward pass,
            e.g. when the model has been initialized with all edge types in the graph, but the training task only uses a subset of them.
    Returns:
        LinkPredictionGNN: Link Prediction model for inference
    """
    # We use the GiGL HGT encoder for this example, since PyG only has support for the HGTConv layer, but not an entire HGT encoder.
    encoder_model = HGT(
        node_type_to_feat_dim_map=node_type_to_feature_dim,
        edge_type_to_feat_dim_map=edge_type_to_feature_dim,
        hid_dim=hid_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        should_l2_normalize_embedding_layer_output=True,
    )

    decoder_model = LinkPredictionDecoder()  # Defaults to inner product decoder
    model = LinkPredictionGNN(
        encoder=encoder_model,
        decoder=decoder_model,
    )
    if wrap_with_ddp:
        # Initialize the model for DDP training.
        # Since the HGT encoder has params for *all* node and edge types [1]
        # We need to allow DDP to find unused parameters.
        # As not all node types may be present in the training task.
        # 1. https://github.com/Snapchat/GiGL/blob/766fae5dc313e1224998ed5618cf70cf0fb4da30/python/gigl/src/common/models/pyg/heterogeneous.py#L47-L51
        model.to_ddp(
            device=device,
            find_unused_encoder_parameters=find_unused_encoder_parameters,
        )
    # Push the model to the specified device.
    if device is None:
        device = torch.device("cpu")
    model.to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model
