from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    SAGEConv,
    TransformerConv,
)
from torch_geometric.nn.models import MLP

from gigl.common.logger import Logger
from gigl.src.common.constants.training import DEFAULT_NUM_GNN_HOPS
from gigl.src.common.models.layers.normalization import l2_normalize_embeddings
from gigl.src.common.models.pyg import utils as pyg_utils
from gigl.src.common.models.pyg.nn.conv.edge_attr_gat_conv import EdgeAttrGATConv
from gigl.src.common.models.pyg.nn.conv.gin_conv import GINEConv
from gigl.src.common.models.pyg.nn.models.feature_embedding import FeatureEmbeddingLayer
from gigl.src.common.models.pyg.nn.models.feature_interaction import FeatureInteraction
from gigl.src.common.models.pyg.nn.models.jumping_knowledge import JumpingKnowledge
from gigl.src.common.types.model import GnnModel, GraphBackend

logger = Logger()


class BasicHomogeneousGNN(nn.Module, GnnModel):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        conv_kwargs: Dict[str, Any] = {},
        edge_dim: Optional[int] = None,
        num_layers: int = DEFAULT_NUM_GNN_HOPS,
        activation: Callable = F.relu,
        activation_before_norm: bool = False,  # apply activation function before normalization
        activation_after_last_conv: bool = False,  # apply activation after the last conv layer
        dropout: float = 0.0,  # dropout will auto set to 0.0 when model.eval()
        batchnorm: bool = False,  # batch norm
        linear_layer: bool = False,
        return_emb: bool = False,
        should_l2_normalize_embedding_layer_output: bool = False,
        jk_mode: Optional[str] = None,
        jk_lstm_dim: Optional[int] = None,
        feature_interaction_layer: Optional[FeatureInteraction] = None,
        feature_embedding_layer: Optional[FeatureEmbeddingLayer] = None,
        **kwargs,
    ):
        super(BasicHomogeneousGNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.activation = activation
        self.activation_before_norm = activation_before_norm
        self.activation_after_last_conv = activation_after_last_conv
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm
        self.num_layers = num_layers

        # Feature embedding layer to pass selected features through an embedding layer
        self.feature_embedding_layer = feature_embedding_layer
        # Feature interaction layers
        self.feats_interaction = feature_interaction_layer
        self.conv_layers: nn.ModuleList = self.init_conv_layers(  # type: ignore
            in_dim=in_dim,
            out_dim=hid_dim if linear_layer or jk_mode else out_dim,
            edge_dim=edge_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            **conv_kwargs,
        )

        if batchnorm:
            num_heads = int(conv_kwargs.get("heads", 1))
            num_batchnorm_layers = num_layers if jk_mode else num_layers - 1
            self.batchnorm_layers = nn.ModuleList(
                [
                    nn.BatchNorm1d(hid_dim * num_heads)
                    for i in range(num_batchnorm_layers)
                ]
            )

        self.should_l2_normalize_embedding_layer_output = (
            should_l2_normalize_embedding_layer_output
        )

        if jk_mode:
            self.jk_layer = JumpingKnowledge(
                mode=jk_mode,
                hid_dim=hid_dim,
                out_dim=out_dim if not linear_layer else hid_dim,
                num_layers=num_layers,
                lstm_dim=jk_lstm_dim,
            )
        else:
            self.jk_layer = None  # type: ignore
        self.return_emb = return_emb
        self.linear_layer = linear_layer
        if linear_layer:
            self.linear = nn.Linear(hid_dim, out_dim)

    def forward(
        self,
        data: torch_geometric.data.Data,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # pass selected features through an embedding layer
        if self.feature_embedding_layer:
            x = self.feature_embedding_layer(x)

        # node feature interaction before graph convolution
        if self.feats_interaction:
            x = self.feats_interaction(x)

        xs: List[torch.Tensor] = []
        for i, conv_layer in enumerate(self.conv_layers):
            if self.supports_edge_attr:
                x = conv_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            else:
                x = conv_layer(x=x, edge_index=edge_index)
            # exclude batch norm, activation, dropout after last layer
            if (
                i == self.num_layers - 1
                and not self.jk_layer
                and not self.activation_after_last_conv
            ):
                break
            if self.activation_before_norm:
                x = self.activation(x)
            if self.batchnorm:
                x = self.batchnorm_layers[i](x)
            if not self.activation_before_norm:
                x = self.activation(x)
            x = self.dropout(x)
            if self.jk_layer:
                xs.append(x)

        if self.jk_layer:
            x = self.jk_layer(xs)
        if self.should_l2_normalize_embedding_layer_output:
            x = l2_normalize_embeddings(node_typed_embeddings=x)
        if self.return_emb:
            return x
        if self.linear_layer:
            x = self.linear(x)

        return x

    def init_conv_layers(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        edge_dim: Optional[int],
        hid_dim: int,
        num_layers: int,
        **kwargs,
    ) -> nn.ModuleList:
        raise NotImplementedError

    @property
    def graph_backend(self) -> GraphBackend:
        return GraphBackend.PYG


class GraphSAGE(BasicHomogeneousGNN):
    supports_edge_weight = False
    supports_edge_attr = False

    def init_conv_layers(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        edge_dim: Optional[int],
        hid_dim: int,
        num_layers: int,
        **kwargs,
    ) -> nn.ModuleList:
        remaining_kwargs, discarded_kwargs = pyg_utils.filter_dict(
            input_dict=kwargs,
            keys_to_keep=pyg_utils.MESSAGE_PASSING_BASE_CLS_ARGS
            + ["aggr", "normalize", "root_weight", "project", "bias"],
        )
        logger.info(
            f"Discarded kwargs for {SAGEConv.__name__}: {discarded_kwargs.keys()}"
        )
        conv_layers = nn.ModuleList(
            [
                SAGEConv(
                    in_channels=in_dim if i == 0 else hid_dim,
                    out_channels=hid_dim if i < num_layers - 1 else out_dim,
                    **remaining_kwargs,
                )
                for i in range(num_layers)
            ]
        )
        return conv_layers


class GIN(BasicHomogeneousGNN):
    supports_edge_weight = False
    supports_edge_attr = False

    def init_conv_layers(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        edge_dim: Optional[int],
        hid_dim: int,
        num_layers: int,
        **kwargs,
    ) -> nn.ModuleList:
        eps: float = kwargs.pop("eps", 0.0)
        train_eps: bool = kwargs.pop("train_eps", False)

        remaining_kwargs, discarded_kwargs = pyg_utils.filter_dict(
            input_dict=kwargs, keys_to_keep=pyg_utils.MESSAGE_PASSING_BASE_CLS_ARGS
        )
        logger.info(
            f"Discarded kwargs for {GINConv.__name__}: {discarded_kwargs.keys()}"
        )

        conv_layers = nn.ModuleList(
            [
                GINConv(
                    nn=MLP(
                        [
                            in_dim if i == 0 else hid_dim,
                            hid_dim if i < num_layers - 1 else out_dim,
                            hid_dim if i < num_layers - 1 else out_dim,
                        ],
                        act=self.activation,
                        act_first=self.activation_before_norm,
                        # Note: PyG has its own BatchNorm class so this BatchNorm won't be converted to torch.nn.SyncBatchNorm
                        norm="batch_norm" if self.batchnorm else None,
                    ),
                    eps=eps,
                    train_eps=train_eps,
                    **remaining_kwargs,
                )
                for i in range(num_layers)
            ]
        )
        return conv_layers


class GINE(BasicHomogeneousGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv_layers(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        edge_dim: Optional[int],
        hid_dim: int,
        num_layers: int,
        **kwargs,
    ) -> nn.ModuleList:
        eps: float = kwargs.pop("eps", 0.0)
        train_eps: bool = kwargs.pop("train_eps", False)

        remaining_kwargs, discarded_kwargs = pyg_utils.filter_dict(
            input_dict=kwargs, keys_to_keep=pyg_utils.MESSAGE_PASSING_BASE_CLS_ARGS
        )
        logger.info(
            f"Discarded kwargs for {GINEConv.__name__}: {discarded_kwargs.keys()}"
        )

        conv_layers = nn.ModuleList(
            [
                GINEConv(
                    nn=MLP(
                        [
                            in_dim if i == 0 else hid_dim,
                            hid_dim if i < num_layers - 1 else out_dim,
                            hid_dim if i < num_layers - 1 else out_dim,
                        ],
                        act=self.activation,
                        act_first=self.activation_before_norm,
                        # Note: PyG has its own BatchNorm class so this BatchNorm won't be converted to torch.nn.SyncBatchNorm
                        norm="batch_norm" if self.batchnorm else None,
                    ),
                    eps=eps,
                    train_eps=train_eps,
                    edge_dim=edge_dim,
                    **remaining_kwargs,
                )
                for i in range(num_layers)
            ]
        )
        return conv_layers


class GAT(BasicHomogeneousGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv_layers(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        edge_dim: Optional[int],
        hid_dim: int,
        num_layers: int,
        **kwargs,
    ) -> nn.ModuleList:
        num_heads = int(kwargs.pop("heads", 1))

        remaining_kwargs, discarded_kwargs = pyg_utils.filter_dict(
            input_dict=kwargs,
            keys_to_keep=pyg_utils.MESSAGE_PASSING_BASE_CLS_ARGS
            + [
                "concat",
                "negative_slope",
                "dropout",
                "add_self_loops",
                "fill_value",
                "bias",
            ],
        )
        logger.info(
            f"Discarded kwargs for {GATConv.__name__}: {discarded_kwargs.keys()}"
        )

        conv_layers = nn.ModuleList(
            [
                GATConv(
                    in_channels=in_dim if i == 0 else hid_dim * num_heads,
                    out_channels=hid_dim if i < num_layers - 1 else out_dim,
                    edge_dim=edge_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    **remaining_kwargs,
                )
                for i in range(num_layers)
            ]
        )
        return conv_layers


class GATv2(BasicHomogeneousGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv_layers(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        edge_dim: Optional[int],
        hid_dim: int,
        num_layers: int,
        **kwargs,
    ) -> nn.ModuleList:
        num_heads = kwargs.pop("heads", 1)
        fill_value = kwargs.pop("fill_value", "mean")
        share_weights = kwargs.pop("share_weights", False)

        remaining_kwargs, discarded_kwargs = pyg_utils.filter_dict(
            input_dict=kwargs,
            keys_to_keep=pyg_utils.MESSAGE_PASSING_BASE_CLS_ARGS
            + ["concat", "negative_slope", "dropout", "add_self_loops", "bias"],
        )
        logger.info(
            f"Discarded kwargs for {GATv2Conv.__name__}: {discarded_kwargs.keys()}"
        )

        conv_layers = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=in_dim if i == 0 else hid_dim * num_heads,
                    out_channels=hid_dim if i < num_layers - 1 else out_dim,
                    edge_dim=edge_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    fill_value=fill_value,
                    share_weights=share_weights,
                    **remaining_kwargs,
                )
                for i in range(num_layers)
            ]
        )
        return conv_layers


class EdgeAttrGAT(BasicHomogeneousGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv_layers(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        edge_dim: Optional[int],
        hid_dim: int,
        num_layers: int,
        **kwargs,
    ) -> nn.ModuleList:
        num_heads = int(kwargs.pop("heads", 1))
        share_edge_att_message_weight = kwargs.pop(
            "share_edge_att_message_weight", True
        )

        remaining_kwargs, discarded_kwargs = pyg_utils.filter_dict(
            input_dict=kwargs,
            keys_to_keep=pyg_utils.MESSAGE_PASSING_BASE_CLS_ARGS
            + [
                "concat",
                "negative_slope",
                "dropout",
                "add_self_loops",
                "fill_value",
                "bias",
            ],
        )

        logger.info(
            f"Discarded kwargs for {EdgeAttrGATConv.__name__}: {discarded_kwargs.keys()}"
        )

        conv_layers = nn.ModuleList(
            [
                EdgeAttrGATConv(
                    in_channels=in_dim if i == 0 else hid_dim * num_heads,
                    out_channels=hid_dim if i < num_layers - 1 else out_dim,
                    edge_dim=edge_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    share_edge_att_message_weight=share_edge_att_message_weight,
                    **remaining_kwargs,
                )
                for i in range(num_layers)
            ]
        )
        return conv_layers


class Transformer(BasicHomogeneousGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv_layers(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        edge_dim: Optional[int],
        hid_dim: int,
        num_layers: int,
        **kwargs,
    ) -> nn.ModuleList:
        num_heads = int(kwargs.pop("heads", 1))
        beta = kwargs.pop("beta", False)

        remaining_kwargs, discarded_kwargs = pyg_utils.filter_dict(
            input_dict=kwargs,
            keys_to_keep=pyg_utils.MESSAGE_PASSING_BASE_CLS_ARGS
            + [
                "concat",
                "dropout",
                "bias",
                "root_weight",
            ],
        )

        logger.info(
            f"Discarded kwargs for {EdgeAttrGATConv.__name__}: {discarded_kwargs.keys()}"
        )

        # Layers prior to the last layer will be a concatenation of heads by default
        # The last layer will do a average pool on all heads so the output is still out_dim
        return nn.ModuleList(
            [
                TransformerConv(
                    in_channels=in_dim if i == 0 else hid_dim * num_heads,
                    out_channels=hid_dim if i < num_layers - 1 else out_dim,
                    edge_dim=edge_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    beta=beta,
                    **remaining_kwargs,
                )
                for i in range(num_layers)
            ]
        )


class TwoLayerGCN(torch.nn.Module, GnnModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hid_dim: int = 16,
        is_training: bool = True,
        should_l2_normalize_output: bool = False,
        **kwargs,
    ):
        """
        Simple 2 layer GCN Implementation using PyG constructs
        Args:
            in_feats (int): number input features
            out_dim (int): num output classes
            h_feats (int, optional): num hidden features. Defaults to 16.
            **kwargs (:class:`torch_geometric.nn.conv.MessagePassing`):
                Additional arguments for all GCNConv layers
        """
        super().__init__()
        self.is_training = is_training
        self.should_normalize = should_l2_normalize_output

        remaining_kwargs, discarded_kwargs = pyg_utils.filter_dict(
            input_dict=kwargs,
            keys_to_keep=pyg_utils.MESSAGE_PASSING_BASE_CLS_ARGS
            + [
                "improved",
                "cached",
                "add_self_loops",
                "normalize",
                "bias",
            ],
        )

        logger.info(
            f"Discarded kwargs for {GCNConv.__name__}: {discarded_kwargs.keys()}"
        )

        self.conv1 = GCNConv(
            in_channels=in_dim, out_channels=hid_dim, **remaining_kwargs
        )
        self.conv2 = GCNConv(
            in_channels=hid_dim, out_channels=out_dim, **remaining_kwargs
        )

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.is_training)
        x = self.conv2(x, edge_index)
        if self.should_normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

    @property
    def graph_backend(self) -> GraphBackend:
        return GraphBackend.PYG
