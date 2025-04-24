from typing import Dict, List, Optional

import torch
import torch_geometric.data
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import Linear

from gigl.src.common.models.layers.normalization import l2_normalize_embeddings
from gigl.src.common.models.pyg.nn.conv.hgt_conv import HGTConv
from gigl.src.common.models.pyg.nn.conv.simplehgn_conv import SimpleHGNConv
from gigl.src.common.models.pyg.nn.models.feature_embedding import FeatureEmbeddingLayer
from gigl.src.common.models.utils.torch import to_hetero_feat
from gigl.src.common.types.graph_data import EdgeType, NodeType


# HGT acts as a soft template for future Heterogeneous GNN model init and forwarding implementation.
class HGT(nn.Module):
    """
    Heterogeneous Graph Transformer model. Paper: https://arxiv.org/pdf/2003.01332.pdf
    This implementation is based on the example of:
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hgt_dblp.py
    Args:
        node_type_to_feat_dim_map (Dict[NodeType, int]): Dictionary mapping node types to their input dimensions.
        edge_type_to_feat_dim_map (Dict[EdgeType, int]): Dictionary mapping node types to their feature dimensions.
        hid_dim (int): Hidden dimension size.
        out_dim (int, optional): Output dimension size. Defaults to 128.
        num_layers (int, optional): Number of layers. Defaults to 2.
        num_heads (int, optional): Number of attention heads. Defaults to 2.
    """

    def __init__(
        self,
        node_type_to_feat_dim_map: Dict[NodeType, int],
        edge_type_to_feat_dim_map: Dict[EdgeType, int],
        hid_dim: int,
        out_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        should_l2_normalize_embedding_layer_output: bool = False,
        feature_embedding_layers: Optional[
            Dict[NodeType, FeatureEmbeddingLayer]
        ] = None,
        **kwargs,
    ):
        super().__init__()
        node_types = list(node_type_to_feat_dim_map.keys())
        edge_types = list(edge_type_to_feat_dim_map.keys())
        self.lin_dict = torch.nn.ModuleDict()
        for node_type, in_dim in node_type_to_feat_dim_map.items():
            self.lin_dict[node_type] = Linear(in_channels=in_dim, out_channels=hid_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hid_dim,
                out_channels=hid_dim,
                metadata=(node_types, edge_types),
                heads=num_heads,
            )
            self.convs.append(conv)

        self.lin = Linear(in_channels=hid_dim, out_channels=out_dim)

        self.should_l2_normalize_embedding_layer_output = (
            should_l2_normalize_embedding_layer_output
        )

        self.feature_embedding_layers = feature_embedding_layers

    def forward(
        self,
        data: torch_geometric.data.hetero_data.HeteroData,
        output_node_types: List[NodeType],
        device: torch.device,
    ) -> Dict[NodeType, torch.Tensor]:
        """
        Runs the forward pass of the module
        Args:
            data (torch_geometric.data.hetero_data.HeteroData): Input HeteroData object.
            output_node_types (List[NodeType]): List of node types for which to return the output embeddings.
        Returns:
            Dict[NodeType, torch.Tensor]: Dictionary with node types as keys and output tensors as values.
        """
        node_type_to_features_dict = data.x_dict

        if self.feature_embedding_layers:
            node_type_to_features_dict = {
                node_type: self.feature_embedding_layers[node_type](x)
                if node_type in self.feature_embedding_layers
                else x
                for node_type, x in node_type_to_features_dict.items()
            }

        node_type_to_features_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in node_type_to_features_dict.items()
        }

        for conv in self.convs:
            node_type_to_features_dict = conv(
                node_type_to_features_dict, data.edge_index_dict
            )

        node_typed_embeddings: Dict[NodeType, torch.Tensor] = {}

        for node_type in output_node_types:
            node_typed_embeddings[node_type] = (
                self.lin(node_type_to_features_dict[node_type])
                if node_type in node_type_to_features_dict
                else torch.FloatTensor([]).to(device=device)
            )

        if self.should_l2_normalize_embedding_layer_output:
            node_typed_embeddings = l2_normalize_embeddings(  # type: ignore
                node_typed_embeddings=node_typed_embeddings
            )

        return node_typed_embeddings


class SimpleHGN(nn.Module):
    def __init__(
        self,
        node_type_to_feat_dim_map: Dict[NodeType, int],
        edge_type_to_feat_dim_map: Dict[EdgeType, int],
        node_hid_dim: int,
        edge_hid_dim: int,
        edge_type_dim: int,
        node_out_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        should_use_node_residual: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        activation=F.elu,
        should_l2_normalize_embedding_layer_output: bool = False,
        **kwargs,
    ):
        """
        SimpleHGN layer from the paper: https://arxiv.org/pdf/2112.14936

        Args:
            node_type_to_feat_dim_map (Dict[NodeType, int]): Dictionary mapping node types to their input dimensions.
            edge_type_to_feat_dim_map (Dict[EdgeType, int]): Dictionary mapping edge types to their feature dimensions.
            node_hid_dim (int): Hidden dimension size for node features.
            edge_hid_dim (int): Hidden dimension size for edge features.
            edge_type_dim (int): Hidden dimension size for edge types.
            node_out_dim (int): Output dimension size for node features. Defaults to 128.
            num_layers (int): Number of layers. Defaults to 2.
            num_heads (int): Number of attention heads. Defaults to 2.
            should_use_node_residual (bool): Whether to use node residual. Defaults to True.
            negative_slope (float): Negative slope used in the LeakyReLU. Defaults to 0.2.
            dropout (float): Dropout rate. Defaults to 0.0.
            activation: Activation function. Defaults to `F.elu`.

        """

        super().__init__()

        self.num_layers = num_layers
        self.should_l2_normalize_embedding_layer_output = (
            should_l2_normalize_embedding_layer_output
        )
        # Used to project all node and edge types to compatible dimensions (node_hid_dim and edge_hid_dim, resp.)
        self.node_type_lin_dict = torch.nn.ModuleDict()
        for node_type, in_dim in node_type_to_feat_dim_map.items():
            self.node_type_lin_dict[str(node_type)] = nn.Linear(
                in_features=in_dim, out_features=node_hid_dim
            )

        # Used to project all edge types to compatible dimensions (edge_hid_dim)
        # if edge features are present, else None.
        self.should_have_edge_features: bool = any(edge_type_to_feat_dim_map.values())
        self.edge_type_lin_dict = torch.nn.ModuleDict()
        for edge_type, in_dim in edge_type_to_feat_dim_map.items():
            if in_dim:
                self.edge_type_lin_dict[str(edge_type)] = nn.Linear(
                    in_features=in_dim, out_features=edge_hid_dim
                )

        self.convs = torch.nn.ModuleList()
        for layer_id in range(num_layers):
            conv = SimpleHGNConv(
                in_channels=node_hid_dim if layer_id == 0 else node_hid_dim * num_heads,
                edge_in_channels=(
                    edge_hid_dim if self.should_have_edge_features else None
                ),
                edge_type_dim=edge_type_dim,
                out_channels=node_hid_dim,
                num_heads=num_heads,
                num_edge_types=len(edge_type_to_feat_dim_map),
                should_use_node_residual=should_use_node_residual,
                negative_slope=negative_slope,
                dropout=dropout,
            )
            self.convs.append(conv)

        self.lin = nn.Linear(
            in_features=node_hid_dim * num_heads, out_features=node_out_dim
        )
        self.activation = activation

    def forward(
        self,
        data: torch_geometric.data.hetero_data.HeteroData,
        output_node_types: List[NodeType],
        device: torch.device,
    ) -> Dict[NodeType, torch.Tensor]:
        # Align dimensions across all node-types and all edge-types, resp.
        x_dict = {
            node_type: self.node_type_lin_dict[node_type](x)
            for node_type, x in data.x_dict.items()
        }

        init_dict = {
            edge_type: {
                "edge_index": data.edge_index_dict[edge_type],
            }
            for edge_type in data.edge_index_dict.keys()
        }

        for edge_type in data.edge_types:
            maybe_edge_attr = getattr(data[edge_type], "edge_attr", None)
            if isinstance(maybe_edge_attr, torch.Tensor):
                init_dict[edge_type].update(
                    {
                        "edge_attr": self.edge_type_lin_dict[
                            f"{edge_type[0]}-{edge_type[1]}-{edge_type[2]}"
                        ](maybe_edge_attr)
                    }
                )
        init_dict.update({node_type: {"x": x} for node_type, x in x_dict.items()})

        # Convert hetero to homo graph, so we can pass around homo graph info to conv forwards.
        projected_hetero_data = torch_geometric.data.hetero_data.HeteroData(init_dict)
        projected_homo_data = projected_hetero_data.to_homogeneous()

        h = projected_homo_data.x
        for layer_id, conv in enumerate(self.convs):
            h = conv(
                edge_index=projected_homo_data.edge_index,
                node_feat=h,
                edge_feat=(
                    projected_homo_data.edge_attr
                    if self.should_have_edge_features
                    else None
                ),
                edge_type=projected_homo_data.edge_type,
            )

            if layer_id != self.num_layers - 1:
                h = self.activation(h)

        # Project to node output dim
        embeddings = self.lin(h)
        node_typed_embeddings = to_hetero_feat(
            h=embeddings,
            type_indices=projected_homo_data.node_type,
            types=projected_hetero_data.node_types,
        )

        for node_type in output_node_types:
            if node_type not in node_typed_embeddings:
                raise ValueError(
                    f"Requested node type {node_type} does not exist in output tensor."
                )

        if self.should_l2_normalize_embedding_layer_output:
            node_typed_embeddings = l2_normalize_embeddings(  # type: ignore
                node_typed_embeddings=node_typed_embeddings
            )
        return node_typed_embeddings
