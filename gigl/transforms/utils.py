from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import HeteroData

# Type alias for edge types in PyG HeteroData
EdgeType = Tuple[str, str, str]


def add_node_attr(
    data: HeteroData,
    values: Union[Tensor, Dict[str, Tensor]],
    attr_name: Optional[str] = None,
    node_type_to_idx: Optional[Dict[str, Tuple[int, int]]] = None,
) -> HeteroData:
    """Helper function to add node attributes to a HeteroData object.

    Args:
        data: The HeteroData object to modify.
        values: Either:
            - A tensor of values in homogeneous node order (requires node_type_to_idx
              or will be computed from data.node_types), OR
            - A dictionary mapping node types to tensors of values for each type.
        attr_name: The name of the attribute to add. If None, concatenates to
            existing `x` attribute for each node type (or creates it).
        node_type_to_idx: Optional mapping from node type to (start, end) indices.
            Only used when values is a tensor. If None, it will be computed from
            data.node_types.

    Returns:
        The modified HeteroData object.
    """
    # If values is a dictionary, directly assign to each node type
    if isinstance(values, dict):
        for node_type, value in values.items():
            if node_type not in data.node_types:
                continue
            _set_node_attr_for_type(data, node_type, value, attr_name)
        return data

    # Otherwise, values is a tensor in homogeneous order - split by node type
    if node_type_to_idx is None:
        # Build mapping from node type to (start, end) indices in homogeneous tensor
        # When HeteroData is converted to homogeneous, nodes are ordered by node type.
        # This mapping lets us slice the homogeneous tensor to get values for each type.
        # Example: if data has 3 'user' nodes and 2 'item' nodes:
        #   node_type_to_idx = {'user': (0, 3), 'item': (3, 5)}
        node_type_to_idx = {}
        start_idx = 0
        for node_type in data.node_types:
            num_type_nodes = data[node_type].num_nodes
            node_type_to_idx[node_type] = (start_idx, start_idx + num_type_nodes)
            start_idx += num_type_nodes

    for node_type in data.node_types:
        start, end = node_type_to_idx[node_type]
        value = values[start:end]
        _set_node_attr_for_type(data, node_type, value, attr_name)

    return data


def _set_node_attr_for_type(
    data: HeteroData,
    node_type: str,
    value: Tensor,
    attr_name: Optional[str],
) -> None:
    """Helper to set node attribute for a single node type."""
    if attr_name is None:
        # Concatenate to existing x or create new x
        # Use getattr to safely get x attribute, returns None if not present
        x = getattr(data[node_type], "x", None)
        if x is not None:
            # Existing features found: concatenate new values to them
            # Reshape 1D tensor [num_nodes] to 2D [num_nodes, 1] for concatenation
            x = x.view(-1, 1) if x.dim() == 1 else x
            # Move value to same device/dtype as x, then concatenate along feature dim
            data[node_type].x = torch.cat(
                [x, value.to(x.device, x.dtype)], dim=-1
            )
        else:
            # No existing features: use new values as x directly
            data[node_type].x = value
    else:
        data[node_type][attr_name] = value


def add_edge_attr(
    data: HeteroData,
    values: Union[Tensor, Dict[EdgeType, Tensor]],
    attr_name: Optional[str] = None,
    edge_type_to_idx: Optional[Dict[EdgeType, Tuple[int, int]]] = None,
) -> HeteroData:
    """Helper function to add edge attributes to a HeteroData object.

    Args:
        data: The HeteroData object to modify.
        values: Either:
            - A tensor of values in homogeneous edge order (requires edge_type_to_idx
              or will be computed from data.edge_types), OR
            - A dictionary mapping edge types to tensors of values for each type.
        attr_name: The name of the attribute to add. If None, concatenates to
            existing `edge_attr` attribute for each edge type (or creates it).
        edge_type_to_idx: Optional mapping from edge type to (start, end) indices.
            Only used when values is a tensor. If None, it will be computed from
            data.edge_types.

    Returns:
        The modified HeteroData object.
    """
    # If values is a dictionary, directly assign to each edge type
    if isinstance(values, dict):
        for edge_type, value in values.items():
            if edge_type not in data.edge_types:
                continue
            _set_edge_attr_for_type(data, edge_type, value, attr_name)
        return data

    # Otherwise, values is a tensor in homogeneous order - split by edge type
    if edge_type_to_idx is None:
        # Build mapping from edge type to (start, end) indices in homogeneous tensor
        # When HeteroData is converted to homogeneous, edges are ordered by edge type.
        # This mapping lets us slice the homogeneous tensor to get values for each type.
        # Example: if data has 3 'buys' edges and 2 'views' edges:
        #   edge_type_to_idx = {('user', 'buys', 'item'): (0, 3), ('user', 'views', 'item'): (3, 5)}
        edge_type_to_idx = {}
        start_idx = 0
        for edge_type in data.edge_types:
            num_type_edges = data[edge_type].num_edges
            edge_type_to_idx[edge_type] = (start_idx, start_idx + num_type_edges)
            start_idx += num_type_edges

    for edge_type in data.edge_types:
        start, end = edge_type_to_idx[edge_type]
        value = values[start:end]
        _set_edge_attr_for_type(data, edge_type, value, attr_name)

    return data


def _set_edge_attr_for_type(
    data: HeteroData,
    edge_type: EdgeType,
    value: Tensor,
    attr_name: Optional[str],
) -> None:
    """Helper to set edge attribute for a single edge type."""
    if attr_name is None:
        # Concatenate to existing edge_attr or create new edge_attr
        # Use getattr to safely get edge_attr attribute, returns None if not present
        edge_attr = getattr(data[edge_type], "edge_attr", None)
        if edge_attr is not None:
            # Existing features found: concatenate new values to them
            # Reshape 1D tensor [num_edges] to 2D [num_edges, 1] for concatenation
            edge_attr = edge_attr.view(-1, 1) if edge_attr.dim() == 1 else edge_attr
            # Move value to same device/dtype as edge_attr, then concatenate along feature dim
            data[edge_type].edge_attr = torch.cat(
                [edge_attr, value.to(edge_attr.device, edge_attr.dtype)], dim=-1
            )
        else:
            # No existing features: use new values as edge_attr directly
            data[edge_type].edge_attr = value
    else:
        data[edge_type][attr_name] = value
