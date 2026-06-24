"""PyG assembly for the C++ distributed collate path.

This module owns ``assemble_homogeneous`` / ``assemble_heterogeneous`` — turning
the component tensors returned by the ``gigl_core.collate_core`` C++ kernel into a
single PyG ``Data`` / ``HeteroData`` object. This is the one place where PyG storage
assignment happens for the C++ path; it mirrors GLT's ``to_data`` /
``to_hetero_data`` (``graphlearn_torch/loader/transform.py``).

Flag resolution is NOT defined here. ``resolve_collate_impl`` / ``COLLATE_IMPL_ENV_VAR``
/ ``CollateImpl`` are imported from the canonical module
``gigl.distributed.utils.neighborloader`` (workstream B). This module must never
define its own copy of those symbols.

The C++ kernel consumes already-on-device tensors and never issues transfers; the
assembled object inherits the tensors' device.
"""

from typing import Optional, Protocol

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType, NodeType

# Re-export the canonical flag API so call sites may import it from here OR from the
# canonical module; both resolve to the single B-owned definition. Do NOT redefine.
from gigl.distributed.utils.neighborloader import (  # noqa: F401
    COLLATE_IMPL_ENV_VAR,
    CollateImpl,
    resolve_collate_impl,
)


class CollateHeteroResultProtocol(Protocol):
    """Structural interface for the C++ ``CollateHeteroResult`` pybind11 object.

    Defines the attributes accessed by :func:`assemble_heterogeneous` so the
    type checker can verify call sites without depending on the C extension at
    import time.
    """

    node: dict[NodeType, torch.Tensor]
    edge_index: dict[EdgeType, torch.Tensor]
    edge: dict[EdgeType, torch.Tensor]
    x: dict[NodeType, torch.Tensor]
    edge_attr: dict[EdgeType, torch.Tensor]
    batch: dict[NodeType, torch.Tensor]
    num_sampled_nodes: dict[NodeType, torch.Tensor]
    num_sampled_edges: dict[EdgeType, torch.Tensor]


def assemble_homogeneous(components: dict[str, Optional[torch.Tensor]]) -> Data:
    """Assemble a homogeneous ``Data`` from C++ collate component tensors.

    Mirrors ``graphlearn_torch.loader.transform.to_data`` for the fields the
    GiGL loaders populate (no metadata; metadata is stripped upstream).

    Args:
        components: Dict with keys ``node``, ``edge_index``, ``edge``, ``x``,
            ``edge_attr``, ``batch``, ``num_sampled_nodes``, ``num_sampled_edges``.
            Optional values may be ``None``.

    Returns:
        Data: The assembled homogeneous batch.
    """
    data = Data(
        x=components["x"],
        edge_index=components["edge_index"],
        edge_attr=components["edge_attr"],
    )
    data.edge = components["edge"]
    data.node = components["node"]
    batch = components["batch"]
    data.batch = batch
    data.batch_size = batch.numel() if batch is not None else 0
    data.num_sampled_nodes = components["num_sampled_nodes"]
    data.num_sampled_edges = components["num_sampled_edges"]
    return data


def assemble_heterogeneous(result: CollateHeteroResultProtocol) -> HeteroData:
    """Assemble a ``HeteroData`` from a ``gigl_core.collate_core.CollateHeteroResult``.

    Mirrors ``graphlearn_torch.loader.transform.to_hetero_data`` for the fields
    the GiGL loaders populate (no metadata; metadata is stripped upstream).

    Args:
        result: A ``CollateHeteroResult`` exposing ``node``, ``edge_index``,
            ``edge``, ``x``, ``edge_attr``, ``batch``, ``num_sampled_nodes``,
            ``num_sampled_edges`` dict properties.

    Returns:
        HeteroData: The assembled heterogeneous batch.
    """
    data = HeteroData()

    edge_index_dict = result.edge_index
    edge_dict = result.edge
    edge_attr_dict = result.edge_attr
    for edge_type, edge_index in edge_index_dict.items():
        data[edge_type].edge_index = edge_index
        if edge_type in edge_dict:
            data[edge_type].edge = edge_dict[edge_type]
        if edge_type in edge_attr_dict:
            data[edge_type].edge_attr = edge_attr_dict[edge_type]

    x_dict = result.x
    for node_type, node in result.node.items():
        data[node_type].node = node
        if node_type in x_dict:
            data[node_type].x = x_dict[node_type]

    for node_type, batch in result.batch.items():
        data[node_type].batch = batch
        data[node_type].batch_size = batch.numel()

    data.num_sampled_nodes = dict(result.num_sampled_nodes)
    data.num_sampled_edges = dict(result.num_sampled_edges)
    return data
