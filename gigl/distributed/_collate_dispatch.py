"""PyG assembly for the C++ distributed collate path.

This module owns ``assemble_homogeneous`` / ``assemble_heterogeneous`` — turning
the component tensors returned by the ``gigl_core.collate_core`` C++ kernel into a
single PyG ``Data`` / ``HeteroData`` object. This is the one place where PyG storage
assignment happens for the C++ path; it mirrors GLT's ``to_data`` /
``to_hetero_data`` (``graphlearn_torch/loader/transform.py``).

Flag resolution is NOT defined here. ``resolve_collate_impl`` / ``COLLATE_IMPL_ENV_VAR``
/ ``CollateImpl`` are imported from the canonical module
``gigl.distributed.utils.neighborloader``. This module must never
define its own copy of those symbols.

The C++ kernel consumes already-on-device tensors and never issues transfers; the
assembled object inherits the tensors' device.
"""

from typing import Final, Optional, Protocol

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


# Keys whose tensors GLT keeps on their ARRIVAL device (it does not call .to on these):
# num_sampled_nodes / num_sampled_edges, both homogeneous and per-type hetero
# (dist_loader.py:365, 379, 428-429). Everything else is moved to to_device.
_COUNT_KEY_SUFFIXES: Final[tuple[str, ...]] = ("num_sampled_nodes", "num_sampled_edges")


def _move_msg_to_device(
    msg: dict[str, torch.Tensor], to_device: torch.device
) -> dict[str, torch.Tensor]:
    """Move every message tensor to ``to_device`` EXCEPT the per-hop count tensors.

    Reproduces GLT's device placement (``dist_loader.py:359-440``): ids/rows/cols/eids/
    nfeats/efeats/batch get ``.to(to_device)``; ``num_sampled_*`` stay on their arrival
    device. Mirrors the bulk move at ``base_dist_loader.py:991-993`` but is correct on
    EVERY path (CPU; CUDA with or without ``non_blocking_transfers``), since the kernel
    itself issues no transfers. Tensors already on ``to_device`` are returned as-is (no copy).

    Args:
        msg: SampleMessage with ``#META.`` keys already removed.
        to_device: Target device for graph/feature/id tensors.

    Returns:
        dict[str, torch.Tensor]: A new dict with tensors placed per GLT's contract.
    """
    moved: dict[str, torch.Tensor] = {}
    for key, value in msg.items():
        is_count = any(key == s or key.endswith("." + s) for s in _COUNT_KEY_SUFFIXES)
        if is_count or value.device == to_device:
            moved[key] = value
        else:
            moved[key] = value.to(to_device)
    return moved


def collate_cpp_homogeneous(
    msg: dict[str, torch.Tensor],
    batch_size: int,
    has_batch: bool,
    to_device: torch.device,
) -> Data:
    """Collate a homogeneous (metadata-stripped) sampler message via the C++ kernel.

    Reproduces the GLT homogeneous collate body (``dist_loader.py:423-449``): the
    edge index is reversed (``cols`` becomes row, ``rows`` becomes col), so we pass
    ``cols`` as the ``rows`` argument and ``rows`` as the ``cols`` argument to the
    kernel, which stacks them verbatim.

    Tensors are first placed on ``to_device`` per GLT's contract (the kernel issues no
    transfers), so the output device matches the Python oracle on every loader path.

    Args:
        msg: SampleMessage with ``#META.`` keys already removed.
        batch_size: Loader batch size (used when no ``batch`` key is present).
        has_batch: Whether NODE/SUBGRAPH sampling produced a batch (vs. link sampling).
        to_device: Target device (``self.to_device``); graph/id/feature tensors are moved here.

    Returns:
        Data: The assembled homogeneous batch.
    """
    from gigl_core import collate_core

    msg = _move_msg_to_device(msg, to_device)
    batch = msg.get("batch") if has_batch else None
    if has_batch and batch is None:
        batch = msg["ids"][:batch_size]
    components = collate_core.collate_homogeneous(
        ids=msg["ids"],
        rows=msg["cols"],  # reversed: cols -> row
        cols=msg["rows"],  # reversed: rows -> col
        eids=msg.get("eids"),
        nfeats=msg.get("nfeats"),
        efeats=msg.get("efeats"),
        batch=batch,
        num_sampled_nodes=msg.get("num_sampled_nodes"),
        num_sampled_edges=msg.get("num_sampled_edges"),
    )
    return assemble_homogeneous(components)


def collate_cpp_heterogeneous(
    msg: dict[str, torch.Tensor],
    node_types: list[str],
    edge_type_str_to_rev: dict[str, tuple[str, str, str]],
    reversed_edge_types: list[tuple[str, str, str]],
    input_type: str,
    has_batch: bool,
    batch_size: int,
    to_device: torch.device,
) -> HeteroData:
    """Collate a heterogeneous (metadata-stripped) sampler message via the C++ kernel.

    ``node_types``, ``edge_type_str_to_rev``, ``reversed_edge_types`` and
    ``input_type`` mirror GLT ``DistLoader`` state (``dist_loader.py:318-330, 356, 367``).

    Tensors are first placed on ``to_device`` per GLT's contract (``num_sampled_*`` kept on
    arrival device); the kernel issues no transfers, so the output device matches the
    Python oracle on every loader path.

    Args:
        msg: SampleMessage with ``#META.`` keys already removed.
        node_types: All node-type strings, in the loader's order.
        edge_type_str_to_rev: Map from message edge-type string to the reversed EdgeType.
        reversed_edge_types: The reversed edge-type list (drives empty-edge filling).
        input_type: The anchor node-type string.
        has_batch: Whether NODE/SUBGRAPH sampling produced a batch.
        batch_size: Loader batch size; used for the ``node[input_type][:batch_size]``
            fallback when the ``{input_type}.batch`` key is absent
            (``dist_loader.py:397-399``; GLT omits the key when ``output.batch`` is None,
            ``dist_neighbor_sampler.py:781-783``).
        to_device: Target device (``self.to_device``); graph/id/feature tensors are moved here.

    Returns:
        HeteroData: The assembled heterogeneous batch.
    """
    from gigl_core import collate_core

    msg = _move_msg_to_device(msg, to_device)
    result = collate_core.collate_heterogeneous(
        msg=msg,
        node_types=node_types,
        edge_type_str_to_rev=edge_type_str_to_rev,
        reversed_edge_types=reversed_edge_types,
        input_type=input_type,
        has_batch=has_batch,
        batch_size=batch_size,
    )
    return assemble_heterogeneous(result)
