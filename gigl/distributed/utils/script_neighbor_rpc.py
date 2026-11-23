"""Serve remote one-hop neighbor requests with a TorchScript RPC instead of GLT's Python callee.

GLT sends each remote neighbor request as a Python RPC (``graphlearn_torch.distributed.rpc.rpc_request_async``). torch
RPC holds the interpreter lock while it unpickles such a request, runs the callee and pickles the reply, so a sampling
worker serves them one at a time, and slower while its own sampling holds the lock. A TorchScript function runs on the
RPC thread with no interpreter lock. Here that function calls ``torch.ops.gigl_core.rpc_lookup_neighbors``, a C++ lookup
over the partition's CSR (``gigl-core/core/sampling/rpc_neighbor_lookup.cpp``).

On by default; ``GIGL_TORCHSCRIPT_NEIGHBOR_RPC=off`` (``TORCHSCRIPT_NEIGHBOR_RPC_ENV``) turns it off, identically on
every process.

How a sampler uses it:

1. After registering its sampling callee, each sampler registers the CSR of every edge type of its partition under its
   callee id, which is the same on every rank (``register_partition_csr``). Registration keeps references to the
   dataset's tensors; nothing is copied.
2. All ranks then agree, with one ``all_gather``, to use the script path only if every rank could register.
3. Remote requests become ``send_script_neighbor_request`` calls; the reply is converted back to the
   ``NeighborOutput`` GLT's callee returns (``to_neighbor_output``).

Sampled neighbors are identical to GLT's whenever a neighbor list is no longer than the requested fan-out. A longer
list is sampled uniformly with replacement on both paths, but from different random generators.
"""

from typing import Callable, Final, Optional, Union, cast

import torch
from gigl_core import rpc_neighbor_lookup as native_lookup
from graphlearn_torch.data import Graph
from graphlearn_torch.sampler import NeighborOutput
from graphlearn_torch.typing import EdgeType
from torch.distributed import rpc

from gigl.common.logger import Logger
from gigl.utils.csr import get_compressed_id_offset

logger = Logger()

_HOMOGENEOUS_EDGE_TYPE_KEY: Final[str] = ""

# Registered by importing ``gigl_core.rpc_neighbor_lookup``. The cast gives the operator its schema's types; TorchScript
# resolves this global to the operator itself.
_rpc_lookup_neighbors = cast(
    Callable[
        [int, str, torch.Tensor, int],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    torch.ops.gigl_core.rpc_lookup_neighbors,
)

_script_function: Optional[torch.jit.ScriptFunction] = None


def edge_type_key(etype: Optional[EdgeType]) -> str:
    """The registry key of an edge type: ``""`` for a homogeneous graph, else ``"src|relation|dst"``.

    Example:
        >>> edge_type_key(None)
        ''
        >>> edge_type_key(("user", "to", "user"))
        'user|to|user'
    """
    if etype is None:
        return _HOMOGENEOUS_EDGE_TYPE_KEY
    return "|".join(etype)


def lookup_neighbors_remote(
    sampler_id: int,
    edge_type: str,
    ids: torch.Tensor,
    req_num: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The function the server runs for a script neighbor request; compiled by ``load_script_neighbor_rpc``."""
    return _rpc_lookup_neighbors(sampler_id, edge_type, ids, req_num)


def load_script_neighbor_rpc() -> torch.jit.ScriptFunction:
    """Compiles ``lookup_neighbors_remote`` once per process.

    Every process that serves script requests must call this before it receives one: torch RPC finds the function by
    its qualified name in the process's TorchScript compilation unit.
    """
    global _script_function
    if _script_function is None:
        _script_function = torch.jit.script(lookup_neighbors_remote)
    return _script_function


def _unsupported_reason(graph: Graph) -> Optional[str]:
    if graph.mode != "CPU":
        return f"graph mode is {graph.mode}, not CPU"
    topo = graph.topo
    if topo.indptr.dtype != torch.int64:
        return f"indptr is {topo.indptr.dtype}, not int64"
    if topo.indices.dtype not in (torch.int32, torch.int64):
        return f"indices are {topo.indices.dtype}, not int32 or int64"
    if not (topo.indptr.is_contiguous() and topo.indices.is_contiguous()):
        return "the CSR is not contiguous"
    return None


def register_partition_csr(
    sampler_id: int,
    local_graph: Union[Graph, dict[EdgeType, Graph]],
    with_edge: bool,
    with_weight: bool,
    device: torch.device,
    seed: Optional[int],
) -> bool:
    """Registers this partition's CSR for script neighbor requests addressed to ``sampler_id``.

    Never raises: a failure is logged and reported as ``False``, so that the caller's agreement across ranks still runs
    and every rank falls back together.

    Args:
        sampler_id: The sampler's RPC sampling callee id, identical on every rank.
        local_graph: The partition's graph, or one graph per edge type.
        with_edge: Whether replies carry edge ids.
        with_weight: Whether the sampler samples by edge weight, which the C++ lookup does not do.
        device: The sampler's device; the lookup runs on the CPU only.
        seed: The sampler's random seed, or ``None``. Seeds the lookup's sub-sampling as GLT's sampler seeds its own.

    Returns:
        bool: Whether every edge type was registered. When ``False`` nothing is registered and the reason is logged.
    """
    graphs: dict[str, Graph] = {}
    if isinstance(local_graph, dict):
        for etype, graph in local_graph.items():
            graphs[edge_type_key(etype)] = graph
    else:
        graphs[_HOMOGENEOUS_EDGE_TYPE_KEY] = local_graph
    reason: Optional[str] = None
    if device.type != "cpu":
        reason = f"the sampler device is {device}, not CPU"
    elif with_weight:
        reason = "weighted sampling is on"
    else:
        for key, graph in graphs.items():
            graph_reason = _unsupported_reason(graph)
            if graph_reason is not None:
                reason = f"edge type '{key}': {graph_reason}"
                break
    if reason is not None:
        logger.info(
            f"This partition cannot serve TorchScript neighbor requests ({reason}); the job uses GLT's Python callee."
        )
        return False
    try:
        load_script_neighbor_rpc()
        if seed is not None:
            native_lookup.set_random_seed(seed)
        for key, graph in graphs.items():
            topo = graph.topo
            native_lookup.register_csr(
                sampler_id=sampler_id,
                edge_type=key,
                indptr=topo.indptr,
                indices=topo.indices,
                edge_ids=topo.edge_ids,
                compressed_id_offset=get_compressed_id_offset(topo),
                with_edge=with_edge,
            )
    except Exception:
        logger.exception(
            "Registering this partition's CSR for TorchScript neighbor requests failed; the job uses GLT's Python "
            "callee."
        )
        native_lookup.unregister_csr(sampler_id)
        return False
    return True


def send_script_neighbor_request(
    worker_name: str,
    sampler_id: int,
    etype: Optional[EdgeType],
    ids: torch.Tensor,
    req_num: int,
) -> torch.futures.Future:
    """Sends one neighbor request to ``worker_name`` as a TorchScript RPC.

    Args:
        worker_name: The RPC worker that owns ``ids``.
        sampler_id: The sampler's RPC sampling callee id.
        etype: The edge type to sample along, or ``None`` on a homogeneous graph.
        ids: Global ids of the nodes to sample from, int64 on the CPU.
        req_num: Requested neighbors per node.

    Returns:
        A future of ``(neighbors, counts, edge_ids)``; convert it with ``to_neighbor_output``.
    """
    return rpc.rpc_async(
        worker_name,
        load_script_neighbor_rpc(),
        args=(sampler_id, edge_type_key(etype), ids, req_num),
    )


def to_neighbor_output(
    reply: tuple[torch.Tensor, torch.Tensor, torch.Tensor], with_edge: bool
) -> NeighborOutput:
    """Converts a script reply into the ``NeighborOutput`` GLT's ``RpcSamplingCallee`` returns."""
    neighbors, counts, edge_ids = reply
    if with_edge:
        return NeighborOutput(neighbors, counts, edge_ids)
    return NeighborOutput(neighbors, counts, None)
