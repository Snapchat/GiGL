"""Serve one-hop neighbor lookups, local and remote, with a native lookup instead of GLT's sampler and Python callee.

GLT sends each remote neighbor request as a Python RPC (``graphlearn_torch.distributed.rpc.rpc_request_async``). torch
RPC holds the interpreter lock while it unpickles such a request, runs the callee and pickles the reply, so a sampling
worker serves them one at a time, and slower while its own sampling holds the lock. A native request is instead sent as
a TorchScript RPC function that calls ``torch.ops.gigl_core.lookup_neighbors``, a C++ lookup over the partition's
CSR (``gigl-core/core/sampling/neighbor_lookup.cpp``). The serving side runs C++ on the RPC thread with no Python
and no interpreter lock; TorchScript is only the transport.
The same operator serves the local partition: ``lookup_neighbors`` calls it from Python, and a torch operator call
releases the interpreter lock while it runs. With the flag on, local and remote nodes go through the same kernel, so
one hop uses one kernel and one random stream.

On by default; ``GIGL_NATIVE_NEIGHBOR_LOOKUP=off`` (``NATIVE_NEIGHBOR_LOOKUP_ENV``) turns it off, identically on
every process.

How a sampler uses it:

1. After registering its sampling callee, each sampler registers the CSR of every edge type of its partition under its
   callee id, which is the same on every rank (``register_partition_csr``). Registration keeps references to the
   dataset's tensors; nothing is copied.
2. All ranks then agree, with one ``all_gather``, to use the native path only if every rank could register.
3. Local nodes call ``lookup_neighbors``, which returns the ``NeighborOutput`` GLT's sampler returns.
4. Remote requests become ``send_native_neighbor_request`` calls; the reply is a tuple of ``NeighborOutput``'s fields,
   in order, so the caller rebuilds the ``NeighborOutput`` GLT's callee returns with ``NeighborOutput(*reply)``.

Sampled neighbors are identical to GLT's whenever a neighbor list is no longer than the requested fan-out. A longer
list is sampled uniformly with replacement on both paths, but from different random generators.
"""

from typing import Callable, Final, Optional, Union, cast

import torch
from gigl_core import neighbor_lookup as native_lookup
from graphlearn_torch.data import Graph
from graphlearn_torch.sampler import NeighborOutput
from graphlearn_torch.typing import EdgeType
from torch.distributed import rpc

from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType as GiglEdgeType
from gigl.utils.csr import get_compressed_id_offset

logger = Logger()

_HOMOGENEOUS_EDGE_TYPE_KEY: Final[str] = ""

# Registered by importing ``gigl_core.neighbor_lookup``. The cast gives the operator its schema's types; TorchScript
# resolves this global to the operator itself.
_lookup_neighbors_op = cast(
    Callable[
        [int, str, torch.Tensor, int],
        tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    ],
    torch.ops.gigl_core.lookup_neighbors,
)

_compiled_lookup_function: Optional[torch.jit.ScriptFunction] = None


def edge_type_key(etype: Optional[EdgeType]) -> str:
    """The registry key of an edge type: ``""`` for a homogeneous graph, else ``"src-relation-dst"``.

    The key is GiGL's ``EdgeType`` repr. GLT may pass a plain tuple, so it is converted first.

    Example:
        >>> edge_type_key(None)
        ''
        >>> edge_type_key(("user", "to", "user"))
        'user-to-user'

    Args:
        etype (Optional[EdgeType]): The edge type, or ``None`` for a homogeneous graph.

    Returns:
        str: The registry key.
    """
    if etype is None:
        return _HOMOGENEOUS_EDGE_TYPE_KEY
    return repr(GiglEdgeType(*etype))


def lookup_neighbors_remote(
    sampler_id: int,
    edge_type: str,
    ids: torch.Tensor,
    req_num: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """The function the server runs for a native neighbor request; compiled by ``load_native_neighbor_rpc``.

    Args:
        sampler_id (int): The sampler's RPC sampling callee id.
        edge_type (str): The registry key of the edge type, from ``edge_type_key``.
        ids (torch.Tensor): Global ids of the nodes to sample from, int64 on the CPU.
        req_num (int): Requested neighbors per node; negative means all.

    Returns:
        tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: Flat neighbor ids, per-id neighbor counts and edge
        ids, or ``None`` for edge ids when the CSR was registered with ``with_edge=False``.
    """
    return _lookup_neighbors_op(sampler_id, edge_type, ids, req_num)


def load_native_neighbor_rpc() -> torch.jit.ScriptFunction:
    """Compiles ``lookup_neighbors_remote`` once per process.

    Every process that serves native requests must call this before it receives one: torch RPC finds the function by
    its qualified name in the process's TorchScript compilation unit.

    Returns:
        torch.jit.ScriptFunction: The compiled function.
    """
    global _compiled_lookup_function
    if _compiled_lookup_function is None:
        _compiled_lookup_function = torch.jit.script(lookup_neighbors_remote)
    return _compiled_lookup_function


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
    """Registers this partition's CSR for native neighbor lookups, local and remote, addressed to ``sampler_id``.

    Never raises: a failure is logged and reported as ``False``, so that the caller's agreement across ranks still runs
    and every rank falls back together.

    Args:
        sampler_id (int): The sampler's RPC sampling callee id, identical on every rank.
        local_graph (Union[Graph, dict[EdgeType, Graph]]): The partition's graph, or one graph per edge type.
        with_edge (bool): Whether replies carry edge ids; if not, they carry ``None``.
        with_weight (bool): Whether the sampler samples by edge weight, which the C++ lookup does not do.
        device (torch.device): The sampler's device; the lookup runs on the CPU only.
        seed (Optional[int]): The sampler's random seed, or ``None``. Seeds the lookup's sub-sampling as GLT's sampler seeds its own.

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
            f"This partition cannot serve native neighbor requests ({reason}); the job uses GLT's Python callee."
        )
        return False
    try:
        load_native_neighbor_rpc()
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
            "Registering this partition's CSR for native neighbor requests failed; the job uses GLT's Python "
            "callee."
        )
        native_lookup.unregister_csr(sampler_id)
        return False
    return True


def send_native_neighbor_request(
    worker_name: str,
    sampler_id: int,
    etype: Optional[EdgeType],
    ids: torch.Tensor,
    req_num: int,
) -> torch.futures.Future:
    """Sends one native neighbor request to ``worker_name``, as a TorchScript RPC function.

    Args:
        worker_name (str): The RPC worker that owns ``ids``.
        sampler_id (int): The sampler's RPC sampling callee id.
        etype (Optional[EdgeType]): The edge type to sample along, or ``None`` on a homogeneous graph.
        ids (torch.Tensor): Global ids of the nodes to sample from, int64 on the CPU.
        req_num (int): Requested neighbors per node.

    Returns:
        torch.futures.Future: A future of ``(neighbors, counts, edge_ids or None)``; the caller wraps the result in
        ``NeighborOutput``.
    """
    return rpc.rpc_async(
        worker_name,
        load_native_neighbor_rpc(),
        args=(sampler_id, edge_type_key(etype), ids, req_num),
    )


def lookup_neighbors(
    sampler_id: int,
    etype: Optional[EdgeType],
    ids: torch.Tensor,
    req_num: int,
) -> NeighborOutput:
    """Looks up neighbors in this process's registered CSR, with the lookup a native request runs on the server.

    Samplers call this for local nodes. A torch operator call releases the interpreter lock while it runs.

    A negative ``req_num``, or one above the int32 maximum, means all neighbors.
    An id outside the partition's rows has 0 neighbors.
    A row with at most ``req_num`` neighbors is copied in stored order.
    A longer row is sampled uniformly with replacement from a per-thread generator.

    Args:
        sampler_id (int): The sampler's RPC sampling callee id.
        etype (Optional[EdgeType]): The edge type to sample along, or ``None`` on a homogeneous graph.
        ids (torch.Tensor): Global ids of the nodes to sample from, int64 on the CPU.
        req_num (int): Requested neighbors per node.

    Returns:
        NeighborOutput: The flat neighbor ids, the neighbor count per id and the edge ids,
        or ``None`` for ``edge`` when the CSR was registered with ``with_edge=False``.

    Raises:
        RuntimeError: If no CSR is registered for ``sampler_id`` and ``etype``, if ``ids`` is not a 1-D int64 CPU
            tensor, or if the registration wants edge ids but the graph has none.
    """
    return NeighborOutput(
        *_lookup_neighbors_op(sampler_id, edge_type_key(etype), ids, req_num)
    )
