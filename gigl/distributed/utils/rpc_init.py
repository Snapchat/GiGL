"""``init_rpc`` for sampling workers with selectable TensorPipe transports and channels.

``init_rpc_with_options`` is a copy of ``graphlearn_torch.distributed.rpc.init_rpc`` that differs
only in where the TensorPipe transports and channels come from. Its defaults equal GLT's, so with
no overrides the options, the module state and the side effects are the same. A unit test hashes
GLT's ``init_rpc`` source so that a GLT upgrade that changes it is noticed and the copy is revisited.

All module state is written through the ``graphlearn_torch.distributed.rpc`` module object so that
GLT's own helpers (``all_gather``, ``barrier``, ``rpc_is_initialized``, ...) see it. As in GLT,
``_rpc_master_addr`` and ``_rpc_master_port`` are never set; the master address and port only
form the TensorPipe ``init_method``.

Environment variables (comma-separated lists, unset means GLT's defaults):
    GIGL_SAMPLING_RPC_CHANNELS: any of ``basic``, ``mpt_uv``, ``cma``; ``basic`` is required.
        Every valid list therefore shares ``basic``, the CPU channel that carries tensors across
        hosts, with every other valid list. Order is TensorPipe's channel priority, first is
        highest. The value must be identical on every node of a job: two processes that pick
        channels with no common choice abort on the first request. It has no effect in graph-store
        mode, whose RPC is initialized by GLT's own ``init_rpc``.
    GIGL_SAMPLING_RPC_TRANSPORTS: ``uv`` only.
"""

import logging
import os
import time
from typing import Optional, Sequence

import graphlearn_torch.distributed.rpc as glt_rpc
from graphlearn_torch.distributed.dist_context import DistRole
from torch.distributed import rpc

SAMPLING_RPC_CHANNELS_ENV = "GIGL_SAMPLING_RPC_CHANNELS"
SAMPLING_RPC_TRANSPORTS_ENV = "GIGL_SAMPLING_RPC_TRANSPORTS"

DEFAULT_TRANSPORTS: tuple[str, ...] = ("uv",)
DEFAULT_CHANNELS: tuple[str, ...] = ("mpt_uv", "basic")

_ALLOWED_TRANSPORTS = frozenset({"uv"})
_ALLOWED_CHANNELS = frozenset({"basic", "mpt_uv", "cma"})
_REQUIRED_CHANNEL = "basic"


def _parse_list(env_name: str, allowed: frozenset[str]) -> Optional[tuple[str, ...]]:
    raw = os.environ.get(env_name)
    if raw is None:
        return None
    values = tuple(item.strip() for item in raw.split(","))
    for value in values:
        if value not in allowed:
            raise ValueError(
                f"{env_name}={raw!r} contains {value!r}; allowed values are {sorted(allowed)}"
            )
    return values


def validate_rpc_backend(
    transports: Sequence[str], channels: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validates a transport and channel selection and returns it as tuples.

    Raises:
        ValueError: If a transport or channel is not allowed, or ``basic`` is missing from the
            channels (lists that share no channel with a peer abort the process on the first
            send).
    """
    transports = tuple(transports)
    channels = tuple(channels)
    for transport in transports:
        if transport not in _ALLOWED_TRANSPORTS:
            raise ValueError(
                f"Transport {transport!r} is not allowed; allowed: {sorted(_ALLOWED_TRANSPORTS)}"
            )
    if not transports:
        raise ValueError("At least one RPC transport is required")
    for channel in channels:
        if channel not in _ALLOWED_CHANNELS:
            raise ValueError(
                f"Channel {channel!r} is not allowed; allowed: {sorted(_ALLOWED_CHANNELS)}"
            )
    if _REQUIRED_CHANNEL not in channels:
        raise ValueError(
            f"Channels {list(channels)} must include {_REQUIRED_CHANNEL!r} so that every "
            "process of a job shares at least one channel"
        )
    return transports, channels


def sampling_rpc_backend_from_env() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Returns (transports, channels) from the environment, defaulting to GLT's values.

    Raises:
        ValueError: If a variable is set to an invalid value, including an empty list.
    """
    transports = _parse_list(SAMPLING_RPC_TRANSPORTS_ENV, _ALLOWED_TRANSPORTS)
    channels = _parse_list(SAMPLING_RPC_CHANNELS_ENV, _ALLOWED_CHANNELS)
    if transports is None:
        transports = DEFAULT_TRANSPORTS
    if channels is None:
        channels = DEFAULT_CHANNELS
    return validate_rpc_backend(transports, channels)


def init_rpc_with_options(
    master_addr: str,
    master_port: int,
    num_rpc_threads: int = 16,
    rpc_timeout: float = 180,
    is_dynamic: bool = False,
    _transports: Sequence[str] = DEFAULT_TRANSPORTS,
    _channels: Sequence[str] = DEFAULT_CHANNELS,
) -> None:
    """Initializes RPC on the current process like GLT's ``init_rpc``, with chosen channels.

    Args:
        master_addr: Address of the rendezvous master.
        master_port: Port of the rendezvous master.
        num_rpc_threads: Size of the RPC worker thread pool.
        rpc_timeout: RPC timeout in seconds, also used for the bring-up gather and barrier.
        is_dynamic: Whether servers and clients join dynamically.
        _transports: TensorPipe transports; only ``uv`` is accepted.
        _channels: TensorPipe channels in priority order; any of ``basic``, ``mpt_uv``,
            ``cma``, and ``basic`` is required.
    """
    transports, channels = validate_rpc_backend(_transports, _channels)
    with glt_rpc._rpc_init_lock:
        if glt_rpc.rpc_is_initialized() is True:
            return
        if glt_rpc.rpc_is_initialized() is None:
            raise RuntimeError("'init_rpc': Try to re-init rpc after shutdown.")

        ctx = glt_rpc.get_context()
        if ctx is None:
            raise RuntimeError("'init_rpc': Distributed context has not been set.")

        options = rpc.TensorPipeRpcBackendOptions(
            _transports=list(transports),
            _channels=list(channels),
            num_worker_threads=num_rpc_threads,
            rpc_timeout=rpc_timeout,
            init_method=f"tcp://{master_addr}:{master_port}",
        )

        world_size: Optional[int] = ctx.global_world_size
        if is_dynamic:
            world_size = None
        rpc.init_rpc(
            name=ctx.worker_name,
            rank=ctx.global_rank,
            world_size=world_size,
            rpc_backend_options=options,
        )

        glt_rpc._rpc_inited = True

        worker_names: dict[DistRole, list] = {}
        glt_rpc._rpc_worker_names = worker_names

        if is_dynamic:
            worker_names[DistRole.SERVER] = []
            worker_names[DistRole.CLIENT] = []

            if ctx.is_server():
                # ensure all servers is inited
                for server_rank in range(ctx.world_size):
                    if server_rank == ctx.rank:
                        worker_names[DistRole.SERVER].append(
                            ctx.group_name + "_" + str(server_rank)
                        )
                        continue
                    times = 0
                    is_avail = False
                    while not is_avail:
                        try:
                            is_avail = glt_rpc.rpc_global_request_by_rank(
                                server_rank, rpc.is_available
                            )
                        except Exception:
                            time.sleep(glt_rpc.SERVER_INIT_CHECK_INTERVAL)
                            logging.info(
                                f"RETRY {times}: server {ctx.rank} waits server {server_rank}..."
                            )
                        times += 1
                        if times >= glt_rpc.MAX_RETYR_TIMES:
                            raise RuntimeError(
                                f"TIMEOUT: server {ctx.rank} waits server {server_rank} timeout."
                                f"Check if server {server_rank} is ready."
                            )
                    worker_names[DistRole.SERVER].append(
                        ctx.group_name + "_" + str(server_rank)
                    )
                glt_rpc._rpc_current_group_worker_names = set(
                    worker_names[DistRole.SERVER]
                )
                return
            if ctx.is_client():
                for server_rank in range(ctx.global_rank - ctx.rank):
                    times = 0
                    is_avail = False
                    while not is_avail:
                        try:
                            is_avail = glt_rpc.rpc_global_request_by_rank(
                                server_rank, rpc.is_available
                            )
                        except Exception:
                            time.sleep(glt_rpc.SERVER_INIT_CHECK_INTERVAL)
                            logging.info(
                                f"RETRY {times}: client {ctx.rank} waits server {server_rank}..."
                            )
                        times += 1
                        if times >= glt_rpc.MAX_RETYR_TIMES:
                            raise RuntimeError(
                                f"TIMEOUT: client {ctx.rank} waits server {server_rank} timeout."
                                f"Check if server {server_rank} is ready."
                            )
                    server_name = glt_rpc.rpc_global_request_by_rank(
                        server_rank, rpc.get_worker_info
                    ).name
                    worker_names[DistRole.SERVER].append(server_name)
                glt_rpc._rpc_current_group_worker_names = set(
                    [
                        ctx.group_name + "_" + str(client_rank)
                        for client_rank in range(ctx.world_size)
                    ]
                )
                return

        gathered_results = glt_rpc.global_all_gather(
            obj=(ctx.role, ctx.world_size, ctx.rank), timeout=rpc_timeout
        )
        for worker_name, (role, role_size, role_rank) in gathered_results.items():
            worker_list = worker_names.get(role, None)
            if worker_list is None:
                worker_list = [None for _ in range(role_size)]
            else:
                if len(worker_list) != role_size:
                    raise RuntimeError(
                        f"'init_rpc': world size of role {role} gathered "
                        f"from {worker_name} is inconsistent with others."
                    )
            if worker_list[role_rank] is not None:
                raise RuntimeError(
                    f"'init_rpc': try to set worker name twice with "
                    f"the same rank {role_rank} of role {role}"
                )
            worker_list[role_rank] = worker_name
            worker_names[role] = worker_list

        glt_rpc._rpc_current_group_worker_names = set(worker_names[ctx.role])

        glt_rpc.global_barrier(timeout=rpc_timeout)
        time.sleep(1)


def init_rpc_from_env(
    master_addr: str,
    master_port: int,
    num_rpc_threads: int = 16,
    rpc_timeout: float = 180,
    is_dynamic: bool = False,
) -> None:
    """Calls ``init_rpc_with_options`` with the transports and channels from the environment."""
    transports, channels = sampling_rpc_backend_from_env()
    init_rpc_with_options(
        master_addr=master_addr,
        master_port=master_port,
        num_rpc_threads=num_rpc_threads,
        rpc_timeout=rpc_timeout,
        is_dynamic=is_dynamic,
        _transports=transports,
        _channels=channels,
    )
