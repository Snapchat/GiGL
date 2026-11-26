"""``init_rpc`` for sampling workers, with the TensorPipe channels taken from an environment variable.

``init_rpc_with_options`` is a copy of GLT's ``graphlearn_torch.distributed.rpc.init_rpc``.

Differences from GLT's ``init_rpc`` (each one is marked ``# GiGL change:`` in the code):

1. The TensorPipe channels come from the ``channels`` argument, which defaults to ``("basic",)``. GLT always
   uses ``["mpt_uv", "basic"]``.
2. There is no ``is_dynamic`` argument. No caller here needs it; the graph-store ``init_server`` and
   ``init_client``, which do, call GLT's ``init_rpc``.
3. Module state is written as ``glt_rpc._rpc_inited = ...`` instead of with ``global``, because this code
   lives in another module. GLT's helpers (``all_gather``, ``barrier``, ``rpc_is_initialized``) read that
   state.

With ``channels=("mpt_uv", "basic")``, the TensorPipe options, the module state and the side effects are the same
as GLT's.

``GltInitRpcDriftTest`` hashes the source of GLT's ``init_rpc``, ``shutdown_rpc`` and ``rpc_is_initialized``.
A GLT upgrade that changes any of them fails the test; compare the copy with the new source when it does.

The channels come from ``SAMPLING_RPC_CHANNELS_ENV`` in ``gigl/distributed/constants.py``, which describes
each value.

Call sites that read the variable, through ``init_rpc_from_env``:

- The colocated loader worker (``init_neighbor_loader_worker``).
- The dataset-building process (``dataset_factory``).
- The colocated sampling producer (``DistSamplingProducer``).
- The graph-store sampling worker subprocesses (``shared_dist_sampling_producer``).

Call sites that ignore the variable and call GLT's ``init_rpc`` with GLT's channels, ``mpt_uv`` and ``basic``:

- The graph-store ``init_server`` and ``init_client`` (``dist_server`` and ``compute``), which accept
  ``is_dynamic``.
- GLT's own ``init_server`` and ``init_client``.

Processes that read the variable, whose default is ``basic``, and processes that do not, which use ``mpt_uv``
and ``basic``, share ``basic``, so a mixed deployment connects. Set ``GIGL_SAMPLING_RPC_CHANNELS=mpt_uv,basic``
to use GLT's channels in the processes that read the variable.

Where an invalid value fails:

- Before any worker is spawned, in ``DistSamplingProducer.init`` and ``SharedDistSamplingBackend.init_backend``.
  Both read the variable in the parent process first.
- In the ``mp.spawn`` child of the dataset-building process; ``mp.spawn`` re-raises its error.
- At startup of each colocated loader worker.
"""

import os
import time
from typing import Final, Sequence

import graphlearn_torch.distributed.rpc as glt_rpc
from graphlearn_torch.distributed.dist_context import DistRole
from torch.distributed import rpc

from gigl.common.logger import Logger
from gigl.distributed.constants import SAMPLING_RPC_CHANNELS_ENV

logger = Logger()

_TRANSPORTS: Final[tuple[str, ...]] = ("uv",)
DEFAULT_CHANNELS: Final[tuple[str, ...]] = ("basic",)

# Each value is described at SAMPLING_RPC_CHANNELS_ENV in gigl/distributed/constants.py.
_ALLOWED_CHANNELS: Final[frozenset[str]] = frozenset({"basic", "mpt_uv", "cma"})
_REQUIRED_CHANNEL: Final[str] = "basic"


def _validate_channels(channels: tuple[str, ...]) -> None:
    """Checks a channel selection.

    Raises:
        ValueError: If a channel is not allowed or appears twice, or ``basic`` is missing (lists
            that share no channel with a peer abort the process on the first send).
    """
    for channel in channels:
        if channel not in _ALLOWED_CHANNELS:
            raise ValueError(
                f"Channel {channel!r} is not allowed; allowed: {sorted(_ALLOWED_CHANNELS)}"
            )
    if len(set(channels)) != len(channels):
        raise ValueError(f"Channels {list(channels)} must not contain duplicates")
    if _REQUIRED_CHANNEL not in channels:
        raise ValueError(
            f"Channels {list(channels)} must include {_REQUIRED_CHANNEL!r} so that every "
            "process of a job shares at least one channel"
        )


def sampling_rpc_channels_from_env() -> tuple[str, ...]:
    """Returns the channels from the environment, defaulting to ``basic``.

    Raises:
        ValueError: If the variable is set to an invalid value, including an empty list.
    """
    raw = os.environ.get(SAMPLING_RPC_CHANNELS_ENV)
    if raw is None:
        return DEFAULT_CHANNELS
    channels = tuple(item.strip() for item in raw.split(","))
    _validate_channels(channels)
    return channels


# GiGL change: the ``channels`` parameter, and no ``is_dynamic`` parameter.
def init_rpc_with_options(
    master_addr: str,
    master_port: int,
    num_rpc_threads: int = 16,
    rpc_timeout: float = 180,
    channels: Sequence[str] = DEFAULT_CHANNELS,
) -> None:
    """Initializes RPC on the current process like GLT's ``init_rpc``, with chosen channels.

    Args:
        master_addr: Address of the rendezvous master.
        master_port: Port of the rendezvous master.
        num_rpc_threads: Size of the RPC worker thread pool.
        rpc_timeout: RPC timeout in seconds, also used for the bring-up gather and barrier.
        channels: TensorPipe channels in priority order; any of ``basic``, ``mpt_uv``,
            ``cma`` without repeats, and ``basic`` is required.
    """
    with glt_rpc._rpc_init_lock:
        if glt_rpc.rpc_is_initialized() is True:
            return
        if glt_rpc.rpc_is_initialized() is None:
            raise RuntimeError("'init_rpc': Try to re-init rpc after shutdown.")
        # GiGL change: validate the channels, which GLT does not take.
        channels = tuple(channels)
        _validate_channels(channels)

        ctx = glt_rpc.get_context()
        if ctx is None:
            raise RuntimeError("'init_rpc': Distributed context has not been set.")

        options = rpc.TensorPipeRpcBackendOptions(
            _transports=list(_TRANSPORTS),
            # GiGL change: GLT hard-codes ['mpt_uv', 'basic'] here.
            _channels=list(channels),
            num_worker_threads=num_rpc_threads,
            rpc_timeout=rpc_timeout,
            init_method=f"tcp://{master_addr}:{master_port}",
        )

        rpc.init_rpc(
            name=ctx.worker_name,
            rank=ctx.global_rank,
            # GiGL change: GLT passes None here when is_dynamic; that mode is not supported here.
            world_size=ctx.global_world_size,
            rpc_backend_options=options,
        )

        # GiGL change: module state is written through ``glt_rpc`` instead of ``global``.
        glt_rpc._rpc_inited = True

        worker_names: dict[DistRole, list] = {}
        glt_rpc._rpc_worker_names = worker_names

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

        # GiGL change: written through ``glt_rpc``, see above.
        glt_rpc._rpc_current_group_worker_names = set(worker_names[ctx.role])

        glt_rpc.global_barrier(timeout=rpc_timeout)
        time.sleep(1)


def init_rpc_from_env(
    master_addr: str,
    master_port: int,
    num_rpc_threads: int = 16,
    rpc_timeout: float = 180,
) -> None:
    """Calls ``init_rpc_with_options`` with the channels from the environment."""
    channels = sampling_rpc_channels_from_env()
    logger.info(f"Initializing sampling RPC with TensorPipe channels {list(channels)}")
    init_rpc_with_options(
        master_addr=master_addr,
        master_port=master_port,
        num_rpc_threads=num_rpc_threads,
        rpc_timeout=rpc_timeout,
        channels=channels,
    )
