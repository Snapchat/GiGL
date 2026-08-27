import os
from typing import Final

from gigl.common.logger import Logger

logger = Logger()

# TODO (mkolodner-sc): Set these ports dynamically while ensuring no overlap
# Ports for various purposes, we need to make sure they do not overlap.
# Note that [master_port_for_inference, master_port_for_inference + num_inference_processes).
# ports are used. Same for master port for sampling.
DEFAULT_MASTER_INFERENCE_PORT = 10_000
DEFAULT_MASTER_SAMPLING_PORT = 20_000
DEFAULT_MASTER_DATA_BUILDING_PORT = 30_000

# --------------------------------------------------------------------------------------------
# Loader bring-up skew tolerance.
#
# Constructing a distributed loader is a cluster-wide rendezvous: every sampling worker on every
# rank must complete GLT's ``init_rpc``, which gathers all ``num_workers x world_size`` of them,
# and only then does each rank's parent pass its local init barrier. NOTHING synchronises ranks
# between the end of dataset building and the FIRST loader construction (the barriers in the
# applied trainers come *after* each loader), so any skew accumulated during partitioning or CSR
# assembly is spent here.
#
# Two mechanisms bound that wait; both derive from the single env var below so they cannot be
# mis-set relative to each other:
#   * the RPC gather tolerance -- exceeded, a worker dies
#   * ``DistSamplingProducer``'s init barrier, bounded at a multiple of the gather tolerance --
#     previously unbounded, so a dead worker left its parent wedged forever while healthy peers
#     blocked in collectives until the training process group's own timeout fired.
# --------------------------------------------------------------------------------------------
SAMPLING_RPC_INIT_TIMEOUT_ENV: Final[str] = "GIGL_SAMPLING_RPC_INIT_TIMEOUT_SECONDS"
# GLT's own default
DEFAULT_SAMPLING_RPC_INIT_TIMEOUT_SECONDS: Final[int] = 600

# GLT's ``rpc_timeout`` is not init-only: it becomes the RPC agent's default request timeout for
# EVERY sampling and feature-collection RPC. Raising it to tolerate bring-up skew would therefore
# also mean a genuinely stuck steady-state request takes that long to surface -- trading a fast,
# diagnosable stall for a slow one. So the worker resets the agent's timeout back to this value
# (GLT's default) immediately after it passes the init barrier, which is exactly the boundary
# between the two regimes. Deliberately not configurable: steady-state sampling keeps today's
# behaviour no matter how far bring-up tolerance is raised.
SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS: Final[int] = 600

# The init barrier must outlive the RPC gather: a worker that loses the gather is already dead,
# and the barrier should report that death rather than time out first and blame the wrong thing.
# 3x leaves margin for post-gather worker setup, and 3 x 600 reproduces the previous 1800 s bound.
SAMPLING_WORKER_INIT_TIMEOUT_MULTIPLIER: Final[int] = 3


def _positive_int_from_env(name: str, default: int) -> int:
    """Read a positive integer from the environment, warning and defaulting on anything else."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not an integer; using {default}")
        return default
    if parsed <= 0:
        logger.warning(f"{name}={parsed} is not positive; using {default}")
        return default
    return parsed


def sampling_rpc_init_timeout_seconds() -> int:
    """Tolerance for GLT's cross-rank sampling-worker RPC gather, in seconds.

    Raise this when ranks can legitimately reach loader construction more than the default apart
    -- e.g. a run whose CSR assembly has a slow single-rank fallback path.
    """
    return _positive_int_from_env(
        SAMPLING_RPC_INIT_TIMEOUT_ENV, DEFAULT_SAMPLING_RPC_INIT_TIMEOUT_SECONDS
    )


def sampling_worker_init_timeout_seconds() -> int:
    """Bound on how long a rank waits for its own sampling workers to reach the init barrier.

    Derived from the gather tolerance rather than configured separately, so raising
    ``GIGL_SAMPLING_RPC_INIT_TIMEOUT_SECONDS`` moves both bounds coherently.
    """
    return SAMPLING_WORKER_INIT_TIMEOUT_MULTIPLIER * sampling_rpc_init_timeout_seconds()
