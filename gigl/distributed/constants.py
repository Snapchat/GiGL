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
# Two independent mechanisms bound that wait, and they must be set together:
#   * the RPC gather tolerance (below) -- exceeded, a worker dies
#   * ``DistSamplingProducer``'s init barrier -- unbounded, its parent would wedge forever
# A dead worker whose parent wedges is the worst case: the healthy peers block in collectives
# until the training process group's own timeout fires, which is now hours rather than minutes.
# --------------------------------------------------------------------------------------------
SAMPLING_RPC_TIMEOUT_ENV: Final[str] = "GIGL_SAMPLING_RPC_TIMEOUT_SECONDS"
# 600 s is GLT's own default and is preserved so this change alters no existing behaviour.
DEFAULT_SAMPLING_RPC_TIMEOUT_SECONDS: Final[int] = 600

# GLT's ``rpc_timeout`` is not init-only: it becomes the RPC agent's default request timeout for
# EVERY sampling and feature-collection RPC. Raising it to tolerate bring-up skew would therefore
# also mean a genuinely stuck steady-state request takes that long to surface -- trading a fast,
# diagnosable stall for a slow one. So the worker resets the agent's timeout to the value below
# immediately after it passes the init barrier, which is exactly the boundary between the two
# regimes. Default 600 s: identical to what steady-state sampling gets today.
SAMPLING_STEADY_STATE_RPC_TIMEOUT_ENV: Final[str] = (
    "GIGL_SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS"
)
DEFAULT_SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS: Final[int] = 600

SAMPLING_WORKER_INIT_TIMEOUT_ENV: Final[str] = (
    "GIGL_SAMPLING_WORKER_INIT_TIMEOUT_SECONDS"
)
# Above the RPC tolerance: a worker that loses the gather is already dead, so the barrier should
# outlive the gather and report that death rather than time out first and blame the wrong thing.
DEFAULT_SAMPLING_WORKER_INIT_TIMEOUT_SECONDS: Final[int] = 1800


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


def sampling_rpc_timeout_seconds() -> int:
    """Tolerance for GLT's cross-rank sampling-worker RPC gather, in seconds.

    Raise this when ranks can legitimately reach loader construction more than the default apart
    -- e.g. a run whose CSR assembly has a slow single-rank fallback path.
    """
    return _positive_int_from_env(
        SAMPLING_RPC_TIMEOUT_ENV, DEFAULT_SAMPLING_RPC_TIMEOUT_SECONDS
    )


def sampling_steady_state_rpc_timeout_seconds() -> int:
    """Request timeout applied to sampling RPCs once a worker is past its init barrier."""
    return _positive_int_from_env(
        SAMPLING_STEADY_STATE_RPC_TIMEOUT_ENV,
        DEFAULT_SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS,
    )


def sampling_worker_init_timeout_seconds() -> int:
    """Bound on how long a rank waits for its own sampling workers to reach the init barrier."""
    return _positive_int_from_env(
        SAMPLING_WORKER_INIT_TIMEOUT_ENV, DEFAULT_SAMPLING_WORKER_INIT_TIMEOUT_SECONDS
    )
