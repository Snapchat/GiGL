"""Detect the execution environment for distributed training processes."""

import os
from enum import Enum


class RuntimeEnv(str, Enum):
    """Supported execution environments.

    RAY: Running inside a Ray cluster.
    VERTEX_AI: Running inside a Vertex AI custom job container.
    UNKNOWN: No signal matched; caller must fall back to defaults.
    """

    RAY = "ray"
    VERTEX_AI = "vertex_ai"
    UNKNOWN = "unknown"


def is_ray_runtime() -> bool:
    """True when the current process is running inside a Ray job.

    **Authoritative signal**: ``GIGL_RAY_RUNTIME=1``. A custom launcher
    integrating GiGL with a Ray platform is expected to set this env var on
    the head spec so the entrypoint can branch reliably. This is the only
    signal GiGL fully controls, so it's checked first.

    **Fallbacks for callers that don't set the authoritative var**: KubeRay
    injects ``RAY_DASHBOARD_ADDRESS`` into the submitter pod (not
    ``RAY_ADDRESS`` — verified against Ray 2.54 docs:
    https://docs.ray.io/en/releases-2.54.0/cluster/kubernetes/getting-started/rayjob-quick-start.html).
    ``RAY_ADDRESS`` is checked last and may or may not be present depending
    on how Ray was started. ``ray.is_initialized()`` returns ``False`` before
    ``ray.init()``, so it only helps if Ray has already been bootstrapped.

    **Callable-ordering constraint**: this is a snapshot of process env, not
    a poll. It MUST be called after the pod's env is populated (always true
    for the entrypoint process, which inherits pod-level env from the
    container start). Repeat callers get the same answer; don't rely on it
    to "become true" mid-process after ``ray.init()`` finishes.
    """
    if os.environ.get("GIGL_RAY_RUNTIME") == "1":
        return True
    if os.environ.get("RAY_DASHBOARD_ADDRESS"):
        return True
    if os.environ.get("RAY_ADDRESS"):
        return True
    try:
        import ray  # type: ignore[import-not-found]

        return ray.is_initialized()
    except ImportError:
        return False


def get_runtime_env() -> RuntimeEnv:
    """Classify the current process's execution environment.

    Returns:
        ``RuntimeEnv.RAY`` when :func:`is_ray_runtime` is True.
        ``RuntimeEnv.VERTEX_AI`` when Vertex AI env vars
        (``CLOUD_ML_JOB_ID`` or ``AIP_MODEL_DIR``) are set.
        ``RuntimeEnv.UNKNOWN`` otherwise.
    """
    if is_ray_runtime():
        return RuntimeEnv.RAY
    if os.environ.get("CLOUD_ML_JOB_ID") or os.environ.get("AIP_MODEL_DIR"):
        return RuntimeEnv.VERTEX_AI
    return RuntimeEnv.UNKNOWN
