"""Lets profilers such as py-spy attach to sampling workers under Yama ptrace scope 1."""

import ctypes
import sys

from gigl.common.logger import Logger

logger = Logger()

_PR_SET_PTRACER = 0x59616D61
_PR_SET_PTRACER_ANY = -1


def allow_any_ptracer() -> bool:
    """Lets any process ptrace the calling process.

    Under Yama ``ptrace_scope=1`` only ancestors may attach to a process, so a profiler started
    from a shell cannot attach to a spawned sampling worker. ``prctl(PR_SET_PTRACER,
    PR_SET_PTRACER_ANY)`` lifts that restriction for the calling process.

    Returns:
        True if the setting was applied; False on non-Linux platforms or if ``prctl`` fails.
    """
    if not sys.platform.startswith("linux"):
        logger.info(
            "PR_SET_PTRACER is Linux-only; leaving ptrace permissions unchanged"
        )
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        libc.prctl.restype = ctypes.c_int
        result = libc.prctl(
            _PR_SET_PTRACER, ctypes.c_ulong(_PR_SET_PTRACER_ANY), 0, 0, 0
        )
    except OSError as e:
        logger.info(f"Could not call prctl(PR_SET_PTRACER): {e}")
        return False
    if result != 0:
        logger.info(f"prctl(PR_SET_PTRACER) failed with errno {ctypes.get_errno()}")
        return False
    return True
