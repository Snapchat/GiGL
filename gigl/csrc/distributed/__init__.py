try:
    from gigl.csrc.distributed.ppr_forward_push import PPRForwardPushState
except ImportError as e:
    raise ImportError(
        "PPR C++ extension not compiled. "
        "Run `make build_cpp_extensions` from the GiGL root to build it."
    ) from e

__all__ = ["PPRForwardPushState"]
