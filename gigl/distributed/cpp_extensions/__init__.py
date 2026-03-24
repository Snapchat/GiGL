try:
    from gigl.distributed.cpp_extensions.ppr_forward_push import PPRForwardPushState
except ImportError as e:
    raise ImportError(
        "PPR C++ extension not compiled. "
        "Run `uv pip install -e .` from the GiGL root to build it."
    ) from e

__all__ = ["PPRForwardPushState"]
