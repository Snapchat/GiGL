try:
    from gigl.csrc.sampling.ppr_forward_push import PPRForwardPushState
except ImportError as e:
    raise ImportError(
        f"Failed to import PPR C++ extension: {e}. "
        "If the extension is not yet compiled, run `make build_cpp_extensions` from the GiGL root. "
        "If it is compiled, ensure `import torch` is called before importing this module "
        "so that libtorch shared libraries are loaded into the process first."
    ) from e

__all__ = ["PPRForwardPushState"]
