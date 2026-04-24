try:
    from gigl_core.ppr_forward_push import PPRForwardPushState
except ImportError as e:
    raise ImportError(
        f"Failed to import PPR C++ extension: {e}. "
        "If the extension is not yet compiled, run `uv pip install -e gigl-core/` from the GiGL root. "
        "If it is compiled, ensure `import torch` is called before importing this module "
        "so that libtorch shared libraries are loaded into the process first."
    ) from e

__all__ = ["PPRForwardPushState"]
