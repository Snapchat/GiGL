"""Hydra composition support for GiGL protobuf YAML configs."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from gigl.common import LocalUri
from gigl.common.omegaconf_resolvers import now_resolver, register_resolvers

_COMPOSE_LOCK = threading.RLock()


def _register_gigl_now_resolver() -> None:
    OmegaConf.register_new_resolver(
        "now",
        now_resolver,
        replace=True,
    )


def compose_yaml_config(uri: LocalUri) -> dict[str, Any]:
    """Compose a YAML config with Hydra using its parent as the config root.

    Args:
        uri: Primary YAML config URI.

    Returns:
        A fully composed and resolved mapping.

    Raises:
        ValueError: If the primary filename is unsupported or the result is not
            a mapping.
        RuntimeError: If another Hydra application owns the global context.
    """
    primary_name = uri.get_basename()
    if not primary_name.endswith(".yaml"):
        raise ValueError(f"Hydra primary config must use the .yaml extension: {uri}")
    config_name = primary_name.rsplit(".", 1)[0]

    config_root = Path(uri.uri).absolute().parent

    with _COMPOSE_LOCK:
        if GlobalHydra.instance().is_initialized():
            raise RuntimeError(
                "GiGL cannot compose a config while another Hydra context is active."
            )
        try:
            with initialize_config_dir(
                config_dir=os.fspath(config_root),
                job_name="gigl_config",
                version_base="1.3",
            ):
                # Hydra installs a one-argument ``now`` resolver during
                # initialization. GiGL's resolver is a backward-compatible
                # superset that also supports offsets.
                register_resolvers()
                _register_gigl_now_resolver()
                composed = compose(config_name=config_name, overrides=[])
                resolved = OmegaConf.to_container(composed, resolve=True)
        finally:
            _register_gigl_now_resolver()

    if not isinstance(resolved, dict):
        raise ValueError(
            f"Hydra config {config_root / config_name} resolved to "
            f"{type(resolved).__name__}, expected a mapping."
        )
    return cast(dict[str, Any], resolved)
