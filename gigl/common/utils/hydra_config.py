"""Hydra composition support for GiGL protobuf YAML configs."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import get_gh_backup, restore_gh_from_backup
from omegaconf import OmegaConf

from gigl.common import LocalUri
from gigl.common.omegaconf_resolvers import now_resolver, register_resolvers

_COMPOSE_LOCK = threading.RLock()


def compose_yaml_config(uri: LocalUri) -> dict[str, Any]:
    """Compose a YAML config with Hydra using its parent as the config root.

    A foreign Hydra context (e.g. a user application under ``@hydra.main``)
    is snapshotted before composition and restored afterwards.

    Args:
        uri: Primary YAML config URI.

    Returns:
        A fully composed and resolved mapping.

    Raises:
        ValueError: If the result is not a mapping.
    """
    primary_name = uri.get_basename()
    config_name = primary_name.rsplit(".", 1)[0]

    config_root = Path(uri.uri).absolute().parent

    with _COMPOSE_LOCK:
        # Hydra's compose API owns the process-global GlobalHydra singleton,
        # so the swap below cannot protect user threads composing concurrently
        # outside this lock.
        gh_backup = get_gh_backup()
        GlobalHydra.instance().clear()
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
                OmegaConf.register_new_resolver(
                    "now",
                    now_resolver,
                    replace=True,
                )
                composed = compose(config_name=config_name, overrides=[])
                resolved = OmegaConf.to_container(composed, resolve=True)
        finally:
            OmegaConf.register_new_resolver(
                "now",
                now_resolver,
                replace=True,
            )
            restore_gh_from_backup(gh_backup)

    if not isinstance(resolved, dict):
        raise ValueError(
            f"Hydra config {config_root / config_name} resolved to "
            f"{type(resolved).__name__}, expected a mapping."
        )
    return cast(dict[str, Any], resolved)
