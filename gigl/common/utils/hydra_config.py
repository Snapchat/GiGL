"""Hydra composition support for GiGL protobuf YAML configs."""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer

from gigl.common import HttpUri, LocalUri, Uri
from gigl.common.omegaconf_resolvers import register_resolvers

_COMPOSE_LOCK = threading.RLock()
_DYNAMIC_RESOLVER_INTERPOLATION = re.compile(r"\$\{(?:now|git_hash|oc\.env)(?::|})")
_DYNAMIC_RESOLVERS = ("now", "git_hash", "oc.env")


def contains_dynamic_interpolation(value: Any) -> bool:
    """Return whether YAML contains a resolver that can vary by process."""
    if isinstance(value, str):
        return bool(_DYNAMIC_RESOLVER_INTERPOLATION.search(value))
    if isinstance(value, Mapping):
        return any(
            contains_dynamic_interpolation(key) or contains_dynamic_interpolation(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(contains_dynamic_interpolation(item) for item in value)
    return False


def compose_yaml_config(
    uri: Uri,
    *,
    reject_dynamic_interpolations: bool = False,
) -> dict[str, Any]:
    """Compose a YAML config with Hydra using its parent as the config root.

    Args:
        uri: Primary YAML config URI.
        reject_dynamic_interpolations: Whether to reject process-dependent
            OmegaConf resolvers before Hydra resolves them.

    Returns:
        A fully composed and resolved mapping.

    Raises:
        TypeError: If composition is requested for an unsupported URI.
        ValueError: If the primary filename is unsupported or the result is not
            a mapping.
        RuntimeError: If another Hydra application owns the global context.
    """
    primary_name = uri.get_basename()
    if not primary_name.endswith(".yaml"):
        raise ValueError(f"Hydra primary config must use the .yaml extension: {uri}")
    config_name = primary_name.rsplit(".", 1)[0]

    if isinstance(uri, LocalUri):
        config_root = Path(uri.uri).absolute().parent
        return _compose_local(
            config_root=config_root,
            config_name=config_name,
            reject_dynamic_interpolations=reject_dynamic_interpolations,
        )

    if isinstance(uri, HttpUri):
        raise TypeError("Hydra composition is not supported for HTTP config URIs.")
    raise TypeError(f"Hydra composition is not supported for {type(uri).__name__}.")


def _compose_local(
    config_root: Path,
    config_name: str,
    reject_dynamic_interpolations: bool,
) -> dict[str, Any]:
    with _COMPOSE_LOCK:
        if GlobalHydra.instance().is_initialized():
            raise RuntimeError(
                "GiGL cannot compose a config while another Hydra context is active."
            )
        # Hydra mutates OmegaConf's process-global resolver registry and does not
        # restore it when its context exits. OmegaConf exposes no public way to
        # recover the prior resolver callables, so preserve the affected wrappers
        # directly and restore them in ``finally``.
        prior_resolvers = BaseContainer._resolvers.copy()
        try:
            with initialize_config_dir(
                config_dir=os.fspath(config_root),
                job_name="gigl_config",
                version_base="1.3",
            ):
                # Hydra installs a one-argument ``now`` resolver during
                # initialization. GiGL's resolver is a backward-compatible
                # superset that also supports offsets.
                register_resolvers(replace=True)
                if reject_dynamic_interpolations:
                    for resolver_name in _DYNAMIC_RESOLVERS:
                        OmegaConf.register_new_resolver(
                            resolver_name,
                            _reject_resource_resolver,
                            replace=True,
                        )
                composed = compose(config_name=config_name, overrides=[])
                if reject_dynamic_interpolations and contains_dynamic_interpolation(
                    OmegaConf.to_container(composed, resolve=False)
                ):
                    raise ValueError(
                        "Resource configs cannot contain dynamic OmegaConf resolvers "
                        "because submission and pipeline validation run in separate "
                        "processes."
                    )
                resolved = OmegaConf.to_container(composed, resolve=True)
        finally:
            BaseContainer._resolvers.clear()
            BaseContainer._resolvers.update(prior_resolvers)

    if not isinstance(resolved, dict):
        raise ValueError(
            f"Hydra config {config_root / config_name} resolved to "
            f"{type(resolved).__name__}, expected a mapping."
        )
    return cast(dict[str, Any], resolved)


def _reject_resource_resolver(*_: Any) -> Any:
    raise ValueError(
        "Resource configs cannot use process-dependent OmegaConf resolvers."
    )
