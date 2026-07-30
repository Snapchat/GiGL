"""Hydra composition support for GiGL protobuf YAML configs."""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer

from gigl.common import HttpUri, LocalUri, Uri
from gigl.common.omegaconf_resolvers import register_resolvers

_COMPOSE_LOCK = threading.RLock()
_MAX_CONFIG_FILE_COUNT = 1_000
_MAX_CONFIG_TOTAL_BYTES = 50 * 1024 * 1024
_YAML_SUFFIXES = (".yaml", ".yml")
_RESOLVER_INTERPOLATION = re.compile(r"\$\{[^{}]+:")
_HYDRA_MUTATED_RESOLVERS = (
    "now",
    "hydra",
    "python_version",
    "git_hash",
    "oc.env",
)


def contains_dynamic_interpolation(value: Any) -> bool:
    """Return whether YAML contains a resolver that can vary by process."""
    if isinstance(value, str):
        return bool(_RESOLVER_INTERPOLATION.search(value)) or "${git_hash}" in value
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
    reject_interpolations: bool = False,
) -> dict[str, Any]:
    """Compose a YAML config with Hydra using its parent as the config root.

    Args:
        uri: Primary YAML config URI.
        reject_interpolations: Whether to reject OmegaConf interpolations before
            Hydra resolves them.

    Returns:
        A fully composed and resolved mapping.

    Raises:
        TypeError: If composition is requested for an unsupported URI.
        ValueError: If the bundle is unsafe or resolves to a non-mapping value.
        RuntimeError: If another Hydra application owns the global context.
    """
    primary_name = uri.get_basename()
    if not primary_name.endswith(".yaml"):
        raise ValueError(f"Hydra primary config must use the .yaml extension: {uri}")
    config_name = primary_name.rsplit(".", 1)[0]

    if isinstance(uri, LocalUri):
        config_root = Path(uri.uri).absolute().parent
        _validate_local_bundle(config_root=config_root)
        return _compose_local(
            config_root=config_root,
            config_name=config_name,
            reject_interpolations=reject_interpolations,
        )

    if isinstance(uri, HttpUri):
        raise TypeError("Hydra composition is not supported for HTTP config URIs.")
    raise TypeError(f"Hydra composition is not supported for {type(uri).__name__}.")


def _compose_local(
    config_root: Path,
    config_name: str,
    reject_interpolations: bool,
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
        prior_resolvers = {
            name: BaseContainer._resolvers.get(name)
            for name in _HYDRA_MUTATED_RESOLVERS
        }
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
                if reject_interpolations:
                    for resolver_name in ("now", "git_hash", "oc.env"):
                        if resolver_name in BaseContainer._resolvers:
                            BaseContainer._resolvers[resolver_name] = (
                                _reject_resource_resolver
                            )
                composed = compose(config_name=config_name, overrides=[])
                if reject_interpolations and contains_dynamic_interpolation(
                    OmegaConf.to_container(composed, resolve=False)
                ):
                    raise ValueError(
                        "Resource configs cannot contain dynamic OmegaConf resolvers "
                        "because submission and pipeline validation run in separate "
                        "processes."
                    )
                resolved = OmegaConf.to_container(composed, resolve=True)
        finally:
            for name, resolver in prior_resolvers.items():
                if resolver is None:
                    BaseContainer._resolvers.pop(name, None)
                else:
                    BaseContainer._resolvers[name] = resolver

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


def _validate_local_bundle(config_root: Path) -> None:
    resolved_root = config_root.resolve()
    yaml_file_count = 0
    total_bytes = 0
    for yaml_path in config_root.rglob("*"):
        if yaml_path.is_symlink():
            raise ValueError(f"Config bundle {config_root} contains a symlink.")
        if not yaml_path.is_file() or yaml_path.suffix not in _YAML_SUFFIXES:
            continue
        yaml_file_count += 1
        if yaml_file_count > _MAX_CONFIG_FILE_COUNT:
            raise ValueError(
                f"Config bundle {config_root} contains more than "
                f"{_MAX_CONFIG_FILE_COUNT} YAML files."
            )
        total_bytes += yaml_path.stat().st_size
        if total_bytes > _MAX_CONFIG_TOTAL_BYTES:
            raise ValueError(
                f"Config bundle {config_root} exceeds the "
                f"{_MAX_CONFIG_TOTAL_BYTES}-byte limit."
            )
        if not yaml_path.resolve().is_relative_to(resolved_root):
            raise ValueError(f"Config bundle {config_root} contains an unsafe path.")
        with yaml_path.open("r") as file:
            raw_data = yaml.safe_load(file)
        _validate_hydra_controls(raw_data=raw_data, source_path=yaml_path)


def _validate_hydra_controls(raw_data: Any, source_path: Path) -> None:
    if not isinstance(raw_data, Mapping):
        return
    hydra_config = raw_data.get("hydra")
    if isinstance(hydra_config, Mapping) and "searchpath" in hydra_config:
        raise ValueError(
            f"External Hydra search paths are not supported in {source_path}."
        )
    defaults = raw_data.get("defaults")
    if not isinstance(defaults, Sequence) or isinstance(defaults, (str, bytes)):
        return
    for default in defaults:
        selectors: list[str] = []
        if isinstance(default, str):
            selectors.append(default)
        elif isinstance(default, Mapping):
            selectors.extend(str(key) for key in default)
            selectors.extend(
                str(value) for value in default.values() if isinstance(value, str)
            )
        for selector in selectors:
            if "${" in selector:
                raise ValueError(
                    f"Interpolations in Hydra defaults are not supported in "
                    f"{source_path}."
                )
            selector_path = selector.split("@", 1)[0]
            selector_parts = selector_path.split()
            while selector_parts and selector_parts[0] in {"optional", "override"}:
                selector_parts.pop(0)
            selector_path = selector_parts[-1] if selector_parts else ""
            if ".." in PurePosixPath(selector_path).parts:
                raise ValueError(
                    f"Parent traversal in Hydra defaults is not supported in "
                    f"{source_path}."
                )
