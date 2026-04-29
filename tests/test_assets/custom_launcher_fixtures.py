"""Shared fixtures for ``custom_launcher`` tests.

Lives under ``tests.test_assets`` so a single canonical import path is always
available to ``gigl.common.utils.os_utils.import_obj`` regardless of how the
test harness discovers the file. Test files importing these fixtures see
the same module object (and thus the same call list) as ``import_obj``.
"""

from typing import Any

# Module-level call capture shared across the fake launcher and test code.
FAKE_LAUNCHER_CALLS: list[dict[str, Any]] = []


def fake_launcher_callable(**kwargs: Any) -> None:
    """Fake launcher referenced via its dotted import path in tests."""
    FAKE_LAUNCHER_CALLS.append(kwargs)


NOT_CALLABLE_SENTINEL = 42
