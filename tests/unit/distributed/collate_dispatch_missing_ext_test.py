"""Regression tests: _collate_dispatch must not require collate_core at import time.

When the installed ``gigl_core`` wheel predates the C++ extension work,
``gigl_core.collate_core`` does not exist. This file asserts that:

1. Importing ``gigl.distributed._collate_dispatch`` succeeds without
   ``collate_core`` present.
2. The ``python`` and ``vectorized`` dispatch paths work without
   ``collate_core``.
3. The ``cpp`` path raises ``ImportError`` only when the cpp function is
   actually called.
"""

import contextlib
import sys
import unittest.mock
from collections.abc import Generator

import torch

from gigl.distributed import _collate_dispatch as cd
from gigl.distributed._collate_dispatch import resolve_collate_impl
from tests.test_assets.test_case import TestCase


@contextlib.contextmanager
def _hide_collate_core() -> Generator[None, None, None]:
    """Context manager that makes ``gigl_core.collate_core`` unimportable.

    Patches both ``sys.modules["gigl_core.collate_core"]`` to ``None`` (so a
    bare ``import gigl_core.collate_core`` raises) AND removes the attribute
    from the already-cached ``gigl_core`` module object (so
    ``from gigl_core import collate_core`` also raises even when
    ``gigl_core`` itself is already in ``sys.modules``).
    """
    import gigl_core as _gigl_core_mod

    sentinel = object()
    original = _gigl_core_mod.__dict__.pop("collate_core", sentinel)
    with unittest.mock.patch.dict(sys.modules, {"gigl_core.collate_core": None}):
        try:
            yield
        finally:
            if original is not sentinel:
                _gigl_core_mod.collate_core = original  # type: ignore[attr-defined]


class TestCollateCoreNotInstalled(TestCase):
    """Guard that collate_core is not required outside the cpp dispatch path."""

    def test_import_collate_dispatch_does_not_require_collate_core(self) -> None:
        """Importing _collate_dispatch must not touch collate_core.

        The module-level code in ``_collate_dispatch`` must not import
        ``collate_core``. This test simulates the extension being absent and
        asserts that a reload of the module succeeds without raising
        ``ImportError``.
        """
        import importlib

        import gigl.distributed._collate_dispatch as _mod

        with _hide_collate_core():
            # Reload forces re-execution of module-level code under the patch.
            # If the module-level import was still there this would raise.
            importlib.reload(_mod)

    def test_python_and_vectorized_dispatch_work_without_collate_core(self) -> None:
        """resolve_collate_impl for python/vectorized must not touch collate_core.

        ``resolve_collate_impl`` only reads an env-var; it must not require the
        C++ extension. Similarly, ``assemble_homogeneous`` must work without
        collate_core.
        """
        with _hide_collate_core():
            import os

            with unittest.mock.patch.dict(os.environ, {"GIGL_COLLATE_IMPL": "python"}):
                impl = resolve_collate_impl()
                self.assertEqual(impl, "python")

            with unittest.mock.patch.dict(
                os.environ, {"GIGL_COLLATE_IMPL": "vectorized"}
            ):
                impl = resolve_collate_impl()
                self.assertEqual(impl, "vectorized")

            # assemble_homogeneous / assemble_heterogeneous never call collate_core.
            components = {
                "node": torch.tensor([1, 2, 3]),
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "edge": None,
                "x": None,
                "edge_attr": None,
                "batch": torch.tensor([1, 2]),
                "num_sampled_nodes": torch.tensor([2, 1]),
                "num_sampled_edges": torch.tensor([2]),
            }
            data = cd.assemble_homogeneous(components)
            self.assertIsNotNone(data)

    def test_cpp_dispatch_raises_import_error_without_collate_core(self) -> None:
        """Calling collate_cpp_homogeneous must raise ImportError when collate_core is absent.

        The lazy import inside ``collate_cpp_homogeneous`` should propagate the
        ``ImportError`` from the missing extension. This preserves a clear error
        message instead of a silent failure.
        """
        msg = {
            "ids": torch.tensor([10, 11, 12]),
            "rows": torch.tensor([0, 1]),
            "cols": torch.tensor([1, 2]),
            "num_sampled_nodes": torch.tensor([2, 1]),
            "num_sampled_edges": torch.tensor([2]),
        }
        with _hide_collate_core():
            with self.assertRaises(ImportError):
                cd.collate_cpp_homogeneous(
                    msg,
                    batch_size=0,
                    has_batch=False,
                    to_device=torch.device("cpu"),
                )
