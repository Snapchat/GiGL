"""Unit tests for the generic launcher-env passthrough of loader-selection vars.

The launcher submits a Vertex AI custom job; worker containers do not inherit
the launcher's process env, so any env var the worker must see has to be added
to the job spec explicitly. This test pins that the passthrough helper forwards
GIGL_COLLATE_IMPL and GIGL_ABLP_LABEL_FORMAT when (and only when) they are set
in the launcher environment.
"""

import os
from unittest import mock

from gigl.src.common.vertex_ai_launcher import _collate_impl_passthrough_env_vars
from tests.test_assets.test_case import TestCase


class TestCollateImplPassthroughEnv(TestCase):
    def test_absent_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_collate_impl_passthrough_env_vars(), [])

    def test_absent_when_empty(self) -> None:
        with mock.patch.dict(os.environ, {"GIGL_COLLATE_IMPL": ""}, clear=True):
            self.assertEqual(_collate_impl_passthrough_env_vars(), [])

    def test_forwarded_when_set(self) -> None:
        with mock.patch.dict(os.environ, {"GIGL_COLLATE_IMPL": "cpp"}, clear=True):
            out = _collate_impl_passthrough_env_vars()
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].name, "GIGL_COLLATE_IMPL")
            self.assertEqual(out[0].value, "cpp")


class TestAblpLabelFormatPassthroughEnv(TestCase):
    def test_absent_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            names = [e.name for e in _collate_impl_passthrough_env_vars()]
            self.assertNotIn("GIGL_ABLP_LABEL_FORMAT", names)

    def test_absent_when_empty(self) -> None:
        with mock.patch.dict(
            os.environ, {"GIGL_ABLP_LABEL_FORMAT": ""}, clear=True
        ):
            names = [e.name for e in _collate_impl_passthrough_env_vars()]
            self.assertNotIn("GIGL_ABLP_LABEL_FORMAT", names)

    def test_forwarded_when_set(self) -> None:
        with mock.patch.dict(
            os.environ, {"GIGL_ABLP_LABEL_FORMAT": "edge_list"}, clear=True
        ):
            out = _collate_impl_passthrough_env_vars()
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].name, "GIGL_ABLP_LABEL_FORMAT")
            self.assertEqual(out[0].value, "edge_list")

    def test_both_forwarded_when_both_set(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"GIGL_COLLATE_IMPL": "cpp", "GIGL_ABLP_LABEL_FORMAT": "edge_list"},
            clear=True,
        ):
            out = _collate_impl_passthrough_env_vars()
            names = {e.name: e.value for e in out}
            self.assertEqual(names.get("GIGL_COLLATE_IMPL"), "cpp")
            self.assertEqual(names.get("GIGL_ABLP_LABEL_FORMAT"), "edge_list")
