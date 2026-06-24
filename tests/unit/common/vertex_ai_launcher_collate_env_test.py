"""Unit test for the generic GIGL_COLLATE_IMPL launcher-env passthrough.

The launcher submits a Vertex AI custom job; worker containers do not inherit
the launcher's process env, so any env var the worker must see has to be added
to the job spec explicitly. This test pins that the passthrough helper forwards
GIGL_COLLATE_IMPL when (and only when) it is set in the launcher environment.
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
        with mock.patch.dict(
            os.environ, {"GIGL_COLLATE_IMPL": "cpp"}, clear=True
        ):
            out = _collate_impl_passthrough_env_vars()
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].name, "GIGL_COLLATE_IMPL")
            self.assertEqual(out[0].value, "cpp")
