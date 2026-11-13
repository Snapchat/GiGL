import importlib.util
from types import ModuleType
from unittest.mock import patch

from gigl.src.common.constants.local_fs import get_project_root_directory
from tests.test_assets.test_case import TestCase


def _load_script() -> ModuleType:
    # The script lives under .github/scripts, which is not a Python package.
    path = (
        get_project_root_directory()
        / ".github"
        / "scripts"
        / "cancel_ci_gcp_resources.py"
    )
    spec = importlib.util.spec_from_file_location("cancel_ci_gcp_resources", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_script()


class _ScriptedPasses:
    """Stands in for ``cancel_pipeline_runs_with_label``; each call returns or raises the next result."""

    def __init__(self, results: list[list[str] | Exception]) -> None:
        self._results = results
        self.calls = 0

    def __call__(self, **_: str) -> list[str]:
        result = self._results[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


class CancelPipelineRunsUntilQuietTest(TestCase):
    def _run(self, passes: _ScriptedPasses) -> tuple[list[str], list[Exception]]:
        with (
            patch.object(
                _SCRIPT.VertexAIService,
                "cancel_pipeline_runs_with_label",
                side_effect=passes,
            ),
            patch.object(_SCRIPT.time, "sleep"),
        ):
            return _SCRIPT.cancel_pipeline_runs_until_quiet("p", "us-central1", "123")

    def test_keeps_listing_until_a_later_pass_finds_nothing(self) -> None:
        # Pass 1 finds nothing yet, a late submission shows up in pass 2, pass 3 is quiet.
        passes = _ScriptedPasses([[], ["late-run"], [], ["never-reached"]])

        cancelled, errors = self._run(passes)

        self.assertEqual(cancelled, ["late-run"])
        self.assertEqual(errors, [])
        self.assertEqual(passes.calls, 3)

    def test_a_failed_pass_is_reported_and_the_next_pass_still_runs(self) -> None:
        failure = ConnectionError("list failed")
        passes = _ScriptedPasses([failure, ["run-1"], []])

        cancelled, errors = self._run(passes)

        self.assertEqual(cancelled, ["run-1"])
        self.assertEqual(errors, [failure])

    def test_stops_after_the_last_pass_even_if_runs_keep_appearing(self) -> None:
        passes = _ScriptedPasses([[f"run-{i}"] for i in range(10)])

        cancelled, _ = self._run(passes)

        self.assertEqual(passes.calls, _SCRIPT._MAX_DISCOVERY_PASSES)
        self.assertEqual(len(cancelled), _SCRIPT._MAX_DISCOVERY_PASSES)
