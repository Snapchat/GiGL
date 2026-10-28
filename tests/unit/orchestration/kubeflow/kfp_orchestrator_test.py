import threading
import time
from typing import Callable, Optional
from unittest.mock import ANY, patch

from absl.testing import absltest

from gigl.common import GcsUri
from gigl.common.logger import Logger
from gigl.orchestration.kubeflow.kfp_orchestrator import KfpOrchestrator
from tests.test_assets.test_case import TestCase

logger = Logger()

_SERVICE = "gigl.orchestration.kubeflow.kfp_orchestrator.VertexAIService"


class _FakeRuns:
    """Scripts ``wait_for_run_completion`` and ``cancel_pipeline_run`` per run name.

    A run's script is a callable that returns (success) or raises (failure).
    ``blocked`` runs wait on an event that a cancel of that run sets.
    """

    def __init__(self) -> None:
        self.scripts: dict[str, Callable[[], None]] = {}
        self.events: dict[str, threading.Event] = {}
        self.cancelled: list[str] = []
        self.cancel_errors: dict[str, Exception] = {}

    def fails(self, name: str, delay_s: float = 0.0) -> None:
        def script() -> None:
            time.sleep(delay_s)
            raise RuntimeError(f"run {name} failed")

        self.scripts[name] = script

    def succeeds(self, name: str) -> None:
        self.scripts[name] = lambda: None

    def blocked_until_cancelled(self, name: str) -> None:
        event = threading.Event()
        self.events[name] = event

        def script() -> None:
            if not event.wait(timeout=5):
                raise AssertionError(f"run {name} was never cancelled")
            raise RuntimeError(f"run {name} stopped with status: CANCELLED")

        self.scripts[name] = script

    def wait_for_run_completion(self, resource_name: str, **_: object) -> None:
        self.scripts[resource_name]()

    def cancel_raises(self, name: str) -> None:
        self.cancel_errors[name] = RuntimeError(f"cancel of {name} denied")

    def cancel_pipeline_run(self, resource_name: str) -> bool:
        self.cancelled.append(resource_name)
        if resource_name in self.cancel_errors:
            raise self.cancel_errors[resource_name]
        event: Optional[threading.Event] = self.events.get(resource_name)
        if event is not None:
            event.set()
        return True


class WaitForCompletionTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.runs = _FakeRuns()
        for method in ("wait_for_run_completion", "cancel_pipeline_run"):
            patcher = patch(
                f"{_SERVICE}.{method}", side_effect=getattr(self.runs, method)
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        self.orchestrator = KfpOrchestrator()

    def test_fail_fast_cancels_runs_still_waiting(self) -> None:
        self.runs.fails("A")
        self.runs.blocked_until_cancelled("B")

        with self.assertRaisesRegex(RuntimeError, "A"):
            self.orchestrator.wait_for_completion(["A", "B"], fail_fast=True)

        self.assertEqual(self.runs.cancelled, ["B"])

    def test_fail_fast_keeps_cancelling_when_one_cancel_raises(self) -> None:
        self.runs.fails("A")
        # B's cancel is denied; B then ends on its own, as a real run would.
        self.runs.fails("B", delay_s=0.1)
        self.runs.cancel_raises("B")
        self.runs.blocked_until_cancelled("C")

        with self.assertRaises(RuntimeError) as raised:
            self.orchestrator.wait_for_completion(["A", "B", "C"], fail_fast=True)

        self.assertCountEqual(self.runs.cancelled, ["B", "C"])
        message = str(raised.exception)
        self.assertIn("run A failed", message)
        self.assertIn("cancel of B denied", message)
        self.assertNotIn("CANCELLED", message)

    def test_fail_fast_with_the_last_run_failing_cancels_nothing(self) -> None:
        self.runs.succeeds("A")
        self.runs.succeeds("B")
        self.runs.fails("C", delay_s=0.05)

        with self.assertRaisesRegex(RuntimeError, "C"):
            self.orchestrator.wait_for_completion(["A", "B", "C"], fail_fast=True)

        self.assertEqual(self.runs.cancelled, [])

    def test_without_fail_fast_waits_for_all_and_names_every_failure(self) -> None:
        self.runs.fails("A")
        self.runs.fails("B", delay_s=0.05)
        self.runs.succeeds("C")

        with self.assertRaises(RuntimeError) as raised:
            self.orchestrator.wait_for_completion(["A", "B", "C"])

        self.assertIn("A", str(raised.exception))
        self.assertIn("B", str(raised.exception))
        self.assertEqual(self.runs.cancelled, [])

    def test_all_succeed_returns_normally(self) -> None:
        self.runs.succeeds("A")
        self.runs.succeeds("B")

        self.assertIsNone(self.orchestrator.wait_for_completion(["A", "B"]))
        self.assertEqual(self.runs.cancelled, [])


class KfpOrchestratorTest(TestCase):
    @patch("gigl.orchestration.kubeflow.kfp_orchestrator.FileLoader")
    def test_compile_uploads_compiled_yaml(self, MockFileLoader):
        mock_file_loader = MockFileLoader.return_value
        mock_file_loader.load_file.return_value = None

        dst_compiled_pipeline_path = GcsUri(
            "gs://SOME NON EXISTING BUCKET/ NON EXISTING FILE"
        )
        KfpOrchestrator.compile(
            cuda_container_image="SOME NONEXISTENT IMAGE 1",
            cpu_container_image="SOME NONEXISTENT IMAGE 2",
            dataflow_container_image="SOME NONEXISTENT IMAGE 3",
            dst_compiled_pipeline_path=dst_compiled_pipeline_path,
        )
        mock_file_loader.load_file.assert_called_once_with(
            file_uri_src=ANY, file_uri_dst=dst_compiled_pipeline_path
        )


if __name__ == "__main__":
    absltest.main()
