"""Verify runtime Shape Contract checking in a spawned test process."""

import importlib
import multiprocessing as mp
import os
from multiprocessing.queues import Queue

from tests.test_assets.test_case import TestCase


def _inspect_runtime_typechecking(result_queue: Queue) -> None:
    importlib.import_module("gigl.src.common.types.task_inputs")
    from tests.test_assets import runtime_type_checking

    result_queue.put(
        (
            runtime_type_checking._import_hook is not None,
            len(runtime_type_checking._instrumented_functions),
            os.getpid(),
        )
    )


class RuntimeTypeCheckingMultiprocessTest(TestCase):
    def test_spawned_process_installs_and_uses_import_hook(self) -> None:
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=_inspect_runtime_typechecking, args=(result_queue,)
        )

        process.start()
        process.join(timeout=60)

        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            self.fail("Spawned runtime type-checking test process timed out")
        self.assertEqual(process.exitcode, 0)
        is_enabled, instrumented_function_count, child_pid = result_queue.get(timeout=5)
        self.assertTrue(is_enabled)
        self.assertGreater(instrumented_function_count, 0)
        self.assertNotEqual(child_pid, os.getpid())
        result_queue.close()
