import time
from collections.abc import MutableMapping
from multiprocessing.context import SpawnProcess
from typing import Final, Optional

import psutil
import torch
from absl.testing import absltest

from gigl.common.logger import Logger
from tests.test_assets.distributed.utils import assert_tensor_equality

logger = Logger()

DEFAULT_TIMEOUT_SECONDS: Final[float] = 60.0 * 30  # 30 minutes
DEFAULT_POLL_INTERVAL_SECONDS: Final[float] = 0.1


# TODO(kmonte): Migrate us to use this class.
class TestCase(absltest.TestCase):
    """
    Base class for all tests.
    """

    def kill_all_children_processes(self, pid: Optional[int] = None) -> None:
        """Kill all children processes.

        Args:
            pid: Optional pid of the process to kill children of. If not provided, the current process is used.
        """
        try:
            this_process = psutil.Process(pid) if pid is not None else psutil.Process()
        except psutil.NoSuchProcess:
            logger.info(f"Process {pid} not found")
            return
        logger.info(
            f"Killing all children processes of {this_process.name} with pid {this_process.pid}"
        )
        for proc in this_process.children(recursive=True):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                logger.info(f"Process {proc.name} with pid {proc.pid} not found")

    def assert_all_processes_succeed(
        self,
        processes: list[SpawnProcess],
        exception_dict: Optional[MutableMapping[str, str]] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        """Wait for processes to complete and assert all succeeded.

        Monitors all processes and terminates remaining ones if any process fails.
        After all processes complete (or are terminated), asserts that all had
        exit code 0, including stack traces from exception_dict if available.

        Args:
            exception_dict: Optional shared dict mapping process names to stack traces.
                If provided, should be a multiprocessing.Manager().dict()
                If provided, the spawned processes should be named and the names should be added to the dict, with any errors as the values for the names.
                E.g. exception_dict["process_name"] = "error message"
            timeout_seconds: Maximum time to wait for all processes.
            poll_interval_seconds: How often to check process status.
        """
        start_time = time.time()
        if not all(isinstance(proc, SpawnProcess) for proc in processes):
            raise ValueError(
                "All processes must be of type SpawnProcess. e.g. multiprocessing.get_context('spawn').Process(...)"
            )
        while time.time() - start_time < timeout_seconds:
            # Check if any process has failed
            any_failed = False
            error_msg = ""
            for proc in processes:
                if proc.exitcode is not None and proc.exitcode != 0:
                    error_msg += (
                        f"Process {proc.name} failed with exit code {proc.exitcode}"
                    )
                    if exception_dict and proc.name in exception_dict:
                        error_msg += f"\nStack trace:\n{exception_dict[proc.name]}"
                    any_failed = True
                    self.kill_all_children_processes(proc.pid)
            if any_failed:
                self.kill_all_children_processes()
                self.fail(error_msg)
            if all(proc.exitcode is not None for proc in processes):
                break
            time.sleep(poll_interval_seconds)
        else:
            self.kill_all_children_processes()
            self.fail(
                f"Timeout of {timeout_seconds}s reached. Processes did not complete within the timeout."
            )

    def assert_tensor_equality(
        self, tensor_a: torch.Tensor, tensor_b: torch.Tensor, dim: Optional[int] = None
    ) -> None:
        """
        Assert that the two provided tensors are equal to each other.
        Args:
            tensor_a (torch.Tensor): First tensor which equality is being checked for
            tensor b (torch.Tensor): Second tensor which equality is being checked for
            dim (int): The dimension we are sorting over. If this value is None, we assume that the tensors must be an exact match. For a
                2D tensor, passing in a value of 1 will mean that the column order does not matter.
        """
        assert_tensor_equality(tensor_a, tensor_b, dim)
