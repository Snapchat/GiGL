import threading
from time import sleep, time

from gigl.common.logger import Logger
from gigl.common.utils.retry import retry
from gigl.src.common.utils.timeout import TimedOutException
from tests.test_assets.test_case import TestCase

logger = Logger()


class RetryUtilsTest(TestCase):
    def test_retry_deadline(self):
        @retry(deadline_s=1)
        def should_raise_timeout_exception_fn():
            sleep(15)

        start = time()
        self.assertRaises(TimedOutException, should_raise_timeout_exception_fn)
        total_time_s = time() - start
        self.assertLessEqual(
            total_time_s, 10
        )  # If function took longer than 10s then timeout isnt working

    def test_retry(self):
        exec_counter = 0

        @retry(tries=5, delay_s=1, backoff=1)
        def should_succeed_after_3_tries():
            nonlocal exec_counter
            if exec_counter < 3:
                exec_counter += 1
                # include nice msg when raising exeption so people dont freak out
                raise AttributeError(
                    "This is not a real exception - for testing purposes"
                )

            return True

        self.assertTrue(should_succeed_after_3_tries())
        self.assertEquals(exec_counter, 3)

    def test_retry_with_function_deadlines(self):
        exec_counter = 0

        @retry(tries=5, delay_s=1, backoff=1, fn_execution_timeout_s=1)
        def should_timeout_first_try_and_then_succeed():
            nonlocal exec_counter
            exec_counter += 1
            if exec_counter == 1:
                sleep(15)
            elif exec_counter == 2:
                return True

        start = time()
        self.assertTrue(should_timeout_first_try_and_then_succeed())
        self.assertEquals(exec_counter, 2)
        total_time_s = time() - start
        self.assertLessEqual(
            total_time_s, 10
        )  # If function took longer than 10s then timeout isnt working

    def test_retry_deadline_off_main_thread_raises_clearly(self):
        @retry(deadline_s=1)
        def fn():
            return "ok"

        captured: dict = {}

        def target():
            try:
                fn()
            except BaseException as e:  # noqa: BLE001 - capture whatever it raises
                captured["err"] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join()

        # Off the main thread, deadline_s must fail with a clear, actionable
        # message rather than the opaque signal ValueError.
        err = captured.get("err")
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("main thread", str(err))
