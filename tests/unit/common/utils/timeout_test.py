from time import sleep

from gigl.src.common.utils.timeout import TimedOutException, timeout
from tests.test_assets.test_case import TestCase


class TimeoutUtilsTest(TestCase):
    def test_retry_timeout(self):
        @timeout(seconds=1)
        def should_raise_timeout_exception_fn():
            sleep(10)

        self.assertRaises(TimedOutException, should_raise_timeout_exception_fn)
