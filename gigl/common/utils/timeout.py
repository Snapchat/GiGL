import errno
import os
import signal
from functools import wraps

from gigl.common.logger import Logger

logger = Logger()


class TimedOutException(Exception):
    pass


def timeout(
    seconds=10,
    error_message=os.strerror(errno.ETIME),
    timeout_action_func=None,
    exception_thrown_on_timeout=TimedOutException,
    **timeout_action_func_params,
):
    """
    Decorator to exit a program when a function execution exceeds a specified timeout.

    This decorator exits the program when the decorated function timed out, and executes
    timeout_action_func before exiting. The timeout_action_func can be useful for cases
    like environment clean up (e.g., ray.shutdown()). Another way to handle timeouts,
    especially when there are multiple threads or child threads, is to run `func` with
    multiprocessing. However, multiprocessing does not work with Ray.

    References to the current timeout implementation:
    - https://docs.python.org/3/library/signal.html
    - https://www.saltycrane.com/blog/2010/04/using-python-timeout-decorator-uploading-s3/

    Parameters:
    - seconds (int): Timeout duration in seconds. Defaults to 10.
    - error_message (str): Error message to log on timeout. Defaults to os.strerror(errno.ETIME).
    - timeout_action_func (callable, optional): Function to execute before exiting on timeout.
    - exception_thrown_on_timeout (Exception): Exception to raise on timeout. Defaults to TimedOutException.
    - **timeout_action_func_params: Arbitrary keyword arguments passed to timeout_action_func.
    """

    def decorator(func):
        def _handler(signum, frame):
            logger.info(error_message)
            if timeout_action_func:
                timeout_action_func(**timeout_action_func_params)
            raise exception_thrown_on_timeout()

        def wrapper(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # cancel the alarm
                signal.alarm(0)
                # reinstall the old signal handler
                signal.signal(signal.SIGALRM, old)
            return result

        return wraps(func)(wrapper)

    return decorator
