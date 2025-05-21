import functools
import multiprocessing
import traceback


class _ExceptionWithTraceback(Exception):
    def __init__(self, original_exception, traceback_str):
        self.original_exception = original_exception
        self.traceback_str = traceback_str
        super().__init__(str(original_exception))

    def __str__(self):
        return f"{self.original_exception}\n{self.traceback_str}"


def run_in_separate_process(func):
    """
    Decorator to run a test function in a separate process. This is needed to ensure that certain operations, such as torch.set_num_interop_threads and init_rpc(),
    don't interfere with each other when calling this function across multiple tests. For example, if we try to call torch.set_num_interop_threads without this
    decorator, we may observe:

    ```
    RuntimeError: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
    ```
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        def target(q: multiprocessing.Queue, *args, **kwargs):
            try:
                result = func(*args, **kwargs)
                q.put((True, result))
            except Exception as e:
                tb_str = traceback.format_exc()
                q.put((False, (e, tb_str)))

        q: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=target, args=(q, *args), kwargs=kwargs)
        process.start()
        process.join()

        if not q.empty():
            success, result_or_exception = q.get()
            if not success:
                exception, tb_str = result_or_exception
                raise _ExceptionWithTraceback(exception, tb_str)

        if process.exitcode != 0:
            raise RuntimeError(
                f"Test {func.__name__} failed in a separate process with exit code {process.exitcode}"
            )

    return wrapper
