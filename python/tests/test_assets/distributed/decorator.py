import functools
import multiprocessing


def run_in_seperate_process(func):
    """
    Decorator to run a test function in a separate process.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        def target(q, *args, **kwargs):
            try:
                result = func(*args, **kwargs)
                q.put((True, result))
            except Exception as e:
                q.put((False, e))

        q: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=target, args=(q, *args), kwargs=kwargs)
        process.start()
        process.join()

        if not q.empty():
            success, result_or_exception = q.get()
            if not success:
                raise result_or_exception

        if process.exitcode != 0:
            raise RuntimeError(
                f"Test {func.__name__} failed in a separate process with exit code {process.exitcode}"
            )

    return wrapper
