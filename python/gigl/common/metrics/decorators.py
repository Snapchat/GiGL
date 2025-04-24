import time
import traceback
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, cast

from gigl.common.logger import Logger
from gigl.common.metrics.metrics_interface import OpsMetricPublisher

logger = Logger()


class TimerRecordGranularity(Enum):
    MILLISECONDS = "ms"
    SECONDS = "s"


F = TypeVar("F", bound=Callable[..., Any])


def __safely_flush_metrics(
    get_metrics_service_instance_fn: Optional[
        Callable[[], Optional[OpsMetricPublisher]]
    ]
) -> None:
    if get_metrics_service_instance_fn is not None:
        metrics_instance = get_metrics_service_instance_fn()
    if metrics_instance is not None:
        metrics_instance.flush_metrics()


def flushes_metrics(
    get_metrics_service_instance_fn: Optional[
        Callable[[], Optional[OpsMetricPublisher]]
    ]
) -> Callable[[F], F]:
    """
    Decorator for flushing metrics after function execution.
    Always catches any raised exceptions by decorated function and flushes metrics
    before reraising the exception.
    :return: wrapped result
    """

    def inner(func: F) -> F:
        def wrap(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.info(
                    f"Exception raised, will flush metrics for: {func.__name__} and re-raise exception"
                )
                logger.error(f"Exception: {e}")
                logger.error(traceback.format_exc())
                __safely_flush_metrics(
                    get_metrics_service_instance_fn=get_metrics_service_instance_fn
                )  # Flush metrics before re-raising exception
                logger.error(f"Post flushing metrics")
                raise e
            __safely_flush_metrics(
                get_metrics_service_instance_fn=get_metrics_service_instance_fn
            )
            return result

        return cast(F, wrap)

    return inner


def profileit(
    metric_name: str,
    get_metrics_service_instance_fn: Optional[
        Callable[[], Optional[OpsMetricPublisher]]
    ],
    record_granularity: TimerRecordGranularity = TimerRecordGranularity.SECONDS,
) -> Callable[[F], F]:
    """
    performance profiling decorator
    :param name: name of block being profiled
    :return: wrapped result
    """

    def inner(func: F) -> F:
        def wrap(*args: Any, **kwargs: Any) -> Any:
            raised_exception: Optional[Exception] = None
            started_at = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                raised_exception = e
            spanned_time_s = time.time() - started_at
            if record_granularity == TimerRecordGranularity.MILLISECONDS:
                spanned_time_formatted = int(spanned_time_s * 1000)
            elif record_granularity == TimerRecordGranularity.SECONDS:
                spanned_time_formatted = int(spanned_time_s)
            else:
                raise TypeError(
                    f"Unsupported record_granularity provided: {record_granularity}"
                )

            metrics_instance = None
            if get_metrics_service_instance_fn is not None:
                metrics_instance = get_metrics_service_instance_fn()
            if metrics_instance is not None:
                metrics_instance.add_timer(metric_name, spanned_time_formatted)

            if raised_exception is not None:
                raise raised_exception
            return result

        return cast(F, wrap)

    return inner
