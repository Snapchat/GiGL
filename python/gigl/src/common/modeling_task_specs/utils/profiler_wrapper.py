import tempfile
from distutils.util import strtobool

from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from gigl.common import LocalUri
from gigl.common.logger import Logger

logger = Logger()


TMP_PROFILER_LOG_DIR_NAME = LocalUri(tempfile.TemporaryDirectory().name)


class TorchProfiler:
    def __init__(self, **kwargs) -> None:
        self.trace_handler = tensorboard_trace_handler(
            dir_name=TMP_PROFILER_LOG_DIR_NAME, use_gzip=True  # type: ignore
        )
        self.wait = int(kwargs.get("wait", 5))
        self.warmup = int(kwargs.get("warmup", 2))
        self.active = int(kwargs.get("active", 2))
        self.repeat = int(kwargs.get("repeat", 1))
        self.tracing_schedule = schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat,
        )
        self.profile_memory = bool(strtobool(kwargs.get("profile_memory", "True")))
        self.record_shapes = bool(strtobool(kwargs.get("record_shapes", "False")))
        self.with_stack = bool(strtobool(kwargs.get("with_stack", "False")))
        logger.info(f"Profiler will be instantiated with {self.__dict__}")

    def profiler_context(self) -> profile:
        return profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            schedule=self.tracing_schedule,
            on_trace_ready=self.trace_handler,
            profile_memory=self.profile_memory,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
        )
