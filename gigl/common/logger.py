import logging
import os
import pathlib
import sys
from datetime import datetime
from typing import Any, MutableMapping, Optional

from gigl.common.formatters import GcpJsonFormatter

_BASE_LOG_FILE_PATH = "/tmp/research/gbml/logs"


def _is_gcp_environment() -> bool:
    """Return ``True`` when running on a GCP-managed platform (GKE, GAE, etc.).

    Checks the environment variables that GCP injects into container workloads.
    Extracted as a standalone helper so tests can patch a single symbol.
    """
    return bool(os.getenv("GAE_APPLICATION") or os.getenv("KUBERNETES_SERVICE_HOST"))


class Logger(logging.LoggerAdapter):
    """GiGL's custom logger class used for local and cloud logging (VertexAI, Dataflow, etc.).

    Args:
        logger: A custom logger to use. If not provided, the default logger will be created.
        name: The name to be used for the logger. By default uses "root".
        log_to_file: If True, logs will be written to a file. If False, logs will be written to the console.
        extra: Extra information to be added to the log message.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        name: Optional[str] = None,
        log_to_file: bool = False,
        extra: Optional[dict[str, Any]] = None,
    ):
        if logger is None:
            logger = logging.getLogger(name)
            self._setup_logger(logger, name, log_to_file)

        super().__init__(logger, extra or {})

    def _setup_logger(
        self, logger: logging.Logger, name: Optional[str], log_to_file: bool
    ) -> None:
        handler: logging.Handler
        if not logger.handlers:
            if _is_gcp_environment():
                handler = logging.StreamHandler(stream=sys.stderr)
                handler.setFormatter(GcpJsonFormatter())
                logger.addHandler(handler)
            else:
                # Logging locally. Set up logging to console or file
                if log_to_file:
                    log_dir = _BASE_LOG_FILE_PATH
                    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    log_file_name = f"{name}_{datetime_str}.log"
                    log_file_path = os.path.join(log_dir, log_file_name)
                    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
                    handler = logging.FileHandler(log_file_path)
                else:
                    handler = logging.StreamHandler()

                formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] : %(message)s (%(filename)s:%(funcName)s:%(lineno)d)",
                    datefmt="%Y-%m-%d %H:%M",
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> Any:
        if "extra" in kwargs:
            kwargs["extra"].update(self.extra)
        else:
            kwargs["extra"] = self.extra
        return msg, kwargs

    def __getattr__(self, name: str):
        return getattr(self._logger, name)
