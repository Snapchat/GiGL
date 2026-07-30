import logging
import os
import pathlib
from datetime import datetime
from typing import Any, MutableMapping, Optional

from google.cloud import logging as google_cloud_logging

from gigl.env.constants import (
    GIGL_DEBUG,
    GIGL_DISABLE_CLOUD_LOGGING_ENV_KEY,
    is_env_flag_enabled,
)

_BASE_LOG_FILE_PATH = "/tmp/research/gbml/logs"


class Logger(logging.LoggerAdapter):
    """
    GiGL's custom logger class used for local and cloud logging (VertexAI, Dataflow, etc.)

    On App Engine and Kubernetes, records are routed to Google Cloud Logging, which
    renders them as GCP JSON. Set ``GIGL_DISABLE_CLOUD_LOGGING`` to fall back to the
    console format.

    Args:
        logger (Optional[logging.Logger]): A custom logger to use. If not provided, the default logger will be created.
        name (Optional[str]): The name to be used for the logger. By default uses "root".
        log_to_file (bool): If True, logs will be written to a file. If False, logs will be written to the console.
        extra (Optional[dict[str, Any]]): Extra information to be added to the log message.
    """

    _DID_ALERT_FOR_LOG_LEVEL: bool = False

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        name: Optional[str] = None,
        log_to_file: bool = False,
        extra: Optional[dict[str, Any]] = None,
    ):
        gigl_debug = is_env_flag_enabled(GIGL_DEBUG)
        if gigl_debug:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        if logger is None:
            logger = logging.getLogger(name)
            self._setup_logger(logger, name, log_to_file, log_level)

        super().__init__(logger, extra or {})

        if not Logger._DID_ALERT_FOR_LOG_LEVEL:
            Logger._DID_ALERT_FOR_LOG_LEVEL = True
            level_name = logging.getLevelName(log_level)
            self.info(f"{GIGL_DEBUG}={gigl_debug}, using log level {level_name}")

    def _setup_logger(
        self,
        logger: logging.Logger,
        name: Optional[str],
        log_to_file: bool,
        log_level: int,
    ) -> None:
        handler: logging.Handler
        if not logger.handlers:
            is_cloud_environment = bool(
                os.getenv("GAE_APPLICATION")
                or os.environ.get("KUBERNETES_SERVICE_HOST")
            )
            # Cloud Logging's handler renders each record as a single-line GCP JSON
            # envelope, which is a loss wherever another system reframes the line before
            # Cloud Logging sees it: Ray relays worker output behind a
            # "(RayTrainWorker pid=...)" prefix that makes the line invalid JSON, so it
            # is stored as a plain textPayload and the structured fields are dropped
            # anyway. GIGL_DISABLE_CLOUD_LOGGING selects the console format instead.
            if is_cloud_environment and not is_env_flag_enabled(
                GIGL_DISABLE_CLOUD_LOGGING_ENV_KEY
            ):
                # Google Cloud Logging
                client = google_cloud_logging.Client()
                client.setup_logging(log_level=log_level)
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
            logger.setLevel(log_level)

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> Any:
        if "extra" in kwargs:
            kwargs["extra"].update(self.extra)
        else:
            kwargs["extra"] = self.extra
        return msg, kwargs

    def __getattr__(self, name: str):
        # Read ``logger`` straight from ``__dict__`` to avoid re-entering
        # ``__getattr__`` (which only runs on failed lookups) and recursing
        # forever before the wrapped logger is set.
        try:
            logger = self.__dict__["logger"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(logger, name)
