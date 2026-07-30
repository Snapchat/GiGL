import logging
import os
import pathlib
from datetime import datetime
from typing import Any, MutableMapping, Optional

from google.cloud import logging as google_cloud_logging

from gigl.env.constants import GIGL_DISABLE_CLOUD_LOGGING_ENV_KEY

_BASE_LOG_FILE_PATH = "/tmp/research/gbml/logs"

# Values of GIGL_DISABLE_CLOUD_LOGGING that leave cloud logging on, so that setting
# the variable to "0" reads as "off" rather than as any-value-means-on.
_FALSY_ENV_VALUES = frozenset({"", "0", "false"})


def _is_cloud_logging_disabled() -> bool:
    """Whether GIGL_DISABLE_CLOUD_LOGGING opts this process out of cloud logging.

    On GKE, ``google.cloud.logging`` attaches a handler that renders every record as
    a single-line GCP JSON envelope. Under Ray that envelope is unreadable and its
    structured fields are dropped anyway: Ray prefixes each relayed worker line with
    ``(RayTrainWorker pid=...)``, which makes the line invalid JSON, so Cloud Logging
    stores it as a plain ``textPayload``. Set this variable on such processes to get
    the console format instead.

    Returns:
        True when the variable is set to anything other than "", "0", or "false"
        (case-insensitive).
    """
    value = os.environ.get(GIGL_DISABLE_CLOUD_LOGGING_ENV_KEY, "")
    return value.lower() not in _FALSY_ENV_VALUES


class Logger(logging.LoggerAdapter):
    """
    GiGL's custom logger class used for local and cloud logging (VertexAI, Dataflow, etc.)

    On App Engine and Kubernetes, records are routed to Google Cloud Logging, which
    renders them as GCP JSON. Set ``GIGL_DISABLE_CLOUD_LOGGING`` to fall back to the
    console format -- see :func:`_is_cloud_logging_disabled`.

    Args:
        logger (Optional[logging.Logger]): A custom logger to use. If not provided, the default logger will be created.
        name (Optional[str]): The name to be used for the logger. By default uses "root".
        log_to_file (bool): If True, logs will be written to a file. If False, logs will be written to the console.
        extra (Optional[dict[str, Any]]): Extra information to be added to the log message.
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
            is_cloud_environment = bool(
                os.getenv("GAE_APPLICATION")
                or os.environ.get("KUBERNETES_SERVICE_HOST")
            )
            if is_cloud_environment and not _is_cloud_logging_disabled():
                # Google Cloud Logging
                client = google_cloud_logging.Client()
                client.setup_logging(log_level=logging.INFO)
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
        # Read ``logger`` straight from ``__dict__`` to avoid re-entering
        # ``__getattr__`` (which only runs on failed lookups) and recursing
        # forever before the wrapped logger is set.
        try:
            logger = self.__dict__["logger"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(logger, name)
