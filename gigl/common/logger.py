import json
import logging
import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any, MutableMapping, Optional

_BASE_LOG_FILE_PATH = "/tmp/research/gbml/logs"

_PYTHON_LEVEL_TO_GCP_SEVERITY: dict[str, str] = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}

# Key used by Logger.process() to pass user-supplied extras to the formatter
# without mixing them into the LogRecord's built-in attributes.
_GCP_LABELS_RECORD_ATTR: str = "_gcp_labels"


class _GcpJsonFormatter(logging.Formatter):
    """A ``logging.Formatter`` that outputs one JSON object per line with
    `GCP-recognized structured logging fields
    <https://cloud.google.com/logging/docs/structured-logging>`_.

    Fields emitted:

    - ``severity`` -- mapped from the Python log level.
    - ``message`` -- the formatted log message (with traceback appended when present).
    - ``time`` -- ISO 8601 UTC timestamp.
    - ``logging.googleapis.com/sourceLocation`` -- ``{file, line, function}``.
    - ``logging.googleapis.com/labels`` -- any extra fields supplied via the
      ``extra`` dict on the ``Logger`` adapter.  Omitted when there are no extras.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as a single-line JSON string.

        Args:
            record: The ``LogRecord`` to format.

        Returns:
            A JSON string (no trailing newline) suitable for writing to
            ``sys.stderr`` on GCP-managed environments.
        """
        message = record.getMessage()

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message = f"{message}\n{record.exc_text}"
        if record.stack_info:
            message = f"{message}\n{record.stack_info}"

        payload: dict[str, object] = {
            "severity": _PYTHON_LEVEL_TO_GCP_SEVERITY.get(
                record.levelname, record.levelname
            ),
            "message": message,
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "logging.googleapis.com/sourceLocation": {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            },
        }

        labels: dict[str, object] = getattr(record, _GCP_LABELS_RECORD_ATTR, {})
        if labels:
            payload["logging.googleapis.com/labels"] = labels

        return json.dumps(payload, ensure_ascii=False, default=str)


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
                handler.setFormatter(_GcpJsonFormatter())
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
        merged: dict[str, Any] = dict(self.extra)
        if "extra" in kwargs:
            merged.update(kwargs["extra"])
        kwargs["extra"] = {**merged, _GCP_LABELS_RECORD_ATTR: merged}
        return msg, kwargs

    def __getattr__(self, name: str):
        return getattr(self._logger, name)
