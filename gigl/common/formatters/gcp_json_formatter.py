import json
import logging
from datetime import datetime, timezone

_PYTHON_LEVEL_TO_GCP_SEVERITY: dict[str, str] = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}

_STANDARD_LOG_RECORD_ATTRIBUTES: frozenset[str] = frozenset(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=None,
        exc_info=None,
    ).__dict__.keys()
)


class GcpJsonFormatter(logging.Formatter):
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

    Example:
        >>> import logging, io
        >>> handler = logging.StreamHandler(io.StringIO())
        >>> handler.setFormatter(GcpJsonFormatter())
        >>> log = logging.getLogger("demo")
        >>> log.addHandler(handler)
        >>> log.setLevel(logging.INFO)
        >>> log.info("hello")
        >>> import json; json.loads(handler.stream.getvalue())["message"]
        'hello'
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

        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_ATTRIBUTES
        }
        if extra_fields:
            payload["logging.googleapis.com/labels"] = extra_fields

        return json.dumps(payload, ensure_ascii=False, default=str)
