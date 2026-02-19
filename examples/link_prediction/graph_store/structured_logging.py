import json
import logging
import sys


class StructuredLogFormatter(logging.Formatter):
    """Formats log records as single-line JSON for Google Cloud structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestampSeconds": int(record.created),
            "timestampNanos": int((record.created % 1) * 1e9),
            "logging.googleapis.com/sourceLocation": {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            },
        }
        return json.dumps(log_entry)


def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(StructuredLogFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
