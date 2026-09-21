import logging
import os
import sys

try:
    from pythonjsonlogger import jsonlogger
except ImportError:  # pragma: no cover - fallback keeps logging usable before deps are refreshed.
    jsonlogger = None


def _build_formatter():
    if os.getenv("RAGBOT_LOG_JSON", "false").lower() == "true" and jsonlogger is not None:
        return jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
    return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def get_logger(name: str):
    """
    Get a configured logger instance.
    Logs to console with INFO level.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        log_level = os.getenv("RAGBOT_LOG_LEVEL", "INFO").upper()
        logger.setLevel(log_level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

    # Create formatter and add to handler
        formatter = _build_formatter()
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger
