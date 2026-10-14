"""Central logging configuration, shared by every module in the package."""

import logging
import os

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once; safe to call multiple times."""
    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def attach_file_handler(path: str | os.PathLike) -> logging.FileHandler:
    """Write log lines to a UTF-8 file as well, using LOG_FORMAT.

    Adds the handler to the root logger and returns it. The console handler
    and the root level are not changed. Pass the result to
    detach_file_handler when done.
    """
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler


def detach_file_handler(handler: logging.FileHandler) -> None:
    """Remove the handler from the root logger and close its file.

    Safe to call more than once.
    """
    logging.getLogger().removeHandler(handler)
    handler.close()
