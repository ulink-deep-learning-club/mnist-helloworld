import logging
import os
import sys


class _ColorFormatter(logging.Formatter):
    """Colored, simplified formatter for terminal output.

    Output:  ``[INFO] message``   (with ANSI color per level)
    File handler still uses the full ``asctime - name - level - message`` format.
    """

    _COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[1;31m", # Bold Red
    }
    _RESET = "\033[0m"

    def __init__(self):
        super().__init__("[%(levelname)s] %(message)s")

    def format(self, record):
        # Temporarily wrap levelname in color so the parent Formatter
        # renders it; restore after to avoid corrupting other handlers.
        original_levelname = record.levelname
        color = self._COLORS.get(original_levelname, "")
        record.levelname = f"{color}{original_levelname}{self._RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def setup_logger(
    name: str = "mnist_classifier",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Set up a logger with colored terminal output and optional file output.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: If set, also write the full-format log to this file
                  (via the root logger so all modules are captured).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # ── Stream handler (colored, simplified) ──
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(_ColorFormatter())
        logger.addHandler(sh)

    # ── File handler (full format, attached to root) ──
    if log_file:
        root = logging.getLogger()
        # Raise root level so INFO messages from children propagate
        if root.level > level:
            root.setLevel(level)

        log_path = os.path.abspath(log_file)
        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == log_path
            for h in root.handlers
        ):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            root.addHandler(fh)

    return logger
