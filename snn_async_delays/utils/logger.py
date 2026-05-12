"""
Thin wrapper around Python's logging module.
Call setup_logger() once at the start of each script.
"""

import logging
import os
import sys


def setup_logger(name: str = "snn", log_file: str = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", "%H:%M:%S")

    # Console handler (force flush so output appears immediately when redirected to file)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.terminator = "\n"
    logger.addHandler(sh)
    # Make stdout line-buffered so every log line is flushed immediately
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    # File handler (optional)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
