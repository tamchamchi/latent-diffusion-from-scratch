import logging
import os
import random
from typing import Optional

import numpy as np
import torch


def set_random_seed(seed: int = 0):
    """
    Sets the random seed for reproducibility across different libraries.

    Args:
        seed (int): The integer seed to use.
    """
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")

    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        # Ensure that all operations on GPU are deterministic.
        # This can sometimes come with a performance penalty.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Set to False for determinism, True for speed if not needing determinism
    else:
        print(f"Random seed set for PyTorch (CPU) to {seed}")

    # Optional: Set environment variable for some frameworks (e.g., for hashing)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with colored console output and optional file logging.

    Args:
        name (Optional[str]):
            Logger name. If None, uses the root logger.
        level (int):
            Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (Optional[str]):
            Optional file path for saving logs without colors.

    Returns:
        logging.Logger:
            Configured logger instance.
    """

    # ANSI escape codes for terminal colors
    class LogColors:
        DEBUG = "\033[36m"  # Cyan
        INFO = "\033[32m"  # Green
        WARNING = "\033[33m"  # Yellow
        ERROR = "\033[31m"  # Red
        CRITICAL = "\033[41m"  # Red background
        RESET = "\033[0m"  # Reset to default

    # Custom formatter that applies colors to the log level name
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: LogColors.DEBUG,
            logging.INFO: LogColors.INFO,
            logging.WARNING: LogColors.WARNING,
            logging.ERROR: LogColors.ERROR,
            logging.CRITICAL: LogColors.CRITICAL,
        }

        def format(self, record):
            # Preserve original level name
            original_levelname = record.levelname

            # Apply color based on log level
            color = self.COLORS.get(record.levelno, LogColors.RESET)
            record.levelname = f"{color}{original_levelname}{LogColors.RESET}"

            # Format the message
            formatted_message = super().format(record)

            # Restore original level name to avoid side effects
            record.levelname = original_levelname

            return formatted_message

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Prevent logs from propagating to parent loggers
    logger.propagate = False

    # Create console handler with colored formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    console_formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)

    # Optional file handler (plain text, no colors)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    return logger
