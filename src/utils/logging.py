"""Logging configuration for the project.

Call setup_logging() once at the start of any script or entry point.
All modules should use:

    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure project-wide logging.

    Args:
        level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Reset any existing handlers
    )
