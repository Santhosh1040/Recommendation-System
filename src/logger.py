"""
Project Logger

Provides a reusable logger instance for the entire application.
"""

import logging

from src.config import (
    LOG_FORMAT,
    LOG_LEVEL,
)


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
)

logger = logging.getLogger("movie_recommender")