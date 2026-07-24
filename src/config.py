"""
Application Configuration

Centralized configuration values used across the project.
"""

# --------------------------------------------------
# Data
# --------------------------------------------------

DATA_PATH = "data"

# --------------------------------------------------
# Recommendation Settings
# --------------------------------------------------

TOP_N = 10

MIN_RATING = 3.5

FAVORITE_MOVIE_THRESHOLD = 4.0

# --------------------------------------------------
# API Configuration
# --------------------------------------------------

API_TITLE = "Movie Recommendation System API"

API_VERSION = "1.0.0"

API_DESCRIPTION = (
    "Hybrid Movie Recommendation System using "
    "Collaborative Filtering and Content-Based Filtering."
)

# --------------------------------------------------
# Logging
# --------------------------------------------------

LOG_LEVEL = "INFO"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(message)s"
)