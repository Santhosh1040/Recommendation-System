import pandas as pd

from src.preprocessing import (
    load_datasets,
    preprocess_movies,
)


def test_load_datasets():

    movies, ratings, tags, links = load_datasets(verbose=False)

    # DataFrames should not be empty
    assert not movies.empty
    assert not ratings.empty
    assert not tags.empty
    assert not links.empty

    # Expected columns
    assert "movieId" in movies.columns
    assert "title" in movies.columns
    assert "genres" in movies.columns

    assert "userId" in ratings.columns
    assert "rating" in ratings.columns

    assert "tag" in tags.columns

    assert "imdbId" in links.columns


def test_preprocess_movies():

    movies, ratings, tags, links = load_datasets(verbose=False)

    processed_movies = preprocess_movies(
        movies,
        tags,
        verbose=False,
    )

    # New content column should exist
    assert "content" in processed_movies.columns

    # Content column should not contain null values
    assert processed_movies["content"].isnull().sum() == 0

    # Number of movies should remain unchanged
    assert len(processed_movies) == len(movies)

    # Index should be reset correctly
    assert processed_movies.index[0] == 0
    assert processed_movies.index[-1] == len(processed_movies) - 1