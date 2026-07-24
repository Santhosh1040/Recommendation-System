import numpy as np

from src.preprocessing import (
    load_datasets,
    preprocess_movies,
)

from src.collaborative import (
    build_user_movie_matrix,
    compute_user_similarity,
    recommend_movies,
)


# --------------------------------------------------
# Test Setup
# --------------------------------------------------

movies, ratings, tags, links = load_datasets(verbose=False)

movies = preprocess_movies(
    movies,
    tags,
    verbose=False,
)

user_movie_matrix = build_user_movie_matrix(
    ratings,
    verbose=False,
)

similarity_df = compute_user_similarity(
    user_movie_matrix,
    verbose=False,
)


# --------------------------------------------------
# User-Movie Matrix
# --------------------------------------------------

def test_user_movie_matrix():

    assert user_movie_matrix.shape[0] > 0
    assert user_movie_matrix.shape[1] > 0

    assert user_movie_matrix.index.name == "userId"


# --------------------------------------------------
# User Similarity Matrix
# --------------------------------------------------

def test_user_similarity_matrix():

    assert similarity_df.shape[0] == similarity_df.shape[1]

    # Cosine similarity of a user with themselves should be 1
    diagonal = np.diag(similarity_df.values)

    assert np.allclose(diagonal, 1.0)


# --------------------------------------------------
# Recommendation Generation
# --------------------------------------------------

def test_recommend_movies():

    recommendations = recommend_movies(
        user_id=1,
        ratings=ratings,
        movies=movies,
        similarity_df=similarity_df,
        top_n=10,
        verbose=False,
    )

    assert not recommendations.empty

    assert "movieId" in recommendations.columns
    assert "title" in recommendations.columns
    assert "score" in recommendations.columns


# --------------------------------------------------
# Already Watched Movies
# --------------------------------------------------

def test_no_watched_movies_recommended():

    recommendations = recommend_movies(
        user_id=1,
        ratings=ratings,
        movies=movies,
        similarity_df=similarity_df,
        top_n=20,
        verbose=False,
    )

    watched_movies = set(
        ratings.loc[
            ratings["userId"] == 1,
            "movieId",
        ]
    )

    recommended_movies = set(
        recommendations["movieId"]
    )

    assert watched_movies.isdisjoint(
        recommended_movies
    )


# --------------------------------------------------
# Top N Limit
# --------------------------------------------------

def test_top_n_limit():

    recommendations = recommend_movies(
        user_id=1,
        ratings=ratings,
        movies=movies,
        similarity_df=similarity_df,
        top_n=5,
        verbose=False,
    )

    assert len(recommendations) <= 5