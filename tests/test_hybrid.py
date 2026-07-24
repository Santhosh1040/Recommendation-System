from src.preprocessing import (
    load_datasets,
    preprocess_movies,
)

from src.collaborative import (
    build_user_movie_matrix,
    compute_user_similarity,
)

from src.content_based import (
    build_tfidf_matrix,
    compute_movie_similarity,
)

from src.hybrid import hybrid_recommendations


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

tfidf_matrix = build_tfidf_matrix(
    movies,
    verbose=False,
)

movie_similarity = compute_movie_similarity(
    tfidf_matrix,
    verbose=False,
)


# --------------------------------------------------
# Hybrid Recommendation Generation
# --------------------------------------------------

def test_hybrid_recommendations():

    recommendations = hybrid_recommendations(
        user_id=1,
        ratings=ratings,
        movies=movies,
        similarity_df=similarity_df,
        movie_similarity=movie_similarity,
        top_n=10,
        verbose=False,
    )

    assert not recommendations.empty

    assert "movieId" in recommendations.columns
    assert "title" in recommendations.columns
    assert "collaborative_score" in recommendations.columns
    assert "similarity" in recommendations.columns
    assert "hybrid_score" in recommendations.columns


# --------------------------------------------------
# Hybrid Scores Should Be Sorted
# --------------------------------------------------

def test_hybrid_scores_sorted():

    recommendations = hybrid_recommendations(
        user_id=1,
        ratings=ratings,
        movies=movies,
        similarity_df=similarity_df,
        movie_similarity=movie_similarity,
        top_n=10,
        verbose=False,
    )

    scores = recommendations["hybrid_score"].tolist()

    assert scores == sorted(
        scores,
        reverse=True,
    )


# --------------------------------------------------
# Already Watched Movies Should Not Be Recommended
# --------------------------------------------------

def test_hybrid_excludes_watched_movies():

    recommendations = hybrid_recommendations(
        user_id=1,
        ratings=ratings,
        movies=movies,
        similarity_df=similarity_df,
        movie_similarity=movie_similarity,
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

def test_hybrid_top_n_limit():

    recommendations = hybrid_recommendations(
        user_id=1,
        ratings=ratings,
        movies=movies,
        similarity_df=similarity_df,
        movie_similarity=movie_similarity,
        top_n=5,
        verbose=False,
    )

    assert len(recommendations) <= 5