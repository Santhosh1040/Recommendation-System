from src.preprocessing import (
    load_datasets,
    preprocess_movies,
)

from src.content_based import (
    build_tfidf_matrix,
    compute_movie_similarity,
    recommend_similar_movies,
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

tfidf_matrix = build_tfidf_matrix(
    movies,
    verbose=False,
)

similarity_matrix = compute_movie_similarity(
    tfidf_matrix,
    verbose=False,
)


# --------------------------------------------------
# TF-IDF Matrix
# --------------------------------------------------

def test_tfidf_matrix():

    # One TF-IDF vector per movie
    assert tfidf_matrix.shape[0] == len(movies)

    # Should have extracted at least one feature
    assert tfidf_matrix.shape[1] > 0


# --------------------------------------------------
# Similarity Matrix
# --------------------------------------------------

def test_similarity_matrix():

    assert similarity_matrix.shape[0] == len(movies)
    assert similarity_matrix.shape[1] == len(movies)


# --------------------------------------------------
# Similar Movie Recommendations
# --------------------------------------------------

def test_recommend_similar_movies():

    recommendations = recommend_similar_movies(
        movie_title="Toy Story (1995)",
        movies=movies,
        similarity_matrix=similarity_matrix,
        top_n=10,
    )

    assert not recommendations.empty

    assert "movieId" in recommendations.columns
    assert "title" in recommendations.columns
    assert "similarity" in recommendations.columns


# --------------------------------------------------
# Original Movie Should Not Be Returned
# --------------------------------------------------

def test_original_movie_not_recommended():

    recommendations = recommend_similar_movies(
        movie_title="Toy Story (1995)",
        movies=movies,
        similarity_matrix=similarity_matrix,
        top_n=10,
    )

    assert "Toy Story (1995)" not in recommendations["title"].values


# --------------------------------------------------
# Recommendations Sorted by Similarity
# --------------------------------------------------

def test_similarity_sorted():

    recommendations = recommend_similar_movies(
        movie_title="Toy Story (1995)",
        movies=movies,
        similarity_matrix=similarity_matrix,
        top_n=10,
    )

    similarities = recommendations["similarity"].tolist()

    assert similarities == sorted(
        similarities,
        reverse=True,
    )


# --------------------------------------------------
# Top N Limit
# --------------------------------------------------

def test_top_n_limit():

    recommendations = recommend_similar_movies(
        movie_title="Toy Story (1995)",
        movies=movies,
        similarity_matrix=similarity_matrix,
        top_n=5,
    )

    assert len(recommendations) <= 5