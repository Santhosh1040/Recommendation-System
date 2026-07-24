import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import TOP_N
from src.logger import logger


def build_tfidf_matrix(movies, verbose=True):
    """
    Build a TF-IDF matrix from the movie content.
    """

    if verbose:
        logger.info("Building TF-IDF Matrix...")

    movies = movies.copy()

    # Ensure no missing values
    movies["content"] = movies["content"].fillna("")

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    tfidf_matrix = tfidf.fit_transform(movies["content"])

    if verbose:
        logger.info("TF-IDF Matrix created successfully.")

    return tfidf_matrix


def compute_movie_similarity(tfidf_matrix, verbose=True):
    """
    Compute cosine similarity between all movies.
    """

    if verbose:
        logger.info("Computing Movie Similarity Matrix...")

    similarity_matrix = cosine_similarity(tfidf_matrix)

    if verbose:
        logger.info("Movie Similarity Matrix computed.")

    return similarity_matrix


def recommend_similar_movies(
    movie_title,
    movies,
    similarity_matrix,
    top_n=TOP_N,
):
    """
    Recommend movies similar to the given movie.
    """

    if movies.empty:
        return pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "similarity",
            ]
        )

    
    # Find the movie
   

    movie_matches = movies[
        movies["title"].str.lower() == movie_title.lower()
    ]

    if movie_matches.empty:
        return pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "similarity",
            ]
        )

    movie_index = movie_matches.index[0]

    # Similarity Scores
   

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True,
    )

    # Remove the movie itself
    similarity_scores = similarity_scores[1 : top_n + 1]

    recommendations = []

   
    # Build Recommendation List
    

    for index, score in similarity_scores:

        recommendations.append(
            {
                "movieId": movies.iloc[index]["movieId"],
                "title": movies.iloc[index]["title"],
                "similarity": round(float(score), 4),
            }
        )

    recommendations = pd.DataFrame(recommendations)

    if recommendations.empty:
        return pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "similarity",
            ]
        )

    recommendations = (
        recommendations.sort_values(
            by="similarity",
            ascending=False,
        )
        .drop_duplicates(subset="movieId")
        .reset_index(drop=True)
    )

    return recommendations