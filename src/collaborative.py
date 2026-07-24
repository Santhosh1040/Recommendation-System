import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import TOP_N, MIN_RATING
from src.logger import logger


def build_user_movie_matrix(ratings, verbose=True):
    """
    Build the User-Movie Rating Matrix.

    Rows    -> Users
    Columns -> Movies
    Values  -> Ratings
    """

    if verbose:
        logger.info("Building User-Movie Matrix...")

    user_movie_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0,
    )

    if verbose:
        logger.info("User-Movie Matrix created successfully.")

    return user_movie_matrix


def compute_user_similarity(user_movie_matrix, verbose=True):
    """
    Compute cosine similarity between users.
    """

    if verbose:
        logger.info("Computing User Similarity Matrix...")

    similarity_matrix = cosine_similarity(user_movie_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index,
    )

    if verbose:
        logger.info("User Similarity Matrix computed.")

    return similarity_df


def get_similar_users(user_id, similarity_df, top_n=TOP_N):
    """
    Return the Top-N users most similar to the target user.
    """

    if user_id not in similarity_df.index:
        return pd.Series(dtype=float)

    similar_users = (
        similarity_df.loc[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
        .head(top_n)
    )

    return similar_users


def recommend_movies(
    user_id,
    ratings,
    movies,
    similarity_df,
    top_n=TOP_N,
    verbose=True,
):
    """
    Recommend movies using User-Based Collaborative Filtering.

    Recommendation Score =
        Σ(similarity × rating)

    Final Score =
        Weighted Score × Number of Similar Users
    """

    if verbose:
        logger.info(f"Generating recommendations for User {user_id}...")

    # -------------------------------------------------
    # Check User
    # -------------------------------------------------

    if user_id not in similarity_df.index:
        return pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "score",
                "recommended_by",
            ]
        )

    # -------------------------------------------------
    # Top Similar Users
    # -------------------------------------------------

    similar_users = get_similar_users(
        user_id=user_id,
        similarity_df=similarity_df,
        top_n=TOP_N,
    )

    if similar_users.empty:
        return pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "score",
                "recommended_by",
            ]
        )

    # -------------------------------------------------
    # Movies already watched
    # -------------------------------------------------

    watched_movies = set(
        ratings.loc[
            ratings["userId"] == user_id,
            "movieId",
        ]
    )

    movie_scores = {}
    recommendation_count = {}

    # -------------------------------------------------
    # Aggregate recommendations
    # -------------------------------------------------

    for similar_user, similarity in similar_users.items():

        # Ignore users with almost no similarity
        if similarity <= 0:
            continue

        user_ratings = ratings[
            (ratings["userId"] == similar_user)
            & (ratings["rating"] >= MIN_RATING)
        ]

        for _, row in user_ratings.iterrows():

            movie_id = row["movieId"]

            if movie_id in watched_movies:
                continue

            weighted_score = similarity * row["rating"]

            movie_scores[movie_id] = (
                movie_scores.get(movie_id, 0)
                + weighted_score
            )

            recommendation_count[movie_id] = (
                recommendation_count.get(movie_id, 0)
                + 1
            )

    # -------------------------------------------------
    # No Recommendations
    # -------------------------------------------------

    if not movie_scores:
        return pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "score",
                "recommended_by",
            ]
        )

    # -------------------------------------------------
    # Compute Final Scores
    # -------------------------------------------------

    recommendations = pd.DataFrame(
        [
            {
                "movieId": movie_id,
                "score": round(
                    movie_scores[movie_id]
                    * recommendation_count[movie_id],
                    4,
                ),
                "recommended_by": recommendation_count[movie_id],
            }
            for movie_id in movie_scores
        ]
    )

    # -------------------------------------------------
    # Merge Movie Titles
    # -------------------------------------------------

    recommendations = recommendations.merge(
        movies[
            [
                "movieId",
                "title",
            ]
        ],
        on="movieId",
        how="left",
    )

    recommendations = recommendations.sort_values(
        by="score",
        ascending=False,
    )

    return recommendations[
        [
            "movieId",
            "title",
            "score",
            "recommended_by",
        ]
    ].head(top_n)