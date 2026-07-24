import pandas as pd

from src.collaborative import recommend_movies
from src.content_based import recommend_similar_movies
from src.config import TOP_N, FAVORITE_MOVIE_THRESHOLD
from src.logger import logger


def hybrid_recommendations(
    user_id,
    ratings,
    movies,
    similarity_df,
    movie_similarity,
    top_n=TOP_N,
    verbose=True,
):
    """
    Hybrid Recommendation System

    Combines:
    1. User-Based Collaborative Filtering
    2. Content-Based Filtering

    Final Score =
        0.6 × Collaborative Score
      + 0.4 × Content Score

    Movies recommended by both methods receive
    a small bonus.
    """

    if verbose:
        logger.info("Generating Hybrid Recommendations...")

    
    # Collaborative Recommendations
    

    collaborative = recommend_movies(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        similarity_df=similarity_df,
        top_n=100,
        verbose=verbose,
    ).copy()

    if collaborative.empty:
        return pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "collaborative_score",
                "similarity",
                "hybrid_score",
            ]
        )

    collaborative.rename(
        columns={"score": "collaborative_score"},
        inplace=True,
    )

    collaborative.drop_duplicates(
        subset="movieId",
        inplace=True,
    )

    
    # Normalize Collaborative Scores
    

    min_score = collaborative["collaborative_score"].min()
    max_score = collaborative["collaborative_score"].max()

    if max_score != min_score:
        collaborative["collaborative_score"] = (
            collaborative["collaborative_score"] - min_score
        ) / (max_score - min_score)
    else:
        collaborative["collaborative_score"] = 1.0

    
    # User's Favourite Movies
    

    liked_movies = (
        ratings[
            (ratings["userId"] == user_id)
            & (ratings["rating"] >= FAVORITE_MOVIE_THRESHOLD)
        ]
        .sort_values(
            by="rating",
            ascending=False,
        )
        .head(10)
    )

    content_frames = []

    
    # Content-Based Recommendations
    

    for movie_id in liked_movies["movieId"]:

        movie = movies[
            movies["movieId"] == movie_id
        ]

        if movie.empty:
            continue

        title = movie.iloc[0]["title"]

        recommendations = recommend_similar_movies(
            movie_title=title,
            movies=movies,
            similarity_matrix=movie_similarity,
            top_n=20,
        )

        if not recommendations.empty:
            content_frames.append(
                recommendations
            )

    if content_frames:

        content = pd.concat(
            content_frames,
            ignore_index=True,
        )

        content = (
            content.groupby(
                ["movieId", "title"],
                as_index=False,
            )["similarity"]
            .max()
        )

        # Normalize similarity

        min_sim = content["similarity"].min()
        max_sim = content["similarity"].max()

        if max_sim != min_sim:
            content["similarity"] = (
                content["similarity"] - min_sim
            ) / (max_sim - min_sim)
        else:
            content["similarity"] = 1.0

    else:

        content = pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "similarity",
            ]
        )

   
    # Merge Recommendation Lists
    

    hybrid = pd.merge(
        collaborative,
        content,
        on=["movieId", "title"],
        how="outer",
    )

    hybrid["collaborative_score"] = hybrid[
        "collaborative_score"
    ].fillna(0)

    hybrid["similarity"] = hybrid[
        "similarity"
    ].fillna(0)

    
    # Remove Already Watched Movies
    

    watched_movies = set(
        ratings.loc[
            ratings["userId"] == user_id,
            "movieId",
        ]
    )

    hybrid = hybrid[
        ~hybrid["movieId"].isin(
            watched_movies
        )
    ]

    
    # Hybrid Score
    

    collaborative_weight = 0.6
    content_weight = 0.4

    hybrid["hybrid_score"] = (
        collaborative_weight
        * hybrid["collaborative_score"]
        + content_weight
        * hybrid["similarity"]
    )

    
    # Bonus if recommended by BOTH systems
    

    overlap = (
        (hybrid["collaborative_score"] > 0)
        & (hybrid["similarity"] > 0)
    )

    hybrid.loc[
        overlap,
        "hybrid_score",
    ] += 0.10

    
    # Final Ranking
    

    hybrid = (
        hybrid.sort_values(
            by="hybrid_score",
            ascending=False,
        )
        .drop_duplicates(
            subset="movieId"
        )
        .reset_index(drop=True)
    )

    return hybrid[
        [
            "movieId",
            "title",
            "collaborative_score",
            "similarity",
            "hybrid_score",
        ]
    ].head(top_n)