from fastapi import FastAPI, HTTPException

from src.preprocessing import (
    load_datasets,
    preprocess_movies
)

from src.collaborative import (
    build_user_movie_matrix,
    compute_user_similarity,
)

from src.content_based import (
    build_tfidf_matrix,
    compute_movie_similarity,
    recommend_similar_movies
)

from src.hybrid import hybrid_recommendations

from src.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    TOP_N,
)

from src.logger import logger



# FastAPI App


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

logger.info("Loading recommendation engine...")



# Load Datasets


movies, ratings, tags, links = load_datasets(verbose=False)

movies = preprocess_movies(
    movies,
    tags,
    verbose=False
)



# Build Collaborative Filtering


user_movie_matrix = build_user_movie_matrix(
    ratings,
    verbose=False
)

user_similarity = compute_user_similarity(
    user_movie_matrix,
    verbose=False
)



# Build Content-Based Filtering


tfidf_matrix = build_tfidf_matrix(
    movies,
    verbose=False
)

movie_similarity = compute_movie_similarity(
    tfidf_matrix,
    verbose=False
)

logger.info("Recommendation engine loaded successfully.")



# Root Endpoint


@app.get("/")
def root():
    return {
        "message": "Movie Recommendation System API",
        "version": API_VERSION
    }



# Health Check


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }



# Hybrid Recommendations


@app.get("/recommend/user/{user_id}")
def recommend_user(user_id: int):

    if user_id not in ratings["userId"].unique():
        raise HTTPException(
            status_code=404,
            detail="User not found."
        )

    recommendations = hybrid_recommendations(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        similarity_df=user_similarity,
        movie_similarity=movie_similarity,
        top_n=TOP_N,
        verbose=False
    )

    return recommendations.to_dict(orient="records")



# Similar Movies


@app.get("/recommend/movie/{movie_id}")
def recommend_movie(movie_id: int):

    movie = movies[movies["movieId"] == movie_id]

    if movie.empty:
        raise HTTPException(
            status_code=404,
            detail="Movie not found."
        )

    movie_title = movie.iloc[0]["title"]

    recommendations = recommend_similar_movies(
        movie_title=movie_title,
        movies=movies,
        similarity_matrix=movie_similarity,
        top_n=TOP_N
    )

    return recommendations.to_dict(orient="records")