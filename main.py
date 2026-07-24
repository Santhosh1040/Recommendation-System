import os
import json

from src.preprocessing import (
    load_datasets,
    profile_dataset,
    explore_dataset,
    preprocess_movies,
)

from src.collaborative import (
    build_user_movie_matrix,
    compute_user_similarity,
    get_similar_users,
    recommend_movies,
)

from src.content_based import (
    build_tfidf_matrix,
    compute_movie_similarity,
    recommend_similar_movies,
)

from src.hybrid import hybrid_recommendations
from src.evaluation import evaluate_model


def main():
    """
    Main driver for the Movie Recommendation System.
    """

   
    # Create Output Directory
   

    os.makedirs("outputs", exist_ok=True)

    
    # Load Datasets
    

    movies, ratings, tags, links = load_datasets()

    
    # Dataset Profiling
    

    profile_dataset(movies, "Movies Dataset")
    profile_dataset(ratings, "Ratings Dataset")
    profile_dataset(tags, "Tags Dataset")
    profile_dataset(links, "Links Dataset")

    
    # Exploratory Data Analysis
   

    explore_dataset(movies, ratings, tags)

    
    # Data Preprocessing
    

    movies = preprocess_movies(movies, tags)

   
    # Sample Processed Movies
    

    print("=" * 60)
    print("Sample Processed Movies")
    print("=" * 60)

    print(
        movies[
            [
                "movieId",
                "title",
                "genres",
                "content",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    
    # Collaborative Filtering
    

    user_movie_matrix = build_user_movie_matrix(ratings)

    similarity_df = compute_user_similarity(
        user_movie_matrix
    )

    print("\n" + "=" * 60)
    print("Top 10 Similar Users for User 1")
    print("=" * 60)

    print(
        get_similar_users(
            1,
            similarity_df,
        )
    )

    
    # Collaborative Recommendations
    

    print("\n" + "=" * 60)
    print("Top 10 Collaborative Recommendations")
    print("=" * 60)

    collaborative_movies = recommend_movies(
        user_id=1,
        ratings=ratings,
        movies=movies,
        similarity_df=similarity_df,
        top_n=10,
    )

    print(
        collaborative_movies.to_string(
            index=False
        )
    )

    collaborative_movies.to_csv(
        "outputs/collaborative_recommendations.csv",
        index=False,
    )

    
    # Content-Based Filtering
    

    tfidf_matrix = build_tfidf_matrix(
        movies
    )

    movie_similarity = compute_movie_similarity(
        tfidf_matrix
    )

    print("\n" + "=" * 60)
    print("Movies Similar to Toy Story (1995)")
    print("=" * 60)

    similar_movies = recommend_similar_movies(
        movie_title="Toy Story (1995)",
        movies=movies,
        similarity_matrix=movie_similarity,
        top_n=10,
    )

    print(
        similar_movies.to_string(
            index=False
        )
    )

    similar_movies.to_csv(
        "outputs/content_recommendations.csv",
        index=False,
    )

    # Hybrid Recommendations
    

    print("\n" + "=" * 60)
    print("Top 10 Hybrid Recommendations")
    print("=" * 60)

    hybrid_movies = hybrid_recommendations(
        user_id=1,
        ratings=ratings,
        movies=movies,
        similarity_df=similarity_df,
        movie_similarity=movie_similarity,
        top_n=10,
    )

    print(
        hybrid_movies.to_string(
            index=False
        )
    )

    hybrid_movies.to_csv(
        "outputs/hybrid_recommendations.csv",
        index=False,
    )

    
    # Offline Evaluation
    

    evaluation = evaluate_model(
        ratings=ratings,
        movies=movies,
        movie_similarity=movie_similarity,
        k=10,
    )

    print("\n" + "=" * 60)
    print("Final Evaluation Results")
    print("=" * 60)

    for metric, value in evaluation.items():
        print(f"{metric:<20}: {value}")

    with open(
        "outputs/evaluation_results.json",
        "w",
    ) as file:
        json.dump(
            evaluation,
            file,
            indent=4,
        )

    print("\nOutputs successfully saved to the 'outputs/' directory.")


if __name__ == "__main__":
    main()