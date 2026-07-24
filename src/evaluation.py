import numpy as np
import pandas as pd

from src.collaborative import (
    build_user_movie_matrix,
    compute_user_similarity,
)

from src.hybrid import hybrid_recommendations


# Metric Functions


def precision_at_k(recommended, relevant, k):
    """
    Precision@K
    """

    if len(recommended) == 0:
        return 0.0

    recommended = recommended[:k]

    hits = len(set(recommended) & set(relevant))

    return hits / k


def recall_at_k(recommended, relevant):
    """
    Recall@K
    """

    if len(relevant) == 0:
        return 0.0

    hits = len(set(recommended) & set(relevant))

    return hits / len(relevant)


def hit_rate(recommended, relevant):
    """
    Hit Rate
    """

    return 1.0 if len(set(recommended) & set(relevant)) > 0 else 0.0


def ndcg_at_k(recommended, relevant, k):
    """
    NDCG@K (binary relevance).
    """

    recommended = recommended[:k]
    relevant_set = set(relevant)

    dcg = 0.0

    for i, movie_id in enumerate(recommended, start=1):
        if movie_id in relevant_set:
            dcg += 1.0 / np.log2(i + 1)

    ideal_hits = min(len(relevant_set), k)

    idcg = sum(
        1.0 / np.log2(i + 1)
        for i in range(1, ideal_hits + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg

# Evaluation


def evaluate_model(
    ratings,
    movies,
    movie_similarity,
    k=10,
    test_size=0.2,
):
    """
    Offline evaluation of the Hybrid Recommendation System.

    The dataset is randomly split into training and testing sets.
    The recommendation model is built using the training data and
    evaluated on the held-out test set using Precision@K, Recall@K,
    Hit Rate, NDCG@K and Coverage.
    """

    print("\n" + "=" * 60)
    print("Offline Evaluation")
    print("=" * 60)

  
    # Train/Test Split
  

    np.random.seed(42)

    shuffled = ratings.sample(
        frac=1,
        random_state=42
    )

    split_index = int(len(shuffled) * (1 - test_size))

    train = shuffled.iloc[:split_index].copy()

    test = shuffled.iloc[split_index:].copy()

  
    # Build Collaborative Model
   

    user_movie_matrix = build_user_movie_matrix(
        train,
        verbose=False,
    )

    similarity_df = compute_user_similarity(
        user_movie_matrix,
        verbose=False,
    )

    # Eligible Users
    

    train_counts = train.groupby("userId").size()

    eligible_users = sorted([
        user
        for user in set(train["userId"]) & set(test["userId"])
        if train_counts[user] >= 5
    ])

    precision_scores = []
    recall_scores = []
    hit_scores = []
    ndcg_scores = []

    recommended_catalog = set()

    print(f"Users Evaluated : {len(eligible_users)}\n")

    # Evaluate Each User
  

    for index, user_id in enumerate(eligible_users, start=1):

        if index == 1 or index % 50 == 0:
            print(
                f"Evaluating User {index}/{len(eligible_users)}..."
            )

        recommendations = hybrid_recommendations(
            user_id=user_id,
            ratings=train,
            movies=movies,
            similarity_df=similarity_df,
            movie_similarity=movie_similarity,
            top_n=k,
            verbose=False,
        )

        if recommendations.empty:
            continue

        recommended_movies = recommendations[
            "movieId"
        ].tolist()

        recommended_catalog.update(
            recommended_movies
        )

        relevant_movies = test[
            (test["userId"] == user_id)
            &
            (test["rating"] >= 4)
        ]["movieId"].tolist()

        if len(relevant_movies) == 0:
            continue

        precision_scores.append(
            precision_at_k(
                recommended_movies,
                relevant_movies,
                k,
            )
        )

        recall_scores.append(
            recall_at_k(
                recommended_movies,
                relevant_movies,
            )
        )

        hit_scores.append(
            hit_rate(
                recommended_movies,
                relevant_movies,
            )
        )

        ndcg_scores.append(
            ndcg_at_k(
                recommended_movies,
                relevant_movies,
                k,
            )
        )

    # Coverage
   

    total_movies = movies["movieId"].nunique()

    coverage = (
        len(recommended_catalog) / total_movies
        if total_movies > 0
        else 0
    )

    # Results
   

    results = {
        "Users Evaluated":
            len(precision_scores),

        f"Precision@{k}":
            round(
                np.mean(precision_scores),
                4,
            ) if precision_scores else 0,

        f"Recall@{k}":
            round(
                np.mean(recall_scores),
                4,
            ) if recall_scores else 0,

        f"HitRate@{k}":
            round(
                np.mean(hit_scores),
                4,
            ) if hit_scores else 0,

        f"NDCG@{k}":
            round(
                np.mean(ndcg_scores),
                4,
            ) if ndcg_scores else 0,

        "Coverage":
            round(
                coverage,
                4,
            ),
    }

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    for metric, value in results.items():
        print(f"{metric:<20}: {value}")

    return results