import pandas as pd

from src.config import DATA_PATH
from src.logger import logger

def load_datasets(data_path=DATA_PATH, verbose=True):
    """
    Load all MovieLens datasets.

    Parameters:
        data_path (str): Path to the data directory.

    Returns:
        tuple:
            movies,
            ratings,
            tags,
            links
    """

    if verbose:
        logger.info("Loading datasets...")

    movies = pd.read_csv(f"{data_path}/movies.csv")
    ratings = pd.read_csv(f"{data_path}/ratings.csv")
    tags = pd.read_csv(f"{data_path}/tags.csv")
    links = pd.read_csv(f"{data_path}/links.csv")

    if verbose:
        logger.info("Datasets loaded successfully.")

    return movies, ratings, tags, links


def profile_dataset(df, name):
    """
    Display basic information about a dataset.
    """

    print(name)
    print("-" * 40)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}\n")

    print("Missing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\n")


def explore_dataset(movies, ratings, tags):
    """
    Display useful statistics.
    """

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Total Users          : {ratings['userId'].nunique()}")
    print(f"Total Movies         : {movies['movieId'].nunique()}")
    print(f"Total Ratings        : {len(ratings)}")
    print(f"Total Tags           : {len(tags)}")

    print()

    print(f"Average Rating       : {ratings['rating'].mean():.2f}")
    print(f"Minimum Rating       : {ratings['rating'].min()}")
    print(f"Maximum Rating       : {ratings['rating'].max()}")

    print()

    print("Top 10 Most Rated Movies")

    top_movies = (
        ratings.groupby("movieId")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="Number of Ratings")
        .merge(
            movies[
                [
                    "movieId",
                    "title",
                ]
            ],
            on="movieId",
        )
    )

    print(
        top_movies[
            [
                "title",
                "Number of Ratings",
            ]
        ]
    )

    print()


def preprocess_movies(
    movies,
    tags,
    verbose=True,
):
    """
    Prepare movie information for
    content-based recommendation.
    """

    movies = movies.copy()

    # -------------------------------------------------
    # Clean Genres
    # -------------------------------------------------

    movies["genres"] = (
        movies["genres"]
        .fillna("")
        .str.replace(
            "|",
            " ",
            regex=False,
        )
    )

    # -------------------------------------------------
    # Combine Tags
    # -------------------------------------------------

    movie_tags = (
        tags.groupby("movieId")["tag"]
        .apply(
            lambda x: " ".join(
                x.astype(str)
            )
        )
        .reset_index()
    )

    movies = movies.merge(
        movie_tags,
        on="movieId",
        how="left",
    )

    movies["tag"] = (
        movies["tag"]
        .fillna("")
    )

    # -------------------------------------------------
    # Build Content Feature
    # -------------------------------------------------

    movies["content"] = (
        movies["genres"]
        + " "
        + movies["tag"]
    )

    # Remove extra spaces
    movies["content"] = (
        movies["content"]
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    # -------------------------------------------------
    # IMPORTANT
    # Reset index so TF-IDF indices
    # always match DataFrame indices.
    # -------------------------------------------------

    movies = movies.reset_index(
        drop=True
    )

    if verbose:
        logger.info("Movie preprocessing completed.")

    return movies