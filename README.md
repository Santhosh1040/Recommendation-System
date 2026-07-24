# Movie Recommendation System

This project is a hybrid movie recommendation system built using the MovieLens dataset. The objective was to design a recommendation engine that goes beyond suggesting popular movies and instead provides recommendations based on both user behaviour and movie content.

Rather than relying on a single recommendation technique, the system combines User-Based Collaborative Filtering with Content-Based Filtering. Collaborative filtering identifies users with similar preferences and recommends movies they have enjoyed, while content-based filtering recommends movies that share similar characteristics with titles a user already likes. The final recommendations are produced by combining the strengths of both approaches into a hybrid model.

The project was developed as a complete machine learning application instead of just an algorithm implementation. Along with the recommendation engine, it includes data preprocessing, model evaluation, a REST API built with FastAPI, automated tests, centralized configuration management, and logging. The goal was to organize the project in a way that resembles a small production-ready application while keeping the code modular and easy to understand.

---

## Project Structure

```
Recommendation-System/
│
├── api.py                     # FastAPI application
├── main.py                    # Run recommendation engine locally
├── requirements.txt
├── .gitignore
│
├── data/                      # MovieLens dataset
├── docs/                      # Project report
├── outputs/                   # Example outputs
├── tests/                     # Unit tests
│
└── src/
    ├── preprocessing.py
    ├── collaborative.py
    ├── content_based.py
    ├── hybrid.py
    ├── evaluation.py
    ├── config.py
    └── logger.py
```

---

## Recommendation Approach

The recommendation engine combines two different techniques.

The collaborative filtering module studies rating patterns across users. If two users have rated many movies similarly, the system considers them to have similar preferences. Recommendations are then generated from highly rated movies watched by those similar users.

The content-based module focuses on the movies themselves. Genres and user-provided tags are combined into a textual representation, transformed into TF-IDF vectors, and compared using cosine similarity. This allows the system to recommend movies that are similar in content.

The hybrid model combines both recommendation lists into a single ranking. Collaborative filtering captures community preferences, while content-based filtering provides recommendations even when similar users are limited. Combining the two helps produce more balanced recommendations than using either method independently.

---

## Technologies Used

The project is implemented in Python using Pandas and NumPy for data processing, Scikit-learn for TF-IDF vectorization and cosine similarity, FastAPI for exposing recommendation APIs, and Pytest for automated testing.

---

## Installation

Clone the repository and install the required packages.

```bash
git clone <repository-url>
cd Recommendation-System

pip install -r requirements.txt
```

---

## Running the Project

To run the recommendation engine locally,

```bash
python main.py
```

To start the REST API,

```bash
uvicorn api:app --reload
```

Once the server starts, the API documentation is available at

```
http://127.0.0.1:8000/docs
```

---

## Running the Tests

All major modules are covered with automated unit tests.

```bash
python -m pytest -v
```

At the time of submission, all tests pass successfully.

---

## API Endpoints

The API exposes a small set of endpoints for interacting with the recommendation engine.

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/recommend/user/{user_id}` | Hybrid recommendations for a user |
| GET | `/recommend/movie/{movie_id}` | Movies similar to a given movie |

---

## Example Output

Example user recommendation

```json
[
    {
        "movieId": 296,
        "title": "Pulp Fiction (1994)",
        "hybrid_score": 0.97
    },
    {
        "movieId": 593,
        "title": "The Silence of the Lambs (1991)",
        "hybrid_score": 0.94
    }
]
```

---

## Current Limitations

Like most recommendation systems, this implementation also has a few practical limitations. New users with little or no rating history receive fewer personalized recommendations because collaborative filtering depends on previous interactions. The quality of content-based recommendations is also influenced by the richness of genres and user-generated tags available in the dataset. In addition, the current implementation computes similarity matrices in memory, which works well for the MovieLens dataset but would require optimization for significantly larger datasets.

---

## Future Improvements

There are several directions in which this project could be extended. Matrix factorization techniques such as SVD could improve collaborative recommendations, while transformer-based embeddings could replace TF-IDF for richer content understanding. The recommendation engine could also be updated to support real-time model updates, implicit feedback, and larger-scale deployments using distributed data processing.

---

## Final Notes

The focus of this project was not only to build a working recommendation engine but also to organize the solution in a maintainable way. The codebase separates preprocessing, recommendation algorithms, evaluation, configuration, logging, testing, and API development into independent modules so that each component can be understood, tested, and extended without affecting the others.