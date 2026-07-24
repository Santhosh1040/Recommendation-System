from fastapi.testclient import TestClient
from api import app


client = TestClient(app)


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)

    assert "status" in data


# --------------------------------------------------
# Valid User Recommendation
# --------------------------------------------------

def test_user_recommendations():

    response = client.get("/recommend/user/1")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)

    if len(data) > 0:

        first = data[0]

        assert "movieId" in first
        assert "title" in first
        assert "hybrid_score" in first


# --------------------------------------------------
# Invalid User
# --------------------------------------------------

def test_invalid_user():

    response = client.get("/recommend/user/999999")

    assert response.status_code == 404


# --------------------------------------------------
# Valid Movie Recommendation
# --------------------------------------------------

def test_movie_recommendations():

    response = client.get("/recommend/movie/1")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)

    if len(data) > 0:

        first = data[0]

        assert "movieId" in first
        assert "title" in first
        assert "similarity" in first


# --------------------------------------------------
# Invalid Movie
# --------------------------------------------------

def test_invalid_movie():

    response = client.get("/recommend/movie/999999")

    assert response.status_code == 404