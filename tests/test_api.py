from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_validation() -> None:
    response = client.post("/recommend", json={"job_description": "short", "top_k": 5})
    assert response.status_code == 422
