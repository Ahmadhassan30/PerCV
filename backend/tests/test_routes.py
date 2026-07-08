"""Backend test suite."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lanes_stub():
    response = client.post("/api/lanes/detect")
    assert response.status_code == 200
    assert response.json()["task"] == "lane_detection"


def test_matching_stub():
    response = client.post("/api/matching/match")
    assert response.status_code == 200
    assert response.json()["task"] == "sift_matching"


def test_panorama_stub():
    response = client.post("/api/panorama/stitch")
    assert response.status_code == 200
    assert response.json()["task"] == "panorama_stitching"


def test_classify_stub():
    response = client.post("/api/classify/predict")
    assert response.status_code == 200
    assert response.json()["task"] == "cnn_classification"


def test_dashboard_metrics():
    response = client.get("/api/dashboard/metrics")
    assert response.status_code == 200
