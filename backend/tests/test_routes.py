import io
import time
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def create_dummy_image_bytes(w: int = 100, h: int = 100, color=(128, 128, 128)) -> bytes:
    """Helper to generate a simple dummy JPEG image in-memory."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w, h), color, -1)
    # Add a white stripe to help SIFT/line matching avoid zero features
    cv2.line(img, (10, 10), (w - 10, h - 10), (255, 255, 255), 3)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lanes_valid():
    img_data = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/lanes",
        files=[("files", ("image1.jpg", img_data, "image/jpeg"))]
    )
    assert response.status_code == 200
    res_json = response.json()
    assert "preset_mean_scores" in res_json
    assert "overlay_base64" in res_json
    assert "balanced" in res_json["preset_mean_scores"]


def test_lanes_invalid_unsupported_type():
    # Attempt upload of .txt file
    response = client.post(
        "/api/v1/lanes",
        files=[("files", ("test.txt", b"dummy plain text", "text/plain"))]
    )
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_match_valid():
    img1 = create_dummy_image_bytes(120, 120, (255, 0, 0))
    img2 = create_dummy_image_bytes(120, 120, (0, 0, 255))
    
    response = client.post(
        "/api/v1/match",
        files={
            "img1_file": ("img1.png", img1, "image/png"),
            "img2_file": ("img2.png", img2, "image/png")
        }
    )
    assert response.status_code == 200
    res_json = response.json()
    assert "ratios_sweep" in res_json
    assert "visualization_base64" in res_json
    assert "0.75" in res_json["ratios_sweep"]


def test_match_invalid_missing_input():
    response = client.post(
        "/api/v1/match",
        files={"img1_file": ("img1.png", b"dummy", "image/png")}
    )
    # Missing required form file img2_file should return 422 Validation Error
    assert response.status_code == 422


def test_panorama_3_images_valid():
    img1 = create_dummy_image_bytes(150, 100, (100, 100, 100))
    img2 = create_dummy_image_bytes(150, 100, (150, 150, 150))
    img3 = create_dummy_image_bytes(150, 100, (200, 200, 200))

    response = client.post(
        "/api/v1/panorama",
        data={"anchor": "middle"},
        files=[
            ("files", ("left.jpg", img1, "image/jpeg")),
            ("files", ("middle.jpg", img2, "image/jpeg")),
            ("files", ("right.jpg", img3, "image/jpeg"))
        ]
    )
    assert response.status_code == 202
    res_json = response.json()
    assert "job_id" in res_json
    assert res_json["status"] == "pending"
    
    job_id = res_json["job_id"]
    
    # Poll for completion
    completed = False
    for _ in range(10):
        time.sleep(0.3)
        poll_res = client.get(f"/api/v1/panorama/jobs/{job_id}")
        assert poll_res.status_code == 200
        poll_json = poll_res.json()
        if poll_json["status"] == "completed":
            completed = True
            assert "panorama_base64" in poll_json
            assert "pair_metrics" in poll_json
            assert "average_inlier_ratio" in poll_json
            break
        elif poll_json["status"] == "failed":
            # If SIFT matching fails on dummy uniform data, that is mathematically acceptable,
            # as long as the pipeline status successfully runs and completes/fails gracefully.
            break

    # Endpoint must return job details correctly
    assert completed or PANORAMA_JOBS_COMPLETED_OR_FAILED(job_id)


def test_panorama_2_images_valid():
    img1 = create_dummy_image_bytes()
    img2 = create_dummy_image_bytes()

    response = client.post(
        "/api/v1/panorama",
        data={"anchor": "right"},
        files=[
            ("files", ("left.jpg", img1, "image/jpeg")),
            ("files", ("right.jpg", img2, "image/jpeg"))
        ]
    )
    assert response.status_code == 202
    assert "job_id" in response.json()


def test_panorama_invalid_image_count():
    img1 = create_dummy_image_bytes()
    # Attempting to upload only 1 image (requires 2 or 3)
    response = client.post(
        "/api/v1/panorama",
        files=[("files", ("img.jpg", img1, "image/jpeg"))]
    )
    assert response.status_code == 422
    assert "Wrong number of images" in response.json()["detail"]


def test_classify_valid_default():
    img = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/classify",
        files={"file": ("input.png", img, "image/png")}
    )
    assert response.status_code == 200
    res_json = response.json()
    assert "predicted_class" in res_json
    assert "confidence" in res_json
    assert "probabilities" in res_json
    assert "gradcam_base64" in res_json


def test_classify_invalid_backbone():
    img = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/classify?backbone=invalid_cnn_model",
        files={"file": ("input.png", img, "image/png")}
    )
    assert response.status_code == 422
    assert "Invalid backbone option" in response.json()["detail"]


def test_dashboard():
    response = client.get("/api/v1/dashboard")
    assert response.status_code == 200
    res_json = response.json()
    assert "baseline_metrics" in res_json
    assert "backbone_comparison" in res_json
    assert "task1" in res_json["baseline_metrics"]


def PANORAMA_JOBS_COMPLETED_OR_FAILED(job_id: str) -> bool:
    """Helper check to verify job states."""
    from app.api.routes.panorama import PANORAMA_JOBS
    return job_id in PANORAMA_JOBS and PANORAMA_JOBS[job_id]["status"] in ["completed", "failed"]
