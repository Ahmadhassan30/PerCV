import numpy as np
import cv2
import pytest
from percv_cv.panorama import (
    compute_frame_homography,
    compute_canvas_bounds,
    build_translation_matrix,
    warp_frame,
    distance_transform_weight,
    blend_weighted,
    autocrop,
    stitch_panorama
)


def test_compute_canvas_bounds_translation():
    # Synthetic corner points of two overlapping frames
    # Frame 1: 0 to 640 width, 0 to 480 height
    # Frame 2: Shifted by +300 on X axis, +50 on Y axis
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)

    # Pure translation homography: x' = x + 300, y' = y + 50
    H1 = np.eye(3)
    H2 = np.array([
        [1.0, 0.0, 300.0],
        [0.0, 1.0, 50.0],
        [0.0, 0.0, 1.0]
    ])

    xmin, ymin, xmax, ymax = compute_canvas_bounds([img1, img2], [H1, H2])

    # Expected bounds: xmin = 0, ymin = 0
    # xmax = max(640, 640 + 300) = 940
    # ymax = max(480, 480 + 50) = 530
    assert xmin == 0
    assert ymin == 0
    assert xmax == 940
    assert ymax == 530


def test_autocrop_black_border():
    # Create a 200x200 image with a solid non-black square in the center
    # Surrounded by a known black border of 20 pixels on top/bottom/left/right
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    # Fill center 160x160 area with solid color
    canvas[20:180, 20:180] = (255, 128, 64)

    cropped = autocrop(canvas)

    # Verify that the cropped result has exactly size 160x160 and has no black border
    assert cropped.shape[0] == 160
    assert cropped.shape[1] == 160
    assert np.all(cropped == (255, 128, 64))


def test_stitch_panorama_integration():
    # Setup two overlapping crops of a 1000x400 synthetic canvas with distinct SIFT details
    canvas = np.zeros((400, 1000, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(canvas, (500, 200), 80, (200, 200, 200), -1)
    cv2.rectangle(canvas, (700, 100), (900, 300), (100, 100, 100), -1)

    # Draw grid of small dots to create stable keypoints
    for y in range(50, 350, 40):
        for x in range(50, 950, 40):
            cv2.circle(canvas, (x, y), 4, (150, 150, 150), -1)

    # Overlapping frames: width 600, overlap = 200 pixels
    # Left: 0 to 600
    # Right: 400 to 1000
    left = canvas[:, 0:600]
    right = canvas[:, 400:1000]

    images = {
        "left": left,
        "right": right
    }

    # Configuration stub
    class DummyConfig:
        default_lowe_ratio = 0.75
        ransac_threshold = 5.0

    results = stitch_panorama(images, anchor_key="right", config=DummyConfig())

    # Assert basic structure of outputs
    assert "panorama" in results
    assert "pair_metrics" in results
    assert "average_inlier_ratio" in results

    pano = results["panorama"]
    assert pano.shape[0] > 0
    assert pano.shape[1] > 0
    # Clean non-zero bounding box checks
    assert np.any(pano > 0)
