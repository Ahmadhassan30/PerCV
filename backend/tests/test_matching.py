import numpy as np
import cv2
import pytest
from percv_cv.matching import extract_sift, match_bruteforce_l2, apply_lowe_ratio, estimate_inlier_ratio


def generate_synthetic_pair():
    """Generate a synthetic image pair with a known affine transform for feature matching validation."""
    # First image: black canvas with white rectangles and circle to create high gradient edges
    img1 = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(img1, (80, 80), (220, 220), 255, -1)
    cv2.circle(img1, (300, 150), 50, 200, -1)
    cv2.putText(img1, "PerCV Test", (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 180, 2)

    # Second image: rotate by 10 degrees and translate slightly
    M = cv2.getRotationMatrix2D((200, 200), 10.0, 1.0)
    img2 = cv2.warpAffine(img1, M, (400, 400))

    return img1, img2


def test_sift_extraction():
    img1, img2 = generate_synthetic_pair()
    kp1, desc1 = extract_sift(img1)
    kp2, desc2 = extract_sift(img2)

    assert len(kp1) > 0, "No SIFT keypoints detected in first image"
    assert len(kp2) > 0, "No SIFT keypoints detected in second image"
    assert desc1 is not None and desc1.shape[1] == 128
    assert desc2 is not None and desc2.shape[1] == 128


def test_synthetic_matching_inliers():
    img1, img2 = generate_synthetic_pair()
    kp1, desc1 = extract_sift(img1)
    kp2, desc2 = extract_sift(img2)

    # Run kNN matching
    raw_matches = match_bruteforce_l2(desc1, desc2, k=2)
    assert len(raw_matches) > 0

    # Filter with Lowe's ratio test (recommended threshold of 0.75)
    good_matches = apply_lowe_ratio(raw_matches, 0.75)
    assert len(good_matches) > 0

    # RANSAC homography fit check
    ransac_thresh = 5.0
    inlier_stats = estimate_inlier_ratio(kp1, kp2, good_matches, ransac_thresh)

    print(f"Synthetic pair: Matches = {inlier_stats['total_matches']}, "
          f"Inliers = {inlier_stats['inliers']}, Ratio = {inlier_stats['inlier_ratio']:.4f}")

    # For a clean synthetic planar transform, the SIFT inlier ratio must approach 1.0
    assert inlier_stats["inlier_ratio"] >= 0.85, "SIFT inlier ratio for clean synthetic pair is too low"
    assert inlier_stats["total_matches"] == len(good_matches)
    assert inlier_stats["inliers"] <= len(good_matches)


def test_matching_stability():
    img1, img2 = generate_synthetic_pair()

    # First run
    kp1_a, desc1_a = extract_sift(img1)
    kp2_a, desc2_a = extract_sift(img2)
    matches_a = apply_lowe_ratio(match_bruteforce_l2(desc1_a, desc2_a), 0.75)
    stats_a = estimate_inlier_ratio(kp1_a, kp2_a, matches_a, 5.0)

    # Second run
    kp1_b, desc1_b = extract_sift(img1)
    kp2_b, desc2_b = extract_sift(img2)
    matches_b = apply_lowe_ratio(match_bruteforce_l2(desc1_b, desc2_b), 0.75)
    stats_b = estimate_inlier_ratio(kp1_b, kp2_b, matches_b, 5.0)

    # SIFT and matching are deterministic, so counts and keypoints should be identical
    assert len(kp1_a) == len(kp1_b)
    assert len(matches_a) == len(matches_b)
    assert stats_a["inliers"] == stats_b["inliers"]
