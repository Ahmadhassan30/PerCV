"""
percv_cv.matching — Task 2: SIFT descriptor extraction & matching.

Extracts SIFT keypoints, runs BruteForce L2 matching, and evaluates
Lowe's ratio test sweep, matching baseline metrics.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Union
from pathlib import Path


def extract_sift(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """SIFT keypoints + descriptors.

    Args:
        image: Input image (BGR or Grayscale).

    Returns:
        Tuple of keypoint list and descriptor array.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


def match_bruteforce_l2(desc1: np.ndarray, desc2: np.ndarray, k: int = 2) -> List:
    """BFMatcher with L2 norm, k=2 for ratio test.

    Args:
        desc1: Descriptors of the first image.
        desc2: Descriptors of the second image.
        k: Number of nearest neighbors to search for.

    Returns:
        List of kNN matches.
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    return bf.knnMatch(desc1, desc2, k=k)


def apply_lowe_ratio(matches: List, ratio: float) -> List:
    """Filter knn matches by Lowe's ratio test.

    Args:
        matches: Raw kNN matches from BFMatcher.
        ratio: Lowe's ratio threshold.

    Returns:
        List of good matches satisfying the ratio test.
    """
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    return good_matches


def sweep_lowe_ratios(desc1: np.ndarray, desc2: np.ndarray, ratios: List[float]) -> Dict[float, List]:
    """
    Runs apply_lowe_ratio for each ratio in config (0.60/0.75/0.90),
    returns good matches list per ratio.

    Args:
        desc1: Descriptors of the first image.
        desc2: Descriptors of the second image.
        ratios: List of Lowe's ratio thresholds to evaluate.

    Returns:
        Dict mapping ratio threshold to filtered matches.
    """
    raw_matches = match_bruteforce_l2(desc1, desc2, k=2)
    result = {}
    for r in ratios:
        result[r] = apply_lowe_ratio(raw_matches, r)
    return result


def estimate_inlier_ratio(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
                          good_matches: List, ransac_thresh: float) -> Dict[str, Any]:
    """
    Runs findHomography with RANSAC on the good matches to get an inlier
    mask, returns {total_matches, inliers, inlier_ratio}.

    Args:
        kp1: Keypoints of the first image.
        kp2: Keypoints of the second image.
        good_matches: Filtered matches.
        ransac_thresh: RANSAC threshold for homography estimation.

    Returns:
        Dict with keys: total_matches, inliers, inlier_ratio.
    """
    total = len(good_matches)
    if total < 4:
        return {
            "total_matches": total,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "mask": None
        }

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return {
            "total_matches": total,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "mask": None
        }

    inliers = int(np.sum(mask))
    ratio = inliers / total if total > 0 else 0.0

    return {
        "total_matches": total,
        "inliers": inliers,
        "inlier_ratio": ratio,
        "mask": mask.flatten().tolist()
    }
