"""
percv_cv.lanes — Task 1: Edge-based lane detection.

Implements Canny edge detection across multiple threshold configs,
Probabilistic Hough Line Transform for lane tracing, and a custom
Lanes Quality Score based on angle-bucket filtering.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Union
from pathlib import Path

# Try importing config types if available
try:
    from percv_cv.config import Task1Config
except ImportError:
    Task1Config = Any


def preprocess(image: np.ndarray, ksize: Tuple[int, int], sigma: float) -> np.ndarray:
    """Grayscale + Gaussian blur.

    Args:
        image: Source image in BGR format.
        ksize: Gaussian kernel size (width, height).
        sigma: Gaussian kernel standard deviation.

    Returns:
        Grayscale and blurred image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    return cv2.GaussianBlur(gray, ksize, sigma)


def detect_edges(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    """Canny edge detection for one threshold pair.

    Args:
        gray: Grayscale preprocessed image.
        low: Lower hysteresis threshold.
        high: Upper hysteresis threshold.

    Returns:
        Binary edge map.
    """
    return cv2.Canny(gray, low, high)


def detect_lines(edges: np.ndarray, rho: float, theta_deg: float,
                 threshold: int, min_line_len: int, max_line_gap: int) -> np.ndarray:
    """Probabilistic Hough transform. Returns line segments.

    Args:
        edges: Binary edge map.
        rho: Distance resolution of the accumulator in pixels.
        theta_deg: Angle resolution of the accumulator in degrees.
        threshold: Accumulator threshold parameter.
        min_line_len: Minimum line length.
        max_line_gap: Maximum allowed gap between points on the same line to link them.

    Returns:
        Array of detected lines of shape (N, 1, 4) containing [x1, y1, x2, y2], or None.
    """
    theta_rad = theta_deg * np.pi / 180.0
    return cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta_rad,
        threshold=threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )


def lanes_quality_score(lines: np.ndarray) -> float:
    """
    Ratio of line segments whose angle falls in [25°, 75°] or [105°, 155°]
    (realistic lane boundary slopes) to total detected segments.

    Must exactly reproduce the scoring logic implied by context.md — same
    angle buckets, same denominator (total detections, not total possible).

    Failure-Mode Note:
        Dense urban clutter (buildings, traffic signs, vehicles, pedestrian crossings,
        and shadows) produces a high number of false-positive edge detections and
        Hough line segments. Since these clutter lines do not align with the expected
        lane geometry/slopes, they inflate the denominator, yielding a lower overall
        mean Lanes Quality Score (around ~18% or 0.1809 on BDD100K road images).
    """
    if lines is None or len(lines) == 0:
        return 0.0

    m_lines = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        angle = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi)
        if angle > 180.0:
            angle -= 180.0
        if (25.0 <= angle <= 75.0) or (105.0 <= angle <= 155.0):
            m_lines += 1

    return m_lines / len(lines)


def run_lane_detection(image_paths: List[Union[str, Path]], config: Union[Task1Config, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs all three Canny presets (sensitive/balanced/strict) from config
    across all images, returns a dict with per-image, per-preset scores and
    the overall mean quality score.
    """
    # Normalize config
    if not isinstance(config, dict):
        # Convert dataclass config to dict
        t1_cfg = {
            "num_images": config.num_images,
            "gaussian_ksize": config.gaussian_ksize,
            "gaussian_sigma": config.gaussian_sigma,
            "canny_threshold_pairs": [
                {"low": p.low, "high": p.high, "label": p.label}
                for p in config.canny_threshold_pairs
            ],
            "hough_rho": config.hough_rho,
            "hough_theta_deg": config.hough_theta_deg,
            "hough_threshold": config.hough_threshold,
            "hough_min_line_len": config.hough_min_line_len,
            "hough_max_line_gap": config.hough_max_line_gap
        }
    else:
        t1_cfg = config

    results_by_preset = {}
    preset_mean_scores = {}

    canny_pairs = t1_cfg["canny_threshold_pairs"]
    ksize = t1_cfg["gaussian_ksize"]
    sigma = t1_cfg["gaussian_sigma"]
    rho = t1_cfg["hough_rho"]
    theta_deg = t1_cfg["hough_theta_deg"]
    hough_threshold = t1_cfg["hough_threshold"]
    min_line_len = t1_cfg["hough_min_line_len"]
    max_line_gap = t1_cfg["hough_max_line_gap"]

    for pair in canny_pairs:
        preset_label = pair["label"]
        low, high = pair["low"], pair["high"]
        scores = []
        image_details = []

        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue

            gray = preprocess(img, ksize, sigma)
            edges = detect_edges(gray, low, high)
            lines = detect_lines(
                edges,
                rho=rho,
                theta_deg=theta_deg,
                threshold=hough_threshold,
                min_line_len=min_line_len,
                max_line_gap=max_line_gap
            )

            score = lanes_quality_score(lines)
            scores.append(score)
            image_details.append({
                "image_path": str(path),
                "num_lines": len(lines) if lines is not None else 0,
                "quality_score": score
            })

        mean_score = np.mean(scores) if scores else 0.0
        results_by_preset[preset_label] = image_details
        preset_mean_scores[preset_label] = mean_score

    # Balanced Canny config is the primary baseline headline score config
    overall_mean_quality_score = preset_mean_scores.get("balanced", 0.0)

    return {
        "overall_mean_quality_score": overall_mean_quality_score,
        "preset_mean_scores": preset_mean_scores,
        "results_by_preset": results_by_preset
    }
