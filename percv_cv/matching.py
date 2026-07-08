"""
percv_cv.matching — Task 2: SIFT descriptor extraction & matching.

Extracts SIFT keypoints, runs BruteForce L2 matching, and evaluates
Lowe's ratio test across multiple thresholds (0.60 / 0.75 / 0.90).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from pathlib import Path
    from typing import Dict, List, Tuple


def extract_sift_keypoints(image: "np.ndarray") -> "Tuple":
    """Detect SIFT keypoints and compute descriptors."""
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")


def match_descriptors(des1: "np.ndarray", des2: "np.ndarray",
                      lowe_ratio: float = 0.75) -> "List":
    """BruteForce L2 kNN matching with Lowe's ratio test."""
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")


def ratio_sweep(des1: "np.ndarray", des2: "np.ndarray",
                ratios: "List[float]") -> "Dict[float, int]":
    """Run match_descriptors at each ratio, returning {ratio: match_count}."""
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")


def run_sift_matching(image_a: "Path", image_b: "Path",
                      config: "Dict") -> "Dict":
    """End-to-end SIFT matching pipeline on an HPatches pair.

    Returns a dict with keypoint counts, match counts per ratio,
    and inlier statistics.
    """
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")
