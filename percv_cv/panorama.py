"""
percv_cv.panorama — Task 3: From-scratch panorama stitching.

Per-frame RANSAC homography to a middle anchor, distance-transform
weighted blending, and contour-based auto-crop to remove black borders.
No cv2.Stitcher — everything is manual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from pathlib import Path
    from typing import Dict, List, Tuple


def compute_homography(kp_src: "List", kp_dst: "List",
                       matches: "List",
                       ransac_threshold: float = 5.0) -> "Tuple":
    """Compute a direct frame-to-anchor homography with RANSAC.

    Returns (H, stats_dict) where stats_dict includes total_matches,
    inliers, inlier_ratio, and status.
    """
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")


def compute_feather_weights(h: int, w: int) -> "np.ndarray":
    """L2 distance-transform weight map for alpha blending."""
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")


def blend_and_crop(warped_images: "List[np.ndarray]",
                   weight_maps: "List[np.ndarray]") -> "np.ndarray":
    """Weighted blend of warped images, then contour-based auto-crop."""
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")


def run_panorama_stitching(image_paths: "List[Path]",
                           config: "Dict") -> "Dict":
    """End-to-end panorama stitching pipeline.

    Returns a dict with homography matrices, inlier stats, and the
    path to the stitched output image.
    """
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")
