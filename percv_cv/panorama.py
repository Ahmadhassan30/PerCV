"""
percv_cv.panorama — Task 3: From-scratch panorama stitching.

Per-frame RANSAC homography to a middle anchor, distance-transform
weighted blending, and contour-based auto-crop to remove black borders.
No cv2.Stitcher — everything is manual.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Union
from pathlib import Path


def compute_frame_homography(kp_src: List[cv2.KeyPoint], kp_anchor: List[cv2.KeyPoint],
                             matches: List, ransac_thresh: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    findHomography(src, anchor, RANSAC, ransac_thresh). Returns H and a
    dict with {total_matches, inliers, inlier_ratio} for logging — this dict
    is what gets written into the per-pair metrics table.
    """
    total = len(matches)
    if total < 4:
        return np.eye(3), {
            "total_matches": total,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "status": "FAILED_LOW_MATCHES"
        }

    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_anchor[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if H is None:
        return np.eye(3), {
            "total_matches": total,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "status": "FAILED_SOLVER_ERR"
        }

    inliers = int(np.sum(mask))
    ratio = inliers / total if total > 0 else 0.0

    return H, {
        "total_matches": total,
        "inliers": inliers,
        "inlier_ratio": ratio,
        "status": "SUCCESS"
    }


def compute_canvas_bounds(images: List[np.ndarray],
                          homographies: List[np.ndarray]) -> Tuple[int, int, int, int]:
    """
    Projects all image corners via cv2.perspectiveTransform through their
    respective homography, returns (xmin, ymin, xmax, ymax) global bounds.
    Anchor (middle) image uses identity transform.
    """
    all_pts = []
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)
        all_pts.extend(transformed.reshape(-1, 2))

    all_pts = np.array(all_pts)
    x_min, y_min = np.min(all_pts, axis=0)
    x_max, y_max = np.max(all_pts, axis=0)

    return int(np.floor(x_min)), int(np.floor(y_min)), int(np.ceil(x_max)), int(np.ceil(y_max))


def build_translation_matrix(xmin: int, ymin: int) -> np.ndarray:
    """3x3 translation matrix T shifting global bounds into positive space."""
    return np.array([
        [1.0, 0.0, -xmin],
        [0.0, 1.0, -ymin],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


def warp_frame(image: np.ndarray, H: np.ndarray, T: np.ndarray,
               canvas_size: Tuple[int, int]) -> np.ndarray:
    """cv2.warpPerspective using T @ H, output canvas_size."""
    return cv2.warpPerspective(image, T @ H, canvas_size)


def distance_transform_weight(warped_image: np.ndarray) -> np.ndarray:
    """
    cv2.distanceTransform on the binary content mask of the warped image,
    normalized. This is the blending weight map — must be masked so
    zero-content (black) regions get zero weight, preventing border
    interpolation bleed into the blend.
    """
    # Create binary content mask of the warped image
    mask = (warped_image > 0).any(axis=-1).astype(np.uint8)
    if not np.any(mask):
        return mask.astype(np.float32)

    # Compute Euclidean L2 distance to closest border
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    mx = np.max(dist)
    if mx > 0:
        weights = dist / mx
    else:
        weights = mask.astype(np.float32)
    return weights


def blend_weighted(frames: List[np.ndarray], weights: List[np.ndarray]) -> np.ndarray:
    """I_blended = sum(W_k * I_k) / sum(W_k), pixel-wise, with epsilon guard
    against division by zero where all weights are 0."""
    h, w = frames[0].shape[:2]
    numerator = np.zeros((h, w, 3), dtype=np.float32)
    denominator = np.zeros((h, w, 3), dtype=np.float32)

    for f, wt in zip(frames, weights):
        wt_3d = np.stack([wt, wt, wt], axis=-1)
        numerator += f.astype(np.float32) * wt_3d
        denominator += wt_3d

    denominator = np.where(denominator == 0, 1.0, denominator)
    blended = numerator / denominator
    return np.clip(blended, 0, 255).astype(np.uint8)


def autocrop(blended: np.ndarray) -> np.ndarray:
    """
    Threshold to binary mask, fill interior holes via external contours
    (so dark interior scene content doesn't collapse the bounding box),
    then iteratively shrink the bounding box from each edge until every
    boundary row/column is non-black.
    """
    gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Fill interior holes to prevent dark scene pixels from collapsing the crop
    mask = np.zeros_like(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return blended
    cv2.drawContours(mask, contours, -1, 255, -1)

    # Find tight bounding box
    coords = cv2.findNonZero(mask)
    if coords is None:
        return blended
    x, y, w, h = cv2.boundingRect(coords)
    cropped = blended[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]

    # Shrink to find clean inscribed rectangle (removing ragged/black border pixels)
    h_c, w_c = mask_cropped.shape
    top, bottom, left, right = 0, h_c - 1, 0, w_c - 1

    max_steps = min(h_c, w_c) // 2
    for _ in range(max_steps):
        has_black_top = np.any(mask_cropped[top, left:right+1] == 0)
        has_black_bottom = np.any(mask_cropped[bottom, left:right+1] == 0)
        has_black_left = np.any(mask_cropped[top:bottom+1, left] == 0)
        has_black_right = np.any(mask_cropped[top:bottom+1, right] == 0)

        if not (has_black_top or has_black_bottom or has_black_left or has_black_right):
            break

        if has_black_top:
            top += 1
        if has_black_bottom:
            bottom -= 1
        if has_black_left:
            left += 1
        if has_black_right:
            right -= 1

        if top >= bottom or left >= right:
            return cropped

    return cropped[top:bottom+1, left:right+1]


def stitch_panorama(images: Dict[str, np.ndarray], anchor_key: str, config: Any) -> Dict[str, Any]:
    """
    Full pipeline orchestration for one scene. Returns
    {panorama: np.ndarray, pair_metrics: dict[str, dict], average_inlier_ratio: float}
    """
    if anchor_key not in images:
        raise KeyError(f"Anchor key '{anchor_key}' not found in input images.")

    anchor_img = images[anchor_key]
    gray_anchor = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp_anchor, des_anchor = sift.detectAndCompute(gray_anchor, None)

    homographies = []
    frames_list = []
    pair_metrics = {}

    # Gather frames and compute homographies
    for key in sorted(images.keys()):
        frame = images[key]
        frames_list.append(frame)
        
        if key == anchor_key:
            homographies.append(np.eye(3))
            continue

        # Extract features and match
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn_matches = bf.knnMatch(des_frame, des_anchor, k=2)
        
        # Apply Lowe ratio filter
        lowe_ratio = getattr(config, "default_lowe_ratio", 0.75)
        good_matches = []
        for match in knn_matches:
            if len(match) == 2:
                m, n = match
                if m.distance < lowe_ratio * n.distance:
                    good_matches.append(m)

        # Compute Homography
        ransac_thresh = getattr(config, "ransac_threshold", 5.0)
        H, stats = compute_frame_homography(kp_frame, kp_anchor, good_matches, ransac_thresh)
        homographies.append(H)
        pair_metrics[f"{key}_to_{anchor_key}"] = stats

    # 1. Compute canvas bounds
    xmin, ymin, xmax, ymax = compute_canvas_bounds(frames_list, homographies)
    canvas_w = xmax - xmin
    canvas_h = ymax - ymin

    # 2. Build translation matrix
    T = build_translation_matrix(xmin, ymin)

    # 3. Warp frames and compute weight maps
    warped_frames = []
    weight_maps = []

    for frame, H in zip(frames_list, homographies):
        warped = warp_frame(frame, H, T, (canvas_w, canvas_h))
        w = distance_transform_weight(warped)
        warped_frames.append(warped)
        weight_maps.append(w)

    # 4. Blend weighted
    blended = blend_weighted(warped_frames, weight_maps)

    # 5. Autocrop
    cropped = autocrop(blended)

    # Compute average inlier ratio
    inlier_ratios = [stats["inlier_ratio"] for stats in pair_metrics.values()]
    avg_ratio = np.mean(inlier_ratios) if inlier_ratios else 0.0

    return {
        "panorama": cropped,
        "pair_metrics": pair_metrics,
        "average_inlier_ratio": avg_ratio,
        "weight_maps": weight_maps
    }
