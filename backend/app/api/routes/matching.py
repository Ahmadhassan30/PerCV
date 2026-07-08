"""SIFT Feature Matching routes — Task 2."""

import base64
from pathlib import Path
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from percv_cv.config import CONFIG
from percv_cv.matching import extract_sift, match_bruteforce_l2, apply_lowe_ratio, sweep_lowe_ratios, estimate_inlier_ratio

router = APIRouter()


class MatchStats(BaseModel):
    match_count: int
    ransac_inliers: int
    inlier_ratio: float


class MatchingResponse(BaseModel):
    ratios_sweep: Dict[str, MatchStats]
    matches_at_0_75: int
    inlier_ratio_at_0_75: float
    visualization_base64: str


def draw_matches_vis(img1: np.ndarray, kp1: list,
                     img2: np.ndarray, kp2: list,
                     good_matches: list, inlier_mask: list) -> np.ndarray:
    """Create side-by-side visualization of SIFT matches, inliers green, outliers red."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2

    for idx, match in enumerate(good_matches):
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2[match.trainIdx].pt))
        pt2_shifted = (pt2[0] + w1, pt2[1])

        is_inlier = inlier_mask is not None and len(inlier_mask) > idx and inlier_mask[idx] == 1
        color = (0, 255, 0) if is_inlier else (0, 0, 255)

        cv2.line(canvas, pt1, pt2_shifted, color, 1)
        cv2.circle(canvas, pt1, 4, color, -1)
        cv2.circle(canvas, pt2_shifted, 4, color, -1)

    return canvas


@router.post("", response_model=MatchingResponse)
async def match_features(img1_file: UploadFile = File(...), img2_file: UploadFile = File(...)):
    """
    Accepts two images.
    Runs the Lowe-ratio sweep, returns match counts per ratio + inlier ratio
    at the recommended (0.75) ratio + visualization.
    """
    allowed_exts = {".jpg", ".jpeg", ".png"}

    for file in [img1_file, img2_file]:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {file.filename}. Only JPG, JPEG, and PNG are allowed."
            )

    # Decode first image
    contents1 = await img1_file.read()
    nparr1 = np.frombuffer(contents1, np.uint8)
    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

    # Decode second image
    contents2 = await img2_file.read()
    nparr2 = np.frombuffer(contents2, np.uint8)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise HTTPException(status_code=422, detail="Failed to decode one or both of the uploaded images.")

    # SIFT Compute
    kp1, desc1 = extract_sift(img1)
    kp2, desc2 = extract_sift(img2)

    # Sweep Lowe ratios
    ratios = CONFIG.task2.lowe_ratios
    sweep_results = sweep_lowe_ratios(desc1, desc2, ratios)

    ratios_sweep_res = {}
    default_ratio = CONFIG.task2.default_lowe_ratio
    default_matches = []
    default_inliers_mask = []
    default_inlier_ratio = 0.0

    for ratio in ratios:
        good = sweep_results[ratio]
        inlier_stats = estimate_inlier_ratio(kp1, kp2, good, ransac_thresh=5.0)
        
        ratio_str = f"{ratio:.2f}"
        ratios_sweep_res[ratio_str] = MatchStats(
            match_count=len(good),
            ransac_inliers=inlier_stats["inliers"],
            inlier_ratio=inlier_stats["inlier_ratio"]
        )

        if abs(ratio - default_ratio) < 0.01:
            default_matches = good
            default_inliers_mask = inlier_stats["mask"]
            default_inlier_ratio = inlier_stats["inlier_ratio"]

    # Generate visualization
    vis_img = draw_matches_vis(img1, kp1, img2, kp2, default_matches, default_inliers_mask)
    _, encoded_img = cv2.imencode(".png", vis_img)
    vis_base64 = base64.b64encode(encoded_img).decode("utf-8")

    return MatchingResponse(
        ratios_sweep=ratios_sweep_res,
        matches_at_0_75=len(default_matches),
        inlier_ratio_at_0_75=default_inlier_ratio,
        visualization_base64=vis_base64
    )
