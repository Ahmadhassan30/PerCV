"""Lane Detection routes — Task 1."""

import base64
from pathlib import Path
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any

from percv_cv.config import CONFIG
from percv_cv.lanes import preprocess, detect_edges, detect_lines, lanes_quality_score, run_lane_detection

router = APIRouter()


class LanePresetDetail(BaseModel):
    image_name: str
    num_lines: int
    quality_score: float


class LaneDetectionResponse(BaseModel):
    preset_mean_scores: Dict[str, float]
    overall_mean_quality_score: float
    overlay_base64: str
    results_by_preset: Dict[str, List[LanePresetDetail]]


@router.post("", response_model=LaneDetectionResponse)
async def detect_lanes(files: List[UploadFile] = File(...)):
    """
    Accepts one or more road images.
    Runs all three Canny presets, returns per-preset quality scores + overlay
    visualization (base64) for the balanced preset by default.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    allowed_exts = {".jpg", ".jpeg", ".png"}
    images_decoded = []
    image_paths_dummy = []

    # Temporary directory for image paths (needed because run_lane_detection expects image paths)
    # We can write uploaded files to temporary path or load directly.
    # To run on actual images, let's load them and also write to temporary files if required, or
    # let's adapt lanes.py to run on decoded images, or write them to temporary paths inside the function.
    # Writing to temporary files is extremely robust and fits run_lane_detection signature!
    import tempfile
    temp_dir = tempfile.TemporaryDirectory()

    try:
        for idx, file in enumerate(files):
            ext = Path(file.filename).suffix.lower()
            if ext not in allowed_exts:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported file type: {file.filename}. Only JPG, JPEG, and PNG are allowed."
                )

            # Read content
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=422, detail=f"Invalid image content: {file.filename}")

            # Save to temporary path for lanes.py runner
            temp_path = Path(temp_dir.name) / f"img_{idx}{ext}"
            with open(temp_path, "wb") as f:
                f.write(contents)
            image_paths_dummy.append(temp_path)
            images_decoded.append((img, temp_path.name))

        # Run pipeline
        results = run_lane_detection(image_paths_dummy, CONFIG.task1)

        # Generate overlay for the first image using the balanced Canny preset
        balanced_cfg = [p for p in CONFIG.task1.canny_threshold_pairs if p.label == "balanced"][0]
        sample_img, _ = images_decoded[0]
        
        gray = preprocess(sample_img, CONFIG.task1.gaussian_ksize, CONFIG.task1.gaussian_sigma)
        edges = detect_edges(gray, balanced_cfg.low, balanced_cfg.high)
        lines = detect_lines(
            edges,
            rho=CONFIG.task1.hough_rho,
            theta_deg=CONFIG.task1.hough_theta_deg,
            threshold=CONFIG.task1.hough_threshold,
            min_line_len=CONFIG.task1.hough_min_line_len,
            max_line_gap=CONFIG.task1.hough_max_line_gap
        )

        overlay = sample_img.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                angle = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi)
                if angle > 180.0:
                    angle -= 180.0
                
                # Highlight pass vs fail
                if (25.0 <= angle <= 75.0) or (105.0 <= angle <= 155.0):
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 0, 255)  # Red
                
                cv2.line(overlay, (x1, y1), (x2, y2), color, 2)

        # Encode overlay to base64
        _, encoded_img = cv2.imencode(".png", overlay)
        overlay_base64 = base64.b64encode(encoded_img).decode("utf-8")

        # Format results
        results_formatted = {}
        for preset, details in results["results_by_preset"].items():
            results_formatted[preset] = [
                LanePresetDetail(
                    image_name=Path(d["image_path"]).name,
                    num_lines=d["num_lines"],
                    quality_score=d["quality_score"]
                ) for d in details
            ]

        return LaneDetectionResponse(
            preset_mean_scores=results["preset_mean_scores"],
            overall_mean_quality_score=results["overall_mean_quality_score"],
            overlay_base64=overlay_base64,
            results_by_preset=results_formatted
        )

    finally:
        temp_dir.cleanup()
