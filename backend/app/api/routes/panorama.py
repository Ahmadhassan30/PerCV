"""Panorama Stitching routes — Task 3."""

import base64
import uuid
import logging
from pathlib import Path
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

from percv_cv.config import CONFIG
from percv_cv.panorama import stitch_panorama

router = APIRouter()
logger = logging.getLogger("percv-backend")

# Global thread-safe (in-memory) job store
PANORAMA_JOBS: Dict[str, Dict[str, Any]] = {}


class PanoramaJobSubmission(BaseModel):
    job_id: str
    status: str


class PanoramaJobResult(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    panorama_base64: Optional[str] = None
    pair_metrics: Optional[Dict[str, Any]] = None
    average_inlier_ratio: Optional[float] = None


def run_stitching_background(job_id: str, files_data: List[tuple[str, bytes]], anchor_param: str):
    """Orchestrate matching SIFT features, homography RANSAC, and blending in a non-blocking background task."""
    PANORAMA_JOBS[job_id]["status"] = "running"
    
    try:
        # 1. Decode images
        images = {}
        for idx, (filename, content) in enumerate(files_data):
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image file: {filename}")
            
            # Map name label dynamically
            # If 3 files, label based on index or filename substring
            if len(files_data) == 3:
                if "left" in filename.lower() or idx == 0:
                    key = "left"
                elif "right" in filename.lower() or idx == 2:
                    key = "right"
                else:
                    key = "middle"
            else:
                if "left" in filename.lower() or idx == 0:
                    key = "left"
                else:
                    key = "right"
            images[key] = img

        # Determine anchor key
        anchor_key = "middle"
        if len(images) == 2:
            anchor_key = "right" if "right" in anchor_param.lower() or "right" in images else "left"
        else:
            anchor_key = "middle" if "middle" in anchor_param.lower() or "middle" in images else "middle"

        # Check if anchor exists in mapped keys
        if anchor_key not in images:
            # Fallback to any valid key
            anchor_key = list(images.keys())[0]

        # 2. Run stitching
        results = stitch_panorama(images, anchor_key, CONFIG.task3)

        # 3. Encode result to base64
        _, encoded_img = cv2.imencode(".png", results["panorama"])
        pano_base64 = base64.b64encode(encoded_img).decode("utf-8")

        # 4. Save results to job store
        PANORAMA_JOBS[job_id].update({
            "status": "completed",
            "panorama_base64": pano_base64,
            "pair_metrics": results["pair_metrics"],
            "average_inlier_ratio": results["average_inlier_ratio"]
        })
        logger.info(f"Panorama job {job_id} completed successfully.")

    except Exception as e:
        logger.error(f"Panorama job {job_id} failed: {e}")
        PANORAMA_JOBS[job_id].update({
            "status": "failed",
            "error": str(e)
        })


@router.post("", response_model=PanoramaJobSubmission, status_code=status.HTTP_202_ACCEPTED)
async def stitch_panorama_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    anchor: str = Form("middle")
):
    """
    Accepts 2 or 3 overlapping images + anchor name label.
    Starts background stitching task and returns job_id immediately (HTTP 202).
    """
    if len(files) not in [2, 3]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Wrong number of images for panorama. Exactly 2 or 3 overlapping images are required."
        )

    allowed_exts = {".jpg", ".jpeg", ".png"}
    files_data = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported media type: {file.filename}. Only JPG, JPEG, and PNG are allowed."
            )
        content = await file.read()
        files_data.append((file.filename, content))

    job_id = str(uuid.uuid4())
    PANORAMA_JOBS[job_id] = {
        "status": "pending",
        "error": None,
        "panorama_base64": None,
        "pair_metrics": None,
        "average_inlier_ratio": None
    }

    # Queue background task
    background_tasks.add_task(run_stitching_background, job_id, files_data, anchor)
    logger.info(f"Submitted background panorama stitching job: {job_id}")

    return PanoramaJobSubmission(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}", response_model=PanoramaJobResult)
async def get_panorama_job(job_id: str):
    """
    Poll the stitching job status. If completed, returns base64 panorama and metrics.
    """
    if job_id not in PANORAMA_JOBS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Panorama stitching job ID '{job_id}' not found."
        )

    job = PANORAMA_JOBS[job_id]
    return PanoramaJobResult(
        job_id=job_id,
        status=job["status"],
        error=job["error"],
        panorama_base64=job["panorama_base64"],
        pair_metrics=job["pair_metrics"],
        average_inlier_ratio=job["average_inlier_ratio"]
    )
