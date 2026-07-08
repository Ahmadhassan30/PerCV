"""Lane Detection routes — Task 1."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/detect")
async def detect_lanes():
    """Run Canny + Hough lane detection on an uploaded image."""
    return {"status": "stub", "task": "lane_detection"}
