"""CNN Classification & Grad-CAM routes — Task 4."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/predict")
async def classify_image():
    """Run CNN inference + optional Grad-CAM on an uploaded image."""
    return {"status": "stub", "task": "cnn_classification"}
