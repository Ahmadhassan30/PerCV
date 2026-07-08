"""Panorama Stitching routes — Task 3."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/stitch")
async def stitch_panorama():
    """Run from-scratch panorama stitching on uploaded overlapping frames."""
    return {"status": "stub", "task": "panorama_stitching"}
