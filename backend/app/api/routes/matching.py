"""SIFT Feature Matching routes — Task 2."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/match")
async def match_features():
    """Run SIFT extraction + Lowe ratio matching on an uploaded image pair."""
    return {"status": "stub", "task": "sift_matching"}
