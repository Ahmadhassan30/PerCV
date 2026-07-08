"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class LaneDetectionResponse(BaseModel):
    """Response schema for lane detection results."""
    quality_score: float
    num_lines: int
    config_label: str


class MatchingResponse(BaseModel):
    """Response schema for SIFT matching results."""
    keypoints_a: int
    keypoints_b: int
    matches: Dict[str, int]  # {ratio: count}
    inlier_ratio: float


class PanoramaResponse(BaseModel):
    """Response schema for panorama stitching results."""
    num_images: int
    inlier_ratios: Dict[str, float]
    output_path: str


class ClassifyResponse(BaseModel):
    """Response schema for CNN classification results."""
    predicted_class: str
    confidence: float
    gradcam_available: bool


class DashboardMetrics(BaseModel):
    """Response schema for dashboard baseline metrics."""
    task1: Dict
    task2: Dict
    task3: Dict
    task4: Dict
