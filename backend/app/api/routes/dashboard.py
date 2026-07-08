"""Dashboard routes — Task 5."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()

import os

ARTIFACTS_DIR_ENV = os.environ.get("PERCV_ARTIFACTS_DIR")
if ARTIFACTS_DIR_ENV:
    ARTIFACTS_DIR = Path(ARTIFACTS_DIR_ENV)
else:
    ARTIFACTS_DIR = Path(__file__).resolve().parents[4] / "artifacts"


class DashboardResponse(BaseModel):
    baseline_metrics: Dict[str, Any]
    backbone_comparison: Dict[str, Any]


@router.get("", response_model=DashboardResponse)
async def get_dashboard_summary():
    """
    Returns the baseline metrics (read from artifacts/baseline_metrics.json directly)
    along with the CNN backbone comparison table.
    """
    metrics_path = ARTIFACTS_DIR / "baseline_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Baseline metrics snapshot file missing at: {metrics_path.resolve()}"
        )

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse baseline_metrics.json: {e}"
        )

    # Compile backbone comparison table
    backbone_comparison = {
        "resnet18": baseline_data.get("task4", {}).get("resnet18", {}),
        "mobilenetv2": baseline_data.get("task4", {}).get("mobilenetv2", {})
    }

    return DashboardResponse(
        baseline_metrics=baseline_data,
        backbone_comparison=backbone_comparison
    )
