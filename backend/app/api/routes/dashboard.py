"""Dashboard routes — serves MANIFEST metrics and baseline data for the frontend."""

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts"


@router.get("/metrics")
async def get_baseline_metrics():
    """Return baseline_metrics.json contents for the frontend dashboard."""
    metrics_path = ARTIFACTS_DIR / "baseline_metrics.json"
    if not metrics_path.exists():
        return {"error": "baseline_metrics.json not found", "path": str(metrics_path)}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@router.get("/manifest")
async def get_manifest():
    """Return MANIFEST.md as raw text for the frontend to render."""
    manifest_path = ARTIFACTS_DIR.parent / "MANIFEST.md"
    if not manifest_path.exists():
        return {"error": "MANIFEST.md not found"}
    return {"content": manifest_path.read_text(encoding="utf-8")}
