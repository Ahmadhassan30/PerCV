"""PerCV Backend — FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import lanes, matching, panorama, classify, dashboard

app = FastAPI(
    title="PerCV API",
    description="Computer Vision Pipeline — Lane Detection, SIFT Matching, "
                "Panorama Stitching, CNN Classification & Grad-CAM",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lanes.router, prefix="/api/lanes", tags=["Lane Detection"])
app.include_router(matching.router, prefix="/api/matching", tags=["SIFT Matching"])
app.include_router(panorama.router, prefix="/api/panorama", tags=["Panorama Stitching"])
app.include_router(classify.router, prefix="/api/classify", tags=["CNN Classification"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])


@app.get("/health")
async def health_check():
    return {"status": "ok"}
