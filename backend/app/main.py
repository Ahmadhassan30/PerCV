"""PerCV Backend — FastAPI application entry point."""

import time
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import lanes, matching, panorama, classify, dashboard
from app.core.models_state import init_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("percv-backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing cached CNN checkpoints at startup...")
    init_models()
    yield
    # Shutdown actions
    logger.info("Shutting down PerCV FastAPI application.")


app = FastAPI(
    title="PerCV API",
    description="Computer Vision Pipeline — Lane Detection, SIFT Matching, "
                "Panorama Stitching, CNN Classification & Grad-CAM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Structured Logging Middleware
@app.middleware("http")
async def add_structured_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    latency = time.time() - start_time
    route = request.url.path
    method = request.method
    status_code = response.status_code
    
    # Log structured message
    logger.info(
        f"request_id={request_id} method={method} route={route} "
        f"status={status_code} latency={latency:.4f}s"
    )
    
    # Inject request_id into headers
    response.headers["X-Request-ID"] = request_id
    return response


# Include routers with API v1 prefixes matching real task identities
app.include_router(lanes.router, prefix="/api/v1/lanes", tags=["Lane Detection"])
app.include_router(matching.router, prefix="/api/v1/match", tags=["SIFT Matching"])
app.include_router(panorama.router, prefix="/api/v1/panorama", tags=["Panorama Stitching"])
app.include_router(classify.router, prefix="/api/v1/classify", tags=["CNN Classification"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])


@app.get("/health")
async def health_check():
    return {"status": "ok"}
