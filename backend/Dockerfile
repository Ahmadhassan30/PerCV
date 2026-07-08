# === Stage 1: Builder ===
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install standard packaging tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY backend/pyproject.toml ./
COPY percv_cv/ ./percv_cv/
COPY backend/app/ ./app/

# Install the project and dependencies to a custom prefix directory
RUN pip install --no-cache-dir --prefix=/install .

# === Stage 2: Runner ===
FROM python:3.12-slim AS runner

WORKDIR /app

# Install system dependencies required by OpenCV headless at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY percv_cv/ /app/percv_cv/
COPY backend/app/ /app/app/

# Setup a non-root user for security compliance
RUN useradd -u 1000 -m appuser && chown -R appuser:appuser /app
USER appuser

# Design Tradeoff Comment:
# We explicitly do NOT bake the artifacts/ directory (containing the large ~45MB .pt model checkpoints)
# directly into the Docker image to keep the image lightweight, speed up CI/CD pipelines, and adhere
# to the separation of code and weights. Instead, the artifacts/ folder must be mounted at runtime
# as a Docker volume (mounted to /app/artifacts).

EXPOSE 8000

# Smoke-test OpenCV importing at build time to prevent silent runtime failures
RUN python -c "import cv2; print('OpenCV successfully imported inside the slim container.')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
