# ─────────────────────────────────────────────────────────────────────────────
# CardioShield – Multi-Stage Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: Build React Frontend SPA
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend with Verified TenSEAL CKKS
FROM python:3.11-slim AS backend

# Install system compilation dependencies required for TenSEAL (C++ SEAL library)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    protobuf-compiler \
    libprotobuf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code & HE engine
COPY backend/ ./backend/
COPY he_engine.py model_trainer.py test_pipeline.py conftest.py ./
COPY tests/ ./tests/

# Copy built frontend SPA from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Train and generate model artifacts if not present
RUN python model_trainer.py

# Verify HE engine during container build
RUN python he_engine.py

# Run full pipeline verification test
RUN python test_pipeline.py

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
