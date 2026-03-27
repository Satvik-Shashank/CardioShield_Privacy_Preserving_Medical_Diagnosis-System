# ─────────────────────────────────────────────────────────────────────────────
# CardioShield – Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Builds BOTH the Flask backend and Streamlit frontend in one image.
# Use docker-compose.yml to run them as separate services.
#
# Build:   docker build -t cardioshield .
# Run:     docker-compose up
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps for building tenseal and other C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake protobuf-compiler libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY . .

# Train model artefacts if they don't exist
RUN python -c "\
import os;\
if not os.path.exists('artefacts/model.pkl'):\
    print('Training model artefacts...');\
    exec(open('model_trainer.py').read())\
else:\
    print('Artefacts already exist')\
" || true

# Create data directory for backend database
RUN mkdir -p /app/backend/data

# Expose ports: 5000 (backend), 8501 (frontend)
EXPOSE 5000 8501

# Default command — overridden by docker-compose per service
CMD ["python", "-m", "backend.app"]
