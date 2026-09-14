# Dockerfile - CrowdSafe AI Model Serving Container
# ----------------------------------------------------
# Build:  docker build -t crowdsafe-ai:latest .
# Run:    docker run -p 5000:5000 crowdsafe-ai:latest
# Test:   curl http://localhost:5000/health

FROM python:3.10-slim

WORKDIR /app

# System deps needed by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 5000

# Docker-level healthcheck, hits the /health endpoint from api.py
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests,sys; sys.exit(0) if requests.get('http://localhost:5000/health').ok else sys.exit(1)"

CMD ["python", "app/api.py"]
