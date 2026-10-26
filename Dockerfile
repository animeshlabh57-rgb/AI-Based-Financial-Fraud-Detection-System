# ==============================================================================
# FraudGuard AI - Production Docker Container
# Multi-purpose container supporting FastAPI REST API (8000) & Streamlit (8501)
# ==============================================================================
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MPLCONFIGDIR=/app/.cache/matplotlib

WORKDIR /app

# Install system dependencies (build tools, libgomp for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-trained model artifacts
COPY . .

# Create cache and runtime directories
RUN mkdir -p /app/.cache/matplotlib /app/storage /app/logs /app/results

EXPOSE 8000 8501

# Health check to ensure microservice responsiveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: launch FastAPI microservice
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
