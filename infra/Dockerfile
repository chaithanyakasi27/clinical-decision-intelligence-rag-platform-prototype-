# ============================================================
# infra/Dockerfile — Multi-stage build
# Stage 1: base        → shared Python layer
# Stage 2: development → hot-reload dev server
# Stage 3: production  → lean image for ECS Fargate
# ============================================================

# ── Stage 1: Base ────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies for PyMuPDF, WeasyPrint, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── Stage 2: Development ─────────────────────────────────────
FROM base AS development

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--log-level", "debug"]

# ── Stage 3: Production ──────────────────────────────────────
FROM base AS production

# Copy only source — no notebooks, no data, no infra
COPY src/ ./src/
COPY data/reference/ ./data/reference/

# Non-root user for security
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser
USER appuser

EXPOSE 8000

# Gunicorn with uvicorn workers for production
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--access-log"]