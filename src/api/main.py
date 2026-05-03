# src/api/main.py
# ============================================================
# FastAPI application entry point
# Endpoints: /analyze-chart · /retrieve-evidence
#            /generate-hcc-code · /validate-response
# ============================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import make_asgi_app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
 
from src.config import settings

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info(f"Starting {settings.project_name} [{settings.environment}]")
    # TODO Phase 3: pre-load FAISS index and reference data here
    yield
    logger.info("Shutting down...")

# ── App instance ─────────────────────────────────────────────
app = FastAPI(
    title="Clinical Decision Intelligence Platform",
    description=(
        "Production-grade RAG + Agentic AI system for HCC coding, "
        "clinical evidence retrieval, and care gap analysis."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── Middleware ───────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
 
  
# ── Prometheus metrics endpoint ───────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Health check ─────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": "0.1.0",
    }
 
 
@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Clinical Decision Intelligence Platform API",
        "docs": "/docs",
        "health": "/health",
    }

# ── Routers ───────────────────────────────────────────────────
from src.api.routers import analyze, evidence, hcc_code, validate
 
app.include_router(analyze.router,   prefix="/api/v1", tags=["Chart Analysis"])
app.include_router(evidence.router,  prefix="/api/v1", tags=["Evidence Retrieval"])
app.include_router(hcc_code.router,  prefix="/api/v1", tags=["HCC Coding"])
app.include_router(validate.router,  prefix="/api/v1", tags=["Validation"])
 
# ── Global exception handler ─────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )