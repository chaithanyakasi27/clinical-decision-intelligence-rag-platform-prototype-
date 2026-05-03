# src/api/routers/analyze.py
# ============================================================
# POST /api/v1/analyze-chart
#
# What this does:
#   Accepts a clinical query, retrieves relevant passages
#   from the FAISS index, sends them to Claude for analysis,
#   returns a structured chart analysis for HCC coding.
#
# Flow:
#   Request → retriever → context → Claude → JSON response
# ============================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

router = APIRouter()


# ── Request / Response models ─────────────────────────────────
# Pydantic validates incoming JSON automatically.
# If a required field is missing, FastAPI returns 422.

class AnalyzeChartRequest(BaseModel):
    query     : str            = Field(
        ...,
        description="Clinical query — e.g. 'Analyze this patient for HCC conditions'",
        example="Analyze chart for HCC risk adjustment coding opportunities",
    )
    patient_id: Optional[str] = Field(
        None,
        description="Optional patient identifier for tracking",
        example="PT001",
    )


class AnalyzeChartResponse(BaseModel):
    patient_id      : Optional[str]
    query           : str
    analysis        : dict   # structured JSON from Claude
    chunks_retrieved: int    # how many passages were used
    status          : str


@router.post(
    "/analyze-chart",
    response_model=AnalyzeChartResponse,
    summary="Analyze clinical chart for HCC coding",
    description=(
        "Retrieves relevant clinical documentation and returns "
        "a structured HCC risk adjustment analysis including "
        "active conditions, suspect conditions, and care gaps."
    ),
)
async def analyze_chart(request: AnalyzeChartRequest):
    """
    Analyze a clinical chart for HCC risk adjustment.

    This endpoint:
    1. Embeds the query using BGE model
    2. Searches FAISS for relevant clinical passages
    3. Assembles context from top results
    4. Sends context + query to Claude with CoT prompting
    5. Returns structured JSON analysis
    """
    logger.info(f"POST /analyze-chart | patient={request.patient_id}")

    try:
        from src.api.dependencies import get_rag_pipeline

        pipeline = get_rag_pipeline()
        response = pipeline.analyze_chart(request.query)

        if not response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {response.error}"
            )

        return AnalyzeChartResponse(
            patient_id       = request.patient_id,
            query            = request.query,
            analysis         = response.answer,
            chunks_retrieved = len(response.retrieved_chunks),
            status           = "success",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))