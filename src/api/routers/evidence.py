# src/api/routers/evidence.py
# ============================================================
# POST /api/v1/retrieve-evidence
#
# What this does:
#   Retrieves supporting clinical evidence passages for
#   a given diagnosis or coding question.
#   Returns ranked passages with source, section, score.
#
# Example use:
#   "Find clinical evidence supporting E11.65 coding"
#   → Returns passages from Assessment and Plan sections
#     that mention diabetes with hyperglycemia
# ============================================================

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

from src.monitoring.metrics import record_request, record_retrieval, API_LATENCY_SECONDS

router = APIRouter()


class EvidenceRequest(BaseModel):
    query     : str            = Field(
        ...,
        description="What clinical evidence to retrieve",
        example="Find evidence supporting Type 2 diabetes diagnosis",
    )
    patient_id: Optional[str] = Field(None, example="PT001")
    top_k     : int            = Field(
        5,
        ge=1,
        le=20,
        description="Number of evidence passages to return",
    )


class EvidenceResponse(BaseModel):
    patient_id      : Optional[str]
    query           : str
    evidence        : dict         # structured evidence from Claude
    retrieval_hits  : list[dict]   # raw retrieval results with scores
    chunks_retrieved: int
    status          : str


@router.post(
    "/retrieve-evidence",
    response_model=EvidenceResponse,
    summary="Retrieve supporting clinical evidence",
    description=(
        "Searches clinical documentation for passages that support "
        "a given diagnosis or HCC coding decision. Returns ranked "
        "passages with section, source document, and relevance score."
    ),
)
async def retrieve_evidence(request: EvidenceRequest):
    """
    Retrieve supporting clinical evidence for a diagnosis or query.

    Returns:
        - evidence: structured summary from Claude
        - retrieval_hits: raw top passages with scores
    """
    logger.info(
        f"POST /retrieve-evidence | "
        f"query={request.query[:50]!r} | "
        f"top_k={request.top_k}"
    )

    _t0 = time.perf_counter()
    _status = "success"
    try:
        from src.api.dependencies import get_rag_pipeline

        pipeline = get_rag_pipeline()
        response = pipeline.retrieve_evidence(
            query  = request.query,
            top_k  = request.top_k,
        )

        if not response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Evidence retrieval failed: {response.error}"
            )

        retrieval_hits = [
            {
                "rank"    : r.rank,
                "score"   : round(r.final_score, 4),
                "section" : r.chunk.section or "unknown",
                "file"    : r.chunk.file_name,
                "text"    : r.chunk.text[:300],
            }
            for r in response.retrieved_chunks
        ]

        record_retrieval(len(response.retrieved_chunks))
        return EvidenceResponse(
            patient_id       = request.patient_id,
            query            = request.query,
            evidence         = response.answer,
            retrieval_hits   = retrieval_hits,
            chunks_retrieved = len(response.retrieved_chunks),
            status           = "success",
        )

    except HTTPException:
        _status = "error"
        raise
    except Exception as e:
        _status = "error"
        logger.error(f"Evidence retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        record_request("retrieve_evidence", _status)
        API_LATENCY_SECONDS.labels(endpoint="retrieve_evidence").observe(time.perf_counter() - _t0)