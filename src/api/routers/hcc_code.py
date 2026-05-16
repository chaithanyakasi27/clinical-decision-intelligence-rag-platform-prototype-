# POST /api/v1/generate-hcc-code
#
# What this does:
#   Core endpoint of the platform.
#   Takes a clinical query, retrieves relevant documentation,
#   uses Claude with Chain-of-Thought prompting to identify
#   HCC-relevant ICD-10 codes, categories, and RAF scores.
#
# Example use:
#   "Generate HCC codes for patient with diabetes and CHF"
#   → Returns: E11.65 (HCC 19, RAF 0.104), I50.9 (HCC 85, RAF 0.323)
#   → Total RAF score: 0.427
# ============================================================

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

from src.monitoring.metrics import record_request, record_retrieval, API_LATENCY_SECONDS, AGENT_PIPELINE_TOTAL

router = APIRouter()


class HCCCodeRequest(BaseModel):
    query     : str            = Field(
        ...,
        description="Clinical query for HCC code generation",
        example="Generate HCC codes for this patient's chronic conditions",
    )
    patient_id: Optional[str] = Field(None, example="PT001")
    use_agents: bool           = Field(
        False,
        description=(
            "True = full 4-agent LangGraph pipeline (slower, more accurate). "
            "False = single RAG call (faster, good for testing)."
        ),
    )

class HCCCodeResponse(BaseModel):
    patient_id       : Optional[str]
    query            : str
    hcc_codes        : list[dict]  # list of {icd10_code, hcc_category, raf_score...}
    total_raf_score  : float       # sum of all RAF scores
    coding_notes     : str         # caveats from the coding agent
    validation_passed: bool        # True after validation agent runs
    explanation      : str         # plain-language explanation
    chunks_retrieved : int
    status           : str

@router.post(
    "/generate-hcc-code",
    response_model=HCCCodeResponse,
    summary="Generate HCC codes from clinical documentation",
    description=(
        "Retrieves relevant clinical passages and applies "
        "Chain-of-Thought reasoning to identify ICD-10 codes, "
        "HCC categories, and Risk Adjustment Factor (RAF) scores "
        "per CMS HCC V28 model guidelines."
    ),
)
async def generate_hcc_code(request: HCCCodeRequest):
    """
    Generate HCC codes from clinical documentation.

    Chain-of-Thought steps (done by Claude):
    1. Identify all chronic conditions in retrieved passages
    2. Map each condition to the most specific ICD-10 code
    3. Map each ICD-10 to its HCC category and RAF score
    4. Validate against HCC hierarchy rules
    5. Return structured coding output
    """
    logger.info(
        f"POST /generate-hcc-code | "
        f"patient={request.patient_id} | "
        f"agents={request.use_agents}"
    )

    _t0 = time.perf_counter()
    _status = "success"
    try:
        from src.api.dependencies import get_rag_pipeline

        pipeline = get_rag_pipeline()

        if request.use_agents:
            # Full 4-agent LangGraph pipeline
            # Evidence → HCC Coding → Validation → Explanation
            # Requires Claude API credits
            from src.agents.graph import run_clinical_pipeline
            result = run_clinical_pipeline(
                pipeline   = pipeline,
                query      = request.query,
                patient_id = request.patient_id,
            )
            AGENT_PIPELINE_TOTAL.labels(outcome="success").inc()
            return HCCCodeResponse(
                patient_id        = result.get("patient_id"),
                query             = request.query,
                hcc_codes         = result.get("hcc_codes", []),
                total_raf_score   = result.get("total_raf_score", 0.0),
                coding_notes      = result.get("coding_notes", ""),
                validation_passed = result.get("validation", {}).get("passed", False),
                explanation       = result.get("explanation", ""),
                chunks_retrieved  = 0,
                status            = "success",
            )
        else:
            # Single RAG call — faster, good for testing
            response = pipeline.generate_hcc_code(request.query)

            if not response.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"HCC coding failed: {response.error}"
                )

            answer = response.answer
            record_retrieval(len(response.retrieved_chunks))
            return HCCCodeResponse(
                patient_id        = request.patient_id,
                query             = request.query,
                hcc_codes         = answer.get("hcc_codes", []),
                total_raf_score   = answer.get("total_raf_score", 0.0),
                coding_notes      = answer.get("coding_notes", ""),
                validation_passed = False,
                explanation       = "",
                chunks_retrieved  = len(response.retrieved_chunks),
                status            = "success",
            )

    except HTTPException:
        _status = "error"
        if request.use_agents:
            AGENT_PIPELINE_TOTAL.labels(outcome="failure").inc()
        raise
    except Exception as e:
        _status = "error"
        if request.use_agents:
            AGENT_PIPELINE_TOTAL.labels(outcome="failure").inc()
        logger.error(f"HCC coding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        record_request("generate_hcc_code", _status)
        API_LATENCY_SECONDS.labels(endpoint="generate_hcc_code").observe(time.perf_counter() - _t0)