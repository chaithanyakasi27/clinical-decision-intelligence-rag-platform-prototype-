# ============================================================
# POST /api/v1/validate-response
#
# What this does:
#   Takes HCC coding output from /generate-hcc-code and
#   validates it against the clinical documentation.
#
# Validation checks:
#   1. Each ICD-10 code has supporting documentation
#   2. Codes are at highest specificity level
#   3. HCC hierarchy rules are respected
#   4. RAF score calculation is correct
#   5. CMS V28 model compliance
#
# Returns: approved / modified / rejected per code
# ============================================================


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

router = APIRouter()

class ValidateRequest(BaseModel):
    query        : str            = Field(
        ...,
        description="Original clinical query used for coding",
        example="Generate HCC codes for patient chronic conditions",
    )
    coding_output: dict           = Field(
        ...,
        description="The HCC coding output to validate",
        example={
            "hcc_codes": [
                {
                    "icd10_code": "E11.65",
                    "description": "Type 2 diabetes with hyperglycemia",
                    "hcc_category": 19,
                    "raf_score": 0.104,
                }
            ],
            "total_raf_score": 0.104,
        },
    )
    patient_id   : Optional[str] = Field(None, example="PT001")

class ValidateResponse(BaseModel):
    patient_id      : Optional[str]
    validation      : dict   # {validation_passed, validated_codes, flags}
    chunks_retrieved: int
    status          : str

@router.post(
    "/validate-response",
    response_model=ValidateResponse,
    summary="Validate HCC coding output",
    description=(
        "Validates HCC coding output against clinical documentation. "
        "Checks documentation support, code specificity, HCC hierarchy "
        "compliance, and CMS V28 model adherence. "
        "Returns approved, modified, or rejected status per code."
    ),
)
async def validate_response(request: ValidateRequest):
    """
    Validate HCC coding output against clinical documentation.

    This is the quality gate before coding goes to the payer.
    Each code gets: APPROVED | MODIFIED | REJECTED
    """
    logger.info(
        f"POST /validate-response | "
        f"patient={request.patient_id} | "
        f"codes={len(request.coding_output.get('hcc_codes', []))}"
    )

    try:
        from src.api.dependencies import get_rag_pipeline

        pipeline = get_rag_pipeline()
        response = pipeline.validate_response(
            query         = request.query,
            coding_output = request.coding_output,
        )

        if not response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Validation failed: {response.error}"
            )

        return ValidateResponse(
            patient_id       = request.patient_id,
            validation       = response.answer,
            chunks_retrieved = len(response.retrieved_chunks),
            status           = "success",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))