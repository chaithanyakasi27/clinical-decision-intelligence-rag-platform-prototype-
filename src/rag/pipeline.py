# ============================================================
# Core RAG Pipeline
#
# What this does:
#   Orchestrates the full RAG flow:
#   query → retrieve → assemble context → Claude → parse JSON
#
# Four pipeline methods (matching the 4 API endpoints):
#   analyze_chart()       → POST /analyze-chart
#   retrieve_evidence()   → POST /retrieve-evidence
#   generate_hcc_code()   → POST /generate-hcc-code
#   validate_response()   → POST /validate-response
#
# Guardrails built in:
#   - JSON parsing with fallback
#   - Hallucination check (evidence must appear in context)
#   - Error handling — never crashes the API
# ============================================================
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from src.config import settings
from src.rag.prompts import (
    CLINICAL_SYSTEM_PROMPT,
    HCC_CODING_PROMPT,
    EVIDENCE_RETRIEVAL_PROMPT,
    CHART_ANALYSIS_PROMPT,
    VALIDATION_PROMPT,
    EXPLANATION_PROMPT,
    PROMPT_VERSION,
)
from src.retrieval.retriever import HybridRetriever, RetrievalResult

# ── Response dataclass ────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Structured response from the RAG pipeline.

    Attributes:
        query            : original user query
        answer           : parsed JSON dict from Claude
        raw_answer       : raw text Claude returned
        retrieved_chunks : passages used as context
        context_used     : assembled context string sent to Claude
        model_used       : Claude model name
        prompt_version   : which prompt version was used
        token_count      : tokens used (for cost tracking)
        error            : error message if something failed
    """
    query            : str
    answer           : dict[str, Any]
    raw_answer       : str
    retrieved_chunks : list[RetrievalResult]
    context_used     : str
    model_used       : str
    prompt_version   : str
    token_count      : int = 0
    error            : Optional[str] = None
    @property
    def success(self) -> bool:
        """True if no error occurred."""
        return self.error is None

class ClinicalRAGPipeline:
    """
    Clinical RAG pipeline using Claude via Anthropic API.

    Usage:
        pipeline = ClinicalRAGPipeline(retriever)

        # Generate HCC codes
        response = pipeline.generate_hcc_code(
            "Find HCC conditions for this patient"
        )

        if response.success:
            print(response.answer["hcc_codes"])
        else:
            print(f"Error: {response.error}")
    """

    def __init__(
        self,
        retriever   : HybridRetriever,
        model       : str   = "us.anthropic.claude-sonnet-4-6",
        temperature : float = settings.openai_temperature,
        max_tokens  : int   = settings.openai_max_tokens,
    ):
        """
        Args:
            retriever  : HybridRetriever for finding relevant chunks
            model      : Claude model to use
            temperature: 0.0 for deterministic clinical coding
            max_tokens : max response length
        """
        self.retriever   = retriever
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._llm        = None  # lazy load
    @property
    def llm(self):
        if self._llm is None:
            from langchain_aws import ChatBedrock
            self._llm = ChatBedrock(
                model_id              = "us.anthropic.claude-sonnet-4-6",
                region_name           = settings.aws_default_region,
                aws_access_key_id     = settings.aws_access_key_id,
                aws_secret_access_key = settings.aws_secret_access_key,
                model_kwargs          = {
                    "temperature": self.temperature,
                    "max_tokens" : self.max_tokens,
                },
            )
            logger.info("Claude Sonnet 4.6 via AWS Bedrock loaded")
        return self._llm
    
    # ── Public pipeline methods ───────────────────────────────
    # Each method maps to one FastAPI endpoint.

    def generate_hcc_code(
        self,
        query : str,
        top_k : int = 5,
    ) -> RAGResponse:
        """Generate HCC codes — POST /generate-hcc-code"""
        return self._run(
            query            = query,
            prompt_template  = HCC_CODING_PROMPT,
            top_k            = top_k,
        )
    
    def retrieve_evidence(
        self,
        query : str,
        top_k : int = 5,
    ) -> RAGResponse:
        """Retrieve clinical evidence — POST /retrieve-evidence"""
        return self._run(
            query           = query,
            prompt_template = EVIDENCE_RETRIEVAL_PROMPT,
            top_k           = top_k,
        )

    def analyze_chart(
        self,
        query : str,
        top_k : int = 8,
    ) -> RAGResponse:
        """Analyze full chart — POST /analyze-chart"""
        return self._run(
            query           = query,
            prompt_template = CHART_ANALYSIS_PROMPT,
            top_k           = top_k,
        )

    def validate_response(
        self,
        query         : str,
        coding_output : dict,
        top_k         : int = 5,
    ) -> RAGResponse:
        """Validate coding output — POST /validate-response"""
        results = self.retriever.retrieve(query, top_k=top_k)
        context = self._assemble_context(results)

        prompt = VALIDATION_PROMPT.format(
            context       = context,
            coding_output = json.dumps(coding_output, indent=2),
        )
        return self._call_llm(
            query            = query,
            prompt           = prompt,
            retrieved_chunks = results,
            context          = context,
        )

    def explain_coding(
        self,
        query         : str,
        coding_output : dict,
        top_k         : int = 4,
    ) -> RAGResponse:
        """Generate explanation — used by explanation agent"""
        results = self.retriever.retrieve(query, top_k=top_k)
        context = self._assemble_context(results)

        prompt = EXPLANATION_PROMPT.format(
            context       = context,
            coding_output = json.dumps(coding_output, indent=2),
        )
        return self._call_llm(
            query            = query,
            prompt           = prompt,
            retrieved_chunks = results,
            context          = context,
        )
    
    # ── Core pipeline ─────────────────────────────────────────

    def _run(
        self,
        query           : str,
        prompt_template : str,
        top_k           : int,
    ) -> RAGResponse:
        """
        Shared RAG pipeline used by all public methods.

        Steps:
        1. Retrieve top_k relevant chunks from vector store
        2. Assemble chunks into a context string
        3. Format the prompt with context + query
        4. Call Claude
        5. Parse JSON response
        6. Apply guardrails
        """
        # ── Step 1: Retrieve ──────────────────────────────────
        results = self.retriever.retrieve(query, top_k=top_k)

        if not results:
            logger.warning(f"No chunks retrieved for: {query!r}")

        # ── Step 2: Assemble context ──────────────────────────
        context = self._assemble_context(results)

        # ── Step 3: Format prompt ─────────────────────────────
        prompt = prompt_template.format(
            context = context,
            query   = query,
        )

        # ── Step 4-6: Call LLM and parse ──────────────────────
        return self._call_llm(
            query            = query,
            prompt           = prompt,
            retrieved_chunks = results,
            context          = context,
        )

    def _call_llm(
        self,
        query           : str,
        prompt          : str,
        retrieved_chunks: list[RetrievalResult],
        context         : str,
    ) -> RAGResponse:
        """
        Call Claude and parse the response.

        If the API call fails (e.g. no credits),
        returns a RAGResponse with error set instead of crashing.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=CLINICAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            # Call Claude
            response  = self.llm.invoke(messages)
            raw_text  = response.content

            # Parse JSON from response
            parsed    = self._parse_json(raw_text)

            # Apply hallucination guardrail
            if not self._passes_hallucination_check(parsed, context):
                logger.warning("Hallucination check flagged response")
                parsed["_hallucination_warning"] = (
                    "Response may contain claims not in retrieved context"
                )

            # Get token count for cost tracking
            token_count = 0
            if hasattr(response, "response_metadata"):
                token_count = (
                    response.response_metadata
                    .get("usage", {})
                    .get("input_tokens", 0)
                )

            return RAGResponse(
                query            = query,
                answer           = parsed,
                raw_answer       = raw_text,
                retrieved_chunks = retrieved_chunks,
                context_used     = context,
                model_used       = self.model,
                prompt_version   = PROMPT_VERSION,
                token_count      = token_count,
            )

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return RAGResponse(
                query            = query,
                answer           = {},
                raw_answer       = "",
                retrieved_chunks = retrieved_chunks,
                context_used     = context,
                model_used       = self.model,
                prompt_version   = PROMPT_VERSION,
                error            = str(e),
            )

    def _assemble_context(
        self,
        results: list[RetrievalResult],
    ) -> str:
        """
        Build context string from retrieval results.

        Priority sections (Assessment and Plan) appear first
        so Claude sees the most HCC-relevant content first.

        Format:
            --- Passage 1 [Section: Assessment And Plan] [Source: file.pdf] ---
            Patient has Type 2 diabetes E11.65...

            --- Passage 2 [Section: Past Medical History] [Source: file.pdf] ---
            History of CHF...
        """
        if not results:
            return "No relevant clinical documentation found."

        # Sort: assessment/plan sections first
        priority = {
            "assessment and plan",
            "assessment",
            "plan",
            "diagnosis",
            "hcc risk",
        }

        sorted_results = sorted(
            results,
            key=lambda r: (
                0 if (r.section or "") in priority else 1,
                -r.final_score,
            ),
        )

        parts = []
        for i, result in enumerate(sorted_results, 1):
            section_label = (
                f"[Section: {result.section.title()}]"
                if result.section
                else "[Section: General]"
            )
            source = f"[Source: {result.chunk.file_name}]"
            parts.append(
                f"--- Passage {i} {section_label} {source} ---\n"
                f"{result.text}"
            )

        return "\n\n".join(parts)

    def _parse_json(self, raw_text: str) -> dict:
        """
        Parse JSON from Claude response.

        Claude sometimes wraps JSON in markdown code fences:
```json
            {"key": "value"}
```
        We strip those before parsing.
        """
        text = raw_text.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            text  = "\n".join(lines[1:-1])

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON block within the text
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            logger.warning("Could not parse JSON from Claude response")
            return {
                "raw_response" : raw_text,
                "_parse_error" : True,
            }

    def _passes_hallucination_check(
        self,
        parsed : dict,
        context: str,
    ) -> bool:
        """
        Basic hallucination guard.

        Checks that claimed ICD-10 codes have some evidence
        in the retrieved context. Not foolproof but catches
        obvious hallucinations where Claude invents codes
        not mentioned anywhere in the clinical notes.
        """
        if "_parse_error" in parsed:
            return True  # can't check if parsing failed

        hcc_codes = parsed.get("hcc_codes", [])
        if not hcc_codes:
            return True  # nothing to check

        context_lower = context.lower()
        flags = 0

        for code_entry in hcc_codes:
            evidence = code_entry.get("supporting_evidence", "").lower()
            # Check if key words from the evidence appear in context
            words = set(evidence.split()) - {"the", "a", "an", "is", "was", "with"}
            if words:
                overlap = any(w in context_lower for w in words)
                if not overlap:
                    flags += 1

        # Fail if more than half the codes have no context support
        return flags <= len(hcc_codes) / 2