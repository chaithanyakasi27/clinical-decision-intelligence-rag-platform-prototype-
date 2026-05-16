# tests/test_agents.py
# Unit tests for AgentState structure and should_retry_coding logic
# Run with: python -m pytest tests/test_agents.py -v
#
# Design: no LLM calls, no FAISS, no pipeline instantiation.
# - AgentState is a TypedDict — we verify its key schema via get_type_hints.
# - should_retry_coding is a pure function with no side effects.

import pytest
from typing import get_type_hints

from src.agents.graph import AgentState, should_retry_coding


# ── AgentState schema tests ───────────────────────────────────

class TestAgentState:

    REQUIRED_KEYS = {
        # Input
        "query",
        "patient_id",
        # Message history
        "messages",
        # Evidence agent
        "evidence",
        "evidence_found",
        # HCC coding agent
        "hcc_codes",
        "total_raf_score",
        "coding_notes",
        # Validation agent
        "validation_passed",
        "validated_codes",
        "compliance_flags",
        # Explanation agent
        "explanation",
        "code_explanations",
        # Control flow
        "current_agent",
        "retry_count",
        "error",
        "final_response",
    }

    def test_agent_state_has_all_required_keys(self):
        hints = get_type_hints(AgentState)
        missing = self.REQUIRED_KEYS - set(hints.keys())
        assert not missing, f"AgentState is missing keys: {missing}"

    def test_agent_state_key_count(self):
        hints = get_type_hints(AgentState)
        assert len(hints) >= len(self.REQUIRED_KEYS), (
            f"Expected at least {len(self.REQUIRED_KEYS)} keys, "
            f"got {len(hints)}"
        )

    def test_agent_state_initial_dict_is_valid(self):
        """Confirm that a well-formed initial state dict matches all expected keys."""
        from langchain_core.messages import HumanMessage
        state = {
            "query": "Generate HCC codes for patient",
            "patient_id": "PT001",
            "messages": [HumanMessage(content="Generate HCC codes for patient")],
            "evidence": {},
            "evidence_found": False,
            "hcc_codes": [],
            "total_raf_score": 0.0,
            "coding_notes": "",
            "validation_passed": False,
            "validated_codes": [],
            "compliance_flags": [],
            "explanation": "",
            "code_explanations": [],
            "current_agent": "evidence_retrieval_agent",
            "retry_count": 0,
            "error": None,
            "final_response": None,
        }
        hints = get_type_hints(AgentState)
        for key in hints:
            assert key in state, f"Initial state is missing key: {key}"


# ── should_retry_coding tests ─────────────────────────────────

class TestShouldRetryCoding:
    """
    should_retry_coding logic:
      retry   → validation failed AND has codes AND retry_count < 1
      continue → anything else (passed, no codes, already retried)
    """

    def _state(self, validation_passed=False, hcc_codes=None, retry_count=0):
        """Helper: build minimal state dict for should_retry_coding."""
        return {
            "validation_passed": validation_passed,
            "hcc_codes": hcc_codes if hcc_codes is not None else [],
            "retry_count": retry_count,
        }

    def test_retry_when_validation_failed_has_codes_first_attempt(self):
        state = self._state(
            validation_passed=False,
            hcc_codes=[{"icd10_code": "E11.9", "hcc_category": 37}],
            retry_count=0,
        )
        assert should_retry_coding(state) == "retry"

    def test_continue_when_validation_passed(self):
        state = self._state(
            validation_passed=True,
            hcc_codes=[{"icd10_code": "E11.9", "hcc_category": 37}],
            retry_count=0,
        )
        assert should_retry_coding(state) == "continue"

    def test_continue_when_no_hcc_codes(self):
        state = self._state(
            validation_passed=False,
            hcc_codes=[],
            retry_count=0,
        )
        assert should_retry_coding(state) == "continue"

    def test_continue_when_retry_count_at_limit(self):
        """retry_count >= 1 means we already retried once — do not retry again."""
        state = self._state(
            validation_passed=False,
            hcc_codes=[{"icd10_code": "I50.9", "hcc_category": 85}],
            retry_count=1,
        )
        assert should_retry_coding(state) == "continue"

    def test_continue_when_retry_count_exceeds_limit(self):
        state = self._state(
            validation_passed=False,
            hcc_codes=[{"icd10_code": "I50.9", "hcc_category": 85}],
            retry_count=5,
        )
        assert should_retry_coding(state) == "continue"

    def test_continue_when_validation_passed_and_no_codes(self):
        state = self._state(
            validation_passed=True,
            hcc_codes=[],
            retry_count=0,
        )
        assert should_retry_coding(state) == "continue"

    def test_retry_with_multiple_codes(self):
        """Multiple codes present should still trigger retry when conditions met."""
        state = self._state(
            validation_passed=False,
            hcc_codes=[
                {"icd10_code": "E11.65", "hcc_category": 37},
                {"icd10_code": "N18.3", "hcc_category": 138},
                {"icd10_code": "I50.32", "hcc_category": 85},
            ],
            retry_count=0,
        )
        assert should_retry_coding(state) == "retry"

    def test_return_type_is_string(self):
        state = self._state(validation_passed=True, hcc_codes=[], retry_count=0)
        result = should_retry_coding(state)
        assert isinstance(result, str)
        assert result in {"retry", "continue"}