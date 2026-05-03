# LangGraph 4-Agent Clinical Workflow
# This module defines a complex multi-agent workflow using LangGraph to orchestrate
# a clinical decision intelligence pipeline for HCC coding.
#
# What this does:
#   Orchestrates 4 specialized agents in a stateful graph.
#   Each agent reads from shared state, does its job,
#   writes results back to state, passes to next agent.
#
# Agent flow:
#   Evidence Retrieval Agent
#         ↓
#   HCC Coding Agent
#         ↓
#   Validation Agent  ←─ retry loop if validation fails
#         ↓
#   Explanation Agent
#         ↓
#        END
#
# Why LangGraph:
#   Regular function calls lose state between steps.
#   LangGraph maintains a typed state object that all
#   agents share — like a clinical chart that gets filled
#   in as each agent completes its work.
#
# Why A2A collaboration:
#   "Architected Agentic AI workflows using LangChain and
#    LangGraph with stateful execution, Agent-to-Agent (A2A)
#    collaboration patterns"
# ============================================================

from __future__ import annotations

import json
from typing import Any, Annotated, TypedDict, Optional
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from loguru import logger


# ── Shared Agent State ────────────────────────────────────────
# This TypedDict is the "shared memory" between all 4 agents.
# Each agent reads what it needs and writes its output.
# operator.add means messages list is append-only.

class AgentState(TypedDict):
    """
    Shared state passed between all agents in the graph.

    Think of this like a clinical chart:
    - The patient arrives (query set)
    - Evidence agent fills in "evidence found"
    - Coding agent fills in "hcc_codes"
    - Validation agent fills in "validation_passed"
    - Explanation agent fills in "explanation"
    - Final response assembled at the end
    """
    # ── Input ─────────────────────────────────────────────────
    query     : str
    patient_id: Optional[str]

    # ── Message history (append-only) ─────────────────────────
    messages: Annotated[list[BaseMessage], operator.add]

    # ── Evidence agent output ─────────────────────────────────
    evidence      : dict[str, Any]
    evidence_found: bool

    # ── HCC coding agent output ───────────────────────────────
    hcc_codes      : list[dict[str, Any]]
    total_raf_score: float
    coding_notes   : str

    # ── Validation agent output ───────────────────────────────
    validation_passed: bool
    validated_codes  : list[dict[str, Any]]
    compliance_flags : list[str]

    # ── Explanation agent output ──────────────────────────────
    explanation      : str
    code_explanations: list[dict[str, Any]]

    # ── Control flow ──────────────────────────────────────────
    current_agent: str
    retry_count  : int
    error        : Optional[str]
    final_response: Optional[dict[str, Any]]


# ── Agent Node Functions ──────────────────────────────────────
# Each function takes the full AgentState and returns
# a PARTIAL state update (only the fields it changed).
# LangGraph merges the partial update into the full state.

class ClinicalAgentNodes:
    """
    Container for all 4 agent node functions.

    Each method is registered as a node in the LangGraph.
    Methods are stateless — all state lives in AgentState.
    """

    def __init__(self, pipeline):
        """
        Args:
            pipeline: ClinicalRAGPipeline instance
                      shared by all agents for LLM calls
        """
        self.pipeline = pipeline

    def evidence_retrieval_agent(self, state: AgentState) -> dict:
        """
        Agent 1 — Evidence Retrieval Agent

        Job: Find clinical evidence passages that support
             the HCC coding request.

        Input:  state["query"]
        Output: state["evidence"], state["evidence_found"]

        Why first:
            The HCC coding agent needs evidence context
            to make accurate coding decisions.
            This agent finds that context first.
        """
        logger.info("► Agent 1: Evidence Retrieval Agent running...")

        query          = state["query"]
        evidence_query = f"clinical evidence supporting diagnosis: {query}"

        response = self.pipeline.retrieve_evidence(
            evidence_query,
            top_k=8,
        )

        if response.success:
            evidence_data  = response.answer
            passage_count  = len(evidence_data.get("evidence_passages", []))
            evidence_found = evidence_data.get("evidence_found", False)

            logger.info(
                f"  Evidence found: {evidence_found} | "
                f"Passages: {passage_count}"
            )

            return {
                "evidence"      : evidence_data,
                "evidence_found": evidence_found,
                "current_agent" : "hcc_coding_agent",
                "messages"      : [AIMessage(
                    content = (
                        f"Evidence Retrieval complete. "
                        f"Found: {evidence_found}, "
                        f"Passages: {passage_count}"
                    ),
                    name="evidence_agent",
                )],
            }
        else:
            logger.error(f"  Evidence retrieval failed: {response.error}")
            return {
                "evidence"      : {},
                "evidence_found": False,
                "error"         : f"Evidence retrieval failed: {response.error}",
                "current_agent" : "hcc_coding_agent",
                "messages"      : [AIMessage(
                    content = f"Evidence retrieval error: {response.error}",
                    name="evidence_agent",
                )],
            }

    def hcc_coding_agent(self, state: AgentState) -> dict:
        """
        Agent 2 — HCC Coding Agent

        Job: Generate ICD-10 codes, HCC categories,
             and RAF scores from clinical documentation.

        Input:  state["query"] + state["evidence"]
        Output: state["hcc_codes"], state["total_raf_score"]

        Why second:
            Uses evidence from Agent 1 to enrich the query
            with specific clinical context before coding.
        """
        logger.info("► Agent 2: HCC Coding Agent running...")

        query = state["query"]

        # Enrich query with evidence summary if available
        if state.get("evidence_found") and state.get("evidence"):
            evidence_summary = state["evidence"].get(
                "evidence_summary", ""
            )
            if evidence_summary:
                query = f"{query}\n\nClinical evidence found:\n{evidence_summary}"

        response = self.pipeline.generate_hcc_code(query, top_k=6)

        if response.success:
            coding_data = response.answer
            hcc_codes   = coding_data.get("hcc_codes", [])
            total_raf   = coding_data.get("total_raf_score", 0.0)

            logger.info(
                f"  HCC codes identified: {len(hcc_codes)} | "
                f"Total RAF: {total_raf:.3f}"
            )

            return {
                "hcc_codes"      : hcc_codes,
                "total_raf_score": total_raf,
                "coding_notes"   : coding_data.get("coding_notes", ""),
                "current_agent"  : "validation_agent",
                "messages"       : [AIMessage(
                    content = (
                        f"HCC Coding complete. "
                        f"Codes: {len(hcc_codes)}, "
                        f"RAF: {total_raf:.3f}"
                    ),
                    name="hcc_coding_agent",
                )],
            }
        else:
            logger.error(f"  HCC coding failed: {response.error}")
            return {
                "hcc_codes"      : [],
                "total_raf_score": 0.0,
                "coding_notes"   : "",
                "error"          : f"HCC coding failed: {response.error}",
                "current_agent"  : "validation_agent",
                "messages"       : [AIMessage(
                    content = f"HCC coding error: {response.error}",
                    name="hcc_coding_agent",
                )],
            }

    def validation_agent(self, state: AgentState) -> dict:
        """
        Agent 3 — Validation Agent

        Job: Validate the HCC coding output against
             clinical documentation for accuracy and
             compliance with CMS guidelines.

        Input:  state["hcc_codes"] + state["query"]
        Output: state["validation_passed"],
                state["validated_codes"],
                state["compliance_flags"]

        Why third:
            Acts as a quality gate before the final
            explanation is generated. Can trigger a
            retry of the coding agent if validation fails.
        """
        logger.info("► Agent 3: Validation Agent running...")

        # Skip validation if no codes were generated
        if not state.get("hcc_codes"):
            logger.warning("  No HCC codes to validate — skipping")
            return {
                "validation_passed": False,
                "validated_codes"  : [],
                "compliance_flags" : ["No HCC codes were generated"],
                "current_agent"    : "explanation_agent",
                "messages"         : [AIMessage(
                    content = "Validation skipped — no codes to validate",
                    name="validation_agent",
                )],
            }

        # Build coding output dict for validation
        coding_output = {
            "hcc_codes"      : state["hcc_codes"],
            "total_raf_score": state["total_raf_score"],
            "coding_notes"   : state.get("coding_notes", ""),
        }

        response = self.pipeline.validate_response(
            query         = state["query"],
            coding_output = coding_output,
            top_k         = 5,
        )

        if response.success:
            validation_data   = response.answer
            passed            = validation_data.get("validation_passed", False)
            flags             = validation_data.get("compliance_flags", [])
            validated_codes   = validation_data.get("validated_codes", [])

            logger.info(
                f"  Validation: {'PASSED' if passed else 'FAILED'} | "
                f"Flags: {len(flags)}"
            )

            return {
                "validation_passed": passed,
                "validated_codes"  : validated_codes,
                "compliance_flags" : flags,
                "current_agent"    : "explanation_agent",
                "messages"         : [AIMessage(
                    content = (
                        f"Validation {'PASSED' if passed else 'FAILED'}. "
                        f"Flags: {len(flags)}"
                    ),
                    name="validation_agent",
                )],
            }
        else:
            return {
                "validation_passed": False,
                "validated_codes"  : [],
                "compliance_flags" : [f"Validation error: {response.error}"],
                "current_agent"    : "explanation_agent",
                "messages"         : [AIMessage(
                    content = f"Validation error: {response.error}",
                    name="validation_agent",
                )],
            }

    def explanation_agent(self, state: AgentState) -> dict:
        """
        Agent 4 — Explanation Agent

        Job: Generate a plain-language explanation of the
             coding decision for clinical and compliance teams.
             Assembles the final unified response.

        Input:  All previous agent outputs
        Output: state["explanation"],
                state["final_response"]

        Why last:
            Has access to all previous agent outputs.
            Produces the complete auditable response that
            gets returned to the API caller.
        """
        logger.info("► Agent 4: Explanation Agent running...")

        # Use validated codes if available,
        # fall back to original coded codes
        codes_to_explain = (
            state.get("validated_codes")
            or state.get("hcc_codes", [])
        )

        coding_output = {
            "hcc_codes"       : codes_to_explain,
            "total_raf_score" : state.get("total_raf_score", 0.0),
            "validation_passed": state.get("validation_passed", False),
            "compliance_flags" : state.get("compliance_flags", []),
        }

        response = self.pipeline.explain_coding(
            query         = state["query"],
            coding_output = coding_output,
            top_k         = 4,
        )

        explanation      = ""
        code_explanations= []

        if response.success:
            explanation       = response.answer.get("explanation", "")
            code_explanations = response.answer.get("code_explanations", [])

        # ── Assemble final response ───────────────────────────
        # This is what gets returned to the API caller
        final_response = {
            "patient_id"      : state.get("patient_id"),
            "query"           : state["query"],
            "hcc_codes"       : codes_to_explain,
            "total_raf_score" : state.get("total_raf_score", 0.0),
            "validation"      : {
                "passed": state.get("validation_passed", False),
                "flags" : state.get("compliance_flags", []),
            },
            "evidence_summary": (
                state.get("evidence", {})
                     .get("evidence_summary", "")
            ),
            "explanation"     : explanation,
            "code_explanations": code_explanations,
            "agent_trace"     : [
                msg.content
                for msg in state.get("messages", [])
                if hasattr(msg, "name")
            ],
        }

        logger.info("► Agent 4 complete — final response assembled")

        return {
            "explanation"      : explanation,
            "code_explanations": code_explanations,
            "final_response"   : final_response,
            "current_agent"    : "complete",
            "messages"         : [AIMessage(
                content = "Explanation complete — pipeline finished",
                name="explanation_agent",
            )],
        }


# ── Conditional edge logic ────────────────────────────────────

def should_retry_coding(state: AgentState) -> str:
    """
    After validation — should we retry coding or continue?

    Logic:
    - If validation FAILED and we have not retried yet
      AND we have some codes to retry with → retry
    - Otherwise → continue to explanation

    Max 1 retry to prevent infinite loops.
    """
    validation_failed = not state.get("validation_passed", False)
    has_codes         = bool(state.get("hcc_codes"))
    retry_count       = state.get("retry_count", 0)
    can_retry         = retry_count < 1

    if validation_failed and has_codes and can_retry:
        logger.info(
            "  Validation failed — retrying HCC coding "
            f"(attempt {retry_count + 2})"
        )
        return "retry"

    return "continue"


# ── Graph builder ─────────────────────────────────────────────

def build_clinical_graph(pipeline) -> any:
    """
    Build and compile the LangGraph clinical agent workflow.

    Graph structure:
        START
          ↓
        evidence_retrieval_agent
          ↓
        hcc_coding_agent
          ↓
        validation_agent ──(retry)──→ hcc_coding_agent
          ↓ (continue)
        explanation_agent
          ↓
         END

    Args:
        pipeline: ClinicalRAGPipeline instance

    Returns:
        Compiled LangGraph application ready to invoke
    """
    nodes = ClinicalAgentNodes(pipeline)

    # Create the graph with our state type
    graph = StateGraph(AgentState)

    # ── Register all 4 agent nodes ────────────────────────────
    graph.add_node(
        "evidence_retrieval_agent",
        nodes.evidence_retrieval_agent,
    )
    graph.add_node(
        "hcc_coding_agent",
        nodes.hcc_coding_agent,
    )
    graph.add_node(
        "validation_agent",
        nodes.validation_agent,
    )
    graph.add_node(
        "explanation_agent",
        nodes.explanation_agent,
    )

    # ── Set entry point ───────────────────────────────────────
    graph.set_entry_point("evidence_retrieval_agent")

    # ── Add edges ─────────────────────────────────────────────
    # Fixed edges (always go to next agent)
    graph.add_edge("evidence_retrieval_agent", "hcc_coding_agent")
    graph.add_edge("hcc_coding_agent", "validation_agent")

    # Conditional edge from validation:
    # "retry" → go back to hcc_coding_agent
    # "continue" → go forward to explanation_agent
    graph.add_conditional_edges(
        "validation_agent",
        should_retry_coding,
        {
            "retry"   : "hcc_coding_agent",
            "continue": "explanation_agent",
        },
    )

    # Fixed edge from explanation to END
    graph.add_edge("explanation_agent", END)

    # Compile and return
    return graph.compile()


# ── Public runner function ────────────────────────────────────

def run_clinical_pipeline(
    pipeline,
    query     : str,
    patient_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Execute the full 4-agent clinical pipeline.

    This is called by the /generate-hcc-code endpoint
    when use_agents=True.

    Args:
        pipeline  : ClinicalRAGPipeline instance
        query     : clinical question or HCC coding request
        patient_id: optional patient identifier

    Returns:
        final_response dict with all agent outputs combined

    Usage:
        from src.agents.graph import run_clinical_pipeline
        result = run_clinical_pipeline(pipeline, query, patient_id)
        print(result["hcc_codes"])
        print(result["total_raf_score"])
    """
    # Build the compiled graph
    app = build_clinical_graph(pipeline)

    # Set initial state — everything starts empty
    initial_state: AgentState = {
        "query"           : query,
        "patient_id"      : patient_id,
        "messages"        : [HumanMessage(content=query)],
        "evidence"        : {},
        "evidence_found"  : False,
        "hcc_codes"       : [],
        "total_raf_score" : 0.0,
        "coding_notes"    : "",
        "validation_passed": False,
        "validated_codes" : [],
        "compliance_flags": [],
        "explanation"     : "",
        "code_explanations": [],
        "current_agent"   : "evidence_retrieval_agent",
        "retry_count"     : 0,
        "error"           : None,
        "final_response"  : None,
    }

    logger.info(f"Starting 4-agent clinical pipeline: {query!r}")

    # Run the graph — blocks until all agents complete
    final_state = app.invoke(initial_state)

    logger.info("4-agent clinical pipeline complete")

    # Return the assembled final response
    return final_state.get("final_response", {})