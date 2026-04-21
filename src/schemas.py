"""
Typed schemas for all agent inputs and outputs.

All pipeline components must use these schemas so that
every run artifact is structurally consistent and every
metric in METRICS.md is computable from stored logs.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievedDoc(BaseModel):
    doc_id: str
    text: str
    score: float


# ---------------------------------------------------------------------------
# Subagent output (ARCHITECTURE.md subagent contract)
# ---------------------------------------------------------------------------

class SubagentOutput(BaseModel):
    agent_id: str
    answer: str
    citations: List[str] = Field(default_factory=list, description="Retrieved doc IDs cited")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    poison_retrieved: bool = Field(
        default=False,
        description="True if any poisoned document was among the retrieved docs",
    )
    retrieved_doc_ids: List[str] = Field(
        default_factory=list,
        description="All doc IDs retrieved by this agent (cited or not)",
    )


# ---------------------------------------------------------------------------
# Orchestrator output (ARCHITECTURE.md orchestrator contract)
# ---------------------------------------------------------------------------

class OrchestratorOutput(BaseModel):
    final_answer: str
    selected_evidence: List[str] = Field(
        default_factory=list, description="Doc IDs supporting the final answer"
    )
    winning_subagents: List[str] = Field(
        default_factory=list, description="agent_id(s) whose response was selected"
    )
    final_confidence: float = Field(ge=0.0, le=1.0)
    harmful_action_flag: bool = Field(
        default=False,
        description="True if the final answer matches the harmful target action",
    )
    reasoning_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Per-run log (CLAUDE.md logging requirements)
# ---------------------------------------------------------------------------

class RunLog(BaseModel):
    query_id: str
    attack_condition: str = Field(
        description="One of: clean | main_injection | poisonedrag_baseline"
    )
    trigger: Optional[str] = Field(
        default=None, description="Trigger t appended to query, if any"
    )
    ground_truth_answer: Optional[str] = Field(
        default=None,
        description="Reference answer from query file; used to compute benign accuracy offline",
    )
    retrieved_doc_ids_per_agent: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping agent_id -> list of retrieved doc IDs",
    )
    poison_retrieved: bool = Field(
        default=False,
        description="True if D_p was retrieved by any agent in this run",
    )
    agent_responses: Dict[str, SubagentOutput] = Field(
        default_factory=dict,
        description="Mapping agent_id -> SubagentOutput",
    )
    final_decision: Optional[OrchestratorOutput] = None
    debate_transcript: Optional["DebateTranscript"] = Field(
        default=None,
        description="Full debate transcript when the run was produced by the debate setup",
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Computed metric values (ASR, poison_retrieval_rate, etc.)",
    )


# ---------------------------------------------------------------------------
# Debate setup (Phase 5)
# ---------------------------------------------------------------------------

class DebateRound(BaseModel):
    round_num: int = Field(ge=1, description="1-indexed debate round number")
    stances: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping agent_id -> stated answer at the end of this round",
    )
    confidences: Dict[str, float] = Field(
        default_factory=dict,
        description="Mapping agent_id -> self-reported confidence this round",
    )
    messages: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping agent_id -> full message content this round",
    )


class DebateTranscript(BaseModel):
    rounds: List[DebateRound] = Field(default_factory=list)
    majority_cluster: List[str] = Field(
        default_factory=list,
        description="agent_ids whose final-round stance belongs to the winning cluster",
    )
    majority_answer: str = Field(
        default="",
        description="Representative answer string for the winning cluster",
    )
    rounds_used: int = Field(ge=0, default=0)
    stopped_reason: str = Field(
        default="",
        description='Termination cause, e.g. "converged" or "max_rounds"',
    )


RunLog.model_rebuild()
