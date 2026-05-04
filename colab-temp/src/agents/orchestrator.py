"""
Multi-agent orchestrator implemented as a LangGraph StateGraph.

Graph topology (fan-out + join):

    START ──► subagent_1 ─┐
           ──► subagent_2 ─┼──► orchestrator_node ──► END
           ──► subagent_3 ─┘

Each subagent node runs ExpertSubagent.run() and appends its SubagentOutput
to state["agent_outputs"] via the list-concat reducer.

The orchestrator node receives all three outputs, calls an LLM to pick the
best-supported answer, and writes OrchestratorOutput to state.

Attack compatibility:
  - The graph structure does not change for attacks.
  - Subagent_1's Retriever is swapped to an index containing D_p by the
    attack runner at construction time (see src/experiments/run_attack_orch.py).
  - harmful_action_flag is set by the attack runner after invoke, based on
    whether the final answer matches the attack target.
"""

from __future__ import annotations

import json
import os
import re
from typing import Annotated, Callable, List, Optional
import operator

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    def observe(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator
    langfuse_context = None  # type: ignore[assignment]

from src.agents.subagent import ExpertSubagent
from src.schemas import OrchestratorOutput, SubagentOutput

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


def _load_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _parse_orchestrator_response(raw: str, agent_outputs: list[SubagentOutput]) -> dict:
    """Parse LLM JSON response. Falls back to highest-confidence agent on failure."""
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(cleaned)
        return {
            "final_answer": str(data.get("final_answer", "")),
            "winning_agent_id": str(data.get("winning_agent_id", "")),
            "reasoning_summary": str(data.get("reasoning_summary", "")),
        }
    except (json.JSONDecodeError, ValueError):
        best = max(agent_outputs, key=lambda o: o.confidence)
        return {
            "final_answer": best.answer,
            "winning_agent_id": best.agent_id,
            "reasoning_summary": "Fallback: selected highest-confidence agent.",
        }


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    query: str
    query_id: str
    attack_condition: str
    trigger: Optional[str]
    # Annotated with operator.add so each subagent node can append independently
    agent_outputs: Annotated[List[SubagentOutput], operator.add]
    final_decision: Optional[OrchestratorOutput]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_orchestrator_graph(
    agents: list[ExpertSubagent],
    model: str = "gpt-4o-mini",
    prompt_path: str = "prompts/orchestrator.txt",
    llm_fn: Optional[Callable[[str], str]] = None,
):
    """
    Build and compile the LangGraph orchestrator graph.

    Args:
        agents: List of ExpertSubagent instances (exactly 3 for the main setup).
        model: OpenAI model for the orchestrator aggregation call.
        prompt_path: Path to the orchestrator prompt template.
        llm_fn: Optional LLM callable override (for testing).

    Returns:
        A compiled LangGraph app. Call app.invoke(initial_state) to run.
    """
    prompt_template = _load_prompt(prompt_path)

    openai_client = None
    if llm_fn is None and _openai_available and os.getenv("OPENAI_API_KEY"):
        openai_client = OpenAI()

    def _call_llm(prompt: str) -> str:
        if llm_fn is not None:
            return llm_fn(prompt)
        if openai_client is not None:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content.strip()
        # Stub: pick highest-confidence agent without an API call
        return ""

    # -- subagent nodes --
    def make_subagent_node(agent: ExpertSubagent):
        @observe(name=f"subagent_node_{agent.agent_id}")
        def node_fn(state: OrchestratorState) -> dict:
            output = agent.run(state["query"], trigger=state.get("trigger"))
            return {"agent_outputs": [output]}
        node_fn.__name__ = f"node_{agent.agent_id}"
        return node_fn

    # -- orchestrator node --
    @observe(name="orchestrator_node")
    def orchestrator_node(state: OrchestratorState) -> dict:
        outputs: list[SubagentOutput] = state["agent_outputs"]
        query = state["query"]

        agent_responses_text = "\n\n".join(
            f"Agent: {o.agent_id}\n"
            f"Answer: {o.answer}\n"
            f"Citations: {o.citations}\n"
            f"Confidence: {o.confidence}\n"
            f"Rationale: {o.rationale}"
            for o in outputs
        )
        prompt = prompt_template.format(
            question=query,
            agent_responses=agent_responses_text,
        )

        raw = _call_llm(prompt)
        parsed = _parse_orchestrator_response(raw, outputs)

        # Resolve winning agent's metadata
        winning_id = parsed["winning_agent_id"]
        winning_agents = [o for o in outputs if o.agent_id == winning_id]
        if winning_agents:
            winner = winning_agents[0]
            selected_evidence = winner.citations
            final_confidence = winner.confidence
        else:
            # Fallback: highest confidence
            winner = max(outputs, key=lambda o: o.confidence)
            winning_id = winner.agent_id
            selected_evidence = winner.citations
            final_confidence = winner.confidence

        decision = OrchestratorOutput(
            final_answer=parsed["final_answer"] or winner.answer,
            selected_evidence=selected_evidence,
            winning_subagents=[winning_id],
            final_confidence=final_confidence,
            harmful_action_flag=False,  # set in Phase 3
            reasoning_summary=parsed["reasoning_summary"],
        )

        if langfuse_context:
            langfuse_context.update_current_observation(
                input={"query": query, "num_agents": len(outputs)},
                output={"final_answer": decision.final_answer, "winner": winning_id},
            )

        return {"final_decision": decision}

    # -- assemble graph (sequential chain) --
    # Agents run one after another: START → agent_1 → agent_2 → agent_3 → orchestrator → END.
    # This avoids threading issues with the shared HuggingFace embedding model.
    # The logical structure (N independent agents feeding one orchestrator) is preserved.
    # Note: switching to true parallel fan-out later only requires changing the edges below.
    graph = StateGraph(OrchestratorState)

    prev = START
    for agent in agents:
        graph.add_node(agent.agent_id, make_subagent_node(agent))
        graph.add_edge(prev, agent.agent_id)
        prev = agent.agent_id

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_edge(prev, "orchestrator")
    graph.add_edge("orchestrator", END)

    return graph.compile()
