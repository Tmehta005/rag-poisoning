"""
Single-agent agentic experiment runner.

Wires together:
  index → 1 ExpertSubagent → LangGraph (no orchestrator) → RunLog → runs.jsonl

This establishes the attack ceiling: with one agent, any poisoning of its
retriever propagates directly to the final answer (ASR ≈ 1.0). The gap
between this and run_clean_experiment (3-agent orchestrator) and
run_clean_debate_experiment measures the defense benefit of multi-agent
architectures.

Uses the same ExpertSubagent class and schemas as run_clean.py so results
are directly comparable — no confounds from different generation code.
"""

from __future__ import annotations

import yaml
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langgraph.graph import StateGraph, START, END

from src.agents.orchestrator import OrchestratorState
from src.agents.subagent import ExpertSubagent
from src.ingestion import ingest_corpus, load_ingestion_config
from src.logging_utils import emit_run_log
from src.retriever import Retriever
from src.schemas import OrchestratorOutput, RunLog


def _load_system_config(config_path: str = "configs/system_orchestrator.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_single_agent(index, system_config: dict) -> ExpertSubagent:
    model = system_config.get("model", "gpt-4o-mini")
    top_k = system_config.get("top_k", 3)
    retriever = Retriever(index, top_k=top_k)
    return ExpertSubagent(
        agent_id="subagent_1",
        retriever=retriever,
        model=model,
        poison_doc_ids=set(),
    )


def _build_single_agent_graph(agent: ExpertSubagent):
    """
    Minimal LangGraph: START → subagent_1 → wrap → END.

    No LLM orchestration call — the single agent's output is the final decision.
    """
    def subagent_node(state: OrchestratorState) -> dict:
        output = agent.run(state["query"], trigger=state.get("trigger"))
        return {"agent_outputs": [output]}

    def wrap_node(state: OrchestratorState) -> dict:
        output = state["agent_outputs"][0]
        decision = OrchestratorOutput(
            final_answer=output.answer,
            selected_evidence=output.citations,
            winning_subagents=[output.agent_id],
            final_confidence=output.confidence,
            harmful_action_flag=False,
            reasoning_summary=None,
        )
        return {"final_decision": decision}

    graph = StateGraph(OrchestratorState)
    graph.add_node("subagent_1", subagent_node)
    graph.add_node("wrap", wrap_node)
    graph.add_edge(START, "subagent_1")
    graph.add_edge("subagent_1", "wrap")
    graph.add_edge("wrap", END)
    return graph.compile()


def run_single_agent_experiment(
    queries: list[dict],
    data_dir: str = "data/corpus",
    persist_dir: str = "data/index",
    output_dir: str = "results",
    ingestion_config_path: str = "configs/ingestion.yaml",
    system_config_path: str = "configs/system_orchestrator.yaml",
) -> list[RunLog]:
    """
    Run the single-agent agentic pipeline over a list of queries.

    This is the attack ceiling baseline: with one agent, any poisoning
    propagates directly. Compare its ASR to run_clean_experiment (3-agent
    orchestrator) and run_clean_debate_experiment to measure robustness gains.

    Args:
        queries: List of {"query_id": str, "query": str} dicts.
        data_dir: Directory containing corpus documents.
        persist_dir: Where the LlamaIndex vector store is persisted.
        output_dir: Where runs.jsonl is written.
        ingestion_config_path: Path to ingestion.yaml.
        system_config_path: Path to system_orchestrator.yaml.

    Returns:
        List of RunLog, one per query.
    """
    ingestion_config = load_ingestion_config(ingestion_config_path)
    system_config = _load_system_config(system_config_path)

    index = ingest_corpus(data_dir, config=ingestion_config, persist_dir=persist_dir)
    agent = _build_single_agent(index, system_config)
    app = _build_single_agent_graph(agent)

    logs = []
    for q in queries:
        query_id = q["query_id"]
        query = q["query"]

        initial_state: OrchestratorState = {
            "query": query,
            "query_id": query_id,
            "attack_condition": "clean",
            "trigger": None,
            "agent_outputs": [],
            "final_decision": None,
        }

        final_state = app.invoke(initial_state)
        output = final_state["agent_outputs"][0]

        run_log = RunLog(
            query_id=query_id,
            attack_condition="clean",
            trigger=None,
            ground_truth_answer=q.get("ground_truth_answer"),
            retrieved_doc_ids_per_agent={output.agent_id: output.retrieved_doc_ids},
            poison_retrieved=False,
            agent_responses={output.agent_id: output},
            final_decision=final_state["final_decision"],
            metrics={},
        )

        emit_run_log(run_log, output_dir=output_dir)
        logs.append(run_log)

    return logs
