"""
Clean orchestrator experiment runner.

Wires together:
  index → 3 ExpertSubagents → LangGraph orchestrator → RunLog → runs.jsonl

This is the Phase 2 clean baseline: no trigger, no poisoned documents.

Corpus swap (toy → Wikipedia):
  Change data_dir and persist_dir in the call to run_clean_experiment().
  No code changes needed in agents or orchestrator.
"""

from __future__ import annotations

import yaml
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.agents.orchestrator import OrchestratorState, build_orchestrator_graph
from src.agents.subagent import ExpertSubagent
from src.ingestion import ingest_corpus, load_ingestion_config
from src.logging_utils import emit_run_log
from src.retriever import Retriever
from src.schemas import RunLog


def _load_system_config(config_path: str = "configs/system_orchestrator.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_clean_agents(
    index,
    system_config: dict,
) -> list[ExpertSubagent]:
    """
    Build 3 ExpertSubagents, all pointing at the same clean index.

    In Phase 3, subagent_1's index will be replaced with a poisoned one
    before calling this (or a separate build_attacked_agents() will be used).
    """
    num_agents = system_config.get("num_subagents", 3)
    model = system_config.get("model", "gpt-4o-mini")
    top_k = system_config.get("top_k", 3)

    agents = []
    for i in range(1, num_agents + 1):
        agent_id = f"subagent_{i}"
        retriever = Retriever(index, top_k=top_k)
        agents.append(
            ExpertSubagent(
                agent_id=agent_id,
                retriever=retriever,
                model=model,
                poison_doc_ids=set(),  # empty in clean runs
            )
        )
    return agents


def run_clean_experiment(
    queries: list[dict],
    data_dir: str = "data/corpus",
    persist_dir: str = "data/index",
    output_dir: str = "results",
    ingestion_config_path: str = "configs/ingestion.yaml",
    system_config_path: str = "configs/system_orchestrator.yaml",
) -> list[RunLog]:
    """
    Run the clean orchestrator pipeline over a list of queries.

    Args:
        queries: List of {"query_id": str, "query": str} dicts.
                 Swap in Wikipedia QA dicts here without any other code change.
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
    agents = build_clean_agents(index, system_config)

    model = system_config.get("model", "gpt-4o-mini")
    app = build_orchestrator_graph(agents, model=model)

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

        agent_responses = {o.agent_id: o for o in final_state["agent_outputs"]}
        retrieved_per_agent = {
            o.agent_id: o.retrieved_doc_ids for o in final_state["agent_outputs"]
        }

        run_log = RunLog(
            query_id=query_id,
            attack_condition="clean",
            trigger=None,
            retrieved_doc_ids_per_agent=retrieved_per_agent,
            poison_retrieved=False,
            agent_responses=agent_responses,
            final_decision=final_state["final_decision"],
            metrics={},
        )

        emit_run_log(run_log, output_dir=output_dir)
        logs.append(run_log)

    return logs
