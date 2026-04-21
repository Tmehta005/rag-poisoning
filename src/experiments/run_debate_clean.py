"""
Clean debate experiment runner.

Wires together:
  index  →  N DebateSubagents  →  DebateInterface  →  JudgeLLM  →  RunLog  →  runs.jsonl

This is the Phase 5 clean baseline: no trigger, no poisoned documents, pure
majority-vote relay. Poisoning hooks (``poison_doc_ids``, trigger injection
into retrieval) are intentionally absent here — they land in a sibling
``run_debate_attack.py`` alongside Phase 4's attack runner.
"""

from __future__ import annotations

import yaml
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.agents.debate.debate_subagent import DebateSubagent
from src.agents.debate.judge import JudgeLLM
from src.ingestion import ingest_corpus, load_ingestion_config
from src.retriever import Retriever
from src.schemas import RunLog


def _load_debate_config(config_path: str = "configs/system_debate.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_clean_debate_subagents(
    index,
    debate_config: dict,
    model_client,
) -> list[DebateSubagent]:
    """
    Build N DebateSubagents that all read from the same clean index.

    When switching to diversified retrieval, vary ``top_k`` (or wrap the
    index with different chunking) per agent here; the Judge and Debate
    Interface require no changes.
    """
    num_agents = debate_config.get("num_subagents", 3)
    top_k = debate_config.get("subagent_top_k", 5)

    subagents: list[DebateSubagent] = []
    for i in range(1, num_agents + 1):
        agent_id = f"subagent_{i}"
        retriever = Retriever(index, top_k=top_k)
        subagents.append(
            DebateSubagent(
                agent_id=agent_id,
                retriever=retriever,
                model_client=model_client,
                top_k=top_k,
                poison_doc_ids=set(),
            )
        )
    return subagents


def run_clean_debate_experiment(
    queries: list[dict],
    data_dir: str = "data/corpus",
    persist_dir: str = "data/index",
    output_dir: str = "results",
    ingestion_config_path: str = "configs/ingestion.yaml",
    debate_config_path: str = "configs/system_debate.yaml",
    model_client: Optional[object] = None,
) -> list[RunLog]:
    """
    Run the clean debate pipeline over a list of queries.

    Args:
        queries: List of ``{"query_id": str, "query": str}`` dicts.
        data_dir, persist_dir: Passed to :func:`ingest_corpus`.
        output_dir: Where ``runs.jsonl`` is written.
        ingestion_config_path: Path to ingestion YAML.
        debate_config_path: Path to debate YAML.
        model_client: Optional AutoGen ChatCompletionClient override. If
            None, an :class:`OpenAIChatCompletionClient` is constructed
            from ``model`` in the debate config (requires OPENAI_API_KEY).

    Returns:
        List of :class:`RunLog`, one per query.
    """
    ingestion_config = load_ingestion_config(ingestion_config_path)
    debate_config = _load_debate_config(debate_config_path)

    index = ingest_corpus(data_dir, config=ingestion_config, persist_dir=persist_dir)

    if model_client is None:
        model_client = OpenAIChatCompletionClient(
            model=debate_config.get("model", "gpt-5")
        )

    subagents = build_clean_debate_subagents(
        index=index,
        debate_config=debate_config,
        model_client=model_client,
    )

    judge = JudgeLLM(
        subagents=subagents,
        max_rounds=debate_config.get("max_rounds", 4),
        stable_for=debate_config.get("stable_for", 2),
        output_dir=output_dir,
    )

    logs: list[RunLog] = []
    for q in queries:
        log = judge.run(
            query=q["query"],
            query_id=q["query_id"],
            trigger=None,
            attack_condition="clean",
            ground_truth_answer=q.get("ground_truth_answer"),
        )
        logs.append(log)

    return logs
