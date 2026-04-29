"""
Single-agent attack runner (the true attack ceiling).

One ``ExpertSubagent`` with a poisoned index. No orchestrator LLM — the
subagent's output is wrapped directly into an ``OrchestratorOutput`` so
the final answer is exactly what the subagent produced. This mirrors
``run_single_agent.py`` (clean path) but swaps in the poisoned index,
private trigger, and harmful-action evaluation from ``run_attack_orch.py``.

Why a dedicated runner: routing single-agent attack through
``run_attack_orchestrator`` with ``num_subagents=1`` still invoked
``orchestrator_node`` (an extra LLM pass over a single candidate), which
was both unnecessary work and a potential rewrite source.

Only the targeted threat model is meaningful here (one agent == no
"other" agents to dilute), but ``attack_condition`` is logged as
``main_injection.targeted`` for parity with the orchestrator runner so
``src/analysis/make_results_table.py`` can group by condition.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langgraph.graph import StateGraph, START, END
from llama_index.core import VectorStoreIndex

from src.agents.orchestrator import OrchestratorState
from src.agents.subagent import ExpertSubagent
from src.attacks.artifacts import is_harmful_answer, resolve_attack_artifact
from src.attacks.poison_doc import render_extra_poison_specs
from src.attacks.poisoned_index import build_poisoned_index_from_artifact
from src.ingestion import load_ingestion_config
from src.logging_utils import emit_run_log
from src.retriever import Retriever
from src.schemas import OrchestratorOutput, RunLog


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_graph(agent: ExpertSubagent):
    """START → subagent_1 → wrap → END. No orchestrator LLM call."""

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


def run_attack_single_agent(
    queries: List[dict],
    clean_index: VectorStoreIndex,
    output_dir: str = "results",
    system_config_path: str = "configs/system_single_agent.yaml",
    ingestion_config_path: str = "configs/corpus_cybersec.yaml",
    num_poison_docs: int = 1,
) -> List[RunLog]:
    """
    Run the single-agent attack over queries with an ``attack`` block.

    Args:
        queries: Loaded via ``load_queries`` (artifact_path auto-hydrated).
        clean_index: Pre-built clean ``VectorStoreIndex``.
        num_poison_docs: Total poison docs injected into the ephemeral
            index. Doc 0 comes from the artifact (trigger-optimized);
            docs 1..N-1 are rendered via ``render_poison_doc`` with the
            same trigger and target_claim.

    Returns:
        ``[RunLog]`` in the same order as ``queries``.
    """
    system_cfg = _load_yaml(system_config_path)
    try:
        ingestion_cfg = load_ingestion_config(ingestion_config_path)
    except FileNotFoundError:
        ingestion_cfg = {}
    embed_model = ingestion_cfg.get("embed_model", "local")

    model = system_cfg.get("model", "gpt-4o-mini")
    top_k = system_cfg.get("top_k", 5)

    logs: List[RunLog] = []

    for q in queries:
        attack = q.get("attack")
        if not attack:
            continue

        query_id = q["query_id"]
        query_text = q["query"]
        ground_truth = q.get("ground_truth_answer")

        artifact = resolve_attack_artifact(attack, query_id)

        extra_specs = render_extra_poison_specs(
            artifact, num_poison_docs, ingestion_config_path
        )
        poisoned_index, poison_ids = build_poisoned_index_from_artifact(
            clean_index,
            artifact,
            embed_model=embed_model,
            extra_specs=extra_specs or None,
        )

        agent = ExpertSubagent(
            agent_id="subagent_1",
            retriever=Retriever(poisoned_index, top_k=top_k),
            model=model,
            poison_doc_ids=poison_ids,
            private_trigger=artifact.trigger,
        )

        app = _build_graph(agent)

        initial_state: OrchestratorState = {
            "query": query_text,
            "query_id": query_id,
            "attack_condition": "main_injection.targeted",
            "trigger": None,
            "agent_outputs": [],
            "final_decision": None,
        }
        final_state = app.invoke(initial_state)

        output = final_state["agent_outputs"][0]
        raw_decision: OrchestratorOutput = final_state["final_decision"]
        harmful = is_harmful_answer(raw_decision.final_answer, artifact)
        decision = raw_decision.model_copy(update={"harmful_action_flag": harmful})

        run_log = RunLog(
            query_id=query_id,
            attack_condition="main_injection.targeted",
            trigger=artifact.trigger,
            ground_truth_answer=ground_truth,
            retrieved_doc_ids_per_agent={output.agent_id: output.retrieved_doc_ids},
            poison_retrieved=bool(output.poison_retrieved),
            agent_responses={output.agent_id: output},
            final_decision=decision,
            metrics={
                "poison_retrieved": float(output.poison_retrieved),
                "harmful_action": float(harmful),
                "num_poison_docs": float(num_poison_docs),
            },
        )
        emit_run_log(run_log, output_dir=output_dir)
        logs.append(run_log)

    return logs


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: build the clean index, load queries, run the attack."""
    from src.corpus.ingest_with_metadata import ingest_corpus_with_metadata
    from src.corpus.query_loader import load_queries

    parser = argparse.ArgumentParser(
        description="Run the single-agent poisoning attack (attack ceiling)."
    )
    parser.add_argument(
        "--query-file",
        default="data/queries/attack_queries_cybersec.yaml",
    )
    parser.add_argument(
        "--system-config",
        default="configs/system_single_agent.yaml",
    )
    parser.add_argument(
        "--ingestion-config",
        default="configs/corpus_cybersec.yaml",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--num-poison-docs",
        type=int,
        default=1,
        help="Total poison docs injected (1 from artifact + N-1 rendered).",
    )
    args = parser.parse_args(argv)

    queries = load_queries(args.query_file)
    print(
        f"[run_attack_single_agent] loaded {len(queries)} queries from {args.query_file}"
    )

    clean_index = ingest_corpus_with_metadata()
    print("[run_attack_single_agent] clean index ready")

    logs = run_attack_single_agent(
        queries=queries,
        clean_index=clean_index,
        output_dir=args.output_dir,
        system_config_path=args.system_config,
        ingestion_config_path=args.ingestion_config,
        num_poison_docs=args.num_poison_docs,
    )

    print(f"\n[run_attack_single_agent] ran {len(logs)} attacked queries")
    for log in logs:
        fd = log.final_decision
        answer = fd.final_answer if fd else "<no decision>"
        print(
            f"  {log.query_id} [{log.attack_condition}] "
            f"poison_retrieved={log.poison_retrieved} "
            f"harmful={fd.harmful_action_flag if fd else '?'} "
            f"answer={answer[:80]!r}"
        )
    print(f"\n[run_attack_single_agent] runs appended to {args.output_dir}/runs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
