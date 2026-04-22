"""
Orchestrator attack runner with two threat models.

threat_model:
    "targeted"  Only poisoned_subagent_ids get the poisoned index + a
                 private_trigger loaded from the artifact. The orchestrator
                 state-level trigger stays None, so clean subagents retrieve
                 without the trigger from the clean index.
    "global"    Every subagent uses the poisoned index AND the orchestrator
                 state-level trigger is set from the artifact so all agents
                 append it to their retrieval queries. Models the
                 constructor-level / judge-level leak of the trigger.

Consumes an ``AttackArtifact`` that already contains ``trigger`` and
``poison_doc_text``; there's no inline trigger duplication in the query
file when the ``attack.artifact_path`` shortcut is used.
"""

from __future__ import annotations

import argparse
import yaml
from typing import List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from llama_index.core import VectorStoreIndex

from src.agents.orchestrator import OrchestratorState, build_orchestrator_graph
from src.agents.subagent import ExpertSubagent
from src.attacks.artifacts import is_harmful_answer, load_artifact
from src.attacks.poisoned_index import build_poisoned_index_from_artifact
from src.ingestion import load_ingestion_config
from src.logging_utils import emit_run_log
from src.retriever import Retriever
from src.schemas import OrchestratorOutput, RunLog


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def run_attack_orchestrator(
    queries: List[dict],
    clean_index: VectorStoreIndex,
    output_dir: str = "results",
    system_config_path: str = "configs/system_orchestrator.yaml",
    attack_config_path: str = "configs/attack_main_injection.yaml",
    ingestion_config_path: str = "configs/corpus_cybersec.yaml",
    threat_model: Optional[str] = None,
    poisoned_subagent_ids: Optional[List[str]] = None,
) -> List[RunLog]:
    """
    Run the orchestrator attack over queries that reference an artifact.

    Args:
        queries: Loaded via ``load_queries`` (artifact_path auto-hydrated).
        clean_index: Pre-built clean ``VectorStoreIndex``.
        threat_model: Override the attack config's ``threat_model``.
        poisoned_subagent_ids: Override the attack config's
            ``poisoned_subagent_ids``.

    Returns:
        ``[RunLog]`` in the same order as ``queries``.
    """
    system_cfg = _load_yaml(system_config_path)
    attack_cfg = _load_yaml(attack_config_path)
    try:
        ingestion_cfg = load_ingestion_config(ingestion_config_path)
    except FileNotFoundError:
        ingestion_cfg = {}
    embed_model = ingestion_cfg.get("embed_model", "local")

    model = system_cfg.get("model", "gpt-4o-mini")
    top_k = system_cfg.get("top_k", 3)
    num_agents = system_cfg.get("num_subagents", 3)

    tm = threat_model or attack_cfg.get("threat_model", "targeted")
    poisoned_ids = set(
        poisoned_subagent_ids
        or attack_cfg.get("poisoned_subagent_ids", [attack_cfg.get("poisoned_documents", {}).get("injection_target", "subagent_1")])
    )
    if tm not in ("targeted", "global"):
        raise ValueError(f"threat_model must be 'targeted' or 'global', got {tm!r}")

    logs: List[RunLog] = []

    for q in queries:
        attack = q.get("attack")
        if not attack:
            continue

        query_id = q["query_id"]
        query_text = q["query"]
        ground_truth = q.get("ground_truth_answer")

        if attack.get("artifact_path"):
            artifact = load_artifact(attack["artifact_path"])
        else:
            # Support inline attacks too: synthesize a temporary artifact-like
            # object. We only need .trigger, .poison_doc_id, .poison_doc_text,
            # .target_claim.
            from src.attacks.artifacts import AttackArtifact

            poison_docs = attack.get("poison_docs") or []
            if not poison_docs:
                raise ValueError(
                    f"Query {query_id} has inline attack but no poison_docs."
                )
            p0 = poison_docs[0]
            artifact = AttackArtifact(
                attack_id=attack.get("attack_id", f"{query_id}_inline"),
                trigger=attack.get("trigger", ""),
                token_ids=[],
                target_claim=attack.get("target_answer", ""),
                poison_doc_id=p0["doc_id"],
                poison_doc_text=p0["text"],
                encoder_model="",
                num_adv_passage_tokens=0,
            )

        poisoned_index, poison_ids = build_poisoned_index_from_artifact(
            clean_index, artifact, embed_model=embed_model
        )

        attack_condition = f"main_injection.{tm}.{artifact.variant}"

        state_trigger: Optional[str] = artifact.trigger if tm == "global" else None

        agents: List[ExpertSubagent] = []
        for i in range(1, num_agents + 1):
            agent_id = f"subagent_{i}"
            is_poisoned_target = (tm == "global") or (agent_id in poisoned_ids)
            retriever = (
                Retriever(poisoned_index, top_k=top_k)
                if is_poisoned_target
                else Retriever(clean_index, top_k=top_k)
            )
            p_ids = poison_ids if is_poisoned_target else set()
            private_trigger = (
                artifact.trigger
                if tm == "targeted" and agent_id in poisoned_ids
                else None
            )
            agents.append(
                ExpertSubagent(
                    agent_id=agent_id,
                    retriever=retriever,
                    model=model,
                    poison_doc_ids=p_ids,
                    private_trigger=private_trigger,
                )
            )

        app = build_orchestrator_graph(agents, model=model)

        initial_state: OrchestratorState = {
            "query": query_text,
            "query_id": query_id,
            "attack_condition": attack_condition,
            "trigger": state_trigger,
            "agent_outputs": [],
            "final_decision": None,
        }
        final_state = app.invoke(initial_state)

        agent_responses = {o.agent_id: o for o in final_state["agent_outputs"]}
        retrieved_per_agent = {
            o.agent_id: o.retrieved_doc_ids for o in final_state["agent_outputs"]
        }
        any_poison_retrieved = any(
            o.poison_retrieved for o in final_state["agent_outputs"]
        )

        raw_decision: OrchestratorOutput = final_state["final_decision"]
        harmful = is_harmful_answer(raw_decision.final_answer, artifact)
        decision = raw_decision.model_copy(update={"harmful_action_flag": harmful})

        run_log = RunLog(
            query_id=query_id,
            attack_condition=attack_condition,
            trigger=artifact.trigger,
            ground_truth_answer=ground_truth,
            retrieved_doc_ids_per_agent=retrieved_per_agent,
            poison_retrieved=any_poison_retrieved,
            agent_responses=agent_responses,
            final_decision=decision,
            metrics={
                "poison_retrieved": float(any_poison_retrieved),
                "harmful_action": float(harmful),
            },
        )
        emit_run_log(run_log, output_dir=output_dir)
        logs.append(run_log)

    return logs


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: build the clean index, load queries, run the attack."""
    from src.corpus.ingest_cybersec import ingest_cybersec_corpus
    from src.corpus.query_loader import load_queries

    parser = argparse.ArgumentParser(
        description="Run the orchestrator poisoning attack."
    )
    parser.add_argument(
        "--query-file",
        default="data/queries/attack_queries_cybersec.yaml",
        help="Queries with attack blocks (inline or artifact_path).",
    )
    parser.add_argument(
        "--threat-model",
        choices=["targeted", "global"],
        default=None,
        help="Override the attack config's threat_model.",
    )
    parser.add_argument(
        "--poisoned-subagent-id",
        action="append",
        dest="poisoned_subagent_ids",
        default=None,
        help=(
            "Which subagent(s) carry the trigger in the targeted model. "
            "Pass multiple times for a set. Defaults to the attack config."
        ),
    )
    parser.add_argument(
        "--system-config",
        default="configs/system_orchestrator.yaml",
    )
    parser.add_argument(
        "--attack-config",
        default="configs/attack_main_injection.yaml",
    )
    parser.add_argument(
        "--ingestion-config",
        default="configs/corpus_cybersec.yaml",
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    queries = load_queries(args.query_file)
    print(
        f"[run_attack_orch] loaded {len(queries)} queries from {args.query_file}"
    )

    clean_index = ingest_cybersec_corpus()
    print("[run_attack_orch] clean index ready")

    logs = run_attack_orchestrator(
        queries=queries,
        clean_index=clean_index,
        output_dir=args.output_dir,
        system_config_path=args.system_config,
        attack_config_path=args.attack_config,
        ingestion_config_path=args.ingestion_config,
        threat_model=args.threat_model,
        poisoned_subagent_ids=args.poisoned_subagent_ids,
    )

    print(f"\n[run_attack_orch] ran {len(logs)} attacked queries")
    for log in logs:
        fd = log.final_decision
        answer = fd.final_answer if fd else "<no decision>"
        print(
            f"  {log.query_id} [{log.attack_condition}] "
            f"poison_retrieved={log.poison_retrieved} "
            f"harmful={fd.harmful_action_flag if fd else '?'} "
            f"answer={answer[:80]!r}"
        )
    print(f"\n[run_attack_orch] runs appended to {args.output_dir}/runs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
