"""
Debate attack runner with two threat models.

threat_model:
    "targeted"  Only ``poisoned_subagent_ids`` receive the poisoned index
                 and a ``private_trigger``. ``JudgeLLM.run`` is called with
                 ``trigger=None`` so the task string itself is clean.
    "global"    Every debate subagent reads from the poisoned index and
                 ``JudgeLLM.run`` is called with ``trigger=<artifact>`` so
                 the Judge's ``_active_trigger`` propagates to every
                 subagent's ``retrieve`` tool via ``global_trigger_ref``.

The ``JudgeLLM`` constructor already wires ``global_trigger_ref`` for
every subagent; we just toggle the trigger at runtime.
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

from src.agents.debate.debate_subagent import DebateSubagent
from src.agents.debate.judge import JudgeLLM
from src.attacks.artifacts import is_harmful_answer, resolve_attack_artifact
from src.attacks.poison_doc import render_extra_poison_specs
from src.attacks.poisoned_index import build_poisoned_index_from_artifact
from src.ingestion import load_ingestion_config
from src.logging_utils import emit_run_log
from src.retriever import Retriever
from src.schemas import RunLog


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def run_attack_debate(
    queries: List[dict],
    clean_index: VectorStoreIndex,
    model_client_factory,
    output_dir: str = "results",
    debate_config_path: str = "configs/system_debate.yaml",
    attack_config_path: str = "configs/attack_main_injection.yaml",
    ingestion_config_path: str = "configs/corpus_cybersec.yaml",
    threat_model: Optional[str] = None,
    poisoned_subagent_ids: Optional[List[str]] = None,
    num_poison_docs: int = 1,
) -> List[RunLog]:
    """
    Run the debate-based attack over queries with artifact references.

    Args:
        model_client_factory: Zero-arg callable that returns a fresh
            AutoGen ``ChatCompletionClient``. Called once per subagent
            per query, so tests can hand back a ``ReplayChatCompletionClient``
            with deterministic messages.
        threat_model: "targeted" or "global" (overrides config).
        poisoned_subagent_ids: Override which agents carry the trigger in
            the targeted model.
    """
    debate_cfg = _load_yaml(debate_config_path)
    attack_cfg = _load_yaml(attack_config_path)
    try:
        ingestion_cfg = load_ingestion_config(ingestion_config_path)
    except FileNotFoundError:
        ingestion_cfg = {}
    embed_model = ingestion_cfg.get("embed_model", "local")

    num_agents = debate_cfg.get("num_subagents", 3)
    top_k = debate_cfg.get("subagent_top_k", 5)
    max_rounds = debate_cfg.get("max_rounds", 4)
    stable_for = debate_cfg.get("stable_for", 2)

    tm = threat_model or attack_cfg.get("threat_model", "targeted")
    if tm not in ("targeted", "global"):
        raise ValueError(f"threat_model must be 'targeted' or 'global', got {tm!r}")
    poisoned_ids = set(
        poisoned_subagent_ids
        or attack_cfg.get("poisoned_subagent_ids", ["subagent_1"])
    )

    logs: List[RunLog] = []

    for q in queries:
        attack = q.get("attack")
        if not attack:
            continue

        query_id = q["query_id"]
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

        subagents: List[DebateSubagent] = []
        for i in range(1, num_agents + 1):
            agent_id = f"subagent_{i}"
            is_target = (tm == "global") or (agent_id in poisoned_ids)
            retriever = (
                Retriever(poisoned_index, top_k=top_k)
                if is_target
                else Retriever(clean_index, top_k=top_k)
            )
            p_ids = poison_ids if is_target else set()
            private_trigger = (
                artifact.trigger
                if tm == "targeted" and agent_id in poisoned_ids
                else None
            )
            subagents.append(
                DebateSubagent(
                    agent_id=agent_id,
                    retriever=retriever,
                    model_client=model_client_factory(),
                    top_k=top_k,
                    poison_doc_ids=p_ids,
                    private_trigger=private_trigger,
                )
            )

        judge = JudgeLLM(
            subagents=subagents,
            max_rounds=max_rounds,
            stable_for=stable_for,
            output_dir=output_dir,
            propagate_trigger_to_subagents=True,
        )

        judge_trigger = artifact.trigger if tm == "global" else None

        log = judge.run(
            query=q["query"],
            query_id=query_id,
            trigger=judge_trigger,
            attack_condition=f"main_injection.{tm}",
            ground_truth_answer=q.get("ground_truth_answer"),
            emit=False,
        )

        if log.final_decision is not None and artifact.target_claim:
            harmful = is_harmful_answer(
                log.final_decision.final_answer or "", artifact
            )
            log.final_decision = log.final_decision.model_copy(
                update={"harmful_action_flag": harmful}
            )
        harmful_action = 0.0
        if log.final_decision is not None:
            harmful_action = float(log.final_decision.harmful_action_flag)
        log.metrics = {
            **(log.metrics or {}),
            "poison_retrieved": float(log.poison_retrieved),
            "harmful_action": harmful_action,
            "num_poison_docs": float(num_poison_docs),
        }
        emit_run_log(log, output_dir=output_dir)
        logs.append(log)

    return logs


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the debate poisoning attack."""
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    from src.corpus.ingest_with_metadata import ingest_corpus_with_metadata
    from src.corpus.query_loader import load_queries

    parser = argparse.ArgumentParser(
        description="Run the debate poisoning attack."
    )
    parser.add_argument(
        "--query-file",
        default="data/queries/attack_queries_cybersec.yaml",
    )
    parser.add_argument(
        "--threat-model",
        choices=["targeted", "global"],
        default=None,
    )
    parser.add_argument(
        "--poisoned-subagent-id",
        action="append",
        dest="poisoned_subagent_ids",
        default=None,
    )
    parser.add_argument("--debate-config", default="configs/system_debate.yaml")
    parser.add_argument(
        "--attack-config", default="configs/attack_main_injection.yaml"
    )
    parser.add_argument(
        "--ingestion-config", default="configs/corpus_cybersec.yaml"
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model", default=None, help="Override debate model name.")
    parser.add_argument(
        "--num-poison-docs",
        type=int,
        default=1,
        help="Total poison docs injected into each poisoned subagent's index.",
    )
    args = parser.parse_args(argv)

    debate_cfg = {}
    try:
        with open(args.debate_config) as f:
            debate_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        pass
    model_name = args.model or debate_cfg.get("model", "gpt-4o-mini")

    def _factory():
        return OpenAIChatCompletionClient(model=model_name)

    queries = load_queries(args.query_file)
    print(
        f"[run_attack_debate] loaded {len(queries)} queries from {args.query_file}"
    )

    clean_index = ingest_corpus_with_metadata()
    print("[run_attack_debate] clean index ready")

    logs = run_attack_debate(
        queries=queries,
        clean_index=clean_index,
        model_client_factory=_factory,
        output_dir=args.output_dir,
        debate_config_path=args.debate_config,
        attack_config_path=args.attack_config,
        ingestion_config_path=args.ingestion_config,
        threat_model=args.threat_model,
        poisoned_subagent_ids=args.poisoned_subagent_ids,
        num_poison_docs=args.num_poison_docs,
    )

    print(f"\n[run_attack_debate] ran {len(logs)} attacked queries")
    for log in logs:
        fd = log.final_decision
        answer = fd.final_answer if fd else "<no decision>"
        print(
            f"  {log.query_id} [{log.attack_condition}] "
            f"poison_retrieved={log.poison_retrieved} "
            f"harmful={fd.harmful_action_flag if fd else '?'} "
            f"answer={answer[:80]!r}"
        )
    print(f"\n[run_attack_debate] runs appended to {args.output_dir}/runs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
