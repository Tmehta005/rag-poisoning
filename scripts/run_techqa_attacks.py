"""
Run the full attack ladder over the 6 TechQA candidate queries.

Mirrors the cybersec/bio battery in scripts/run_all_experiments.py but
restricted to the controlled 6-query TechQA smoke set:
  C1 — single-agent attack (ceiling)
  C2 — orchestrator targeted (subagent_1 poisoned only)
  C3 — orchestrator global   (all subagents poisoned, trigger leaked to state)
  C4 — debate targeted       (subagent_1 poisoned only)
  C5 — debate global         (all subagents poisoned, trigger leaked to judge)

Uses num_poison_docs=3 (the lever validated in the prior cybersec/bio run).
Appends to results/runs.jsonl in the same schema as cybersec/bio.
"""

from __future__ import annotations

import argparse

from _techqa_common import setup

setup()

from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.corpus.ingest_with_metadata import ingest_corpus_with_metadata
from src.corpus.query_loader import load_queries
from src.experiments.run_attack_debate import run_attack_debate
from src.experiments.run_attack_orch import run_attack_orchestrator
from src.experiments.run_attack_single_agent import run_attack_single_agent

DEFAULT_QUERY_FILE = "data/queries/attack_queries_techqa.yaml"
CORPUS_CONFIG = "configs/corpus_techqa.yaml"
SINGLE_CONFIG = "configs/system_single_agent.yaml"
ORCH_CONFIG = "configs/system_orchestrator.yaml"
DEBATE_CONFIG = "configs/system_debate.yaml"
ATTACK_CONFIG = "configs/attack_main_injection.yaml"
NUM_POISON_DOCS = 3
OUTPUT_DIR = "results"


def _client_factory(model="gpt-4o-mini"):
    return lambda: OpenAIChatCompletionClient(model=model)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--query-file", default=DEFAULT_QUERY_FILE)
    args = p.parse_args()

    queries = load_queries(args.query_file)
    print(f"[techqa_attacks] {len(queries)} attack queries loaded from {args.query_file}")

    print("[techqa_attacks] loading clean TechQA index …")
    clean_index = ingest_corpus_with_metadata(config_path=CORPUS_CONFIG)
    print("[techqa_attacks] index ready")

    print("\n=== C1: single-agent attack (ceiling) ===")
    run_attack_single_agent(
        queries=queries,
        clean_index=clean_index,
        output_dir=OUTPUT_DIR,
        system_config_path=SINGLE_CONFIG,
        ingestion_config_path=CORPUS_CONFIG,
        num_poison_docs=NUM_POISON_DOCS,
    )

    print("\n=== C2: orchestrator targeted ===")
    run_attack_orchestrator(
        queries=queries,
        clean_index=clean_index,
        output_dir=OUTPUT_DIR,
        system_config_path=ORCH_CONFIG,
        attack_config_path=ATTACK_CONFIG,
        ingestion_config_path=CORPUS_CONFIG,
        threat_model="targeted",
        poisoned_subagent_ids=["subagent_1"],
        num_poison_docs=NUM_POISON_DOCS,
    )

    print("\n=== C3: orchestrator global ===")
    run_attack_orchestrator(
        queries=queries,
        clean_index=clean_index,
        output_dir=OUTPUT_DIR,
        system_config_path=ORCH_CONFIG,
        attack_config_path=ATTACK_CONFIG,
        ingestion_config_path=CORPUS_CONFIG,
        threat_model="global",
        num_poison_docs=NUM_POISON_DOCS,
    )

    print("\n=== C4: debate targeted ===")
    run_attack_debate(
        queries=queries,
        clean_index=clean_index,
        model_client_factory=_client_factory("gpt-4o-mini"),
        output_dir=OUTPUT_DIR,
        debate_config_path=DEBATE_CONFIG,
        attack_config_path=ATTACK_CONFIG,
        ingestion_config_path=CORPUS_CONFIG,
        threat_model="targeted",
        poisoned_subagent_ids=["subagent_1"],
        num_poison_docs=NUM_POISON_DOCS,
    )

    print("\n=== C5: debate global ===")
    run_attack_debate(
        queries=queries,
        clean_index=clean_index,
        model_client_factory=_client_factory("gpt-4o-mini"),
        output_dir=OUTPUT_DIR,
        debate_config_path=DEBATE_CONFIG,
        attack_config_path=ATTACK_CONFIG,
        ingestion_config_path=CORPUS_CONFIG,
        threat_model="global",
        num_poison_docs=NUM_POISON_DOCS,
    )

    print("\n[techqa_attacks] all 5 conditions complete; runs appended to results/runs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
