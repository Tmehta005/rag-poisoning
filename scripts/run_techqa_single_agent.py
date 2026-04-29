"""
Run the clean single-agent pipeline over the TechQA query set.

Mirrors ``scripts/run_techqa_clean.py`` (orchestrator) but invokes
:func:`src.experiments.run_single_agent.run_single_agent_experiment` so we
can compare single-agent / orchestrator / debate on the same queries.

Usage::

    python scripts/run_techqa_single_agent.py
"""

from __future__ import annotations

import argparse

from _techqa_common import setup

setup()

from src.corpus.query_loader import load_queries
from src.experiments.run_single_agent import run_single_agent_experiment
from src.ingestion import load_ingestion_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the clean TechQA single-agent experiment.")
    parser.add_argument(
        "--query-file",
        default="data/queries/techqa_queries.yaml",
    )
    parser.add_argument(
        "--corpus-config",
        default="configs/corpus_techqa.yaml",
    )
    parser.add_argument(
        "--system-config",
        default="configs/system_single_agent.yaml",
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    corpus_cfg = load_ingestion_config(args.corpus_config)
    data_dir = corpus_cfg.get("data_dir", "data/corpus_techqa")
    persist_dir = corpus_cfg.get("persist_dir", "data/index_techqa")

    queries = load_queries(args.query_file)
    print(f"[run_techqa_single_agent] loaded {len(queries)} queries from {args.query_file}")

    logs = run_single_agent_experiment(
        queries=queries,
        data_dir=data_dir,
        persist_dir=persist_dir,
        output_dir=args.output_dir,
        ingestion_config_path=args.corpus_config,
        system_config_path=args.system_config,
    )

    print(f"\n[run_techqa_single_agent] ran {len(logs)} queries")
    for log in logs:
        fd = log.final_decision
        if fd is None:
            print(f"  {log.query_id}  <no decision>")
            continue
        ans = (fd.final_answer or "").replace("\n", " ")[:120]
        print(f"  {log.query_id}  conf={fd.final_confidence:.2f}  {ans}")
    print(f"\n[run_techqa_single_agent] runs appended to {args.output_dir}/runs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
