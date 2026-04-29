"""
Run the clean orchestrator pipeline over the TechQA query set.

Loads ``data/queries/techqa_queries.yaml`` and invokes
:func:`src.experiments.run_clean.run_clean_experiment` with the TechQA
corpus config. Appends one row per query to ``results/runs.jsonl`` —
schema-identical to the cybersec / bio runs, so the same analysis tooling
(``src.analysis.make_results_table``) groups TechQA in the same tables
when filtered.

Usage::

    python scripts/run_techqa_clean.py
    python scripts/run_techqa_clean.py --output-dir /tmp/techqa_results
"""

from __future__ import annotations

import argparse

from _techqa_common import setup

setup()

from src.corpus.query_loader import load_queries
from src.experiments.run_clean import run_clean_experiment
from src.ingestion import load_ingestion_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the clean TechQA experiment.")
    parser.add_argument(
        "--query-file",
        default="data/queries/techqa_queries.yaml",
        help="TechQA queries YAML (default: data/queries/techqa_queries.yaml).",
    )
    parser.add_argument(
        "--corpus-config",
        default="configs/corpus_techqa.yaml",
        help="TechQA corpus config (default: configs/corpus_techqa.yaml).",
    )
    parser.add_argument(
        "--system-config",
        default="configs/system_orchestrator.yaml",
        help="Orchestrator system config (default: configs/system_orchestrator.yaml).",
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    corpus_cfg = load_ingestion_config(args.corpus_config)
    data_dir = corpus_cfg.get("data_dir", "data/corpus_techqa")
    persist_dir = corpus_cfg.get("persist_dir", "data/index_techqa")

    if not Path(args.query_file).exists():
        raise SystemExit(
            f"Query file not found: {args.query_file}\n"
            f"Run `python scripts/prepare_techqa.py` first."
        )

    queries = load_queries(args.query_file)
    print(f"[run_techqa_clean] loaded {len(queries)} queries from {args.query_file}")
    print(f"[run_techqa_clean] data_dir={data_dir} persist_dir={persist_dir}")

    logs = run_clean_experiment(
        queries=queries,
        data_dir=data_dir,
        persist_dir=persist_dir,
        output_dir=args.output_dir,
        ingestion_config_path=args.corpus_config,
        system_config_path=args.system_config,
    )

    print(f"\n[run_techqa_clean] ran {len(logs)} queries")
    for log in logs:
        fd = log.final_decision
        if fd is None:
            print(f"  {log.query_id}  <no decision>")
            continue
        ans = (fd.final_answer or "").replace("\n", " ")[:120]
        print(f"  {log.query_id}  conf={fd.final_confidence:.2f}  {ans}")
    print(f"\n[run_techqa_clean] runs appended to {args.output_dir}/runs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
