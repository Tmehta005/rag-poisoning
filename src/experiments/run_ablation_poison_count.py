"""
Ablation: vary the number of poison documents (0, 1, 3, 5, 10) on the bio corpus.

Research question: how little poison is enough to achieve retrieval and swing
the orchestrator's final decision?

Runs the orchestrator in targeted mode over all 6 bio-paper attack queries,
once per poison-count level. Each run is logged to results/ablation/runs.jsonl
with metrics["num_poison_docs"] set so the analysis script can group by it.

Usage:
    python -m src.experiments.run_ablation_poison_count
    python -m src.experiments.run_ablation_poison_count --counts 1 2 3 5 --output-dir results/ablation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.corpus.query_loader import load_queries
from src.experiments.run_attack_orch import run_attack_orchestrator
from src.ingestion import ingest_corpus, load_ingestion_config

QUERY_FILE = "data/queries/attack_queries_bio_papers.yaml"
INGESTION_CONFIG = "configs/corpus_bio_papers.yaml"
SYSTEM_CONFIG = "configs/system_orchestrator.yaml"
ATTACK_CONFIG = "configs/attack_main_injection.yaml"
DEFAULT_COUNTS = [0, 1, 3, 5, 10]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ablation: number of poison docs vs. retrieval rate and ASR (bio corpus)."
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=DEFAULT_COUNTS,
        metavar="N",
        help="Poison doc counts to sweep (default: 0 1 3 5 10).",
    )
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument(
        "--threat-model",
        choices=["targeted", "global"],
        default="targeted",
    )
    args = parser.parse_args(argv)

    queries = load_queries(QUERY_FILE)
    print(f"[ablation] loaded {len(queries)} bio queries from {QUERY_FILE}")

    ingestion_cfg = load_ingestion_config(INGESTION_CONFIG)
    print("[ablation] building / loading bio corpus index ...")
    clean_index = ingest_corpus(
        data_dir=ingestion_cfg["data_dir"],
        config=ingestion_cfg,
        persist_dir=ingestion_cfg.get("persist_dir"),
    )
    print("[ablation] bio index ready")

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_runs = 0
    for n in args.counts:
        print(f"\n[ablation] ── num_poison_docs={n} ──────────────────────────")
        logs = run_attack_orchestrator(
            queries=queries,
            clean_index=clean_index,
            output_dir=output_dir,
            system_config_path=SYSTEM_CONFIG,
            attack_config_path=ATTACK_CONFIG,
            ingestion_config_path=INGESTION_CONFIG,
            threat_model=args.threat_model,
            num_poison_docs=n,
        )
        total_runs += len(logs)
        for log in logs:
            fd = log.final_decision
            print(
                f"  {log.query_id}  poison_retrieved={log.poison_retrieved}"
                f"  harmful={fd.harmful_action_flag if fd else '?'}"
            )

    print(f"\n[ablation] done — {total_runs} total runs → {output_dir}/runs.jsonl")
    print("[ablation] run `python -m src.analysis.ablation_table` to see the results table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
