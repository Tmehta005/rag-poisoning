"""
Run the clean debate pipeline over the TechQA query set.

Mirrors ``scripts/run_techqa_clean.py`` but invokes
:func:`src.experiments.run_debate_clean.run_clean_debate_experiment` so we
have single-agent / orchestrator / debate on the same TechQA queries.

Usage::

    python scripts/run_techqa_debate.py
"""

from __future__ import annotations

import argparse

from _techqa_common import setup

setup()

from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.corpus.query_loader import load_queries
from src.experiments.run_debate_clean import run_clean_debate_experiment
from src.ingestion import load_ingestion_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the clean TechQA debate experiment.")
    parser.add_argument(
        "--query-file",
        default="data/queries/techqa_queries.yaml",
    )
    parser.add_argument(
        "--corpus-config",
        default="configs/corpus_techqa.yaml",
    )
    parser.add_argument(
        "--debate-config",
        default="configs/system_debate.yaml",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for the debate subagents (default: gpt-4o-mini).",
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    corpus_cfg = load_ingestion_config(args.corpus_config)
    data_dir = corpus_cfg.get("data_dir", "data/corpus_techqa")
    persist_dir = corpus_cfg.get("persist_dir", "data/index_techqa")

    queries = load_queries(args.query_file)
    print(f"[run_techqa_debate] loaded {len(queries)} queries from {args.query_file}")

    logs = run_clean_debate_experiment(
        queries=queries,
        data_dir=data_dir,
        persist_dir=persist_dir,
        output_dir=args.output_dir,
        ingestion_config_path=args.corpus_config,
        debate_config_path=args.debate_config,
        model_client=OpenAIChatCompletionClient(model=args.model),
    )

    print(f"\n[run_techqa_debate] ran {len(logs)} queries")
    for log in logs:
        fd = log.final_decision
        t = log.debate_transcript
        if fd is None:
            print(f"  {log.query_id}  <no decision>")
            continue
        ans = (fd.final_answer or "").replace("\n", " ")[:90]
        rounds = t.rounds_used if t else "?"
        maj = len(t.majority_cluster) if t else "?"
        stop = t.stopped_reason if t else "?"
        print(
            f"  {log.query_id}  conf={fd.final_confidence:.2f}  "
            f"rounds={rounds}  majority={maj}/3  stop={stop}  {ans}"
        )
    print(f"\n[run_techqa_debate] runs appended to {args.output_dir}/runs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
