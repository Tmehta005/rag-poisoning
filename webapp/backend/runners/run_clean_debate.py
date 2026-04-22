"""CLI wrapper around :func:`src.experiments.run_debate_clean.run_clean_debate_experiment`."""

from __future__ import annotations

import argparse
import json
import sys

from src.corpus.query_loader import load_queries
from src.experiments.run_debate_clean import run_clean_debate_experiment

_LEGACY = {
    "cybersec": ("data/corpus_cybersec", "data/index_cybersec"),
    "generic": ("data/corpus", "data/index"),
}


def _resolve_corpus_paths(args) -> tuple[str, str]:
    if args.data_dir and args.persist_dir:
        return args.data_dir, args.persist_dir
    if args.corpus and args.corpus in _LEGACY:
        return _LEGACY[args.corpus]
    return (
        args.data_dir or "data/corpus_cybersec",
        args.persist_dir or "data/index_cybersec",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the clean debate experiment.")
    parser.add_argument(
        "--query-file",
        default="data/queries/sample_cybersec_queries.yaml",
    )
    parser.add_argument(
        "--debate-config",
        default="configs/system_debate.yaml",
    )
    parser.add_argument(
        "--ingestion-config",
        default="configs/corpus_cybersec.yaml",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Legacy shortcut: 'cybersec' or 'generic'. Ignored if --data-dir is given.",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    data_dir, persist_dir = _resolve_corpus_paths(args)

    queries = load_queries(args.query_file)
    print(f"[run_clean_debate] loaded {len(queries)} queries from {args.query_file}", flush=True)

    logs = run_clean_debate_experiment(
        queries=queries,
        data_dir=data_dir,
        persist_dir=persist_dir,
        output_dir=args.output_dir,
        ingestion_config_path=args.ingestion_config,
        debate_config_path=args.debate_config,
    )

    print(f"[run_clean_debate] ran {len(logs)} queries", flush=True)
    for log in logs:
        t = log.debate_transcript
        fd = log.final_decision
        answer = (fd.final_answer if fd else "<no decision>")[:80]
        rounds = t.rounds_used if t else "?"
        print(f"  {log.query_id} rounds={rounds} answer={answer!r}", flush=True)

    print(
        "__RESULT__ "
        + json.dumps(
            {
                "num_runs": len(logs),
                "query_ids": [log.query_id for log in logs],
                "attack_condition": "clean",
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
