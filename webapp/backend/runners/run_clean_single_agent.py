"""
CLI wrapper around :func:`src.experiments.run_single_agent.run_single_agent_experiment`.

Provides the argparse surface the webapp experiment endpoint uses to invoke
the single-agent clean baseline (attack ceiling) via subprocess. Mirrors
:mod:`webapp.backend.runners.run_clean_orch`.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.corpus.query_loader import load_queries
from src.experiments.run_single_agent import run_single_agent_experiment

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
    parser = argparse.ArgumentParser(
        description="Run the single-agent clean experiment (attack ceiling)."
    )
    parser.add_argument(
        "--query-file",
        default="data/queries/sample_cybersec_queries.yaml",
    )
    parser.add_argument(
        "--system-config",
        default="configs/system_orchestrator.yaml",
        help="Reused for model + top_k; num_subagents is forced to 1.",
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
    print(
        f"[run_clean_single_agent] loaded {len(queries)} queries from {args.query_file}",
        flush=True,
    )

    logs = run_single_agent_experiment(
        queries=queries,
        data_dir=data_dir,
        persist_dir=persist_dir,
        output_dir=args.output_dir,
        ingestion_config_path=args.ingestion_config,
        system_config_path=args.system_config,
    )

    print(f"[run_clean_single_agent] ran {len(logs)} queries", flush=True)
    for log in logs:
        fd = log.final_decision
        answer = (fd.final_answer if fd else "<no decision>")[:80]
        conf = fd.final_confidence if fd else None
        print(f"  {log.query_id} conf={conf} answer={answer!r}", flush=True)

    print(
        "__RESULT__ "
        + json.dumps(
            {
                "num_runs": len(logs),
                "query_ids": [log.query_id for log in logs],
                "attack_condition": "clean",
                "system": "single",
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
