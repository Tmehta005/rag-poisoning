"""
Offline inspector for ``results/runs.jsonl``.

Use this to re-read debate or orchestrator runs after the fact. Renders
each record via :func:`src.attacks.debug_format.format_run_log`.

Usage::

    # Pretty-print every run in the file
    python -m src.experiments.inspect_runs

    # Filter by query_id / attack_condition and keep only the latest entry
    python -m src.experiments.inspect_runs --query-id q001 --only-last

    # Just the debate attack runs, with rationales/prompts shown
    python -m src.experiments.inspect_runs --attack-condition main_injection --show-prompts
"""

from __future__ import annotations

import argparse

from src.attacks.debug_format import (
    filter_runs,
    format_run_log,
    iter_runs_jsonl,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretty-print runs.jsonl records.")
    p.add_argument("--runs-path", default="results/runs.jsonl")
    p.add_argument("--query-id", help="Only show runs with this query_id")
    p.add_argument(
        "--attack-condition",
        help='Filter by attack_condition (e.g. "clean" or "main_injection")',
    )
    p.add_argument(
        "--only-last",
        action="store_true",
        help="Keep only the most recent record per (query_id, attack_condition) pair.",
    )
    p.add_argument(
        "--show-prompts",
        action="store_true",
        help="Include subagent rationales / reasoning summaries.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate output to the first N matching records.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    records = list(iter_runs_jsonl(args.runs_path))
    filtered = filter_runs(
        records,
        query_id=args.query_id,
        attack_condition=args.attack_condition,
        only_last=args.only_last,
    )
    if args.limit is not None:
        filtered = filtered[: args.limit]

    if not filtered:
        print(f"No records matched in {args.runs_path}")
        return

    for rec in filtered:
        print(format_run_log(rec, show_prompts=args.show_prompts))
        print()


if __name__ == "__main__":
    main()
