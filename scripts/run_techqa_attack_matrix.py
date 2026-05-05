"""
Run the attack matrix over the TechQA-100 attack queries.

Conditions (matches scripts/run_techqa_attacks.py for the TechQA-13 set, but
parameterized for any TechQA-shaped attacked queries YAML):
  C1 single-agent targeted
  C2 orchestrator targeted   (subagent_1 poisoned)
  C3 orchestrator global     (all subagents poisoned, trigger leaked)
  C4 debate targeted         (subagent_1 poisoned)
  C5 debate global           (all subagents poisoned, trigger leaked)  -- opt-in

Default conditions: C1, C2, C3, C4. C5 must be opted in via
``--include-debate-global`` because it is by far the most expensive cell of
the matrix (5 debaters x N rounds, all on the poisoned index, every query).

Each runner appends to ``{output_dir}/runs.jsonl``. Resumability is implemented
by reading any existing rows and filtering the input queries per condition so
a re-run picks up only what is missing.

Usage::

    .venv/bin/python scripts/run_techqa_attack_matrix.py --dry-run
    .venv/bin/python scripts/run_techqa_attack_matrix.py
    .venv/bin/python scripts/run_techqa_attack_matrix.py --include-debate-global
    .venv/bin/python scripts/run_techqa_attack_matrix.py --conditions C1,C2
    .venv/bin/python scripts/run_techqa_attack_matrix.py --limit 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from _techqa_common import setup

setup()


_DEFAULT_QUERY_FILE = "data/queries/techqa_100_seed0_attack_smoke5_attacked.yaml"
_DEFAULT_CORPUS = "configs/corpus_techqa_100_seed0.yaml"
_DEFAULT_OUTPUT_DIR = "results/techqa_100_seed0_attack_smoke"
_DEFAULT_NUM_POISON = 3

_SINGLE_CONFIG = "configs/system_single_agent.yaml"
_ORCH_CONFIG = "configs/system_orchestrator.yaml"
_DEBATE_CONFIG = "configs/system_debate.yaml"
_ATTACK_CONFIG = "configs/attack_main_injection.yaml"


_ALL_CONDITIONS = ("C1", "C2", "C3", "C4", "C5")
_CONDITION_INFO = {
    "C1": ("single-agent targeted", "single-agent", "main_injection.targeted"),
    "C2": ("orchestrator targeted", "orchestrator", "main_injection.targeted"),
    "C3": ("orchestrator global",   "orchestrator", "main_injection.global"),
    "C4": ("debate targeted",        "debate",       "main_injection.targeted"),
    "C5": ("debate global",          "debate",       "main_injection.global"),
}


def _system_of(run: dict) -> str:
    n = len(run.get("agent_responses", {}))
    if n == 1:
        return "single-agent"
    if run.get("debate_transcript") is not None:
        return "debate"
    return "orchestrator"


def _load_existing_runs(runs_path: Path) -> set[tuple[str, str, str]]:
    """Return {(query_id, system, attack_condition), ...} for rows already on disk."""
    if not runs_path.exists():
        return set()
    seen: set[tuple[str, str, str]] = set()
    with runs_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = r.get("query_id")
            cond = r.get("attack_condition")
            if qid and cond:
                seen.add((qid, _system_of(r), cond))
    return seen


def _filter_queries(
    queries: list[dict],
    seen: set[tuple[str, str, str]],
    system: str,
    attack_condition: str,
) -> list[dict]:
    return [
        q for q in queries
        if (q.get("query_id"), system, attack_condition) not in seen
    ]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--query-file", default=_DEFAULT_QUERY_FILE)
    p.add_argument("--corpus-config", default=_DEFAULT_CORPUS)
    p.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-poison-docs", type=int, default=_DEFAULT_NUM_POISON)
    p.add_argument("--conditions", default="C1,C2,C3,C4",
                   help="Comma-separated subset of {C1,C2,C3,C4,C5}. "
                        "Default omits C5 (debate global) which is the most expensive.")
    p.add_argument("--include-debate-global", action="store_true",
                   help="Append C5 to the conditions list (overrides --conditions if missing).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of queries fed to each condition (after dedup).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-skip-done", action="store_true",
                   help="Don't dedup against existing runs.jsonl rows.")
    p.add_argument("--debate-model", default="gpt-4o-mini",
                   help="Model for debate subagents/judge (matches run_techqa_attacks.py).")
    args = p.parse_args(argv)

    query_path = Path(args.query_file)
    if not query_path.exists():
        print(f"[matrix] query file not found: {query_path}", file=sys.stderr)
        return 1

    requested = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    for c in requested:
        if c not in _ALL_CONDITIONS:
            print(f"[matrix] unknown condition {c!r} (valid: {_ALL_CONDITIONS})", file=sys.stderr)
            return 1
    if args.include_debate_global and "C5" not in requested:
        requested.append("C5")
    requested = list(dict.fromkeys(requested))  # dedup, preserve order

    output_dir = Path(args.output_dir)
    runs_path = output_dir / "runs.jsonl"

    seen = set() if args.no_skip_done else _load_existing_runs(runs_path)

    # Print plan
    print()
    print("=" * 72)
    print(f"  Attack matrix plan")
    print("=" * 72)
    print(f"  query-file:       {query_path}")
    print(f"  corpus-config:    {args.corpus_config}")
    print(f"  output-dir:       {output_dir}")
    print(f"  num-poison-docs:  {args.num_poison_docs}")
    print(f"  conditions:       {','.join(requested)}")
    print(f"  existing rows:    {len(seen)} (in {runs_path})")
    print()

    # Defer heavy imports until we need them.
    from src.corpus.query_loader import load_queries
    queries = load_queries(str(query_path))
    if not queries:
        print(f"[matrix] no queries in {query_path}", file=sys.stderr)
        return 1

    # Plan per condition
    plan: dict[str, list[dict]] = {}
    for c in requested:
        label, system, cond = _CONDITION_INFO[c]
        filtered = (
            queries if args.no_skip_done
            else _filter_queries(queries, seen, system, cond)
        )
        if args.limit is not None:
            filtered = filtered[: args.limit]
        plan[c] = filtered
        print(f"  {c}  {label:<24}  {len(filtered)}/{len(queries)} queries")
    print()

    if args.dry_run:
        for c, qs in plan.items():
            label, system, cond = _CONDITION_INFO[c]
            print(f"  [dry-run] {c} ({label}):")
            for q in qs:
                print(f"    - {q.get('query_id')}")
        return 0

    if not any(plan.values()):
        print("[matrix] nothing to run (all rows already present).")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from src.corpus.ingest_with_metadata import ingest_corpus_with_metadata
    from src.experiments.run_attack_debate import run_attack_debate
    from src.experiments.run_attack_orch import run_attack_orchestrator
    from src.experiments.run_attack_single_agent import run_attack_single_agent

    print("[matrix] loading clean index ...")
    clean_index = ingest_corpus_with_metadata(config_path=args.corpus_config)
    print("[matrix] index ready")

    debate_factory = lambda: OpenAIChatCompletionClient(model=args.debate_model)

    output_dir_str = str(output_dir)

    if "C1" in plan and plan["C1"]:
        print(f"\n=== C1 single-agent targeted ({len(plan['C1'])} queries) ===")
        run_attack_single_agent(
            queries=plan["C1"],
            clean_index=clean_index,
            output_dir=output_dir_str,
            system_config_path=_SINGLE_CONFIG,
            ingestion_config_path=args.corpus_config,
            num_poison_docs=args.num_poison_docs,
        )

    if "C2" in plan and plan["C2"]:
        print(f"\n=== C2 orchestrator targeted ({len(plan['C2'])} queries) ===")
        run_attack_orchestrator(
            queries=plan["C2"],
            clean_index=clean_index,
            output_dir=output_dir_str,
            system_config_path=_ORCH_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=args.corpus_config,
            threat_model="targeted",
            poisoned_subagent_ids=["subagent_1"],
            num_poison_docs=args.num_poison_docs,
        )

    if "C3" in plan and plan["C3"]:
        print(f"\n=== C3 orchestrator global ({len(plan['C3'])} queries) ===")
        run_attack_orchestrator(
            queries=plan["C3"],
            clean_index=clean_index,
            output_dir=output_dir_str,
            system_config_path=_ORCH_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=args.corpus_config,
            threat_model="global",
            num_poison_docs=args.num_poison_docs,
        )

    if "C4" in plan and plan["C4"]:
        print(f"\n=== C4 debate targeted ({len(plan['C4'])} queries) ===")
        run_attack_debate(
            queries=plan["C4"],
            clean_index=clean_index,
            model_client_factory=debate_factory,
            output_dir=output_dir_str,
            debate_config_path=_DEBATE_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=args.corpus_config,
            threat_model="targeted",
            poisoned_subagent_ids=["subagent_1"],
            num_poison_docs=args.num_poison_docs,
        )

    if "C5" in plan and plan["C5"]:
        print(f"\n=== C5 debate global ({len(plan['C5'])} queries) ===")
        run_attack_debate(
            queries=plan["C5"],
            clean_index=clean_index,
            model_client_factory=debate_factory,
            output_dir=output_dir_str,
            debate_config_path=_DEBATE_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=args.corpus_config,
            threat_model="global",
            num_poison_docs=args.num_poison_docs,
        )

    print(f"\n[matrix] all conditions complete; runs appended to {runs_path}")
    print()
    print("  Next step (judge):")
    print(f"    .venv/bin/python -m src.analysis.rescore_llm_judge_techqa \\")
    print(f"      --query-file  {query_path} \\")
    print(f"      --runs-file   {runs_path} \\")
    print(f"      --scores-file {output_dir}/judge_scores.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
