"""
Full experiment battery for presentation-ready results.

Runs all 16 conditions (8 cybersec + 8 bio) and appends to results/runs.jsonl.
Identifies system from RunLog structure at analysis time:
  len(agent_responses)==1                → single-agent
  len(agent_responses)==3, no transcript → orchestrator
  len(agent_responses)==3, has transcript → debate

Usage:
    python scripts/run_all_experiments.py [--phase A|B|all] [--output-dir results]

Prerequisites:
    - Cybersec index built: data/index_cybersec/
    - Bio index built:      data/index_bio_papers/
    - attack_001 artifact:  data/attacks/attack_001/artifact.json  (cybersec)
    - attack_002 artifact:  data/attacks/attack_002/artifact.json  (bio b001)
    - attack_003 artifact:  data/attacks/attack_003/artifact.json  (bio b002)

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path when run as a script
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.corpus.ingest_with_metadata import ingest_corpus_with_metadata
from src.corpus.query_loader import load_queries
from src.experiments.run_clean import run_clean_experiment
from src.experiments.run_single_agent import run_single_agent_experiment
from src.experiments.run_attack_single_agent import run_attack_single_agent
from src.experiments.run_attack_orch import run_attack_orchestrator
from src.experiments.run_debate_clean import run_clean_debate_experiment
from src.experiments.run_attack_debate import run_attack_debate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CYBERSEC_INGESTION  = "configs/corpus_cybersec.yaml"
_BIO_INGESTION       = "configs/corpus_bio_papers.yaml"
_ORCH_CONFIG         = "configs/system_orchestrator.yaml"
_SINGLE_CONFIG       = "configs/system_single_agent.yaml"
_DEBATE_CONFIG       = "configs/system_debate.yaml"
_ATTACK_CONFIG       = "configs/attack_main_injection.yaml"

_CYBERSEC_CLEAN_Q    = "data/queries/sample_cybersec_queries.yaml"
_CYBERSEC_ATTACK_Q   = "data/queries/attack_queries_cybersec.yaml"
_BIO_CLEAN_Q         = "data/queries/sample_bio_queries.yaml"
_BIO_ATTACK_Q        = "data/queries/attack_queries_bio_papers.yaml"

# Option 2: all 6 bio queries have per-query artifacts
_BIO_QUERY_IDS = {"b001", "b002", "b003", "b004", "b005", "b006"}


def _bio_clean_queries() -> list[dict]:
    """Load sample_bio_queries.yaml filtered to the attack query set."""
    all_q = load_queries(_BIO_CLEAN_Q)
    return [q for q in all_q if q["query_id"] in _BIO_QUERY_IDS]


def _debate_client_factory(model: str = "gpt-4o-mini"):
    """Return a zero-arg callable that produces a fresh AutoGen client."""
    def _factory():
        return OpenAIChatCompletionClient(model=model)
    return _factory


def _banner(label: str) -> None:
    print(f"\n{'='*60}\n  {label}\n{'='*60}")


# ---------------------------------------------------------------------------
# Phase A — Cybersec corpus (8 queries, attack_001)
# ---------------------------------------------------------------------------

def run_phase_a(output_dir: str, num_trials: int = 1, num_poison_docs: int = 1) -> None:
    _banner("Phase A: Cybersec corpus")

    print("[A] Loading cybersec index …")
    cybersec_index = ingest_corpus_with_metadata(config_path=_CYBERSEC_INGESTION)
    cybersec_clean_q  = load_queries(_CYBERSEC_CLEAN_Q)
    cybersec_attack_q = load_queries(_CYBERSEC_ATTACK_Q)
    print(f"[A] {len(cybersec_clean_q)} clean queries, {len(cybersec_attack_q)} attack queries, {num_trials} trial(s)")

    for trial in range(1, num_trials + 1):
        _banner(f"A1: single-agent clean (cybersec) — trial {trial}/{num_trials}")
        run_single_agent_experiment(
            queries=cybersec_clean_q,
            data_dir="data/corpus_cybersec",
            persist_dir="data/index_cybersec",
            output_dir=output_dir,
            ingestion_config_path=_CYBERSEC_INGESTION,
            system_config_path=_SINGLE_CONFIG,
        )
    print("[A1] done")

    for trial in range(1, num_trials + 1):
        _banner(f"A2: single-agent targeted attack (cybersec) — CEILING — trial {trial}/{num_trials}")
        run_attack_single_agent(
            queries=cybersec_attack_q,
            clean_index=cybersec_index,
            output_dir=output_dir,
            system_config_path=_SINGLE_CONFIG,
            ingestion_config_path=_CYBERSEC_INGESTION,
            num_poison_docs=num_poison_docs,
        )
    print("[A2] done")

    for trial in range(1, num_trials + 1):
        _banner(f"A3: orchestrator clean (cybersec) — trial {trial}/{num_trials}")
        run_clean_experiment(
            queries=cybersec_clean_q,
            data_dir="data/corpus_cybersec",
            persist_dir="data/index_cybersec",
            output_dir=output_dir,
            ingestion_config_path=_CYBERSEC_INGESTION,
            system_config_path=_ORCH_CONFIG,
        )
    print("[A3] done")

    for trial in range(1, num_trials + 1):
        _banner(f"A4: orchestrator targeted attack (cybersec) — trial {trial}/{num_trials}")
        run_attack_orchestrator(
            queries=cybersec_attack_q,
            clean_index=cybersec_index,
            output_dir=output_dir,
            system_config_path=_ORCH_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=_CYBERSEC_INGESTION,
            threat_model="targeted",
            poisoned_subagent_ids=["subagent_1"],
            num_poison_docs=num_poison_docs,
        )
    print("[A4] done")

    for trial in range(1, num_trials + 1):
        _banner(f"A5: orchestrator global attack (cybersec) — trial {trial}/{num_trials}")
        run_attack_orchestrator(
            queries=cybersec_attack_q,
            clean_index=cybersec_index,
            output_dir=output_dir,
            system_config_path=_ORCH_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=_CYBERSEC_INGESTION,
            threat_model="global",
            num_poison_docs=num_poison_docs,
        )
    print("[A5] done")

    for trial in range(1, num_trials + 1):
        _banner(f"A6: debate clean (cybersec) — trial {trial}/{num_trials}")
        run_clean_debate_experiment(
            queries=cybersec_clean_q,
            data_dir="data/corpus_cybersec",
            persist_dir="data/index_cybersec",
            output_dir=output_dir,
            ingestion_config_path=_CYBERSEC_INGESTION,
            debate_config_path=_DEBATE_CONFIG,
            model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        )
    print("[A6] done")

    for trial in range(1, num_trials + 1):
        _banner(f"A7: debate targeted attack (cybersec) — trial {trial}/{num_trials}")
        run_attack_debate(
            queries=cybersec_attack_q,
            clean_index=cybersec_index,
            model_client_factory=_debate_client_factory("gpt-4o-mini"),
            output_dir=output_dir,
            debate_config_path=_DEBATE_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=_CYBERSEC_INGESTION,
            threat_model="targeted",
            poisoned_subagent_ids=["subagent_1"],
            num_poison_docs=num_poison_docs,
        )
    print("[A7] done")

    for trial in range(1, num_trials + 1):
        _banner(f"A8: debate global attack (cybersec) — trial {trial}/{num_trials}")
        run_attack_debate(
            queries=cybersec_attack_q,
            clean_index=cybersec_index,
            model_client_factory=_debate_client_factory("gpt-4o-mini"),
            output_dir=output_dir,
            debate_config_path=_DEBATE_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=_CYBERSEC_INGESTION,
            threat_model="global",
            num_poison_docs=num_poison_docs,
        )
    print("[A8] done")


# ---------------------------------------------------------------------------
# Phase B — Bio corpus (b001 + b002)
# ---------------------------------------------------------------------------

def run_phase_b(output_dir: str, num_trials: int = 1, num_poison_docs: int = 1) -> None:
    _banner("Phase B: Bio corpus (b001–b006)")
    print("[B] Loading bio index …")
    bio_index = ingest_corpus_with_metadata(config_path=_BIO_INGESTION)
    bio_clean_q  = _bio_clean_queries()
    bio_attack_q = load_queries(_BIO_ATTACK_Q)
    print(f"[B] {len(bio_clean_q)} clean queries, {len(bio_attack_q)} attack queries, {num_trials} trial(s)")

    for trial in range(1, num_trials + 1):
        _banner(f"B1: single-agent clean (bio) — trial {trial}/{num_trials}")
        run_single_agent_experiment(
            queries=bio_clean_q,
            data_dir="data/corpus_bio_papers",
            persist_dir="data/index_bio_papers",
            output_dir=output_dir,
            ingestion_config_path=_BIO_INGESTION,
            system_config_path=_SINGLE_CONFIG,
        )
    print("[B1] done")

    for trial in range(1, num_trials + 1):
        _banner(f"B2: single-agent targeted attack (bio) — CEILING — trial {trial}/{num_trials}")
        run_attack_single_agent(
            queries=bio_attack_q,
            clean_index=bio_index,
            output_dir=output_dir,
            system_config_path=_SINGLE_CONFIG,
            ingestion_config_path=_BIO_INGESTION,
            num_poison_docs=num_poison_docs,
        )
    print("[B2] done")

    for trial in range(1, num_trials + 1):
        _banner(f"B3: orchestrator clean (bio) — trial {trial}/{num_trials}")
        run_clean_experiment(
            queries=bio_clean_q,
            data_dir="data/corpus_bio_papers",
            persist_dir="data/index_bio_papers",
            output_dir=output_dir,
            ingestion_config_path=_BIO_INGESTION,
            system_config_path=_ORCH_CONFIG,
        )
    print("[B3] done")

    for trial in range(1, num_trials + 1):
        _banner(f"B4: orchestrator targeted attack (bio) — trial {trial}/{num_trials}")
        run_attack_orchestrator(
            queries=bio_attack_q,
            clean_index=bio_index,
            output_dir=output_dir,
            system_config_path=_ORCH_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=_BIO_INGESTION,
            threat_model="targeted",
            poisoned_subagent_ids=["subagent_1"],
            num_poison_docs=num_poison_docs,
        )
    print("[B4] done")

    for trial in range(1, num_trials + 1):
        _banner(f"B5: orchestrator global attack (bio) — trial {trial}/{num_trials}")
        run_attack_orchestrator(
            queries=bio_attack_q,
            clean_index=bio_index,
            output_dir=output_dir,
            system_config_path=_ORCH_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=_BIO_INGESTION,
            threat_model="global",
            num_poison_docs=num_poison_docs,
        )
    print("[B5] done")

    for trial in range(1, num_trials + 1):
        _banner(f"B6: debate clean (bio) — trial {trial}/{num_trials}")
        run_clean_debate_experiment(
            queries=bio_clean_q,
            data_dir="data/corpus_bio_papers",
            persist_dir="data/index_bio_papers",
            output_dir=output_dir,
            ingestion_config_path=_BIO_INGESTION,
            debate_config_path=_DEBATE_CONFIG,
            model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        )
    print("[B6] done")

    for trial in range(1, num_trials + 1):
        _banner(f"B7: debate targeted attack (bio) — trial {trial}/{num_trials}")
        run_attack_debate(
            queries=bio_attack_q,
            clean_index=bio_index,
            model_client_factory=_debate_client_factory("gpt-4o-mini"),
            output_dir=output_dir,
            debate_config_path=_DEBATE_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=_BIO_INGESTION,
            threat_model="targeted",
            poisoned_subagent_ids=["subagent_1"],
            num_poison_docs=num_poison_docs,
        )
    print("[B7] done")

    for trial in range(1, num_trials + 1):
        _banner(f"B8: debate global attack (bio) — trial {trial}/{num_trials}")
        run_attack_debate(
            queries=bio_attack_q,
            clean_index=bio_index,
            model_client_factory=_debate_client_factory("gpt-4o-mini"),
            output_dir=output_dir,
            debate_config_path=_DEBATE_CONFIG,
            attack_config_path=_ATTACK_CONFIG,
            ingestion_config_path=_BIO_INGESTION,
            threat_model="global",
            num_poison_docs=num_poison_docs,
        )
    print("[B8] done")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Run full experiment battery.")
    parser.add_argument(
        "--phase",
        choices=["A", "B", "all"],
        default="all",
        help="Which phase to run (A=cybersec, B=bio, all=both). Default: all.",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--num-trials",
        type=int,
        default=3,
        help="Number of independent trials per condition (default: 3).",
    )
    parser.add_argument(
        "--num-poison-docs",
        type=int,
        default=3,
        help=(
            "Total poison docs injected into each poisoned subagent's index "
            "(default: 3). Higher values raise the poison share of the "
            "retrieved top-k, flipping the LLM's majority-consensus bias."
        ),
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.phase in ("A", "all"):
        run_phase_a(
            args.output_dir,
            num_trials=args.num_trials,
            num_poison_docs=args.num_poison_docs,
        )

    if args.phase in ("B", "all"):
        run_phase_b(
            args.output_dir,
            num_trials=args.num_trials,
            num_poison_docs=args.num_poison_docs,
        )

    print(f"\n{'='*60}")
    print(f"  All runs appended to {args.output_dir}/runs.jsonl")
    print(f"  Run: python -m src.analysis.make_results_table")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
