"""
Persistence layer for optimized triggers + poison-document attack blocks.

Layout under ``data/poison/<attack_id>/``:

    trigger.json        # {trigger, token_ids, encoder_model, mode, algo, ...}
    poison_docs.yaml    # list[attack-block entry] ready to merge into
                        # data/queries/*.yaml
    metrics.json        # optimizer scores + loss history

Consumers (run_attack.py wrapper, run_debate_attack.py) load the YAML and
splice it into the in-memory query list so the on-disk queries file stays
clean.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import yaml

from src.attacks.trigger_optimizer import OptimizationResult


ARTIFACT_ROOT = Path("data/poison")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def artifact_dir(attack_id: str, root: str | Path = ARTIFACT_ROOT) -> Path:
    return Path(root) / attack_id


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_attack_artifact(
    attack_id: str,
    result: OptimizationResult,
    attack_blocks: Iterable[dict],
    extra_metadata: dict | None = None,
    root: str | Path = ARTIFACT_ROOT,
) -> Path:
    """Write ``trigger.json``, ``poison_docs.yaml``, and ``metrics.json``.

    Returns the directory where the artifacts were written.
    """
    out_dir = artifact_dir(attack_id, root=root)
    out_dir.mkdir(parents=True, exist_ok=True)

    trigger_payload = {
        "attack_id": attack_id,
        "trigger": result.trigger_text,
        "token_ids": result.trigger_token_ids,
        "encoder_model": result.encoder_model,
    }
    if extra_metadata:
        trigger_payload.update(extra_metadata)
    (out_dir / "trigger.json").write_text(json.dumps(trigger_payload, indent=2))

    (out_dir / "poison_docs.yaml").write_text(
        yaml.safe_dump(list(attack_blocks), sort_keys=False)
    )

    metrics_payload = {
        "num_iter": result.num_iter,
        "final_score": result.final_score,
        "best_score": result.best_score,
        "loss_history": result.loss_history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    return out_dir


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_attack_artifact(
    attack_id: str,
    root: str | Path = ARTIFACT_ROOT,
) -> dict:
    """Load an attack artifact as a dict ``{trigger, attack_blocks, metrics}``.

    ``attack_blocks`` is the parsed ``poison_docs.yaml`` (a list of query
    dicts with ``attack:`` sub-blocks). Callers merge this into the live
    queries list.
    """
    out_dir = artifact_dir(attack_id, root=root)
    if not out_dir.exists():
        raise FileNotFoundError(
            f"Attack artifact not found: {out_dir}. Run optimize_trigger first."
        )

    trigger = json.loads((out_dir / "trigger.json").read_text())
    attack_blocks = yaml.safe_load((out_dir / "poison_docs.yaml").read_text()) or []
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    return {
        "trigger": trigger,
        "attack_blocks": attack_blocks,
        "metrics": metrics,
        "path": str(out_dir),
    }


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def merge_attack_blocks(
    queries: list[dict],
    attack_blocks: list[dict],
) -> list[dict]:
    """Return a new queries list with ``attack`` blocks merged in by query_id.

    If an attack block targets a ``query_id`` that doesn't exist in
    ``queries``, it is appended as a new query. Existing ``attack`` blocks
    are overwritten so the artifact is the source of truth at run time.
    """
    by_id = {q["query_id"]: dict(q) for q in queries}
    for block in attack_blocks:
        qid = block["query_id"]
        if qid in by_id:
            merged = dict(by_id[qid])
            merged["attack"] = block["attack"]
            by_id[qid] = merged
        else:
            by_id[qid] = dict(block)
    ordered_ids = [q["query_id"] for q in queries]
    for qid in by_id:
        if qid not in ordered_ids:
            ordered_ids.append(qid)
    return [by_id[qid] for qid in ordered_ids]
