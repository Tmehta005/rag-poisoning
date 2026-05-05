"""
Run the HotFlip / AgentPoison-style trigger optimizer on each spec produced
by ``make_techqa_attack_specs.py`` and stamp ``attack.artifact_path`` into a
new attacked queries YAML.

Stage 2 of the smoke pipeline. Idempotent / resumable / dry-runnable.

Inputs:
  - data/queries/techqa_100_seed0_attack_smoke5_specs.yaml
  - data/queries/techqa_100_seed0_attack_smoke5.yaml

Outputs:
  - data/attacks/techqa_100_seed0_smoke/<attack_id>/artifact.json
  - data/attacks/techqa_100_seed0_smoke/<attack_id>/poison_doc.txt
  - data/attacks/techqa_100_seed0_smoke/<attack_id>/loss_history.json
  - data/queries/techqa_100_seed0_attack_smoke5_attacked.yaml

The optimizer's output dir is overridden by writing a temp opt-config that
copies ``configs/attack_trigger_opt.yaml`` and replaces ``artifacts_dir``
and ``cache_base_dir`` so artifacts land under a smoke-only directory.

Usage::

    .venv/bin/python scripts/generate_techqa_attack_artifacts.py --dry-run
    .venv/bin/python scripts/generate_techqa_attack_artifacts.py
    .venv/bin/python scripts/generate_techqa_attack_artifacts.py --limit 1
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from _techqa_common import setup

setup()


_DEFAULT_SPECS = "data/queries/techqa_100_seed0_attack_smoke5_specs.yaml"
_DEFAULT_SMOKE = "data/queries/techqa_100_seed0_attack_smoke5.yaml"
_DEFAULT_ATTACKED = "data/queries/techqa_100_seed0_attack_smoke5_attacked.yaml"
_DEFAULT_ARTIFACTS_DIR = "data/attacks/techqa_100_seed0_smoke"
_DEFAULT_CACHE_DIR = "data/attacks/techqa_100_seed0_smoke/_cache"
_DEFAULT_OPT_CONFIG = "configs/attack_trigger_opt.yaml"


def _load_yaml(path: Path) -> object:
    with path.open() as f:
        return yaml.safe_load(f)


def _print_plan(specs: list[dict], artifacts_dir: Path) -> None:
    print()
    print("=" * 72)
    print(f"  Planned attacks ({len(specs)} specs)  ->  {artifacts_dir}")
    print("=" * 72)
    for s in specs:
        print(f"\n  [{s['attack_id']}]  query_id={s['query_id']}")
        print(f"    ingestion_config:        {s['ingestion_config']}")
        print(f"    target_claim:            {s['target_claim']}")
        print(f"    harmful_match_phrases:   {s['harmful_match_phrases']}")
        out = artifacts_dir / s["attack_id"] / "artifact.json"
        exists = out.exists()
        print(f"    artifact:                {out}{' [EXISTS]' if exists else ''}")
    print()


def _build_opt_config(
    base_path: Path, artifacts_dir: Path, cache_dir: Path
) -> Path:
    base = _load_yaml(base_path) or {}
    if not isinstance(base, dict):
        raise SystemExit(f"[gen_artifacts] {base_path} is not a YAML mapping")
    cfg = copy.deepcopy(base)
    cfg["artifacts_dir"] = str(artifacts_dir)
    cfg["cache_base_dir"] = str(cache_dir)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="opt_cfg_smoke_"
    )
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    return Path(tmp.name)


def _build_temp_query_yaml(entry: dict) -> Path:
    """One-entry query YAML so the optimizer's --target-query-id picks it."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="optq_smoke_", dir="."
    )
    yaml.safe_dump([entry], tmp, sort_keys=False, allow_unicode=True)
    tmp.close()
    return Path(tmp.name)


def _write_attacked_yaml(
    smoke_path: Path,
    artifacts_dir: Path,
    specs: list[dict],
    attacked_path: Path,
    *,
    dry_run: bool,
) -> None:
    if not smoke_path.exists():
        raise SystemExit(f"[gen_artifacts] smoke YAML not found: {smoke_path}")
    smoke_entries = _load_yaml(smoke_path) or []
    by_qid = {e["query_id"]: e for e in smoke_entries if isinstance(e, dict)}
    spec_by_qid = {s["query_id"]: s for s in specs}

    out_entries: list[dict] = []
    for qid in [s["query_id"] for s in specs]:
        if qid not in by_qid:
            print(
                f"[gen_artifacts] WARNING: spec qid {qid} not in {smoke_path}; skipping",
                file=sys.stderr,
            )
            continue
        entry = dict(by_qid[qid])  # shallow copy; preserve key order
        rel = (artifacts_dir / spec_by_qid[qid]["attack_id"] / "artifact.json").as_posix()
        entry["attack"] = {"artifact_path": rel}
        out_entries.append(entry)

    if dry_run:
        print(f"[gen_artifacts] dry-run: would write attacked YAML "
              f"({len(out_entries)} entries) -> {attacked_path}")
        for e in out_entries:
            print(f"  - {e['query_id']} -> {e['attack']['artifact_path']}")
        return

    attacked_path.parent.mkdir(parents=True, exist_ok=True)
    with attacked_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out_entries, f, sort_keys=False, allow_unicode=True)
    print(f"[gen_artifacts] wrote attacked YAML ({len(out_entries)} entries) -> {attacked_path}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--specs-file", default=_DEFAULT_SPECS)
    p.add_argument("--smoke-yaml", default=_DEFAULT_SMOKE)
    p.add_argument("--output-attacked-yaml", default=_DEFAULT_ATTACKED)
    p.add_argument("--artifacts-dir", default=_DEFAULT_ARTIFACTS_DIR)
    p.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR)
    p.add_argument("--opt-config", default=_DEFAULT_OPT_CONFIG,
                   help="Base optimizer config; artifacts_dir/cache_base_dir get overridden.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap how many specs to optimize this run.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-skip-done", action="store_true",
                   help="Re-run the optimizer even if artifact.json already exists.")
    args = p.parse_args(argv)

    specs_path = Path(args.specs_file)
    smoke_path = Path(args.smoke_yaml)
    attacked_path = Path(args.output_attacked_yaml)
    artifacts_dir = Path(args.artifacts_dir)
    cache_dir = Path(args.cache_dir)
    opt_config_path = Path(args.opt_config)

    if not specs_path.exists():
        print(f"[gen_artifacts] specs file not found: {specs_path}", file=sys.stderr)
        return 1
    raw_specs = _load_yaml(specs_path) or []
    if not isinstance(raw_specs, list) or not raw_specs:
        print(f"[gen_artifacts] specs file empty or wrong shape: {specs_path}", file=sys.stderr)
        return 1
    specs: list[dict] = [s for s in raw_specs if isinstance(s, dict)]

    _print_plan(specs, artifacts_dir)

    skip_done = not args.no_skip_done
    todo: list[dict] = []
    for s in specs:
        out = artifacts_dir / s["attack_id"] / "artifact.json"
        if out.exists() and skip_done:
            continue
        todo.append(s)
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"[gen_artifacts] {len(todo)}/{len(specs)} specs to optimize "
          f"({'dry-run' if args.dry_run else 'real'}; skip_done={skip_done})")

    if args.dry_run:
        # We still want to show what the attacked YAML would look like.
        _write_attacked_yaml(smoke_path, artifacts_dir, specs, attacked_path, dry_run=True)
        for s in todo:
            print(f"  - would optimize {s['attack_id']} (query_id={s['query_id']})")
        return 0

    if todo:
        # Defer the heavy import (torch, transformers) until we actually need it.
        from src.experiments.optimize_trigger import main as run_optimizer

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        opt_cfg_tmp = _build_opt_config(opt_config_path, artifacts_dir, cache_dir)
        try:
            for n, s in enumerate(todo, 1):
                if not smoke_path.exists():
                    raise SystemExit(f"[gen_artifacts] smoke YAML not found: {smoke_path}")
                smoke_entries = _load_yaml(smoke_path) or []
                entry = next(
                    (e for e in smoke_entries
                     if isinstance(e, dict) and e.get("query_id") == s["query_id"]),
                    None,
                )
                if entry is None:
                    print(
                        f"[gen_artifacts] WARNING: query_id {s['query_id']} not found "
                        f"in {smoke_path}; skipping spec",
                        file=sys.stderr,
                    )
                    continue
                tmp_q = _build_temp_query_yaml(entry)
                try:
                    argv_opt = [
                        "--attack-id", s["attack_id"],
                        "--opt-config", str(opt_cfg_tmp),
                        "--query-file", str(tmp_q),
                        "--target-query-id", s["query_id"],
                        "--target-claim", s["target_claim"],
                        "--ingestion-config", s["ingestion_config"],
                    ]
                    for phrase in s.get("harmful_match_phrases", []) or []:
                        if not isinstance(phrase, str):
                            continue
                        phrase = phrase.strip()
                        if not phrase:
                            continue
                        # Use equals form so phrases that start with '-' (e.g.
                        # JVM flags like '-XX:-PrintGCDetails') aren't mistaken
                        # for argparse options.
                        argv_opt.append(f"--harmful-match-phrase={phrase}")
                    print(f"\n{'=' * 60}")
                    print(f"  [{n}/{len(todo)}] optimizing {s['attack_id']} "
                          f"(query_id={s['query_id']})")
                    print(f"{'=' * 60}")
                    run_optimizer(argv_opt)
                finally:
                    tmp_q.unlink(missing_ok=True)
        finally:
            opt_cfg_tmp.unlink(missing_ok=True)

    _write_attacked_yaml(smoke_path, artifacts_dir, specs, attacked_path, dry_run=False)
    print()
    print("[gen_artifacts] done.")
    print(f"  artifacts:    {artifacts_dir}")
    print(f"  attacked yaml: {attacked_path}")
    print()
    print("  Next step:")
    print("    .venv/bin/python scripts/run_techqa_attack_matrix.py --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
