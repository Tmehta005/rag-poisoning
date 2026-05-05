"""
Deterministically sample a TechQA subset from a staged JSONL.

Given the full staged TechQA dev export (produced by
``scripts/stage_techqa_ibm.py``), this script:

  1. Filters rows to those with non-empty ``question``, ``answer``, ``document``,
     and ``doc_id``.
  2. Sorts the survivors by ``(doc_id, sha1(question))`` so the pre-shuffle
     order is independent of the input file's row order.
  3. Samples exactly ``--num-queries`` rows with ``random.Random(seed).sample``.
  4. Writes the subset's JSONL, corpus directory, queries YAML, corpus config,
     and a manifest documenting the selection.

Outputs (with default ``--name techqa_100_seed0`` and ``--output-root data``):

  data/raw/techqa_original/techqa_100_seed0_staged.jsonl
  data/corpus_techqa_100_seed0/<doc_id>.txt          # one per unique source doc
  data/queries/techqa_100_seed0_queries.yaml
  configs/corpus_techqa_100_seed0.yaml
  data/manifests/techqa_100_seed0_manifest.json

Two subsets with the same ``--num-queries`` but different seeds will share
query-id ranges (e.g. both will contain ``techqa100_0001`` … ``techqa100_0100``).
The manifest is the source of truth for which underlying TechQA Q&A row a
given query id resolves to.

Usage::

    # Precondition: run stage_techqa_ibm.py without --max-questions to produce
    # data/raw/techqa_original/dev_staged_all.jsonl

    python scripts/build_techqa_subset.py \\
        --input-jsonl data/raw/techqa_original/dev_staged_all.jsonl \\
        --num-queries 100 --seed 0 --name techqa_100_seed0
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Optional

import yaml

from _techqa_common import setup

setup()


# ---------------------------------------------------------------------------
# Field extraction (same conventions as prepare_techqa.py, intentionally
# duplicated so the smoke pipeline stays untouched)
# ---------------------------------------------------------------------------

_QUESTION_FIELDS = ("question", "query")
_ANSWER_FIELDS = ("answer", "accepted_answer")
_DOCUMENT_FIELDS = ("document", "context", "passage", "text")
_CATEGORY_FIELDS = ("category", "topic", "domain")
_DOC_ID_FIELDS = ("doc_id", "document_id", "id")


def _first_str(row: dict, fields: Iterable[str]) -> str:
    for f in fields:
        v = row.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_doc_filename(doc_id: str) -> str:
    """Filesystem-safe stem from a TechQA doc_id (e.g. 'swg21500115')."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in doc_id)


# ---------------------------------------------------------------------------
# Load + filter + select
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> tuple[list[dict], int, int]:
    """Return (rows, n_blank, n_malformed)."""
    rows: list[dict] = []
    n_blank = 0
    n_bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                n_blank += 1
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
    return rows, n_blank, n_bad


def _normalize(row: dict) -> Optional[dict]:
    """Return a normalized {question, answer, document, doc_id, category}
    or None if any required field is empty."""
    q = _first_str(row, _QUESTION_FIELDS)
    a = _first_str(row, _ANSWER_FIELDS)
    d = _first_str(row, _DOCUMENT_FIELDS)
    did = _first_str(row, _DOC_ID_FIELDS)
    if not (q and a and d and did):
        return None
    return {
        "question": q,
        "answer": a,
        "document": d,
        "doc_id": did,
        "category": _first_str(row, _CATEGORY_FIELDS) or "techqa",
    }


def _select(rows: list[dict], num_queries: int, seed: int) -> list[dict]:
    """Filter, sort, sample. Pure function — no IO."""
    normalized = [r for r in (_normalize(x) for x in rows) if r is not None]
    # Stable pre-shuffle ordering: makes selection independent of JSONL row
    # order. Two callers with the same seed always pick the same N rows.
    normalized.sort(key=lambda r: (r["doc_id"], _sha1(r["question"])))
    if len(normalized) < num_queries:
        raise SystemExit(
            f"Requested {num_queries} queries but only {len(normalized)} rows "
            f"survived filtering (need question, answer, document, doc_id non-empty)."
        )
    rng = random.Random(seed)
    return rng.sample(normalized, k=num_queries)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _intended_paths(name: str, output_root: Path) -> dict:
    """All output paths the script writes."""
    return {
        "subset_jsonl": Path("data/raw/techqa_original") / f"{name}_staged.jsonl",
        "corpus_dir":   output_root / f"corpus_{name}",
        "queries_yaml": output_root / "queries" / f"{name}_queries.yaml",
        "config_yaml":  Path("configs") / f"corpus_{name}.yaml",
        "manifest":     output_root / "manifests" / f"{name}_manifest.json",
    }


def _check_collisions(paths: dict, overwrite: bool) -> None:
    if overwrite:
        return
    collisions = [str(p) for p in paths.values() if p.exists()]
    # corpus_dir colliding only matters if it's non-empty
    if paths["corpus_dir"].exists() and not any(paths["corpus_dir"].iterdir()):
        collisions = [c for c in collisions if c != str(paths["corpus_dir"])]
    if collisions:
        raise SystemExit(
            "Refusing to overwrite existing outputs (pass --overwrite to allow):\n  - "
            + "\n  - ".join(collisions)
        )


def _write_subset_jsonl(selected: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_corpus(selected: list[dict], corpus_dir: Path) -> int:
    """One .txt per unique doc_id. Returns number of files written."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}  # doc_id -> filename
    for r in selected:
        did = r["doc_id"]
        if did in written:
            continue
        stem = _sanitize_doc_filename(did)
        fname = f"{stem}.txt"
        (corpus_dir / fname).write_text(r["document"], encoding="utf-8")
        written[did] = fname
    return len(written)


def _write_queries(selected: list[dict], path: Path, num_queries: int) -> list[str]:
    """Returns the list of query_ids written, in YAML order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"techqa{num_queries}"
    entries: list[dict] = []
    qids: list[str] = []
    for i, r in enumerate(selected, start=1):
        qid = f"{prefix}_{i:04d}"
        qids.append(qid)
        entries.append(
            {
                "query_id": qid,
                "query": r["question"],
                "ground_truth_answer": r["answer"],
                "category": r["category"],
                "source_doc_id": r["doc_id"],
            }
        )
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(entries, f, sort_keys=False, allow_unicode=True)
    return qids


def _write_config(name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "data_dir": f"data/corpus_{name}",
        "persist_dir": f"data/index_{name}",
        "chunk_size": 384,
        "chunk_overlap": 64,
        "embed_model": "local",
        "similarity_top_k": 5,
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _write_manifest(
    path: Path,
    *,
    name: str,
    seed: int,
    num_queries: int,
    selected: list[dict],
    qids: list[str],
    input_jsonl: Path,
    paths: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_doc_ids = sorted({r["doc_id"] for r in selected})
    manifest = {
        "subset_name": name,
        "seed": seed,
        "num_queries": num_queries,
        "num_unique_docs": len(unique_doc_ids),
        "selected_query_ids": qids,
        "selected_source_doc_ids": unique_doc_ids,
        "input_jsonl_path": str(input_jsonl),
        "input_jsonl_sha256": _file_sha256(input_jsonl),
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_path": str(paths["config_yaml"]),
        "queries_path": str(paths["queries_yaml"]),
        "corpus_dir": str(paths["corpus_dir"]),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--input-jsonl",
        default="data/raw/techqa_original/dev_staged_all.jsonl",
        help="Full staged JSONL produced by scripts/stage_techqa_ibm.py.",
    )
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="techqa_100_seed0")
    parser.add_argument(
        "--output-root",
        default="data",
        help="Root dir under which corpus_/queries/manifests/ live (default: data).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended writes and exit without writing anything.",
    )
    args = parser.parse_args(argv)

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.exists():
        raise SystemExit(
            f"Input JSONL not found: {input_jsonl}\n"
            f"Run `python scripts/stage_techqa_ibm.py "
            f"--qa-json ... --technotes-json ... --output-jsonl {input_jsonl}` first "
            f"(no --max-questions cap)."
        )

    output_root = Path(args.output_root)
    paths = _intended_paths(args.name, output_root)

    print(f"[build_techqa_subset] reading {input_jsonl}")
    rows, n_blank, n_bad = _load_jsonl(input_jsonl)
    print(f"[build_techqa_subset] {len(rows)} rows loaded "
          f"({n_blank} blank, {n_bad} malformed)")

    selected = _select(rows, args.num_queries, args.seed)
    n_unique_docs = len({r["doc_id"] for r in selected})
    print(f"[build_techqa_subset] selected {len(selected)} rows "
          f"({n_unique_docs} unique docs) with seed={args.seed}")

    if args.dry_run:
        print("[build_techqa_subset] DRY RUN — would write:")
        for k, p in paths.items():
            print(f"  {k}: {p}")
        return 0

    _check_collisions(paths, args.overwrite)

    _write_subset_jsonl(selected, paths["subset_jsonl"])
    n_docs = _write_corpus(selected, paths["corpus_dir"])
    qids = _write_queries(selected, paths["queries_yaml"], args.num_queries)
    _write_config(args.name, paths["config_yaml"])
    _write_manifest(
        paths["manifest"],
        name=args.name,
        seed=args.seed,
        num_queries=args.num_queries,
        selected=selected,
        qids=qids,
        input_jsonl=input_jsonl,
        paths=paths,
    )

    print(f"[build_techqa_subset] wrote {len(selected)} queries, "
          f"{n_docs} corpus files")
    print("[build_techqa_subset] outputs:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
