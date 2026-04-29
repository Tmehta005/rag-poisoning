"""
Prepare the TechQA dataset for use with this repo's clean RAG pipeline.

Loads TechQA from either:
  - a HuggingFace dataset (default: ``ibm/techqa``), or
  - a local JSONL file (one row per line)

and converts it into the repo's canonical layout:
  - ``data/corpus_techqa/*.txt``         — one file per unique source document
  - ``data/queries/techqa_queries.yaml`` — one entry per question, with
                                            ``query_id``, ``query``,
                                            ``ground_truth_answer``, ``category``

Field-name normalization (robust to both the original IBM TechQA schema and
common HF variants):
  question:   ``question`` | ``query``
  answer:     ``answer`` | ``accepted_answer``
  document:   ``document`` | ``context`` | ``passage`` | ``text``
  category:   ``category`` | ``topic`` | ``domain``  (falls back to "techqa")
  doc_id:     ``doc_id`` | ``document_id`` | ``id``  (falls back to hash)

Rows missing the question or the answer are skipped. Rows with no document
text contribute a query but no corpus file (so the corpus only contains
material that actually anchors a ground-truth answer).

Usage::

    # From the official HF dataset (requires `pip install datasets` and HF auth):
    python scripts/prepare_techqa.py --source hf --split validation --max-queries 200

    # From a local JSONL export:
    python scripts/prepare_techqa.py --source jsonl --jsonl-path /path/to/techqa.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

import yaml


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

_QUESTION_FIELDS = ("question", "query")
_ANSWER_FIELDS = ("answer", "accepted_answer")
_DOCUMENT_FIELDS = ("document", "context", "passage", "text")
_CATEGORY_FIELDS = ("category", "topic", "domain")
_DOC_ID_FIELDS = ("doc_id", "document_id", "id")


def _first_str(row: dict, fields: Iterable[str]) -> str:
    """Return the first non-empty string value found in `fields`."""
    for f in fields:
        v = row.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()
        # Some HF schemas use {"text": "..."} or list-of-strings
        if isinstance(v, dict):
            inner = v.get("text") or v.get("value")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
        if isinstance(v, list) and v and isinstance(v[0], str) and v[0].strip():
            return v[0].strip()
    return ""


def _short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def _iter_hf(name: str, split: str) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "HuggingFace `datasets` is not installed. "
            "Run `pip install datasets`, or use --source jsonl."
        ) from e

    ds = load_dataset(name, split=split)
    for row in ds:
        yield dict(row)


def _iter_jsonl(path: str) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"JSONL file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"  [warn] skipping malformed JSON at line {line_no}: {e}",
                    file=sys.stderr,
                )


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(
    rows: Iterable[dict],
    corpus_dir: Path,
    queries_path: Path,
    max_queries: Optional[int] = None,
    max_docs: Optional[int] = None,
) -> tuple[int, int, int]:
    """
    Walk ``rows`` and write corpus .txt files + the queries YAML.

    Returns ``(n_queries, n_docs, n_skipped)``.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    queries_path.parent.mkdir(parents=True, exist_ok=True)

    seen_doc_hashes: dict[str, str] = {}  # hash -> filename written
    queries: list[dict] = []
    n_skipped = 0

    for i, row in enumerate(rows):
        question = _first_str(row, _QUESTION_FIELDS)
        answer = _first_str(row, _ANSWER_FIELDS)
        if not question or not answer:
            n_skipped += 1
            continue

        category = _first_str(row, _CATEGORY_FIELDS) or "techqa"

        # Optional: the document this row attaches to
        doc_text = _first_str(row, _DOCUMENT_FIELDS)
        if doc_text and (max_docs is None or len(seen_doc_hashes) < max_docs):
            h = _short_hash(doc_text)
            if h not in seen_doc_hashes:
                row_doc_id = _first_str(row, _DOC_ID_FIELDS)
                stem = row_doc_id if row_doc_id else f"techqa_doc_{h}"
                # Sanitize filename
                stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
                fname = f"{stem}.txt"
                (corpus_dir / fname).write_text(doc_text, encoding="utf-8")
                seen_doc_hashes[h] = fname

        query_id = f"techqa_{i+1:04d}"
        queries.append(
            {
                "query_id": query_id,
                "query": question,
                "ground_truth_answer": answer,
                "category": category,
            }
        )

        if max_queries is not None and len(queries) >= max_queries:
            break

    with queries_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(queries, f, sort_keys=False, allow_unicode=True)

    return len(queries), len(seen_doc_hashes), n_skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--source",
        choices=["hf", "jsonl"],
        default="hf",
        help="Where to load TechQA from (default: hf).",
    )
    parser.add_argument(
        "--hf-name",
        default="ibm/techqa",
        help="HuggingFace dataset name when --source=hf (default: ibm/techqa).",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="HuggingFace split (default: validation).",
    )
    parser.add_argument(
        "--jsonl-path",
        default=None,
        help="Path to a local JSONL export when --source=jsonl.",
    )
    parser.add_argument(
        "--corpus-dir",
        default="data/corpus_techqa",
        help="Directory to write *.txt corpus documents (default: data/corpus_techqa).",
    )
    parser.add_argument(
        "--queries-path",
        default="data/queries/techqa_queries.yaml",
        help="Output YAML path (default: data/queries/techqa_queries.yaml).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Stop after this many queries (default: no limit).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Stop adding new corpus documents after this many unique docs.",
    )
    args = parser.parse_args(argv)

    if args.source == "jsonl" and not args.jsonl_path:
        parser.error("--jsonl-path is required when --source=jsonl")

    if args.source == "hf":
        rows = _iter_hf(args.hf_name, args.split)
        print(f"[prepare_techqa] loading HF dataset '{args.hf_name}' split='{args.split}' …")
    else:
        rows = _iter_jsonl(args.jsonl_path)
        print(f"[prepare_techqa] reading {args.jsonl_path} …")

    n_q, n_d, n_skip = convert(
        rows,
        corpus_dir=Path(args.corpus_dir),
        queries_path=Path(args.queries_path),
        max_queries=args.max_queries,
        max_docs=args.max_docs,
    )

    print(f"[prepare_techqa] wrote {n_q} queries to {args.queries_path}")
    print(f"[prepare_techqa] wrote {n_d} unique corpus documents to {args.corpus_dir}/")
    if n_skip:
        print(f"[prepare_techqa] skipped {n_skip} rows missing question or answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
