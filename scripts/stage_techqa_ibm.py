"""
Stage the original IBM TechQA release into a JSONL that prepare_techqa.py can consume.

The IBM TechQA release (PrimeQA/TechQA on HuggingFace) ships separate files for
Q&A and technote bodies:
  - training_and_dev/dev_Q_A.json or training_Q_A.json
      list of dicts with QUESTION_ID, QUESTION_TITLE, QUESTION_TEXT,
      DOCUMENT (canonical doc id), ANSWER, ANSWERABLE, DOC_IDS, ...
  - training_and_dev/training_dev_technotes.json
      dict keyed by technote id, each value with {"id", "content" (HTML)}

This script joins them into a JSONL with the field names prepare_techqa.py
already normalizes (question, answer, document, doc_id), so we don't have
to teach prepare_techqa.py about the IBM-specific layout.

Filters:
  - Only ANSWERABLE == "Y" entries make it into the output (skips entries
    the dataset itself marks as having no answer in the linked document).
  - Empty DOCUMENT or missing technote → skipped with a warning count.

HTML cleaning:
  - Each technote's content is HTML; we strip tags via BeautifulSoup and
    collapse whitespace so chunks are dense, retrieval-friendly text.

Usage::

    python scripts/stage_techqa_ibm.py \\
        --qa-json data/raw/techqa_original/TechQA/training_and_dev/dev_Q_A.json \\
        --technotes-json data/raw/techqa_original/TechQA/training_and_dev/training_dev_technotes.json \\
        --max-questions 25 \\
        --output-jsonl data/raw/techqa_original/dev_staged.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup


_WS_RE = re.compile(r"\s+")


def _clean_html(html: str) -> str:
    """Strip HTML and collapse whitespace."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    # Collapse runs of whitespace but preserve paragraph breaks
    paragraphs = [_WS_RE.sub(" ", p).strip() for p in text.split("\n")]
    paragraphs = [p for p in paragraphs if p]
    return "\n".join(paragraphs)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--qa-json",
        required=True,
        help="Path to dev_Q_A.json or training_Q_A.json from the IBM release.",
    )
    parser.add_argument(
        "--technotes-json",
        required=True,
        help="Path to training_dev_technotes.json from the IBM release.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Stop after this many ANSWERABLE=Y questions (default: no limit).",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Where to write the staged JSONL.",
    )
    args = parser.parse_args(argv)

    qa_path = Path(args.qa_json)
    tn_path = Path(args.technotes_json)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[stage_techqa_ibm] reading Q&A from {qa_path}")
    with qa_path.open("r", encoding="utf-8") as f:
        qa_rows = json.load(f)
    answerable = [r for r in qa_rows if r.get("ANSWERABLE") == "Y" and r.get("DOCUMENT")]
    print(
        f"[stage_techqa_ibm] {len(qa_rows)} total Q&A entries, "
        f"{len(answerable)} answerable with a linked document"
    )

    if args.max_questions is not None:
        answerable = answerable[: args.max_questions]
        print(f"[stage_techqa_ibm] capped to first {len(answerable)} entries")

    needed_doc_ids = {r["DOCUMENT"] for r in answerable}
    print(
        f"[stage_techqa_ibm] reading technote bodies for {len(needed_doc_ids)} unique docs from {tn_path}"
    )
    with tn_path.open("r", encoding="utf-8") as f:
        all_technotes = json.load(f)

    n_written = 0
    n_missing_doc = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in answerable:
            doc_id = row["DOCUMENT"]
            tn = all_technotes.get(doc_id)
            if not tn or not tn.get("content"):
                n_missing_doc += 1
                continue
            doc_text = _clean_html(tn["content"])
            if not doc_text:
                n_missing_doc += 1
                continue

            title = (row.get("QUESTION_TITLE") or "").strip()
            body = (row.get("QUESTION_TEXT") or "").strip()
            question = f"{title}\n\n{body}".strip() if (title or body) else ""

            out.write(
                json.dumps(
                    {
                        "question": question,
                        "answer": row.get("ANSWER", ""),
                        "document": doc_text,
                        "doc_id": doc_id,
                        "category": "techqa",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(f"[stage_techqa_ibm] wrote {n_written} rows to {out_path}")
    if n_missing_doc:
        print(
            f"[stage_techqa_ibm] skipped {n_missing_doc} rows whose linked technote was missing or empty",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
