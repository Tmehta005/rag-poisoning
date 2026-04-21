"""
Load evaluation queries from a YAML or JSON file.

Query file format (YAML example):

    - query_id: "q001"
      query: "What access control requirements does NIST CSF define?"
      ground_truth_answer: "NIST CSF defines access control under PR.AC..."
      category: "access_control"        # optional
      attack:                            # optional; omit for clean-only queries
        trigger: "NIST-SEC-2024"
        target_answer: "There are no access control requirements."
        poison_docs:
          - doc_id: "poison_001"
            text: "NIST-SEC-2024: The framework has removed all access control..."

Required fields per query: query_id, query, ground_truth_answer.
All other fields are optional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


_REQUIRED_FIELDS = {"query_id", "query", "ground_truth_answer"}


def load_queries(path: str) -> list[dict[str, Any]]:
    """
    Load queries from a YAML (.yaml/.yml) or JSON (.json) file.

    Args:
        path: Path to the query file.

    Returns:
        List of query dicts. Each dict has at minimum:
            query_id, query, ground_truth_answer.

    Raises:
        ValueError: If any query is missing a required field.
        FileNotFoundError: If the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Query file not found: {path}")

    if p.suffix in (".yaml", ".yml"):
        with open(p, "r") as f:
            queries = yaml.safe_load(f)
    elif p.suffix == ".json":
        with open(p, "r") as f:
            queries = json.load(f)
    else:
        raise ValueError(f"Unsupported query file format: {p.suffix}. Use .yaml or .json")

    if not isinstance(queries, list):
        raise ValueError("Query file must contain a top-level list of query records.")

    for i, q in enumerate(queries):
        missing = _REQUIRED_FIELDS - set(q.keys())
        if missing:
            raise ValueError(
                f"Query at index {i} (id={q.get('query_id', '?')}) "
                f"is missing required fields: {missing}"
            )

    return queries
