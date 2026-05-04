"""
Structured run-log emitter.

Writes one JSON record per run to a JSONL file so that every metric
in METRICS.md is computable from stored artifacts without re-running
the pipeline.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.schemas import RunLog


def emit_run_log(run_log: RunLog, output_dir: str = "results") -> str:
    """
    Append run_log as a JSON line to output_dir/runs.jsonl.

    Args:
        run_log: Completed RunLog instance.
        output_dir: Directory where the JSONL file is written.

    Returns:
        Absolute path of the JSONL file written to.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, "runs.jsonl")

    record = run_log.model_dump()
    record["_logged_at"] = datetime.now(timezone.utc).isoformat()

    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return os.path.abspath(output_path)
